import glob
import os
import numpy as np
import cv2

# -------- USER SETTINGS ----------
CALIB_DIR = "calib"
LEFT_GLOB  = os.path.join(CALIB_DIR, "left_*.png")
RIGHT_GLOB = os.path.join(CALIB_DIR, "right_*.png")

# inner corners (NOT squares). Example: a "7x5" board means (7,5) inner corners.
CHESSBOARD = (7, 7)

# Square size in meters (e.g. 20 mm = 0.020)
SQUARE_SIZE = 0.010

# If True, shows detections; press any key to step, ESC to quit preview
SHOW_DETECTIONS = True

# Output file
OUT_NPZ = "stereo_calib_result.npz"
# -------------------------------

def sort_key(path):
    # Extract number from "..._12.png"
    base = os.path.splitext(os.path.basename(path))[0]
    num = "".join([c for c in base if c.isdigit()])
    return int(num) if num else 0

left_paths  = sorted(glob.glob(LEFT_GLOB), key=sort_key)
right_paths = sorted(glob.glob(RIGHT_GLOB), key=sort_key)

if len(left_paths) == 0 or len(right_paths) == 0:
    raise RuntimeError("No images found. Check CALIB_DIR and filename patterns.")
if len(left_paths) != len(right_paths):
    raise RuntimeError(f"Mismatched counts: left={len(left_paths)} right={len(right_paths)}")

# Prepare object points (0,0,0), (1,0,0), ... scaled by square size
objp = np.zeros((CHESSBOARD[1] * CHESSBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []        # 3D points in world coords
imgpoints_l = []      # 2D points in left image
imgpoints_r = []      # 2D points in right image

criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
flags_find = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

image_size = None
used = 0

for lp, rp in zip(left_paths, right_paths):
    img_l = cv2.imread(lp, cv2.IMREAD_COLOR)
    img_r = cv2.imread(rp, cv2.IMREAD_COLOR)
    if img_l is None or img_r is None:
        print(f"Skip unreadable pair: {lp}, {rp}")
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = (gray_l.shape[1], gray_l.shape[0])

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD, flags_find)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD, flags_find)

    if not (ret_l and ret_r):
        print(f"Chessboard not found in pair: {os.path.basename(lp)} / {os.path.basename(rp)}")
        continue

    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria_subpix)
    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria_subpix)

    objpoints.append(objp.copy())
    imgpoints_l.append(corners_l)
    imgpoints_r.append(corners_r)
    used += 1

    if SHOW_DETECTIONS:
        vis_l = img_l.copy()
        vis_r = img_r.copy()
        cv2.drawChessboardCorners(vis_l, CHESSBOARD, corners_l, True)
        cv2.drawChessboardCorners(vis_r, CHESSBOARD, corners_r, True)
        both = np.hstack([vis_l, vis_r])
        cv2.imshow("detections (left | right)", both)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            SHOW_DETECTIONS = False
            cv2.destroyAllWindows()

if used < 10:
    raise RuntimeError(f"Too few valid pairs ({used}). Aim for 20–50 good pairs.")

print(f"Using {used} valid stereo pairs. Image size: {image_size}")

# ---- 1) Calibrate each camera intrinsics ----
ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objpoints, imgpoints_l, image_size, None, None
)
ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objpoints, imgpoints_r, image_size, None, None
)

print("Mono calibration RMS:")
print("  Left :", ret_l)
print("  Right:", ret_r)

# ---- 2) Stereo calibrate (extrinsics) ----
stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
stereo_flags = cv2.CALIB_FIX_INTRINSIC  # keep K,D from mono calibration

ret_stereo, K_l2, D_l2, K_r2, D_r2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_l, D_l, K_r, D_r,
    image_size,
    criteria=stereo_criteria,
    flags=stereo_flags
)

print("Stereo calibration RMS:", ret_stereo)
baseline = float(np.linalg.norm(T))
print("Estimated baseline (same units as SQUARE_SIZE):", baseline)

# ---- 3) Rectification + maps ----
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_l, D_l, K_r, D_r,
    image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0  # alpha=0 crops to valid region
)

mapLx, mapLy = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, image_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, image_size, cv2.CV_32FC1)

np.savez(
    OUT_NPZ,
    image_size=image_size,
    CHESSBOARD=CHESSBOARD,
    SQUARE_SIZE=SQUARE_SIZE,

    K_l=K_l, D_l=D_l,
    K_r=K_r, D_r=D_r,

    R=R, T=T, E=E, F=F,

    R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    roi1=roi1, roi2=roi2,

    mapLx=mapLx, mapLy=mapLy,
    mapRx=mapRx, mapRy=mapRy,
)

print(f"Saved calibration to: {OUT_NPZ}")
