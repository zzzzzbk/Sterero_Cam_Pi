import glob
import os
import numpy as np
import cv2

# -------- USER SETTINGS ----------
CALIB_DIR = "calib"
LEFT_GLOB  = os.path.join(CALIB_DIR, "left_*.png")
RIGHT_GLOB = os.path.join(CALIB_DIR, "right_*.png")

# ChArUco board definition (SQUARES, not inner corners)
BOARD_COLS = 8   # number of squares along X
BOARD_ROWS = 6   # number of squares along Y

# Physical sizes (meters)
SQUARE_LENGTH = 0.010   # e.g. 10mm squares
MARKER_LENGTH = 0.007   # e.g. 7mm markers (must be < square_length)

# ArUco dictionary
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

SHOW_DETECTIONS = True
OUT_NPZ = "stereo_calib_charuco.npz"
# -------------------------------

def sort_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    num = "".join([c for c in base if c.isdigit()])
    return int(num) if num else 0

left_paths  = sorted(glob.glob(LEFT_GLOB), key=sort_key)
right_paths = sorted(glob.glob(RIGHT_GLOB), key=sort_key)

if len(left_paths) == 0 or len(right_paths) == 0:
    raise RuntimeError("No images found. Check CALIB_DIR and filename patterns.")
if len(left_paths) != len(right_paths):
    raise RuntimeError(f"Mismatched counts: left={len(left_paths)} right={len(right_paths)}")

# ---- Build ChArUco board ----
board = cv2.aruco.CharucoBoard(
    (BOARD_COLS, BOARD_ROWS),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=ARUCO_DICT
)

# Detector parameters (OpenCV 4.7+ uses ArucoDetector; fallback for older)
detector_params = cv2.aruco.DetectorParameters()
try:
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, detector_params)
    use_new_api = True
except AttributeError:
    use_new_api = False

def detect_charuco(gray):
    """
    Returns (charuco_corners, charuco_ids, vis_debug)
    corners: (N,1,2), ids: (N,1)
    """
    if use_new_api:
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=detector_params)

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, (marker_corners, marker_ids)

    # Refine marker detection (optional but helps)
    cv2.aruco.refineDetectedMarkers(gray, board, marker_corners, marker_ids, rejected)

    # Interpolate ChArUco corners
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )

    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None, (marker_corners, marker_ids)

    return charuco_corners, charuco_ids, (marker_corners, marker_ids)

# ---- Collect detections ----
image_size = None

# For mono calibration (opencv wants per-frame charuco corners/ids)
all_charuco_corners_L = []
all_charuco_ids_L = []
all_charuco_corners_R = []
all_charuco_ids_R = []

# For stereo calibration (we'll build matched object/image points per pair)
objpoints_stereo = []
imgpointsL_stereo = []
imgpointsR_stereo = []

used_pairs = 0

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

    cL, idsL, debugL = detect_charuco(gray_l)
    cR, idsR, debugR = detect_charuco(gray_r)

    if cL is None or cR is None:
        print(f"ChArUco not found in pair: {os.path.basename(lp)} / {os.path.basename(rp)}")
        continue

    # Save for mono calibration
    all_charuco_corners_L.append(cL)
    all_charuco_ids_L.append(idsL)
    all_charuco_corners_R.append(cR)
    all_charuco_ids_R.append(idsR)

    # Build stereo correspondences by common corner IDs
    idsL_flat = idsL.flatten()
    idsR_flat = idsR.flatten()
    common = np.intersect1d(idsL_flat, idsR_flat)

    if len(common) < 8:
        print(f"Too few common ChArUco corners in pair: {os.path.basename(lp)} / {os.path.basename(rp)}")
        continue

    # Map ID -> corner for each view
    dictL = {int(i): cL[k, 0, :] for k, i in enumerate(idsL_flat)}
    dictR = {int(i): cR[k, 0, :] for k, i in enumerate(idsR_flat)}

    # Get object points for those IDs from the board model
    # board.getChessboardCorners() returns all corners in ID order (0..N-1)
    board_obj = board.getChessboardCorners()  # shape (Nc,3)
    obj = []
    ptsL = []
    ptsR = []
    for cid in common:
        cid = int(cid)
        obj.append(board_obj[cid])
        ptsL.append(dictL[cid])
        ptsR.append(dictR[cid])

    obj = np.array(obj, dtype=np.float32).reshape(-1, 3)
    ptsL = np.array(ptsL, dtype=np.float32).reshape(-1, 1, 2)
    ptsR = np.array(ptsR, dtype=np.float32).reshape(-1, 1, 2)

    objpoints_stereo.append(obj)
    imgpointsL_stereo.append(ptsL)
    imgpointsR_stereo.append(ptsR)
    used_pairs += 1

    if SHOW_DETECTIONS:
        visL = img_l.copy()
        visR = img_r.copy()
        marker_corners_L, marker_ids_L = debugL
        marker_corners_R, marker_ids_R = debugR

        if marker_ids_L is not None:
            cv2.aruco.drawDetectedMarkers(visL, marker_corners_L, marker_ids_L)
        if marker_ids_R is not None:
            cv2.aruco.drawDetectedMarkers(visR, marker_corners_R, marker_ids_R)

        cv2.aruco.drawDetectedCornersCharuco(visL, cL, idsL)
        cv2.aruco.drawDetectedCornersCharuco(visR, cR, idsR)

        both = np.hstack([visL, visR])
        window_name = "ChArUco detections (L | R)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1750, 500)
        cv2.imshow(window_name, both)
        key = cv2.waitKey(0)
        if key == 27:
            SHOW_DETECTIONS = False
            cv2.destroyAllWindows()

if used_pairs < 10:
    raise RuntimeError(f"Too few valid stereo pairs ({used_pairs}). Aim for 20–50 good pairs.")

print(f"Using {used_pairs} valid stereo pairs. Image size: {image_size}")

# ---- 1) Mono calibrate each camera using ChArUco ----
# OpenCV provides a direct function:
# calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, ...)
retL, K_l, D_l, rvecs_l, tvecs_l = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners_L, all_charuco_ids_L, board, image_size, None, None
)
retR, K_r, D_r, rvecs_r, tvecs_r = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners_R, all_charuco_ids_R, board, image_size, None, None
)

print("Mono (ChArUco) RMS:")
print("  Left :", retL)
print("  Right:", retR)

# ---- 2) Stereo calibration (fix intrinsics, solve extrinsics) ----
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
flags = cv2.CALIB_FIX_INTRINSIC

retStereo, K_l2, D_l2, K_r2, D_r2, R, T, E, F = cv2.stereoCalibrate(
    objpoints_stereo,
    imgpointsL_stereo,
    imgpointsR_stereo,
    K_l, D_l,
    K_r, D_r,
    image_size,
    criteria=criteria,
    flags=flags
)

print("Stereo RMS:", retStereo)
baseline = float(np.linalg.norm(T))
print("Estimated baseline (meters):", baseline)

# ---- 3) Rectification & maps ----
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_l, D_l, K_r, D_r,
    image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=-1
)

mapLx, mapLy = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, image_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, image_size, cv2.CV_32FC1)

np.savez(
    OUT_NPZ,
    image_size=image_size,

    # Board settings
    BOARD_COLS=BOARD_COLS, BOARD_ROWS=BOARD_ROWS,
    SQUARE_LENGTH=SQUARE_LENGTH, MARKER_LENGTH=MARKER_LENGTH,

    # Intrinsics
    K_l=K_l, D_l=D_l,
    K_r=K_r, D_r=D_r,

    # Extrinsics
    R=R, T=T, E=E, F=F,

    # Rectification
    R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2,
    mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy,
)

print(f"Saved calibration to: {OUT_NPZ}")
