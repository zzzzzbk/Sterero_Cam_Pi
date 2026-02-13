import numpy as np
import cv2

# -------- USER SETTINGS --------
CALIB_NPZ = "stereo_calib_charuco.npz"

LEFT_IMG  = "output\\octopus_left.png"
RIGHT_IMG = "output\\octopus_right.png"
# LEFT_IMG  = "calib\\left_09.png"
# RIGHT_IMG = "calib\\right_09.png"

OUT_PREFIX = "result"

# Stereo matcher parameters
NUM_DISPARITIES = 16 * 28   # must be multiple of 16
BLOCK_SIZE = 7             # odd: 5,7,9...

DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 3.0
# -------------------------------


# ---- Load calibration ----
data = np.load(CALIB_NPZ, allow_pickle=True)

image_size = tuple(data["image_size"])
mapLx, mapLy = data["mapLx"], data["mapLy"]
mapRx, mapRy = data["mapRx"], data["mapRy"]
Q = data["Q"]

# ---- Load images ----
imgL = cv2.imread(LEFT_IMG, cv2.IMREAD_COLOR)
imgR = cv2.imread(RIGHT_IMG, cv2.IMREAD_COLOR)

if imgL is None or imgR is None:
    raise RuntimeError("Failed to load left/right images")

h, w = imgL.shape[:2]
if (w, h) != image_size:
    raise RuntimeError(
        f"Image size {(w,h)} does not match calibration size {image_size}"
    )

# ---- Rectify ----
rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

cv2.imwrite(f"{OUT_PREFIX}_rectL.png", rectL)
cv2.imwrite(f"{OUT_PREFIX}_rectR.png", rectR)

# ---- Convert to grayscale ----
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

# ---- Stereo disparity ----
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=NUM_DISPARITIES,
    blockSize=BLOCK_SIZE,
    P1=8 * BLOCK_SIZE * BLOCK_SIZE,
    P2=32 * BLOCK_SIZE * BLOCK_SIZE,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# ---- Save disparity visualization ----
disp_vis = disp.copy()
disp_vis[disp_vis <= 0] = np.nan
disp_norm = cv2.normalize(
    np.nan_to_num(disp_vis, nan=0.0),
    None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(f"{OUT_PREFIX}_disparity.png", disp_norm)

# ---- Reproject to 3D ----
points_3d = cv2.reprojectImageTo3D(disp, Q)
depth_m = points_3d[:, :, 2]

# ---- Save depth visualization ----
depth_vis = depth_m.copy()
depth_vis[(depth_vis < DEPTH_MIN_M) | (depth_vis > DEPTH_MAX_M)] = np.nan
depth_norm = cv2.normalize(
    np.nan_to_num(depth_vis, nan=0.0),
    None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(f"{OUT_PREFIX}_depth.png", depth_norm)

# ---- Quick sanity check ----
cy, cx = depth_m.shape[0] // 2, depth_m.shape[1] // 2
print("Center depth (m):", float(depth_m[cy, cx]))

print("Saved:")
print(f"  {OUT_PREFIX}_rectL.png")
print(f"  {OUT_PREFIX}_rectR.png")
print(f"  {OUT_PREFIX}_disparity.png")
print(f"  {OUT_PREFIX}_depth.png")
