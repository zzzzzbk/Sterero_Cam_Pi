import numpy as np
import cv2

# -------- USER SETTINGS --------
CAMERA_ID = 1
CALIB_NPZ = f"camera_data/camera{CAMERA_ID}/stereo_calib_charuco.npz"
name="shared"
LEFT_IMG  = f"camera_data/camera{CAMERA_ID}/output/{name}{CAMERA_ID}_left.png"
RIGHT_IMG = f"camera_data/camera{CAMERA_ID}/output/{name}{CAMERA_ID}_right.png"
# LEFT_IMG  = "calib\\left_09.png"
# RIGHT_IMG = "calib\\right_09.png"

OUT_PREFIX = f"processed/{name}{CAMERA_ID}"

# Stereo matcher parameters



DEPTH_MIN_M = 0.01
DEPTH_MAX_M = 0.50
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

#rectR = np.roll(rectR, shift=-4, axis=0)

cv2.imwrite(f"{OUT_PREFIX}_rectL.png", rectL)
cv2.imwrite(f"{OUT_PREFIX}_rectR.png", rectR)

# ---- Convert to grayscale ----
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

# ---- Stereo disparity ----
BLOCK_SIZE =5      # odd: 5,7,9...
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities= 16 *20,   # must be multiple of 16,
    blockSize=BLOCK_SIZE,
    P1=8 * BLOCK_SIZE * BLOCK_SIZE,
    P2=32 * BLOCK_SIZE * BLOCK_SIZE,
    disp12MaxDiff=-1,
    uniquenessRatio=1,
    speckleWindowSize=200,
    speckleRange=1,
    preFilterCap = 63,
    mode=cv2.STEREO_SGBM_MODE_SGBM,
)
# grayL=cv2.GaussianBlur(grayL,(3,3),0)
# grayR=cv2.GaussianBlur(grayR,(3,3),0)
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0


# --- Ensure grayscale is 8-bit ---
# grayL, grayR should be uint8; if not, convert.
if grayL.dtype != np.uint8:
    grayL_u8 = cv2.normalize(grayL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    grayL_u8 = grayL

if grayR.dtype != np.uint8:
    grayR_u8 = cv2.normalize(grayR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    grayR_u8 = grayR

# --- Your left matcher (stereo) already exists ---
# stereo = cv2.StereoSGBM_create(...)

# --- Create a right matcher with the same settings ---
try:
    stereoR = cv2.ximgproc.createRightMatcher(stereo)
except AttributeError:
    raise RuntimeError(
        "cv2.ximgproc not found. Install opencv-contrib-python (not opencv-python)."
    )

# --- Compute raw disparities (fixed-point int16) ---
dispL_raw = stereo.compute(grayL_u8, grayR_u8)      # int16, scaled by 16
dispR_raw = stereoR.compute(grayR_u8, grayL_u8)     # int16, scaled by 16

# Convert to float disparities in pixels
dispL = dispL_raw.astype(np.float32) / 16.0
dispR = dispR_raw.astype(np.float32) / 16.0

# --- WLS filter ---
wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)

# Typical good starting values:
# lambda: smoothness strength (higher = smoother, less noise, more bleeding)
# sigmaColor: edge sensitivity (higher = edges preserved more strongly)
wls.setLambda(3000)        # try 5000, 8000, 12000, 20000
wls.setSigmaColor(0.8)     # try 0.8 - 2.0

# Filter expects int16 disparities (scaled by 16) + left view image as guidance
dispL_wls_raw = wls.filter(dispL_raw, grayL_u8, None, dispR_raw)

# Back to float disparity in pixels
disp_wls = dispL_wls_raw.astype(np.float32) / 16.0

# --- Clean invalids / clamp ---
# WLS may output negatives/zeros; mask them out
disp_wls_clean = disp_wls.copy()
disp_wls_clean[disp_wls_clean <= 0.0] = np.nan

# Optional: remove tiny disparities if you only trust nearer depths
min_disp_valid = 1.0
disp_wls_clean[disp_wls_clean < min_disp_valid] = np.nan

# If you want a displayable version (0 for invalid)
disp_wls_vis = np.nan_to_num(disp_wls_clean, nan=0.0).astype(np.float32)

# #filtering disparity
# disp_clean = disp.copy()
# disp_clean[disp_clean <= 0] = np.nan

# # median filter on a filled version
# disp_filled = np.nan_to_num(disp_clean, nan=0.0).astype(np.float32)
# disp_med = cv2.medianBlur(disp_filled, 5)

# # keep only where original was valid
# valid = np.isfinite(disp_clean) & (disp_clean > 1.0)
# disp_med[~valid] = 0.0

# ---- Save disparity visualization ----
disp_vis = disp_wls_vis.copy()
disp_vis[disp_vis <= 0] = np.nan
disp_norm = cv2.normalize(
    np.nan_to_num(disp_vis, nan=0.0),
    None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(f"{OUT_PREFIX}_disparity.png", disp_norm)



# ---- Reproject to 3D ----
points_3d = cv2.reprojectImageTo3D(disp_vis, Q)
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


def export_pointcloud_ply(points_3d, colors_bgr, disp,
                          ply_path="cloud.ply",
                          depth_min=0.05, depth_max=3.0,
                          disp_min=1.0):
    """
    points_3d: (H,W,3) float32 from reprojectImageTo3D
    colors_bgr: (H,W,3) uint8 (use rectified left image)
    disp: (H,W) float32 disparity in pixels
    """
    X = points_3d[:, :, 0]
    Y = points_3d[:, :, 1]
    Z = points_3d[:, :, 2]

    # Validity mask
    mask = np.isfinite(Z) & (Z > depth_min) & (Z < depth_max) & np.isfinite(disp) & (disp > disp_min)

    pts = points_3d[mask]  # (N,3)
    col = colors_bgr[mask] # (N,3) BGR

    # Convert BGR->RGB for PLY
    col = col[:, ::-1]

    # Write ASCII PLY
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, col):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    print(f"Saved point cloud: {ply_path}   points={len(pts)}")

export_pointcloud_ply(points_3d, rectL, disp_vis, ply_path=f"{OUT_PREFIX}_cloud.ply",
                      depth_min=DEPTH_MIN_M, depth_max=DEPTH_MAX_M, disp_min=1.0)