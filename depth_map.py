#!/usr/bin/env python3
"""
Compute a stereo depth map and export a point cloud from a calibrated stereo pair.

Reads a stereo calibration .npz file (produced by calibration_ChArUco.py) and
a left/right image pair, then saves:
  - Rectified left/right images
  - Disparity map (WLS-filtered)
  - Depth map
  - Coloured point cloud (.ply)

Usage (standalone):
    python depth_map.py --calib <calib.npz> --left <left.png> --right <right.png> \\
                        --out-dir <output_dir>

Can also be imported and called programmatically:
    from depth_map import compute_depth
    compute_depth("calib.npz", "left.png", "right.png", "output/")
"""

import argparse
import os
import sys

import numpy as np
import cv2

# -------- DEFAULTS (kept for backward-compatible standalone usage) --------
_DEFAULT_CAMERA_ID = 1
_DEFAULT_NAME       = "shared"
_DEFAULT_CALIB_NPZ  = f"camera_data/camera{_DEFAULT_CAMERA_ID}/stereo_calib_charuco.npz"
_DEFAULT_LEFT_IMG   = f"camera_data/camera{_DEFAULT_CAMERA_ID}/output/{_DEFAULT_NAME}{_DEFAULT_CAMERA_ID}_left.png"
_DEFAULT_RIGHT_IMG  = f"camera_data/camera{_DEFAULT_CAMERA_ID}/output/{_DEFAULT_NAME}{_DEFAULT_CAMERA_ID}_right.png"
_DEFAULT_OUT_DIR    = f"processed/{_DEFAULT_NAME}{_DEFAULT_CAMERA_ID}"

DEPTH_MIN_M = 0.01
DEPTH_MAX_M = 0.50
# -------------------------------------------------------------------------


def export_pointcloud_ply(
    points_3d,
    colors_bgr,
    disp,
    ply_path="cloud.ply",
    depth_min=0.05,
    depth_max=3.0,
    disp_min=1.0,
):
    """
    Write a coloured point cloud to an ASCII PLY file.

    Args:
        points_3d:  (H, W, 3) float32 from cv2.reprojectImageTo3D.
        colors_bgr: (H, W, 3) uint8 – use the rectified left image for colour.
        disp:       (H, W) float32 disparity in pixels.
        ply_path:   Destination file path.
        depth_min:  Minimum valid depth in metres.
        depth_max:  Maximum valid depth in metres.
        disp_min:   Minimum valid disparity in pixels.
    """
    X = points_3d[:, :, 0]
    Y = points_3d[:, :, 1]
    Z = points_3d[:, :, 2]

    mask = (
        np.isfinite(Z) & (Z > depth_min) & (Z < depth_max)
        & np.isfinite(disp) & (disp > disp_min)
    )

    pts = points_3d[mask]   # (N, 3)
    col = colors_bgr[mask]  # (N, 3) BGR -> RGB below
    col = col[:, ::-1]

    os.makedirs(os.path.dirname(os.path.abspath(ply_path)), exist_ok=True)
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, col):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    print(f"Saved point cloud: {ply_path}   points={len(pts)}")


def compute_depth(
    disparity: int,
    calib_npz: str,
    left_img: str,
    right_img: str,
    out_dir: str,
    depth_min_m: float = DEPTH_MIN_M,
    depth_max_m: float = DEPTH_MAX_M,
    USE_WLS: bool = False,
) -> dict:
    """
    Compute disparity, depth map, and point cloud for one stereo image pair.

    Args:
        calib_npz:   Path to the stereo calibration .npz file.
        left_img:    Path to the left-camera image.
        right_img:   Path to the right-camera image.
        out_dir:     Directory to write output files into.
        depth_min_m: Minimum depth (metres) shown in the depth visualisation.
        depth_max_m: Maximum depth (metres) shown in the depth visualisation.
        USE_WLS:     Whether to use the WLS filter for disparity smoothing.
    Returns:
        Dictionary with keys ``rectL``, ``rectR``, ``disparity``, ``depth``,
        ``cloud`` mapping to the absolute paths of the saved files.
    """
    out_path = out_dir.rstrip("/\\")
    os.makedirs(out_path, exist_ok=True)

    # ---- Load calibration ----
    data = np.load(calib_npz, allow_pickle=True)
    image_size = tuple(data["image_size"])
    mapLx, mapLy = data["mapLx"], data["mapLy"]
    mapRx, mapRy = data["mapRx"], data["mapRy"]
    Q = data["Q"]

    # ---- Load images ----
    imgL = cv2.imread(left_img, cv2.IMREAD_COLOR)
    imgR = cv2.imread(right_img, cv2.IMREAD_COLOR)
    if imgL is None or imgR is None:
        raise RuntimeError(
            f"Failed to load images:\n  left:  {left_img}\n  right: {right_img}"
        )

    h, w = imgL.shape[:2]
    if (w, h) != image_size:
        raise RuntimeError(
            f"Image size {(w, h)} does not match calibration size {image_size}. "
            "Rectification maps are resolution-specific."
        )

    # ---- Rectify ----
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    path_rectL = os.path.join(out_path, "rectL.png")
    path_rectR = os.path.join(out_path, "rectR.png")
    cv2.imwrite(path_rectL, rectL)
    cv2.imwrite(path_rectR, rectR)

    # ---- Grayscale ----
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    grayL_u8 = (grayL if grayL.dtype == np.uint8
                else cv2.normalize(grayL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    grayR_u8 = (grayR if grayR.dtype == np.uint8
                else cv2.normalize(grayR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    # ---- Stereo disparity (SGBM + WLS filter) ----
    BLOCK_SIZE = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * disparity,
        blockSize=BLOCK_SIZE,
        P1=8  * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=-1,
        uniquenessRatio=1,
        speckleWindowSize=200,
        speckleRange=1,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )

    try:
        stereoR = cv2.ximgproc.createRightMatcher(stereo)
    except AttributeError:
        raise RuntimeError(
            "cv2.ximgproc not found. "
            "Install opencv-contrib-python (not opencv-python)."
        )

    dispL_raw = stereo.compute(grayL_u8, grayR_u8)
    dispR_raw = stereoR.compute(grayR_u8, grayL_u8)

    if USE_WLS:
        dispL_raw = stereo.compute(grayL_u8, grayR_u8)
        dispR_raw = stereoR.compute(grayR_u8, grayL_u8)

        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls.setLambda(3000)
        wls.setSigmaColor(0.8)
        dispL_wls_raw = wls.filter(dispL_raw, grayL_u8, None, dispR_raw)

        disp_wls = dispL_wls_raw.astype(np.float32) / 16.0
        disp_wls_clean = disp_wls.copy()
        disp_wls_clean[disp_wls_clean <= 1.0] = np.nan
        disp_vis = np.nan_to_num(disp_wls_clean, nan=0.0).astype(np.float32)
    else:
        disp_raw = stereo.compute(grayL_u8, grayR_u8)
        disp = disp_raw.astype(np.float32) / 16.0
        disp[disp <= 1.0] = np.nan
        disp_vis = np.nan_to_num(disp, nan=0.0).astype(np.float32)

    # ---- Save disparity visualisation ----
    disp_norm = cv2.normalize(
        np.nan_to_num(disp_vis, nan=0.0), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    path_disp = os.path.join(out_path, "disparity.png")
    cv2.imwrite(path_disp, disp_norm)

    # ---- Reproject to 3-D ----
    points_3d = cv2.reprojectImageTo3D(disp_vis, Q)
    depth_m = points_3d[:, :, 2]

    # ---- Save depth visualisation ----
    depth_vis = depth_m.copy()
    depth_vis[(depth_vis < depth_min_m) | (depth_vis > depth_max_m)] = np.nan
    depth_norm = cv2.normalize(
        np.nan_to_num(depth_vis, nan=0.0), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    path_depth = os.path.join(out_path, "depth.png")
    cv2.imwrite(path_depth, depth_norm)

    # ---- Quick sanity check ----
    cy, cx = depth_m.shape[0] // 2, depth_m.shape[1] // 2
    print(f"Center depth (m): {float(depth_m[cy, cx]):.3f}")

    # ---- Export point cloud ----
    path_cloud = os.path.join(out_path, "cloud.ply")
    export_pointcloud_ply(
        points_3d, rectL, disp_vis,
        ply_path=path_cloud,
        depth_min=depth_min_m,
        depth_max=depth_max_m,
        disp_min=1.0,
    )

    print("Saved:")
    for label, p in [("rectL", path_rectL), ("rectR", path_rectR),
                     ("disparity", path_disp), ("depth", path_depth),
                     ("cloud", path_cloud)]:
        print(f"  {p}")

    return {
        "rectL":     os.path.abspath(path_rectL),
        "rectR":     os.path.abspath(path_rectR),
        "disparity": os.path.abspath(path_disp),
        "depth":     os.path.abspath(path_depth),
        "cloud":     os.path.abspath(path_cloud),
    }


def main():
    p = argparse.ArgumentParser(
        description="Compute stereo depth map and point cloud from a calibrated stereo pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--calib",  default=_DEFAULT_CALIB_NPZ, help="Path to calibration .npz file")
    p.add_argument("--left",   default=_DEFAULT_LEFT_IMG,  help="Path to left-camera image")
    p.add_argument("--right",  default=_DEFAULT_RIGHT_IMG, help="Path to right-camera image")
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR,  help="Output directory for results")
    p.add_argument("--depth-min", type=float, default=DEPTH_MIN_M, help="Minimum depth (m) for visualisation")
    p.add_argument("--depth-max", type=float, default=DEPTH_MAX_M, help="Maximum depth (m) for visualisation")
    args = p.parse_args()

    compute_depth(
        calib_npz=args.calib,
        left_img=args.left,
        right_img=args.right,
        out_dir=args.out_dir,
        depth_min_m=args.depth_min,
        depth_max_m=args.depth_max,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())