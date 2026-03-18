#!/usr/bin/env python3
"""
Compute a stereo depth map using the FoundationStereo neural network.

Replaces the OpenCV SGBM-based disparity estimation in depth_map.py with
the NVlabs FoundationStereo foundation model for significantly improved
accuracy and zero-shot generalisation.

Requirements
------------
1. Run setup_foundation_stereo.sh to populate the FoundationStereo/
   directory as plain source files (no nested git / submodule):

       bash setup_foundation_stereo.sh

2. Install FoundationStereo dependencies:

       conda env create -f FoundationStereo/environment.yml
       conda activate foundation_stereo
       pip install flash-attn

3. Download pretrained model weights from
   https://github.com/NVlabs/FoundationStereo#model-weights
   and place the folder under FoundationStereo/pretrained_models/, e.g.:
       FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth

Usage (standalone)
------------------
    python depth_map_foundation.py \\
        --calib  data/calibration/calib.npz \\
        --left   data/sessions/my_scene/frames/left_0000.png \\
        --right  data/sessions/my_scene/frames/right_0000.png \\
        --out-dir data/sessions/my_scene/output/ \\
        --ckpt   FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth

Programmatic usage
------------------
    from depth_map_foundation import compute_depth_foundation
    results = compute_depth_foundation(
        calib_npz="data/calibration/calib.npz",
        left_img="left.png",
        right_img="right.png",
        out_dir="output/",
        ckpt_dir="FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
    )
"""

import argparse
import os
import sys
import pickle
import cv2
import numpy as np

# -------- paths --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_FOUNDATION_STEREO_DIR = os.path.join(_HERE, "FoundationStereo")

# -------- defaults (kept for backward-compatible standalone usage) --------
_DEFAULT_CAMERA_ID = 1
_DEFAULT_NAME      = "shared"
_DEFAULT_CALIB_NPZ = (
    f"camera_data/camera{_DEFAULT_CAMERA_ID}/stereo_calib_charuco.npz"
)
_DEFAULT_LEFT_IMG  = (
    f"camera_data/camera{_DEFAULT_CAMERA_ID}/output/"
    f"{_DEFAULT_NAME}{_DEFAULT_CAMERA_ID}_left.png"
)
_DEFAULT_RIGHT_IMG = (
    f"camera_data/camera{_DEFAULT_CAMERA_ID}/output/"
    f"{_DEFAULT_NAME}{_DEFAULT_CAMERA_ID}_right.png"
)
_DEFAULT_OUT_DIR   = f"processed/{_DEFAULT_NAME}{_DEFAULT_CAMERA_ID}"
_DEFAULT_CKPT      = os.path.join(
    _FOUNDATION_STEREO_DIR,
    "pretrained_models", "23-51-11", "model_best_bp2.pth",
)

DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 10.0
# -------------------------------------------------------------------------


def _check_foundation_stereo():
    """Raise an informative error if FoundationStereo source is not present."""
    marker = os.path.join(_FOUNDATION_STEREO_DIR, "core", "foundation_stereo.py")
    if not os.path.isfile(marker):
        raise RuntimeError(
            f"FoundationStereo not found at {_FOUNDATION_STEREO_DIR}.\n"
            "Run:  bash setup_foundation_stereo.sh\n"
            "See README.md for full setup instructions."
        )


def export_k_txt(calib_data, out_path: str):
    """
    Write a K.txt intrinsic file from calibration .npz data.

    FoundationStereo K.txt format:
      Line 1: space-separated row-major 3×3 intrinsic matrix (9 values)
      Line 2: baseline in metres

    Uses the *rectified* projection matrices (P1, P2) stored in the .npz
    file, so the values correctly describe the rectified image pair that is
    passed to FoundationStereo.

    For a rectified stereo rig:
      P1 = [K | 0]          (left camera)
      P2 = [K | [-fx·b, 0, 0]ᵀ]   (right camera, b = baseline in metres)

    So:  baseline = -P2[0,3] / fx

    Args:
        calib_data: Loaded numpy .npz archive (from np.load).
        out_path:   Destination path for K.txt.

    Returns:
        Tuple (K, baseline) where K is the 3×3 float64 intrinsic matrix and
        baseline is a float in metres.
    """
    P1 = calib_data["P1"].astype(np.float64)  # shape (3, 4)
    P2 = calib_data["P2"].astype(np.float64)  # shape (3, 4)

    K = P1[:3, :3]           # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = float(P1[0, 0])
    baseline = float(-P2[0, 3]) / fx  # metres

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(" ".join(f"{v:.8f}" for v in K.flatten()) + "\n")
        f.write(f"{baseline:.8f}\n")

    return K, baseline


def export_pointcloud_ply(
    points_3d,
    colors_bgr,
    disp,
    ply_path="cloud.ply",
    depth_min=0.1,
    depth_max=10.0,
    disp_min=1.0,
):
    """
    Write a coloured point cloud to an ASCII PLY file.

    Args:
        points_3d:  (H, W, 3) float32 from cv2.reprojectImageTo3D.
        colors_bgr: (H, W, 3) uint8 – rectified left image for colour.
        disp:       (H, W) float32 disparity in pixels.
        ply_path:   Destination file path.
        depth_min:  Minimum valid depth in metres.
        depth_max:  Maximum valid depth in metres.
        disp_min:   Minimum valid disparity in pixels.
    """
    Z = points_3d[:, :, 2]
    mask = (
        np.isfinite(Z) & (Z > depth_min) & (Z < depth_max)
        & np.isfinite(disp) & (disp > disp_min)
    )

    pts = points_3d[mask]
    col = colors_bgr[mask][:, ::-1]  # BGR -> RGB

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


def compute_depth_foundation(
    calib_npz: str,
    left_img: str,
    right_img: str,
    out_dir: str,
    ckpt_dir: str = _DEFAULT_CKPT,
    depth_min_m: float = DEPTH_MIN_M,
    depth_max_m: float = DEPTH_MAX_M,
    scale: float = 1.0,
    valid_iters: int = 32,
) -> dict:
    """
    Compute disparity, depth map, and point cloud using FoundationStereo.

    Workflow
    --------
    1. Load stereo calibration (.npz produced by calibration_ChArUco.py).
    2. Rectify the raw left/right images with the stored remap matrices.
    3. Generate K.txt (rectified intrinsics + baseline) for FoundationStereo.
    4. Run FoundationStereo forward pass on the rectified pair.
    5. Convert disparity → metric depth:  depth = fx · baseline / disparity
    6. Reproject to 3-D using the Q matrix and export a coloured point cloud.
    7. Save all outputs to *out_dir* in the same layout as depth_map.py.

    Args:
        calib_npz:   Path to the stereo calibration .npz file.
        left_img:    Path to the left-camera image.
        right_img:   Path to the right-camera image.
        out_dir:     Directory to write output files into.
        ckpt_dir:    Path to the FoundationStereo model checkpoint (.pth).
        depth_min_m: Minimum depth (metres) for the depth visualisation.
        depth_max_m: Maximum depth (metres) for the depth visualisation.
        scale:       Resize factor applied before inference (must be ≤1).
                     Use < 1 to speed up at the cost of resolution.
        valid_iters: Number of recurrent update iterations (quality vs speed).

    Returns:
        Dictionary mapping labels to absolute paths of the saved files:
        ``rectL``, ``rectR``, ``K_txt``, ``disparity``, ``depth``,
        ``depth_npy``, ``cloud``.
    """
    _check_foundation_stereo()

    # ---- Add FoundationStereo to Python path ----
    if _FOUNDATION_STEREO_DIR not in sys.path:
        sys.path.insert(0, _FOUNDATION_STEREO_DIR)

    import torch
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo  # noqa: E402 (dynamic import)
    from core.utils.utils import InputPadder             # noqa: E402

    if scale > 1.0:
        raise ValueError(f"scale must be <= 1.0, got {scale}")

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

    # ---- Generate K.txt from rectified calibration ----
    path_k_txt = os.path.join(out_path, "K.txt")
    K, baseline = export_k_txt(data, path_k_txt)
    print(
        f"Generated K.txt  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
        f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}  baseline={baseline:.4f} m"
    )

    # ---- Load FoundationStereo model ----
    ckpt_file = ckpt_dir
    cfg_path = os.path.join(os.path.dirname(ckpt_file), "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["scale"] = scale
    cfg["valid_iters"] = valid_iters
    cfg["hiera"] = 0

    model = FoundationStereo(cfg)
    # PyTorch 2.6+: default weights_only=True can fail on older checkpoint formats.
    try:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch that doesn't support weights_only kwarg
        ckpt = torch.load(ckpt_file, map_location="cpu")
    except pickle.UnpicklingError:
        print(
            "weights_only=True failed for this checkpoint format; "
            "retrying with weights_only=False (trusted checkpoint required)."
        )
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    # ckpt = torch.load(ckpt_file, map_location="cpu")
    # model.load_state_dict(ckpt["model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Loaded FoundationStereo from {ckpt_file}  (device={device})")

    # ---- Prepare tensors (FoundationStereo expects RGB float32) ----
    img0_np = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB).astype(np.float32)
    img1_np = cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB).astype(np.float32)

    if scale != 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img0_np = cv2.resize(img0_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img1_np = cv2.resize(img1_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    H_inf, W_inf = img0_np.shape[:2]

    img0_t = torch.as_tensor(img0_np, device=device)[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1_np, device=device)[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    # ---- FoundationStereo inference ----
    torch.autograd.set_grad_enabled(False)
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        disp_t = model.forward(img0_t, img1_t, iters=valid_iters, test_mode=True)

    disp_t = padder.unpad(disp_t.float())
    disp_np = disp_t.cpu().numpy().reshape(H_inf, W_inf)  # pixels at inference scale

    # ---- Convert disparity to metric depth ----
    # depth = fx * baseline / disparity   (fx adjusted for inference scale)
    fx_scaled = float(K[0, 0]) * scale
    disp_pos = disp_np.copy()
    disp_pos[disp_pos <= 0] = np.nan
    depth_m = np.where(np.isfinite(disp_pos), fx_scaled * baseline / disp_pos, np.nan)

    # ---- Save disparity visualisation ----
    disp_vis = np.nan_to_num(disp_pos, nan=0.0).astype(np.float32)
    disp_norm = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    path_disp = os.path.join(out_path, "disparity.png")
    cv2.imwrite(path_disp, disp_norm)

    # ---- Save depth visualisation ----
    depth_vis = depth_m.copy()
    depth_vis[(depth_vis < depth_min_m) | (depth_vis > depth_max_m)] = np.nan
    depth_norm = cv2.normalize(
        np.nan_to_num(depth_vis, nan=0.0), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    path_depth = os.path.join(out_path, "depth.png")
    cv2.imwrite(path_depth, depth_norm)

    # ---- Quick sanity check ----
    cy_inf, cx_inf = depth_m.shape[0] // 2, depth_m.shape[1] // 2
    print(f"Center depth (m): {float(depth_m[cy_inf, cx_inf]):.3f}")

    # ---- Save metric depth as numpy array for downstream processing ----
    path_depth_npy = os.path.join(out_path, "depth_meter.npy")
    np.save(path_depth_npy, depth_m.astype(np.float32))

    # ---- Export point cloud via Q-matrix reprojection ----
    # If inference used a smaller scale, upscale disparity back to full resolution
    # before reprojecting so the PLY coordinates match the colour image.
    if scale != 1.0:
        disp_full = (
            cv2.resize(disp_np, (w, h), interpolation=cv2.INTER_LINEAR) / scale
        )
    else:
        disp_full = disp_np

    disp_full_pos = disp_full.copy()
    disp_full_pos[disp_full_pos <= 0] = np.nan

    points_3d = cv2.reprojectImageTo3D(
        np.nan_to_num(disp_full_pos, nan=0.0).astype(np.float32), Q
    )
    path_cloud = os.path.join(out_path, "cloud.ply")
    export_pointcloud_ply(
        points_3d, rectL,
        np.nan_to_num(disp_full_pos, nan=0.0).astype(np.float32),
        ply_path=path_cloud,
        depth_min=depth_min_m,
        depth_max=depth_max_m,
        disp_min=1.0,
    )

    print("Saved:")
    for p in [path_rectL, path_rectR, path_k_txt, path_disp, path_depth,
              path_depth_npy, path_cloud]:
        print(f"  {p}")

    return {
        "rectL":     os.path.abspath(path_rectL),
        "rectR":     os.path.abspath(path_rectR),
        "K_txt":     os.path.abspath(path_k_txt),
        "disparity": os.path.abspath(path_disp),
        "depth":     os.path.abspath(path_depth),
        "depth_npy": os.path.abspath(path_depth_npy),
        "cloud":     os.path.abspath(path_cloud),
    }


def main():
    p = argparse.ArgumentParser(
        description=(
            "Compute stereo depth map and point cloud using FoundationStereo "
            "(replaces the SGBM-based depth_map.py)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--calib", default=_DEFAULT_CALIB_NPZ,
        help="Path to stereo calibration .npz file",
    )
    p.add_argument(
        "--left", default=_DEFAULT_LEFT_IMG,
        help="Path to left-camera image",
    )
    p.add_argument(
        "--right", default=_DEFAULT_RIGHT_IMG,
        help="Path to right-camera image",
    )
    p.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR,
        help="Output directory for results",
    )
    p.add_argument(
        "--ckpt", default=_DEFAULT_CKPT,
        help="Path to FoundationStereo checkpoint (.pth file)",
    )
    p.add_argument(
        "--depth-min", type=float, default=DEPTH_MIN_M,
        help="Minimum depth (m) for visualisation",
    )
    p.add_argument(
        "--depth-max", type=float, default=DEPTH_MAX_M,
        help="Maximum depth (m) for visualisation",
    )
    p.add_argument(
        "--scale", type=float, default=1.0,
        help=(
            "Inference scale factor (must be ≤1). "
            "Use 0.5 for faster inference at half resolution."
        ),
    )
    p.add_argument(
        "--valid-iters", type=int, default=32,
        help="Number of recurrent update iterations (quality vs speed trade-off)",
    )
    args = p.parse_args()

    compute_depth_foundation(
        calib_npz=args.calib,
        left_img=args.left,
        right_img=args.right,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt,
        depth_min_m=args.depth_min,
        depth_max_m=args.depth_max,
        scale=args.scale,
        valid_iters=args.valid_iters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
