#!/usr/bin/env python3
"""
Stereo camera calibration using a ChArUco board.

Reads left_NNNN.png / right_NNNN.png pairs from a frames directory and
saves the full stereo calibration result (intrinsics, extrinsics,
rectification maps) to a .npz file.

Usage (standalone):
    python calibration_ChArUco.py --frames-dir <dir> --out-npz <file.npz>

Can also be imported and called programmatically:
    from calibration_ChArUco import run_calibration
    run_calibration("path/to/frames", "path/to/calib.npz")
"""

import argparse
import glob
import os
import sys

import numpy as np
import cv2

# -------- BOARD DEFAULTS ----------
BOARD_COLS = 8       # number of squares along X
BOARD_ROWS = 6       # number of squares along Y
SQUARE_LENGTH = 0.015   # square size in metres (15 mm)
MARKER_LENGTH = 0.011   # marker size in metres (11 mm, must be < SQUARE_LENGTH)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
# ----------------------------------

# Legacy defaults (kept for backward compatibility)
_DEFAULT_FRAMES_DIR = "data/calibration/frames"
_DEFAULT_OUT_NPZ    = "data/calibration/calib.npz"


def _sort_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    num = "".join([c for c in base if c.isdigit()])
    return int(num) if num else 0


def _calibrate_charuco_standard(all_corners, all_ids, board, img_size):
    """Prepare object/image point lists and run cv2.calibrateCamera."""
    all_object_points = []
    all_image_points = []
    board_corners_3d = board.getChessboardCorners()
    for corners, ids in zip(all_corners, all_ids):
        if ids is not None and len(ids) > 0:
            current_obj_points = board_corners_3d[ids, :]
            all_object_points.append(current_obj_points)
            all_image_points.append(corners)
    print(f"  Prepared {len(all_object_points)} sets for mono calibration.")
    return cv2.calibrateCamera(all_object_points, all_image_points, img_size, None, None)


def run_calibration(
    frames_dir: str,
    out_npz: str,
    show_detections: bool = False,
    skip:int = 0,
    BOARD_COLS: int = 8,       # number of squares along X
    BOARD_ROWS: int = 6,       # number of squares along Y
    SQUARE_LENGTH: float = 0.015,   # square size in metres (15 mm)
    MARKER_LENGTH: float = 0.011,   # marker size in metres (11 mm, must be < SQUARE_LENGTH)
) -> str:
    """
    Run stereo ChArUco calibration on frame pairs in *frames_dir* and save
    the result to *out_npz*.

    Args:
        frames_dir:       Directory containing left_NNNN.png / right_NNNN.png
                          frame pairs produced by extract_frames.py.
        out_npz:          Destination path for the calibration .npz file.
                          Parent directories are created automatically.
        show_detections:  If True, display detection overlays in a window
                          (requires a graphical display).

    Returns:
        Absolute path to the saved .npz file.
    """
    left_paths  = sorted(glob.glob(os.path.join(frames_dir, "left_*.png")),  key=_sort_key)
    right_paths = sorted(glob.glob(os.path.join(frames_dir, "right_*.png")), key=_sort_key)

    if len(left_paths) == 0 or len(right_paths) == 0:
        raise RuntimeError(f"No images found in '{frames_dir}'. Check the path and filename patterns.")
    if len(left_paths) != len(right_paths):
        raise RuntimeError(f"Mismatched counts: left={len(left_paths)} right={len(right_paths)}")

    # ---- Build ChArUco board ----
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=ARUCO_DICT,
    )
    board.setLegacyPattern(True)
    detector = cv2.aruco.CharucoDetector(board)

    # ---- Collect detections ----
    image_size = None

    all_charuco_corners_L, all_charuco_ids_L = [], []
    all_charuco_corners_R, all_charuco_ids_R = [], []

    objpoints_stereo  = []
    imgpointsL_stereo = []
    imgpointsR_stereo = []

    used_pairs = 0
    processed_pairs = 0
    last_id = -1

    for lp, rp in zip(left_paths, right_paths):
        processed_pairs += 1

        img_l = cv2.imread(lp, cv2.IMREAD_COLOR)
        img_r = cv2.imread(rp, cv2.IMREAD_COLOR)
        if img_l is None or img_r is None:
            print(f"  Skip unreadable pair: {lp}, {rp}")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])

        charuco_corners_L, charuco_id_L, marker_corners_L, marker_ids_L = detector.detectBoard(gray_l)
        charuco_corners_R, charuco_id_R, marker_corners_R, marker_ids_R = detector.detectBoard(gray_r)

        if charuco_corners_L is None or charuco_corners_R is None:
            print(f"  ChArUco not found in pair: {os.path.basename(lp)} / {os.path.basename(rp)}")
            continue

        idsL_flat = charuco_id_L.flatten()
        idsR_flat = charuco_id_R.flatten()
        common = np.intersect1d(idsL_flat, idsR_flat)

        if len(common) < 15:
            print(f"  Too few common corners in pair: {os.path.basename(lp)} / {os.path.basename(rp)}")
            continue

        if last_id != -1 and processed_pairs - last_id < 10:
            print(f"  Skipping pair (temporal proximity, ID {processed_pairs})")
            continue

        # Save for mono calibration
        all_charuco_corners_L.append(charuco_corners_L)
        all_charuco_ids_L.append(charuco_id_L)
        all_charuco_corners_R.append(charuco_corners_R)
        all_charuco_ids_R.append(charuco_id_R)

        # Build stereo correspondences from common corner IDs
        dictL = {int(i): charuco_corners_L[k, 0, :] for k, i in enumerate(idsL_flat)}
        dictR = {int(i): charuco_corners_R[k, 0, :] for k, i in enumerate(idsR_flat)}
        board_obj = board.getChessboardCorners()

        obj, ptsL, ptsR = [], [], []
        for cid in common:
            cid = int(cid)
            obj.append(board_obj[cid])
            ptsL.append(dictL[cid])
            ptsR.append(dictR[cid])

        obj  = np.array(obj,  dtype=np.float32).reshape(-1, 3)
        ptsL = np.array(ptsL, dtype=np.float32).reshape(-1, 1, 2)
        ptsR = np.array(ptsR, dtype=np.float32).reshape(-1, 1, 2)

        objpoints_stereo.append(obj)
        imgpointsL_stereo.append(ptsL)
        imgpointsR_stereo.append(ptsR)
        used_pairs += 1
        last_id = processed_pairs

        if show_detections:
            visL, visR = img_l.copy(), img_r.copy()
            if marker_ids_L is not None:
                cv2.aruco.drawDetectedMarkers(visL, marker_corners_L, marker_ids_L)
            if marker_ids_R is not None:
                cv2.aruco.drawDetectedMarkers(visR, marker_corners_R, marker_ids_R)
            cv2.aruco.drawDetectedCornersCharuco(visL, charuco_corners_L, charuco_id_L)
            cv2.aruco.drawDetectedCornersCharuco(visR, charuco_corners_R, charuco_id_R)
            both = np.hstack([visL, visR])
            window_name = "ChArUco detections (L | R)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1750, 500)
            cv2.imshow(window_name, both)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if used_pairs < 10:
        raise RuntimeError(
            f"Too few valid stereo pairs ({used_pairs}). "
            "Aim for 20-50 good pairs spread across the frame."
        )

    print(f"Using {used_pairs} valid stereo pairs. Image size: {image_size}")

    # ---- 1) Mono calibration for each camera ----
    print("Calibrating left camera ...")
    retL, K_l, D_l, _, _ = _calibrate_charuco_standard(
        all_charuco_corners_L, all_charuco_ids_L, board, image_size
    )
    print("Calibrating right camera ...")
    retR, K_r, D_r, _, _ = _calibrate_charuco_standard(
        all_charuco_corners_R, all_charuco_ids_R, board, image_size
    )
    print(f"Mono RMS  Left: {retL:.4f}  Right: {retR:.4f}")

    # ---- 2) Stereo calibration (fix intrinsics, solve extrinsics) ----
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    retStereo, K_l2, D_l2, K_r2, D_r2, R, T, E, F = cv2.stereoCalibrate(
        objpoints_stereo,
        imgpointsL_stereo,
        imgpointsR_stereo,
        K_l, D_l,
        K_r, D_r,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print(f"Stereo RMS: {retStereo:.4f}")
    print(f"Estimated baseline (m): {float(np.linalg.norm(T)):.4f}")

    # ---- 3) Rectification & maps ----
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, D_l, K_r, D_r,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1,
    )
    mapLx, mapLy = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, image_size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, image_size, cv2.CV_32FC1)

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez(
        out_npz,
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
    print(f"Saved calibration to: {out_npz}")
    return os.path.abspath(out_npz)


def main():
    p = argparse.ArgumentParser(
        description="Stereo camera calibration using a ChArUco board.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--frames-dir",
        default=_DEFAULT_FRAMES_DIR,
        help="Directory containing left_NNNN.png / right_NNNN.png frame pairs",
    )
    p.add_argument(
        "--out-npz",
        default=_DEFAULT_OUT_NPZ,
        help="Output .npz calibration file path",
    )
    p.add_argument(
        "--show-detections",
        action="store_true",
        help="Show ChArUco detection overlays (requires a graphical display)",
    )
    args = p.parse_args()

    run_calibration(
        frames_dir=args.frames_dir,
        out_npz=args.out_npz,
        show_detections=args.show_detections,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
