#!/usr/bin/env python3
"""
Stereo Camera Pipeline
======================

Orchestrates the full stereo-vision workflow in two modes:

  calibrate  – Capture a calibration video, extract frames, run ChArUco
               stereo calibration, and save the result.

  depth      – Capture a scene video, extract one frame pair, compute a
               depth map and export a coloured point cloud.

Data layout produced
--------------------
  data/
    calibration/
      <timestamp>/
        videos/     raw .mkv files from capture_sync_video
        frames/     left_NNNN.png / right_NNNN.png extracted frames
      calib.npz     latest calibration result (overwritten each run)
    sessions/
      <session>/
        videos/     raw .mkv files
        frames/     extracted frames
        output/
          rectL.png       rectified left image
          rectR.png       rectified right image
          disparity.png   WLS-filtered disparity visualisation
          depth.png       depth map visualisation
          cloud.ply       coloured point cloud

Quick-start examples
--------------------
  # Step 1+2 – move the ChArUco board in view of both cameras during capture
  python pipeline.py calibrate --time 10s

  # Step 3 – capture a scene and compute depth
  python pipeline.py depth --session my_scene --time 2s

  # Re-run calibration on existing frames (skip capture)
  python pipeline.py calibrate --no-capture --frames-dir data/calibration/20240101_120000/frames

  # Re-run depth on existing images (skip capture)
  python pipeline.py depth --no-capture --session my_scene
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

# Locate sibling scripts relative to this file so the pipeline works
# regardless of the current working directory.
_HERE = Path(__file__).parent

DATA_DIR         = Path("data")
CALIB_DATA_DIR   = DATA_DIR / "calibration"
SESSIONS_DIR     = DATA_DIR / "sessions"
DEFAULT_CALIB_NPZ = CALIB_DATA_DIR / "calib.npz"

CAPTURE_SCRIPT = _HERE / "capture_sync_video"  # intentionally no .py extension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, check: bool = True):
    """Print and run a command, raising on failure when *check* is True."""
    print(">>", " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd])
    if check and result.returncode != 0:
        sys.exit(f"Command failed (rc={result.returncode}): {' '.join(str(c) for c in cmd)}")
    return result.returncode


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _find_videos(video_dir: Path):
    """Return (cam0_path, cam1_path) from a videos directory."""
    cam0_files = sorted(video_dir.glob("*_cam0_*"))
    cam1_files = sorted(video_dir.glob("*_cam1_*"))
    if not cam0_files or not cam1_files:
        sys.exit(
            f"No cam0/cam1 video files found in {video_dir}.\n"
            "Expected filenames matching *_cam0_* and *_cam1_*."
        )
    return cam0_files[0], cam1_files[0]


# ---------------------------------------------------------------------------
# Calibration step
# ---------------------------------------------------------------------------

def cmd_calibrate(args):
    ts = _timestamp()

    session_dir = CALIB_DATA_DIR / ts
    videos_dir  = session_dir / "videos"
    frames_dir  = session_dir / "frames"

    # Allow the user to point at a pre-existing frames directory so that
    # both capture AND extraction can be skipped in one go.
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        if not frames_dir.exists():
            sys.exit(f"--frames-dir does not exist: {frames_dir}")

    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Capture ----
    if not args.no_capture:
        print("\n=== Step 1/3: Capturing calibration video ===")
        print(
            "  Move the ChArUco board slowly in front of both cameras.\n"
            f"  Recording for {args.time} ..."
        )
        _run([
            sys.executable, CAPTURE_SCRIPT,
            "--time", args.time,
            "--base", "calib",
            "--outdir", str(videos_dir),
            "--ext", ".mkv",
        ])
    else:
        print("\n=== Step 1/3: Capture skipped (--no-capture) ===")

    # ---- Step 2a: Extract frames (skip if --frames-dir was given) ----
    if args.frames_dir:
        print(f"\n=== Step 2/3: Using existing frames in {frames_dir} ===")
    else:
        print("\n=== Step 2/3: Extracting calibration frames ===")
        cam0, cam1 = _find_videos(videos_dir)
        from extract_frames import extract_frames as _extract_frames
        n = _extract_frames(
            str(cam0),
            str(cam1),
            out_dir=str(frames_dir),
            every_n=args.every_n,
        )
        if n == 0:
            sys.exit("Frame extraction produced no frames. Check the video files.")

    # ---- Step 2b: Calibrate ----
    print("\n=== Step 3/3: Running stereo calibration ===")
    out_npz = args.out_npz or str(DEFAULT_CALIB_NPZ)
    DEFAULT_CALIB_NPZ.parent.mkdir(parents=True, exist_ok=True)

    from calibration_ChArUco import run_calibration
    run_calibration(
        frames_dir=str(frames_dir),
        out_npz=out_npz,
        show_detections=False,
    )

    print(f"\nCalibration complete. Result saved to: {out_npz}")


# ---------------------------------------------------------------------------
# Depth step
# ---------------------------------------------------------------------------

def cmd_depth(args):
    session_name = args.session or _timestamp()
    session_dir  = SESSIONS_DIR / session_name
    videos_dir   = session_dir / "videos"
    frames_dir   = session_dir / "frames"
    output_dir   = session_dir / "output"

    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Capture ----
    if not args.no_capture:
        print("\n=== Step 1/3: Capturing scene video ===")
        _run([
            sys.executable, CAPTURE_SCRIPT,
            "--time", args.time,
            "--base", "scene",
            "--outdir", str(videos_dir),
            "--ext", ".mkv",
        ])
    else:
        print("\n=== Step 1/3: Capture skipped (--no-capture) ===")

    # ---- Step 2: Extract one frame pair ----
    print("\n=== Step 2/3: Extracting frame pair ===")
    cam0, cam1 = _find_videos(videos_dir)
    from extract_frames import extract_frames as _extract_frames
    n = _extract_frames(
        str(cam0),
        str(cam1),
        out_dir=str(frames_dir),
        every_n=1,
        max_frames=args.frame + 1,
    )
    if n == 0:
        sys.exit("Frame extraction produced no frames. Check the video files.")

    left_img  = frames_dir / f"left_{args.frame:04d}.png"
    right_img = frames_dir / f"right_{args.frame:04d}.png"
    if not left_img.exists() or not right_img.exists():
        sys.exit(
            f"Frame {args.frame} not found after extraction. "
            f"The video may be shorter than expected."
        )

    # ---- Step 3: Compute depth ----
    print("\n=== Step 3/3: Computing depth map and point cloud ===")
    calib_npz = args.calib or str(DEFAULT_CALIB_NPZ)
    if not Path(calib_npz).exists():
        sys.exit(
            f"Calibration file not found: {calib_npz}\n"
            "Run 'python pipeline.py calibrate' first."
        )

    from depth_map import compute_depth
    results = compute_depth(
        calib_npz=calib_npz,
        left_img=str(left_img),
        right_img=str(right_img),
        out_dir=str(output_dir),
        depth_min_m=args.depth_min,
        depth_max_m=args.depth_max,
    )

    print(f"\nDepth estimation complete. Results saved to: {output_dir}")
    for name, path in results.items():
        print(f"  {name}: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    root = argparse.ArgumentParser(
        description="Stereo Camera Pipeline – orchestrates capture, calibration, and depth estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ---- calibrate ----
    cal = sub.add_parser(
        "calibrate",
        help="Capture calibration video, extract frames, and run stereo calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cal.add_argument(
        "--time", default="10s",
        help="Capture duration passed to rpicam-vid (e.g. 10s, 15000ms).",
    )
    cal.add_argument(
        "--every-n", type=int, default=30, metavar="N",
        help="Use every Nth frame for calibration (30 = 1 fps from a 30 fps video).",
    )
    cal.add_argument(
        "--out-npz", default=None,
        help=f"Destination for calibration .npz (default: {DEFAULT_CALIB_NPZ}).",
    )
    cal.add_argument(
        "--no-capture", action="store_true",
        help="Skip video capture; use videos already present in the session directory.",
    )
    cal.add_argument(
        "--frames-dir", default=None,
        help="Use a pre-existing frames directory instead of extracting from video "
             "(implies --no-capture is also needed to skip capture).",
    )
    cal.set_defaults(func=cmd_calibrate)

    # ---- depth ----
    dep = sub.add_parser(
        "depth",
        help="Capture a scene video, extract a frame pair, and compute depth + point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dep.add_argument(
        "--time", default="1s",
        help="Capture duration passed to rpicam-vid.",
    )
    dep.add_argument(
        "--session", default=None,
        help="Session name / directory under data/sessions/ (default: current timestamp).",
    )
    dep.add_argument(
        "--frame", type=int, default=0,
        help="Index of the frame pair to process (0-based).",
    )
    dep.add_argument(
        "--calib", default=None,
        help=f"Path to calibration .npz file (default: {DEFAULT_CALIB_NPZ}).",
    )
    dep.add_argument(
        "--depth-min", type=float, default=0.01,
        help="Minimum depth (metres) shown in depth visualisation.",
    )
    dep.add_argument(
        "--depth-max", type=float, default=0.50,
        help="Maximum depth (metres) shown in depth visualisation.",
    )
    dep.add_argument(
        "--no-capture", action="store_true",
        help="Skip video capture; use videos already present in the session directory.",
    )
    dep.set_defaults(func=cmd_depth)

    args = root.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
