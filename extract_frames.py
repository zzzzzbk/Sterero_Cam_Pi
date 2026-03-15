#!/usr/bin/env python3
"""
Extract synchronized frames from a pair of stereo video files.

Frames are saved as left_NNNN.png / right_NNNN.png in the output directory.
"""

import argparse
import sys

import cv2
from pathlib import Path


def extract_frames(
    video_left: str,
    video_right: str,
    out_dir: str,
    every_n: int = 1,
    max_frames: int = None,
) -> int:
    """
    Extract synchronized frame pairs from two stereo video files.

    Args:
        video_left:  Path to the left-camera video file.
        video_right: Path to the right-camera video file.
        out_dir:     Directory to write left_NNNN.png / right_NNNN.png files.
        every_n:     Save every Nth frame (default 1 = every frame).
        max_frames:  Stop after this many saved pairs (default None = all frames).

    Returns:
        Number of frame pairs saved.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    capL = cv2.VideoCapture(str(video_left))
    capR = cv2.VideoCapture(str(video_right))

    if not capL.isOpened():
        raise RuntimeError(f"Cannot open left video: {video_left}")
    if not capR.isOpened():
        raise RuntimeError(f"Cannot open right video: {video_right}")

    frame_idx = 0
    saved = 0

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL or not retR:
            break

        if frame_idx % every_n == 0:
            cv2.imwrite(str(out_path / f"left_{saved:04d}.png"), frameL)
            cv2.imwrite(str(out_path / f"right_{saved:04d}.png"), frameR)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    capL.release()
    capR.release()

    print(f"Extracted {saved} frame pairs to: {out_dir}")
    return saved


def main():
    p = argparse.ArgumentParser(
        description="Extract frames from synchronized stereo videos."
    )
    p.add_argument("video_left", help="Path to left-camera video file")
    p.add_argument("video_right", help="Path to right-camera video file")
    p.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for frame pairs (default: current directory)",
    )
    p.add_argument(
        "--every-n",
        type=int,
        default=1,
        metavar="N",
        help="Save every Nth frame (default: 1 = all frames)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="M",
        help="Stop after M saved pairs (default: all frames)",
    )
    args = p.parse_args()

    n = extract_frames(
        args.video_left,
        args.video_right,
        out_dir=args.out_dir,
        every_n=args.every_n,
        max_frames=args.max_frames,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
