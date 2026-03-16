#!/usr/bin/env python3
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

def build_cmd(camera: int,role:str, duration: str, output: Path) -> list[str]:
    return [
        "rpicam-vid",
        "--camera", str(camera),
        "--level", "4.2",
        "--framerate", "30",
        "--width", "1920",
        "--height", "1080",
        "--denoise", "cdn_off",
        "--profile", "baseline",
        "--inline",
        "--lens-position", "6",
        "--shutter", "2000",
        "--sync", role,
        "-n",
        "-t", duration,
        "-o", str(output),
    ]

def main():
    p = argparse.ArgumentParser(description="Record two synchronized rpicam-vid streams.")
    p.add_argument("-t", "--time", default="1s",
                   help="Capture duration passed to rpicam-vid, e.g. 1s, 2000ms (default: 1s)")
    p.add_argument("--base", default="test",
                   help="Base output name (default: test). Files become <base>_cam0_<ts>.mkv and <base>_cam1_<ts>.mkv")
    p.add_argument("--outdir", default=".",
                   help="Output directory (default: current directory)")
    p.add_argument("--ext", default=".mkv",
                   help="Output extension (default: .mkv). Use .h264 if you want raw H.264, etc.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = args.ext if args.ext.startswith(".") else "." + args.ext

    out0 = outdir / f"{args.base}_cam0_{ts}{ext}"
    out1 = outdir / f"{args.base}_cam1_{ts}{ext}"

    cmd0 = build_cmd(0,"client" ,args.time, out0)
    cmd1 = build_cmd(1, "server", args.time, out1)

    print("Starting camera 0:", " ".join(cmd0))
    p0 = subprocess.Popen(cmd0)

    print("Starting camera 1:", " ".join(cmd1))
    p1 = subprocess.Popen(cmd1)

    rc0 = p0.wait()
    rc1 = p1.wait()

    if rc0 != 0 or rc1 != 0:
        print(f"One or both commands failed: cam0 rc={rc0}, cam1 rc={rc1}", file=sys.stderr)
        return 1

    print("Done.")
    print("Saved:", out0)
    print("Saved:", out1)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())