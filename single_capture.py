from picamera2 import Picamera2
from libcamera import controls
import time
import cv2
import glob
import os
import argparse

def shot(args):
    DIR = "output"

    FULL_RESOLUTION=(4608, 2592)
    FAST_RESOLUTION=(2304, 1296)
    LENS_POS = 5.38           # try 0.5 ~ 3.0 (float)
    cam0 = Picamera2(camera_num=0)
    cam1 = Picamera2(camera_num=1)

    cfg0 = cam0.create_still_configuration(main={"size": FULL_RESOLUTION})
    cfg1 = cam1.create_still_configuration(main={"size": FULL_RESOLUTION})

    cam0.configure(cfg0)
    cam1.configure(cfg1)

    cam0.start()
    cam1.start()

    time.sleep(1.0)  # warm up sensor

    # Set manual focus
    cam1.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": float(LENS_POS)
    })

    cam0.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": float(LENS_POS)
    })

    # Small delay to let lens settle
    time.sleep(0.3)

    img0 = cam0.capture_array()
    img1 = cam1.capture_array()

    cv2.imwrite(f"{DIR}/{args.filename}_left.png", img0)
    cv2.imwrite(f"{DIR}/{args.filename}_right.png", img1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Capture a single stereo pair for calibration.")
    parser.add_argument('--filename', type=str, default='test',help='a str: data file name')
    args = parser.parse_args()
    shot(args)