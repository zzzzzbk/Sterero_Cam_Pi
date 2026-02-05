# capture_single_focus.py

from picamera2 import Picamera2
from libcamera import controls
import time
import cv2

# --------- USER PARAMETERS ----------
CAMERA_NUM = 0              # 0 or 1
IMAGE_SIZE = (1920, 1080)   # output resolution
FULL_RESOLUTION=(4608, 2592)
FAST_RESOLUTION=(2304, 1296)
LENS_POS = 5.38           # try 0.5 ~ 3.0 (float)
OUT_FILE = "output/test_2K.png"
# -----------------------------------

cam = Picamera2(camera_num=CAMERA_NUM)

# Use a still configuration (good quality, no preview overhead)
config = cam.create_still_configuration(
    main={"size": FULL_RESOLUTION}
)
cam.configure(config)

cam.start()
time.sleep(1.0)  # warm up sensor

# Set manual focus
cam.set_controls({
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": float(LENS_POS)
})

# Small delay to let lens settle
time.sleep(0.3)

# Capture image
image = cam.capture_array()
cv2.imwrite(OUT_FILE, image)

print(f"Saved image to {OUT_FILE}")

cam.stop()
