from picamera2 import Picamera2
from libcamera import controls
import cv2, time, os



outdir = "calib"
#outdir = "output"
os.makedirs(outdir, exist_ok=True)

LENS_POS = 5.38    
SIZE = (1920, 1080)
FULL_RESOLUTION=(4608, 2592)
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

for i in range(20):
    img0 = cam0.capture_array()
    img1 = cam1.capture_array()

    cv2.imwrite(f"{outdir}/left_{i:02d}.png", img0)
    cv2.imwrite(f"{outdir}/right_{i:02d}.png", img1)

    print("Saved pair", i)
    time.sleep(4)

cam0.stop()
cam1.stop()