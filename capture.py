from picamera2 import Picamera2
import cv2, time, os

outdir = "calib"
os.makedirs(outdir, exist_ok=True)

SIZE = (1920, 1080)

cam0 = Picamera2(camera_num=0)
cam1 = Picamera2(camera_num=1)

cfg0 = cam0.create_still_configuration(main={"size": SIZE})
cfg1 = cam1.create_still_configuration(main={"size": SIZE})

cam0.configure(cfg0)
cam1.configure(cfg1)

cam0.start()
cam1.start()
time.sleep(1.0)  # warm up

for i in range(10):
    img0 = cam0.capture_array()
    img1 = cam1.capture_array()

    cv2.imwrite(f"{outdir}/left_{i:02d}.png", img0)
    cv2.imwrite(f"{outdir}/right_{i:02d}.png", img1)

    print("Saved pair", i)
    time.sleep(2)

cam0.stop()
cam1.stop()