from picamera2 import Picamera2
from libcamera import controls
import time

SIZE = (1920, 1080)

cam = Picamera2(0)
cam.configure(cam.create_preview_configuration(main={"size": SIZE}))
cam.start()
time.sleep(1)

# 1) Autofocus
cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
cam.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
time.sleep(0.7)

# 2) Read the focused lens position
req = cam.capture_request()
meta = req.get_metadata()
lens_pos = meta.get("LensPosition", None)
req.release()
print("LensPosition:", lens_pos)

# 3) Lock it
if lens_pos is not None:
    cam.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": float(lens_pos)
    })
