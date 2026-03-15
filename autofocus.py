from picamera2 import Picamera2
from libcamera import controls
import time
import numpy as np

SIZE = (2304, 1296)
lp_list=[]
for i in range(2):
    cam = Picamera2(i)
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
    lp_list.append(lens_pos)

np.savez("AF_lens.npz", lens_left=lp_list[0], lens_right=lp_list[1])

# # 3) Lock it
# if lens_pos is not None:
#     cam.set_controls({
#         "AfMode": controls.AfModeEnum.Manual,
#         "LensPosition": float(lens_pos)
#     })
