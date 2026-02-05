from picamera2 import Picamera2
cam = Picamera2(0)
for m in cam.sensor_modes:
    print(m)
