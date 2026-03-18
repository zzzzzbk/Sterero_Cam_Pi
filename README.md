# Stereo Camera Pi

This project turns a Raspberry Pi 5 + dual camera setup into a practical stereo depth pipeline. In one flow, you record synchronized camera data, calibrate the stereo pair with a ChArUco board, then generate a depth map and colored point cloud for a scene.

## What you get

- Synchronized capture from two Raspberry Pi Camera 3 modules
- Stereo calibration file: `data/calibration/calib.npz`
- Scene outputs: rectified images, disparity, depth map, and `cloud.ply`
- Two depth backends:
  - OpenCV SGBM: fast, CPU-friendly
  - FoundationStereo: highest quality, requires CUDA GPU

## 1) Setup (once)

Hardware:

- Raspberry Pi 5 (dual CSI)
- 2x Raspberry Pi Camera 3
- 3d printed Rigid stereo mount (located in hardware/)


## 2) Calibrate the stereo rig (do this first)

Goal: create `data/calibration/calib.npz`, used by all depth runs.

```bash
python pipeline.py calibrate --time 30s
```

What this does:

1. Records calibration videos from both cameras
2. Extracts frame pairs
3. Runs ChArUco stereo calibration
4. Saves `data/calibration/calib.npz`

Recalibration without capture new videos:

```bash
# recalibration, use existing videos/frames instead of new capture
python pipeline.py calibrate --no-capture

```

## 3) Run depth on a scene (default: OpenCV SGBM)

Goal: produce depth outputs for one scene session.

```bash
python pipeline.py depth --session my_scene --time 2s
```

What this does:

1. Captures synchronized scene videos
2. Extracts frame pair(s)
3. Computes disparity/depth/point cloud
4. Writes outputs to `data/sessions/my_scene/output/`

Common variant:

```bash
# Re-run processing on an existing session without recording again
python pipeline.py depth --session my_scene --no-capture
```

## 4) Use FoundationStereo backend (optional, higher quality)

### 4.1 Install FoundationStereo (once)

```bash
bash setup_foundation_stereo.sh
conda env create -f FoundationStereo/environment.yml
conda activate foundation_stereo
pip install flash-attn
```

Then download model weights from:

- https://github.com/NVlabs/FoundationStereo#model-weights

Place them here:

```text
FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
FoundationStereo/pretrained_models/23-51-11/cfg.yaml
```

### 4.2 Run depth with FoundationStereo

```bash
python pipeline.py depth --session my_scene --no-capture \
    --use-foundation-stereo \
    --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --depth-min 0.1 --depth-max 10.0
```

Speed/quality knobs:

```bash
# Faster, lower resolution inference
python pipeline.py depth --session my_scene --no-capture \
    --use-foundation-stereo \
    --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --scale 0.5

# Faster, slightly lower quality iterative refinement
python pipeline.py depth --session my_scene --no-capture \
    --use-foundation-stereo \
    --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --valid-iters 16
```

## 5) Output files you will check most often

In `data/sessions/<session>/output/`:

- `rectL.png`, `rectR.png`: rectified image pair
- `disparity.png`: disparity visualization
- `depth.png`: depth visualization
- `cloud.ply`: colored 3D point cloud
- `depth_meter.npy`: metric depth array (FoundationStereo path)
- `K.txt`: generated camera intrinsics + baseline (FoundationStereo path)

To view the generated point cloud
```bash
# open and visualize an .ply file 
python Visual_PointCloud.py

# to save a roated view of pointcloud 
pyhton rotated_visualizatin.py
```

## 6) Minimal direct-script examples

Use these if you want to bypass `pipeline.py`.

```bash
# Stereo calibration from extracted frame pairs
python calibration_ChArUco.py --frames-dir frames/ --out-npz calib.npz

# OpenCV depth from a single stereo pair
python depth_map.py --calib calib.npz --left left.png --right right.png --out-dir output/

# FoundationStereo depth from a single stereo pair
python depth_map_foundation.py --calib calib.npz --left left.png --right right.png \
    --out-dir output/ --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
```

## 7) Quick troubleshooting

- Calibration looks bad: recapture with more board poses and better lighting.
- Images fail to rectify: make sure frame resolution matches calibration resolution.
- FoundationStereo checkpoint load errors on PyTorch 2.6+: use the updated `depth_map_foundation.py` fallback logic (`weights_only=False` for trusted checkpoints).
- Point cloud too sparse/noisy: tune depth range (`--depth-min`, `--depth-max`) and recapture with better texture/lighting.

## ChArUco board settings

- Board: 8x6, `DICT_4X4_100`
- Square size: 15 mm
- Marker size: 11 mm

Print at correct physical scale and keep the board flat for best calibration accuracy.
