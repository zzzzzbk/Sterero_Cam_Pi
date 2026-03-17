# Stereo Camera Pi

A stereo-vision pipeline for Raspberry Pi 5 using two synchronized camera 3

The pipeline covers the full workflow:

1. **Capture** – record synchronized video from both cameras
2. **Calibrate** – extract frames and compute the stereo camera calibration (ChArUco board)
3. **Depth** – apply the calibration to compute a depth map and export a coloured point cloud

Two depth estimation backends are available:

| Backend | Script | Quality | Hardware |
|---------|--------|---------|----------|
| OpenCV SGBM (default) | `depth_map.py` | Good | CPU only |
| **FoundationStereo** (neural network) | `depth_map_foundation.py` | **Best-in-class** | CUDA GPU required |

---

## Project structure

```
.
├── pipeline.py                  ← main orchestration script (start here)
├── capture_sync_video           ← records two synchronized rpicam-vid streams
├── extract_frames.py            ← extracts left/right frame pairs from video files
├── calibration_ChArUco.py       ← stereo calibration from ChArUco board images
├── depth_map.py                 ← disparity + depth map + point cloud (OpenCV SGBM)
├── depth_map_foundation.py      ← disparity + depth map + point cloud (FoundationStereo)
├── setup_foundation_stereo.sh   ← downloads FoundationStereo source (no nested git)
├── check_rectify.py             ← visual epipolar-line check of rectification
├── check_sync.py                ← side-by-side sync check for extracted frames
├── capture_single_image.py      ← one-shot stereo image capture (picamera2)
├── autofocus.py                 ← autofocus both cameras and save lens positions
│
├── FoundationStereo/            ← FoundationStereo source (populated by setup script)
│
└── data/                        ← all generated data (contents gitignored)
    ├── calibration/
    │   ├── videos/      ← raw .mkv files from capture_sync_video
    │   └── frames/      ← left_NNNN.png / right_NNNN.png pairs
    │   └── calib.npz        ← saved calibration (overwritten each run)
    └── sessions/
        └── <session_name>/
            ├── videos/      ← raw .mkv files
            ├── frames/      ← extracted frame pairs
            └── output/
                ├── rectL.png        rectified left image
                ├── rectR.png        rectified right image
                ├── K.txt            camera intrinsics + baseline (FoundationStereo format)
                ├── disparity.png    disparity visualisation
                ├── depth.png        depth map visualisation
                ├── depth_meter.npy  metric depth array (FoundationStereo backend)
                └── cloud.ply        coloured point cloud
```

---

## Quick start

### Prerequisites

hardware: 

raspberry pi 5 with dual CSI port

2 raspberry pi camera 3

3d printed camera mount

```
pip install opencv-contrib-python numpy open3d 
```

### Step 1 + 2 – Calibrate

Place the ChArUco board in view of both cameras, then run:

```bash
python pipeline.py calibrate --time 10s
```

Move the board around during the 10-second recording so many different poses
are captured.  The script will:

1. Record `data/calibration/<timestamp>/videos/calib_cam0_*.mkv` and `calib_cam1_*.mkv`
2. Extract one frame per second (`--every-n 30`) into `data/calibration/<timestamp>/frames/`
3. Run the ChArUco stereo calibration and save the result to `data/calibration/calib.npz`

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--time` | `10s` | Recording duration (passed to `rpicam-vid`) |
| `--every-n N` | `30` | Use every Nth frame (30 = 1 fps from 30 fps video) |
| `--out-npz FILE` | `data/calibration/calib.npz` | Calibration output path |
| `--no-capture` | off | Skip video recording; use existing videos |
| `--frames-dir DIR` | auto | Use a pre-existing frames directory |

### Step 3 – Depth estimation (OpenCV SGBM — default)

Point the cameras at a scene and run:

```bash
python pipeline.py depth --session my_scene --time 2s
```

The script will:

1. Record `data/sessions/my_scene/videos/scene_cam0_*.mkv` and `scene_cam1_*.mkv`
2. Extract the first frame pair into `data/sessions/my_scene/frames/`
3. Compute the depth map and point cloud, saving everything to `data/sessions/my_scene/output/`

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--time` | `1s` | Recording duration |
| `--session NAME` | timestamp | Session directory name under `data/sessions/` |
| `--frame N` | `0` | Frame index to process (0-based) |
| `--calib FILE` | `data/calibration/calib.npz` | Calibration .npz to use |
| `--depth-min M` | `0.01` | Minimum depth (metres) for visualisation |
| `--depth-max M` | `0.50` | Maximum depth (metres) for visualisation |
| `--no-capture` | off | Skip video recording; use existing videos |

### Step 3 – Depth estimation (FoundationStereo — neural network)

See the **FoundationStereo Integration** section below for full setup instructions.
Once set up, pass `--use-foundation-stereo` to the `depth` subcommand:

```bash
python pipeline.py depth --session my_scene --time 2s \
    --use-foundation-stereo \
    --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --depth-min 0.1 --depth-max 10.0
```

Additional FoundationStereo flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--use-foundation-stereo` | off | Enable FoundationStereo backend |
| `--ckpt PATH` | see script | Path to model checkpoint (.pth) |
| `--scale F` | `1.0` | Inference scale (≤1; use `0.5` for speed) |
| `--valid-iters N` | `32` | Recurrent update iterations (quality vs speed) |

---

## FoundationStereo Integration

[FoundationStereo](https://github.com/NVlabs/FoundationStereo) (CVPR 2025 Oral, Best Paper Nomination) is a
foundation model for stereo depth estimation with state-of-the-art zero-shot accuracy.

`depth_map_foundation.py` integrates FoundationStereo into this pipeline:

1. Uses the same `.npz` calibration file as `depth_map.py`
2. Rectifies images with the stored remap matrices
3. Automatically generates `K.txt` (rectified intrinsics + baseline) from the calibration
4. Runs FoundationStereo inference for a dense, high-quality disparity map
5. Converts disparity to metric depth: `depth = fx × baseline / disparity`
6. Saves the same output layout as `depth_map.py` (plus `K.txt` and `depth_meter.npy`)

### Setup

**Step 1 — Get FoundationStereo source (no nested git)**

```bash
bash setup_foundation_stereo.sh
```

This clones `NVlabs/FoundationStereo` into `FoundationStereo/` and **removes the nested
`.git`** so the code is plain files — not a submodule.  To make it available when cloning
this repo to another machine, commit it once:

```bash
git add FoundationStereo/
git commit -m "Add FoundationStereo source (no nested git)"
```

**Step 2 — Install dependencies**

```bash
conda env create -f FoundationStereo/environment.yml
conda activate foundation_stereo
pip install flash-attn
```

**Step 3 — Download pretrained model weights**

Download the `23-51-11` folder from the
[FoundationStereo model page](https://github.com/NVlabs/FoundationStereo#model-weights)
and place it under `FoundationStereo/pretrained_models/`:

```
FoundationStereo/
└── pretrained_models/
    └── 23-51-11/
        ├── model_best_bp2.pth
        └── cfg.yaml
```

> **Note:** Model weights are excluded from git via `.gitignore` because they are
> several GB.  Each developer downloads them separately.

**Step 4 — Run**

Standalone:

```bash
python depth_map_foundation.py \
    --calib  data/calibration/calib.npz \
    --left   data/sessions/my_scene/frames/left_0000.png \
    --right  data/sessions/my_scene/frames/right_0000.png \
    --out-dir data/sessions/my_scene/output/ \
    --ckpt   FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
```

Via pipeline:

```bash
python pipeline.py depth --session my_scene --no-capture \
    --use-foundation-stereo \
    --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
```

For high-resolution images or to speed up inference:

```bash
# Half-resolution inference (faster, less GPU memory)
python depth_map_foundation.py ... --scale 0.5

# Fewer recurrent iterations (faster, slightly lower quality)
python depth_map_foundation.py ... --valid-iters 16
```

### K.txt format

`K.txt` is generated automatically from the calibration `.npz`.  Its format matches
what FoundationStereo's `run_demo.py` expects:

```
fx 0 cx  0 fy cy  0 0 1   ← row-major 3×3 rectified intrinsic matrix (9 values)
0.065                      ← baseline between left and right cameras (metres)
```

---

## Running individual scripts

Each script can also be run on its own with `--help` for all options.

### `capture_sync_video`

Records two synchronized streams:

```bash
python capture_sync_video --time 5s --base test --outdir ./raw_video
```

### `extract_frames.py`

Extracts frame pairs from two video files:

```bash
python extract_frames.py raw_video/test_cam0.mkv raw_video/test_cam1.mkv \
       --out-dir frames/ --every-n 30
```

### `calibration_ChArUco.py`

Runs stereo calibration from a directory of frame pairs:

```bash
python calibration_ChArUco.py --frames-dir frames/ --out-npz calib.npz
```

### `depth_map.py` (OpenCV SGBM)

Computes depth from a calibrated stereo pair using SGBM:

```bash
python depth_map.py --calib calib.npz \
       --left left.png --right right.png --out-dir output/
```

### `depth_map_foundation.py` (FoundationStereo)

Computes depth using the FoundationStereo neural network:

```bash
python depth_map_foundation.py --calib calib.npz \
       --left left.png --right right.png --out-dir output/ \
       --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
```

### `check_rectify.py`

Visual epipolar-line check after calibration:

```bash
python check_rectify.py
```

---

## ChArUco board

The calibration uses an **8 × 6** ChArUco board (DICT_4X4_100) with:

- Square size: **15 mm**
- Marker size: **11 mm**

Print `board.png` (or generate it with OpenCV) at the correct physical size and
mount it flat on a rigid backing for best results.
