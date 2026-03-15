# Stereo Camera Pi

A stereo-vision pipeline for Raspberry Pi using two synchronized `rpicam-vid`
cameras and OpenCV.  The pipeline covers the full workflow:

1. **Capture** – record synchronized video from both cameras
2. **Calibrate** – extract frames and compute the stereo camera calibration (ChArUco board)
3. **Depth** – apply the calibration to compute a depth map and export a coloured point cloud

---

## Project structure

```
.
├── pipeline.py              ← main orchestration script (start here)
├── capture_sync_video       ← records two synchronized rpicam-vid streams
├── extract_frames.py        ← extracts left/right frame pairs from video files
├── calibration_ChArUco.py   ← stereo calibration from ChArUco board images
├── depth_map.py             ← disparity + depth map + point cloud (.ply)
├── check_rectify.py         ← visual epipolar-line check of rectification
├── check_sync.py            ← side-by-side sync check for extracted frames
├── capture_single_image.py  ← one-shot stereo image capture (picamera2)
├── autofocus.py             ← autofocus both cameras and save lens positions
│
└── data/                    ← all generated data (contents gitignored)
    ├── calibration/
    │   ├── <timestamp>/
    │   │   ├── videos/      ← raw .mkv files from capture_sync_video
    │   │   └── frames/      ← left_NNNN.png / right_NNNN.png pairs
    │   └── calib.npz        ← saved calibration (overwritten each run)
    └── sessions/
        └── <session_name>/
            ├── videos/      ← raw .mkv files
            ├── frames/      ← extracted frame pairs
            └── output/
                ├── rectL.png       rectified left image
                ├── rectR.png       rectified right image
                ├── disparity.png   WLS-filtered disparity visualisation
                ├── depth.png       depth map visualisation
                └── cloud.ply       coloured point cloud
```

---

## Quick start

### Prerequisites

```bash
pip install opencv-contrib-python numpy
# rpicam-vid must be available (Raspberry Pi OS with camera stack)
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

### Step 3 – Depth estimation

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

### `depth_map.py`

Computes depth from a calibrated stereo pair:

```bash
python depth_map.py --calib calib.npz \
       --left left.png --right right.png --out-dir output/
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
