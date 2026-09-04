# Intelligent Retail Analytics System (SIH Problem Statement 179)
## Multi-Camera/Video Computer Vision Foundation & Multi-Object Tracking

This repository contains the computer-vision and multi-object tracking foundation for an **Intelligent Retail Analytics System** designed for Smart India Hackathon (SIH) Problem Statement 179.

The system supports real-time person detection, ByteTrack tracking, and experimental BoT-SORT tracking optimized for laptops powered by Intel processors (such as the Intel Core i5-1334U with Intel Iris Xe graphics) using CPU inference without requiring an NVIDIA CUDA GPU.

---

## Features

### Phase 1: Modular Person Detection
- **Ultralytics YOLO11 Nano (`yolo11n.pt`)**: Ultra-lightweight model delivering high accuracy with low CPU footprint.
- **Strict Person Filtering**: Only detects COCO class `0` (`person`), ignoring non-human objects.
- **Decoupled Architecture**: `PersonDetector` isolates YOLO inference and warmup, producing clean detection structures.

### Phase 2: Multi-Object Tracking & Trajectories
- **Dual Tracker Support**:
  - **ByteTrack** (`--tracker bytetrack`): Fast, lightweight spatial tracking using two-stage IoU association.
  - **BoT-SORT** (`--tracker botsort`): Advanced multi-object tracking integrating Global Motion Compensation (GMC) and appearance ReID options.
- **Persistent Anonymous Session IDs**: Assigns clean IDs (`Person 1`, `Person 2`, `Person 3`, etc.).
- **In-Memory Trajectory History**: Records spatial coordinate trails (`(cx, cy)` center and `(cx, y2)` ground-plane bottom-center) in `TrajectoryManager`.
- **Motion Trail Visualization**: Renders smooth trajectory paths showing shopper movement.
- **Detection-Only Mode**: Disable tracking via `--no-track` for Phase-1 benchmarking.

---

## Directory Structure

```text
sih-retail-analytics/
├── .gitignore             # Git ignore for venv, models, and caches
├── configs/
│   ├── __init__.py
│   └── config.py          # Centralized configuration (Detector, Tracker, Visualizer, Pipeline)
├── src/
│   ├── __init__.py
│   ├── detector.py        # PersonDetector (YOLO11n, CPU-enforced, structured DetectionBatch)
│   ├── tracker.py         # PersonTracker (ByteTrack on CPU), TrajectoryManager, TrackedPerson
│   ├── tracker_botsort.py # Experimental BotSortTracker (BoT-SORT on CPU)
│   ├── visualizer.py      # Bounding boxes, persistent ID labels, motion trails, HUD
│   └── pipeline.py        # Video capture loop, tracking orchestration, smoothed FPS
├── videos/
│   └── test.mp4           # Pedestrian surveillance sample video (OpenCV official sample, BSD 3-Clause)
├── main.py                # Configurable CLI entrypoint
├── requirements.txt       # Core dependencies: ultralytics, opencv-python, numpy, lap
└── README.md              # Project documentation
```

---

## Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows, Linux, or macOS

### 2. Activate Virtual Environment
```powershell
# In PowerShell (Windows):
.\venv\Scripts\Activate.ps1

# Or in Command Prompt (cmd):
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## How to Run

### 1. ByteTrack Tracking (Default)
```powershell
python main.py
```

### 2. BoT-SORT Tracking (Experimental)
```powershell
python main.py --tracker botsort
```

### 3. BoT-SORT with Static Camera Optimization (`--gmc-method none`)
For static retail CCTV cameras, disabling GMC saves significant CPU cycles:
```powershell
python main.py --tracker botsort --gmc-method none
```

### 4. Run with a Custom Video File
```powershell
python main.py --tracker botsort --source "path/to/your/retail_cctv.mp4"
```

### 5. Automated / Headless Benchmark Test
```powershell
python main.py --tracker botsort --video videos/test.mp4 --no-show --max-frames 100
```

---

## CLI Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` / `--source` | `str` | `videos/test.mp4` | Path to input video file |
| `--tracker` | `str` | `bytetrack` | Tracker algorithm: `bytetrack` or `botsort` |
| `--model` | `str` | `yolo11n.pt` | YOLO model name or local `.pt` path |
| `--conf` | `float` | `0.40` | Detection confidence threshold |
| `--iou` | `float` | `0.45` | NMS IoU threshold |
| `--imgsz` | `int` | `640` | Inference resolution (`640`, `480`, `320`) |
| `--device` | `str` | `cpu` | Hardware device (`cpu` explicitly configured) |
| `--no-track` | `flag`| `False`| Disable tracking (detection-only mode) |
| `--no-trail` | `flag`| `False`| Disable rendering trajectory motion trails |
| `--track-buffer` | `int` | `60` | Frame buffer before lost tracks are removed |
| `--track-high-thresh` | `float` | `0.35` | High-confidence 1st-stage association threshold |
| `--track-low-thresh` | `float` | `0.05` | Low-confidence 2nd-stage association threshold |
| `--new-track-thresh` | `float` | `0.35` | Minimum score to initiate a new tracklet |
| `--match-thresh` | `float` | `0.80` | IoU cost threshold for track association |
| `--gmc-method` | `str` | `sparseOptFlow`| BoT-SORT GMC method: `sparseOptFlow`, `none`, etc. |
| `--with-reid` | `flag`| `False`| Enable appearance ReID in BoT-SORT |
| `--output` | `str` | `None` | Optional path to write annotated MP4 video |
| `--max-frames` | `int` | `None` | Maximum number of frames to process |
| `--db-path` | `str` | `data/retail_edge.db` | Local SQLite database file path |
| `--db-interval` | `float` | `5.0` | Telemetry snapshot interval in seconds |
| `--api` | `flag`| `False`| Start local FastAPI Edge REST API server |
| `--api-host` | `str` | `127.0.0.1` | Host address for FastAPI Edge server |
| `--api-port` | `int` | `8000` | Port for FastAPI Edge server |

---

## Phase 6B: Local FastAPI Edge REST API

The local FastAPI Edge REST API exposes anonymous telemetry snapshots from SQLite (`data/retail_edge.db`) for in-store dashboards, POS integrations, and diagnostics without cloud dependency or PII transmission.

### 1. Starting the API Server

Using `uvicorn` directly:
```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Or using `main.py`:
```powershell
python main.py --api --api-host 127.0.0.1 --api-port 8000
```

### 2. Interactive API Documentation (Swagger UI)
Open your browser and navigate to:
```text
http://127.0.0.1:8000/docs
```

### 3. Available Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Service identification banner & online status |
| `/health` | `GET` | Health check verifying SQLite database connectivity (`200` or `503`) |
| `/api/latest` | `GET` | Most recent telemetry snapshot (`404` if empty) |
| `/api/snapshots?limit=100` | `GET` | List recent snapshots (limit capped at 1000) |
| `/api/summary` | `GET` | Aggregated metrics (entries, exits, occupancy, dwell, queue, sync count) |
| `/api/sync/status` | `GET` | Sync status (`pending_snapshots`, `sync_required`) |

---

## Tracker Comparison (795 Frames on `videos/test.mp4` - CPU)

| Metric | ByteTrack | BoT-SORT (`sparseOptFlow`) | BoT-SORT (`gmc=none`) |
| :--- | :--- | :--- | :--- |
| **Total Frames** | 795 | 795 | 795 |
| **Processing FPS (CPU)** | **18.16 FPS** | 13.88 FPS | 14.80 FPS |
| **Max Simultaneous Tracks** | 9 | 9 | 9 |
| **Total Unique Track IDs** | **32** | 34 | 36 |
| **ReID Support** | No (pure motion/IoU) | Supported (`--with-reid`) | Supported (`--with-reid`) |
