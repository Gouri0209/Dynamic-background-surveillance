# Smart Surveillance System with Dynamic Background Modeling

> **CSE3010 Computer Vision — BYOP Capstone Project**

A real-time object detection and tracking system designed for surveillance scenarios where the background is **not static** — swaying trees, flickering lights, moving crowds, or camera jitter. Classical fixed-background subtraction fails in these conditions; this project addresses that using adaptive background modeling, morphological filtering, and optional optical flow tracking.

---

## Problem Statement

Standard surveillance systems struggle when the background is dynamic. A tree swaying in the wind, a flickering light, or waves in a water body confuse simple frame differencing into generating thousands of false positives. This project implements a **robust detection pipeline** that separates true foreground objects from dynamic background noise.

---

## Syllabus Coverage (CSE3010)

| Module | Concept Applied |
|--------|----------------|
| Module 3 | Feature extraction, Edge detection, Contour analysis |
| Module 3 | Image segmentation (foreground/background separation) |
| Module 4 | Background subtraction & modeling (MOG2) |
| Module 4 | Optical Flow — Lucas-Kanade (KLT) tracking |
| Module 4 | HOG descriptor for pedestrian detection |
| Experiment 12 | Object detection from dynamic background for surveillance |

---

## Project Structure

```
surveillance_project/
├── src/
│   ├── detector.py          # Core detection engine (MOG2 + morphology + LK flow)
│   ├── main.py              # Entry point — webcam or video file
│   ├── evaluate.py          # Offline evaluation + threshold sensitivity analysis
│   └── demo_synthetic.py   # Self-contained demo (no camera needed)
├── output/                  # Saved videos / screenshots / reports (auto-created)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/smart-surveillance-cv.git
cd smart-surveillance-cv
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

All commands are run from the `src/` directory:
```bash
cd src
```

### Run on webcam (default)
```bash
python main.py
```

### Run on a video file
```bash
python main.py --source path/to/video.mp4
```

### Enable Optical Flow tracking + foreground mask display
```bash
python main.py --source video.mp4 --flow --show-mask
```

### Run headless (no GUI, just process and save)
```bash
python main.py --source video.mp4 --save --no-display
```

### Tune MOG2 sensitivity
```bash
# Lower threshold = more sensitive (more detections, possibly more noise)
python main.py --threshold 30 --min-area 500
```

---

## Synthetic Demo (no camera needed)

If you don't have a video file or webcam available, run the built-in synthetic scene:
```bash
python demo_synthetic.py
python demo_synthetic.py --objects 5 --flow --mask --save
```

This generates a procedural scene with:
- A slowly shifting background (sine-wave brightness drift + per-frame noise)
- Multiple colored rectangles moving and bouncing across the frame

---

## Evaluation

Run the detector on a video file and get a statistical report:
```bash
python evaluate.py path/to/video.mp4
```

Compare MOG2 sensitivity at different thresholds (helps with tuning):
```bash
python evaluate.py path/to/video.mp4 --compare-thresholds
```

This prints a table like:

```
── Threshold Sensitivity Analysis ──────────────────
   Threshold   TotalDet  AvgDet/Frame    AvgMs
──────────────────────────────────────────────────
        20.0        847          4.24      8.3
        40.0        521          2.61      8.1
        60.0        312          1.56      8.0
        80.0        198          0.99      7.9
       100.0        103          0.52      7.8
```

---

## Keyboard Controls (during live playback)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause / Resume |
| `S` | Save screenshot to `output/` |

---

## How It Works

```
Frame input
    │
    ▼
MOG2 Background Subtraction
(adaptive Gaussian mixture model)
    │
    ▼
Shadow removal + Morphological cleaning
(open → close → dilate)
    │
    ▼
Contour detection + Area / aspect-ratio filtering
    │
    ├──► [Optional] HOG pedestrian confirmation
    │
    ├──► [Optional] Lucas-Kanade Optical Flow tracking
    │
    ▼
Bounding box drawing + HUD overlay
    │
    ▼
Display / Save
```

**MOG2** models each pixel as a mixture of Gaussians, adapting over time. This allows it to absorb slow background changes while still flagging fast-moving foreground objects.

**Morphological operations** clean up the binary foreground mask — opening removes tiny noise blobs, closing fills gaps inside objects, dilation merges fragmented detections.

**Lucas-Kanade Optical Flow** tracks feature points inside each detected bounding box across consecutive frames, visualising motion vectors.

---

## Example Output

| Panel | Description |
|-------|-------------|
| Annotated frame | Original frame with green bounding boxes and object labels |
| FG Mask | Binary foreground mask after MOG2 + morphological filtering |
| Optical Flow | Frame with green motion vectors and red point markers |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python >= 4.8` | All CV operations (MOG2, HOG, LK flow, drawing) |
| `numpy >= 1.24` | Array operations |

---

## Author

**[Your Name]**  
CSE3010 — Computer Vision  
[Your University Name]  
[Your Roll Number]
