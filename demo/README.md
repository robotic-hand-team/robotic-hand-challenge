# Grasp Inference & Visualization

Quick tool to run all models on a session folder and get a result image.

## Input format

A session folder must contain these two files:

```
my_session/
  tactile_summary.csv    ← tactile sensor readings (one row per frame)
  actuators.csv          ← finger joint angles (one row per frame)
```

Optionally (sim only — enables ground-truth overlay):
```
  object_pose.csv        ← true object position + phase label
```

Same structure as the dataset folders in `models/dataset_sim/`.

## Usage

```bash
# analyse a specific session → saves <session_name>_result.png
python demo/run.py path/to/my_session/

# use a default sim session (for testing)
python demo/run.py
```

The script saves a PNG next to `run.py` and prints a summary:

```
Session : my_session
Frames  : 87  |  ground truth: no
Position: X=+42.3  Y=+98.7  Z=+112.1  mm (mean over hold frames)
Tilt    : upright  (81% conf)
Slip    :  3% of frames flagged as slipping
```

## What the image shows

| Panel | Description |
|-------|-------------|
| **3D view** (top-left) | Hand skeleton + red sphere = predicted object centre |
| **Info** (top-centre) | Position, tilt, slip values for the selected frame |
| **Slip timeline** (top-right) | Per-frame slip probability (orange) |
| **Front / Side / Top** (bottom-left) | 2D projections of hand + object |
| **XYZ over time** (bottom-right) | Predicted position per frame (or error if GT available) |

Green triangle = ground truth (only visible when `object_pose.csv` present).

## Interactive mode

Uncomment the last line in `run.py` to open an animated window instead of saving a PNG:

```python
# run(session, save_png=png_out)   ← comment this out
run(session, animate=True)         ← uncomment this
```

Controls: `SPACE` = pause/resume · `←` / `→` = step through frames

## Models used

All models live in `classifier/model/` and are loaded automatically.

| Model | Task | Accuracy |
|-------|------|----------|
| `pose_regressor.pkl` | XYZ position (mm) | ~9.6 mm RMSE |
| `tilt_classifier.pkl` | upright vs tilted (≥10°) | ~70 % |
| `slip_detector.pkl` | stable vs slipping | frame-level |

## Requirements

```
numpy pandas scikit-learn xgboost joblib scipy matplotlib
```
