# Grasp Inference — Quick Start

Run on any session folder to get a prediction image you can compare with the real world.

---

## 1. Prepare your session folder

Your folder needs exactly these two files:

```
my_session/
  tactile_summary.csv     ← tactile sensor readings (one row per frame)
  actuators.csv           ← finger joint angles    (one row per frame)
```

Same structure as the folders in `models/dataset_sim/`.  
`object_pose.csv` and `meta.json` are optional (sim only — enables ground-truth overlay).

---

## 2. Run

```bash
python3 demo/run.py path/to/my_session/
```

**Example with a sim session (for testing):**
```bash
python3 demo/run.py models/dataset_sim/20261017_110506_w0_0000
```

---

## 3. Output

Two images are saved in `demo/`:

### `<session>_compare.png` ← the one to compare with the camera

Front view of the hand (lateral × height):

| Marker | Meaning |
|--------|---------|
| **Red circle + arrow** | Predicted object centre + predicted tilt direction |
| **Green triangle + arrow** | Ground-truth position + actual tilt (sim only) |

Hold this image next to your camera feed and check:
- Is the red circle roughly where the object is?
- Does the red arrow point in the tilt direction you see?

### `<session>_result.png` ← full diagnostics (8 panels)

3D hand skeleton, slip timeline, front/side/top projections, error over time.

---

## 4. What the models predict

| Output | Model | Accuracy |
|--------|-------|----------|
| XYZ position (mm) | `pose_regressor.pkl` | ~9.6 mm RMSE |
| Tilt: upright / TILTED (≥10°) | `tilt_classifier.pkl` | ~91 % |
| Slip: stable / slipping | `slip_detector.pkl` | frame-level |

Models are loaded automatically from `classifier/model/`.

---

## 5. Known limitations

- **Tilt direction** shown in red arrow comes from predicted pitch/roll — accuracy is limited (~57° RMSE). Use it as a rough indicator, not a precise angle.
- **Tilt around Z axis** (object rotating away from palm) is barely covered in training data (max ±10°). Predictions outside this range are unreliable.
- Models were trained on **simulation data only**. Real hardware may show a distribution shift.
