# `classifier/`: learned models

This folder holds the training, evaluation and visualisation scripts for the machine-learning models,
plus the trained models in `model/`. The models answer three questions:

1. **Which object is held?** Sensor A (VEGABAR 29) or Sensor B (VEGABAR 38).
2. **Where is it, and how is it tilted?** Position in mm, plus orientation or tilt class.
3. **Is it slipping?**

Run every script **from the repo root**, for example `python classifier/train_sim.py`.

## Scripts

### Object identity (Sensor A vs B)

| Script | Data | Model file(s) in `model/` |
|---|---|---|
| `train.py` | Real (13 labelled sessions) | `xgb_sensor.pkl`, `scaler.pkl`, `feature_cols.pkl` |
| `predict.py <session>` | Inference with the model above | prints per-frame predictions and a majority vote, writes `predictions.csv` into the session folder |
| `train_sim.py` | Simulation | `xgb_sensor_sim.pkl`, `scaler_sim.pkl`, `feature_cols_sim.pkl` |
| `train_combined.py` | Real + sim | `xgb_sensor_combined.pkl`, `scaler_combined.pkl`, `feature_cols_combined.pkl` |
| `train_mixed.py` | Real + sim, per-session z-score | `xgb_mixed.pkl` |
| `minirocket_train.py` | Real, time series (MiniROCKET) | `minirocket_sensor.pkl` (needs `aeon`) |
| `cnn_train.py` | Real, raw taxel grids (2D CNN) | `cnn_sensor.pt`, not committed (needs `torch`) |
| `evaluate_cross_domain.py` | Real ↔ sim | prints how well each model transfers to the other domain |

### Pose and tilt

These are trained on simulation only, with ground truth from `object_pose.csv` and `meta.json`.

| Script | Predicts | Model file |
|---|---|---|
| `train_pose.py` | Position xyz (mm) + roll/pitch/yaw | `pose_regressor.pkl` |
| `physics_pose.py` | Position from force-weighted contact centroids (interpretable baseline) | `physics_pose_ridge.pkl` |
| `train_tilt.py` | Upright vs tilted (≥ 10°) | `tilt_classifier.pkl` |
| `train_orientation.py` | 4-class roll (neutral / right / inverted / left) | `orientation_classifier.pkl` |
| `minirocket_orientation.py` | Upright vs tilted, MiniROCKET | `minirocket_orientation.pkl` |
| `cnn_orientation.py` | 3-class pitch (vertical / tilted / horizontal), CNN | needs `torch` |

### Slip

| Script | Predicts | Model file |
|---|---|---|
| `train_slip.py` | Per frame: stable vs slipping (object speed > 0.015 m/s while held) | `slip_detector.pkl` |

### Visualisation

| Script | Output |
|---|---|
| `visualize_pose.py [session]` | 3D hand skeleton + predicted (red) vs true (green) object position |
| `visualize_orientation.py` | `position_and_orientation.png` in the repo root: tilt directions and tilt histogram |

`visualize_pose.py` hardcodes `matplotlib.use("MacOSX")`. On Windows or Linux, change it to `"TkAgg"`.

## `model/`

Every model has its own **scaler** and **feature-column list**: `scaler_<name>.pkl` and
`feature_cols_<name>.pkl`. Always load the three files together. The sensor classifiers use 108
features: the tactile summary columns plus hand registers.

The models are pickles. [`classifier/requirements.txt`](../../classifier/requirements.txt) pins **scikit-learn 1.9.1** and **xgboost 3.2.0** exactly, because loading the models with other versions can fail or warn. Optional extras that aren't pinned: `matplotlib` for `visualize_*.py`, `torch` for `cnn_*.py` and `aeon` for `minirocket_*.py`.

## Who uses these models

| User | Models |
|---|---|
| `demo/run.py` | `pose_regressor`, `tilt_classifier`, `slip_detector` |
| `backend/src/` (dashboard) | `xgb_sensor*` for replay and the `per_frame` live mode; `physics_pose_ridge` (optional position shift). The default live Sensor A/B model is `final_model/sensor_classifier.pkl` |
| `simulation/hand_sim/classifier_bridge.py` | `xgb_sensor{,_sim,_combined}`, chosen by the `CLASSIFIER_SUFFIX` env var |

## How far to trust the numbers

- **Real-data object identity:** the trustworthy figure is leave-one-session-out cross-validation, about **89%**.
- **Cross-domain:** a model trained on real data is near chance on simulated data, and a sim-only model is *below* chance on real data. Simulated tactile values, current and temperature are approximations. The combined model's 100% on its 4-session real holdout is not a meaningful result.
- **Pose, tilt and slip** have only been validated on simulation. Expect them to be less accurate on the real hand.
