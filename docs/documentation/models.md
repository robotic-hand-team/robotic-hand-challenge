# `models/`: grasp recordings

This folder holds the **data** every other part of the repo trains on or reads: real recordings
from the hand and simulated recordings from the MuJoCo twin. Despite the folder name, the trained
models live in `classifier/model/` and `final_model/`.

## Contents

| Path | What it is |
|---|---|
| `dataset/` | **Real** sessions recorded on the physical hand (18 sessions) |
| `dataset/logs/` | CAD STEP-export logs. These are **not** sensor data |
| `dataset_sim/` | **Simulated** sessions from `simulation/hand_sim/generate_dataset.py` (about 700) |
| `classify_sensor.py` | Early Sensor A/B XGBoost classifier with exhaustive cross-validation |

## Session folder format

Every session is one folder. Real sessions are named `YYYYMMDD_HHMMSS`; simulated ones are named
`<timestamp>_w<worker>_<episode>`. Join the CSVs on `frame` or `t_unix`.

| File | Contents |
|---|---|
| `actuators.csv` | Per DOF: `angle_set_*`, `angle_act_*`, `force_act_*`, `current_*`, `error_*`, `status_*`, plus `pos_set_*`, `force_set_*`, `speed_set_*`, `pos_act_*`, `temp_*` in newer sessions |
| `tactile_raw.csv` | Every taxel, `<group>_r<row>_c<col>`, raw 0–4095 |
| `tactile_summary.csv` | Per sensor group: `sum`, `max`, `contacts` (taxels > 50), centroid `cy`/`cx`, mean `force` |
| `meta.json` | Sensor layout, calibration `offsets`, sample rate. Simulated sessions also store `object` and episode info such as `tilt_x_rad` and `tilt_z_rad` |
| `object_pose.csv` | **Simulation only:** ground-truth object position and orientation per frame |

The DOF suffixes are `little, ring, middle, index, thumb_bend, thumb_rot`.

There are two `actuators.csv` schemas:
- The first two real sessions (`20261016_201150`, `20261016_201443`) have 45 columns.
- Every later session has 69 columns. Treat that as the current schema.

## Labelled real sessions

| Object | Label | Sessions |
|---|---|---|
| VEGABAR 29 | Sensor A = 0 | `20261016_204344` `204402` `204429` `204457` `204513` `204524` `204548` |
| VEGABAR 38 | Sensor B = 1 | `20261016_204558` `204609` `204619` `204627` `204637` `204649` |

The other real sessions are unlabelled: early tests, plus `20261016_201150`, in which the hand wasn't initialised.

## Regenerating the simulated data

```bash
cd simulation
python -m hand_sim.generate_dataset --episodes 300 --out ../models/dataset_sim --workers 8
```

`dataset_sim/` is committed because the pose, tilt and slip models and `demo/` were trained on it.
Regenerating it changes the training data, so retrain afterwards and check the accuracy numbers again.

## `classify_sensor.py`

```bash
python models/classify_sensor.py
```

This trains an XGBoost Sensor A/B classifier on the 13 labelled real sessions. It evaluates every way of
holding out 2 sessions per class, 315 combinations in total. It is the earliest version of the classifier;
the maintained versions are in `classifier/` and `final_model/`.
