# `demo/`: one-command grasp inference

Quick start: [README.md](../../demo/README.md)

This demo runs the learned models from `classifier/model/` on a recorded session. It produces
pictures that you can hold next to a camera image of the real grasp to compare.

## What it predicts

| Output | Model | Reported accuracy |
|---|---|---|
| Object position xyz (mm) | `pose_regressor.pkl` | about 9.6 mm RMSE |
| Upright vs tilted (≥ 10°) | `tilt_classifier.pkl` | about 70–91% depending on the measure |
| Stable vs slipping (per frame) | `slip_detector.pkl` | — |

All three models were trained on **simulation data only**.

## Running

From the repo root:

```bash
python demo/run.py models/dataset_sim/20261017_110506_w0_0000     # a simulated session
python demo/run.py models/dataset/20261016_204344                 # a real session
python demo/run.py                                                # first sim session
```

The input folder needs `tactile_summary.csv` and `actuators.csv`. `object_pose.csv`, which only
simulated sessions have, adds the ground truth to the pictures.

An interactive window opens (**SPACE** pauses, **← / →** step through frames), and two images are saved in
`demo/gen_pics/`:

| Image | Contents |
|---|---|
| `<session>_compare.png` | Front view of the hand. Red circle + arrow = predicted position + tilt; green triangle + arrow = ground truth (sim only) |
| `<session>_result.png` | 8-panel diagnostics: 3D hand skeleton, slip timeline, front / side / top projections, error over time |

## Files

| File | What it is |
|---|---|
| `run.py` | The demo script |
| `README.md` | How to read the output |
| `gen_pics/` | Generated images, created automatically |

## Limits

- The **tilt direction** arrow comes from the predicted pitch and roll, which are only rough (about 57° RMSE).
- Rotation that tips the object away from the palm is barely covered by the training data (±10° at most).
- Expect lower accuracy on real sessions than on simulated ones, because the models never saw real data.
