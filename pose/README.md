# Tactile pose prototype (VEGA challenge, Black Forest Hackathon)

Logged grasp → 3D contact points → fit the object's STL → pose + 3D image of hand and object.

## Folder layout

```
tactile_pose/
  logs/                    <- unzip logs.zip here (one subfolder per session)
  inspire_hand/            <- the WHOLE folder from dex-urdf: robots/hands/inspire_hand
    inspire_hand_right.urdf
    meshes/
  objects/
    vegabar29.stl          <- rename "...-VEGABAR_29_STL"
    vegabar38.stl          <- rename "...-VEGABAR_38"
    standin_pressure_sensor.stl
  config.py, run_pipeline.py, ...
```

Windows hides `.stl` endings ("RoboDK Object"): turn on View > Show > File name extensions
before renaming.

## Run

```bash
pip install trimesh yourdfpy plotly rtree scipy matplotlib pandas fast_simplification
python run_pipeline.py --session 20261016_204637      # one session, object from its label
python run_pipeline.py                                # all labelled sessions
python run_pipeline.py --object standin --session 20261016_204637   # force an object
python synthetic_test.py                              # checks the fitter on a known pose
```

`--object auto` (default) picks the STL from `SESSION_LABELS` and `LABEL_TO_OBJECT` in
`config.py`: Sensor A and B from your session list. Unlabelled test sessions are skipped.
One session takes about 20 seconds.

Output per session in `output/<object>/`: `<session>.png`, interactive `<session>.html`
(open in a browser, rotate with the mouse) and `results.csv` with the pose of every session.

## Before trusting results (placeholders in `config.py`)

1. **Hand model**: copy the whole `robots/hands/inspire_hand` folder from
   https://github.com/dexsuite/dex-urdf into the project (see layout above).
   Set `HAND_SIDE` to the hand you used.
   The URDF is the standard RH56 hand; check link lengths against the RH56DFTP manual.
2. **VEGA STLs**: copy `VEGABAR 29 STL.stl` and `VEGABAR 38.stl` into `objects/`. Convert the
   pressure transducer STEP to STL (FreeCAD: File > Export) and save it as
   `objects/1109903_00-Druckaufnehmer.stl`. `standin` is a guessed shape for testing only.
3. **Tactile layout** (`REGION_LAYOUT`, `ROW0_AT_TIP`, `PALM_*_RANGE`): where each grid sits on
   the finger and which way its rows run. Check against `hand.png` and the manual: press one
   corner of a patch and see which row/column lights up in `hand_gui.py`.
4. **Angle mapping** (`RAW_OPEN`, `DOF_TO_JOINT`): linear 1000 = open, 0 = closed. Replace with the
   manual's angle table if it has one.
5. **Object choice**: the classifier isn't ready, so session labels choose the STL. Check that
   `LABEL_TO_OBJECT` maps Sensor A and B to the right STLs.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Load + hold window | `data_io.py` | Reads a session, keeps frames where fingers are closed and pressure is near peak, averages them |
| Hand kinematics | `hand_model.py` | Raw 0–1000 angles → joint angles (coupled joints from URDF) → link poses |
| Taxels → 3D | `hand_model.py` | Ray-casts every taxel grid onto the palm-facing surface of its link mesh |
| Contacts | `contacts.py` | Per region: pressure-weighted 3D centroid + surface normal |
| Pose fit | `pose_fit.py` | Multi-start least squares: contacts on the STL surface, normals opposite, no penetration of the hand |
| Quality | `pose_fit.py` | Fit RMS (mm), normal error, penetration, ambiguity (other poses fitting almost as well) |
| Slip | `run_pipeline.py` | Contact-centroid drift and pressure change during the hold |
| Output | `render.py` | PNG + interactive HTML of hand, object, taxels and contacts |

## What the results mean

- `pos_*_mm`, `axis_*`: object centre and long axis in the hand frame (`hand_base_link`).
- `T_hand_obj`: 4x4 pose of the object frame (centred, long axis = z).
  `T_hand_cad`: pose of the original CAD frame; use this for placement.
- `status`: `ok` (RMS < 2 mm, unique), `poor fit` or `ambiguous`.
- Rotation about the object's own axis and position along it are only observable where the
  shape changes (thread, hex, connector). The synthetic test shows this: axis within about 1–6°,
  lateral position within 2 mm, along-axis error up to about 9 mm when contacts sit on the plain body.

## Status of the current run

All real sessions use the stand-in shape, so fits are poor (2–18 mm RMS). This is expected:
the stand-in is 33 mm wide while the contacts suggest a smaller gap, and the layout placeholders
aren't calibrated yet. Swap in the real STL and check items 1–4 above.
