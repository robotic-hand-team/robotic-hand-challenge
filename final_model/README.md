# Final Model — Two-Stage Session Matcher

Given an unknown grasp session folder, returns the most similar session from the reference library.

---

## How it works

**Stage 1 — XGBoost sensor classifier**  
Extracts the mean tactile + joint-angle features over the detected hold phase and classifies the session as `sensor_A` (slim silver cylinder, yellow cap) or `sensor_B` (compact sensor with large round sensing element).

**Stage 2 — Multivariate DTW**  
Searches only within library sessions of the predicted sensor class using Dynamic Time Warping on a 12-channel hold-phase time series (per-finger tactile activation × 3 weight + joint angles).

---

## Setup

Install dependencies (from repo root):

```bash
pip install xgboost scikit-learn dtaidistance pandas numpy
```

---

## Usage

```bash
python3 final_model/match_session.py  <query_session_folder>  [--lib <dir>]  [--top <k>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `query` | — | Path to the query session folder |
| `--lib` | `~/Documents/20261017_17October_8pm` | Reference library directory |
| `--top` | `5` | Number of best matches to display |
| `--rebuild-index` | off | Force re-indexing the library (needed after new sessions are added) |

**Example:**
```bash
python3 final_model/match_session.py ~/Documents/20261017_17October_8pm/20261017_202049
```

The session folder must contain:
```
<session>/
  tactile_summary.csv
  actuators.csv
```

---

## Output

```
Query   : 20261017_202049
  XGBoost prediction → sensor_B
  Hold window: frames 176–284  →  60 frames × 12 channels

Library : /Users/.../20261017_17October_8pm
  Using cached index: library_index.pkl

  DTW search within sensor_B: 18 candidates…

Rank     DTW dist  Session
----------------------------------------------------------
  1         17.84  20261017_202449  ← best match
  2         20.49  20261017_202245
  ...

Best match: /Users/.../20261017_17October_8pm/20261017_202449
```

The library index is cached in `final_model/library_index.pkl` after the first run (~10 s to build). Pass `--rebuild-index` if you add new sessions to the library.

---

## Accuracy (leave-one-out on 33 labeled sessions)

| Stage | Accuracy |
|-------|----------|
| XGBoost sensor classification | 81.8 % (27/33) |
| DTW within correct class | 100 % |
| Combined end-to-end | **81.8 %** |

---

## Files

| File | Description |
|------|-------------|
| `match_session.py` | Main entry point |
| `sensor_classifier.pkl` | XGBoost + scaler + label encoder (retrained on real 8pm data) |
| `library_index.pkl` | Cached DTW series for the 72-session library (auto-generated) |
