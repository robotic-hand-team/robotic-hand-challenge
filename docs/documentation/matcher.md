# `matcher/`: DTW session matcher

This is the first version of the session matcher, which uses **DTW only**. It finds the recorded
grasp most similar to a query session. The final model in [`final_model/`](../../final_model/) adds
an XGBoost object classifier in front of it and **imports its functions from here**, so this folder
is still needed.

## Files

| File | What it does |
|---|---|
| `match_session.py` | Hold-window detection, 12-channel series extraction, multivariate DTW distance, top-k search |
| `eval_loo.py` | Leave-one-out accuracy of the matcher on labelled sessions |

### Functions that `final_model/` uses

| Function | Purpose |
|---|---|
| `detect_hold_window()` | Longest window where every finger moves less than 20 units per frame (fallback: middle 50%) |
| `extract_series()` | Hold window → 12 channels: per-finger + palm tactile activation (×3 weight) and 6 joint angles. Down-sampled to 60 frames or fewer and z-scored |
| `dtw_distance()` | Multivariate DTW (`dtaidistance.dtw_ndim`) |

## Running

```bash
python matcher/match_session.py <query_session> --lib <library_dir> --top 5
```

Like `final_model/`, the default `--lib` is a folder on a teammate's PC, so pass your own, for
example `--lib models/dataset`.

`eval_loo.py` has its library and label paths **hardcoded** to a teammate's PC
(`~/Documents/20261017_17October_8pm` and `~/Documents/labelled_data/manual_labels.csv`). Neither is
in the repo, so it won't run as-is. Edit `LIB` and `LABELS` at the top of the file first.
