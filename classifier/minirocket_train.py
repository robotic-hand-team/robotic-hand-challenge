"""
MiniROCKET sensor A/B classifier.

Each session = one multivariate time series (108 features × T frames).
MiniROCKET extracts random convolutional features, Ridge classifier on top.

Evaluation: Leave-One-Session-Out CV (13 folds)
Then trains final model on all data and saves to model/minirocket_sensor.pkl
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

HERE        = Path(__file__).parent
DATASET_DIR = HERE.parent / "models" / "dataset"
MODEL_DIR   = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

SESSIONS_A = [
    "20261016_204344", "20261016_204402", "20261016_204429",
    "20261016_204457", "20261016_204513", "20261016_204524", "20261016_204548",
]
SESSIONS_B = [
    "20261016_204558", "20261016_204609", "20261016_204619",
    "20261016_204627", "20261016_204637", "20261016_204649",
]

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
# Fixed window length (frames). Taken from the active-contact phase.
# 200 frames = 10s at 20Hz — covers the full stable-hold phase.
WINDOW_LEN = 200


def load_session_series(session: str, label: int) -> tuple[np.ndarray, int]:
    """Returns (T, F) float32 array for one session + label."""
    p = DATASET_DIR / session
    tac = pd.read_csv(p / "tactile_summary.csv")
    act = pd.read_csv(p / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df  = tac.merge(act, on="frame")

    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)

    # Use only active-contact frames (hand closed around object)
    contact_cols = [c for c in df.columns if c.endswith("_contacts")]
    active = df[contact_cols].sum(axis=1) > 20
    df_active = df[active].reset_index(drop=True)

    series = df_active[feat_cols].values.astype(np.float32)  # (T, F)
    return series, label


def crop_or_pad(series: np.ndarray, length: int) -> np.ndarray:
    """Crop to length from start, or zero-pad at end."""
    T, F = series.shape
    if T >= length:
        return series[:length]
    pad = np.zeros((length - T, F), dtype=np.float32)
    return np.concatenate([series, pad], axis=0)


# ── Load all sessions ─────────────────────────────────────────────────────────
print("Loading sessions...")
series_list, labels, session_names = [], [], []

for s in SESSIONS_A:
    ser, lbl = load_session_series(s, 0)
    series_list.append(ser); labels.append(lbl); session_names.append(s)
    print(f"  [A] {s}  active frames: {len(ser)}")

for s in SESSIONS_B:
    ser, lbl = load_session_series(s, 1)
    series_list.append(ser); labels.append(lbl); session_names.append(s)
    print(f"  [B] {s}  active frames: {len(ser)}")

labels        = np.array(labels)
session_names = np.array(session_names)

# Crop/pad to WINDOW_LEN, then reshape to (N, F, T) as aeon expects
X = np.stack([crop_or_pad(s, WINDOW_LEN) for s in series_list])  # (13, 200, F)
X = X.transpose(0, 2, 1)                                          # (13, F, 200)

print(f"\nInput shape: {X.shape}  (sessions × features × time)")
print(f"Sensor A: {(labels==0).sum()}  |  Sensor B: {(labels==1).sum()}")

# ── Leave-One-Session-Out CV ──────────────────────────────────────────────────
print(f"\nLeave-One-Session-Out CV ({len(session_names)} folds)...")
cv_accs, cv_preds_gt = [], []

for i, test_sess in enumerate(session_names):
    mask_test  = session_names == test_sess
    mask_train = ~mask_test

    X_train, y_train = X[mask_train], labels[mask_train]
    X_test,  y_test  = X[mask_test],  labels[mask_test]

    pipe = Pipeline([
        ("rocket", MiniRocket(n_kernels=10_000, random_state=42)),
        ("ridge",  RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)[0]

    correct = pred == y_test[0]
    cv_accs.append(float(correct))
    cv_preds_gt.append((pred, y_test[0]))

    sym  = "✓" if correct else "✗"
    lbl  = "A" if y_test[0] == 0 else "B"
    pred_lbl = "A" if pred == 0 else "B"
    print(f"  {sym} [{lbl}] {test_sess}  predicted={pred_lbl}")

cv_accs = np.array(cv_accs)
print(f"\n{'='*50}")
print(f"  MiniROCKET LOSO-CV")
print(f"  Accuracy: {cv_accs.mean():.3f}  ({int(cv_accs.sum())}/{len(cv_accs)} correct)")
print(f"{'='*50}")

# ── Final model on all data ───────────────────────────────────────────────────
print("\nTraining final model on all data...")
final_pipe = Pipeline([
        ("rocket", MiniRocket(n_kernels=10_000, random_state=42)),
    ("ridge",  RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
])
final_pipe.fit(X, labels)

joblib.dump({"pipeline": final_pipe, "window_len": WINDOW_LEN,
             "meta_cols": META_COLS, "actuator_cols": ACTUATOR_COLS},
            MODEL_DIR / "minirocket_sensor.pkl")

print(f"Saved → {MODEL_DIR}/minirocket_sensor.pkl")
