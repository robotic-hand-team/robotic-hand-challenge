"""
MiniROCKET orientation classifier on sim sessions.

Detects object tilt: are the fingers holding the object nearly horizontal or tilted?

  Class 0: "upright"  tilt_magnitude <  10°  (small tilt from nominal grasp pose)
  Class 1: "tilted"   tilt_magnitude >= 10°  (significant tilt)

where tilt_magnitude = sqrt(tilt_x_rad² + tilt_z_rad²) from meta.json episode_info.

Input: all-phase tactile_summary + actuator angles as multivariate time series.
Model: MiniROCKET (10 000 fixed kernels → PPV features) + Ridge classifier.
Split: 80/20 at session level, random seed 42.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

HERE      = Path(__file__).parent
SIM_DIR   = HERE.parent / "models" / "dataset_sim"
MODEL_DIR = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
TILT_THRESH_DEG = 10.0    # tilt_magnitude threshold for class 0 vs 1
MAX_FRAMES      = 120     # crop/pad each session to this length
CLASS_NAMES     = ["upright (<10°)", "tilted (≥10°)"]


def tilt_label(meta: dict) -> int:
    ep   = meta["episode_info"]
    tx   = np.rad2deg(ep["tilt_x_rad"])
    tz   = np.rad2deg(ep["tilt_z_rad"])
    mag  = np.sqrt(tx ** 2 + tz ** 2)
    return int(mag >= TILT_THRESH_DEG)


def load_session(session_path: Path, feat_cols: list | None = None):
    """Return (T, F) float32 array and int label, or None."""
    meta_path = session_path / "meta.json"
    if not meta_path.exists():
        return None

    meta  = json.loads(meta_path.read_text())
    label = tilt_label(meta)

    tac = pd.read_csv(session_path / "tactile_summary.csv")
    act = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df  = tac.merge(act, on="frame")
    if len(df) < 5:
        return None

    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in META_COLS]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feat_cols] = df[feat_cols].fillna(0)

    X = df[feat_cols].values.astype(np.float32)

    # Crop or pad to MAX_FRAMES
    T = len(X)
    if T >= MAX_FRAMES:
        X = X[:MAX_FRAMES]
    else:
        pad = np.zeros((MAX_FRAMES - T, X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=0)

    return X, label


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading sim sessions (all phases)...")
sessions = sorted(SIM_DIR.iterdir())

# First pass: gather feature columns from first valid session
feat_cols = None
for s in sessions:
    tac = pd.read_csv(s / "tactile_summary.csv")
    act = pd.read_csv(s / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df  = tac.merge(act, on="frame")
    feat_cols = [c for c in df.columns if c not in META_COLS]
    break

session_data = {}
for s in sessions:
    res = load_session(s, feat_cols)
    if res is not None:
        session_data[s.name] = res  # {name: (X, label)}

print(f"  Loaded {len(session_data)} sessions  |  {len(feat_cols)} features  |  T={MAX_FRAMES}")

labels_all = np.array([v[1] for v in session_data.values()])
for i, name in enumerate(CLASS_NAMES):
    n = (labels_all == i).sum()
    print(f"  [{i}] {name}: {n} sessions ({n/len(labels_all)*100:.1f}%)")

# ── 80/20 session split ───────────────────────────────────────────────────────
all_sessions = np.array(sorted(session_data.keys()))
np.random.seed(42)
idx        = np.random.permutation(len(all_sessions))
n_train    = int(len(all_sessions) * 0.8)
train_sess = set(all_sessions[idx[:n_train]])
test_sess  = set(all_sessions[idx[n_train:]])

print(f"\n  Train: {len(train_sess)} sessions  |  Test: {len(test_sess)} sessions")

X_train = np.stack([session_data[s][0] for s in all_sessions[idx[:n_train]]])
y_train = np.array([session_data[s][1] for s in all_sessions[idx[:n_train]]])
X_test  = np.stack([session_data[s][0] for s in all_sessions[idx[n_train:]]])
y_test  = np.array([session_data[s][1] for s in all_sessions[idx[n_train:]]])

# MiniROCKET expects (n_samples, n_channels, n_timesteps)
X_train = X_train.transpose(0, 2, 1)
X_test  = X_test.transpose(0, 2, 1)
print(f"  Input shape: {X_train.shape}  (sessions, features, timesteps)")

# ── MiniROCKET ────────────────────────────────────────────────────────────────
print("\nFitting MiniROCKET + Ridge...")
pipe = make_pipeline(
    MiniRocket(n_kernels=10_000, random_state=42),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced"),
)
pipe.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = pipe.predict(X_test)
acc   = accuracy_score(y_test, preds)

print("\n" + "="*52)
print("  MiniROCKET Orientation — Final Results")
print("="*52)
print(f"  Session accuracy: {acc:.3f}  ({int(acc*len(y_test))}/{len(y_test)} correct)")
print(f"\n{classification_report(y_test, preds, target_names=['upright','tilted'])}")

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print(f"  Confusion matrix:")
print(f"    True upright → pred upright: {tn:3d}  ({tn/(tn+fp)*100:.1f}%)")
print(f"    True upright → pred tilted:  {fp:3d}  ({fp/(tn+fp)*100:.1f}% false alarm)")
print(f"    True tilted  → pred upright: {fn:3d}  ({fn/(fn+tp)*100:.1f}% missed)")
print(f"    True tilted  → pred tilted:  {tp:3d}  ({tp/(fn+tp)*100:.1f}% caught)")

# Feature importance proxy: predict on held-out and inspect Ridge coefficients
ridge = pipe[-1]
print(f"\n  Ridge best alpha: {ridge.alpha_:.4g}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(pipe,      MODEL_DIR / "minirocket_orientation.pkl")
joblib.dump(feat_cols, MODEL_DIR / "feature_cols_orientation_mr.pkl")
print(f"\nSaved → {MODEL_DIR}/minirocket_orientation.pkl")
