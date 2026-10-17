"""
Tilt orientation classifier from tactile data.

Detects how much the grasped object tilts from the nominal grasp pose.
Uses tilt_x_rad and tilt_z_rad from sim meta.json as ground-truth labels:

  tilt_magnitude = sqrt(tilt_x_rad² + tilt_z_rad²)  (degrees)
  Class 0: "upright"  tilt_mag <  10°  (object close to nominal orientation)
  Class 1: "tilted"   tilt_mag >= 10°  (object noticeably tilted)

Key insight: tilt changes the contact asymmetry across fingers and the
cy (along-finger) centroid of each patch — these spatial patterns are
captured by hold-phase mean tactile features.

Split: 80/20 session level.  Training only on simulation data.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

HERE      = Path(__file__).parent
SIM_DIR   = HERE.parent / "models" / "dataset_sim"
MODEL_DIR = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
TILT_THRESH_DEG = 10.0
CLASS_NAMES = ["upright (<10°)", "tilted (≥10°)"]


def tilt_label(meta: dict) -> int:
    ep  = meta["episode_info"]
    tx  = np.rad2deg(ep["tilt_x_rad"])
    tz  = np.rad2deg(ep["tilt_z_rad"])
    mag = np.sqrt(tx ** 2 + tz ** 2)
    return int(mag >= TILT_THRESH_DEG)


def load_session(session_path: Path) -> tuple | None:
    meta = json.loads((session_path / "meta.json").read_text())
    label = tilt_label(meta)

    tac  = pd.read_csv(session_path / "tactile_summary.csv")
    act  = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    pose = pd.read_csv(session_path / "object_pose.csv")[["frame", "phase"]]
    df   = tac.merge(act, on="frame").merge(pose, on="frame")
    hold = df[df["phase"] == "hold"]
    if len(hold) < 5:
        return None

    feat_cols = [c for c in tac.columns if c not in META_COLS] + ACTUATOR_COLS
    hold = hold.fillna(0)
    row  = hold[feat_cols].mean()  # session summary = mean over hold frames
    return row.values.astype(np.float32), label, feat_cols


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading simulation sessions (hold-phase mean)...")
records = []
for s in sorted(SIM_DIR.iterdir()):
    res = load_session(s)
    if res is not None:
        records.append(res)

feat_cols = records[0][2]
X_all = np.stack([r[0] for r in records])
y_all = np.array([r[1] for r in records])
print(f"  {len(records)} sessions  |  {len(feat_cols)} features")
for i, name in enumerate(CLASS_NAMES):
    n = (y_all == i).sum()
    print(f"  [{i}] {name}: {n} sessions ({n/len(y_all)*100:.1f}%)")

# ── 80/20 split ───────────────────────────────────────────────────────────────
np.random.seed(42)
idx    = np.random.permutation(len(records))
n_tr   = int(len(records) * 0.8)
tr_idx = idx[:n_tr]
te_idx = idx[n_tr:]

X_train, y_train = X_all[tr_idx], y_all[tr_idx]
X_test,  y_test  = X_all[te_idx], y_all[te_idx]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nSplit: {len(tr_idx)} train  /  {len(te_idx)} test sessions")

# ── Train ─────────────────────────────────────────────────────────────────────
spw = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
print(f"Training XGBoost (scale_pos_weight={spw:.2f})...")
model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)

print("\n" + "="*52)
print("  Tilt Orientation Classifier Results")
print("="*52)
print(f"  Session accuracy: {acc:.3f}  ({int(acc*len(y_test))}/{len(y_test)} correct)")
print(f"\n{classification_report(y_test, preds, target_names=['upright','tilted'])}")

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print(f"  Confusion matrix:")
print(f"    True upright → pred upright: {tn:3d}  ({tn/(tn+fp)*100:.1f}%)")
print(f"    True upright → pred tilted:  {fp:3d}  ({fp/(tn+fp)*100:.1f}%)")
print(f"    True tilted  → pred upright: {fn:3d}  ({fn/(fn+tp)*100:.1f}%)")
print(f"    True tilted  → pred tilted:  {tp:3d}  ({tp/(fn+tp)*100:.1f}%)")

# Top features
imp = model.feature_importances_
top = np.argsort(imp)[::-1][:10]
print(f"\n  Top 10 features:")
for i in top:
    print(f"    {feat_cols[i]:35s}  {imp[i]:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,     MODEL_DIR / "tilt_classifier.pkl")
joblib.dump(scaler,    MODEL_DIR / "scaler_tilt.pkl")
joblib.dump(feat_cols, MODEL_DIR / "feature_cols_tilt.pkl")
print(f"\nSaved → {MODEL_DIR}/tilt_classifier.pkl")
