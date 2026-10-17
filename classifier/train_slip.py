"""
Small XGBoost slip detector.
Frame-level binary: stable (0) vs slipping (1)

Key insight: rate-of-change features are most informative for slip.
Features = current tactile summary + delta from previous frame.

Label: object velocity > 0.015 m/s during hold phase → slipping
Class imbalance handled via scale_pos_weight (~20:1).

Split: 80/20 session level.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

HERE      = Path(__file__).parent
SIM_DIR   = HERE.parent / "models" / "dataset_sim"
MODEL_DIR = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
VEL_THRESH = 0.015   # m/s — above this = slipping


def load_session(session_path: Path) -> pd.DataFrame | None:
    tac  = pd.read_csv(session_path / "tactile_summary.csv")
    act  = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    pose = pd.read_csv(session_path / "object_pose.csv")

    df = tac.merge(act, on="frame").merge(pose, on="frame")
    df = df[df["phase"] == "hold"].reset_index(drop=True)
    if len(df) < 5:
        return None

    # Slip label from velocity
    vel = np.sqrt(df["lin_vel_x"]**2 + df["lin_vel_y"]**2 + df["lin_vel_z"]**2)
    df["slip"] = (vel > VEL_THRESH).astype(int)

    feat_cols = [c for c in tac.columns if c not in META_COLS]
    for c in ACTUATOR_COLS:
        feat_cols.append(c)
    df[feat_cols] = df[feat_cols].fillna(0)

    # Delta features: change from previous frame (slip onset signal)
    delta = df[feat_cols].diff().fillna(0)
    delta.columns = [f"d_{c}" for c in feat_cols]
    df = pd.concat([df, delta], axis=1)

    df["session"] = session_path.name
    return df


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading sim sessions (hold phase)...")
frames = [df for s in sorted(SIM_DIR.iterdir())
          if (df := load_session(s)) is not None]
data = pd.concat(frames, ignore_index=True)

base_feat  = [c for c in data.columns
              if c not in META_COLS | {"slip","session","phase",
              "pos_x","pos_y","pos_z","quat_w","quat_x","quat_y","quat_z",
              "lin_vel_x","lin_vel_y","lin_vel_z","ang_vel_x","ang_vel_y","ang_vel_z","in_hand",
              "t_unix_x","t_unix_y","t_rel_x","t_rel_y","frame_x","frame_y"}
              and not c.startswith("d_")]
delta_feat = [c for c in data.columns if c.startswith("d_") and
              not any(x in c for x in ["t_unix","t_rel","frame"])]
feat_cols  = base_feat + delta_feat

n_stable = (data["slip"] == 0).sum()
n_slip   = (data["slip"] == 1).sum()
print(f"  {len(frames)} sessions  |  stable={n_stable}  slip={n_slip}  "
      f"ratio={n_stable//n_slip}:1")

# ── 80/20 session split ───────────────────────────────────────────────────────
sessions   = np.array(data["session"].unique())
np.random.seed(42)
idx        = np.random.permutation(len(sessions))
n_train    = int(len(sessions) * 0.8)
train_sess = set(sessions[idx[:n_train]])
test_sess  = set(sessions[idx[n_train:]])

train = data[data["session"].isin(train_sess)]
test  = data[data["session"].isin(test_sess)]

X_train = train[feat_cols].values
y_train = train["slip"].values
X_test  = test[feat_cols].values
y_test  = test["slip"].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"  Train: {len(train_sess)} sessions ({len(y_train)} frames)  "
      f"Test: {len(test_sess)} sessions ({len(y_test)} frames)")

# ── Small XGBoost ─────────────────────────────────────────────────────────────
spw = int(n_stable / n_slip)   # ~20 — weights slip frames 20× more
print(f"\nTraining small XGBoost (scale_pos_weight={spw})...")

model = XGBClassifier(
    n_estimators=100,          # small
    max_depth=3,               # shallow trees
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.6,
    scale_pos_weight=spw,      # handle 20:1 imbalance
    eval_metric="logloss",
    verbosity=0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("  Slip Detector Results")
print("="*50)
print(classification_report(y_test, preds, target_names=["stable","slip"],
                             digits=3))

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print(f"  Confusion matrix:")
print(f"    True  stable → predicted stable: {tn:4d}  ({tn/(tn+fp)*100:.1f}%)")
print(f"    True  stable → predicted slip:   {fp:4d}  ({fp/(tn+fp)*100:.1f}% false alarm)")
print(f"    True  slip   → predicted stable: {fn:4d}  ({fn/(fn+tp)*100:.1f}% missed)")
print(f"    True  slip   → predicted slip:   {tp:4d}  ({tp/(fn+tp)*100:.1f}% caught)")

# Top features
importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print(f"\n  Top 10 features:")
for i in top_idx:
    print(f"    {feat_cols[i]:35s}  {importances[i]:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,     MODEL_DIR / "slip_detector.pkl")
joblib.dump(scaler,    MODEL_DIR / "scaler_slip.pkl")
joblib.dump(feat_cols, MODEL_DIR / "feature_cols_slip.pkl")
print(f"\nSaved → {MODEL_DIR}/slip_detector.pkl")
