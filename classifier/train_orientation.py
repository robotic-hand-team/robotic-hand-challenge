"""
Orientation classifier from tactile data.
4 classes based on roll angle (which side of object faces palm):

  Class 0  neutral    roll  -45° –  +45°
  Class 1  right      roll  +45° – +135°
  Class 2  inverted   roll ±135° –  180°
  Class 3  left       roll -135° –  -45°

Train: sim hold-phase frames (80/20 session split)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.transform import Rotation
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

HERE      = Path(__file__).parent
SIM_DIR   = HERE.parent / "models" / "dataset_sim"
MODEL_DIR = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
CLASS_NAMES = ["neutral (-45°–+45°)", "right (+45°–+135°)",
               "inverted (±135°–180°)", "left (-135°––45°)"]


def roll_to_class(roll_deg: float) -> int:
    r = roll_deg
    if   -45  <= r <  45:  return 0  # neutral
    elif  45  <= r <  135: return 1  # right
    elif  135 <= r or r < -135: return 2  # inverted
    else:                  return 3  # left  (-135 to -45)


def load_sim_session(session_path: Path) -> pd.DataFrame | None:
    tac  = pd.read_csv(session_path / "tactile_summary.csv")
    act  = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    pose = pd.read_csv(session_path / "object_pose.csv")

    df = tac.merge(act, on="frame").merge(pose, on="frame")
    df = df[df["phase"] == "hold"]
    if len(df) < 5:
        return None

    quats = df[["quat_w", "quat_x", "quat_y", "quat_z"]].values
    rolls = Rotation.from_quat(quats[:, [1,2,3,0]]).as_euler("xyz", degrees=True)[:, 0]
    df["orientation"] = [roll_to_class(r) for r in rolls]

    feat_cols = [c for c in tac.columns if c not in META_COLS]
    for c in ACTUATOR_COLS:
        feat_cols.append(c)
    df[feat_cols] = df[feat_cols].fillna(0)
    df["session"] = session_path.name
    return df


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading simulation sessions...")
frames = [df for s in sorted(SIM_DIR.iterdir())
          if (df := load_sim_session(s)) is not None]
data = pd.concat(frames, ignore_index=True)

feat_cols = [c for c in data.columns
             if c not in META_COLS | {"session", "orientation", "phase",
             "quat_w","quat_x","quat_y","quat_z","pos_x","pos_y","pos_z",
             "lin_vel_x","lin_vel_y","lin_vel_z","ang_vel_x","ang_vel_y","ang_vel_z","in_hand"}]

print(f"  {len(frames)} sessions  |  {len(data)} frames  |  {len(feat_cols)} features")
print(f"\nClass distribution:")
for i, name in enumerate(CLASS_NAMES):
    n = (data["orientation"] == i).sum()
    print(f"  [{i}] {name}: {n} frames ({n/len(data)*100:.1f}%)")

# ── 80/20 session split ───────────────────────────────────────────────────────
sessions = np.array(data["session"].unique())
np.random.seed(42)
idx = np.random.permutation(len(sessions))
n_train     = int(len(sessions) * 0.8)
train_sess  = set(sessions[idx[:n_train]])
test_sess   = set(sessions[idx[n_train:]])

train = data[data["session"].isin(train_sess)]
test  = data[data["session"].isin(test_sess)]

X_train = train[feat_cols].values
y_train = train["orientation"].values
X_test  = test[feat_cols].values
y_test  = test["orientation"].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nSplit: {len(train_sess)} train / {len(test_sess)} test sessions")

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", verbosity=0, random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)

print("\n" + "="*52)
print("  Orientation Classification Results")
print("="*52)
print(f"  Frame accuracy: {acc:.3f}")
print(f"\n{classification_report(y_test, preds, target_names=['neutral','right','inverted','left'])}")

# Session-level majority vote
print("  Per-session majority vote:")
sess_correct = []
for s in sorted(test_sess):
    mask  = test["session"] == s
    y_s   = y_test[mask.values]
    p_s   = preds[mask.values]
    maj   = int(np.bincount(p_s, minlength=4).argmax())
    true  = int(y_s[0])
    correct = maj == true
    sess_correct.append(correct)
    sym = "✓" if correct else "✗"
    print(f"    {sym} true={CLASS_NAMES[true][:7]}  pred={CLASS_NAMES[maj][:7]}")

print(f"\n  Session accuracy: {np.mean(sess_correct):.3f}  ({sum(sess_correct)}/{len(sess_correct)})")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,     MODEL_DIR / "orientation_classifier.pkl")
joblib.dump(scaler,    MODEL_DIR / "scaler_orientation.pkl")
joblib.dump(feat_cols, MODEL_DIR / "feature_cols_orientation.pkl")
print(f"\nSaved → {MODEL_DIR}/orientation_classifier.pkl")
