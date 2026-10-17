"""
XGBoost multi-output regressor: position (xyz mm) + orientation (Euler deg)
Trained on simulation hold-phase frames with ground-truth object pose.

Output: [pos_x, pos_y, pos_z, roll, pitch, yaw]  (3 mm + 3 degrees)
Split: 80/20 at session level (no leakage)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.transform import Rotation
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

HERE      = Path(__file__).parent
SIM_DIR   = HERE.parent / "models" / "dataset_sim"
MODEL_DIR = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
TARGET_COLS = ["pos_x", "pos_y", "pos_z",
               "sin_roll", "cos_roll",
               "sin_pitch", "cos_pitch",
               "sin_yaw", "cos_yaw"]


def quat_to_euler(df: pd.DataFrame) -> pd.DataFrame:
    """Add roll/pitch/yaw columns from quaternion columns (degrees)."""
    quats = df[["quat_w", "quat_x", "quat_y", "quat_z"]].values
    r = Rotation.from_quat(quats[:, [1, 2, 3, 0]])  # scipy: [x,y,z,w]
    euler = r.as_euler("xyz", degrees=True)
    df = df.copy()
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    df["roll"]      = roll
    df["pitch"]     = pitch
    df["yaw"]       = yaw
    df["sin_roll"]  = np.sin(np.deg2rad(roll))
    df["cos_roll"]  = np.cos(np.deg2rad(roll))
    df["sin_pitch"] = np.sin(np.deg2rad(pitch))
    df["cos_pitch"] = np.cos(np.deg2rad(pitch))
    df["sin_yaw"]   = np.sin(np.deg2rad(yaw))
    df["cos_yaw"]   = np.cos(np.deg2rad(yaw))
    return df


def load_sim_session(session_path: Path) -> pd.DataFrame | None:
    tac  = pd.read_csv(session_path / "tactile_summary.csv")
    act  = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    pose = pd.read_csv(session_path / "object_pose.csv")

    df = tac.merge(act, on="frame").merge(pose, on="frame")
    df = df[df["phase"] == "hold"]
    if len(df) < 5:
        return None

    df = quat_to_euler(df)
    # Convert position m → mm
    df["pos_x"] *= 1000
    df["pos_y"] *= 1000
    df["pos_z"] *= 1000

    feat_cols = [c for c in tac.columns if c not in META_COLS]
    for c in ACTUATOR_COLS:
        feat_cols.append(c)
    df[feat_cols] = df[feat_cols].fillna(0)
    df["session"] = session_path.name
    return df


# ── Load all sim sessions ─────────────────────────────────────────────────────
print("Loading simulation sessions (hold phase)...")
frames = [df for s in sorted(SIM_DIR.iterdir())
          if (df := load_sim_session(s)) is not None]
data = pd.concat(frames, ignore_index=True)

feat_cols = [c for c in data.columns
             if c not in META_COLS | set(TARGET_COLS) |
             {"session", "phase", "roll", "pitch", "yaw",
              "quat_w", "quat_x", "quat_y", "quat_z",
              "lin_vel_x", "lin_vel_y", "lin_vel_z",
              "ang_vel_x", "ang_vel_y", "ang_vel_z", "in_hand"}]

print(f"  {len(frames)} sessions  |  {len(data)} frames  |  {len(feat_cols)} features")
print(f"  Position range:  x={data['pos_x'].min():.1f}–{data['pos_x'].max():.1f} mm"
      f"  y={data['pos_y'].min():.1f}–{data['pos_y'].max():.1f} mm"
      f"  z={data['pos_z'].min():.1f}–{data['pos_z'].max():.1f} mm")
print(f"  Euler range:     roll={data['roll'].min():.1f}°–{data['roll'].max():.1f}°"
      f"  pitch={data['pitch'].min():.1f}°–{data['pitch'].max():.1f}°"
      f"  yaw={data['yaw'].min():.1f}°–{data['yaw'].max():.1f}°")

# ── 80/20 session-level split ─────────────────────────────────────────────────
sessions = data["session"].unique()
np.random.seed(42)
np.random.shuffle(sessions)
n_train  = int(len(sessions) * 0.8)
train_sess = set(sessions[:n_train])
test_sess  = set(sessions[n_train:])

train = data[data["session"].isin(train_sess)]
test  = data[data["session"].isin(test_sess)]

print(f"\nSplit: {len(train_sess)} train sessions ({len(train)} frames)"
      f"  /  {len(test_sess)} test sessions ({len(test)} frames)")

X_train = train[feat_cols].values
y_train = train[TARGET_COLS].values
X_test  = test[feat_cols].values
y_test  = test[TARGET_COLS].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining MultiOutput XGBoost...")
base = XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    verbosity=0, random_state=42, n_jobs=-1,
)
model = MultiOutputRegressor(base, n_jobs=6)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)

print("\n" + "="*50)
print("  Position + Orientation Prediction Results")
print("="*50)

labels = ["pos_x (mm)", "pos_y (mm)", "pos_z (mm)",
          "sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "sin_yaw", "cos_yaw"]
rmses, r2s = [], []
for i, lbl in enumerate(labels):
    rmse = np.sqrt(np.mean((preds[:, i] - y_test[:, i])**2))
    r2   = r2_score(y_test[:, i], preds[:, i])
    rmses.append(rmse); r2s.append(r2)
    bar = "█" * int(max(0, r2) * 20)
    print(f"  {lbl:16s}  RMSE={rmse:5.3f}  R²={r2:+.3f}  {bar}")

# Convert sin/cos back to angular error in degrees
def angular_rmse(sin_pred, cos_pred, sin_true, cos_true):
    angle_pred = np.rad2deg(np.arctan2(sin_pred, cos_pred))
    angle_true = np.rad2deg(np.arctan2(sin_true, cos_true))
    diff = np.abs(angle_pred - angle_true)
    diff = np.minimum(diff, 360 - diff)  # wrap to [0, 180]
    return np.sqrt(np.mean(diff**2))

roll_rmse  = angular_rmse(preds[:,3], preds[:,4], y_test[:,3], y_test[:,4])
pitch_rmse = angular_rmse(preds[:,5], preds[:,6], y_test[:,5], y_test[:,6])
yaw_rmse   = angular_rmse(preds[:,7], preds[:,8], y_test[:,7], y_test[:,8])

print(f"\n  Mean position RMSE : {np.mean(rmses[:3]):.2f} mm")
print(f"  Angular RMSE       : roll={roll_rmse:.1f}°  pitch={pitch_rmse:.1f}°  yaw={yaw_rmse:.1f}°")
print(f"  Mean rotation RMSE : {np.mean([roll_rmse, pitch_rmse, yaw_rmse]):.1f} °")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,     MODEL_DIR / "pose_regressor.pkl")
joblib.dump(scaler,    MODEL_DIR / "scaler_pose.pkl")
joblib.dump(feat_cols, MODEL_DIR / "feature_cols_pose.pkl")
print(f"\nSaved → {MODEL_DIR}/pose_regressor.pkl")
