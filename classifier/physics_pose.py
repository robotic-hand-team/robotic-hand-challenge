"""
Physics-inspired position estimator.

Computes a weighted contact centroid across all active tactile patches:
  - for each active patch: weight = force  (contact magnitude)
                           centroid = (cy, cx) fraction within the patch
  - final estimate = linear combination of these weighted patch features

Two models are trained:
  1. CentroidRegressor: 3-output ridge regression on the physics features alone
     (interpretable baseline, shows what cy/cx encode)
  2. XGBRegressor: full tactile summary + actuators (reference model)

Evaluation metric: per-axis RMSE and mean angular RMSE (roll/pitch/yaw).
Split: 80/20 at session level.

Physics features per patch (17 patches):
  (cy_norm, cx_norm, force_weighted)
  cy_norm = cy / max_row  (0=proximal, 1=fingertip)
  cx_norm = cx / max_col  (0=medial, 1=lateral)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

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

# Sensor patch layout: (name, max_rows, max_cols)
PATCHES = [
    ("little_tip",  3,  3), ("little_nail", 12,  8), ("little_pad", 10,  8),
    ("ring_tip",    3,  3), ("ring_nail",   12,  8), ("ring_pad",   10,  8),
    ("middle_tip",  3,  3), ("middle_nail", 12,  8), ("middle_pad", 10,  8),
    ("index_tip",   3,  3), ("index_nail",  12,  8), ("index_pad",  10,  8),
    ("thumb_tip",   3,  3), ("thumb_nail",  12,  8), ("thumb_mid",   3,  3),
    ("thumb_pad",  12,  8), ("palm",         8, 14),
]


def quat_to_sincos(df: pd.DataFrame) -> pd.DataFrame:
    quats = df[["quat_w", "quat_x", "quat_y", "quat_z"]].values
    r = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
    euler = r.as_euler("xyz", degrees=True)
    df = df.copy()
    for i, ax in enumerate(["roll", "pitch", "yaw"]):
        df[f"sin_{ax}"] = np.sin(np.deg2rad(euler[:, i]))
        df[f"cos_{ax}"] = np.cos(np.deg2rad(euler[:, i]))
    return df


def physics_features(df: pd.DataFrame) -> np.ndarray:
    """3 features per patch: (cy_norm, cx_norm, force_w). NaN when patch inactive → 0."""
    rows = []
    for name, max_r, max_c in PATCHES:
        contacts = df[f"{name}_contacts"].values.astype(float)
        force    = df[f"{name}_force"].values.astype(float)
        cy       = pd.to_numeric(df[f"{name}_cy"],   errors="coerce").values
        cx       = pd.to_numeric(df[f"{name}_cx"],   errors="coerce").values

        active      = contacts > 0
        cy_norm     = np.where(active & ~np.isnan(cy), cy / max_r, 0.0)
        cx_norm     = np.where(active & ~np.isnan(cx), cx / max_c, 0.0)
        force_w     = np.where(active, force, 0.0)
        rows.append(np.stack([cy_norm, cx_norm, force_w], axis=1))   # (N, 3)
    return np.concatenate(rows, axis=1)   # (N, 51)


def load_session(session_path: Path) -> pd.DataFrame | None:
    tac  = pd.read_csv(session_path / "tactile_summary.csv")
    act  = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    pose = pd.read_csv(session_path / "object_pose.csv")

    df = tac.merge(act, on="frame").merge(pose, on="frame")
    df = df[df["phase"] == "hold"].reset_index(drop=True)
    if len(df) < 5:
        return None

    df = quat_to_sincos(df)
    df["pos_x"] *= 1000
    df["pos_y"] *= 1000
    df["pos_z"] *= 1000
    df["session"] = session_path.name
    return df


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading simulation sessions (hold phase)...")
frames = [df for s in sorted(SIM_DIR.iterdir())
          if (df := load_session(s)) is not None]
data = pd.concat(frames, ignore_index=True)
print(f"  {len(frames)} sessions  |  {len(data)} frames")

# Full feature set (tactile_summary + actuators)
full_feat = [c for c in data.columns
             if c not in META_COLS | set(TARGET_COLS) |
             {"session", "phase", "roll", "pitch", "yaw",
              "quat_w", "quat_x", "quat_y", "quat_z",
              "lin_vel_x", "lin_vel_y", "lin_vel_z",
              "ang_vel_x", "ang_vel_y", "ang_vel_z", "in_hand",
              "t_unix_x", "t_unix_y", "t_rel_x", "t_rel_y",
              "frame_x", "frame_y"}]

# ── Split ─────────────────────────────────────────────────────────────────────
sessions = data["session"].unique()
np.random.seed(42)
np.random.shuffle(sessions)
n_train    = int(len(sessions) * 0.8)
train_sess = set(sessions[:n_train])
test_sess  = set(sessions[n_train:])

train = data[data["session"].isin(train_sess)]
test  = data[data["session"].isin(test_sess)]
print(f"\nSplit: {len(train_sess)} train / {len(test_sess)} test sessions")

y_train = train[TARGET_COLS].values
y_test  = test[TARGET_COLS].values

# ── Model 1: Physics centroid features ────────────────────────────────────────
print("\n[1] Physics centroid features (cy/cx/force per patch, 51 features)...")
Xp_train = physics_features(train)
Xp_test  = physics_features(test)
print(f"    Feature shape: {Xp_train.shape}")

sp = StandardScaler()
Xp_train_s = sp.fit_transform(Xp_train)
Xp_test_s  = sp.transform(Xp_test)

ridge = MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=-1)
ridge.fit(Xp_train_s, y_train)
pred_phys = ridge.predict(Xp_test_s)

# ── Model 2: Full XGBoost (reference) ─────────────────────────────────────────
print("[2] Full XGBoost (reference, 108+ features)...")
X_train = train[full_feat].values
X_test  = test[full_feat].values
sx = StandardScaler()
X_train_s = sx.fit_transform(X_train)
X_test_s  = sx.transform(X_test)

base = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    verbosity=0, random_state=42, n_jobs=-1)
xgb = MultiOutputRegressor(base, n_jobs=6)
xgb.fit(X_train_s, y_train)
pred_xgb = xgb.predict(X_test_s)


def angular_rmse(sin_pred, cos_pred, sin_true, cos_true):
    ap = np.rad2deg(np.arctan2(sin_pred, cos_pred))
    at = np.rad2deg(np.arctan2(sin_true, cos_true))
    d  = np.abs(ap - at)
    d  = np.minimum(d, 360 - d)
    return np.sqrt(np.mean(d ** 2))


def print_results(name, pred, y_true):
    print(f"\n  {'─'*44}")
    print(f"  {name}")
    print(f"  {'─'*44}")
    labels = ["pos_x (mm)", "pos_y (mm)", "pos_z (mm)",
              "sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "sin_yaw", "cos_yaw"]
    rmses = []
    for i, lbl in enumerate(labels):
        rmse = np.sqrt(np.mean((pred[:, i] - y_true[:, i]) ** 2))
        r2   = r2_score(y_true[:, i], pred[:, i])
        rmses.append(rmse)
        bar  = "█" * int(max(0, r2) * 15)
        print(f"  {lbl:16s}  RMSE={rmse:5.2f}  R²={r2:+.3f}  {bar}")
    roll_e  = angular_rmse(pred[:,3], pred[:,4], y_true[:,3], y_true[:,4])
    pitch_e = angular_rmse(pred[:,5], pred[:,6], y_true[:,5], y_true[:,6])
    yaw_e   = angular_rmse(pred[:,7], pred[:,8], y_true[:,7], y_true[:,8])
    print(f"\n  Pos RMSE  : {np.mean(rmses[:3]):.2f} mm")
    print(f"  Angle RMSE: roll={roll_e:.1f}°  pitch={pitch_e:.1f}°  yaw={yaw_e:.1f}°")


print("\n" + "="*52)
print("  Position + Orientation Estimation Results")
print("="*52)
print_results("Physics centroid (Ridge, 51 features)", pred_phys, y_test)
print_results("Full XGBoost (108+ features, reference)", pred_xgb, y_test)

# ── Save physics model ────────────────────────────────────────────────────────
joblib.dump(ridge, MODEL_DIR / "physics_pose_ridge.pkl")
joblib.dump(sp,    MODEL_DIR / "scaler_physics_pose.pkl")
joblib.dump(xgb,   MODEL_DIR / "pose_regressor.pkl")
joblib.dump(sx,    MODEL_DIR / "scaler_pose.pkl")
joblib.dump(full_feat, MODEL_DIR / "feature_cols_pose.pkl")
print(f"\nSaved → {MODEL_DIR}/physics_pose_ridge.pkl  (physics model)")
print(f"Saved → {MODEL_DIR}/pose_regressor.pkl  (XGBoost reference)")
