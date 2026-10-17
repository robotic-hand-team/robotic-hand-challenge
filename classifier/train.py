"""
Train XGBoost sensor classifier on all 13 labeled sessions.
Saves model + scaler to model/ for later inference.

Labels: 0 = Sensor A,  1 = Sensor B
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
DATASET_DIR = HERE.parent / "models" / "dataset"
MODEL_DIR   = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── Session labels ────────────────────────────────────────────────────────────
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


def load_session(session: str, label: int) -> pd.DataFrame:
    p = DATASET_DIR / session
    tac = pd.read_csv(p / "tactile_summary.csv")
    act = pd.read_csv(p / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df = tac.merge(act, on="frame")
    df["label"]   = label
    df["session"] = session
    feat_cols = [c for c in df.columns if c not in META_COLS | {"label", "session"}]
    df[feat_cols] = df[feat_cols].fillna(0)
    return df


# ── Load all 13 sessions ──────────────────────────────────────────────────────
print("Loading sessions...")
frames = [load_session(s, 0) for s in SESSIONS_A] + \
         [load_session(s, 1) for s in SESSIONS_B]
data = pd.concat(frames, ignore_index=True)

feature_cols = [c for c in data.columns if c not in META_COLS | {"label", "session"}]
X = data[feature_cols].values
y = data["label"].values

print(f"  {len(data)} frames  |  {len(feature_cols)} features")
print(f"  Sensor A: {(y==0).sum()} frames  |  Sensor B: {(y==1).sum()} frames")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    verbosity=0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_scaled, y)

train_acc = (model.predict(X_scaled) == y).mean()
print(f"  Train accuracy (in-sample): {train_acc:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,        MODEL_DIR / "xgb_sensor.pkl")
joblib.dump(scaler,       MODEL_DIR / "scaler.pkl")
joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

print(f"\nSaved to {MODEL_DIR}/")
print("  xgb_sensor.pkl   — trained model")
print("  scaler.pkl       — fitted StandardScaler")
print("  feature_cols.pkl — ordered feature list")
