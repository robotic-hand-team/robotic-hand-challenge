"""
Mixed training: Simulation (12k frames) + Real (2.9k frames).

Two fixes for Sim→Real gap:
  1. Per-session z-score normalization (removes absolute value domain shift)
  2. Mixed training so model sees both domains

Evaluation: LOSO-CV on real sessions
  (each fold: all sim + remaining 12 real sessions → test on 1 real session)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

HERE      = Path(__file__).parent
SIM_DIR   = HERE.parent / "models" / "dataset_sim"
REAL_DIR  = HERE.parent / "models" / "dataset"
MODEL_DIR = HERE / "model"
MODEL_DIR.mkdir(exist_ok=True)

REAL_SESSIONS_A = [
    "20261016_204344", "20261016_204402", "20261016_204429",
    "20261016_204457", "20261016_204513", "20261016_204524", "20261016_204548",
]
REAL_SESSIONS_B = [
    "20261016_204558", "20261016_204609", "20261016_204619",
    "20261016_204627", "20261016_204637", "20261016_204649",
]

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]


def session_normalize(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Z-score each session independently — removes absolute domain shift."""
    df = df.copy()
    vals = df[feat_cols].values.astype(np.float64)
    mu   = vals.mean(axis=0)
    sig  = vals.std(axis=0) + 1e-8
    df[feat_cols] = (vals - mu) / sig
    return df


def load_sim_session(session_path: Path, feat_cols: list | None = None) -> pd.DataFrame | None:
    tac  = pd.read_csv(session_path / "tactile_summary.csv")
    act  = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    pose = pd.read_csv(session_path / "object_pose.csv")[["frame", "phase"]]
    df   = tac.merge(act, on="frame").merge(pose, on="frame")
    df   = df[df["phase"] == "hold"].drop(columns=["phase"])
    if len(df) < 5:
        return None
    meta  = json.loads((session_path / "meta.json").read_text())
    obj   = meta.get("object") or meta.get("episode_info", {}).get("object", "")
    label = 0 if "29" in obj else 1
    fc    = feat_cols or [c for c in df.columns if c not in META_COLS]
    df[fc] = df[fc].fillna(0)
    df["label"]   = label
    df["session"] = session_path.name
    df["domain"]  = "sim"
    return session_normalize(df, fc)


def load_real_session(session: str, label: int, feat_cols: list) -> pd.DataFrame:
    p   = REAL_DIR / session
    tac = pd.read_csv(p / "tactile_summary.csv")
    act = pd.read_csv(p / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df  = tac.merge(act, on="frame")
    contact_cols = [c for c in df.columns if c.endswith("_contacts")]
    df   = df[df[contact_cols].sum(axis=1) > 20]
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feat_cols] = df[feat_cols].fillna(0)
    df["label"]   = label
    df["session"] = session
    df["domain"]  = "real"
    return session_normalize(df, feat_cols)


# ── Load sim ──────────────────────────────────────────────────────────────────
print("Loading simulation sessions...")
sim_frames = [df for s in sorted(SIM_DIR.iterdir())
              if (df := load_sim_session(s)) is not None]
sim_data  = pd.concat(sim_frames, ignore_index=True)
feat_cols = [c for c in sim_data.columns if c not in META_COLS | {"label","session","domain"}]
print(f"  {len(sim_frames)} sessions  |  {len(sim_data)} hold frames")
print(f"  A: {(sim_data['label']==0).sum()}  B: {(sim_data['label']==1).sum()}")

# ── Load real ─────────────────────────────────────────────────────────────────
print("\nLoading real sessions...")
real_frames = (
    [load_real_session(s, 0, feat_cols) for s in REAL_SESSIONS_A] +
    [load_real_session(s, 1, feat_cols) for s in REAL_SESSIONS_B]
)
real_data = pd.concat(real_frames, ignore_index=True)
print(f"  13 sessions  |  {len(real_data)} frames")

# ── LOSO-CV on real sessions ──────────────────────────────────────────────────
all_real_sessions = REAL_SESSIONS_A + REAL_SESSIONS_B
print(f"\nLOSO-CV on real sessions ({len(all_real_sessions)} folds)")
print("  Train: all sim + 12 real  →  Test: 1 real session")
print("-" * 55)

cv_correct = []
for test_sess in all_real_sessions:
    train_real = real_data[real_data["session"] != test_sess]
    test_real  = real_data[real_data["session"] == test_sess]

    train_df = pd.concat([sim_data, train_real], ignore_index=True)
    X_train  = train_df[feat_cols].values
    y_train  = train_df["label"].values
    X_test   = test_real[feat_cols].values
    y_test   = test_real["label"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    maj   = int(np.bincount(preds).argmax())
    true  = int(y_test[0])
    conf  = model.predict_proba(X_test).max(axis=1).mean()

    correct  = maj == true
    lbl      = "A" if true == 0 else "B"
    pred_lbl = "A" if maj  == 0 else "B"
    sym      = "✓" if correct else "✗"
    cv_correct.append(correct)
    print(f"  {sym} [{lbl}] {test_sess}  → {pred_lbl}  conf={conf:.1%}")

print("-" * 55)
print(f"  Session accuracy: {np.mean(cv_correct):.3f}  ({sum(cv_correct)}/13 correct)")

# ── Final model: all sim + all real ──────────────────────────────────────────
print("\nTraining final model on all sim + all real...")
all_data = pd.concat([sim_data, real_data], ignore_index=True)
X_all = all_data[feat_cols].values
y_all = all_data["label"].values

scaler_final = StandardScaler()
X_all_s = scaler_final.fit_transform(X_all)

final_model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1,
)
final_model.fit(X_all_s, y_all)

joblib.dump(final_model,   MODEL_DIR / "xgb_mixed.pkl")
joblib.dump(scaler_final,  MODEL_DIR / "scaler_mixed.pkl")
joblib.dump(feat_cols,     MODEL_DIR / "feature_cols.pkl")
print(f"Saved → {MODEL_DIR}/xgb_mixed.pkl")
