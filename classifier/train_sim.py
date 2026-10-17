"""
Train the same XGBoost sensor A/B classifier as train.py, but on simulator-generated sessions
(simulation/hand_sim/generate_dataset.py) instead of the 13 real hardware recordings.

Ground truth comes straight from each session's meta.json["object"] field (the simulator knows
exactly what it grasped), mapped via OBJECT_TO_LABEL below:
    vegabar_29 (Sensor A) = 0, vegabar_38 (Sensor B) = 1
confirmed against the real dataset's 13 labeled sessions (2026-10-17).

Splits by SESSION, not by frame, for train/val (frames within a session are highly correlated;
a frame-level split would leak information and overstate accuracy).

Usage (from repo root):
    venv/bin/python classifier/train_sim.py --dataset models/dataset_sim --val-frac 0.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"

META_COLS = {"frame", "t_unix", "t_rel"}
XGBoost trained on 300 simulation sessions (hold-phase only).
Validated on 13 real hardware sessions (contacts > 20 as hold-proxy).

Sim labels:  vegabar_29 (w0-w3) = Sensor A (0)
             vegabar_38 (w4-w7) = Sensor B (1)
Real labels: sessions 6-12     = Sensor A (0)
             sessions 13-18    = Sensor B (1)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

HERE        = Path(__file__).parent
SIM_DIR     = HERE.parent / "models" / "dataset_sim"
REAL_DIR    = HERE.parent / "models" / "dataset"
MODEL_DIR   = HERE / "model"
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
OBJECT_TO_LABEL = {"vegabar_29": 0, "vegabar_38": 1}  # 0 = Sensor A, 1 = Sensor B
LABEL_NAMES = {0: "Sensor A", 1: "Sensor B"}


def load_session(session_dir: Path) -> pd.DataFrame | None:
    meta = json.loads((session_dir / "meta.json").read_text())
    obj = meta.get("object")
    if obj not in OBJECT_TO_LABEL:
        return None
    tac = pd.read_csv(session_dir / "tactile_summary.csv")
    act = pd.read_csv(session_dir / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df = tac.merge(act, on="frame")
    df["label"] = OBJECT_TO_LABEL[obj]
    df["session"] = session_dir.name
    feat_cols = [c for c in df.columns if c not in META_COLS | {"label", "session"}]
    df[feat_cols] = df[feat_cols].fillna(0)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(HERE.parent / "models" / "dataset_sim"))
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of sessions held out, per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="_sim", help="model files are saved as xgb_sensor<suffix>.pkl etc.")
    args = ap.parse_args()

    session_dirs = sorted(p.parent for p in Path(args.dataset).glob("*/meta.json"))
    print(f"Loading {len(session_dirs)} sessions from {args.dataset} ...")
    frames = [df for d in session_dirs if (df := load_session(d)) is not None]
    data = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in data.columns if c not in META_COLS | {"label", "session"}]

    rng = np.random.default_rng(args.seed)
    sessions = data[["session", "label"]].drop_duplicates()
    val_sessions = set()
    for label in sessions["label"].unique():
        s = sessions.loc[sessions.label == label, "session"].tolist()
        rng.shuffle(s)
        val_sessions.update(s[: max(1, round(len(s) * args.val_frac))])
    is_val = data["session"].isin(val_sessions)
    train_df, val_df = data[~is_val], data[is_val]

    print(f"  {len(data)} frames from {sessions.shape[0]} sessions | {len(feature_cols)} features")
    print(f"  train: {len(train_df)} frames / {sessions.shape[0] - len(val_sessions)} sessions | "
          f"val: {len(val_df)} frames / {len(val_sessions)} sessions")
    for label, name in LABEL_NAMES.items():
        print(f"  {name}: {(data.label == label).sum()} frames total "
              f"({(sessions.label == label).sum()} sessions)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        verbosity=0, random_state=args.seed, n_jobs=-1,
    )
    model.fit(X_train, train_df["label"].values)

    train_acc = (model.predict(X_train) == train_df["label"].values).mean()
    val_pred = model.predict(X_val)
    val_acc = (val_pred == val_df["label"].values).mean()
    print(f"\n  Train accuracy (frame-level): {train_acc:.3f}")
    print(f"  Val accuracy   (frame-level): {val_acc:.3f}  (held-out sessions, not frames)")

    # Session-level majority vote, closer to how predict.py is actually used.
    val_df = val_df.assign(pred=val_pred)
    votes = val_df.groupby("session").apply(
        lambda g: (g["pred"].mode()[0] == g["label"].iloc[0]), include_groups=False
    )
    print(f"  Val accuracy   (session majority vote): {votes.mean():.3f} ({votes.sum()}/{len(votes)} sessions)")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"xgb_sensor{args.suffix}.pkl")
    joblib.dump(scaler, MODEL_DIR / f"scaler{args.suffix}.pkl")
    joblib.dump(feature_cols, MODEL_DIR / f"feature_cols{args.suffix}.pkl")
    print(f"\nSaved to {MODEL_DIR}/*{args.suffix}.pkl")


if __name__ == "__main__":
    main()


def get_label_from_meta(session_path: Path) -> int:
    meta = json.loads((session_path / "meta.json").read_text())
    obj  = meta.get("object") or meta.get("episode_info", {}).get("object", "")
    return 0 if "29" in obj else 1


def load_sim_session(session_path: Path) -> pd.DataFrame | None:
    """Load one sim session, keep only hold-phase frames."""
    tac  = pd.read_csv(session_path / "tactile_summary.csv")
    act  = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    pose = pd.read_csv(session_path / "object_pose.csv")[["frame", "phase"]]

    df = tac.merge(act, on="frame").merge(pose, on="frame")
    df = df[df["phase"] == "hold"].drop(columns=["phase"])

    if len(df) < 5:
        return None

    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    df["label"]   = get_label_from_meta(session_path)
    return df


def load_real_session(session: str, label: int) -> pd.DataFrame:
    """Load one real session, keep active-contact frames as hold proxy."""
    p   = REAL_DIR / session
    tac = pd.read_csv(p / "tactile_summary.csv")
    act = pd.read_csv(p / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df  = tac.merge(act, on="frame")

    contact_cols = [c for c in df.columns if c.endswith("_contacts")]
    df = df[df[contact_cols].sum(axis=1) > 20]

    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    df["label"]   = label
    df["session"] = session
    return df


# ── Load sim data ─────────────────────────────────────────────────────────────
print("Loading simulation sessions (hold phase)...")
sim_frames = []
skipped = 0
for s in sorted(SIM_DIR.iterdir()):
    df = load_sim_session(s)
    if df is None:
        skipped += 1
        continue
    sim_frames.append(df)

sim_data = pd.concat(sim_frames, ignore_index=True)
feature_cols = [c for c in sim_data.columns if c not in META_COLS | {"label", "session"}]

n_a = (sim_data["label"] == 0).sum()
n_b = (sim_data["label"] == 1).sum()
print(f"  {len(sim_frames)} sessions loaded ({skipped} skipped <5 hold frames)")
print(f"  Sensor A: {n_a} frames  |  Sensor B: {n_b} frames  |  Total: {len(sim_data)}")

# ── Load real data ────────────────────────────────────────────────────────────
print("\nLoading real sessions (contacts > 20)...")
real_frames = []
for s in REAL_SESSIONS_A:
    real_frames.append(load_real_session(s, 0))
for s in REAL_SESSIONS_B:
    real_frames.append(load_real_session(s, 1))
real_data = pd.concat(real_frames, ignore_index=True)
print(f"  {len(real_data)} frames across 13 sessions")

# Align feature cols (sim and real share the same schema)
for col in feature_cols:
    if col not in real_data.columns:
        real_data[col] = 0.0

# ── Train on sim ──────────────────────────────────────────────────────────────
print("\nTraining XGBoost on sim data...")
X_sim = sim_data[feature_cols].values
y_sim = sim_data["label"].values

scaler  = StandardScaler()
X_sim_s = scaler.fit_transform(X_sim)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    verbosity=0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_sim_s, y_sim)
train_acc = accuracy_score(y_sim, model.predict(X_sim_s))
print(f"  In-sample accuracy (sim): {train_acc:.3f}")

# ── Validate on real ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Sim → Real validation (13 hardware sessions)")
print("="*55)

X_real = real_data[feature_cols].values
y_real = real_data["label"].values
X_real_s = scaler.transform(X_real)
preds = model.predict(X_real_s)

overall_acc = accuracy_score(y_real, preds)
print(f"\n  Overall frame accuracy: {overall_acc:.3f}")
print(f"\n{classification_report(y_real, preds, target_names=['Sensor A','Sensor B'])}")

# Per-session majority vote
print("  Per-session majority vote:")
for s in REAL_SESSIONS_A + REAL_SESSIONS_B:
    mask    = real_data["session"] == s
    y_s     = y_real[mask.values]
    p_s     = preds[mask.values]
    maj     = int(np.bincount(p_s).argmax())
    correct = maj == y_s[0]
    lbl     = "A" if y_s[0] == 0 else "B"
    pred_lbl = "A" if maj == 0 else "B"
    sym     = "✓" if correct else "✗"
    conf    = model.predict_proba(X_real_s[mask.values]).max(axis=1).mean()
    print(f"    {sym} [{lbl}] {s}  → {pred_lbl}  conf={conf:.1%}")

# Session-level accuracy
session_preds = []
for s in REAL_SESSIONS_A + REAL_SESSIONS_B:
    mask = real_data["session"] == s
    p_s  = preds[mask.values]
    maj  = int(np.bincount(p_s).argmax())
    true = int(real_data[mask]["label"].iloc[0])
    session_preds.append(maj == true)
print(f"\n  Session-level accuracy: {np.mean(session_preds):.3f}  ({sum(session_preds)}/13 correct)")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,        MODEL_DIR / "xgb_sim.pkl")
joblib.dump(scaler,       MODEL_DIR / "scaler_sim.pkl")
joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")
print(f"\nSaved → {MODEL_DIR}/xgb_sim.pkl")
