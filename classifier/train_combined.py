"""
Train on REAL hardware sessions + SIMULATED sessions combined, to check whether simulation data
can usefully augment the small (13-session) real dataset rather than replace it.

Validation is held out by (source, session) so no session ever appears in both train and val, and
reports real-only / sim-only / combined val accuracy separately -- the real-only number is what
matters for deployment on the physical hand.

Usage (from repo root):
    venv/bin/python classifier/train_combined.py --sim-sessions 60
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluate_cross_domain import ACTUATOR_COLS, META_COLS, REAL_SESSIONS, load_session

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
REAL_DIR = HERE.parent / "models" / "dataset"
SIM_DIR = HERE.parent / "models" / "dataset_sim"
OBJECT_TO_LABEL = {"vegabar_29": 0, "vegabar_38": 1}


def load_real():
    frames = [load_session(REAL_DIR / s, lbl) for s, lbl in REAL_SESSIONS.items()]
    for f in frames:
        f["source"] = "real"
    return frames


def load_sim(max_sessions: int, rng: np.random.Generator):
    per_class = {}
    for meta_path in sorted(SIM_DIR.glob("*/meta.json")):
        obj = json.loads(meta_path.read_text()).get("object")
        if obj in OBJECT_TO_LABEL:
            per_class.setdefault(OBJECT_TO_LABEL[obj], []).append(meta_path.parent)
    frames = []
    for label, dirs in per_class.items():
        chosen = rng.permutation(dirs)[: max_sessions // len(per_class)]
        for d in chosen:
            df = load_session(d, label)
            if df is not None:
                df["source"] = "sim"
                frames.append(df)
    return frames


def split(frames, val_frac, rng):
    sessions = pd.DataFrame({"session": [f["session"].iloc[0] for f in frames],
                             "label": [f["label"].iloc[0] for f in frames],
                             "source": [f["source"].iloc[0] for f in frames]})
    val = set()
    for (label, source), g in sessions.groupby(["label", "source"]):
        s = rng.permutation(g["session"].tolist())
        val.update(s[: max(1, round(len(s) * val_frac))])
    train_frames = [f for f in frames if f["session"].iloc[0] not in val]
    val_frames = [f for f in frames if f["session"].iloc[0] in val]
    return pd.concat(train_frames, ignore_index=True), pd.concat(val_frames, ignore_index=True)


def session_vote_acc(df, pred_col="pred"):
    votes = df.groupby("session").apply(
        lambda g: (g[pred_col].mode()[0] == g["label"].iloc[0]), include_groups=False
    )
    return votes.mean(), votes.sum(), len(votes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-sessions", type=int, default=60, help="total sim sessions to mix in (split across classes)")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    real_frames = load_real()
    sim_frames = load_sim(args.sim_sessions, rng)
    print(f"real: {len(real_frames)} sessions | sim: {len(sim_frames)} sessions (of up to {args.sim_sessions} requested)")

    train_df, val_df = split(real_frames + sim_frames, args.val_frac, rng)
    feature_cols = [c for c in train_df.columns if c not in META_COLS | {"label", "session", "source"}]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                           colsample_bytree=0.8, eval_metric="logloss", verbosity=0,
                           random_state=args.seed, n_jobs=-1)
    model.fit(X_train, train_df["label"].values)

    val_df = val_df.assign(pred=model.predict(scaler.transform(val_df[feature_cols].values)))
    print(f"\ntrain: {len(train_df)} frames ({train_df['session'].nunique()} sessions, "
          f"{(train_df.source == 'real').sum()} real frames, {(train_df.source == 'sim').sum()} sim frames)")
    for source in ("real", "sim"):
        sub = val_df[val_df.source == source]
        if len(sub) == 0:
            continue
        acc, n_ok, n = session_vote_acc(sub)
        print(f"val [{source:4s}]: session majority-vote acc {acc:.3f} ({n_ok}/{n} sessions)")
    acc, n_ok, n = session_vote_acc(val_df)
    print(f"val [all ]: session majority-vote acc {acc:.3f} ({n_ok}/{n} sessions)")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "xgb_sensor_combined.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_combined.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols_combined.pkl")
    print(f"\nSaved to {MODEL_DIR}/*_combined.pkl")


if __name__ == "__main__":
    main()
