"""
Predict Sensor A or B on an unseen session folder.

Usage:
    python predict.py <session_folder>

    session_folder must contain:
        tactile_summary.csv
        actuators.csv

Output:
    Per-frame predictions + majority vote for the session.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

HERE      = Path(__file__).parent
MODEL_DIR = HERE / "model"

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
LABEL_NAMES = {0: "Sensor A", 1: "Sensor B"}


def load_model():
    model        = joblib.load(MODEL_DIR / "xgb_sensor.pkl")
    scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return model, scaler, feature_cols


def load_session(session_path: Path, feature_cols: list) -> pd.DataFrame:
    tac = pd.read_csv(session_path / "tactile_summary.csv")
    act = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df = tac.merge(act, on="frame")
    feat_cols_in_df = [c for c in df.columns if c not in META_COLS]
    df[feat_cols_in_df] = df[feat_cols_in_df].fillna(0)
    # Align to training feature order, fill missing with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df


def predict(session_path: Path):
    model, scaler, feature_cols = load_model()
    df = load_session(session_path, feature_cols)

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    preds      = model.predict(X_scaled)
    probs      = model.predict_proba(X_scaled)   # shape (N, 2)
    conf       = probs.max(axis=1)

    # Filter low-contact frames (hand open / no object)
    contact_cols  = [c for c in df.columns if c.endswith("_contacts")]
    total_contacts = df[contact_cols].sum(axis=1).values
    active_mask   = total_contacts > 20            # ignore pre/post frames

    active_preds = preds[active_mask]
    active_conf  = conf[active_mask]

    if len(active_preds) == 0:
        print("No active contact frames found — is the session valid?")
        return

    majority = int(np.bincount(active_preds).argmax())
    mean_conf = active_conf.mean()

    print(f"\nSession: {session_path.name}")
    print(f"  Total frames   : {len(df)}")
    print(f"  Active frames  : {active_mask.sum()} (contact > 20)")
    print(f"  Prediction     : {LABEL_NAMES[majority]}  (majority vote)")
    print(f"  Mean confidence: {mean_conf:.1%}")
    print(f"  A frames: {(active_preds==0).sum()}  |  B frames: {(active_preds==1).sum()}")

    # Per-frame CSV output
    out = df[["frame", "t_rel"]].copy()
    out["prediction"] = [LABEL_NAMES[p] for p in preds]
    out["confidence"] = conf.round(3)
    out["active"]     = active_mask
    out_path = session_path / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"  Per-frame predictions saved → {out_path}")

    return majority, mean_conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <session_folder>")
        sys.exit(1)

    session_path = Path(sys.argv[1])
    if not session_path.exists():
        print(f"Folder not found: {session_path}")
        sys.exit(1)

    predict(session_path)
