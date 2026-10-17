"""
Cross-domain check: does a classifier trained on one data source (real hardware or simulation)
generalize to the other? Evaluates session-level majority-vote accuracy in both directions.

Usage (from repo root):
    venv/bin/python classifier/evaluate_cross_domain.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
REAL_DIR = HERE.parent / "models" / "dataset"
SIM_DIR = HERE.parent / "models" / "dataset_sim"

META_COLS = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
LABEL_NAMES = {0: "Sensor A", 1: "Sensor B"}

# train.py's real-session labels (the only ground truth we have for the hardware recordings).
REAL_SESSIONS = {
    "20261016_204344": 0, "20261016_204402": 0, "20261016_204429": 0, "20261016_204457": 0,
    "20261016_204513": 0, "20261016_204524": 0, "20261016_204548": 0,
    "20261016_204558": 1, "20261016_204609": 1, "20261016_204619": 1,
    "20261016_204627": 1, "20261016_204637": 1, "20261016_204649": 1,
}


def load_session(session_dir: Path, label: int) -> pd.DataFrame | None:
    try:
        tac = pd.read_csv(session_dir / "tactile_summary.csv")
        act = pd.read_csv(session_dir / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    except FileNotFoundError:
        return None
    df = tac.merge(act, on="frame")
    df["label"], df["session"] = label, session_dir.name
    feat_cols = [c for c in df.columns if c not in META_COLS | {"label", "session"}]
    df[feat_cols] = df[feat_cols].fillna(0)
    return df


def load_real():
    frames = [load_session(REAL_DIR / s, lbl) for s, lbl in REAL_SESSIONS.items()]
    return pd.concat([f for f in frames if f is not None], ignore_index=True)


def load_sim():
    import json
    object_to_label = {"vegabar_29": 0, "vegabar_38": 1}
    frames = []
    for meta_path in sorted(SIM_DIR.glob("*/meta.json")):
        obj = json.loads(meta_path.read_text()).get("object")
        if obj in object_to_label:
            frames.append(load_session(meta_path.parent, object_to_label[obj]))
    return pd.concat([f for f in frames if f is not None], ignore_index=True)


def evaluate(suffix: str, data: pd.DataFrame, source_name: str):
    model = joblib.load(MODEL_DIR / f"xgb_sensor{suffix}.pkl")
    scaler = joblib.load(MODEL_DIR / f"scaler{suffix}.pkl")
    feature_cols = joblib.load(MODEL_DIR / f"feature_cols{suffix}.pkl")
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0.0
    X = scaler.transform(data[feature_cols].values)
    pred = model.predict(X)
    frame_acc = (pred == data["label"].values).mean()

    data = data.assign(pred=pred)
    votes = data.groupby("session").apply(
        lambda g: (g["pred"].mode()[0] == g["label"].iloc[0]), include_groups=False
    )
    print(f"  model{suffix:<5s} on {source_name:5s} data: frame acc {frame_acc:.3f} | "
          f"session majority-vote acc {votes.mean():.3f} ({votes.sum()}/{len(votes)})")


def main():
    real, sim = load_real(), load_sim()
    print(f"real: {len(real)} frames / {real['session'].nunique()} sessions  |  "
          f"sim: {len(sim)} frames / {sim['session'].nunique()} sessions\n")

    print("Trained on REAL hardware data (classifier/model/xgb_sensor.pkl, Lars's original):")
    evaluate("", real, "real")
    evaluate("", sim, "sim")
    print("\nTrained on SIMULATED data (classifier/model/xgb_sensor_sim.pkl):")
    evaluate("_sim", real, "real")
    evaluate("_sim", sim, "sim")


if __name__ == "__main__":
    main()
