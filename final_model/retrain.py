"""Retrains sensor_classifier.pkl (the hold-mean sensor A/B model) from labelled sessions.

    venv/bin/python -m final_model.retrain [--labels final_model/labels.csv] [--logs hardware/logs]

labels.csv: one row per session folder in --logs, label sensor_A or sensor_B. Features are exactly
match_session.py's: the mean of the 108 feature_cols over the detected hold window, as float32.
Same model as before (StandardScaler + XGBoost, 50 trees, depth 2, learning rate 0.1). Prints the
leave-one-session-out accuracy first, then trains on all sessions and overwrites the model.

Train on recordings of the grasp you will use: on 18.10 the 23 labelled grasps alone scored 17/23 in
leave-one-session-out CV, while adding the 16.10 / 17.10 sessions (another grasp style) dropped
that to 11–12/23, and the previous model (17.10 data) got 0 of the 13 sensor_B grasps right.
After retraining, rebuild the matcher's library: match_session.py ... --rebuild-index
"""
import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    import dtaidistance  # noqa: F401  (only the matcher's DTW stage needs it)
except ImportError:
    _stub = types.ModuleType("dtaidistance")
    _stub.dtw_ndim = None
    sys.modules["dtaidistance"] = _stub

from final_model.match_session import _hold_mean_features

HERE = Path(__file__).parent
MODEL = HERE / "sensor_classifier.pkl"
PARAMS = dict(n_estimators=50, max_depth=2, learning_rate=0.1, eval_metric="logloss",
              random_state=0, n_jobs=1)


def fit(X, y):
    scaler = StandardScaler().fit(X)
    return scaler, XGBClassifier(**PARAMS).fit(scaler.transform(X), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=HERE / "labels.csv")
    ap.add_argument("--logs", type=Path, default=HERE.parent / "hardware" / "logs")
    args = ap.parse_args()

    with open(MODEL, "rb") as f:
        feature_cols = pickle.load(f)["feature_cols"]
    labels = pd.read_csv(args.labels)
    rows = []
    for s, label in zip(labels.session, labels.label):
        v = _hold_mean_features(args.logs / s, feature_cols)
        if v is None:
            print(f"  skipped {s}: missing or too short")
            continue
        rows.append((s, v, label))
    le = LabelEncoder().fit(["sensor_A", "sensor_B"])
    X = np.array([v for _, v, _ in rows])
    y = le.transform([label for *_, label in rows])
    print(f"{len(rows)} sessions: {dict(zip(le.classes_, np.bincount(y, minlength=2)))}")

    hits = []
    for i, (s, v, label) in enumerate(rows):
        keep = np.arange(len(rows)) != i
        scaler, xgb = fit(X[keep], y[keep])
        pred = le.inverse_transform(xgb.predict(scaler.transform(v.reshape(1, -1))))[0]
        hits.append(pred == label)
        if pred != label:
            print(f"  LOO miss: {s} is {label}, predicted {pred}")
    loo = f"{100 * np.mean(hits):.1f}% ({sum(hits)}/{len(hits)})"
    print(f"leave-one-session-out accuracy: {loo}")

    scaler, xgb = fit(X, y)
    with open(MODEL, "wb") as f:
        pickle.dump({"xgb": xgb, "scaler": scaler, "feature_cols": feature_cols,
                     "label_encoder": le, "loo_accuracy": loo}, f)
    print(f"wrote {MODEL}")


if __name__ == "__main__":
    main()
