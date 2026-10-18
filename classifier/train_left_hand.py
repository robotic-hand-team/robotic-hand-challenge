"""Train the live left-hand models on models/dataset_left and score them leave-one-grasp-out.

Two random forests on the mean of the final model's 108 features over the grasp's hold window (the vector
the live pipeline averages, backend/src/sessions.py HoldMeanSensorClassifier.vector):
  sensor     Sensor A or B. Trained on the photo and the inferred labels; scored on the photo labels.
  direction  where the sensor's yellow end points in the palm plane, as an angle (0 deg = towards the
             thumb, 90 deg = towards the fingertips), regressed as (cos, sin). Photo labels only.
             Live, its axis line is replaced by the palm's measured strip (backend/src/palm_axis.py) when
             the palm shows one; the model then only decides which end is which. The palm's calibration
             offset is fitted here and saved with the models.

Run from the repo root:  python -m classifier.train_left_hand
Writes final_model/left_hand_models.pkl.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from backend.src.palm_axis import fuse, palm_line, reliable, wrap90
from classifier.left_hand_data import DATA_DIR, load, palm_hold_mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_model" / "left_hand_models.pkl"
FEATURES = list(joblib.load(ROOT / "final_model" / "sensor_classifier.pkl")["feature_cols"])


def sensor_model():
    return RandomForestClassifier(500, min_samples_leaf=2, class_weight="balanced", random_state=0)


def direction_model():
    return RandomForestRegressor(500, min_samples_leaf=2, random_state=0)


def angle_error(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def main():
    warnings.filterwarnings("ignore")
    df = load()
    X = df.reindex(columns=FEATURES, fill_value=0).to_numpy(np.float32)
    y = (df["sensor"] == "B").astype(int).to_numpy()
    photo = (df["label_source"] == "photo").to_numpy()
    n = len(df)

    # sensor: leave one grasp out, score on the photo labels
    p_b = np.array([sensor_model().fit(X[np.arange(n) != i], y[np.arange(n) != i])
                    .predict_proba(X[i:i + 1])[0, 1] for i in range(n)])
    ok = (p_b >= 0.5).astype(int) == y
    sensor_acc = f"{ok[photo].sum()}/{photo.sum()} photo-labelled grasps ({ok[photo].mean():.0%})"
    print(f"sensor A/B, leave one grasp out: {sensor_acc}; "
          f"A {ok[photo & (y == 0)].mean():.0%}, B {ok[photo & (y == 1)].mean():.0%}")

    # direction: photo labels only
    d = df[photo].reset_index(drop=True)
    Xd = X[photo]
    theta = np.degrees(np.arctan2(d["end_up"].astype(float), d["end_thumb"].astype(float)))
    Y = np.c_[np.cos(np.radians(theta)), np.sin(np.radians(theta))]
    pred = []
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        c, s = direction_model().fit(Xd[tr], Y[tr]).predict(Xd[i:i + 1])[0]
        pred.append(np.degrees(np.arctan2(s, c)))
    def summary(e):
        return (f"median error {np.median(e):.0f} deg, {np.mean(e <= 30):.0%} within 30 deg, "
                f"{np.mean(e > 90):.0%} pointing the wrong way")

    err = angle_error(pred, theta)
    fixed = angle_error(np.where(d["sensor"] == "A", 0.0, 180.0), theta)
    print(f"direction, model only, leave one grasp out: {summary(err)}")
    print(f"  (old fixed pose: {summary(fixed)})")

    # palm axis line (backend/src/palm_axis.py): calibrate its offset against the photo labels, leave-one-out
    palm = np.array([palm_line(palm_hold_mean(DATA_DIR / s)) for s in d["session"]])
    line_true = wrap90(theta)
    ok = np.array([np.isfinite(l) and reliable(e, t) for l, e, t in palm])
    fused = []
    for i in range(len(d)):
        others = ok & (np.arange(len(d)) != i)
        bias = float(np.median(wrap90(line_true[others] - palm[others, 0])))
        fused.append(fuse(pred[i], *palm[i], bias))
    err_fused = angle_error(fused, theta)
    palm_bias = float(np.median(wrap90(line_true[ok] - palm[ok, 0])))
    direction_acc = summary(err_fused)
    print(f"direction, palm line + model end, leave one grasp out: {direction_acc} "
          f"(palm line used on {ok.sum()}/{len(d)}, calibration offset {palm_bias:+.1f} deg)")

    bundle = {
        "features": FEATURES,
        "sensor": sensor_model().fit(X, y),
        "direction": direction_model().fit(Xd, Y),
        "sensor_loo": sensor_acc,
        "direction_loo": direction_acc,
        "palm_bias_deg": palm_bias,
        "trained_on": f"models/dataset_left: {n} grasps for the sensor, {len(d)} with photo directions",
    }
    joblib.dump(bundle, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
