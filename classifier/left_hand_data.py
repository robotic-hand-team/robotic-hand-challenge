"""Labelled left-hand grasps (models/dataset_left, 17 Oct 2026) as one row per grasp.

Each grasp is reduced to the mean over its hold window (the same window detection as final_model/, so the
features match what the live pipeline averages over a settled hold). Labels come from
models/dataset_left/labels.csv: sensor A/B from the session photos ("photo") or from the recording order
("inferred", a session between two of the same sensor), plus the direction of the sensor's yellow end in
the hand (end_thumb = towards the thumb, end_up = towards the fingertips).

Run from the repo root: python -m classifier.left_hand_data   (prints a summary)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "models" / "dataset_left"
LABELS = DATA_DIR / "labels.csv"

DOF = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
FINGERS = ["little", "ring", "middle", "index"]
GROUPS = [f"{f}_{p}" for f in FINGERS for p in ("tip", "nail", "pad")] + \
         ["thumb_tip", "thumb_nail", "thumb_mid", "thumb_pad", "palm"]
STATS = ["sum", "max", "contacts", "cy", "cx", "force"]

# final_model/match_session.py (via matcher/): hold = the longest stretch where every pos_act column's
# rolling std stays below STABILITY_STD; fall back to the middle half of the session.
STABILITY_STD, STABILITY_WIN, MIN_HOLD_FRAMES = 20.0, 10, 15


def hold_window(act: pd.DataFrame) -> tuple[int, int]:
    cols = [f"pos_act_{d}" for d in DOF if f"pos_act_{d}" in act.columns]
    n = len(act)
    if not cols or n < MIN_HOLD_FRAMES:
        return n // 4, max(n // 4 + 1, 3 * n // 4)
    rstd = act[cols].rolling(STABILITY_WIN, center=True).std().fillna(9999).max(axis=1).to_numpy()
    stable = rstd < STABILITY_STD
    best, start = (0, 0), None
    for i, s in enumerate(list(stable) + [False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best[1] - best[0] < MIN_HOLD_FRAMES:
        return n // 4, max(n // 4 + 1, 3 * n // 4)
    return best


def load_frames(session: Path) -> pd.DataFrame:
    tac = pd.read_csv(session / "tactile_summary.csv")
    act = pd.read_csv(session / "actuators.csv")
    keep = ["frame"] + [c for c in act.columns if c.split("_")[0] in ("angle", "force", "current", "pos")
                        and c.startswith(("angle_act", "force_act", "current_", "pos_act"))]
    return tac.merge(act[keep], on="frame")


def hold_mean(session: Path) -> pd.Series:
    df = load_frames(session)
    a, b = hold_window(df)
    # cy/cx are blank (NaN) when a patch has < 2 contacts. They count as 0 in every frame before averaging,
    # exactly like the live pipeline (HoldMeanSensorClassifier.vector) and final_model/match_session.py.
    hold = df.iloc[a:b].fillna(0.0)
    m = hold.mean(numeric_only=True)
    m["hold_frames"] = len(hold)
    return m


def load(sources=("photo", "inferred")) -> pd.DataFrame:
    """One row per labelled grasp: hold-mean columns + label columns."""
    labels = pd.read_csv(LABELS, dtype={"session": str})
    labels = labels[labels["label_source"].isin(sources)].reset_index(drop=True)
    rows = []
    for _, r in labels.iterrows():
        m = hold_mean(DATA_DIR / r["session"])
        rows.append(pd.concat([m, r]))
    return pd.DataFrame(rows).reset_index(drop=True)


if __name__ == "__main__":
    df = load()
    print(df.groupby(["label_source", "sensor"]).size())
    print("hold frames: median", int(df["hold_frames"].median()), "min", int(df["hold_frames"].min()))
