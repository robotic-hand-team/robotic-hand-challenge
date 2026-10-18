"""Replay labelled left-hand sessions through the live pipeline, frame by frame, as the hand GUI would send
them, and compare its one decision per grasp (sensor A/B, direction of the yellow end) with the labels.

The logs have neither the GUI state nor a live clock, so both are rebuilt: CLOSING before the hold window
(classifier/left_hand_data.hold_window), HOLD from its start until contact is gone (the GUI holds until the
hand opens), OPENED after; time = the logged t_unix.

The left-hand models were trained on these grasps, so this checks the live plumbing (the decision the live
code makes from its own averaging matches the labels), not accuracy on new grasps. The honest numbers are
the leave-one-grasp-out ones from classifier/train_left_hand.py.

Run from the repo root:  python -m backend.eval_live [--sources photo,inferred]
"""
import argparse
import math
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.src import live
from backend.src.twin import Twin
from classifier.left_hand_data import DATA_DIR, GROUPS, LABELS, PALM_COLS, hold_window, load_frames

STATS = ("sum", "max", "contacts", "cy", "cx", "force")
DOF = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]


class Clock:
    """Stands in for live.time: time() returns the replayed frame's timestamp."""
    t = 0.0

    def time(self):
        return self.t


def messages(session: Path):
    """The GUI's UDP messages for one logged session (hardware/config.py broadcast())."""
    df = load_frames(session)
    a, _ = hold_window(df)
    # the GUI stays in HOLD until the hand opens: here, until contact is gone after the hold started
    contacts = df[[f"{g}_contacts" for g in GROUPS]].fillna(0).sum(axis=1).to_numpy()
    after = np.flatnonzero((np.arange(len(df)) >= a) & (contacts <= live.C.MIN_CONTACTS_ACTIVE))
    b = int(after[0]) if len(after) else len(df)
    palm = pd.read_csv(session / "tactile_raw.csv", usecols=["frame"] + PALM_COLS)
    palm = df[["frame"]].merge(palm, on="frame", how="left").fillna(0)[PALM_COLS].to_numpy(int)
    for i, r in enumerate(df.itertuples(index=False)):
        r = r._asdict()
        tac = {g: {s: (None if (v := r.get(f"{g}_{s}")) is None or (isinstance(v, float) and math.isnan(v))
                       else v) for s in STATS} for g in GROUPS}
        for g in tac:
            tac[g]["contacts"] = int(tac[g]["contacts"] or 0)
            tac[g]["sum"] = tac[g]["sum"] or 0
        state = "CLOSING" if i < a else "HOLD" if i < b else "OPENED"
        yield r["t_unix"], {"seq": i, "t": r["t_rel"], "angle": [int(r[f"angle_act_{d}"]) for d in DOF],
                            "force_act": [int(r[f"force_act_{d}"]) for d in DOF], "tactile": tac, "state": state,
                            "palm": palm[i].tolist(), "hand_side": "left"}


def angle_error(a, b):
    return abs((a - b + 180) % 360 - 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="photo", help="label sources to replay, comma-separated")
    args = ap.parse_args()
    clock = Clock()
    live.time = clock                                    # the pipeline's time.time() follows the replay
    pipe = live.LivePipeline(Twin())
    labels = pd.read_csv(LABELS, dtype={"session": str})
    labels = labels[labels["label_source"].isin(args.sources.split(","))]
    rows = []
    t0 = _time.time()
    for _, lab in labels.iterrows():
        pipe._reset_grasp()
        pipe.state = None
        decided = None
        for t, msg in messages(DATA_DIR / lab["session"]):
            clock.t = t
            out = pipe.process(msg, ("replay", 0))
            if out["kpi"]["sensor"] and decided is None:
                decided = (out["kpi"]["sensor"]["label"], out["kpi"]["direction_deg"], out["object"] is not None,
                           out["kpi"].get("direction_source"))
        pred, theta, shown, source = decided or (None, None, False, None)
        truth_theta = (math.degrees(math.atan2(lab["end_up"], lab["end_thumb"]))
                       if pd.notna(lab["end_thumb"]) else None)
        rows.append({"session": lab["session"][9:], "sensor": lab["sensor"], "pred": pred, "shown": shown,
                     "dir_source": source,
                     "dir_true": truth_theta, "dir_pred": theta,
                     "dir_err": None if truth_theta is None or theta is None else angle_error(theta, truth_theta)})
    res = pd.DataFrame(rows)
    ok = res["pred"] == res["sensor"]
    print(f"replayed {len(res)} sessions in {_time.time() - t0:.0f} s")
    print(f"decided: {res['pred'].notna().sum()}/{len(res)}, object drawn: {res['shown'].sum()}/{len(res)}")
    print(f"sensor agrees with the label: {ok.sum()}/{len(res)}")
    print(f"direction from: {res['dir_source'].value_counts().to_dict()}")
    e = res["dir_err"].dropna().to_numpy(float)
    if len(e):
        print(f"direction: median error {np.median(e):.0f} deg, {np.mean(e <= 30):.0%} within 30 deg, "
              f"{np.mean(e > 135):.0%} pointing the wrong way ({len(e)} grasps)")
    bad = res[~ok | (res["dir_err"] > 45)]
    if len(bad):
        print("misses:\n" + bad.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
