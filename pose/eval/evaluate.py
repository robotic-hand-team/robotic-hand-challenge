"""Score the pose fit against the photo-labelled real grasps in labels.py.

Fits every labelled session (no axis hint) with the current pose/ settings and compares the
direction of the sensor's CAD +y end with the photo. Takes ~1 min per session.

Usage (from pose/):  python eval/evaluate.py            # all labelled sessions
                     python eval/evaluate.py 20261016_204637 20261016_204609
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from contacts import extract_contacts  # noqa: E402
from data_io import hold_window, joint_raw, load_session  # noqa: E402
from hand_model import HandModel  # noqa: E402
from labels import LABELS, end_dir  # noqa: E402
from pose_fit import ObjectModel, fit_pose  # noqa: E402
from run_pipeline import hand_surface_points  # noqa: E402


def family(v):
    a = np.degrees(np.arctan2(abs(v[1]), abs(v[2])))  # 0 = across the palm, 90 = along the fingers
    return "across" if a < 30 else "along" if a > 60 else "diagonal"


def main():
    sessions = sys.argv[1:] or list(LABELS)
    C.set_hand("right")  # the 13 labelled grasps (2026-10-16) were recorded with the right hand
    hand, objs, rows = HandModel(), {}, []
    print(f"{'session':16s} {'sensor':9s} {'photo':9s} {'fit':9s} {'end dir err':>11s} {'axis err':>8s} {'rms':>5s}")
    for sess in sessions:
        name, _, note = LABELS[sess]
        obj = objs.setdefault(name, ObjectModel(C.OBJECTS[name], name))
        s = load_session(C.LOG_DIR / sess)
        fr = hold_window(s)
        hand.set_raw(joint_raw(s, fr))
        cs, _ = extract_contacts(s, fr, hand)
        m = hand.posed_meshes()
        f = fit_pose(obj, cs, hand_surface_points(m), axis_hint=None, check_points=hand_surface_points(m, 600))
        d = (f.T @ obj.cad_to_obj)[:3, 1]                       # CAD +y end in the hand frame
        p = np.array([0, d[1], d[2]]) / (np.hypot(d[1], d[2]) + 1e-9)
        signed = np.degrees(np.arccos(np.clip(p @ end_dir(sess), -1, 1)))
        unsigned = min(signed, 180 - signed)
        rows.append((signed, unsigned))
        print(f"{sess:16s} {name:9s} {family(end_dir(sess)):9s} {family(p):9s} {signed:11.0f} {unsigned:8.0f} {f.rms_mm:5.1f}",
              flush=True)
    s_, u_ = np.array(rows).T
    print(f"\naxis error (either end): median {np.median(u_):.0f} deg, within 20 deg: {(u_ <= 20).sum()}/{len(u_)}")
    print(f"end direction right (within 45 deg): {(s_ <= 45).sum()}/{len(s_)}, flipped (>135 deg): {(s_ >= 135).sum()}/{len(s_)}")


if __name__ == "__main__":
    main()
