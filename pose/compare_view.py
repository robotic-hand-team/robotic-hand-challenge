"""Render an estimated pose from the same viewpoint as the ground-truth photos
(fingers up, palm toward the camera, thumb on the right), plus a side view.

Run run_pipeline.py for the session first, then:
    python compare_view.py --session 20261016_204513 --object vegabar29
Output: output/<object>/<session>_photo_view.png
"""
import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import config as C
import render as R
from data_io import hold_window, joint_raw, load_session
from hand_model import HandModel
from pose_fit import ObjectModel


def find_result(object_key, session):
    for folder in (C.OUTPUT_DIR / object_key, C.OUTPUT_DIR / "auto"):
        f = folder / "results.csv"
        if f.exists():
            for row in csv.DictReader(open(f)):
                if row["session"] == session and row.get("T_hand_obj"):
                    return row, folder
    raise SystemExit(f"No result for {session} / {object_key}. Run run_pipeline.py first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--object", required=True, choices=list(C.OBJECTS))
    a = ap.parse_args()

    row, folder = find_result(a.object, a.session)
    T = np.array(json.loads(row["T_hand_obj"]))
    obj = ObjectModel(C.OBJECTS[a.object], a.object)
    om = obj.mesh.copy()
    om.apply_transform(T)

    s = load_session(C.LOG_DIR / a.session)
    hand = HandModel()
    hand.set_raw(joint_raw(s, hold_window(s)))
    meshes = hand.posed_meshes()

    parts = [R._faces(R._light(m, 6000 if link == "hand_base_link" else 2500), R.HAND_COLOR, 1.0)
             for link, m in meshes.items()]
    parts.append(R._faces(R._light(om, 20000), R.OBJECT_COLOR, 1.0))
    polys = np.concatenate([p for p, _ in parts])
    cols = np.concatenate([c for _, c in parts])
    # hand frame -> photo frame: across = -z, depth = x (palm faces -x), up = -y (fingers up)
    if C.HAND_SIDE == "left":
        polys = polys * np.array([1, 1, -1])
    Pp = np.stack([-polys[..., 2], polys[..., 0], -polys[..., 1]], -1)

    fig = plt.figure(figsize=(11, 6))
    for i, (elev, azim, title) in enumerate([(8, -90, "Front: palm toward camera (photo view)"),
                                             (8, 0, "Side view")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(Pp.copy(), facecolors=cols, edgecolors=cols,
                                             linewidths=0.1))
        lo, hi = Pp.reshape(-1, 3).min(0), Pp.reshape(-1, 3).max(0)
        mid, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(f"{a.session} | {a.object} | {row.get('status', '')} | "
                 f"fit {row.get('fit_rms_mm', '?')} mm | hint {row.get('axis_hint', 'none')}",
                 fontsize=12)
    fig.tight_layout()
    out = folder / f"{a.session}_photo_view.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
