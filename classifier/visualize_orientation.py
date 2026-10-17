"""
Orientation + position overview plot (top-down view).

Generates position_and_orientation.png in the project root:
  Left:  top-down hand view — one arrow per test session showing GT tilt
         direction, coloured green (correct) / red (misclassified)
  Right: tilt-magnitude histogram with misclassified sessions overlaid

Usage:
    python classifier/visualize_orientation.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE    = Path(__file__).resolve().parent
SIM_DIR = HERE.parent / "models" / "dataset_sim"
MODEL   = HERE / "model"
OUT     = HERE.parent / "position_and_orientation.png"

ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]

BG        = "#1E1E2E"
CORRECT_C = "#4CAF50"
WRONG_C   = "#F44336"
HAND_C    = "#607D8B"
HIST_C    = "#607D8B"
THRESH_C  = "#FF9800"


def load_models():
    return (
        joblib.load(MODEL / "pose_regressor.pkl"),
        joblib.load(MODEL / "scaler_pose.pkl"),
        joblib.load(MODEL / "feature_cols_pose.pkl"),
        joblib.load(MODEL / "tilt_classifier.pkl"),
        joblib.load(MODEL / "scaler_tilt.pkl"),
        joblib.load(MODEL / "feature_cols_tilt.pkl"),
    )


def process_session(s, pos_model, pos_scaler, pos_cols,
                       tilt_model, tilt_scaler, tilt_cols):
    meta = json.loads((s / "meta.json").read_text())
    ep   = meta["episode_info"]
    tx   = np.rad2deg(ep["tilt_x_rad"])
    tz   = np.rad2deg(ep["tilt_z_rad"])
    tilt_mag = np.sqrt(tx ** 2 + tz ** 2)
    true_lbl = int(tilt_mag >= 10.0)

    tac  = pd.read_csv(s / "tactile_summary.csv")
    act  = pd.read_csv(s / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    pose = pd.read_csv(s / "object_pose.csv")
    df   = tac.merge(act, on="frame").merge(pose, on="frame")
    hold = df[df["phase"] == "hold"].reset_index(drop=True)
    if len(hold) < 5:
        return None

    for c in pos_cols:
        if c not in hold.columns:
            hold[c] = 0.0
    pred_xyz = pos_model.predict(
        pos_scaler.transform(hold[pos_cols].fillna(0).values)
    )[:, :3].mean(axis=0)

    for c in tilt_cols:
        if c not in hold.columns:
            hold[c] = 0.0
    pred_lbl = int(tilt_model.predict(
        tilt_scaler.transform(hold[tilt_cols].fillna(0).mean().values.reshape(1, -1))
    )[0])

    gt_xyz  = hold[["pos_x", "pos_y", "pos_z"]].values * 1000.0
    pos_err = np.linalg.norm(pred_xyz - gt_xyz.mean(axis=0))

    return dict(pred_xyz=pred_xyz, pos_err=pos_err,
                tx=tx, tz=tz, tilt_mag=tilt_mag,
                true_lbl=true_lbl, pred_lbl=pred_lbl)


def main():
    print("Loading models...")
    pos_model, pos_scaler, pos_cols, tilt_model, tilt_scaler, tilt_cols = load_models()

    sessions = sorted(SIM_DIR.iterdir())
    np.random.seed(42)
    idx      = np.random.permutation(len(sessions))
    test_idx = idx[int(len(sessions) * 0.8):]

    print(f"Processing {len(test_idx)} test sessions...")
    records = [
        r for i in test_idx
        if (r := process_session(sessions[i], pos_model, pos_scaler, pos_cols,
                                              tilt_model, tilt_scaler, tilt_cols)) is not None
    ]

    correct = sum(r["true_lbl"] == r["pred_lbl"] for r in records)
    errs    = np.array([r["pos_err"] for r in records])
    print(f"  Tilt accuracy : {correct}/{len(records)} = {correct/len(records)*100:.1f}%")
    print(f"  Pos err median: {np.median(errs):.1f} mm")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(BG)
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555566")

    # ── Panel 1: top-down hand + tilt arrows ──────────────────────────────────
    palm_x = [0, 60, 65, 50, 20, 0]
    palm_y = [60, 60, 120, 130, 130, 60]
    ax1.fill(palm_x, palm_y, color="#37474F", alpha=0.6, zorder=1)
    ax1.plot(palm_x, palm_y, color=HAND_C, lw=1.5, zorder=2)
    for fx, fy in [(10, 155), (25, 160), (40, 158), (52, 152), (-10, 115)]:
        ax1.add_patch(plt.Circle((fx, fy), 6, color=HAND_C, zorder=2))

    for r in records:
        px, py = r["pred_xyz"][0], r["pred_xyz"][1]
        mag    = r["tilt_mag"]
        dx     = r["tx"] / mag * 18 if mag > 0.5 else 0
        dy     = r["tz"] / mag * 18 if mag > 0.5 else 0
        color  = CORRECT_C if r["true_lbl"] == r["pred_lbl"] else WRONG_C
        ax1.annotate("", xy=(px + dx, py + dy), xytext=(px - dx, py - dy),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                     zorder=5)
        ax1.scatter(px, py, s=30, c=color, zorder=6, alpha=0.7, edgecolors="none")

    ax1.set_xlim(-25, 110); ax1.set_ylim(40, 185)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X  (palm depth mm)", fontsize=9)
    ax1.set_ylabel("Y  (lateral mm)",    fontsize=9)
    ax1.set_title("Predicted grasp positions + tilt direction\n(top-down view)",
                  fontsize=10, pad=8)
    ax1.legend(
        handles=[
            mpatches.Patch(color=CORRECT_C, label=f"correct tilt  ({correct}/{len(records)})"),
            mpatches.Patch(color=WRONG_C,   label=f"wrong  ({len(records)-correct}/{len(records)})"),
        ],
        fontsize=8, loc="lower right", facecolor="#2A2A3E",
        labelcolor="white", framealpha=0.8,
    )

    # ── Panel 2: tilt histogram ────────────────────────────────────────────────
    mags  = np.array([r["tilt_mag"] for r in records])
    wrong = np.array([r["true_lbl"] != r["pred_lbl"] for r in records])
    bins  = np.linspace(0, 25, 14)
    ax2.hist(mags,        bins=bins, color=HIST_C, alpha=0.7, label="all sessions", zorder=2)
    ax2.hist(mags[wrong], bins=bins, color=WRONG_C, alpha=0.85, label="misclassified", zorder=3)
    ax2.axvline(10, color=THRESH_C, lw=2, ls="--", label="threshold (10°)")
    ax2.set_xlabel("Tilt magnitude (degrees)", fontsize=9)
    ax2.set_ylabel("Sessions",                 fontsize=9)
    ax2.set_title(
        f"Tilt distribution — test set ({len(records)} sessions)\n"
        f"accuracy {correct/len(records)*100:.0f}%  |  "
        f"pos error median {np.median(errs):.1f} mm",
        fontsize=10, pad=8,
    )
    ax2.legend(fontsize=8, facecolor="#2A2A3E", labelcolor="white")

    plt.tight_layout(pad=2.0)
    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
