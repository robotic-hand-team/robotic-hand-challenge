"""
Plots a logging session recorded by config.py.

  python3 plot_logs.py                          -> latest session in ./logs
  python3 plot_logs.py logs/20261016_201443     -> a specific session
  python3 plot_logs.py --frame 20               -> also show tactile heatmaps of frame 20
  python3 plot_logs.py --save                   -> save PNGs into the session folder instead of opening windows

Needs only numpy and matplotlib (pip install numpy matplotlib).
"""
import os
import sys
import csv
import json
import argparse

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is missing – install it with:  pip install matplotlib")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOF_LABELS = ["Little", "Ring", "Middle", "Index", "Thumb bend", "Thumb rot"]


# ----------------------------
# Loading
# ----------------------------
def latest_session():
    log_dir = os.path.join(BASE_DIR, "logs")
    if not os.path.isdir(log_dir):
        sys.exit(f"No logs folder found at {log_dir}")
    sessions = sorted(d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)))
    if not sessions:
        sys.exit("No sessions found in ./logs")
    return os.path.join(log_dir, sessions[-1])


def load_csv(path):
    """Reads a CSV into {column: float array}; empty cells become NaN."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if len(r) == len(header)]
    if not rows:
        sys.exit(f"{os.path.basename(path)} contains no data rows.")
    data = np.array([[float(v) if v != "" else np.nan for v in r] for r in rows])
    return {col: data[:, i] for i, col in enumerate(header)}


def group_key(name):
    return name.lower().replace(" ", "_")


# ----------------------------
# Plots
# ----------------------------
def plot_actuators(act, meta):
    dofs = meta["dof"]
    t = act["t_rel"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 10))
    for i, d in enumerate(dofs):
        line, = axes[0].plot(t, act[f"angle_act_{d}"], label=DOF_LABELS[i])
        axes[0].plot(t, act[f"angle_set_{d}"], "--", color=line.get_color(), alpha=0.5)
        axes[1].plot(t, act[f"force_act_{d}"], label=DOF_LABELS[i])
        axes[2].plot(t, act[f"current_{d}"], label=DOF_LABELS[i])
        axes[3].plot(t, act[f"temp_{d}"], label=DOF_LABELS[i])
    axes[0].set_ylabel("Angle\n(solid = actual, dashed = target)")
    axes[1].set_ylabel("Force")
    axes[2].set_ylabel("Current")
    axes[3].set_ylabel("Temperature")
    axes[3].set_xlabel("Time [s]")
    axes[0].legend(ncol=6, fontsize=8, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Actuators")
    fig.tight_layout()
    return fig


def plot_tactile_summary(summ, meta):
    t = summ["t_rel"]
    groups = meta["sensor_groups"]
    fingers = []
    for g in groups:
        if g["finger"] not in fingers:
            fingers.append(g["finger"])

    fig, axes = plt.subplots(len(fingers), 1, sharex=True, figsize=(11, 2 * len(fingers)))
    for ax, finger in zip(axes, fingers):
        for g in groups:
            if g["finger"] == finger:
                ax.plot(t, summ[f"{group_key(g['name'])}_sum"], label=g["name"])
        ax.set_ylabel(f"{finger}\nsum")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Tactile sensors – total pressure per group")
    fig.tight_layout()
    return fig


def plot_frame(raw, meta, frame):
    idx = np.where(raw["frame"] == frame)[0]
    if len(idx) == 0:
        print(f"Frame {frame} not found (recorded frames: 0–{int(raw['frame'][-1])}).")
        return None
    k = idx[0]
    groups = meta["sensor_groups"]

    mats = []
    for g in groups:
        key = group_key(g["name"])
        vals = [raw[f"{key}_r{i}_c{j}"][k] for i in range(g["rows"]) for j in range(g["cols"])]
        mats.append(np.array(vals).reshape(g["rows"], g["cols"]))
    vmax = max(1.0, max(np.nanmax(m) for m in mats))

    cols = 6
    rows = int(np.ceil(len(groups) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.8 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    im = None
    for ax, g, m in zip(axes.ravel(), groups, mats):
        im = ax.imshow(m, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(g["name"], fontsize=9)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    fig.suptitle(f"Tactile heatmaps – frame {frame} (t = {raw['t_rel'][k]:.2f} s)")
    return fig


# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("session", nargs="?", help="session folder (default: latest)")
    p.add_argument("--frame", type=int, help="frame to show as tactile heatmaps")
    p.add_argument("--save", action="store_true", help="save PNGs instead of showing windows")
    args = p.parse_args()

    session = os.path.abspath(args.session) if args.session else latest_session()
    print(f"Session: {session}")

    with open(os.path.join(session, "meta.json")) as f:
        meta = json.load(f)
    act = load_csv(os.path.join(session, "actuators.csv"))
    summ = load_csv(os.path.join(session, "tactile_summary.csv"))
    print(f"{len(act['frame'])} frames, {act['t_rel'][-1]:.1f} s")

    figs = {
        "actuators": plot_actuators(act, meta),
        "tactile_summary": plot_tactile_summary(summ, meta),
    }

    raw_path = os.path.join(session, "tactile_raw.csv")
    if args.frame is not None:
        if os.path.exists(raw_path):
            fig = plot_frame(load_csv(raw_path), meta, args.frame)
            if fig is not None:
                figs[f"tactile_frame_{args.frame}"] = fig
        else:
            print("tactile_raw.csv not found – heatmaps skipped.")

    # Fall back to saving PNGs if no display/GUI backend is available
    no_gui = matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps")
    if args.save or no_gui:
        for name, fig in figs.items():
            path = os.path.join(session, f"{name}.png")
            fig.savefig(path, dpi=120)
            print(f"Saved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
