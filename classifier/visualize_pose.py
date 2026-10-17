"""
Pose visualization: robotic hand skeleton + predicted object position.

Usage:
    python visualize_pose.py <session_path>       # animate through frames
    python visualize_pose.py                       # pick a random sim session

The 3D plot shows:
  - Hand skeleton at the session's mean joint angles (grey sticks)
  - Predicted XYZ position  (red sphere)
  - Ground truth XYZ (green sphere, only available for sim sessions)
  - Three 2D projection panels below (front / side / top)

Place the plot window next to your camera feed for visual comparison.
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" or "MacOSX" if TkAgg is absent
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.spatial.transform import Rotation

HERE      = Path(__file__).resolve().parent
SIM_DIR   = HERE.parent / "models" / "dataset_sim"
MODEL_DIR = HERE / "model"
XML_PATH  = HERE.parent / "simulation" / "Inspire_hand" / "xml" / "inspire_hand_right.xml"

META_COLS     = {"frame", "t_unix", "t_rel"}
ACTUATOR_COLS = [
    "angle_act_little", "angle_act_ring", "angle_act_middle",
    "angle_act_index", "angle_act_thumb_bend", "angle_act_thumb_rot",
]
DOF_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
DOF_TO_JOINT = {
    "little":     "right_pinky_proximal_joint",
    "ring":       "right_ring_proximal_joint",
    "middle":     "right_middle_proximal_joint",
    "index":      "right_index_proximal_joint",
    "thumb_bend": "right_thumb_proximal_pitch_joint",
    "thumb_rot":  "right_thumb_proximal_yaw_joint",
}
DOF_QMAX = {"little": 1.47, "ring": 1.47, "middle": 1.47, "index": 1.47,
            "thumb_bend": 0.6, "thumb_rot": 1.308}

# Kinematic chains to draw (list of body names, connected in order)
CHAINS = [
    ["right_hand_root", "right_pinky_proximal",  "right_pinky_intermediate"],
    ["right_hand_root", "right_ring_proximal",   "right_ring_intermediate"],
    ["right_hand_root", "right_middle_proximal", "right_middle_intermediate"],
    ["right_hand_root", "right_index_proximal",  "right_index_intermediate"],
    ["right_hand_root", "right_thumb_proximal_base",
     "right_thumb_proximal", "right_thumb_intermediate", "right_thumb_distal"],
]


# ── Forward kinematics ────────────────────────────────────────────────────────

def _parse_vec(s):
    return np.array([float(x) for x in s.split()]) if s else np.zeros(3)

def _parse_quat(s):
    return np.array([float(x) for x in s.split()]) if s else np.array([1.,0.,0.,0.])

def _R_from_mj_quat(q):          # w,x,y,z → rotation matrix
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()

def _rot_axis(axis, angle):
    ax = np.asarray(axis, float)
    norm = np.linalg.norm(ax)
    if norm < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(ax / norm * angle).as_matrix()


def build_body_tree(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bodies, parents = {}, {}

    def traverse(el, parent=None):
        for child in el:
            if child.tag == "body":
                name = child.get("name")
                pos  = _parse_vec(child.get("pos", ""))
                quat = _parse_quat(child.get("quat", ""))
                joints = []
                for sub in child:
                    if sub.tag == "joint":
                        joints.append({
                            "name":  sub.get("name", ""),
                            "axis":  _parse_vec(sub.get("axis", "0 0 1")),
                            "range": [float(x) for x in sub.get("range", "0 1").split()],
                        })
                bodies[name]  = {"pos": pos, "R": _R_from_mj_quat(quat), "joints": joints}
                parents[name] = parent
                traverse(child, name)

    traverse(root.find("worldbody"))
    return bodies, parents


BODY_INFO, BODY_PARENT = build_body_tree(XML_PATH)


def world_pose(body_name, joint_q, _cache=None):
    if _cache is None:
        _cache = {}
    if body_name in _cache:
        return _cache[body_name]

    parent = BODY_PARENT.get(body_name)
    b = BODY_INFO[body_name]

    if parent is None or parent not in BODY_INFO:
        # Root level
        pos, R = b["pos"].copy(), b["R"].copy()
    else:
        pp, pR = world_pose(parent, joint_q, _cache)
        joint_R = np.eye(3)
        for j in b["joints"]:
            q = joint_q.get(j["name"], 0.0)
            joint_R = _rot_axis(j["axis"], q) @ joint_R
        pos = pp + pR @ b["pos"]
        R   = pR @ b["R"] @ joint_R

    _cache[body_name] = (pos, R)
    return pos, R


def angle_acts_to_joint_q(angle_act_row: pd.Series) -> dict:
    q = {}
    for dof in DOF_NAMES:
        col = f"angle_act_{dof}"
        if col in angle_act_row:
            act = float(angle_act_row[col])
            q[DOF_TO_JOINT[dof]] = (1.0 - act / 1000.0) * DOF_QMAX[dof]
    return q


def finger_positions(joint_q):
    """Return {body_name: world_pos (mm)} for all bodies in CHAINS."""
    cache = {}
    pos   = {}
    for chain in CHAINS:
        for name in chain:
            if name not in pos:
                p, _ = world_pose(name, joint_q, cache)
                pos[name] = p * 1000.0    # metres → mm
    return pos


# ── Prediction ────────────────────────────────────────────────────────────────

def load_predictor():
    model     = joblib.load(MODEL_DIR / "pose_regressor.pkl")
    scaler    = joblib.load(MODEL_DIR / "scaler_pose.pkl")
    feat_cols = joblib.load(MODEL_DIR / "feature_cols_pose.pkl")
    return model, scaler, feat_cols


def predict_position(session_df, model, scaler, feat_cols):
    """Returns (N,3) array of predicted XYZ in mm."""
    missing = [c for c in feat_cols if c not in session_df.columns]
    for c in missing:
        session_df[c] = 0.0
    X = session_df[feat_cols].fillna(0).values
    X_s = scaler.transform(X)
    pred = model.predict(X_s)          # (N, 9): [pos_x_mm, pos_y_mm, pos_z_mm, sin/cos …]
    return pred[:, :3]                 # only XYZ


# ── Visualization ─────────────────────────────────────────────────────────────

PALM_COLOR   = "#607D8B"
FINGER_COLOR = "#90A4AE"
THUMB_COLOR  = "#78909C"
PRED_COLOR   = "#F44336"
GT_COLOR     = "#4CAF50"
OBJECT_ALPHA = 0.55


def draw_hand_3d(ax, body_pos, lw=2):
    for i, chain in enumerate(CHAINS):
        color = THUMB_COLOR if "thumb" in chain[-1] else FINGER_COLOR
        xs = [body_pos[n][0] for n in chain]
        ys = [body_pos[n][1] for n in chain]
        zs = [body_pos[n][2] for n in chain]
        ax.plot(xs, ys, zs, "-o", color=color, lw=lw, ms=4, zorder=2)
    # Palm rectangle (index → pinky proximal)
    palm_pts = [body_pos[c[1]] for c in CHAINS[:4]]   # the 4 proximal finger bases
    palm_pts.append(body_pos["right_hand_root"])
    px = [p[0] for p in palm_pts] + [palm_pts[0][0]]
    py = [p[1] for p in palm_pts] + [palm_pts[0][1]]
    pz = [p[2] for p in palm_pts] + [palm_pts[0][2]]
    ax.plot(px, py, pz, color=PALM_COLOR, lw=lw, zorder=1)


def scatter_object(ax, pos_mm, color, label, size=180, zorder=5, marker="o"):
    ax.scatter(*pos_mm, s=size, c=color, marker=marker, label=label,
               edgecolors="white", linewidths=1.5, zorder=zorder, alpha=OBJECT_ALPHA)


def setup_axes_3d(ax):
    ax.set_xlabel("X  (palm depth, mm)", fontsize=7, labelpad=2)
    ax.set_ylabel("Y  (lateral, mm)", fontsize=7, labelpad=2)
    ax.set_zlabel("Z  (height, mm)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_xlim(-10, 100); ax.set_ylim(50, 160); ax.set_zlim(40, 180)
    ax.view_init(elev=20, azim=-60)


def project_views(axes_2d, body_pos, pred_mm, gt_mm=None):
    """Draw top/side/front projections."""
    projections = [
        (0, 2, "front",  "X (palm depth)", "Z (height)"),
        (1, 2, "side",   "Y (lateral)",    "Z (height)"),
        (0, 1, "top",    "X (palm depth)", "Y (lateral)"),
    ]
    for ax, (xi, yi, title, xl, yl) in zip(axes_2d, projections):
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal")
        # Draw hand skeleton
        for chain in CHAINS:
            color = THUMB_COLOR if "thumb" in chain[-1] else FINGER_COLOR
            xs = [body_pos[n][xi] for n in chain]
            ys = [body_pos[n][yi] for n in chain]
            ax.plot(xs, ys, "-o", color=color, lw=1.5, ms=3)
        # Object markers
        ax.scatter(pred_mm[xi], pred_mm[yi],
                   s=120, c=PRED_COLOR, zorder=5, edgecolors="white",
                   linewidths=1, label="predicted")
        if gt_mm is not None:
            ax.scatter(gt_mm[xi], gt_mm[yi],
                       s=120, c=GT_COLOR, zorder=4, marker="^",
                       edgecolors="white", linewidths=1, label="ground truth")
        ax.legend(fontsize=6, loc="upper right")


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize_session(session_path: Path, animate=True, save_png: Path | None = None):
    session_path = Path(session_path)
    print(f"Loading session: {session_path.name}")

    tac  = pd.read_csv(session_path / "tactile_summary.csv")
    act  = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df   = tac.merge(act, on="frame")

    # Ground truth (only available for sim sessions)
    gt_available = (session_path / "object_pose.csv").exists()
    if gt_available:
        pose = pd.read_csv(session_path / "object_pose.csv")
        df   = df.merge(pose[["frame", "phase",
                               "pos_x", "pos_y", "pos_z"]], on="frame", how="left")
        hold = df[df["phase"] == "hold"] if "phase" in df.columns else df
    else:
        hold = df

    hold = hold.reset_index(drop=True)
    print(f"  {len(hold)} hold frames")

    model, scaler, feat_cols = load_predictor()
    pred_xyz = predict_position(hold, model, scaler, feat_cols)   # (N, 3) mm

    # Mean joint angles for hand skeleton display
    mean_acts  = hold[[f"angle_act_{d}" for d in DOF_NAMES]].mean()
    mean_acts.index = mean_acts.index.str.replace("angle_act_", "")
    joint_q    = {DOF_TO_JOINT[d]: (1.0 - mean_acts[d] / 1000.0) * DOF_QMAX[d]
                  for d in DOF_NAMES}
    body_pos   = finger_positions(joint_q)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#1E1E2E")
    gs  = gridspec.GridSpec(2, 4, figure=fig, wspace=0.35, hspace=0.4,
                            height_ratios=[2, 1])

    ax3d  = fig.add_subplot(gs[0, :2], projection="3d")
    ax3d.set_facecolor("#1E1E2E")

    ax_info = fig.add_subplot(gs[0, 2:])
    ax_info.axis("off")

    axes_2d = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax_err  = fig.add_subplot(gs[1, 3])
    for ax in axes_2d + [ax_info, ax_err]:
        ax.set_facecolor("#1E1E2E")
    for ax in axes_2d + [ax_err]:
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555566")

    # Dark 3D axes
    for pane in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        pane.pane.fill = False
        pane.pane.set_edgecolor("#444455")
    ax3d.tick_params(colors="white", labelsize=6)
    ax3d.xaxis.label.set_color("white")
    ax3d.yaxis.label.set_color("white")
    ax3d.zaxis.label.set_color("white")

    # Title
    fig.suptitle(f"Pose estimator  —  {session_path.name}",
                 color="white", fontsize=11, y=0.98)

    n_frames = len(pred_xyz)
    STEP     = max(1, n_frames // 40)     # at most 40 animation frames

    frame_indices = list(range(0, n_frames, STEP))
    gt_avail_col  = gt_available and "pos_x" in hold.columns

    # Error over time (for info panel)
    if gt_avail_col:
        gt_xyz = hold[["pos_x", "pos_y", "pos_z"]].values * 1000.0
        errs   = np.linalg.norm(pred_xyz - gt_xyz, axis=1)

    def draw_frame(fi):
        ax3d.cla()
        for ax in axes_2d:
            ax.cla()
        ax_info.cla(); ax_err.cla()
        ax_info.axis("off"); ax_err.set_facecolor("#1E1E2E")

        setup_axes_3d(ax3d)
        draw_hand_3d(ax3d, body_pos)

        pred_mm = pred_xyz[fi]
        gt_mm   = gt_xyz[fi] if gt_avail_col else None

        scatter_object(ax3d, pred_mm, PRED_COLOR, "predicted")
        if gt_mm is not None:
            scatter_object(ax3d, gt_mm, GT_COLOR, "ground truth", marker="^")

        ax3d.legend(fontsize=7, loc="upper right",
                    facecolor="#2A2A3E", labelcolor="white")

        project_views(axes_2d, body_pos, pred_mm, gt_mm)

        # Info text
        lines = [
            f"Frame  {fi+1}/{n_frames}",
            "",
            f"Predicted position",
            f"  X = {pred_mm[0]:+.1f} mm  (palm depth)",
            f"  Y = {pred_mm[1]:+.1f} mm  (lateral)",
            f"  Z = {pred_mm[2]:+.1f} mm  (height)",
        ]
        if gt_mm is not None:
            err = np.linalg.norm(pred_mm - gt_mm)
            lines += [
                "",
                "Ground truth",
                f"  X = {gt_mm[0]:+.1f} mm",
                f"  Y = {gt_mm[1]:+.1f} mm",
                f"  Z = {gt_mm[2]:+.1f} mm",
                "",
                f"Error  {err:.1f} mm",
            ]
        for i, line in enumerate(lines):
            color = "#AAAACC" if not line.startswith("  ") else "#DDDDFF"
            ax_info.text(0.05, 0.95 - i*0.075, line,
                         transform=ax_info.transAxes,
                         color=color, fontsize=8.5, va="top",
                         fontfamily="monospace")

        # Error timeline
        if gt_avail_col:
            ax_err.plot(errs, color="#607D8B", lw=1, alpha=0.5)
            ax_err.axvline(fi, color=PRED_COLOR, lw=1.5)
            ax_err.axhline(np.mean(errs), color=GT_COLOR, lw=1, ls="--",
                           label=f"mean {np.mean(errs):.1f}mm")
            ax_err.set_xlim(0, n_frames - 1)
            ax_err.set_ylim(bottom=0)
            ax_err.set_xlabel("frame", fontsize=7, color="white")
            ax_err.set_ylabel("error (mm)", fontsize=7, color="white")
            ax_err.set_title("position error", fontsize=8, color="white", pad=3)
            ax_err.tick_params(colors="white")
            for sp in ax_err.spines.values():
                sp.set_edgecolor("#555566")
            ax_err.legend(fontsize=7, facecolor="#2A2A3E", labelcolor="white")

        fig.canvas.draw_idle()

    if save_png:
        draw_frame(n_frames // 2)
        plt.savefig(save_png, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {save_png}")
        return

    # Interactive animation
    state = {"fi": 0, "running": animate}

    def on_key(event):
        if event.key == " ":
            state["running"] = not state["running"]
        elif event.key == "right":
            state["fi"] = min(n_frames - 1, state["fi"] + 1)
            draw_frame(state["fi"])
        elif event.key == "left":
            state["fi"] = max(0, state["fi"] - 1)
            draw_frame(state["fi"])

    fig.canvas.mpl_connect("key_press_event", on_key)

    draw_frame(0)

    if animate:
        import matplotlib.animation as animation

        def _step(i):
            if state["running"]:
                state["fi"] = (state["fi"] + 1) % n_frames
                draw_frame(state["fi"])

        ani = animation.FuncAnimation(fig, _step,
                                      interval=200, cache_frame_data=False)
        plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        session = Path(sys.argv[1])
    else:
        # Default: first sim session
        sessions = sorted(SIM_DIR.iterdir())
        session  = sessions[10]   # pick a mid-range one

    # Also export a static PNG alongside the session
    png_out = HERE / f"pose_viz_{Path(session).name}.png"
    visualize_session(session, animate=True, save_png=None)
