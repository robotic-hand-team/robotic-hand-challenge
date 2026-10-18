"""
Robotic Hand — Grasp Inference & Visualization
===============================================

Usage:
    python run.py <session_folder>        # analyse a single session
    python run.py                         # pick the first sim session

Input folder must contain:
    tactile_summary.csv   (tactile sensor data)
    actuators.csv         (finger joint angles)
    object_pose.csv       (optional — sim only, enables ground-truth overlay)

Output:
    gen_pics/<session_name>_result.png   — 8-panel diagnostics
    gen_pics/<session_name>_compare.png  — front view to compare with the camera
    Interactive matplotlib window (press SPACE to pause, ←/→ to step)

Models used  (../classifier/model/):
    pose_regressor.pkl    → XYZ position of grasped object  (≈ 9.6 mm RMSE)
    tilt_classifier.pkl   → upright / tilted  (≈ 70 % session accuracy)
    slip_detector.pkl     → stable / slipping  (frame-level)
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
sys.stdout.reconfigure(encoding="utf-8")   # Windows console defaults to cp1252 (no →)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
OUT_DIR   = HERE / "gen_pics"          # every generated image goes here
ROOT      = HERE.parent
MODEL_DIR = ROOT / "classifier" / "model"
XML_PATH  = ROOT / "simulation" / "Inspire_hand" / "xml" / "inspire_hand_right.xml"
SIM_DIR   = ROOT / "models" / "dataset_sim"

# ── Column definitions ────────────────────────────────────────────────────────
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
DOF_QMAX = {
    "little": 1.47, "ring": 1.47, "middle": 1.47, "index": 1.47,
    "thumb_bend": 0.6, "thumb_rot": 1.308,
}

CHAINS = [
    ["right_hand_root", "right_pinky_proximal",  "right_pinky_intermediate"],
    ["right_hand_root", "right_ring_proximal",   "right_ring_intermediate"],
    ["right_hand_root", "right_middle_proximal", "right_middle_intermediate"],
    ["right_hand_root", "right_index_proximal",  "right_index_intermediate"],
    ["right_hand_root", "right_thumb_proximal_base",
     "right_thumb_proximal", "right_thumb_intermediate", "right_thumb_distal"],
]

# ── Colours ───────────────────────────────────────────────────────────────────
BG         = "#1E1E2E"
FINGER_C   = "#90A4AE"
THUMB_C    = "#78909C"
PALM_C     = "#607D8B"
PRED_C     = "#F44336"
GT_C       = "#4CAF50"
SLIP_C     = "#FF9800"
UPRIGHT_C  = "#4CAF50"
TILTED_C   = "#FF5722"


# ══════════════════════════════════════════════════════════════════════════════
# Forward kinematics (from XML, no MuJoCo required)
# ══════════════════════════════════════════════════════════════════════════════

def _vec(s):
    return np.array([float(x) for x in s.split()]) if s else np.zeros(3)

def _quat(s):
    return np.array([float(x) for x in s.split()]) if s else np.array([1., 0., 0., 0.])

def _R_from_mj_quat(q):
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()

def _rot_axis(axis, angle):
    ax = np.asarray(axis, float)
    n  = np.linalg.norm(ax)
    return np.eye(3) if n < 1e-9 else Rotation.from_rotvec(ax / n * angle).as_matrix()


def _build_body_tree(xml_path):
    bodies, parents = {}, {}

    def traverse(el, parent=None):
        for child in el:
            if child.tag == "body":
                name = child.get("name")
                joints = [
                    {"name": sub.get("name", ""),
                     "axis": _vec(sub.get("axis", "0 0 1"))}
                    for sub in child if sub.tag == "joint"
                ]
                bodies[name]  = {"pos": _vec(child.get("pos", "")),
                                  "R":   _R_from_mj_quat(_quat(child.get("quat", ""))),
                                  "joints": joints}
                parents[name] = parent
                traverse(child, name)

    root = ET.parse(xml_path).getroot()
    traverse(root.find("worldbody"))
    return bodies, parents


_BODY_INFO, _BODY_PARENT = _build_body_tree(XML_PATH)


def _world_pose(name, joint_q, cache):
    if name in cache:
        return cache[name]
    parent = _BODY_PARENT.get(name)
    b      = _BODY_INFO[name]
    if parent is None or parent not in _BODY_INFO:
        pos, R = b["pos"].copy(), b["R"].copy()
    else:
        pp, pR = _world_pose(parent, joint_q, cache)
        jR = np.eye(3)
        for j in b["joints"]:
            jR = _rot_axis(j["axis"], joint_q.get(j["name"], 0.0)) @ jR
        pos = pp + pR @ b["pos"]
        R   = pR @ b["R"] @ jR
    cache[name] = (pos, R)
    return pos, R


def finger_positions_mm(joint_q: dict) -> dict:
    cache = {}
    return {name: _world_pose(name, joint_q, cache)[0] * 1000.0
            for chain in CHAINS for name in chain}


def actuators_to_joint_q(row: pd.Series) -> dict:
    return {
        DOF_TO_JOINT[d]: (1.0 - float(row[f"angle_act_{d}"]) / 1000.0) * DOF_QMAX[d]
        for d in DOF_NAMES if f"angle_act_{d}" in row
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model loading & inference
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    pos_model  = joblib.load(MODEL_DIR / "pose_regressor.pkl")
    pos_scaler = joblib.load(MODEL_DIR / "scaler_pose.pkl")
    pos_cols   = joblib.load(MODEL_DIR / "feature_cols_pose.pkl")

    tilt_model  = joblib.load(MODEL_DIR / "tilt_classifier.pkl")
    tilt_scaler = joblib.load(MODEL_DIR / "scaler_tilt.pkl")
    tilt_cols   = joblib.load(MODEL_DIR / "feature_cols_tilt.pkl")

    slip_model  = joblib.load(MODEL_DIR / "slip_detector.pkl")
    slip_scaler = joblib.load(MODEL_DIR / "scaler_slip.pkl")
    slip_cols   = joblib.load(MODEL_DIR / "feature_cols_slip.pkl")

    return (pos_model,  pos_scaler,  pos_cols,
            tilt_model, tilt_scaler, tilt_cols,
            slip_model, slip_scaler, slip_cols)


def predict_position(df, model, scaler, feat_cols):
    """Returns (N, 3) XYZ mm and (N, 6) sin/cos orientation columns."""
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    X    = scaler.transform(df[feat_cols].fillna(0).values)
    pred = model.predict(X)          # (N, 9): xyz + sin_roll,cos_roll,sin_pitch,cos_pitch,sin_yaw,cos_yaw
    return pred[:, :3], pred[:, 3:]  # (N,3), (N,6)


def predict_tilt(df, model, scaler, feat_cols):
    """Returns (label, proba): 0=upright, 1=tilted."""
    row = df[feat_cols].fillna(0).mean().values.reshape(1, -1)
    X   = scaler.transform(row)
    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return label, proba


def predict_slip(df, model, scaler, feat_cols):
    """Returns per-frame slip probability array (N,)."""
    base_feat = [c for c in feat_cols if not c.startswith("d_")]
    delta_feat = [c for c in feat_cols if c.startswith("d_")]

    avail_base  = [c for c in base_feat  if c in df.columns]
    avail_delta = [c for c in delta_feat if c in df.columns]

    df2 = df[avail_base].fillna(0).copy()
    delta = df2.diff().fillna(0)
    delta.columns = [f"d_{c}" for c in avail_base]

    combined = pd.concat([df2, delta], axis=1)
    for c in feat_cols:
        if c not in combined.columns:
            combined[c] = 0.0
    X = scaler.transform(combined[feat_cols].fillna(0).values)
    return model.predict_proba(X)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_session(session_path: Path):
    tac = pd.read_csv(session_path / "tactile_summary.csv")
    act = pd.read_csv(session_path / "actuators.csv")[["frame"] + ACTUATOR_COLS]
    df  = tac.merge(act, on="frame")

    gt_available = (session_path / "object_pose.csv").exists()
    if gt_available:
        pose = pd.read_csv(session_path / "object_pose.csv")
        df   = df.merge(pose[["frame", "phase", "pos_x", "pos_y", "pos_z"]],
                        on="frame", how="left")
        hold = df[df["phase"] == "hold"] if "phase" in df.columns else df
    else:
        hold = df

    return hold.reset_index(drop=True), gt_available


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=6)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555566")


def draw_hand_3d(ax, body_pos, lw=2):
    for chain in CHAINS:
        color = THUMB_C if "thumb" in chain[-1] else FINGER_C
        xs = [body_pos[n][0] for n in chain]
        ys = [body_pos[n][1] for n in chain]
        zs = [body_pos[n][2] for n in chain]
        ax.plot(xs, ys, zs, "-o", color=color, lw=lw, ms=4, zorder=2)
    palm = [body_pos[c[1]] for c in CHAINS[:4]] + [body_pos["right_hand_root"]]
    ax.plot([p[0] for p in palm] + [palm[0][0]],
            [p[1] for p in palm] + [palm[0][1]],
            [p[2] for p in palm] + [palm[0][2]],
            color=PALM_C, lw=lw, zorder=1)


def draw_hand_2d(ax, body_pos, xi, yi):
    for chain in CHAINS:
        color = THUMB_C if "thumb" in chain[-1] else FINGER_C
        ax.plot([body_pos[n][xi] for n in chain],
                [body_pos[n][yi] for n in chain],
                "-o", color=color, lw=1.5, ms=3)


def setup_3d(ax):
    ax.set_xlabel("X  (palm depth mm)",  fontsize=7, labelpad=2)
    ax.set_ylabel("Y  (lateral mm)",     fontsize=7, labelpad=2)
    ax.set_zlabel("Z  (height mm)",      fontsize=7, labelpad=2)
    ax.set_xlim(-10, 100); ax.set_ylim(50, 160); ax.set_zlim(40, 180)
    ax.view_init(elev=20, azim=-60)
    ax.tick_params(colors="white", labelsize=6)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#444455")
        axis.label.set_color("white")


# ══════════════════════════════════════════════════════════════════════════════
# Main visualisation
# ══════════════════════════════════════════════════════════════════════════════

def run(session_path: Path, save_png: Path | None = None, animate: bool = True,
        _preloaded: dict | None = None):
    """
    Full 8-panel result view.
    Pass _preloaded dict to skip re-loading data & models (used by __main__).
    """
    if _preloaded:
        hold        = _preloaded["hold"]
        gt_available = _preloaded["gt_available"]
        pred_xyz    = _preloaded["pred_xyz"]
        pred_ori    = _preloaded["pred_ori"]  # noqa: F841 (unused in 8-panel view)
        tilt_lbl    = _preloaded["tilt_lbl"]
        tilt_proba  = _preloaded["tilt_proba"]
        slip_prob   = _preloaded["slip_prob"]
    else:
        hold, gt_available = load_session(session_path)
        (pos_model,  pos_scaler,  pos_cols,
         tilt_model, tilt_scaler, tilt_cols,
         slip_model, slip_scaler, slip_cols) = load_models()
        pred_xyz, pred_ori   = predict_position(hold, pos_model, pos_scaler, pos_cols)
        tilt_lbl, tilt_proba = predict_tilt(hold, tilt_model, tilt_scaler, tilt_cols)
        slip_prob            = predict_slip(hold, slip_model, slip_scaler, slip_cols)

    tilt_str  = "TILTED" if tilt_lbl else "upright"
    tilt_conf = tilt_proba[tilt_lbl] * 100
    slip_pct  = (slip_prob > 0.5).mean() * 100
    mean_pos  = pred_xyz.mean(axis=0)

    if gt_available:
        gt_xyz = hold[["pos_x", "pos_y", "pos_z"]].values * 1000.0
        errs   = np.linalg.norm(pred_xyz - gt_xyz, axis=1)

    # Hand skeleton at mean joint angles
    mean_act = hold[[f"angle_act_{d}" for d in DOF_NAMES]].mean()
    joint_q  = {DOF_TO_JOINT[d]: (1.0 - mean_act[f"angle_act_{d}"] / 1000.0) * DOF_QMAX[d]
                for d in DOF_NAMES}
    body_pos = finger_positions_mm(joint_q)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.38, hspace=0.42,
                           height_ratios=[2.2, 1])

    ax3d    = fig.add_subplot(gs[0, :2], projection="3d")
    ax_info = fig.add_subplot(gs[0, 2])
    ax_slip = fig.add_subplot(gs[0, 3])
    ax_front, ax_side, ax_top = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax_err  = fig.add_subplot(gs[1, 3])

    ax3d.set_facecolor(BG)
    for ax in [ax_info, ax_slip, ax_front, ax_side, ax_top, ax_err]:
        _style_ax(ax)
    ax_info.axis("off")

    fig.suptitle(f"Grasp Analysis  —  {session_path.name}",
                 color="white", fontsize=11, y=0.99)

    n = len(pred_xyz)

    def draw_frame(fi):
        ax3d.cla(); ax_info.cla(); ax_slip.cla()
        ax_front.cla(); ax_side.cla(); ax_top.cla(); ax_err.cla()
        for ax in [ax_info, ax_slip, ax_front, ax_side, ax_top, ax_err]:
            _style_ax(ax)
        ax_info.axis("off")

        # ── 3D ────────────────────────────────────────────────────────────────
        setup_3d(ax3d)
        draw_hand_3d(ax3d, body_pos)

        pred_mm = pred_xyz[fi]
        ax3d.scatter(*pred_mm, s=200, c=PRED_C, marker="o", label="predicted",
                     edgecolors="white", linewidths=1.5, zorder=5, alpha=0.7)

        if gt_available:
            gt_mm = gt_xyz[fi]
            ax3d.scatter(*gt_mm, s=200, c=GT_C, marker="^", label="ground truth",
                         edgecolors="white", linewidths=1.5, zorder=4, alpha=0.7)
            ax3d.plot([pred_mm[0], gt_mm[0]], [pred_mm[1], gt_mm[1]], [pred_mm[2], gt_mm[2]],
                      "--", color="white", lw=0.7, alpha=0.4)

        ax3d.legend(fontsize=7, loc="upper right",
                    facecolor="#2A2A3E", labelcolor="white")

        # ── Info panel ────────────────────────────────────────────────────────
        tilt_color = TILTED_C if tilt_lbl else UPRIGHT_C
        lines = [
            (f"Frame  {fi+1} / {n}",  "#AAAACC"),
            ("",                       "#AAAACC"),
            ("── Position ──────────", "#666688"),
            (f"  X = {pred_mm[0]:+6.1f} mm  (depth)",  "#DDDDFF"),
            (f"  Y = {pred_mm[1]:+6.1f} mm  (lateral)", "#DDDDFF"),
            (f"  Z = {pred_mm[2]:+6.1f} mm  (height)",  "#DDDDFF"),
            ("",                       "#AAAACC"),
            ("── Tilt ───────────────", "#666688"),
            (f"  {tilt_str}  ({tilt_conf:.0f}% conf)", tilt_color),
            ("",                       "#AAAACC"),
            ("── Slip ───────────────", "#666688"),
            (f"  {'SLIP' if slip_prob[fi]>0.5 else 'stable'}  "
             f"(p={slip_prob[fi]:.2f})",
             SLIP_C if slip_prob[fi] > 0.5 else "#4CAF50"),
        ]
        if gt_available:
            err = float(np.linalg.norm(pred_mm - gt_xyz[fi]))
            lines += [
                ("",               "#AAAACC"),
                ("── GT error ───", "#666688"),
                (f"  {err:.1f} mm",  GT_C),
            ]
        for i, (txt, col) in enumerate(lines):
            ax_info.text(0.05, 0.97 - i * 0.072, txt,
                         transform=ax_info.transAxes,
                         color=col, fontsize=8, va="top",
                         fontfamily="monospace")

        # ── Slip timeline ─────────────────────────────────────────────────────
        ax_slip.plot(slip_prob, color=SLIP_C, lw=1, alpha=0.7, label="slip prob")
        ax_slip.axhline(0.5, color="white", lw=0.8, ls="--", alpha=0.4)
        ax_slip.axvline(fi, color="white", lw=1.2, alpha=0.6)
        ax_slip.set_xlim(0, n - 1); ax_slip.set_ylim(0, 1)
        ax_slip.set_xlabel("frame", fontsize=7, color="white")
        ax_slip.set_ylabel("slip probability", fontsize=7, color="white")
        ax_slip.set_title("slip detector", fontsize=8, color="white", pad=3)
        ax_slip.legend(fontsize=6, facecolor="#2A2A3E", labelcolor="white")

        # ── 2D projections ────────────────────────────────────────────────────
        proj_cfg = [
            (ax_front, 0, 2, "front",  "X (depth)",  "Z (height)"),
            (ax_side,  1, 2, "side",   "Y (lateral)", "Z (height)"),
            (ax_top,   0, 1, "top",    "X (depth)",   "Y (lateral)"),
        ]
        for ax, xi, yi, title, xl, yl in proj_cfg:
            ax.set_title(title, fontsize=8, pad=3)
            ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
            ax.set_aspect("equal")
            draw_hand_2d(ax, body_pos, xi, yi)
            ax.scatter(pred_mm[xi], pred_mm[yi], s=120, c=PRED_C,
                       zorder=5, edgecolors="white", linewidths=1, label="predicted")
            if gt_available:
                ax.scatter(gt_xyz[fi][xi], gt_xyz[fi][yi], s=120, c=GT_C,
                           marker="^", zorder=4, edgecolors="white",
                           linewidths=1, label="ground truth")
            ax.legend(fontsize=6, loc="upper right",
                      facecolor="#2A2A3E", labelcolor="white",
                      framealpha=0.6)

        # ── Error timeline (only if GT available) ─────────────────────────────
        if gt_available:
            ax_err.plot(errs, color=PALM_C, lw=1, alpha=0.6)
            ax_err.axvline(fi, color=PRED_C, lw=1.5)
            ax_err.axhline(errs.mean(), color=GT_C, lw=1, ls="--",
                           label=f"mean {errs.mean():.1f} mm")
            ax_err.set_xlim(0, n - 1); ax_err.set_ylim(bottom=0)
            ax_err.set_xlabel("frame", fontsize=7); ax_err.set_ylabel("error (mm)", fontsize=7)
            ax_err.set_title("position error", fontsize=8, pad=3)
            ax_err.legend(fontsize=6, facecolor="#2A2A3E", labelcolor="white")
        else:
            ax_err.plot(pred_xyz[:, 0], color=PRED_C,   lw=1, label="X")
            ax_err.plot(pred_xyz[:, 1], color=UPRIGHT_C, lw=1, label="Y")
            ax_err.plot(pred_xyz[:, 2], color=SLIP_C,   lw=1, label="Z")
            ax_err.axvline(fi, color="white", lw=1.2, alpha=0.5)
            ax_err.set_xlim(0, n - 1)
            ax_err.set_xlabel("frame", fontsize=7); ax_err.set_ylabel("mm", fontsize=7)
            ax_err.set_title("predicted XYZ over time", fontsize=8, pad=3)
            ax_err.legend(fontsize=6, facecolor="#2A2A3E", labelcolor="white")

        fig.canvas.draw_idle()

    # ── Save PNG ───────────────────────────────────────────────────────────────
    if save_png:
        draw_frame(n // 2)
        plt.savefig(save_png, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"\nSaved  → {save_png}")
        return

    # ── Interactive animation ──────────────────────────────────────────────────
    state = {"fi": 0, "running": animate}

    def on_key(event):
        if event.key == " ":
            state["running"] = not state["running"]
        elif event.key == "right":
            state["fi"] = min(n - 1, state["fi"] + 1)
            draw_frame(state["fi"])
        elif event.key == "left":
            state["fi"] = max(0, state["fi"] - 1)
            draw_frame(state["fi"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame(0)

    if animate:
        import matplotlib.animation as anim
        def _step(_):
            if state["running"]:
                state["fi"] = (state["fi"] + 1) % n
                draw_frame(state["fi"])
        anim.FuncAnimation(fig, _step, interval=200, cache_frame_data=False)

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Compare view: simple top-down hand + prediction vs ground truth
# ══════════════════════════════════════════════════════════════════════════════

def save_compare(session_path: Path, out_png: Path,
                 pred_xyz_mean: np.ndarray, pred_ori_mean: np.ndarray,
                 tilt_lbl: int, tilt_conf: float,
                 hold: pd.DataFrame, gt_available: bool,
                 body_pos: dict):
    """
    Front-view (Y=lateral, Z=height): hand skeleton in background,
    prediction + GT arrows in foreground.

    Red arrow  = predicted object centre + tilt direction
    Green arrow = GT position + actual tilt direction (sim only)
    """
    import json
    import matplotlib.patches as mpatches

    # Projection: Y (lateral) on x-axis, Z (height) on y-axis  →  front view
    XI, YI = 1, 2   # body_pos indices

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    # ── Real hand skeleton, front view (Y × Z) ────────────────────────────────
    palm_bodies = [CHAINS[i][1] for i in range(4)] + ["right_hand_root"]
    palm_pts    = [body_pos[b] for b in palm_bodies]
    ax.fill([p[XI] for p in palm_pts] + [palm_pts[0][XI]],
            [p[YI] for p in palm_pts] + [palm_pts[0][YI]],
            color="#2E3D45", alpha=0.8, zorder=1)

    for chain in CHAINS:
        color = THUMB_C if "thumb" in chain[-1] else FINGER_C
        ax.plot([body_pos[n][XI] for n in chain],
                [body_pos[n][YI] for n in chain],
                "-o", color=color, lw=3, ms=6, alpha=0.55, zorder=2)

    ax.plot([p[XI] for p in palm_pts] + [palm_pts[0][XI]],
            [p[YI] for p in palm_pts] + [palm_pts[0][YI]],
            color=PALM_C, lw=2, alpha=0.55, zorder=3)

    # Object positions in Y, Z
    py_pred, pz_pred = pred_xyz_mean[1], pred_xyz_mean[2]

    # ── Predicted tilt arrow ──────────────────────────────────────────────────
    # sin/cos: [sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw]
    pred_pitch = np.rad2deg(np.arctan2(pred_ori_mean[2], pred_ori_mean[3]))
    pred_roll  = np.rad2deg(np.arctan2(pred_ori_mean[0], pred_ori_mean[1]))
    pred_mag   = np.sqrt(pred_pitch**2 + pred_roll**2)
    if pred_mag > 1.0:
        # pitch → tilt in YZ plane (dy), roll → tilt visible in Z
        d_py = pred_pitch / pred_mag * 22
        d_pz = pred_roll  / pred_mag * 22
        ax.annotate("", xy=(py_pred + d_py, pz_pred + d_pz),
                        xytext=(py_pred - d_py, pz_pred - d_pz),
                    arrowprops=dict(arrowstyle="->", color=PRED_C, lw=2.5), zorder=8)

    ax.scatter(py_pred, pz_pred, s=280, c=PRED_C, marker="o",
               zorder=9, edgecolors="white", linewidths=2)
    tilt_str  = "TILTED" if tilt_lbl else "upright"
    tilt_color = TILTED_C if tilt_lbl else UPRIGHT_C
    ax.text(py_pred + 3, pz_pred + 3,
            f"({py_pred:+.0f}, {pz_pred:+.0f}) mm\n{tilt_str} {tilt_conf:.0f}%",
            color=tilt_color, fontsize=8, fontfamily="monospace", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E1E2E", alpha=0.75))

    pred_handle = mpatches.Patch(color=PRED_C,
                                  label=f"Pred Y={py_pred:+.0f} Z={pz_pred:+.0f} mm  {tilt_str}")

    # ── Ground truth (sim only) ───────────────────────────────────────────────
    gt_handle = None
    if gt_available:
        gy = hold["pos_y"].values.mean() * 1000.0
        gz = hold["pos_z"].values.mean() * 1000.0

        meta_path = session_path / "meta.json"
        if meta_path.exists():
            ep  = json.loads(meta_path.read_text())["episode_info"]
            tx  = np.rad2deg(ep["tilt_x_rad"])
            tz  = np.rad2deg(ep["tilt_z_rad"])
            mag = np.sqrt(tx**2 + tz**2)
            if mag > 0.5:
                # tz → rotation around Z → displacement in Y; tx → rot around X → displacement in Z
                dy_gt = tz / mag * 22
                dz_gt = tx / mag * 22
                ax.annotate("", xy=(gy + dy_gt, gz + dz_gt),
                                xytext=(gy - dy_gt, gz - dz_gt),
                            arrowprops=dict(arrowstyle="->", color=GT_C, lw=2.5), zorder=6)
            ax.scatter(gy, gz, s=220, c=GT_C, marker="^",
                       zorder=7, edgecolors="white", linewidths=1.5)
            gt_handle = mpatches.Patch(
                color=GT_C,
                label=f"GT  Y={gy:+.0f} Z={gz:+.0f} mm  tilt {mag:.1f}°",
            )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xlabel("Y  (lateral mm)",  fontsize=9)
    ax.set_ylabel("Z  (height mm)",   fontsize=9)
    ax.set_title(f"{session_path.name}\nFront view  (lateral × height)", fontsize=9,
                 color="white", pad=8)

    handles = [h for h in [pred_handle, gt_handle] if h is not None]
    ax.legend(handles=handles, fontsize=8, loc="lower right",
              facecolor="#2A2A3E", labelcolor="white", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved  → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1:
        session = Path(sys.argv[1])
    else:
        sessions = sorted(SIM_DIR.iterdir())
        session  = sessions[10]
        print(f"No session given — using default: {session.name}")

    OUT_DIR.mkdir(exist_ok=True)

    # ── Load once, use for both outputs ───────────────────────────────────────
    hold, gt_available = load_session(session)
    (pos_model,  pos_scaler,  pos_cols,
     tilt_model, tilt_scaler, tilt_cols,
     slip_model, slip_scaler, slip_cols) = load_models()

    pred_xyz, pred_ori     = predict_position(hold, pos_model, pos_scaler, pos_cols)
    tilt_lbl, tilt_proba   = predict_tilt(hold, tilt_model, tilt_scaler, tilt_cols)
    slip_prob              = predict_slip(hold, slip_model, slip_scaler, slip_cols)

    mean_pos  = pred_xyz.mean(axis=0)
    mean_ori  = pred_ori.mean(axis=0)   # mean sin_roll,cos_roll,sin_pitch,cos_pitch,...
    tilt_conf = tilt_proba[tilt_lbl] * 100

    print(f"\nSession : {session.name}")
    print(f"Frames  : {len(hold)}  |  GT: {'yes' if gt_available else 'no'}")
    print(f"Position: X={mean_pos[0]:+.1f}  Y={mean_pos[1]:+.1f}  Z={mean_pos[2]:+.1f} mm")
    print(f"Tilt    : {'TILTED' if tilt_lbl else 'upright'}  ({tilt_conf:.0f}% conf)")
    print(f"Slip    : {(slip_prob > 0.5).mean()*100:.0f}% of frames")
    if gt_available:
        gt_xyz = hold[["pos_x", "pos_y", "pos_z"]].values * 1000.0
        errs   = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
        print(f"GT error: mean={errs.mean():.1f} mm  median={np.median(errs):.1f} mm")

    # Hand skeleton at mean joint angles (shared between both outputs)
    mean_act = hold[[f"angle_act_{d}" for d in DOF_NAMES]].mean()
    joint_q  = {DOF_TO_JOINT[d]: (1.0 - mean_act[f"angle_act_{d}"] / 1000.0) * DOF_QMAX[d]
                for d in DOF_NAMES}
    body_pos = finger_positions_mm(joint_q)

    preloaded = dict(hold=hold, gt_available=gt_available,
                     pred_xyz=pred_xyz, pred_ori=pred_ori,
                     tilt_lbl=tilt_lbl, tilt_proba=tilt_proba, slip_prob=slip_prob)

    # Full 8-panel result
    run(session, save_png=OUT_DIR / f"{session.name}_result.png", _preloaded=preloaded)

    # Simple top-down compare view
    save_compare(session,
                 out_png       = OUT_DIR / f"{session.name}_compare.png",
                 pred_xyz_mean = mean_pos,
                 pred_ori_mean = mean_ori,
                 tilt_lbl      = tilt_lbl,
                 tilt_conf     = tilt_conf,
                 hold          = hold,
                 gt_available  = gt_available,
                 body_pos      = body_pos)
    # run(session, animate=True)   # uncomment for interactive window
