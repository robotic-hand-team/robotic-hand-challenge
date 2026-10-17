"""Tactile pose prototype: logged grasp -> contacts -> STL pose fit -> 3D image.

Usage:
    python run_pipeline.py                                  # labelled sessions, object from labels
    python run_pipeline.py --session 20261016_204637        # one session
    python run_pipeline.py --object standin --session 20261016_204637   # force an object
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation

import config as C
from contacts import extract_contacts
from data_io import hold_window, joint_raw, load_session
from hand_model import HandModel
from pose_fit import ObjectModel, fit_pose
from render import save_html, save_png

CONTACT_LINKS = ["hand_base_link", "thumb_distal", "thumb_intermediate", "thumb_proximal",
                 "index_proximal", "index_intermediate", "middle_proximal",
                 "middle_intermediate", "ring_proximal", "ring_intermediate",
                 "pinky_proximal", "pinky_intermediate"]


def hand_surface_points(meshes, n_per_link=40):
    """Points on the hand surface: the object must not pass through them."""
    pts = []
    for link in CONTACT_LINKS:
        if link in meshes:
            n = n_per_link * (6 if link == "hand_base_link" else 1)
            pts.append(trimesh.sample.sample_surface_even(meshes[link], n, seed=1)[0])
    return np.vstack(pts)


def slip_indicator(session, frames):
    """Crude slip check over the hold window: contact drift and pressure loss."""
    summ = pd.read_csv(C.LOG_DIR / session.name / "tactile_summary.csv")
    w = summ.iloc[frames]
    first, last = w.iloc[: max(len(w) // 5, 1)], w.iloc[-max(len(w) // 5, 1):]
    drift = 0.0
    for g in session.groups:
        k = g["name"].lower().replace(" ", "_")
        if first[f"{k}_sum"].mean() < C.MIN_REGION_SUM:
            continue
        d = np.hypot(first[f"{k}_cy"].mean() - last[f"{k}_cy"].mean(),
                     first[f"{k}_cx"].mean() - last[f"{k}_cx"].mean())
        if np.isfinite(d):
            drift = max(drift, d)
    sums = [c for c in summ.columns if c.endswith("_sum")]
    p0, p1 = first[sums].sum(axis=1).mean(), last[sums].sum(axis=1).mean()
    loss = 100 * (1 - p1 / p0) if p0 > 0 else 0.0
    return {"max_centroid_drift_cells": round(float(drift), 2),
            "pressure_change_pct": round(float(-loss), 1),
            "slip_suspected": bool(drift > 1.0 or loss > 25)}


def run_session(name, obj, hand, outdir, axis_hint=None):
    s = load_session(C.LOG_DIR / name)
    frames = hold_window(s)
    row = {"session": name, "object": obj.name}
    if len(frames) == 0:
        row["status"] = "no grasp found"
        return row
    hand.set_raw(joint_raw(s, frames))
    contacts, taxels = extract_contacts(s, frames, hand)
    meshes = hand.posed_meshes()
    fit = fit_pose(obj, contacts, hand_surface_points(meshes), axis_hint=axis_hint,
                   check_points=hand_surface_points(meshes, 600))

    obj_mesh = None
    if fit is not None:
        obj_mesh = obj.mesh.copy()
        obj_mesh.apply_transform(fit.T)
    info = {"session": name, "object": obj.name, "fit": fit, "hint": axis_hint}
    save_png(outdir / f"{name}.png", meshes, obj_mesh, contacts, taxels, info)
    save_html(outdir / f"{name}.html", meshes, obj_mesh, contacts, taxels, info)

    row.update(n_hold_frames=len(frames), contacts=";".join(c.region for c in contacts),
               **slip_indicator(s, frames))
    if fit is None:
        row["status"] = "too few contacts"
        return row
    T_cad = fit.T @ obj.cad_to_obj                # pose of the original CAD frame
    euler = Rotation.from_matrix(fit.T[:3, :3]).as_euler("xyz", degrees=True)
    row.update(
        status=("collides with hand" if fit.collides else "ok" if fit.ok
                else "ambiguous" if fit.ambiguous else "poor fit"),
        pos_x_mm=round(fit.T[0, 3] * 1000, 1), pos_y_mm=round(fit.T[1, 3] * 1000, 1),
        pos_z_mm=round(fit.T[2, 3] * 1000, 1),
        axis_x=round(fit.axis[0], 3), axis_y=round(fit.axis[1], 3), axis_z=round(fit.axis[2], 3),
        roll_deg=round(euler[0], 1), pitch_deg=round(euler[1], 1), yaw_deg=round(euler[2], 1),
        fit_rms_mm=round(fit.rms_mm, 2), normal_err_deg=round(fit.normal_err_deg, 1),
        penetration_mm=round(fit.max_penetration_mm, 2), alternatives=fit.alternatives,
        T_hand_obj=json.dumps(np.round(fit.T, 5).tolist()),
        T_hand_cad=json.dumps(np.round(T_cad, 5).tolist()),
    )
    export_for_sim(outdir / f"{name}.sim.json", row, T_cad, hand)
    return row


# Pose object names -> simulation/hand_sim object ids
SIM_OBJECT_IDS = {"vegabar29": "vegabar_29", "vegabar38": "vegabar_38"}


def export_for_sim(path, row, T_cad, hand):
    """Write what simulation/hand_sim needs to replay this grasp (load with `pose <file>`)."""
    joints = {dof: float(hand.urdf.cfg[hand.urdf.actuated_joint_names.index(j)])
              for dof, j in C.DOF_TO_JOINT.items()}
    out = {
        "session": row["session"],
        "object": SIM_OBJECT_IDS.get(row["object"], row["object"]),
        "status": row["status"],
        "fit_rms_mm": row["fit_rms_mm"],
        "frame": "hand_base_link",           # URDF frame, metres
        "T_hand_cad": np.round(T_cad, 6).tolist(),   # STL vertices (mm * 0.001) -> hand frame
        "joint_rad": joints,                  # actuated joints, keyed by DOF name
        "hand": C.HAND_SIDE,                  # left: the simulation (right hand only) shows it mirrored
    }
    path.write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="auto", choices=["auto"] + list(C.OBJECTS),
                    help="auto = use SESSION_LABELS in config.py (classifier stand-in)")
    ap.add_argument("--session", default="all")
    ap.add_argument("--hand", default=C.HAND_SIDE, choices=["right", "left"], help="which hand recorded the logs")
    ap.add_argument("--log-dir", help="folder with session subfolders (default: config.LOG_DIR)")
    ap.add_argument("--axis-hint", default="config",
                    choices=["config", "none", "along_palm", "across_palm", "out_of_palm"],
                    help="rough grasp type; 'config' uses SESSION_AXIS_HINTS / AXIS_HINT")
    a = ap.parse_args()
    C.set_hand(a.hand)
    if a.log_dir:
        C.LOG_DIR = Path(a.log_dir)

    if not C.HAND_URDF.exists() or not (C.HAND_URDF.parent / "meshes").exists():
        raise SystemExit(f"Hand model incomplete: need {C.HAND_URDF} and a meshes/ folder "
                         "next to it (download the whole inspire_hand folder).")
    cache = {}

    def get_object(key):
        if key not in cache:
            stl = C.OBJECTS[key]
            if not stl.exists():
                raise SystemExit(f"STL not found: {stl}\nCopy it there, or use --object standin.")
            cache[key] = ObjectModel(stl, key)
        return cache[key]

    hand = HandModel()
    outdir = C.OUTPUT_DIR / a.object
    outdir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in C.LOG_DIR.iterdir() if p.is_dir()) \
        if a.session == "all" else [a.session]
    rows = []
    for n in names:
        if a.object == "auto":
            label = C.SESSION_LABELS.get(n)
            if label is None:
                print(f"{n}: skipped (no label)")
                continue
            obj = get_object(C.LABEL_TO_OBJECT[label])
        else:
            obj = get_object(a.object)
        if a.axis_hint == "config":
            hint = C.SESSION_AXIS_HINTS.get(n, C.AXIS_HINT)
        else:
            hint = None if a.axis_hint == "none" else a.axis_hint
        r = run_session(n, obj, hand, outdir, hint)
        r["axis_hint"] = hint or "none"
        rows.append(r)
        print(f"{n}: {r.get('status')}  rms={r.get('fit_rms_mm', '-')} mm  "
              f"contacts={r.get('contacts', '')}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k)
                  if k in rows[-1] else 99)
    with open(outdir / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"Results in {outdir}")


if __name__ == "__main__":
    main()
