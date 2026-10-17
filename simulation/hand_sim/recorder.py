"""Writes simulation episodes in the same file layout as models/dataset, plus a ground-truth
object_pose.csv (the real dataset has no pose labels).

Matches the newer, 69-column actuators.csv schema used by 16 of the 18 real sessions (the first
two recordings used an older, shorter schema missing pos_set/force_set/speed_set/pos_act/temp -
see AGENTS.md). tactile_raw.csv / tactile_summary.csv / meta.json schemas are stable across all
real sessions and are matched exactly, including the summary rounding rules.

Usage (from simulation/):
    ../venv/bin/python -m hand_sim.recorder --episodes 20 --out ../models/dataset_sim
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config as C
from .episode import HandoverEpisode
from .policy import ScriptedGraspPolicy
from .world import World

# pos_act is a derived/approximate register (hand_manual.pdf 2.6.10 marks it "not recommended").
# Fit against models/dataset/*/actuators.csv: pos_act ~= POS_FIT[0]*angle_act + POS_FIT[1], +-108.
POS_FIT = (-1.824, 1903.1)
TEMP_BASE_C, TEMP_PER_CURRENT = 26.0, 0.01  # current (mA) -> actuator temperature, roughly tuned


def _actuator_row(frame, t_unix, t_rel, hand_read, rng):
    r = hand_read
    pos_act = np.clip(np.round(POS_FIT[0] * r["angle_act"] + POS_FIT[1] + rng.normal(0, 20, 6)), 0, 2000).astype(int)
    temp = np.clip(np.round(TEMP_BASE_C + TEMP_PER_CURRENT * r["current"]), 0, 100).astype(int)
    row = {"frame": frame, "t_unix": f"{t_unix:.4f}", "t_rel": f"{t_rel:.4f}"}
    row.update({f"pos_set_{n}": "-1" for n in C.DOF_NAMES})  # POS_SET isn't driven by this controller
    for key in ("angle_set", "force_set", "speed_set"):
        for n, v in zip(C.DOF_NAMES, r[key]):
            row[f"{key}_{n}"] = int(v)
    for n, v in zip(C.DOF_NAMES, pos_act):
        row[f"pos_act_{n}"] = int(v)
    for key in ("angle_act", "force_act", "current", "error", "status"):
        for n, v in zip(C.DOF_NAMES, r[key]):
            row[f"{key}_{n}"] = int(v)
    for n, v in zip(C.DOF_NAMES, temp):
        row[f"temp_{n}"] = int(v)
    return row


ACTUATOR_FIELDS = (
    ["frame", "t_unix", "t_rel"]
    + [f"{k}_{n}" for k in ("pos_set", "angle_set", "force_set", "speed_set", "pos_act", "angle_act",
                             "force_act", "current", "error", "status", "temp") for n in C.DOF_NAMES]
)


def _tactile_fields(prefix_rowcol: bool):
    fields = ["frame", "t_unix", "t_rel"]
    for name, _, rows, cols, *_ in C.SENSOR_GROUPS:
        if prefix_rowcol:
            fields += [f"{name}_r{r}_c{c}" for r in range(rows) for c in range(cols)]
        else:
            fields += [f"{name}_{k}" for k in ("sum", "max", "contacts", "cy", "cx", "force")]
    return fields


RAW_FIELDS = _tactile_fields(True)
SUMMARY_FIELDS = _tactile_fields(False)
POSE_FIELDS = ["frame", "t_unix", "t_rel", "phase", "pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y",
               "quat_z", "ori_x", "ori_y", "ori_z", "lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y",
               "ang_vel_z", "in_hand"]


class DatasetRecorder:
    """Accumulates CONTROL_HZ-rate rows for one session; write() dumps them like a real logger session."""

    def __init__(self, world: World, rng: np.random.Generator):
        self.world, self.rng = world, rng
        self.started_at = datetime.now()
        self.frame = 0
        self.rows = {"actuators": [], "tactile_raw": [], "tactile_summary": [], "object_pose": []}

    def record(self, episode: HandoverEpisode, obs):
        w = self.world
        t_unix = time.time()
        t_rel = episode.t
        self.rows["actuators"].append(_actuator_row(self.frame, t_unix, t_rel, obs.registers, self.rng))

        raw = w.tactile.step(w.d)
        raw_row = {"frame": self.frame, "t_unix": f"{t_unix:.4f}", "t_rel": f"{t_rel:.4f}"}
        for name, _, rows, cols, *_ in C.SENSOR_GROUPS:
            grid = np.round(raw[name]).astype(int)
            for r in range(rows):
                for c in range(cols):
                    raw_row[f"{name}_r{r}_c{c}"] = int(grid[r, c])
        self.rows["tactile_raw"].append(raw_row)

        summary = w.tactile.summary(raw)
        summ_row = {"frame": self.frame, "t_unix": f"{t_unix:.4f}", "t_rel": f"{t_rel:.4f}"}
        for name, *_ in C.SENSOR_GROUPS:
            s = summary[name]
            summ_row.update({f"{name}_sum": s["sum"], f"{name}_max": s["max"], f"{name}_contacts": s["contacts"],
                              f"{name}_cy": s["cy"], f"{name}_cx": s["cx"], f"{name}_force": s["force"]})
        self.rows["tactile_summary"].append(summ_row)

        pos, quat = w.object_pose(episode.obj_id)
        vel = w.d.qvel[w._ids[episode.obj_id]["vadr"] : w._ids[episode.obj_id]["vadr"] + 6]
        ori = np.zeros(3)
        import mujoco
        mujoco.mju_rotVecQuat(ori, np.array([0.0, 1.0, 0.0]), np.asarray(quat, dtype=float))
        self.rows["object_pose"].append({
            "frame": self.frame, "t_unix": f"{t_unix:.4f}", "t_rel": f"{t_rel:.4f}", "phase": obs.phase,
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "quat_w": quat[0], "quat_x": quat[1], "quat_y": quat[2], "quat_z": quat[3],
            "ori_x": ori[0], "ori_y": ori[1], "ori_z": ori[2],
            "lin_vel_x": vel[0], "lin_vel_y": vel[1], "lin_vel_z": vel[2],
            "ang_vel_x": vel[3], "ang_vel_y": vel[4], "ang_vel_z": vel[5],
            "in_hand": int(obs.phase != "approach"),
        })
        self.frame += 1

    def write(self, out_dir: Path, episode_info: dict):
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, fields in (("actuators", ACTUATOR_FIELDS), ("tactile_raw", RAW_FIELDS),
                            ("tactile_summary", SUMMARY_FIELDS), ("object_pose", POSE_FIELDS)):
            with open(out_dir / f"{key}.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.rows[key])

        offsets = {g["name"]: np.round(g["offset"], 1).tolist() for g in self.world.tactile.groups}
        meta = {
            "start": self.started_at.isoformat(),
            "ip": "sim",
            "dof": C.DOF_NAMES,
            "sample_rate_target_hz": C.CONTROL_HZ,
            "contact_threshold": C.TACTILE_CONTACT_THRESHOLD,
            "calibrated": True,
            "sensor_groups": [
                {"name": name.replace("_", " ").title(), "addr": addr, "rows": rows, "cols": cols,
                 "finger": name.split("_")[0].title()}
                for name, addr, rows, cols, *_ in C.SENSOR_GROUPS
            ],
            "offsets": {name.replace("_", " ").title(): v for name, v in offsets.items()},
            "source": "sim",
            "object": episode_info["object"],
            "episode_info": {k: v for k, v in episode_info.items() if k not in ("target_pos", "target_quat")},
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object", choices=list(C.OBJECTS), default=None)
    ap.add_argument("--out", default=str(C.SIM_DIR.parent / "models" / "dataset_sim"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World(seed=args.seed)
    out_root = Path(args.out)
    for i in range(args.episodes):
        rec = DatasetRecorder(world, rng)
        ep = HandoverEpisode(world, ScriptedGraspPolicy(), rng, obj_id=args.object, on_frame=rec.record)
        result = ep.run()
        session_dir = out_root / f"{rec.started_at.strftime('%Y%m%d_%H%M%S')}_{i:03d}"
        rec.write(session_dir, {**ep.info, **result})
        print(f"#{i:3d} {ep.obj_id:11s} {'OK  ' if result['success'] else 'FAIL'} "
              f"{result['failure'] or '':8s} {rec.frame:4d} frames -> {session_dir}")


if __name__ == "__main__":
    main()
