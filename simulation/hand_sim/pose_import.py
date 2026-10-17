"""Replay a tactile pose estimate (pose/run_pipeline.py -> <session>.sim.json) in the simulation.

The pose pipeline works in the URDF `hand_base_link` frame and the original STL frame (mm * 0.001).
The simulation places the hand root (= URDF `base` link) in the world and centres each object mesh on
its bounding box (hand_sim/assets.py). This module converts between the two and teleports the hand
and object to the estimated grasp.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import config as C

HAND_ROOT_BODY = "right_hand_root"
# URDF fixed joint base -> hand_base_link: rpy="-1.57079 0 3.14159" (same in pose/ and simulation/ URDF).
R_BASE_HBL = Rotation.from_euler("xyz", [-1.57079, 0.0, 3.14159]).as_matrix()


def _T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def hand_base_link_in_world(m, d) -> np.ndarray:
    b = m.body(HAND_ROOT_BODY).id
    T_world_base = _T(d.xmat[b].reshape(3, 3), d.xpos[b])
    return T_world_base @ _T(R_BASE_HBL)


def load(path) -> dict:
    est = json.loads(Path(path).read_text())
    if est["object"] not in C.OBJECTS:
        raise KeyError(f"object {est['object']!r} not in simulation (have {list(C.OBJECTS)})")
    return est


def _set_coupled_joints(m, d):
    """Put the mimic joints (joint equalities) exactly on their polynomial so nothing snaps."""
    for e in range(m.neq):
        if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or not d.eq_active[e]:
            continue
        j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
        c = m.eq_data[e][:5]
        q2 = d.qpos[m.jnt_qposadr[j2]]
        d.qpos[m.jnt_qposadr[j1]] = c[0] + c[1] * q2 + c[2] * q2**2 + c[3] * q2**3 + c[4] * q2**4


def apply(world, est: dict) -> dict:
    """Teleport hand and object to the estimated grasp. Object stays held by its carrier weld."""
    m, d = world.m, world.d
    obj_id = est["object"]

    # Hand: joint angles (rad) -> raw ANGLE_SET, then teleport.
    q = np.array([est["joint_rad"][n] for n in C.DOF_NAMES])
    raw = np.clip(np.round(C.ANGLE_MAX * (1.0 - q / world.hand.qmax)), 0, C.ANGLE_MAX).astype(int)
    world.reset(raw)
    _set_coupled_joints(m, d)
    mujoco.mj_forward(m, d)

    # Object: STL frame in hand_base_link -> world -> simulation object frame (bbox-centred).
    meta = json.loads((C.OBJECTS_DIR / obj_id / "object.json").read_text())
    offset = np.array(meta["cad_to_object_offset_m"])
    T_world_cad = hand_base_link_in_world(m, d) @ np.array(est["T_hand_cad"])
    T_world_obj = T_world_cad @ _T(p=-offset)
    pos = T_world_obj[:3, 3]
    quat = Rotation.from_matrix(T_world_obj[:3, :3]).as_quat()  # x y z w
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # MuJoCo w x y z

    world.active = obj_id
    world.place(obj_id, pos, quat)
    world.hand.set_forces([C.OBJECTS[obj_id]["grasp_force_g"]] * 6)
    return {"raw_angles": raw.tolist(), "pos_m": pos, "quat_wxyz": quat}


def report(world, est: dict, placed: dict) -> str:
    """Summary line plus the worst hand/object overlaps at the estimated pose."""
    mujoco.mj_forward(world.m, world.d)
    pen = sorted(world.object_contacts(est["object"]), key=lambda c: c[1])
    lines = [
        f"session {est['session']}: {est['object']}  status={est['status']}  rms={est['fit_rms_mm']} mm",
        f"  ANGLE_SET {placed['raw_angles']}  object pos(mm) {np.round(placed['pos_m'] * 1000, 1)}",
    ]
    if pen:
        worst = {}
        for body, dist in pen:
            name = world.m.body(body).name
            worst[name] = min(worst.get(name, 0.0), dist)
        lines.append("  overlap with hand (mm): " + ", ".join(
            f"{k} {v * 1000:.1f}" for k, v in sorted(worst.items(), key=lambda kv: kv[1])))
    else:
        lines.append("  no overlap with the hand")
    return "\n".join(lines)


# --- Physical replay: approach in the estimated orientation, then grasp ----------------------------

APPROACH_S = 0.8  # sim seconds to move from the clear start pose toward the estimate
CLOSE_S = 1.5  # time for the fingers to close and force-stop on the object
HOLD_S = 0.5  # settle with the carrier still attached
RELEASE_S = 0.3  # carrier weld off, weight handed over to the fingers over this time
TEST_S = 1.0  # then check whether the fingers alone hold the object
BLOCKED_LAG_M = 0.004  # approach stops when the hand holds the object back this far behind the carrier
HELD_MAX_SHIFT_M = 0.01
CLEAR_MARGIN_M = 0.003
MAX_BACKOFF_M = 0.10


def _hand_overlap(world, obj_id, tol=-0.0005):
    """Deepest penetration (m, <=0) of the object into any hand body; 0 if clear."""
    depth = 0.0
    for body, dist in world.object_contacts(obj_id, tol):
        if world.m.body(body).name.startswith("right_"):
            depth = min(depth, dist)
    return depth


def _rot_deg(q1, q2):
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return float(np.degrees((r1.inv() * r2).magnitude()))


class GraspReplay:
    """Open hand -> object in the estimated orientation, clear of the hand -> slide in until the hand
    blocks it -> close the fingers with the object's grasp force -> settle -> report -> let go and
    check whether the fingers alone hold it.

    Call update() once per viewer frame (or per step headless); it returns a message when a phase
    finishes and sets .done at the end.
    """

    def __init__(self, world, est: dict):
        self.world, self.est = world, est
        m, d = world.m, world.d
        obj_id = est["object"]

        # Target from the tactile estimate (same conversion as apply()).
        q = np.array([est["joint_rad"][n] for n in C.DOF_NAMES])
        self.logged_raw = np.clip(np.round(C.ANGLE_MAX * (1.0 - q / world.hand.qmax)), 0, C.ANGLE_MAX).astype(int)
        meta = json.loads((C.OBJECTS_DIR / obj_id / "object.json").read_text())
        offset = np.array(meta["cad_to_object_offset_m"])

        # Open fingers; thumb rotated as in the logged grasp so it comes in from the same side.
        open_raw = [C.ANGLE_MAX] * 5 + [int(self.logged_raw[5])]
        world.reset(open_raw)
        _set_coupled_joints(m, d)
        mujoco.mj_forward(m, d)

        T_world_obj = hand_base_link_in_world(m, d) @ np.array(est["T_hand_cad"]) @ _T(p=-offset)
        self.target_pos = T_world_obj[:3, 3]
        xyzw = Rotation.from_matrix(T_world_obj[:3, :3]).as_quat()
        self.quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        # Back off out of the palm (world +x, with some up/down/sideways options) until clear.
        self.start_pos, self.backoff_mm = self._clear_start(obj_id)
        world.active = obj_id
        world.place(obj_id, self.start_pos, self.quat)
        world.hand.set_forces([C.OBJECTS[obj_id]["grasp_force_g"]] * 6)
        self.phase, self.t0, self.done = "approach", d.time, False

    def _clear_start(self, obj_id):
        dirs = [[1, 0, 0], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, -1, 0], [0, 0, 1]]
        best = None
        for v in dirs:
            v = np.array(v, float) / np.linalg.norm(v)
            for s in np.arange(0.0, MAX_BACKOFF_M, 0.002):
                self.world.place(obj_id, self.target_pos + s * v, self.quat)
                if _hand_overlap(self.world, obj_id) == 0.0:
                    if best is None or s < best[0]:
                        best = (s, v)
                    break
        s, v = best if best else (MAX_BACKOFF_M, np.array([1.0, 0, 0]))
        return self.target_pos + (s + CLEAR_MARGIN_M) * v, s * 1000

    def update(self, world):
        if self.done:
            return None
        d, obj_id = world.d, self.est["object"]
        t = d.time - self.t0
        if self.phase == "approach":
            frac = min(1.0, t / APPROACH_S)
            pos, _ = world.object_pose(obj_id)
            carrier = world.d.mocap_pos[world._ids[obj_id]["mocap"]]
            blocked = np.linalg.norm(carrier - pos) > BLOCKED_LAG_M and _hand_overlap(world, obj_id, 0.0) < 0
            if blocked or frac >= 1.0:
                world.move_carrier(obj_id, pos, self.quat)  # stop where the hand blocks it
                world.hand.set_angles([0, 0, 0, 0, 0, -1])  # close all, keep thumb rotation
                self.phase, self.t0 = "close", d.time
                stop = np.linalg.norm(pos - self.target_pos) * 1000
                why = "blocked by the hand" if blocked else "reached the estimate"
                return f"[pose] {why}, {stop:.1f} mm from the estimate; closing fingers"
            world.move_carrier(obj_id, self.start_pos + frac * (self.target_pos - self.start_pos), self.quat)
        elif self.phase == "close" and t >= CLOSE_S:
            self.phase, self.t0 = "hold", d.time
        elif self.phase == "hold" and t >= HOLD_S:
            self.grasped_pos, self.grasped_quat = world.object_pose(obj_id)
            self.grasp_text = self.report()
            world.release(keep_support=True)
            self.phase, self.t0 = "release", d.time
            return self.grasp_text + "\n[pose] letting go of the object: do the fingers hold it?"
        elif self.phase == "release":
            world.set_support(max(0.0, 1.0 - t / RELEASE_S))
            if t >= RELEASE_S:
                self.phase, self.t0 = "test", d.time
        elif self.phase == "test" and t >= TEST_S:
            self.done = True
            pos, quat = world.object_pose(obj_id)
            shift = np.linalg.norm(pos - self.grasped_pos)
            verdict = "HELD" if shift < HELD_MAX_SHIFT_M else "DROPPED / SLIPPED"
            return (f"[pose] {verdict}: object moved {shift * 1000:.1f} mm and turned "
                    f"{_rot_deg(quat, self.grasped_quat):.1f} deg after the hand took its weight")
        return None

    def report(self) -> str:
        w, obj_id = self.world, self.est["object"]
        pos, quat = w.object_pose(obj_id)
        r = w.hand.read()
        touching = [n for n, s in zip(C.DOF_NAMES, r["status"]) if s == C.STATUS_FORCE_STOP]
        return "\n".join([
            f"[pose] grasp done  session {self.est['session']}  ({self.est['object']}, estimate {self.est['status']})",
            f"  start was backed off {self.backoff_mm:.0f} mm out of the hand",
            f"  object vs estimate: shift {np.linalg.norm(pos - self.target_pos) * 1000:.1f} mm, "
            f"rotation {_rot_deg(quat, self.quat):.1f} deg",
            f"  ANGLE_ACT sim    {r['angle_act'].tolist()}",
            f"  ANGLE_ACT logged {self.logged_raw.tolist()}   (order {C.DOF_NAMES})",
            f"  fingers touching: {', '.join(touching) or 'none'}",
            f"  max overlap with hand: {-_hand_overlap(w, obj_id, 0.0) * 1000:.1f} mm",
        ])
