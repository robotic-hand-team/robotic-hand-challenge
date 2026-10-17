"""Simulation state shared by the viewer and headless runners: model, hand, object carriers."""

import mujoco
import numpy as np

from . import config as C
from .hand import HandController
from .scene import PARK_POS, build_model
from .tactile import TactileModel


class World:
    def __init__(self, seed: int | None = None):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.hand = HandController(self.m, self.d, self.rng)
        self.tactile = TactileModel(self.m, self.rng)
        self.active = next(iter(C.OBJECTS))
        self._ids = {}
        for obj_id in C.OBJECTS:
            self._ids[obj_id] = {
                "body": self.m.body(obj_id).id,
                "mocap": self.m.body_mocapid[self.m.body(f"{obj_id}_carrier").id],
                "qadr": self.m.jnt_qposadr[self.m.joint(f"{obj_id}_free").id],
                "vadr": self.m.jnt_dofadr[self.m.joint(f"{obj_id}_free").id],
                "eq": self.m.equality(f"{obj_id}_weld").id,
            }

    def reset(self, hand_angles=None):
        """Fresh physics state: hand teleported to a posture, all objects parked and welded."""
        mujoco.mj_resetData(self.m, self.d)
        self.hand.reset(hand_angles)
        for obj_id in C.OBJECTS:
            self.place(obj_id, PARK_POS[obj_id], [1, 0, 0, 0])

    def object_contacts(self, obj_id: str, min_dist: float = 0.0) -> list[tuple[int, float]]:
        """(other body id, penetration distance) for contacts of obj_id deeper than min_dist (<0)."""
        b = self._ids[obj_id]["body"]
        out = []
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            if b in (b1, b2) and con.dist < min_dist:
                out.append((b2 if b1 == b else b1, con.dist))
        return out

    def step(self, n: int = 1):
        for _ in range(n):
            self.hand.step()
            mujoco.mj_step(self.m, self.d)

    def place(self, obj_id: str, pos, quat):
        """Teleport an object and its carrier to a pose and weld them together."""
        ids = self._ids[obj_id]
        self.d.mocap_pos[ids["mocap"]] = pos
        self.d.mocap_quat[ids["mocap"]] = quat
        self.d.qpos[ids["qadr"] : ids["qadr"] + 3] = pos
        self.d.qpos[ids["qadr"] + 3 : ids["qadr"] + 7] = quat
        self.d.qvel[ids["vadr"] : ids["vadr"] + 6] = 0
        self.d.eq_active[ids["eq"]] = 1
        self._support(obj_id, 1.0)
        mujoco.mj_forward(self.m, self.d)

    def _support(self, obj_id: str, fraction: float):
        """Cancel `fraction` of the object's weight (carried objects: 1, so the soft weld does not sag)."""
        b = self._ids[obj_id]["body"]
        weight = self.m.body_subtreemass[b] * -self.m.opt.gravity[2]
        self.d.xfrc_applied[b] = [0, 0, weight * float(fraction), 0, 0, 0]

    def set_support(self, fraction: float):
        self._support(self.active, fraction)

    def select(self, obj_id: str):
        """Make obj_id the active object at its default grasp pose; park all others."""
        for other in C.OBJECTS:
            if other != obj_id:
                self.place(other, PARK_POS[other], [1, 0, 0, 0])
        self.active = obj_id
        cfg = C.OBJECTS[obj_id]
        self.place(obj_id, cfg["grasp_pos"], cfg["grasp_quat"])

    def release(self, keep_support: bool = False):
        """Disable the carrier weld. With keep_support the weight stays compensated (ramp it via set_support)."""
        self.d.eq_active[self._ids[self.active]["eq"]] = 0
        if not keep_support:
            self._support(self.active, 0.0)

    def move_carrier(self, obj_id: str, pos, quat=None):
        ids = self._ids[obj_id]
        self.d.mocap_pos[ids["mocap"]] = pos
        if quat is not None:
            self.d.mocap_quat[ids["mocap"]] = quat

    def object_pose(self, obj_id: str | None = None):
        b = self._ids[obj_id or self.active]["body"]
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()
