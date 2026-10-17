"""Emulates the RH56DFTP register-level behaviour on top of the MuJoCo joint servos.

Inputs mirror ANGLE_SET / FORCE_SET / SPEED_SET, outputs mirror ANGLE_ACT / FORCE_ACT /
CURRENT / ERROR / STATUS (docs/internal/hand_manual.pdf, section 2.6).
"""

import mujoco
import numpy as np

from . import config as C

N_DOF = len(C.DOF_NAMES)


def _tip_lever_arm(model, data, joint_id, tip_body_id):
    """Distance from the DOF joint axis anchor to the farthest collision vertex of the tip body."""
    anchor = data.xanchor[joint_id]
    best = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != tip_body_id or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[g] == 0:
            continue
        mid = model.geom_dataid[g]
        start, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[start : start + n]
        world = data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T
        best = max(best, float(np.linalg.norm(world - anchor, axis=1).max()))
    return best


class HandController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator | None = None):
        self.m, self.d = model, data
        self.rng = rng or np.random.default_rng()
        self.joint_ids = [model.joint(j).id for j in C.DOF_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.qmax = np.array([model.jnt_range[j][1] for j in self.joint_ids])
        self.act_ids = [model.actuator(f"act_{n}").id for n in C.DOF_NAMES]

        mujoco.mj_forward(model, data)
        tips = [model.body(b).id for b in C.DOF_TIP_BODIES]
        self.lever = np.array([_tip_lever_arm(model, data, j, b) for j, b in zip(self.joint_ids, tips)])
        # Bodies moved by each DOF (joint body and all descendants); contacts on them load the actuator.
        self.dof_bodies = []
        for j in self.joint_ids:
            root = model.jnt_bodyid[j]
            bodies = {root}
            for b in range(root + 1, model.nbody):
                if model.body_parentid[b] in bodies:
                    bodies.add(b)
            self.dof_bodies.append(bodies)
        self._f6 = np.zeros(6)

        self.angle_set = np.full(N_DOF, C.ANGLE_MAX, dtype=int)
        self.force_set = np.full(N_DOF, 1000, dtype=int)
        self.speed_set = np.full(N_DOF, C.SPEED_MAX, dtype=int)
        self.q_cmd = data.qpos[self.qadr].copy()
        self.force_stopped = np.zeros(N_DOF, dtype=bool)
        self.closing = np.zeros(N_DOF, dtype=bool)
        self._force_g = np.full(N_DOF, C.FORCE_ACT_IDLE_G)

    def reset(self, angles=None):
        """Teleport the DOFs to the given ANGLE_SET posture (default: fully open) and clear state."""
        angles = np.asarray(C.PRESHAPE_ANGLES if angles is None else angles, dtype=int)
        self.angle_set = angles.copy()
        self.force_set = np.full(N_DOF, 1000, dtype=int)
        self.speed_set = np.full(N_DOF, C.SPEED_MAX, dtype=int)
        self.q_cmd = self.qmax * (1.0 - angles / C.ANGLE_MAX)
        self.d.qpos[self.qadr] = self.q_cmd
        self.d.ctrl[self.act_ids] = self.q_cmd
        self.force_stopped[:] = False
        self.closing[:] = False
        self._force_g[:] = C.FORCE_ACT_IDLE_G

    # --- register writes -------------------------------------------------------------------
    def set_angles(self, values):
        for i, v in enumerate(values):
            v = int(v)
            if v == -1:  # -1 = no action for this DOF
                continue
            v = int(np.clip(v, 0, C.ANGLE_MAX))
            if v != self.angle_set[i] or self.force_stopped[i]:
                self.closing[i] = self._angle_to_q(v, i) > self.d.qpos[self.qadr[i]]
                self.force_stopped[i] = False
            self.angle_set[i] = v

    def set_forces(self, values):
        self.force_set = np.clip(np.asarray(values, dtype=int), 0, C.FORCE_SET_MAX)

    def set_speeds(self, values):
        self.speed_set = np.clip(np.asarray(values, dtype=int), 0, C.SPEED_MAX)

    # --- simulation step -------------------------------------------------------------------
    def _angle_to_q(self, angle, i):
        return self.qmax[i] * (1.0 - angle / C.ANGLE_MAX)

    def load_torques(self) -> np.ndarray:
        """Torque that external contacts exert against the closing direction of each DOF (Nm, >= 0)."""
        m, d = self.m, self.d
        tau = np.zeros(N_DOF)
        for c_id in range(d.ncon):
            con = d.contact[c_id]
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            mujoco.mj_contactForce(m, d, c_id, self._f6)
            # mj_contactForce returns the force exerted by geom1 on geom2, in the contact frame.
            f_on_2 = con.frame.reshape(3, 3).T @ self._f6[:3]
            for i in range(N_DOF):
                bodies = self.dof_bodies[i]
                if (b1 in bodies) == (b2 in bodies):
                    continue
                f = f_on_2 if b2 in bodies else -f_on_2
                j = self.joint_ids[i]
                t = np.dot(np.cross(con.pos - d.xanchor[j], f), d.xaxis[j])
                tau[i] += t
        return np.maximum(-tau, 0.0)

    def step(self):
        """Call once per physics step, before mj_step."""
        dt = self.m.opt.timestep
        q = self.d.qpos[self.qadr]
        raw_g = self.load_torques() / self.lever / 9.81e-3
        self._force_g += (raw_g - self._force_g) * min(1.0, dt / 0.01)  # ~10 ms sensor low-pass

        rate = self.qmax / C.FULL_STROKE_TIME_S * (self.speed_set / C.SPEED_MAX)
        for i in range(N_DOF):
            if self.force_stopped[i]:
                continue
            if self.closing[i] and self._force_g[i] >= self.force_set[i]:
                self.force_stopped[i] = True
                self.q_cmd[i] = q[i]
                continue
            goal = self._angle_to_q(self.angle_set[i], i)
            self.q_cmd[i] += np.clip(goal - self.q_cmd[i], -rate[i] * dt, rate[i] * dt)
        self.d.ctrl[self.act_ids] = self.q_cmd

    # --- register reads --------------------------------------------------------------------
    def read(self) -> dict:
        q = self.d.qpos[self.qadr]
        qd = self.d.qvel[self.vadr]
        angle_act = np.clip(np.round(C.ANGLE_MAX * (1.0 - q / self.qmax)), 0, C.ANGLE_MAX).astype(int)
        force_act = np.round(self._force_g + C.FORCE_ACT_IDLE_G + self.rng.normal(0, 1.0, N_DOF)).astype(int)
        moving = np.abs(qd) > 0.05
        # Placeholder current model until fitted against models/dataset (milestone 4).
        current = np.round(0.25 * self._force_g + 150.0 * moving).astype(int)
        status = np.where(
            self.force_stopped,
            C.STATUS_FORCE_STOP,
            np.where(self.closing, C.STATUS_GRASPING, C.STATUS_RELEASING),
        )
        return {
            "angle_set": self.angle_set.copy(),
            "force_set": self.force_set.copy(),
            "speed_set": self.speed_set.copy(),
            "angle_act": angle_act,
            "force_act": np.clip(force_act, -4000, 4000),
            "current": np.clip(current, 0, 2000),
            "error": np.zeros(N_DOF, dtype=int),
            "status": status.astype(int),
        }
