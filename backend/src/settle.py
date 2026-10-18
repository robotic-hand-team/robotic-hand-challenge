"""Physics settle for the live object estimate.

The live position estimate (a ridge regression on tactile centroids) knows nothing about the fingers:
placed as is, with the nominal grasp orientation, the object overlaps the hand by ~13 mm on median.
Here the estimate is only the anchor of a soft spring. The hand is held fixed at the measured finger
pose, the object starts from the least-overlapping of a few candidate poses (the previous settled pose
first), and MuJoCo steps let the contacts push it out of the fingers and tilt it to fit between them.

Gravity is off and the carrier welds are released, so the spring and the contacts are the only forces.
Contacts use a stiffer override than the grasp simulation: here they only have to separate bodies
quickly, not reproduce a compliant grasp.
"""
import mujoco
import numpy as np

from .twin import SC, build_model

MIN_STEPS = 40              # 40 ms of sim time at the 1 ms timestep
MAX_STEPS = 100             # keep stepping while the object still overlaps the hand by more than DONE_PEN
DONE_PEN = 0.003            # m
SPRING_HZ = 12.0            # pull toward the estimate: weak enough that contacts win
ROT_SPRING_HZ = 6.0         # pull back toward the nominal grasp orientation (30 Hz goes unstable)
DAMPING_RATIO = 1.5         # overdamped: an object pushed out of deep overlap must not be flung away
SETTLE_SOLREF = [0.003, 1.0]
WORKSPACE_RADIUS = 0.030    # m: estimates further than this from the nominal grasp position are clamped
PALM_NORMAL = np.array([1.0, 0.0, 0.0])  # sim hand frame: palm faces +x
CANDIDATE_OFFSETS = (0.010, 0.020, 0.030)  # m along the palm normal: starts outside fingers that pierce the object
PARK = np.array([5.0, 5.0, 5.0])


def clamp_to_workspace(p: np.ndarray, obj: str) -> np.ndarray:
    g = np.asarray(SC.OBJECTS[obj]["grasp_pos"], float)
    off = p - g
    r = np.linalg.norm(off)
    return p if r <= WORKSPACE_RADIUS else g + off * (WORKSPACE_RADIUS / r)


class PoseSettler:
    def __init__(self, twin):
        self.twin = twin
        self.m = build_model()   # own model and data: twin.d is shared with the replay endpoints
        m = self.m
        m.opt.gravity[:] = 0
        m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
        m.opt.o_solref[:] = SETTLE_SOLREF
        m.opt.o_solimp[:] = m.geom_solimp[0]
        m.opt.o_margin = 0
        self.d = mujoco.MjData(m)
        for e in range(m.neq):
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD:
                self.d.eq_active[e] = 0
        hand = set(twin.hand_bodies)
        self.hand_ids = {m.body(n).id for n in hand}
        joints = [j for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name in hand]
        self.hand_qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self.hand_dadr = np.array([m.jnt_dofadr[j] for j in joints])
        self.obj = {}
        for o in twin.objects:
            jid = m.joint(f"{o}_free").id
            b = m.body(o).id
            mass = float(m.body_mass[b])
            inertia = float(m.body_inertia[b].max())
            w, wr = 2 * np.pi * SPRING_HZ, 2 * np.pi * ROT_SPRING_HZ
            self.obj[o] = dict(body=b, qadr=m.jnt_qposadr[jid], dadr=m.jnt_dofadr[jid],
                               k=mass * w * w, c=2 * DAMPING_RATIO * mass * w,
                               kr=inertia * wr * wr, cr=2 * DAMPING_RATIO * inertia * wr)
        self.hand_q = None
        self.last_pen = 0.0

    def _hand_qpos(self, angles: np.ndarray) -> np.ndarray:
        # Same mapping as Twin.poses: linear DOF angles, coupled joints from the equality polynomials.
        q = np.zeros(self.m.nq)
        t = self.twin
        q[t.dof_qadr] = t.dof_qmax * (1.0 - np.clip(angles, 0, 1000) / 1000.0)
        for q1, q2, c in t.couplings:
            x = q[q2]
            q[q1] = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4
        return q[self.hand_qadr]

    def _overlap(self, obj: str) -> float:
        """Deepest hand-object overlap (m, >= 0) at the current qpos, hand reset to the measured pose."""
        m, d, b = self.m, self.d, self.obj[obj]["body"]
        d.qpos[self.hand_qadr] = self.hand_q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        worst = 0.0
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom[0]], m.geom_bodyid[c.geom[1]]
            if (b1 == b and b2 in self.hand_ids) or (b2 == b and b1 in self.hand_ids):
                worst = min(worst, c.dist)
        return -worst

    def _best_start(self, obj, target, q_nom, start):
        a = self.obj[obj]["qadr"]
        cands = [] if start is None else [np.asarray(start, float)]
        cands.append(np.array([*target, *q_nom]))
        cands += [np.array([*(target + PALM_NORMAL * off), *q_nom]) for off in CANDIDATE_OFFSETS]
        cands.append(np.array([*SC.OBJECTS[obj]["grasp_pos"], *q_nom]))
        best, best_score = None, np.inf
        # Least overlap wins; near-ties go to the earliest candidate, i.e. the previous settled pose, which
        # keeps the object steady between frames. (Preferring the candidate closest to the estimate
        # instead was measured to leave more overlap.)
        for c in cands:
            self.d.qpos[a:a + 7] = c
            score = max(self._overlap(obj) - 0.001, 0.0)
            if score < best_score - 1e-4:
                best, best_score = c, score
        return best

    def settle(self, angles: np.ndarray, obj: str, target: np.ndarray, start: np.ndarray | None) -> np.ndarray:
        """Settled object pose [x, y, z, qw, qx, qy, qz] (sim world frame) for the measured finger angles,
        pulled toward `target` (position) and the nominal grasp orientation."""
        m, d, o = self.m, self.d, self.obj[obj]
        q_nom = np.asarray(SC.OBJECTS[obj]["grasp_quat"], float)
        self.hand_q = self._hand_qpos(angles)
        d.qvel[:] = 0
        d.xfrc_applied[:] = 0
        for other, p in self.obj.items():
            if other != obj:
                d.qpos[p["qadr"]:p["qadr"] + 7] = [*PARK, 1, 0, 0, 0]
        a, v = o["qadr"], o["dadr"]
        d.qpos[a:a + 7] = self._best_start(obj, target, q_nom, start)
        neg, dq, err, R = np.zeros(4), np.zeros(4), np.zeros(3), np.zeros(9)
        for step in range(MAX_STEPS):
            if step >= MIN_STEPS and step % 20 == 0 and self._overlap(obj) <= DONE_PEN:
                break
            d.qpos[self.hand_qadr] = self.hand_q
            d.qvel[self.hand_dadr] = 0
            pos, quat = d.qpos[a:a + 3], d.qpos[a + 3:a + 7]
            mujoco.mju_quat2Mat(R, quat)
            omega = R.reshape(3, 3) @ d.qvel[v + 3:v + 6]  # free joint: linear velocity world, angular local
            mujoco.mju_negQuat(neg, quat)
            mujoco.mju_mulQuat(dq, q_nom, neg)
            mujoco.mju_quat2Vel(err, dq, 1.0)   # rotation vector from current to nominal, world frame
            d.xfrc_applied[o["body"], :3] = o["k"] * (target - pos) - o["c"] * d.qvel[v:v + 3]
            d.xfrc_applied[o["body"], 3:] = o["kr"] * err - o["cr"] * omega
            mujoco.mj_step(m, d)
        out = d.qpos[a:a + 7].copy()
        out[3:] /= np.linalg.norm(out[3:])
        self.last_pen = self._overlap(obj)
        return out
