"""Random handover episodes: a mocap carrier brings an object into the open hand, a policy grasps it,
the carrier lets go and the hand has to hold the object on its own.

Headless batch evaluation (from simulation/):
    ../venv/bin/python -m hand_sim.episode --episodes 50 --seed 0
"""

import argparse
import time
from collections import Counter, defaultdict

import mujoco
import numpy as np

from . import config as C
from .policy import PHASE_APPROACH, PHASE_GRASP, PHASE_HOLD, Action, Observation, Policy, ScriptedGraspPolicy
from .world import World

MAX_SAMPLE_TRIES = 80
PATH_CHECK_POINTS = 12
PENETRATION_TOL_M = -0.001
DROP_SHIFT_M = 0.05


def _quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), float(angle))
    return q


def _qmul(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return q


def _rotate(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=float), np.asarray(q, dtype=float))
    return out


def _quat_angle(a, b):
    return 2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0))


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class HandoverEpisode:
    def __init__(self, world: World, policy: Policy, rng: np.random.Generator, obj_id: str | None = None,
                 on_frame=None, step_fn=None, randomize: bool = True):
        self.world, self.policy, self.rng = world, policy, rng
        self.step_fn = step_fn or (lambda n: world.step(n))  # the viewer substitutes a locking/pacing stepper
        self.obj_id = obj_id or str(rng.choice(list(C.OBJECTS)))
        self.randomize = randomize  # False: start exactly at the object's nominal grasp_pos/grasp_quat
        self.on_frame = on_frame  # callable(episode, Observation) once per control tick (recorder hook)
        self._forced = False
        self.substeps = round(1.0 / (C.CONTROL_HZ * world.m.opt.timestep))
        self.done = False
        self.result: dict | None = None
        self._setup()

    # --- setup ---------------------------------------------------------------------------------
    def _sample_target(self):
        cfg = C.OBJECTS[self.obj_id]
        pivot = np.asarray(cfg["pivot"])
        q_nom = np.asarray(cfg["grasp_quat"], dtype=float)
        pivot_nom = np.asarray(cfg["grasp_pos"]) + _rotate(q_nom, pivot)

        roll = self.rng.uniform(*C.ROLL_RANGE_RAD)
        tilt_x = self.rng.uniform(*C.TILT_X_RANGE_RAD)
        tilt_z = self.rng.uniform(*C.TILT_Z_RANGE_RAD)
        offset = np.array([self.rng.uniform(lo, hi) for lo, hi in cfg["offset_range"]])
        quat = _qmul(_qmul(_quat([0, 0, 1], tilt_z), _quat([1, 0, 0], tilt_x)), _qmul(q_nom, _quat([0, 1, 0], roll)))
        # Occasional extreme orientations (edge cases).
        if self.rng.random() < 0.2:
            quat = _qmul(_quat([1, 0, 0], np.pi), quat)  # flipped end-over-end
        if self.rng.random() < 0.2:
            quat = _qmul(_quat([0, 0, 1], np.pi), quat)  # mirrored across palm midplane
        pivot_world = pivot_nom + offset
        pos = pivot_world - _rotate(quat, pivot)

        direction = np.array([self.rng.uniform(-0.3, 0.8), self.rng.uniform(-0.8, 0.8), 0.5])
        direction /= np.linalg.norm(direction)
        info = {
            "roll_rad": roll,
            "tilt_x_rad": tilt_x,
            "tilt_z_rad": tilt_z,
            "pivot_offset_m": offset.tolist(),
            "approach_dir": direction.tolist(),
        }
        return pos, quat, pos + direction * C.APPROACH_DIST_M, info

    def _path_clear(self, start, target, quat) -> bool:
        for s in np.linspace(0.0, 1.0, PATH_CHECK_POINTS):
            self.world.place(self.obj_id, start + (target - start) * s, quat)
            if self.world.object_contacts(self.obj_id, PENETRATION_TOL_M):
                return False
        return True

    def _setup(self):
        w = self.world
        w.reset(C.PRESHAPE_ANGLES)
        w.active = self.obj_id
        if self.randomize:
            for tries in range(1, MAX_SAMPLE_TRIES + 1):
                target, quat, start, info = self._sample_target()
                if self._path_clear(start, target, quat):
                    break
            else:
                raise RuntimeError(f"no collision-free handover found for {self.obj_id} in {MAX_SAMPLE_TRIES} tries")
        else:
            cfg = C.OBJECTS[self.obj_id]
            target, quat, start = np.array(cfg["grasp_pos"]), np.array(cfg["grasp_quat"]), np.array(cfg["grasp_pos"])
            tries, info = 0, {"roll_rad": 0.0, "tilt_x_rad": 0.0, "tilt_z_rad": 0.0,
                               "pivot_offset_m": [0.0, 0.0, 0.0], "approach_dir": [0.0, 0.0, 1.0]}
        self.target_pos, self.target_quat, self.start_pos = target, quat, start
        w.place(self.obj_id, start, quat)
        self.info = {"object": self.obj_id, "sample_tries": tries, "target_pos": target.tolist(),
                     "target_quat": quat.tolist(), **info}
        self.policy.reset(self.info)
        self.t0 = w.d.time
        self.phase, self.phase_t0 = PHASE_APPROACH, w.d.time
        self.release_reason = None
        self.release_pose = None

    # --- stepping ------------------------------------------------------------------------------
    @property
    def t(self):
        return self.world.d.time - self.t0

    def observation(self) -> Observation:
        raw = self.world.tactile.step(self.world.d)
        summary = self.world.tactile.summary(raw)
        pos, _ = self.world.object_pose(self.obj_id)
        palm_id = self.world.m.body("right_hand_root").id
        proximity = float(np.linalg.norm(pos - self.world.d.xpos[palm_id]))
        return Observation(t=self.t, phase=self.phase, phase_t=self.world.d.time - self.phase_t0,
                           registers=self.world.hand.read(), tactile=summary, proximity=proximity)

    def _apply(self, action: Action):
        hand = self.world.hand
        if action.force_set is not None:
            hand.set_forces(action.force_set)
        if action.speed_set is not None:
            hand.set_speeds(action.speed_set)
        if action.angle_set is not None:
            hand.set_angles(action.angle_set)

    def _enter(self, phase):
        self.phase, self.phase_t0 = phase, self.world.d.time

    def _pivot_pose(self):
        pos, quat = self.world.object_pose(self.obj_id)
        return pos + _rotate(quat, C.OBJECTS[self.obj_id]["pivot"]), quat

    def tick(self) -> bool:
        """Advance one control period (1 / CONTROL_HZ). Returns False once the episode has finished."""
        if self.done:
            return False
        obs = self.observation()
        if self.on_frame:
            self.on_frame(self, obs)
        action = self.policy.act(obs)
        self._apply(action)

        if self.phase == PHASE_GRASP and (action.release or obs.phase_t >= C.GRASP_TIMEOUT_S):
            self.release_reason = "policy" if action.release else "timeout"
            self.release_pose = self._pivot_pose()
            self.info["grasp_time_s"] = obs.phase_t
            self.world.release(keep_support=True)
            self._enter(PHASE_HOLD)

        w = self.world
        for _ in range(self.substeps):
            if self.phase == PHASE_APPROACH:
                s = _smoothstep((w.d.time - self.phase_t0) / C.APPROACH_TIME_S)
                w.move_carrier(self.obj_id, self.start_pos + (self.target_pos - self.start_pos) * s)
            elif self.phase == PHASE_HOLD:
                w.set_support(1.0 - _smoothstep((w.d.time - self.phase_t0) / C.RELEASE_RAMP_S))
            self.step_fn(1)

        phase_t = w.d.time - self.phase_t0
        if self.phase == PHASE_APPROACH and phase_t >= C.APPROACH_TIME_S:
            self._enter(PHASE_GRASP)
        elif self.phase == PHASE_HOLD and phase_t >= C.HOLD_TIME_S:
            self._finish()
        return not self.done

    def _finish(self):
        self.done = True
        pivot0, quat0 = self.release_pose
        pivot1, quat1 = self._pivot_pose()
        shift = float(np.linalg.norm(pivot1 - pivot0))
        rot = _quat_angle(quat0, quat1)
        contacts = self.world.object_contacts(self.obj_id, min_dist=0.0005)
        floor = any(b == 0 for b, _ in contacts)
        hand_bodies = {b for b, _ in contacts if self.world.m.body(b).name.startswith("right_")}
        if floor or shift > DROP_SHIFT_M:
            failure = "dropped"
        elif shift > C.SUCCESS_MAX_SHIFT_M:
            failure = "slipped"
        elif rot > C.SUCCESS_MAX_ROT_RAD:
            failure = "rotated"
        else:
            failure = None
        self.result = {
            **self.info,
            "release_reason": self.release_reason,
            "shift_mm": round(shift * 1000, 2),
            "rot_deg": round(np.degrees(rot), 2),
            "floor_contact": floor,
            "hand_bodies_in_contact": len(hand_bodies),
            "success": failure is None,
            "failure": failure,
        }

    def run(self) -> dict:
        while self.tick():
            pass
        return self.result


def main():
    ap = argparse.ArgumentParser(description="Run random handover episodes headless and report success rates.")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object", choices=list(C.OBJECTS), default=None, help="default: random per episode")
    ap.add_argument("--force", type=int, default=None, help="override the per-object grasp force (g)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = World(seed=args.seed)
    policy = ScriptedGraspPolicy(force_g=args.force)
    stats, failures = defaultdict(list), defaultdict(Counter)
    t_start = time.perf_counter()
    for i in range(args.episodes):
        res = HandoverEpisode(world, policy, rng, obj_id=args.object).run()
        stats[res["object"]].append(res["success"])
        if res["failure"]:
            failures[res["object"]][res["failure"]] += 1
        if args.verbose or not res["success"]:
            print(f"#{i:3d} {res['object']:11s} {'OK  ' if res['success'] else 'FAIL'} {res['failure'] or '':8s} "
                  f"shift={res['shift_mm']:6.1f}mm rot={res['rot_deg']:5.1f}deg release={res['release_reason']} "
                  f"roll={np.degrees(res['roll_rad']):6.1f} tilt_x={np.degrees(res['tilt_x_rad']):5.1f} "
                  f"tilt_z={np.degrees(res['tilt_z_rad']):5.1f} tries={res['sample_tries']}")
    wall = time.perf_counter() - t_start
    print(f"\n{args.episodes} episodes in {wall:.1f}s ({wall / args.episodes:.2f}s each)")
    for obj_id, ok in stats.items():
        print(f"  {obj_id:11s} success {sum(ok)}/{len(ok)} ({100 * np.mean(ok):.0f}%)  failures: {dict(failures[obj_id])}")


if __name__ == "__main__":
    main()
