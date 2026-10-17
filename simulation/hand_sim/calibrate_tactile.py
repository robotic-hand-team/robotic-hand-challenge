"""One-off dev tool: sample real contact points during many grasps to find where each tactile
sensor patch sits on the hand mesh (position, tangent axes, extent), in each body's local frame.

The output feeds the SENSOR_GROUPS patch parameters in config.py; it is not needed at runtime.

Usage (from simulation/):  ../venv/bin/python -m hand_sim.calibrate_tactile [--episodes N]
"""

import argparse
from collections import defaultdict

import mujoco
import numpy as np

from .episode import HandoverEpisode
from .policy import PHASE_GRASP, PHASE_HOLD, ScriptedGraspPolicy
from .world import World

# Bodies that carry at least one tactile group (see hand_manual.pdf p.26 + AGENTS.md mapping).
TACTILE_BODIES = [
    "right_pinky_intermediate", "right_pinky_proximal",
    "right_ring_intermediate", "right_ring_proximal",
    "right_middle_intermediate", "right_middle_proximal",
    "right_index_intermediate", "right_index_proximal",
    "right_thumb_distal", "right_thumb_intermediate", "right_thumb_proximal",
    "right_hand_root",
]


def collect(world: World, n_episodes: int, rng: np.random.Generator) -> dict:
    m, d = world.m, world.d
    body_ids = {name: m.body(name).id for name in TACTILE_BODIES}
    points = defaultdict(list)  # body name -> list of (local_pos(3), local_normal(3))

    def on_frame(ep, obs):
        if obs.phase not in (PHASE_GRASP, PHASE_HOLD):
            return
        for i in range(d.ncon):
            con = d.contact[i]
            for geom, sign in ((con.geom1, 1.0), (con.geom2, -1.0)):
                b = m.geom_bodyid[geom]
                name = m.body(b).name
                if name not in body_ids or body_ids[name] != b:
                    continue
                xmat = d.xmat[b].reshape(3, 3)
                local_pos = xmat.T @ (con.pos - d.xpos[b])
                normal_world = con.frame[:3] * sign
                local_normal = xmat.T @ normal_world
                points[name].append((local_pos, local_normal))

    for obj_id in ("vegabar_29", "vegabar_38"):
        for _ in range(n_episodes):
            HandoverEpisode(world, ScriptedGraspPolicy(), rng, obj_id=obj_id, on_frame=on_frame).run()
    return points


def fit_patch(samples: list) -> dict:
    pos = np.array([p for p, _ in samples])
    normal = np.array([n for _, n in samples])
    mean_normal = normal.mean(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    centroid = pos.mean(axis=0)

    # Tangent basis: PCA of contact points projected onto the plane perpendicular to mean_normal.
    rel = pos - centroid
    rel -= np.outer(rel @ mean_normal, mean_normal)
    cov = rel.T @ rel
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    t1, t2 = eigvec[:, order[0]], eigvec[:, order[1]]  # t1 = long axis (finger length), t2 = width
    if np.dot(np.cross(t1, t2), mean_normal) < 0:
        t2 = -t2
    u, v = rel @ t1, rel @ t2
    return {
        "n_contacts": len(samples),
        "centroid": np.round(centroid, 5).tolist(),
        "normal": np.round(mean_normal, 3).tolist(),
        "tangent_long": np.round(t1, 3).tolist(),
        "tangent_wide": np.round(t2, 3).tolist(),
        "extent_long_mm": round(float(u.max() - u.min()) * 1000, 1),
        "extent_wide_mm": round(float(v.max() - v.min()) * 1000, 1),
        "offset_along_long_mm": round(float(u.mean()) * 1000, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25, help="per object")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = World(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    points = collect(world, args.episodes, rng)
    for name in TACTILE_BODIES:
        samples = points.get(name, [])
        if len(samples) < 10:
            print(f"{name:28s} only {len(samples)} contacts, skipping")
            continue
        print(f"{name}: {fit_patch(samples)}")


if __name__ == "__main__":
    main()
