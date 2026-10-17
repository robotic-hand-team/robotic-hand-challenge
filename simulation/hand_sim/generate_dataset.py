"""Parallel data generation: run many simulated grasp sessions across worker processes and write
them in the models/dataset layout (via recorder.DatasetRecorder), for training classifier/.

Each worker builds its own World (MuJoCo models aren't picklable/shareable across processes) and
runs an independent slice of episodes, so throughput scales close to linearly with worker count.

Usage (from simulation/):
    ../venv/bin/python -m hand_sim.generate_dataset --episodes-per-object 60 --workers 8 \
        --out ../models/dataset_sim
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import config as C
from .episode import HandoverEpisode
from .policy import ScriptedGraspPolicy
from .recorder import DatasetRecorder
from .world import World


def _worker(worker_id: int, obj_id: str, n_episodes: int, seed: int, out_dir: str, min_frames: int):
    """Runs in a separate process: generates n_episodes sessions for one object, returns stats."""
    out_root = Path(out_dir)
    world = World(seed=seed)
    rng = np.random.default_rng(seed)
    written = successes = skipped = 0
    for i in range(n_episodes):
        rec = DatasetRecorder(world, rng)
        ep = HandoverEpisode(world, ScriptedGraspPolicy(), rng, obj_id=obj_id, on_frame=rec.record)
        result = ep.run()
        if rec.frame < min_frames:
            skipped += 1
            continue
        session_dir = out_root / f"{rec.started_at.strftime('%Y%m%d_%H%M%S')}_w{worker_id}_{i:04d}"
        rec.write(session_dir, {**ep.info, **result})
        written += 1
        successes += result["success"]
    return worker_id, obj_id, written, successes, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-per-object", type=int, default=60)
    ap.add_argument("--workers", type=int, default=max(1, __import__("os").cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(C.SIM_DIR.parent / "models" / "dataset_sim"))
    ap.add_argument("--min-frames", type=int, default=20, help="drop sessions shorter than this (failed setup, etc.)")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    objects = list(C.OBJECTS)
    jobs = []  # (worker_id, obj_id, n_episodes, seed)
    wid = 0
    for obj_id in objects:
        remaining = args.episodes_per_object
        n_workers_for_obj = max(1, args.workers // len(objects))
        per_worker = -(-remaining // n_workers_for_obj)  # ceil
        for w in range(n_workers_for_obj):
            n = min(per_worker, remaining)
            if n <= 0:
                break
            jobs.append((wid, obj_id, n, args.seed * 1000 + wid))
            remaining -= n
            wid += 1

    print(f"{len(jobs)} worker jobs, {args.episodes_per_object} episodes/object, "
          f"{sum(j[2] for j in jobs)} total episodes requested")
    t0 = time.perf_counter()
    totals = {obj: [0, 0, 0] for obj in objects}  # written, successes, skipped
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, wid, obj_id, n, seed, args.out, args.min_frames)
                   for wid, obj_id, n, seed in jobs]
        for fut in as_completed(futures):
            wid, obj_id, written, successes, skipped = fut.result()
            totals[obj_id][0] += written
            totals[obj_id][1] += successes
            totals[obj_id][2] += skipped
            print(f"  worker {wid:2d} ({obj_id}): {written} sessions written, "
                  f"{successes} successful grasps, {skipped} skipped (too short)")

    dt = time.perf_counter() - t0
    print(f"\nDone in {dt:.1f}s with {args.workers} workers")
    for obj_id, (written, successes, skipped) in totals.items():
        print(f"  {obj_id:11s}: {written} sessions ({successes} successful grasps, {skipped} skipped) -> {args.out}")


if __name__ == "__main__":
    main()
