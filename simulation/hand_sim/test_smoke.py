"""Fast regression checks for the grasp simulation (no viewer, no CAD regen needed).

Plain functions so this runs without pytest (the venv has no internet access to install it):
    ../venv/bin/python -m hand_sim.test_smoke
If pytest ever becomes available, `pytest hand_sim/test_smoke.py -q` also works (test_* functions,
no fixtures/parametrize so nothing extra is required).
"""

import csv
import glob
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import config as C
from .episode import HandoverEpisode
from .policy import ScriptedGraspPolicy
from .recorder import DatasetRecorder
from .world import World

REAL_DATASET_DIR = C.SIM_DIR.parent / "models" / "dataset"

# Success rates are stochastic; thresholds are well below what ScriptedGraspPolicy achieves in
# practice (~65-85%, see hand_sim/episode.py --episodes) so this catches regressions, not noise.
MIN_SUCCESS_RATE = {"vegabar_29": 0.2, "vegabar_38": 0.2}
N_EPISODES_PER_OBJECT = 15


def test_objects_built():
    for obj_id in C.OBJECTS:
        obj_dir = C.OBJECTS_DIR / obj_id
        assert (obj_dir / "object.json").exists(), f"run `python -m hand_sim.assets` first ({obj_id})"


def test_nominal_grasp_holds():
    """Grasping each object at its default (un-randomized) pose and releasing should not drop it."""
    world = World(seed=0)
    rng = np.random.default_rng(0)
    for obj_id in C.OBJECTS:
        result = HandoverEpisode(world, ScriptedGraspPolicy(), rng, obj_id=obj_id, randomize=False).run()
        assert result["success"], f"{obj_id} nominal grasp failed: {result}"


def test_random_handover_success_rate():
    """Scripted policy should succeed on most random handovers (catches scene/physics regressions)."""
    world = World(seed=0)
    rng = np.random.default_rng(1)
    for obj_id in C.OBJECTS:
        results = [HandoverEpisode(world, ScriptedGraspPolicy(), rng, obj_id=obj_id).run()
                   for _ in range(N_EPISODES_PER_OBJECT)]
        rate = np.mean([r["success"] for r in results])
        assert rate >= MIN_SUCCESS_RATE[obj_id], (
            f"{obj_id} success rate {rate:.0%} below {MIN_SUCCESS_RATE[obj_id]:.0%}: "
            f"{[r['failure'] for r in results]}"
        )


def test_recorder_matches_real_dataset_headers():
    """Column headers must match a real (new-schema) session exactly, or NN training code breaks."""
    real_sessions = sorted(glob.glob(str(REAL_DATASET_DIR / "2026*" / "actuators.csv")))
    real_dir = next((Path(f).parent for f in real_sessions if "pos_act_little" in open(f).readline()), None)
    assert real_dir is not None, f"no real session with the newer actuators.csv schema found under {REAL_DATASET_DIR}"

    world = World(seed=0)
    rng = np.random.default_rng(0)
    rec = DatasetRecorder(world, rng)
    HandoverEpisode(world, ScriptedGraspPolicy(), rng, obj_id="vegabar_29", on_frame=rec.record).run()
    tmp_dir = Path(tempfile.mkdtemp()) / "session"
    try:
        rec.write(tmp_dir, {"object": "vegabar_29"})
        for fname in ("actuators.csv", "tactile_raw.csv", "tactile_summary.csv"):
            real_header = open(real_dir / fname).readline().strip().split(",")
            sim_header = open(tmp_dir / fname).readline().strip().split(",")
            assert real_header == sim_header, f"{fname} header mismatch vs {real_dir}"
    finally:
        shutil.rmtree(tmp_dir.parent, ignore_errors=True)


def test_hand_registers_match_manual_ranges():
    world = World(seed=0)
    world.reset(C.PRESHAPE_ANGLES)
    r = world.hand.read()
    assert np.all((r["angle_act"] >= 0) & (r["angle_act"] <= C.ANGLE_MAX))
    assert np.all((r["force_act"] >= -4000) & (r["force_act"] <= 4000))
    assert np.all((r["current"] >= 0) & (r["current"] <= 2000))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        print(f"{test.__name__} ... ", end="", flush=True)
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"FAIL\n  {e}")
        else:
            print("ok")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
