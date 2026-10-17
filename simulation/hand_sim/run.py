"""Interactive MuJoCo viewer for the simulated hand.

Run from simulation/ (macOS needs mjpython for the viewer):
    ../venv/bin/mjpython -m hand_sim.run

Type commands in the terminal while the viewer is open (see HELP). Space pauses the physics.
"""

import argparse
import queue
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from . import config as C
from . import pose_import
from .world import World

HELP = f"""commands:
  obj <id>            switch object ({', '.join(C.OBJECTS)}), hand pre-shapes, object placed at grasp pose
  preshape            fingers open, thumb opposed
  close [force_g]     close all fingers with a force limit (default: object's grasp_force_g)
  open                open all DOFs
  angles a b c d e f  raw ANGLE_SET in DOF order {C.DOF_NAMES} (-1 = keep)
  force  a b c d e f  raw FORCE_SET (grams)
  speed  a b c d e f  raw SPEED_SET (0-1000)
  release             let go of the object (gravity takes over)
  attach              put the object back at its grasp pose
  status              print the register readout
  pose <file.json>    replay a tactile pose estimate: object in the estimated orientation, slid into the
                      open hand until it touches, then the fingers close on it (pose/output/.../<session>.sim.json)
  pose_raw <file>     only teleport hand and object to the raw estimate (no physics, may overlap)
  help | quit"""


def _stdin_reader(q: queue.Queue):
    for line in sys.stdin:
        q.put(line.strip())


def _print_status(world: World):
    r = world.hand.read()
    pos, _ = world.object_pose()
    print(f"t={world.d.time:6.2f}s  object={world.active} pos(mm)={np.round(pos * 1000, 1)}")
    for key in ("angle_set", "angle_act", "force_act", "current", "status"):
        print(f"  {key:10s} {r[key]}")


_replay = {"seq": None}  # running pose_import.GraspReplay, advanced by the viewer loop


def _handle(cmd: str, world: World) -> bool:
    parts = cmd.split()
    if not parts:
        return True
    op, args = parts[0], parts[1:]
    if op not in ("status", "help"):
        _replay["seq"] = None  # any other command takes over from a running replay
    hand = world.hand
    try:
        if op in ("quit", "exit"):
            return False
        if op == "help":
            print(HELP)
        elif op == "obj":
            hand.set_angles(C.PRESHAPE_ANGLES)
            world.select(args[0])
        elif op == "preshape":
            hand.set_angles(C.PRESHAPE_ANGLES)
        elif op == "close":
            force = int(args[0]) if args else C.OBJECTS[world.active]["grasp_force_g"]
            hand.set_forces([force] * 6)
            hand.set_angles([0, 0, 0, 0, 0, -1])
        elif op == "open":
            hand.set_angles([C.ANGLE_MAX] * 6)
        elif op in ("angles", "force", "speed"):
            vals = [int(a) for a in args]
            assert len(vals) == 6, "need 6 values"
            {"angles": hand.set_angles, "force": hand.set_forces, "speed": hand.set_speeds}[op](vals)
        elif op == "release":
            world.release()
        elif op == "attach":
            cfg = C.OBJECTS[world.active]
            world.place(world.active, cfg["grasp_pos"], cfg["grasp_quat"])
        elif op == "status":
            _print_status(world)
        elif op == "pose":
            est = pose_import.load(" ".join(args).strip('"'))
            _replay["seq"] = pose_import.GraspReplay(world, est)
            print(f"[pose] {est['session']}: {est['object']} placed {_replay['seq'].backoff_mm:.0f} mm out of the hand, "
                  "approaching (runs about 4 s; don't pause with space)")
        elif op == "pose_raw":
            est = pose_import.load(" ".join(args).strip('"'))
            placed = pose_import.apply(world, est)
            print(pose_import.report(world, est, placed))
        else:
            print(f"unknown command {op!r}; type help")
    except (IndexError, ValueError, KeyError, AssertionError, OSError) as e:
        print(f"bad command: {e}")
    return True


def _demo_script():
    """Endless (sim time in s, command) sequence: grasp, release, hold, open, next object."""
    t = 0.0
    while True:
        for obj_id in C.OBJECTS:
            for dt, cmd in ((0.5, f"obj {obj_id}"), (1.0, "close"), (1.5, "release"), (3.0, "open"), (1.5, "preshape")):
                t += dt
                yield t, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default=next(iter(C.OBJECTS)), choices=list(C.OBJECTS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--demo", action="store_true", help="loop grasp/release/open on all objects automatically")
    ap.add_argument("--pose", help="start by replaying a tactile pose estimate (<session>.sim.json)")
    args = ap.parse_args()

    world = World(seed=args.seed)
    world.hand.set_angles(C.PRESHAPE_ANGLES)
    world.step(int(0.8 / world.m.opt.timestep))
    world.select(args.object)
    if args.pose:
        _handle(f"pose {args.pose}", world)

    commands: queue.Queue = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(commands,), daemon=True).start()
    paused = [False]

    def key_callback(keycode):
        if keycode == 32:
            paused[0] = not paused[0]
            print("[paused]" if paused[0] else "[running]")

    demo = _demo_script() if args.demo else None
    next_demo = next(demo) if demo else None
    demo_t0 = world.d.time

    print(HELP)
    with mujoco.viewer.launch_passive(world.m, world.d, key_callback=key_callback) as viewer:
        viewer.opt.geomgroup[2] = 0  # hand collision meshes
        viewer.opt.geomgroup[3] = 0  # object collision hulls
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 150, -15, 0.45
        viewer.cam.lookat[:] = [0.03, 0.1, 0.12]

        steps_per_frame = max(1, round((1 / 60) / world.m.opt.timestep))
        while viewer.is_running():
            frame_start = time.perf_counter()
            while next_demo and world.d.time - demo_t0 >= next_demo[0]:
                print(f"[demo] {next_demo[1]}")
                with viewer.lock():
                    _handle(next_demo[1], world)
                if next_demo[1] == "release":
                    with viewer.lock():
                        _print_status(world)
                next_demo = next(demo)
            while not commands.empty():
                with viewer.lock():
                    if not _handle(commands.get(), world):
                        return
            if not paused[0]:
                with viewer.lock():
                    if _replay["seq"]:
                        msg = _replay["seq"].update(world)
                        if msg:
                            print(msg)
                        if _replay["seq"].done:
                            _replay["seq"] = None
                    world.step(steps_per_frame)
            viewer.sync()
            time.sleep(max(0.0, steps_per_frame * world.m.opt.timestep - (time.perf_counter() - frame_start)))


if __name__ == "__main__":
    main()
