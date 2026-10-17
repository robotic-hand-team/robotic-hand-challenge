"""Interactive MuJoCo viewer for the simulated hand.

Run from simulation/ (macOS needs mjpython for the viewer):
    ../venv/bin/mjpython -m hand_sim.run --demo

Type commands in the terminal while the viewer is open (see HELP). Space pauses the physics.

The top-left overlay shows the live Sensor A/B estimate from classifier/ next to the simulation's
ground-truth object, so a sim-to-real prediction can be eyeballed. Model variant via
CLASSIFIER_SUFFIX (see hand_sim/classifier_bridge.py), or --no-classifier to turn it off.

The hand's tactile sensors are shown as a live thermal heat map: a smooth, continuous field over the
hand's silhouette in the bottom-left corner (deep blue idle -> green/yellow transition -> orange/red
with yellow-white hot cores on contact), optionally plus glowing tiles on the 3D hand.
--heatmap panel|3d|both|off, --heatmap-size N, --heatmap-gain F, --heatmap-stride N.
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
from .episode import HandoverEpisode
from .policy import PHASE_APPROACH, ScriptedGraspPolicy
from .tactile_panel import TactilePanel
from .taxel_overlay import TaxelOverlay
from .world import World

try:
    from .classifier_bridge import ClassifierBridge, LABEL_NAMES, OBJECT_TO_LABEL
except ImportError as e:  # joblib/xgboost not installed
    ClassifierBridge = None
    LABEL_NAMES = {0: "Sensor A (VEGABAR 29)", 1: "Sensor B (VEGABAR 38)"}
    OBJECT_TO_LABEL = {"vegabar_29": 0, "vegabar_38": 1}
    print(f"classifier bridge unavailable: {e}")

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
  status               print the register readout
  tactile              print tactile group sums/contacts above the contact threshold
  help | quit
(run with --demo for an automatic loop of random handover episodes instead)
(flags: --heatmap panel|3d|both|off, --heatmap-size N, --heatmap-gain F, --heatmap-stride N, --no-classifier)"""


def _stdin_reader(q: queue.Queue):
    for line in sys.stdin:
        q.put(line.strip())


def _print_status(world: World):
    r = world.hand.read()
    pos, _ = world.object_pose()
    print(f"t={world.d.time:6.2f}s  object={world.active} pos(mm)={np.round(pos * 1000, 1)}")





def _handle(cmd: str, world: World) -> bool:
    parts = cmd.split()
    if not parts:
        return True
    op, args = parts[0], parts[1:]
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
        elif op == "tactile":
            raw = world.tactile.step(world.d)
            summary = world.tactile.summary(raw)
            active = {k: v for k, v in summary.items() if v["contacts"] > 0}
            print(active or "(no taxel above the contact threshold)")
        else:
            print(f"unknown command {op!r}; type help")
    except (IndexError, ValueError, KeyError, AssertionError) as e:
        print(f"bad command: {e}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default=next(iter(C.OBJECTS)), choices=list(C.OBJECTS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--demo", action="store_true",
                    help="loop random handover episodes (scripted policy) on random objects")
    ap.add_argument("--no-classifier", action="store_true",
                    help="skip the Sensor A/B classifier overlay")
    ap.add_argument("--heatmap", choices=("panel", "3d", "both", "off"), default="panel",
                    help="tactile heat map: thermal panel in the corner (default), tiles on the 3D "
                         "hand, both, or off")
    ap.add_argument("--heatmap-size", type=int, default=300,
                    help="height of the thermal heat map panel in pixels (default 300)")
    ap.add_argument("--heatmap-gain", type=float, default=1.0,
                    help="scale tactile counts before the thermal colormap; >1 makes light contact "
                         "glow (the sim saturates around 4095 raw counts)")
    ap.add_argument("--heatmap-stride", type=int, default=1,
                    help="3D tiles: draw every Nth taxel (raise it if that costs too much FPS)")
    args = ap.parse_args()

    world = World(seed=args.seed)
    world.hand.set_angles(C.PRESHAPE_ANGLES)
    world.step(int(0.8 / world.m.opt.timestep))
    world.select(args.object)

    commands: queue.Queue = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(commands,), daemon=True).start()
    paused = [False]

    def key_callback(keycode):
        if keycode == 32:
            paused[0] = not paused[0]
            print("[paused]" if paused[0] else "[running]")

    demo_rng = np.random.default_rng(args.seed)
    demo_policy = ScriptedGraspPolicy()
    episode = None
    n_episodes = n_success = 0
    pace = {"steps": 0}

    classifier = None
    if not args.no_classifier and ClassifierBridge is not None:
        try:
            classifier = ClassifierBridge()
            print(f"[viewer] classifier: xgb_sensor{classifier.suffix}.pkl "
                  f"({len(classifier.feature_cols)} features) -- overlay top-left")
        except Exception as e:
            print(f"[viewer] classifier unavailable ({e}); overlay shows ground truth only")

    tiles = None
    if args.heatmap in ("3d", "both"):
        tiles = TaxelOverlay(world)
    panel = None
    if args.heatmap in ("panel", "both"):
        panel = TactilePanel(world, height=args.heatmap_size, gain=args.heatmap_gain)
    if tiles is not None or panel is not None:
        print(f"[viewer] tactile heat map: {args.heatmap} mode"
              + (f", {tiles.n_taxels} tiles on the hand (stride {args.heatmap_stride})" if tiles else "")
              + (f", {panel.width}x{panel.height} px thermal panel bottom-left of the 3D view" if panel else ""))

    print(HELP)
    with mujoco.viewer.launch_passive(world.m, world.d, key_callback=key_callback) as viewer:
        viewer.opt.geomgroup[2] = 0  # hand collision meshes
        viewer.opt.geomgroup[3] = 0  # object collision hulls
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 150, -15, 0.45
        viewer.cam.lookat[:] = [0.03, 0.1, 0.12]

        steps_per_frame = max(1, round((1 / 60) / world.m.opt.timestep))

        def paced_step(n):
            """Episode stepper: step under the viewer lock, sync and sleep about once per rendered frame."""
            with viewer.lock():
                world.step(n)
            pace["steps"] += n
            if pace["steps"] >= steps_per_frame:
                pace["steps"] = 0
                viewer.sync()
                time.sleep(steps_per_frame * world.m.opt.timestep)

        class_stats = {"ok": 0, "n": 0}  # demo: episodes whose majority vote matched the truth

        def sample_tactile():
            """Sample the tactile sensors once and redraw the heat map; returns (raw, registers)."""
            with viewer.lock():
                raw = world.tactile.step(world.d)
                registers = world.hand.read()
                if tiles is not None:
                    tiles.update(viewer.user_scn, raw, stride=args.heatmap_stride)
            if panel is not None:
                viewer.set_images([(panel.rect(viewer), panel.render(raw))])
            return raw, registers

        def feed_classifier(raw, registers):
            if classifier is not None:
                classifier.predict(world.tactile.summary(raw), registers)

        def sample_sensors():
            """Refresh the heat map and (if enabled) run the classifier on the same tactile frame."""
            raw, registers = sample_tactile()
            feed_classifier(raw, registers)

        def overlay_text(truth_obj):
            truth = OBJECT_TO_LABEL.get(truth_obj, -1)
            lines = ["Sensor estimate (ML)", f"truth : {LABEL_NAMES.get(truth, truth_obj)}"]
            _pos, _quat = world.object_pose()
            _ori = np.zeros(3)
            mujoco.mju_rotVecQuat(_ori, np.array([0.0, 1.0, 0.0]), np.asarray(_quat, dtype=float))
            lines.append(f"ori   : {np.round(_ori, 2)}")
            recent = classifier.recent() if classifier is not None else None
            if classifier is None:
                lines.append("model : unavailable")
            elif recent is None:
                lines.append("model : (waiting for contact)")
            else:
                verdict = "CORRECT" if recent["label"] == truth else "WRONG"
                lines.append(f"model : {recent['name']}  ({recent['agreement'] * 100:.0f}% of {recent['n_votes']})")
                lines.append(f"status: {verdict}")
            if class_stats["n"]:
                lines.append(f"score : {class_stats['ok']}/{class_stats['n']} episodes correct")
            return "\n".join(lines)

        def refresh_overlay(truth_obj):
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT, overlay_text(truth_obj), None))

        interactive_frame = 0
        if tiles is not None or panel is not None or classifier is not None:
            sample_tactile()
        refresh_overlay(world.active)

        while viewer.is_running():
            frame_start = time.perf_counter()
            while not commands.empty():
                with viewer.lock():
                    if not _handle(commands.get(), world):
                        return
            if args.demo:
                if paused[0]:
                    viewer.sync()
                    time.sleep(0.02)
                    continue
                if episode is None:
                    with viewer.lock():
                        episode = HandoverEpisode(world, demo_policy, demo_rng, step_fn=paced_step)
                    if classifier is not None:
                        classifier.reset()
                    print(f"[demo] episode {n_episodes + 1}: {episode.obj_id}, roll {np.degrees(episode.info['roll_rad']):.0f} deg")
                finished = not episode.tick()
                if tiles is not None or panel is not None or classifier is not None:
                    raw, registers = sample_tactile()
                    if not finished and episode.phase != PHASE_APPROACH:
                        feed_classifier(raw, registers)  # one frame per control tick (20 Hz)
                refresh_overlay(episode.obj_id)
                if finished:
                    res = episode.result
                    n_episodes += 1
                    n_success += res["success"]
                    vote = classifier.majority_vote() if classifier is not None else None
                    if vote is not None:
                        correct = vote["label"] == OBJECT_TO_LABEL.get(episode.obj_id)
                        class_stats["n"] += 1
                        class_stats["ok"] += int(correct)
                        print(f"[demo]   classifier: {vote['name']} ({'CORRECT' if correct else 'WRONG'})"
                              f"  | {class_stats['ok']}/{class_stats['n']} episodes correct")
                    print(f"[demo]   {'OK' if res['success'] else 'FAILED (' + res['failure'] + ')'}: shift {res['shift_mm']} mm, "
                          f"rot {res['rot_deg']} deg  | {n_success}/{n_episodes} successful")
                    viewer.sync()
                    time.sleep(1.0)
                    episode = None
                continue
            if not paused[0]:
                with viewer.lock():
                    world.step(steps_per_frame)
                interactive_frame += 1
                if interactive_frame % 3 == 0:  # ~20 Hz sensor sampling while stepping at 60 Hz
                    if tiles is not None or panel is not None or classifier is not None:
                        sample_sensors()
                    refresh_overlay(world.active)
            viewer.sync()
            time.sleep(max(0.0, steps_per_frame * world.m.opt.timestep - (time.perf_counter() - frame_start)))


if __name__ == "__main__":
    main()
