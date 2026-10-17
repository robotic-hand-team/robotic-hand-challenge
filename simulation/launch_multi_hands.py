"""Launch N independent MuJoCo viewer windows, each running one hand in --demo mode.

Usage (from simulation/):
    ../venv/bin/python launch_multi_hands.py --hands 10
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE.parent / "venv"
MJPYTHON = VENV / "bin" / "mjpython"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heatmap", default="panel")
    args = ap.parse_args()

    procs = []
    for i in range(args.hands):
        seed = args.seed + i
        cmd = [str(MJPYTHON), "-m", "hand_sim.run", "--demo", "--seed", str(seed),
               "--heatmap", args.heatmap]
        print(f"[launcher] starting hand {i+1}/{args.hands} (seed {seed})")
        p = subprocess.Popen(cmd, cwd=str(HERE))
        procs.append(p)

    print(f"[launcher] {len(procs)} viewers running. Close all windows to exit.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()

if __name__ == "__main__":
    main()
