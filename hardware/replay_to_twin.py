"""Replays a recorded hand session as the hand GUI's UDP broadcast (config.py, broadcast()).

Tests the live digital twin without the hand:
    python hardware/replay_to_twin.py                      # newest session in models/dataset
    python hardware/replay_to_twin.py 20261016_204513 --loop
    python hardware/replay_to_twin.py --no-tactile         # exactly what broadcast() sends today

Besides broadcast()'s fields it sends the optional ones the dashboard uses (force_act, tactile
summary, hand_side), unless --no-tactile.
"""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = Path(__file__).resolve().parents[1] / "models" / "dataset"
DOF = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
ANGLE_DEG = [(20, 176)] * 4 + [(70, -13), (90, 165)]  # as in config.py


def summary(m, threshold):
    """DataLogger / contact_info: sum, max, contacts, centroid and mean force of one group."""
    active = m > threshold
    n = int(active.sum())
    out = {"sum": int(m.sum()), "max": int(m.max()), "contacts": n, "cy": None, "cx": None, "force": 0.0}
    if n >= 2:
        rows, cols = np.where(active)
        out.update(cy=round(float(rows.mean()), 2), cx=round(float(cols.mean()), 2),
                   force=round(float(m[active].mean()), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", help="folder name in models/dataset (default: newest)")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    # 127.0.0.1: Windows drops a 255.255.255.255 broadcast to this PC unless the firewall allows it
    ap.add_argument("--host", default="127.0.0.1", help="as BROADCAST_HOST in config.py")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--no-tactile", action="store_true", help="only broadcast()'s current fields")
    ap.add_argument("--hand", choices=("left", "right"), help="default: from the recording's IP")
    args = ap.parse_args()

    sessions = sorted(p for p in DATASET.iterdir() if (p / "meta.json").is_file())
    folder = DATASET / args.session if args.session else sessions[-1]
    meta = json.loads((folder / "meta.json").read_text())
    act = pd.read_csv(folder / "actuators.csv")
    raw = pd.read_csv(folder / "tactile_raw.csv")
    groups = [(g["name"], g["rows"], g["cols"]) for g in meta["sensor_groups"]]
    threshold = meta.get("contact_threshold", 30)
    # 192.168.123.211 = the right hand (all of models/dataset); 192.168.123.210 = the left hand
    hand_side = args.hand or ("right" if meta.get("ip") == "192.168.123.211" else "left")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"replaying {folder.name} ({len(act)} frames, {hand_side} hand) to udp://{args.host}:{args.port}")

    seq = 0
    while True:
        t_prev = None
        for i in range(len(act)):
            a, r = act.iloc[i], raw.iloc[i]
            angles = [int(a[f"angle_act_{d}"]) for d in DOF]
            tactile = {}
            for name, h, w in groups:
                key = name.lower().replace(" ", "_")
                m = np.array([r[f"{key}_r{y}_c{x}"] for y in range(h) for x in range(w)], float).reshape(h, w)
                tactile[key] = summary(m, threshold)
            contacts = sum(s["contacts"] for s in tactile.values())
            closing = any(int(a[f"status_{d}"]) == 1 for d in DOF)
            state = "HOLD" if contacts > 20 else "CLOSING" if closing else "OPENED"  # rough stand-in
            msg = {
                "t": round(time.time(), 3), "seq": seq, "dof": DOF, "angle": angles,
                "angle_deg": [round(d0 + (d1 - d0) * x / 1000, 1) for x, (d0, d1) in zip(angles, ANGLE_DEG)],
                "state": state, "slip": {"slipping": False, "status": "", "reasons": []}, "frozen": False,
            }
            if not args.no_tactile:
                msg.update(force_act=[int(a[f"force_act_{d}"]) for d in DOF], tactile=tactile,
                           hand_side=hand_side, ip=f"replay:{folder.name}")
            sock.sendto(json.dumps(msg).encode(), (args.host, args.port))
            seq += 1
            t = float(a["t_rel"])
            if t_prev is not None:
                time.sleep(max(0.0, (t - t_prev) / args.speed))
            t_prev = t
        if not args.loop:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
