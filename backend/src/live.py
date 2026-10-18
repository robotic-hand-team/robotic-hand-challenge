"""Live digital twin: the real hand GUI's UDP broadcast (hardware/config.py, broadcast()) -> twin
poses, sensor estimate and KPIs, pushed to the dashboard over a WebSocket.

Broadcast message (one JSON datagram per acquisition frame):
  t, seq, dof, angle [6, 0-1000], angle_deg, state, slip {slipping, status, reasons}, frozen
Optional fields the dashboard uses when present:
  force_act [6]          grip force KPI
  tactile {group: {sum, max, contacts, cy, cx, force}}   (DataLogger summary, group = "little_tip")
                         -> sensor A/B estimate and object position; without it only the hand moves
  hand_side "left"/"right", ip

Per frame:
  hand pose      forward kinematics of the MuJoCo model from angle_act (always the right-hand model;
                 the dashboard mirrors it for the left hand, like simulation/hand_sim/left_hand.py)
  sensor A/B     classifier (xgb_sensor<LIVE_CLASSIFIER_SUFFIX>), majority vote over the current grasp
  object         position from classifier/model/physics_pose_ridge.pkl (sim-trained), smoothed and
                 clamped to the grasp workspace (~11 mm median error on sim hold frames). That estimate
                 knows nothing about the fingers, so the 3D pose is settled with physics (settle.py):
                 starting near it, contacts push the object out of the measured finger pose and tilt it
                 to fit. The ridge model's orientation output is unusable (~120 deg median error), so
                 orientation comes only from the settle, pulled toward the nominal grasp orientation.
                 The settled pose is held through short contact dropouts (HOLD_POSE_S).
  slip           the GUI's own SlipDetector result
"""
import asyncio
import json
import threading
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from . import config as C
from .sessions import DOF, OBJECT_LONG_AXIS, SensorClassifier
from .settle import PoseSettler, clamp_to_workspace
from .twin import SC

LABEL_TO_OBJECT = {v: k for k, v in C.OBJECT_TO_LABEL.items()}
GRASP_STATES = {"CLOSING", "CONTACT", "HOLD"}
PATCHES = [("little_tip", 3, 3), ("little_nail", 12, 8), ("little_pad", 10, 8),
           ("ring_tip", 3, 3), ("ring_nail", 12, 8), ("ring_pad", 10, 8),
           ("middle_tip", 3, 3), ("middle_nail", 12, 8), ("middle_pad", 10, 8),
           ("index_tip", 3, 3), ("index_nail", 12, 8), ("index_pad", 10, 8),
           ("thumb_tip", 3, 3), ("thumb_nail", 12, 8), ("thumb_mid", 3, 3),
           ("thumb_pad", 12, 8), ("palm", 8, 14)]  # classifier/physics_pose.py
POS_SMOOTHING = 0.15  # EMA weight of a new position estimate
HOLD_POSE_S = 1.0     # keep showing the last settled pose this long when contact briefly drops out


class LivePipeline:
    def __init__(self, twin):
        self.twin = twin
        self.root = twin.root_pos()
        try:
            self.clf = SensorClassifier(C.LIVE_CLASSIFIER_SUFFIX)
        except Exception as e:
            print(f"[live] classifier unavailable: {e}")
            self.clf = None
        try:
            self.ridge = joblib.load(C.CLASSIFIER_DIR / "physics_pose_ridge.pkl")
            self.ridge_scaler = joblib.load(C.CLASSIFIER_DIR / "scaler_physics_pose.pkl")
        except Exception as e:
            print(f"[live] pose model unavailable: {e}")
            self.ridge = None
        try:
            self.settler = PoseSettler(twin)
        except Exception as e:
            print(f"[live] pose settle unavailable, showing the raw estimate: {e}")
            self.settler = None
        self.latest = None        # last processed frame for the dashboard
        self.last_rx = 0.0
        self.rate_hz = 0.0
        self.source = ""
        self.hand_side = C.LIVE_HAND_SIDE
        self.has_tactile = False
        self._reset_grasp()

    def _reset_grasp(self):
        self.votes, self.probs = [], []
        self.pos = None
        self.settled, self.settled_obj, self.settled_t = None, None, 0.0

    # --- models -------------------------------------------------------------------------------
    def _classify(self, row: dict):
        if not self.clf:
            return None
        p_b = float(self.clf.predict(pd.DataFrame([row]))[0])
        return p_b

    def _position(self, tac: dict):
        if not self.ridge:
            return None
        feats = []
        for name, max_r, max_c in PATCHES:
            s = tac.get(name) or {}
            active = s.get("contacts", 0) > 0
            cy, cx = s.get("cy"), s.get("cx")
            feats += [cy / max_r if active and cy is not None else 0.0,
                      cx / max_c if active and cx is not None else 0.0,
                      s.get("force", 0.0) if active else 0.0]
        y = self.ridge.predict(self.ridge_scaler.transform(np.array([feats])))[0]
        return y[:3] / 1000.0  # world frame of the right-hand simulation, metres

    # --- per frame ----------------------------------------------------------------------------
    def process(self, msg: dict, addr) -> dict:
        now = time.time()
        if self.last_rx:
            self.rate_hz = 0.85 * self.rate_hz + 0.15 / max(1e-3, now - self.last_rx)
        self.last_rx = now
        self.source = msg.get("ip") or addr[0]
        self.hand_side = msg.get("hand_side") or C.LIVE_HAND_SIDE

        angle_act = [int(a) for a in msg["angle"]]
        force_act = [int(f) for f in msg.get("force_act") or []]
        tac = msg.get("tactile") or {}
        self.has_tactile = bool(tac)
        state = msg.get("state", "")
        slip = msg.get("slip") or {}
        act = {"angle_act": angle_act, "force_act": force_act or [0] * 6}
        angles = np.array(angle_act, float)
        contacts = sum(s["contacts"] for s in tac.values())
        active = contacts > C.MIN_CONTACTS_ACTIVE

        if state and state not in GRASP_STATES:
            self._reset_grasp()  # new grasp, new vote

        row = {f"angle_act_{d}": v for d, v in zip(DOF, act["angle_act"])}
        for g, s in tac.items():
            for stat in ("sum", "max", "contacts", "cy", "cx", "force"):
                row[f"{g}_{stat}"] = 0.0 if s.get(stat) is None else s[stat]
        p_b = self._classify(row) if active else None
        if p_b is not None:
            self.votes.append(p_b >= 0.5)
            self.probs.append(p_b)

        estimate = None
        if self.votes:
            n_b = int(sum(self.votes))
            label = "B" if n_b * 2 > len(self.votes) else "A"
            conf = float(np.mean([q if label == "B" else 1 - q for q in self.probs]))
            estimate = {"label": label, "confidence": round(conf, 3),
                        "agreement": round(max(n_b, len(self.votes) - n_b) / len(self.votes), 3),
                        "n_votes": len(self.votes)}

        obj, obj_pose, pos_rel, axis, tilt = None, None, None, None, None
        if estimate:
            obj = LABEL_TO_OBJECT[estimate["label"]]
            if active:
                p = self._position(tac)
                if p is not None:
                    self.pos = p if self.pos is None else (1 - POS_SMOOTHING) * self.pos + POS_SMOOTHING * p
            if active and self.pos is not None:
                target = clamp_to_workspace(self.pos, obj)
                q_nom = SC.OBJECTS[obj]["grasp_quat"]  # nominal grasp orientation (w, x, y, z)
                if self.settler:
                    start = self.settled if self.settled_obj == obj else None
                    obj_pose = self.settler.settle(angles, obj, target, start)
                else:
                    obj_pose = np.array([*target, *q_nom])
                self.settled, self.settled_obj, self.settled_t = obj_pose, obj, now
            elif self.settled_obj == obj and now - self.settled_t < HOLD_POSE_S:
                obj_pose = self.settled  # contact dropped out for a moment: hold, don't jump or vanish
            if obj_pose is None:
                obj = None
            else:
                # The position KPI is the smoothed, clamped estimate: the settle trades ~6 mm of accuracy
                # for a pose that does not overlap the fingers, so it drives the 3D view only.
                pos_rel = (clamp_to_workspace(self.pos, obj) - self.root) * 1000
                q, q_nom = obj_pose[3:], SC.OBJECTS[obj]["grasp_quat"]
                axis = Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply(OBJECT_LONG_AXIS)
                axis_nom = Rotation.from_quat([q_nom[1], q_nom[2], q_nom[3], q_nom[0]]).apply(OBJECT_LONG_AXIS)
                tilt = float(np.degrees(np.arccos(np.clip(axis @ axis_nom, -1, 1))))
                if self.hand_side == "left":  # report in the left hand's own (mirrored) frame
                    pos_rel[1] *= -1
                    axis[1] *= -1

        bodies = self.twin.poses(angles, obj, obj_pose) if obj_pose is not None else self.twin.poses(angles)
        self.latest = {
            "live": True, "seq": msg.get("seq"), "t": msg.get("t"), "phase": state or "LIVE",
            "object": obj, "bodies": bodies, "angles": angle_act, "force_g": force_act or None,
            "tactile": {g: s["sum"] for g, s in tac.items()}, "has_tactile": self.has_tactile,
            "contacts": int(contacts) if tac else None, "in_hand": bool(active),
            "frozen": bool(msg.get("frozen")),
            "kpi": {
                "sensor": estimate, "p_sensor_b": None if p_b is None else round(p_b, 3),
                "pos_mm": None if pos_rel is None else np.round(pos_rel, 1).tolist(),
                "axis": None if axis is None else np.round(axis, 3).tolist(),
                "tilt_deg": None if tilt is None else round(tilt, 1),
                "orientation": "settled" if self.settler and tilt is not None else "nominal",
                "grip_force_g": int(sum(force_act[:5])) if force_act else None,
                "speed_mm_s": None,
                "slipping": bool(slip.get("slipping")), "slip_text": slip.get("status", ""),
            },
        }
        return self.latest

    def status(self) -> dict:
        age = time.time() - self.last_rx if self.last_rx else None
        return {"connected": age is not None and age < 1.5, "age_s": None if age is None else round(age, 2),
                "rate_hz": round(self.rate_hz, 1), "source": self.source, "hand_side": self.hand_side,
                "udp": f"{C.LIVE_UDP_HOST}:{C.LIVE_UDP_PORT}", "has_tactile": self.has_tactile,
                "classifier": f"xgb_sensor{C.LIVE_CLASSIFIER_SUFFIX}" if self.clf else None,
                "pose_model": "physics_pose_ridge" if self.ridge else None}


class _Udp(asyncio.DatagramProtocol):
    """Hands datagrams to a worker thread that always processes the newest one. The physics settle takes
    5-50 ms, so running it here would stall the event loop (WebSocket, replay endpoints) and let frames
    queue up behind it; stale frames are dropped instead (mj_step releases the GIL)."""

    def __init__(self, pipeline: LivePipeline):
        self.pipeline = pipeline
        self.pending = None
        self.cond = threading.Condition()
        threading.Thread(target=self._worker, name="live-pipeline", daemon=True).start()

    def datagram_received(self, data, addr):
        with self.cond:
            self.pending = (data, addr)
            self.cond.notify()

    def _worker(self):
        while True:
            with self.cond:
                while self.pending is None:
                    self.cond.wait()
                data, addr = self.pending
                self.pending = None
            try:
                self.pipeline.process(json.loads(data), addr)
            except Exception as e:  # a malformed frame must not stop the listener
                print(f"[live] bad frame from {addr}: {e}")


async def start_udp(pipeline: LivePipeline):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _Udp(pipeline), local_addr=(C.LIVE_UDP_HOST, C.LIVE_UDP_PORT))
    print(f"[live] listening for hand frames on udp://{C.LIVE_UDP_HOST}:{C.LIVE_UDP_PORT}")
    return transport
