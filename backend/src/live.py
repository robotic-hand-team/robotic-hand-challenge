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
  sensor A/B     LIVE_SENSOR_MODEL "hold_mean" (default): final_model/sensor_classifier.pkl on the
                 running mean of the grasp's settled hold frames, one estimate per grasp;
                 "per_frame": xgb_sensor<LIVE_CLASSIFIER_SUFFIX>, majority vote over the grasp
  object         the real-grasp pose (twin.fit_real_grasp: against the palm, where the fingers at
                 the real hold angles touch it, plus the photo calibration LIVE_GRASP_OFFSET); the
                 sim-trained ridge model may shift it by POS_MAX_OFFSET_MM (0 by default). The twin's
                 fingers stop at the object's surface (LIVE_FINGER_CLAMP)
  slip           the GUI's own SlipDetector result
"""
import asyncio
import json
import threading
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from . import config as C
from .sessions import DOF, OBJECT_LONG_AXIS, HoldMeanSensorClassifier, SensorClassifier
from .twin import SC

LABEL_TO_OBJECT = {v: k for k, v in C.OBJECT_TO_LABEL.items()}
GRASP_STATES = {"CLOSING", "CONTACT", "PROBING", "HOLD"}
PATCHES = [("little_tip", 3, 3), ("little_nail", 12, 8), ("little_pad", 10, 8),
           ("ring_tip", 3, 3), ("ring_nail", 12, 8), ("ring_pad", 10, 8),
           ("middle_tip", 3, 3), ("middle_nail", 12, 8), ("middle_pad", 10, 8),
           ("index_tip", 3, 3), ("index_nail", 12, 8), ("index_pad", 10, 8),
           ("thumb_tip", 3, 3), ("thumb_nail", 12, 8), ("thumb_mid", 3, 3),
           ("thumb_pad", 12, 8), ("palm", 8, 14)]  # classifier/physics_pose.py


class LivePipeline:
    def __init__(self, twin):
        self.twin = twin
        self.root = twin.root_pos()
        try:
            self.clf = SensorClassifier(C.LIVE_CLASSIFIER_SUFFIX)
        except Exception as e:
            print(f"[live] classifier unavailable: {e}")
            self.clf = None
        self.hold_clf = None
        if C.LIVE_SENSOR_MODEL == "hold_mean":
            try:
                self.hold_clf = HoldMeanSensorClassifier()
                print(f"[live] sensor model: {self.hold_clf.name}")
            except Exception as e:
                print(f"[live] hold-mean model unavailable ({e}) – using the per-frame classifier")
        try:
            self.ridge = joblib.load(C.CLASSIFIER_DIR / "physics_pose_ridge.pkl")
            # saved with n_jobs=-1: every single-frame predict started worker processes (~2 s)
            self.ridge.set_params(n_jobs=1)
            self.ridge_scaler = joblib.load(C.CLASSIFIER_DIR / "scaler_physics_pose.pkl")
        except Exception as e:
            print(f"[live] pose model unavailable: {e}")
            self.ridge = None
        self.grasp_poses = {o: self._real_grasp_pose(o) for o in C.OBJECT_TO_LABEL}
        self.latest = None        # last processed frame for the dashboard
        self.last_rx = 0.0
        self.rate_hz = 0.0
        self.source = ""
        self.hand_side = C.LIVE_HAND_SIDE
        self.has_tactile = False
        self.state, self.state_since = None, 0.0  # hand GUI state and when it last changed
        self._reset_grasp()

    def _reset_grasp(self):
        self.votes, self.probs = [], []
        self.hold_sum, self.hold_n = None, 0   # hold_mean: running sum of the grasp's feature rows
        self.pre_rows = deque(maxlen=C.PRE_PROBE_FRAMES)   # last readings before probing
        self.probe_est = None        # {"p", "n", "t"}: estimate made at probing start
        self.pos = None
        self.pos_obj = None          # object the position history belongs to
        self.pos_hist = deque()      # (time, position) of the last POS_MEDIAN_S

    def _real_grasp_pose(self, obj: str) -> np.ndarray:
        """[x, y, z, qw, qx, qy, qz] of obj in the real hand's grasp: fitted to the palm and the real
        hold angles (twin.fit_real_grasp), plus the photo calibration LIVE_GRASP_OFFSET."""
        pose = self.twin.fit_real_grasp(obj, C.LIVE_HOLD_ANGLES)
        off = C.LIVE_GRASP_OFFSET.get(obj, {})
        pose[:3] += np.array(off.get("mm", [0, 0, 0]), float) / 1000.0
        if any(off.get("deg", [0, 0, 0])):
            w, x, y, z = pose[3:]
            r = Rotation.from_quat([x, y, z, w]) * Rotation.from_euler("xyz", off["deg"], degrees=True)
            qx, qy, qz, qw = r.as_quat()
            pose[3:] = [qw, qx, qy, qz]
        print(f"[live] {obj} real-grasp pose: pos {np.round(pose[:3] * 1000, 1)} mm, quat {np.round(pose[3:], 3)}")
        return pose

    def _held_position(self, obj: str, p, now: float):
        """Nominal grasp position of obj, shifted by the model's estimate p but by at most
        POS_MAX_OFFSET_MM, as the median over the last POS_MEDIAN_S: a held object doesn't move,
        and the sim-trained model is neither accurate nor steady on real tactile data."""
        if obj != self.pos_obj:      # the vote switched objects: their anchors differ
            self.pos_obj, self.pos_hist = obj, deque()
        anchor = self.grasp_poses[obj][:3].copy()
        if p is not None:
            offset = p - anchor
            limit = C.POS_MAX_OFFSET_MM / 1000.0
            dist = np.linalg.norm(offset)
            if dist > limit:
                offset *= limit / dist
            self.pos_hist.append((now, anchor + offset))
        while self.pos_hist and now - self.pos_hist[0][0] > C.POS_MEDIAN_S:
            self.pos_hist.popleft()
        if not self.pos_hist:
            return anchor
        return np.median(np.array([q for _, q in self.pos_hist]), axis=0)

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
        row = {f"angle_act_{d}": v for d, v in zip(DOF, act["angle_act"])}
        for g, s in tac.items():
            for stat in ("sum", "max", "contacts", "cy", "cx", "force"):
                row[f"{g}_{stat}"] = 0.0 if s.get(stat) is None else s[stat]
        if state != self.state:
            # a new Close (also straight from HOLD) is a new grasp – not the CLOSING after probing
            if state == "CLOSING" and self.state != "PROBING":
                self._reset_grasp()
            if state == "PROBING" and self.hold_clf and len(self.pre_rows) >= C.HOLD_MEAN_MIN_FRAMES:
                p = self.hold_clf.p_sensor_b(np.mean(self.pre_rows, axis=0))
                self.probe_est = {"p": p, "n": len(self.pre_rows), "t": now}
                print(f"[live] probing started: P(sensor B) = {p:.3f} from the last "
                      f"{len(self.pre_rows)} readings, shown in {C.PROBE_ESTIMATE_DELAY_S:g} s")
            self.state, self.state_since = state, now
        if self.hold_clf and state in ("CLOSING", "CONTACT") and contacts > C.MIN_CONTACTS_ACTIVE:
            self.pre_rows.append(self.hold_clf.vector(row))
        # predict only on a settled grasp: HOLD for at least PREDICT_AFTER_HOLD_S. With the hand
        # open, a few loaded taxels (sensor drift, the palm resting on something) must not produce
        # an estimate, and while closing the tactile picture is not the final grasp yet.
        settled = not state or (state == "HOLD" and now - self.state_since >= C.PREDICT_AFTER_HOLD_S)
        active = settled and contacts > C.MIN_CONTACTS_ACTIVE   # no state in the message: old behaviour

        if state and state not in GRASP_STATES:
            self._reset_grasp()  # new grasp, new vote

        estimate, p_b = None, None
        if self.hold_clf and self.probe_est:
            # probed grasp: the estimate from probing start, shown PROBE_ESTIMATE_DELAY_S later
            if now - self.probe_est["t"] >= C.PROBE_ESTIMATE_DELAY_S:
                p_b = self.probe_est["p"]
                label = "B" if p_b >= 0.5 else "A"
                estimate = {"label": label, "confidence": round(p_b if label == "B" else 1 - p_b, 3),
                            "agreement": None, "n_votes": self.probe_est["n"]}
        elif self.hold_clf:
            # one estimate per grasp from the mean over its settled hold frames (as trained)
            if active:
                v = self.hold_clf.vector(row)
                self.hold_sum = v if self.hold_sum is None else self.hold_sum + v
                self.hold_n += 1
            if self.hold_n >= C.HOLD_MEAN_MIN_FRAMES:
                p_b = self.hold_clf.p_sensor_b(self.hold_sum / self.hold_n)
                label = "B" if p_b >= 0.5 else "A"
                estimate = {"label": label, "confidence": round(p_b if label == "B" else 1 - p_b, 3),
                            "agreement": None, "n_votes": self.hold_n}
            if not active:
                p_b = None
        else:
            p_b = self._classify(row) if active else None
            if p_b is not None:
                self.votes.append(p_b >= 0.5)
                self.probs.append(p_b)

        if not self.hold_clf and self.votes:
            n_b = int(sum(self.votes))
            label = "B" if n_b * 2 > len(self.votes) else "A"
            conf = float(np.mean([q if label == "B" else 1 - q for q in self.probs]))
            estimate = {"label": label, "confidence": round(conf, 3),
                        "agreement": round(max(n_b, len(self.votes) - n_b) / len(self.votes), 3),
                        "n_votes": len(self.votes)}

        # a probed grasp's estimate is shown during probing too, as long as the object is held
        if self.probe_est and estimate and contacts > C.MIN_CONTACTS_ACTIVE:
            active = True
        obj, obj_pose, pos_rel, axis = None, None, None, None
        if estimate and active:
            obj = LABEL_TO_OBJECT[estimate["label"]]
            shift = self._position(tac) if C.POS_MAX_OFFSET_MM > 0 else None   # unused at 0 mm
            self.pos = self._held_position(obj, shift, now)
            if self.pos is not None:
                q = self.grasp_poses[obj][3:]  # real-grasp orientation (w, x, y, z)
                obj_pose = np.array([*self.pos, *q])
                axis = Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply(OBJECT_LONG_AXIS)
                pos_rel = (self.pos - self.root) * 1000
                if self.hand_side == "left":  # report in the left hand's own (mirrored) frame
                    pos_rel[1] *= -1
                    axis[1] *= -1

        bodies = (self.twin.poses(angles, obj, obj_pose, clamp=C.LIVE_FINGER_CLAMP)
                  if obj_pose is not None else self.twin.poses(angles))
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
                "tilt_deg": None, "orientation": "nominal",
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
                "classifier": (self.hold_clf.name if self.hold_clf else
                               f"xgb_sensor{C.LIVE_CLASSIFIER_SUFFIX}" if self.clf else None),
                "pose_model": "physics_pose_ridge" if self.ridge else None}


class _Udp(asyncio.DatagramProtocol):
    """Receives on the event loop, processes in a worker thread – always the newest datagram.
    Processing in datagram_received meant that whenever it was slower than the hand's ~40 Hz, the
    datagrams queued up in the socket buffer and the twin lagged seconds behind the real hand;
    now older frames are dropped, and the event loop stays free for the WebSocket."""

    def __init__(self, pipeline: LivePipeline):
        self.pipeline = pipeline
        self.pending = None
        self.lock = threading.Lock()
        self.wake = threading.Event()
        threading.Thread(target=self._work, daemon=True).start()

    def datagram_received(self, data, addr):
        with self.lock:
            self.pending = (data, addr)
        self.wake.set()

    def _work(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            with self.lock:
                item, self.pending = self.pending, None
            if item is None:
                continue
            data, addr = item
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
