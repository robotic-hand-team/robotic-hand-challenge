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
  sensor A/B     LIVE_SENSOR_MODEL "left_hand" (default): final_model/left_hand_models.pkl on the
                 running mean of the grasp's settled hold frames, one estimate per grasp; "hold_mean":
                 final_model/sensor_classifier.pkl, same averaging; "per_frame": xgb_sensor<suffix>,
                 majority vote over the grasp
  object         "left_hand": turned in the palm so its yellow end points the estimated direction, its
                 grip point on the fist centre, resting on the palm (twin.pose_in_palm). Other models:
                 the nominal real-grasp pose (twin.fit_real_grasp plus LIVE_GRASP_OFFSET); the sim-trained
                 ridge model may shift it by POS_MAX_OFFSET_MM (0 by default). The twin's fingers stop at
                 the object's surface (LIVE_FINGER_CLAMP)
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
from .palm_axis import fuse, palm_line, palm_point, reliable
from .sessions import DOF, OBJECT_LONG_AXIS, HoldMeanSensorClassifier, LeftHandModels, SensorClassifier
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
        if C.LIVE_SENSOR_MODEL == "left_hand":
            try:
                self.hold_clf = LeftHandModels()
                print(f"[live] sensor model: {self.hold_clf.name}")
                print(f"[live] direction model: {self.hold_clf.direction_name}")
            except Exception as e:
                print(f"[live] left-hand models unavailable ({e}) – trying the hold-mean classifier")
        if self.hold_clf is None and C.LIVE_SENSOR_MODEL in ("left_hand", "hold_mean"):
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
        self.fist = np.array(C.LIVE_FIST_CENTER_MM, float) / 1000.0   # fist centre (y, z), metres
        self._pose_cache = {}     # (object, direction in 2 deg steps) -> pose, see _directed_pose
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
        self.decision = None         # (estimate, p_sensor_b, direction_deg | None): fixed once made
        self.quiet_since = None      # stateless streams: when contact was last lost
        self.hold_sum, self.hold_n = None, 0   # hold_mean: running sum of the grasp's feature rows
        self.palm_sum, self.palm_n = None, 0   # the same frames' palm taxels (message field "palm")
        self.pre_rows = deque(maxlen=C.PRE_PROBE_FRAMES)   # last readings before probing
        self.pre_palm = deque(maxlen=C.PRE_PROBE_FRAMES)
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

    def _direction(self, mean_vector, palm_mean=None):
        """(direction of the yellow end in the palm plane in deg, source, palm point (y, z) mm | None)
        when the sensor model provides a direction, else (None, None, None). The axis line comes from the
        palm's pressure strip when it shows a clear one (palm_axis.fuse), the end from the model; without
        a clear strip, both from the model. The palm point is where the strip lies (palm_axis.palm_point)."""
        f = getattr(self.hold_clf, "direction_deg", None)
        if not f:
            return None, None, None
        model = f(mean_vector)
        if palm_mean is None:
            return model, "model", None
        point = palm_point(palm_mean, C.PALM_SENSOR_CENTER_MM, C.PALM_SENSOR_PITCH_MM)
        line, elong, total = palm_line(palm_mean)
        if not (np.isfinite(line) and reliable(elong, total)):
            return model, "model", point
        return fuse(model, line, elong, total, self.hold_clf.palm_bias), "palm", point

    def _directed_pose(self, obj: str, theta: float, point_mm=None) -> np.ndarray:
        """obj turned in the palm so its yellow end points theta (twin.pose_in_palm). Its axis runs over the
        palm point when there is one, with the grip point as close to the fist centre as that allows; else
        through the fist centre. Cached in 2 deg / 2 mm steps: the palm fit runs once per placement."""
        grip_at = self.fist
        if point_mm is not None:
            p = np.asarray(point_mm, float) / 1000.0
            u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
            grip_at = p + np.dot(self.fist - p, u) * u
        key = (obj, int(round(theta / 2.0)) * 2, *np.round(grip_at * 500).astype(int))   # 2 mm steps
        if key not in self._pose_cache:
            self._pose_cache[key] = self.twin.pose_in_palm(
                obj, key[1], self.grasp_poses[obj], C.LIVE_NOMINAL_DIRECTION_DEG[obj],
                C.LIVE_GRIP_POINT[obj], np.array(key[2:]) / 500.0)
        return self._pose_cache[key]

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
            # pre_rows holds at most PRE_PROBE_FRAMES readings, so that is the most a probe can wait for
            if state == "PROBING" and self.hold_clf and \
                    len(self.pre_rows) >= min(C.HOLD_MEAN_MIN_FRAMES, C.PRE_PROBE_FRAMES):
                pre = np.mean(self.pre_rows, axis=0)
                p = self.hold_clf.p_sensor_b(pre)
                pre_palm = np.mean(self.pre_palm, axis=0) if len(self.pre_palm) == len(self.pre_rows) else None
                self.probe_est = {"p": p, "n": len(self.pre_rows), "t": now,
                                  "theta": self._direction(pre, pre_palm)}
                self.decision = None   # a probed grasp is decided by this estimate, not the hold mean
                print(f"[live] probing started: P(sensor B) = {p:.3f} from the last "
                      f"{len(self.pre_rows)} readings, shown in {C.PROBE_ESTIMATE_DELAY_S:g} s")
            self.state, self.state_since = state, now
        palm = msg.get("palm")
        palm = np.asarray(palm, float) if palm is not None and len(palm) == 112 else None
        if self.hold_clf and state in ("CLOSING", "CONTACT") and contacts > C.MIN_CONTACTS_ACTIVE:
            self.pre_rows.append(self.hold_clf.vector(row))
            if palm is not None:
                self.pre_palm.append(palm)
        # predict only on a settled grasp: HOLD for at least PREDICT_AFTER_HOLD_S. With the hand
        # open, a few loaded taxels (sensor drift, the palm resting on something) must not produce
        # an estimate, and while closing the tactile picture is not the final grasp yet.
        settled = not state or (state == "HOLD" and now - self.state_since >= C.PREDICT_AFTER_HOLD_S)
        active = settled and contacts > C.MIN_CONTACTS_ACTIVE   # no state in the message: old behaviour

        if state and state not in GRASP_STATES:
            self._reset_grasp()  # new grasp, new vote
        elif not state:          # no state in the message: a grasp ends when contact has been gone a while
            if contacts > C.MIN_CONTACTS_ACTIVE:
                self.quiet_since = None
            elif self.quiet_since is None:
                self.quiet_since = now
            elif now - self.quiet_since >= C.GRASP_END_S and self.decision:
                self._reset_grasp()

        # One decision per grasp. Until HOLD_MEAN_MIN_FRAMES settled hold frames are in nothing is
        # shown ("analysing"); the answer made then is fixed until the grasp ends, not re-voted per frame.
        estimate, p_b, progress = None, None, 0.0
        if self.hold_clf and self.probe_est:
            # probed grasp: the estimate from probing start, shown PROBE_ESTIMATE_DELAY_S later
            progress = min(1.0, (now - self.probe_est["t"]) / C.PROBE_ESTIMATE_DELAY_S)
            if self.decision is None and progress >= 1.0:
                p = self.probe_est["p"]
                label = "B" if p >= 0.5 else "A"
                self.decision = ({"label": label, "confidence": round(p if label == "B" else 1 - p, 3),
                                  "agreement": None, "n_votes": self.probe_est["n"]}, p, self.probe_est["theta"])
        elif self.hold_clf:
            # one estimate per grasp from the mean over its settled hold frames (as trained)
            if self.decision is None:
                if active:
                    v = self.hold_clf.vector(row)
                    self.hold_sum = v if self.hold_sum is None else self.hold_sum + v
                    self.hold_n += 1
                    if palm is not None:
                        self.palm_sum = palm if self.palm_sum is None else self.palm_sum + palm
                        self.palm_n += 1
                progress = min(1.0, self.hold_n / C.HOLD_MEAN_MIN_FRAMES)
                if self.hold_n >= C.HOLD_MEAN_MIN_FRAMES:
                    mean = self.hold_sum / self.hold_n
                    palm_mean = self.palm_sum / self.palm_n if self.palm_n == self.hold_n else None
                    p = self.hold_clf.p_sensor_b(mean)
                    label = "B" if p >= 0.5 else "A"
                    self.decision = ({"label": label, "confidence": round(p if label == "B" else 1 - p, 3),
                                      "agreement": None, "n_votes": self.hold_n}, p,
                                     self._direction(mean, palm_mean))
        else:
            p_b = self._classify(row) if active else None   # per-frame value, for the confidence chart
            if self.decision is None:
                if p_b is not None:
                    self.votes.append(p_b >= 0.5)
                    self.probs.append(p_b)
                progress = min(1.0, len(self.votes) / C.HOLD_MEAN_MIN_FRAMES)
                if len(self.votes) >= C.HOLD_MEAN_MIN_FRAMES:
                    n_b = int(sum(self.votes))
                    label = "B" if n_b * 2 > len(self.votes) else "A"
                    conf = float(np.mean([q if label == "B" else 1 - q for q in self.probs]))
                    self.decision = ({"label": label, "confidence": round(conf, 3),
                                      "agreement": round(max(n_b, len(self.votes) - n_b) / len(self.votes), 3),
                                      "n_votes": len(self.votes)}, p_b, (None, None, None))
        theta, theta_source, palm_pt = None, None, None
        if self.decision:
            estimate, progress = self.decision[0], 1.0
            theta, theta_source, palm_pt = self.decision[2]
            if self.hold_clf and (self.probe_est or active):
                p_b = self.decision[1]

        # a probed grasp's estimate is shown during probing too, as long as the object is held
        if self.probe_est and estimate and contacts > C.MIN_CONTACTS_ACTIVE:
            active = True
        in_grasp = (state in GRASP_STATES) if state else contacts > C.MIN_CONTACTS_ACTIVE
        obj, obj_pose, pos_rel, axis = None, None, None, None
        if estimate and active:
            obj = LABEL_TO_OBJECT[estimate["label"]]
            if theta is not None:
                # turned in the palm so the yellow end points the estimated way (left_hand direction model)
                obj_pose = self._directed_pose(obj, theta, palm_pt)
                self.pos = obj_pose[:3]
            else:
                shift = self._position(tac) if C.POS_MAX_OFFSET_MM > 0 else None   # unused at 0 mm
                self.pos = self._held_position(obj, shift, now)
                if self.pos is not None:
                    obj_pose = np.array([*self.pos, *self.grasp_poses[obj][3:]])  # real-grasp orientation
            if obj_pose is not None:
                q = obj_pose[3:]  # (w, x, y, z)
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
                "sensor_state": ("final" if estimate else "analysing" if in_grasp else "idle"),
                "analysis_progress": round(progress, 3),
                "pos_mm": None if pos_rel is None else np.round(pos_rel, 1).tolist(),
                "axis": None if axis is None else np.round(axis, 3).tolist(),
                **self._orientation_kpi(theta if obj_pose is not None else None),
                "direction_source": theta_source if obj_pose is not None else None,
                "position_source": ("palm" if palm_pt is not None else "fist centre") if obj_pose is not None else None,
                "grip_force_g": int(sum(force_act[:5])) if force_act else None,
                "speed_mm_s": None,
                "slipping": bool(slip.get("slipping")), "slip_text": slip.get("status", ""),
            },
        }
        return self.latest

    @staticmethod
    def _orientation_kpi(theta) -> dict:
        """Orientation KPIs. theta: where the yellow end points in the palm plane (0 deg = the thumb,
        90 = the fingertips), or None (no direction estimate: the object keeps its nominal orientation).
        tilt_deg = angle of the long axis against the across-the-palm line, 0-90."""
        if theta is None:
            return {"tilt_deg": None, "orientation": "nominal", "direction_deg": None, "direction_text": None}
        t = (theta + 180.0) % 360.0 - 180.0
        tilt = abs((t + 90.0) % 180.0 - 90.0)
        side = "thumb" if abs(t) < 90 else "little finger"
        if tilt >= 60:
            side = "fingertips" if t > 0 else "wrist"
        return {"tilt_deg": round(tilt, 1), "orientation": "estimated", "direction_deg": round(t, 1),
                "direction_text": f"yellow end towards the {side}"}

    def status(self) -> dict:
        age = time.time() - self.last_rx if self.last_rx else None
        return {"connected": age is not None and age < 1.5, "age_s": None if age is None else round(age, 2),
                "rate_hz": round(self.rate_hz, 1), "source": self.source, "hand_side": self.hand_side,
                "udp": f"{C.LIVE_UDP_HOST}:{C.LIVE_UDP_PORT}", "has_tactile": self.has_tactile,
                "classifier": (self.hold_clf.name if self.hold_clf else
                               f"xgb_sensor{C.LIVE_CLASSIFIER_SUFFIX}" if self.clf else None),
                "pose_model": (getattr(self.hold_clf, "direction_name", None)
                               or ("physics_pose_ridge" if self.ridge else None))}


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
