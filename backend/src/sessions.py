"""Session data for the dashboard: mock sessions from models/dataset_sim, sensor classifier, KPIs."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from . import config as C
from .twin import SC

DOF = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
META_COLS = {"frame", "t_unix", "t_rel"}
OBJECT_LONG_AXIS = np.array([0.0, 1.0, 0.0])  # hand_sim.config: long axis = object y


@lru_cache(maxsize=1)
def list_sessions() -> list[dict]:
    out = []
    for folder in sorted(C.DATASET_DIR.iterdir()):
        meta_file = folder / "meta.json"
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text())
        info = meta.get("episode_info", {})
        obj = meta.get("object") or info.get("object")
        out.append({"id": folder.name, "object": obj, "label": C.OBJECT_TO_LABEL.get(obj),
                    "success": info.get("success"), "start": meta.get("start")})
    return out


class SensorClassifier:
    def __init__(self, suffix: str = C.CLASSIFIER_SUFFIX):
        self.suffix = suffix
        self.model = joblib.load(C.CLASSIFIER_DIR / f"xgb_sensor{suffix}.pkl")
        self.scaler = joblib.load(C.CLASSIFIER_DIR / f"scaler{suffix}.pkl")
        self.features = joblib.load(C.CLASSIFIER_DIR / f"feature_cols{suffix}.pkl")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P(Sensor B) per frame, same feature handling as classifier/predict.py."""
        X = df.reindex(columns=self.features).fillna(0.0).to_numpy(float)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


class HoldMeanSensorClassifier:
    """final_model/sensor_classifier.pkl: one estimate per grasp from the mean of its features over
    the hold phase (same features and NaN -> 0 handling as final_model/match_session.py)."""

    def __init__(self, path=C.HOLD_MEAN_MODEL):
        import pickle
        import warnings
        with warnings.catch_warnings(record=True) as caught:   # saved with newer sklearn/xgboost
            warnings.simplefilter("always")
            with open(path, "rb") as f:
                bundle = pickle.load(f)
        if caught:
            print(f"[backend] {path.name}: saved with other library versions "
                  f"({len(caught)} warnings) – re-export it if the estimates look wrong")
        self.model, self.scaler = bundle["xgb"], bundle["scaler"]
        self.features = list(bundle["feature_cols"])
        self.index_b = list(bundle["label_encoder"].classes_).index("sensor_B")
        self.name = f"final_model/{path.name} (hold mean, LOO {bundle.get('loo_accuracy', '?')})"

    def vector(self, row: dict) -> np.ndarray:
        """One frame's features in model order (missing / None -> 0)."""
        return np.array([row.get(c) or 0.0 for c in self.features], float)

    def p_sensor_b(self, mean_vector: np.ndarray) -> float:
        # float32 like match_session.py: some trees split right at a training session's value, and
        # float64 vs float32 means flipped 20261017_202049 from p_B 0.33 to 0.84
        X = self.scaler.transform(mean_vector.astype(np.float32).reshape(1, -1))
        return float(self.model.predict_proba(X)[0, self.index_b])


class LeftHandModels:
    """final_model/left_hand_models.pkl (classifier/train_left_hand.py): random forests trained on the live
    left hand's labelled grasps (models/dataset_left), on the same hold-mean vector as
    HoldMeanSensorClassifier. Sensor A/B, and the direction the sensor's yellow end points in the palm."""

    def __init__(self, path=C.LEFT_HAND_MODELS):
        bundle = joblib.load(path)
        self.features = list(bundle["features"])
        self.sensor, self.direction = bundle["sensor"], bundle["direction"]
        self.index_b = list(self.sensor.classes_).index(1)
        self.name = f"final_model/{path.name} (sensor LOO {bundle['sensor_loo']})"
        self.direction_name = f"final_model/{path.name} (direction LOO {bundle['direction_loo']})"

    def vector(self, row: dict) -> np.ndarray:
        """One frame's features in model order (missing / None -> 0), as in training."""
        return np.array([row.get(c) or 0.0 for c in self.features], float)

    def p_sensor_b(self, mean_vector: np.ndarray) -> float:
        X = mean_vector.astype(np.float32).reshape(1, -1)
        return float(self.sensor.predict_proba(X)[0, self.index_b])

    def direction_deg(self, mean_vector: np.ndarray) -> float:
        """Where the yellow end points in the palm plane: 0 deg = towards the thumb, 90 = the fingertips."""
        c, s = self.direction.predict(mean_vector.astype(np.float32).reshape(1, -1))[0]
        return float(np.degrees(np.arctan2(s, c)))


@lru_cache(maxsize=1)
def classifier() -> SensorClassifier | None:
    try:
        return SensorClassifier()
    except Exception as e:  # missing xgboost or unreadable model: dashboard falls back to ground truth
        print(f"[backend] classifier unavailable: {e}")
        return None


def grasp_decision(probs: list[float]) -> dict:
    """The one sensor decision of a grasp: the label of the mean P(Sensor B) over its settled hold frames."""
    p = float(np.mean(probs))
    label = "B" if p >= 0.5 else "A"
    agree = float(np.mean([(q >= 0.5) == (label == "B") for q in probs]))
    return {"label": label, "confidence": round(p if label == "B" else 1 - p, 3),
            "agreement": round(agree, 3), "n_votes": len(probs)}


def _decisions(p_b: np.ndarray, active: np.ndarray, phase: np.ndarray):
    """Per frame (sensor estimate, state, progress) of a replayed grasp: one decision per grasp.

    Nothing is shown ("analysing") until HOLD_MEAN_MIN_FRAMES settled hold frames (phase "hold", object
    touched) have been seen; the decision made then is fixed for the rest of the grasp instead of being
    re-voted every frame. A grasp whose hold ends sooner (the object dropped) is decided at the end of
    its hold with the frames it has."""
    need = C.HOLD_MEAN_MIN_FRAMES
    settled = active & (phase == "hold")
    hold_idx = np.flatnonzero(phase == "hold")
    last_hold = int(hold_idx[-1]) if len(hold_idx) else -1
    out, probs, decision = [], [], None
    for i in range(len(p_b)):
        if decision is None:
            if settled[i]:
                probs.append(float(p_b[i]))
            if len(probs) >= need or (probs and i == last_hold):
                decision = grasp_decision(probs)
        if decision:
            out.append((decision, "final", 1.0))
        elif phase[i] in ("grasp", "hold"):
            out.append((None, "analysing", round(min(1.0, len(probs) / need), 3)))
        else:
            out.append((None, "idle", 0.0))
    return out


def load_session(session_id: str, twin) -> dict:
    folder = C.DATASET_DIR / session_id
    meta = json.loads((folder / "meta.json").read_text())
    info = meta.get("episode_info", {})
    obj = meta.get("object") or info.get("object")
    act = pd.read_csv(folder / "actuators.csv")
    summ = pd.read_csv(folder / "tactile_summary.csv")
    pose = pd.read_csv(folder / "object_pose.csv")
    df = summ.merge(act[["frame"] + [f"angle_act_{d}" for d in DOF] + [f"force_act_{d}" for d in DOF]],
                    on="frame").merge(pose, on="frame", suffixes=("", "_pose"))

    groups = [g["name"].lower().replace(" ", "_") for g in meta["sensor_groups"]]
    contacts = df[[f"{g}_contacts" for g in groups]].sum(axis=1).to_numpy()
    active = contacts > C.MIN_CONTACTS_ACTIVE

    clf = classifier()
    p_b = clf.predict(df) if clf else None
    decisions = (_decisions(p_b, active, df["phase"].to_numpy()) if clf
                 else [(None, "idle", 0.0)] * len(df))
    estimates = [d[0] for d in decisions]

    root = twin.root_pos()
    grasp_quat = SC.OBJECTS[obj]["grasp_quat"]
    R0 = Rotation.from_quat([*grasp_quat[1:], grasp_quat[0]])
    axis0 = R0.apply(OBJECT_LONG_AXIS)
    t0 = float(df["t_rel"].iloc[0])

    frames = []
    for i, r in df.iterrows():
        q = [r.quat_w, r.quat_x, r.quat_y, r.quat_z]
        pos = np.array([r.pos_x, r.pos_y, r.pos_z])
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        axis = R.apply(OBJECT_LONG_AXIS)
        tilt = np.degrees(np.arccos(np.clip(axis @ axis0, -1, 1)))
        speed = float(np.linalg.norm([r.lin_vel_x, r.lin_vel_y, r.lin_vel_z]) * 1000)
        in_hold = r.phase == "hold"
        frames.append({
            "frame": int(r.frame), "t": round(float(r.t_rel) - t0, 3), "phase": r.phase,
            "bodies": twin.poses(np.array([r[f"angle_act_{d}"] for d in DOF], float), obj, np.array([*pos, *q])),
            "angles": [int(r[f"angle_act_{d}"]) for d in DOF],
            "force_g": [int(r[f"force_act_{d}"]) for d in DOF],
            "tactile": {g: float(r[f"{g}_sum"]) for g in groups},
            "contacts": int(contacts[i]),
            "in_hand": bool(r.in_hand),
            "kpi": {
                "sensor": estimates[i], "sensor_state": decisions[i][1], "analysis_progress": decisions[i][2],
                "p_sensor_b": None if p_b is None else round(float(p_b[i]), 3),
                "pos_mm": np.round((pos - root) * 1000, 1).tolist(),
                "axis": np.round(axis, 3).tolist(),
                "tilt_deg": round(float(tilt), 1),
                "grip_force_g": int(sum(r[f"force_act_{d}"] for d in DOF[:5])),
                "speed_mm_s": round(speed, 1),
                "slipping": bool(in_hold and speed > C.SLIP_SPEED_MM_S),
            },
        })

    final = next((e for e in reversed(estimates) if e), None)
    truth = C.OBJECT_TO_LABEL.get(obj)
    return {
        "id": session_id, "object": obj, "truth": truth, "dt": 1.0 / meta.get("sample_rate_target_hz", 20),
        "hand_side": C.HAND_SIDE, "groups": groups, "frames": frames,
        "summary": {
            "estimate": final, "correct": None if final is None else final["label"] == truth,
            "classifier": f"xgb_sensor{clf.suffix}" if clf else None,
            "success": info.get("success"), "failure": info.get("failure"),
            "shift_mm": info.get("shift_mm"), "rot_deg": info.get("rot_deg"),
        },
    }
