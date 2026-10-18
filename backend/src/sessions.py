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


@lru_cache(maxsize=1)
def classifier() -> SensorClassifier | None:
    try:
        return SensorClassifier()
    except Exception as e:  # missing xgboost or unreadable model: dashboard falls back to ground truth
        print(f"[backend] classifier unavailable: {e}")
        return None


def _running_estimate(p_b: np.ndarray, active: np.ndarray) -> list[dict | None]:
    """Session estimate as it would be known at each frame: majority over active frames so far."""
    out, votes, probs = [], [], []
    for p, a in zip(p_b, active):
        if a:
            votes.append(p >= 0.5)
            probs.append(p)
        if not votes:
            out.append(None)
            continue
        n_b = int(sum(votes))
        label = "B" if n_b * 2 > len(votes) else "A"
        conf = float(np.mean([q if label == "B" else 1 - q for q in probs]))
        out.append({"label": label, "confidence": round(conf, 3),
                    "agreement": round(max(n_b, len(votes) - n_b) / len(votes), 3), "n_votes": len(votes)})
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
    estimates = _running_estimate(p_b, active) if clf else [None] * len(df)

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
                "sensor": estimates[i],
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
