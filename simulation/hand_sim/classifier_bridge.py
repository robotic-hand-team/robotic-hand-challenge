"""Runs classifier/ (Lars's XGBoost Sensor A/B model, or a retrained variant) live against the
simulated tactile + actuator stream, for comparison against the sim's own ground-truth object id.

vegabar_29 = Sensor A, vegabar_38 = Sensor B (confirmed mapping, 2026-10-17).

The model variant is chosen with CLASSIFIER_SUFFIX (env var, or the constructor argument):
    ""          real hardware recordings only  -> classifier/model/xgb_sensor.pkl
    "_combined" real + simulated (default)     -> classifier/model/xgb_sensor_combined.pkl
    "_sim"      simulated only                 -> classifier/model/xgb_sensor_sim.pkl

Note the model only *identifies* the held object; it does not drive the grasp (that is the job of a
Policy). Here it runs alongside the scripted policy so the viewer can show prediction vs. truth.
"""

import os

import joblib
import numpy as np

from . import config as C

CLASSIFIER_DIR = C.SIM_DIR.parent / "classifier" / "model"
OBJECT_TO_LABEL = {"vegabar_29": 0, "vegabar_38": 1}
LABEL_NAMES = {0: "Sensor A (VEGABAR 29)", 1: "Sensor B (VEGABAR 38)"}
MIN_CONTACTS_ACTIVE = 20  # matches classifier/predict.py's "ignore pre/post frames" filter


class ClassifierBridge:
    def __init__(self, suffix: str | None = None):
        self.suffix = os.environ.get("CLASSIFIER_SUFFIX", "_combined") if suffix is None else suffix
        self.model = joblib.load(CLASSIFIER_DIR / f"xgb_sensor{self.suffix}.pkl")
        self.scaler = joblib.load(CLASSIFIER_DIR / f"scaler{self.suffix}.pkl")
        self.feature_cols = joblib.load(CLASSIFIER_DIR / f"feature_cols{self.suffix}.pkl")
        self.votes: list[int] = []  # active-frame labels this episode, for the session majority vote
        self.last: dict | None = None  # most recent per-frame prediction

    def reset(self):
        self.votes.clear()
        self.last = None

    def predict(self, tactile_summary: dict, hand_registers: dict) -> dict:
        """One control-tick prediction from TactileModel.summary() plus HandController.read()."""
        row = {}
        for name, summary in tactile_summary.items():
            for stat in ("sum", "max", "contacts", "cy", "cx", "force"):
                value = summary[stat]
                row[f"{name}_{stat}"] = 0.0 if value == "" else value
        for n, v in zip(C.DOF_NAMES, hand_registers["angle_act"]):
            row[f"angle_act_{n}"] = v

        x = np.array([[row.get(c, 0.0) for c in self.feature_cols]])
        x_scaled = self.scaler.transform(x)
        label = int(self.model.predict(x_scaled)[0])
        confidence = float(self.model.predict_proba(x_scaled)[0].max())

        total_contacts = sum(s["contacts"] for s in tactile_summary.values())
        active = total_contacts > MIN_CONTACTS_ACTIVE  # hand open / nothing grasped
        if active:
            self.votes.append(label)
        self.last = {"label": label, "name": LABEL_NAMES[label], "confidence": confidence,
                     "active": active, "contacts": total_contacts}
        return self.last

    def recent(self, window: int = 10, min_votes: int = 5) -> dict | None:
        """Majority label over the last `window` active frames; None until `min_votes` have arrived."""
        votes = self.votes[-window:]
        if len(votes) < min_votes:
            return None
        label = int(np.bincount(votes).argmax())
        return {"label": label, "name": LABEL_NAMES[label],
                "agreement": votes.count(label) / len(votes), "n_votes": len(votes)}

    def majority_vote(self) -> dict | None:
        """Majority label over a whole episode (session-level prediction, like predict.py)."""
        if not self.votes:
            return None
        label = int(np.bincount(self.votes).argmax())
        return {"label": label, "name": LABEL_NAMES[label], "n_votes": len(self.votes),
                "agreement": self.votes.count(label) / len(self.votes)}
