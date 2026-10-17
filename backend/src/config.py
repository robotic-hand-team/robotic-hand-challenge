"""Paths and settings for the dashboard backend."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = REPO_ROOT / "simulation"
DATASET_DIR = Path(os.environ.get("DATASET_DIR", REPO_ROOT / "models" / "dataset_sim"))
CLASSIFIER_DIR = REPO_ROOT / "classifier" / "model"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"

# The mock data in models/dataset_sim was generated with the right hand (before left-hand support).
HAND_SIDE = os.environ.get("TWIN_HAND_SIDE", "right")

# Sim-trained sensor classifier (xgb_sensor_sim.pkl). The real-data models on main currently fail to
# load ("input stream corrupted"), see README.
CLASSIFIER_SUFFIX = os.environ.get("CLASSIFIER_SUFFIX", "_sim")

OBJECT_TO_LABEL = {"vegabar_29": "A", "vegabar_38": "B"}
LABEL_NAMES = {"A": "Sensor A · VEGABAR 29", "B": "Sensor B · VEGABAR 38"}
MIN_CONTACTS_ACTIVE = 20       # classifier/predict.py: frames with fewer contacts are pre/post grasp
SLIP_SPEED_MM_S = 20.0         # object speed relative to the hand counted as slipping during the hold
