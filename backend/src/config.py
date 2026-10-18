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

# Live hand: hardware/config.py broadcasts one JSON datagram per frame (BROADCAST_HOST/PORT).
# 0.0.0.0: broadcasts to 255.255.255.255 never reach a socket bound to 127.0.0.1.
LIVE_UDP_HOST = os.environ.get("LIVE_UDP_HOST", "0.0.0.0")
LIVE_UDP_PORT = int(os.environ.get("LIVE_UDP_PORT", "5005"))
LIVE_HAND_SIDE = os.environ.get("LIVE_HAND_SIDE", "left")  # the right hand is out of service (pose/config.py)
LIVE_CLASSIFIER_SUFFIX = os.environ.get("LIVE_CLASSIFIER_SUFFIX", "_combined")  # real + sim training data
# Live sensor A/B model:
#   "hold_mean"  final_model/sensor_classifier.pkl (XGBoost on the *average* of the grasp's hold phase,
#                trained on 33 labelled real sessions, leave-one-out 81.8%): one estimate per grasp
#   "per_frame"  xgb_sensor<LIVE_CLASSIFIER_SUFFIX>, one vote per frame, majority over the grasp
# If the hold_mean model can't be loaded, the live pipeline falls back to per_frame.
LIVE_SENSOR_MODEL = os.environ.get("LIVE_SENSOR_MODEL", "hold_mean")
HOLD_MEAN_MODEL = REPO_ROOT / "final_model" / "sensor_classifier.pkl"
HOLD_MEAN_MIN_FRAMES = 15      # final_model/match_session.py MIN_HOLD_FRAMES
# With Read Thumb Press (GUI state PROBING) the hold_mean estimate is made at probing start from the
# last PRE_PROBE_FRAMES readings before it (index closed and settled, thumb not moved yet: the pose
# the model was trained on; after probing the thumb sits elsewhere), shown PROBE_ESTIMATE_DELAY_S
# after probing started, and kept until the hand opens.
PRE_PROBE_FRAMES = 20
PROBE_ESTIMATE_DELAY_S = float(os.environ.get("PROBE_ESTIMATE_DELAY_S", "4.0"))

OBJECT_TO_LABEL ={"vegabar_29": "A", "vegabar_38": "B"}
LABEL_NAMES = {"A": "Sensor A · VEGABAR 29", "B": "Sensor B · VEGABAR 38"}
MIN_CONTACTS_ACTIVE = 20       # classifier/predict.py: frames with fewer contacts are pre/post grasp
# Live hand: predict only once the GUI has reported HOLD for this long. While closing (and right
# after) the fingers still move and the tactile picture is not the final grasp yet.
PREDICT_AFTER_HOLD_S = float(os.environ.get("PREDICT_AFTER_HOLD_S", "1.0"))
# Live object position. The pose model is sim-trained; on real tactile data it sits ~40 mm off
# (towards the wrist / little finger) and jumps up to ~60 mm between frames (thumb_mid_cx alone
# moves it ~27 mm per taxel). So the object starts at its nominal grasp position (in the hand)
# and the model may only shift it by POS_MAX_OFFSET_MM, as the median over the last POS_MEDIAN_S.
# 0 now: the live object starts from the real-grasp pose below, and any shift by the sim-trained
# model moves it into the fingers or off the palm.
POS_MAX_OFFSET_MM = float(os.environ.get("POS_MAX_OFFSET_MM", "0"))
POS_MEDIAN_S = float(os.environ.get("POS_MEDIAN_S", "2.0"))
# Live object pose for the real hand's grasp (instead of the simulation's grasp pose, where the fingers
# stop half-open and the thumb is opposed). Fitted at startup by twin.fit_real_grasp: nominal
# orientation (long axis across the fingers), resting 1 mm off the drawn palm, at the height where the
# fingers at LIVE_HOLD_ANGLES just touch it. LIVE_HOLD_ANGLES = median finger angles of the real holds
# in hardware/logs (17 Oct, 20:07-20:39): the real fingers close almost fully around the sensor.
LIVE_HOLD_ANGLES = [3, 14, 13, 18, 0, 608]
# Calibration from a photo of the real grasp, per object, in the (right-hand) model frame:
# mm = [x out of the palm, y across the fingers towards the thumb, z towards the fingertips],
# deg = rotation [about x, y, z] applied on top of the nominal orientation.
LIVE_GRASP_OFFSET = {
    "vegabar_29": {"mm": [0.0, 0.0, 0.0], "deg": [0.0, 0.0, 0.0]},
    "vegabar_38": {"mm": [0.0, 0.0, 0.0], "deg": [0.0, 0.0, 0.0]},
}
# Twin fingers (and thumb bend) stop at the object's surface instead of passing into it: the real
# fingers read 3-18 while holding, which the model draws as a closed fist through the object.
LIVE_FINGER_CLAMP = os.environ.get("LIVE_FINGER_CLAMP", "1") != "0"
SLIP_SPEED_MM_S = 20.0         # object speed relative to the hand counted as slipping during the hold
