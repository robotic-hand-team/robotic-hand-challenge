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
#   "left_hand"  final_model/left_hand_models.pkl (classifier/train_left_hand.py): random forests on the
#                average of the grasp's hold phase, trained on the 58 labelled grasps of the live left hand
#                (models/dataset_left). Leave-one-grasp-out on the 37 photo labels: sensor 76%, direction
#                median error 7 deg. Also gives the direction the object points, which drives its pose.
#   "hold_mean"  final_model/sensor_classifier.pkl (XGBoost, same hold average, 33 real sessions; 73% left-
#                one-out on the same photo labels): sensor only, the object keeps its nominal pose
#   "per_frame"  xgb_sensor<LIVE_CLASSIFIER_SUFFIX>, one vote per frame, majority over the grasp
# If the chosen model can't be loaded, the live pipeline falls back to the next one down.
LIVE_SENSOR_MODEL = os.environ.get("LIVE_SENSOR_MODEL", "left_hand")
LEFT_HAND_MODELS = REPO_ROOT / "final_model" / "left_hand_models.pkl"
HOLD_MEAN_MODEL = REPO_ROOT / "final_model" / "sensor_classifier.pkl"
# Settled hold frames averaged before the grasp's one decision. The models were trained on whole hold windows
# (median 87 frames); 30 (~0.8 s at the GUI's ~38 Hz) is closer to that than final_model's minimum of 15 and
# in the live replay (backend/eval_live.py) decides as many grasps.
HOLD_MEAN_MIN_FRAMES = 30
# One decision per grasp: nothing is shown ("analysing") until HOLD_MEAN_MIN_FRAMES settled hold frames
# are in, then that answer is fixed until the grasp ends (the hand opens, a new Close starts, or - for a
# stream without a state field - contact has been gone for GRASP_END_S).
GRASP_END_S = 1.0
# With Read Thumb Press (GUI state PROBING) the hold_mean estimate is made at probing start from the
# last PRE_PROBE_FRAMES readings before it (index closed and settled, thumb not moved yet: the pose
# the model was trained on; after probing the thumb sits elsewhere), shown PROBE_ESTIMATE_DELAY_S
# after probing started, and kept until the hand opens.
PRE_PROBE_FRAMES = 20
PROBE_ESTIMATE_DELAY_S = float(os.environ.get("PROBE_ESTIMATE_DELAY_S", "10.0"))

OBJECT_TO_LABEL ={"vegabar_29": "A", "vegabar_38": "B"}
LABEL_NAMES = {"A": "Sensor A · VEGABAR 29", "B": "Sensor B · VEGABAR 38"}
# Grip force: the hand's force_act has a zero offset per finger (about -150…+250 g with nothing
# touching, different after every power-up). The backend averages force_act while the GUI reports
# OPENED (exponential average, this weight per frame), subtracts it and clips at 0.
FORCE_BASE_ALPHA = 0.05
MIN_CONTACTS_ACTIVE = 20       # classifier/predict.py: frames with fewer contacts are pre/post grasp
# The sensor model runs only while an object is in the hand: the fingers (little–index) and the
# palm together carry at least OBJECT_MIN_SUM counts above baseline on more than OBJECT_MIN_CONTACTS
# taxels. Thumb groups don't count. Measured on the 98 hardware/logs holds: empty closes (fist or
# open hand: drift, fingers touching each other) reach 1.6–2.5k counts on 18–23 taxels, held
# objects 7.7k+ on 35+. This gate passes 1% of empty-hold frames (total contacts > 20: 42%) and
# 100% of object-hold frames.
OBJECT_MIN_SUM = float(os.environ.get("OBJECT_MIN_SUM", "5000"))
OBJECT_MIN_CONTACTS = int(os.environ.get("OBJECT_MIN_CONTACTS", "30"))
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
# Live object pose from the "left_hand" direction estimate (twin.pose_in_palm): the object is turned in the
# palm plane so its yellow end (object +y: VEGABAR 29's connector, VEGABAR 38's head) points the estimated
# way, with its grip point on the fist centre. Grip point (object frame, m): where the fingers hold it, from
# the session photos: VEGABAR 29 around its middle, VEGABAR 38 on the shaft next to the head (shaft axis
# at x = -14 mm, simulation pivot). The fist centre (right-hand model frame, mm: y across the fingers, z
# up the palm) is where the photos show the object: across the middle of the four fingers (their bases
# span y 74-132 mm) and just below the finger bases (z 136 mm), so the closing fingers wrap it.
# fit_real_grasp's height (75 mm) put it under the curled fingers, low in front of the palm.
LIVE_FIST_CENTER_MM = [float(v) for v in os.environ.get("LIVE_FIST_CENTER_MM", "103,118").split(",")]
# Palm tactile sensor (8 rows x 14 columns) in the right-hand model frame, from the hand manual's diagram
# (p.26): directly below the finger bases (z 136 mm), centred under the four fingers, about 60 x 34 mm,
# i.e. ~4.3 mm taxel pitch. The object's axis is put through the centroid of the palm's pressure strip
# (palm_axis.palm_point); along its axis the grip point stays as close to the fist centre as it can.
PALM_SENSOR_CENTER_MM = (103.0, 113.0)
PALM_SENSOR_PITCH_MM = 4.3
# Nominal direction of each object's real-grasp pose (its grasp_quat): 29's connector towards the thumb
# (0 deg), 38's head towards the little finger (180 deg).
LIVE_GRIP_POINT = {"vegabar_29": [0.0, 0.0, 0.0], "vegabar_38": [-0.014, -0.045, 0.0]}
LIVE_NOMINAL_DIRECTION_DEG = {"vegabar_29": 0.0, "vegabar_38": 180.0}
# Twin fingers (and thumb bend) stop at the object's surface instead of passing into it: the real
# fingers read 3-18 while holding, which the model draws as a closed fist through the object.
LIVE_FINGER_CLAMP = os.environ.get("LIVE_FINGER_CLAMP", "1") != "0"
SLIP_SPEED_MM_S = 20.0         # object speed relative to the hand counted as slipping during the hold
