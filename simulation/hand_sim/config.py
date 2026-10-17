import os
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent.parent
# Which hand to simulate. "left" is built by mirroring the right-hand model (hand_sim/left_hand.py).
# Override per run with the environment variable HAND_SIDE=right.
HAND_SIDE = os.environ.get("HAND_SIDE", "left")
HAND_XML = SIM_DIR / "Inspire_hand" / "xml" / "inspire_hand_right.xml"  # source model (right)
CAD_DIR = SIM_DIR / "assets"
OBJECTS_DIR = SIM_DIR / "objects"

# DOF order of the real hand registers (ANGLE_SET(0..5) etc.) and of models/dataset.
DOF_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
DOF_JOINTS = [
    f"{HAND_SIDE}_pinky_proximal_joint",
    f"{HAND_SIDE}_ring_proximal_joint",
    f"{HAND_SIDE}_middle_proximal_joint",
    f"{HAND_SIDE}_index_proximal_joint",
    f"{HAND_SIDE}_thumb_proximal_pitch_joint",
    f"{HAND_SIDE}_thumb_proximal_yaw_joint",
]
# Body whose tip defines the lever arm used to convert actuator torque to fingertip force (grams).
DOF_TIP_BODIES = [
    f"{HAND_SIDE}_pinky_intermediate",
    f"{HAND_SIDE}_ring_intermediate",
    f"{HAND_SIDE}_middle_intermediate",
    f"{HAND_SIDE}_index_intermediate",
    f"{HAND_SIDE}_thumb_distal",
    f"{HAND_SIDE}_thumb_distal",
]

# Hand register limits (docs/internal/hand_manual.pdf, section 2.6).
ANGLE_MAX = 1000  # 1000 = open (joint at 0 rad), 0 = closed (joint at upper range limit)
FORCE_SET_MAX = 3000  # grams at the fingertip
SPEED_MAX = 1000  # 1000 = full stroke in FULL_STROKE_TIME_S without load
FULL_STROKE_TIME_S = 0.6
FORCE_ACT_IDLE_G = 77.0  # idle FORCE_ACT reading observed in models/dataset

# STATUS codes as observed in models/dataset (not documented in the manual).
STATUS_RELEASING = 0
STATUS_GRASPING = 1
STATUS_FORCE_STOP = 2

# Collision bit masks: hand collides with objects (and floor), not with itself.
HAND_CONTYPE, HAND_CONAFFINITY = 2, 1
OBJECT_CONTYPE, OBJECT_CONAFFINITY = 1, 3

# Grasp objects prepared from CAD by `python -m hand_sim.assets`.
# Masses are placeholders until confirmed from the VEGA datasheets.
# grasp_pos/grasp_quat: default object pose (world frame) for a power grasp in front of the palm;
# the hand's palm faces +x, fingers point +z, thumb is on the +y side.
OBJECTS = {
    "vegabar_29": {
        "cad": "0edecf86-fa9a-4a2b-9fc1-a49439372a67-VEGABAR_29_STL.stl",
        "mass_kg": 0.25,
        "rgba": [0.75, 0.75, 0.78, 1.0],
        "grasp_pos": [0.044, 0.09, 0.11],
        "grasp_quat": [1.0, 0.0, 0.0, 0.0],
        "grasp_force_g": 1000,
        # Point (object frame) on the long axis that random rotations pivot around.
        "pivot": [0.0, 0.0, 0.0],
        # Random handover offsets of the pivot (world frame, metres): (min, max) per axis.
        "offset_range": [[0.0, 0.008], [-0.015, 0.015], [-0.012, 0.012]],
    },
    "vegabar_38": {
        "cad": "f8a929b6-d3c2-485c-9162-e154690c0d41-VEGABAR_38.stl",
        "mass_kg": 0.6,
        "rgba": [0.55, 0.6, 0.7, 1.0],
        # Shaft across the fingers, heavy head just past the little finger (close to the COM).
        "grasp_pos": [0.030, 0.076, 0.11],
        "grasp_quat": [0.0, 0.0, 0.0, 1.0],
        "grasp_force_g": 2500,
        "pivot": [-0.014, 0.0, 0.0],  # shaft axis
        "offset_range": [[0.0, 0.008], [0.0, 0.015], [-0.012, 0.012]],
    },
}

# Random handover orientation (applied about the object's pivot).
ROLL_RANGE_RAD = (-3.14159, 3.14159)  # about the object's long axis (object y)
TILT_X_RANGE_RAD = (-0.35, 0.35)  # long axis tilts within the palm plane
TILT_Z_RANGE_RAD = (-0.17, 0.17)  # long axis tilts toward/away from the palm

# Episode timing (seconds of sim time) and rates.
CONTROL_HZ = 20  # policy / register update rate, matches hardware/logger.py target_hz
APPROACH_DIST_M = 0.08
APPROACH_TIME_S = 0.6
GRASP_TIMEOUT_S = 2.0
RELEASE_RAMP_S = 0.3  # after release the carried weight is handed over gradually
HOLD_TIME_S = 2.0
# Success after the hold phase: pivot moved less than this, rotated less than this, no floor contact.
SUCCESS_MAX_SHIFT_M = 0.015
SUCCESS_MAX_ROT_RAD = 0.26

# Pre-grasp posture: fingers open, thumb rotated into opposition (ANGLE_SET order = DOF_NAMES).
PRESHAPE_ANGLES = [1000, 1000, 1000, 1000, 1000, 0]


def _mirror_object_defaults():
    """Left hand: the hand is mirrored across the plane y = hand-root y, so are the default grasp poses.
    A mirrored pose S R is not a rotation; the sensors are mirror-symmetric about their own x-y plane
    (long axis y, the VEGABAR 38 shaft offset along x), so S R M with M = diag(1, 1, -1) is the proper
    rotation that puts the same geometry in the mirrored place."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    y0 = 0.1  # right_hand_root / left_hand_root position in the scene
    S, M = np.diag([1.0, -1.0, 1.0]), np.diag([1.0, 1.0, -1.0])
    for cfg in OBJECTS.values():
        cfg["grasp_pos"][1] = 2 * y0 - cfg["grasp_pos"][1]
        w, x, y, z = cfg["grasp_quat"]
        R = S @ Rotation.from_quat([x, y, z, w]).as_matrix() @ M
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        cfg["grasp_quat"] = [float(w), float(x), float(y), float(z)]
        lo, hi = cfg["offset_range"][1]
        cfg["offset_range"][1] = [-hi, -lo]


if HAND_SIDE == "left":
    _mirror_object_defaults()
