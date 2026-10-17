from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent.parent
HAND_XML = SIM_DIR / "Inspire_hand" / "xml" / "inspire_hand_right.xml"
CAD_DIR = SIM_DIR / "assets"
OBJECTS_DIR = SIM_DIR / "objects"

# DOF order of the real hand registers (ANGLE_SET(0..5) etc.) and of models/dataset.
DOF_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
DOF_JOINTS = [
    "right_pinky_proximal_joint",
    "right_ring_proximal_joint",
    "right_middle_proximal_joint",
    "right_index_proximal_joint",
    "right_thumb_proximal_pitch_joint",
    "right_thumb_proximal_yaw_joint",
]
# Body whose tip defines the lever arm used to convert actuator torque to fingertip force (grams).
DOF_TIP_BODIES = [
    "right_pinky_intermediate",
    "right_ring_intermediate",
    "right_middle_intermediate",
    "right_index_intermediate",
    "right_thumb_distal",
    "right_thumb_distal",
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

# Collision bit masks. A pair collides when (contype_a & conaffinity_b) or (contype_b & conaffinity_a).
# Bits: 1 = object, 2 = hand, 4 = world. The hand keeps bit 1 in its conaffinity so hand-hand pairs
# collide (fingers must not pass through each other); the world moves to its own bit so the hand
# still cannot touch the ground.
#   hand vs object : 2 & 3 = 2  -> collide
#   hand vs hand   : 2 & 3 = 2  -> collide,  0 -> pass through (HAND_ALLOWS_SELF_COLLISION)
#   hand vs floor  : 2 & 1 = 0  -> no contact either way
#   object vs floor: 1 & 1 = 1  -> collide
HAND_ALLOWS_SELF_COLLISION = True
HAND_CONTYPE = 2
HAND_CONAFFINITY = 3 if HAND_ALLOWS_SELF_COLLISION else 1
OBJECT_CONTYPE, OBJECT_CONAFFINITY = 1, 3
FLOOR_CONTYPE, FLOOR_CONAFFINITY = 4, 1

# --- Physics stability (Phase 1) ---------------------------------------------------------------
# Measured against models/dataset-independent acceptance tests; see hand_sim/physics_audit.py.
# TIMESTEP: solver temporal resolution. The position servo (SERVO_KP) was originally tuned to be
# stable at 2 ms; a shorter step makes it more stable, not less.
TIMESTEP = 0.001
SOLVER_ITERATIONS = 50
LS_ITERATIONS = 50
NOSLIP_ITERATIONS = 5  # MuJoCo default is 0 (no slip / friction post-processing at all)
# Contact spring-damper and impedance. solref = [timeconst, dampratio]; solimp = [dmin, dmax, width,
# midpoint, power]. Higher dmin/dmax = firmer contact = less interpenetration ("bleeding").
# timeconst 0.01 s is 10 solver steps at TIMESTEP, the same ratio the MuJoCo default (0.02 at 2 ms)
# had, but twice as firm in absolute terms.
# Measured with hand_sim.physics_audit.py, 20 paired episodes: hand-object penetration deepest
# -1.65 mm with 351 ticks deeper than 1 mm at the MuJoCo defaults, versus -0.84 mm and 0 ticks here.
# A blanket setting is used because the fingers have a single collision geom each, so there is no
# separate pad geom to soften; [0.95, 0.99] measured no better and leaves less solver headroom.
# None = leave the MuJoCo defaults.
CONTACT_SOLREF = [0.01, 1.0]
CONTACT_SOLIMP = [0.93, 0.97, 0.001, 0.5, 2.0]
# Max vertices when MuJoCo builds a collision mesh's convex hull from a mesh file. Every object
# collision part exceeds 64 verts (74-2366), so 64 truncated them all; see physics_audit.py.
MAX_HULL_VERT = 256

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
        "grasp_force_g": 3000,
        # Point (object frame) on the long axis that random rotations pivot around.
        "pivot": [0.0, 0.0, 0.0],
        # Random handover offsets of the pivot (world frame, metres): (min, max) per axis.
        "offset_range": [[0.0, 0.015], [-0.03, 0.03], [-0.025, 0.025]],
    },
    "vegabar_38": {
        "cad": "f8a929b6-d3c2-485c-9162-e154690c0d41-VEGABAR_38.stl",
        "mass_kg": 0.6,
        "rgba": [0.55, 0.6, 0.7, 1.0],
        # Shaft across the fingers, heavy head just past the little finger (close to the COM).
        "grasp_pos": [0.030, 0.076, 0.11],
        "grasp_quat": [0.0, 0.0, 0.0, 1.0],
        "grasp_force_g": 3000,
        "pivot": [-0.014, 0.0, 0.0],  # shaft axis
        "offset_range": [[0.0, 0.015], [0.0, 0.03], [-0.025, 0.025]],
    },
}

# Random handover orientation (applied about the object's pivot).
ROLL_RANGE_RAD = (-3.14159, 3.14159)  # about the object's long axis (object y)
TILT_X_RANGE_RAD = (-0.5, 0.5)  # long axis tilts within the palm plane
TILT_Z_RANGE_RAD = (-0.3, 0.3)  # long axis tilts toward/away from the palm

# Episode timing (seconds of sim time) and rates.
CONTROL_HZ = 20  # policy / register update rate, matches hardware/logger.py target_hz
APPROACH_DIST_M = 0.08
APPROACH_TIME_S = 1.5
GRASP_TIMEOUT_S = 2.0
RELEASE_RAMP_S = 0.3  # after release the carried weight is handed over gradually
HOLD_TIME_S = 2.0
# Success after the hold phase: pivot moved less than this, rotated less than this, no floor contact.
SUCCESS_MAX_SHIFT_M = 0.015
SUCCESS_MAX_ROT_RAD = 0.26

# Pre-grasp posture: fingers open, thumb rotated into opposition (ANGLE_SET order = DOF_NAMES).
PRESHAPE_ANGLES = [1000, 1000, 1000, 1000, 1000, 0]

# --- Tactile sensors (docs/internal/hand_manual.pdf, section 2.6.20 / p.26-27) -------------------
# 17 sensor groups, matching the register addresses, sizes and names used in models/dataset.
# Column key = the real dataset's lowercase_with_underscore name, e.g. "little_tip".
#
# Patch geometry (where each group sits on its body) is NOT hardcoded here: tactile.py derives it
# at startup from the live mesh, using the conventions below plus a (start_frac, end_frac) window
# along the body's row axis, measured from the joint end (0.0) to the fingertip/distal end (1.0).
# This was calibrated by sampling real contact points during many grasps (see calibrate_tactile.py)
# and cross-checked against each mesh's own bounding box; it is an approximation of the real sensor
# placement in the hand manual's diagram (p.26), not a measured fact.
TACTILE_CONTACT_THRESHOLD = 30  # raw units; matches real hand (hand_gui.py uses 30)
TACTILE_RAW_MAX = 4095

# Local axis convention per body category: row_axis points from the joint (0) toward the fingertip
# (1); col_axis is the other in-plane direction; normal points out of the surface into the contact.
FINGER_AXES = {"row": [0, 1, 0], "col": [0, 0, -1], "normal": [-1, 0, 0]}
THUMB_AXES = {"row": [1, 0, 0], "col": [0, 0, 1], "normal": [0, 1, 0]}
PALM_AXES = {"row": [0, 0, 1], "col": [0, 1, 0], "normal": [1, 0, 0]}
COL_FRAC = (0.1, 0.9)  # shared margin across the width of every non-palm patch

SENSOR_GROUPS = [
    # name, addr, rows, cols, body, axes, row_frac
    ("little_tip", 3000, 3, 3, "right_pinky_intermediate", FINGER_AXES, (0.76, 0.97)),
    ("little_nail", 3018, 12, 8, "right_pinky_intermediate", FINGER_AXES, (0.08, 0.72)),
    ("little_pad", 3210, 10, 8, "right_pinky_proximal", FINGER_AXES, (0.12, 0.85)),
    ("ring_tip", 3370, 3, 3, "right_ring_intermediate", FINGER_AXES, (0.76, 0.97)),
    ("ring_nail", 3388, 12, 8, "right_ring_intermediate", FINGER_AXES, (0.08, 0.72)),
    ("ring_pad", 3580, 10, 8, "right_ring_proximal", FINGER_AXES, (0.12, 0.85)),
    ("middle_tip", 3740, 3, 3, "right_middle_intermediate", FINGER_AXES, (0.76, 0.97)),
    ("middle_nail", 3758, 12, 8, "right_middle_intermediate", FINGER_AXES, (0.08, 0.72)),
    ("middle_pad", 3950, 10, 8, "right_middle_proximal", FINGER_AXES, (0.12, 0.85)),
    ("index_tip", 4110, 3, 3, "right_index_intermediate", FINGER_AXES, (0.76, 0.97)),
    ("index_nail", 4128, 12, 8, "right_index_intermediate", FINGER_AXES, (0.08, 0.72)),
    ("index_pad", 4320, 10, 8, "right_index_proximal", FINGER_AXES, (0.12, 0.85)),
    ("thumb_tip", 4480, 3, 3, "right_thumb_distal", THUMB_AXES, (0.68, 0.95)),
    ("thumb_nail", 4498, 12, 8, "right_thumb_distal", THUMB_AXES, (0.05, 0.62)),
    ("thumb_mid", 4690, 3, 3, "right_thumb_intermediate", THUMB_AXES, (0.45, 0.75)),
    ("thumb_pad", 4708, 12, 8, "right_thumb_proximal", THUMB_AXES, (0.10, 0.85)),
    ("palm", 4900, 8, 14, "right_hand_root", PALM_AXES, (0.25, 0.80)),
]
TACTILE_BODIES = sorted({g[4] for g in SENSOR_GROUPS})
