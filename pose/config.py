"""Central settings for the tactile pose prototype.

Everything marked PLACEHOLDER must be checked against the real hardware,
the VEGA manual (RH56DFTP user manual) or hand.png before trusting results.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Data ----------------------------------------------------------------- 
REPO_ROOT = ROOT.parent
LOG_DIR = REPO_ROOT / "models" / "dataset"                           # folder with session subfolders
OUTPUT_DIR = ROOT / "output"

# Which sensor was grasped in each session (stands in for the classifier).
# Sessions not listed here are skipped when running with --object auto.
SESSION_LABELS = {
    "20261016_204344": "A", "20261016_204402": "A", "20261016_204429": "A",
    "20261016_204457": "A", "20261016_204513": "A", "20261016_204524": "A",
    "20261016_204548": "A",
    "20261016_204558": "B", "20261016_204609": "B", "20261016_204619": "B",
    "20261016_204627": "B", "20261016_204637": "B", "20261016_204649": "B",
}
# PLACEHOLDER: which STL is Sensor A and which is Sensor B? Swap if wrong.
LABEL_TO_OBJECT = {"A": "vegabar29", "B": "vegabar38"}

# --- Hand model -----------------------------------------------------------
# Download the WHOLE folder (urdf + meshes/) from
# https://github.com/dexsuite/dex-urdf  ->  robots/hands/inspire_hand/
# PLACEHOLDER: set HAND_SIDE to the hand you actually used ("right" or "left").
HAND_SIDE = "right"
HAND_URDF = ROOT / "inspire_hand" / f"inspire_hand_{HAND_SIDE}.urdf"

# Raw actuator value -> joint angle.  Inspire convention: 1000 = open, 0 = closed.
# joint = (1000 - raw) / 1000 * joint_upper_limit  (linear approximation)
# PLACEHOLDER: replace with the manual's angle table if it gives one.
RAW_OPEN = 1000.0
# Fraction of the URDF joint range reached at raw 0 (fully closed). 1.0 = model's maximum bend.
# PLACEHOLDER: calibrate by photographing a fully closed finger from the side.
FINGER_ANGLE_SCALE = 1.0
THUMB_ANGLE_SCALE = 1.0
DOF_TO_JOINT = {
    "little": "pinky_proximal_joint",
    "ring": "ring_proximal_joint",
    "middle": "middle_proximal_joint",
    "index": "index_proximal_joint",
    "thumb_bend": "thumb_proximal_pitch_joint",
    "thumb_rot": "thumb_proximal_yaw_joint",
}

# --- Objects --------------------------------------------------------------
# PLACEHOLDER: put VEGA's STL files here (from the challenge attachments).
# The pressure transducer only comes as STEP -> convert to STL first
# (FreeCAD: File > Export > STL, or CadQuery).
OBJECTS = {
    "vegabar29": ROOT / "objects" / "vegabar29.stl",
    "vegabar38": ROOT / "objects" / "vegabar38.stl",
    "druckaufnehmer": ROOT / "objects" / "druckaufnehmer.stl",
  
}
OBJECT_UNITS_TO_M = 0.001     # CAD is in mm, URDF is in metres
OBJECT_MAX_FACES = 10**9      # decimate for fast fitting

# --- Tactile layout --------------------------------------------------------
CONTACT_THRESHOLD = 50        # taxel value counted as contact (from meta.json)
MIN_REGION_SUM = 200          # ignore regions with less total pressure

# Where each tactile region sits on the hand model.
# link: URDF link carrying the patch
# span: (start, end) along the link as a fraction of its length (0 = joint, 1 = tip)
# width: fraction of the link width covered by the patch
# PLACEHOLDER: spans and row direction are estimates; check against hand.png.
REGION_LAYOUT = {
    "tip":  {"segment": "distal",   "span": (0.82, 0.98), "width": 0.6},
    "nail": {"segment": "distal",   "span": (0.10, 0.78), "width": 0.8},
    "pad":  {"segment": "proximal", "span": (0.15, 0.85), "width": 0.8},
    "mid":  {"segment": "middle",   "span": (0.20, 0.80), "width": 0.6},  # thumb only
}
ROW0_AT_TIP = True            # PLACEHOLDER: is taxel row 0 at the fingertip end?

FINGER_LINKS = {
    "little": {"proximal": "pinky_proximal", "distal": "pinky_intermediate"},
    "ring":   {"proximal": "ring_proximal",  "distal": "ring_intermediate"},
    "middle": {"proximal": "middle_proximal", "distal": "middle_intermediate"},
    "index":  {"proximal": "index_proximal", "distal": "index_intermediate"},
    "thumb":  {"proximal": "thumb_proximal", "middle": "thumb_intermediate",
               "distal": "thumb_distal"},
}
# Palm patch, in hand_base_link coordinates (metres). PLACEHOLDER.
# Palm patch along the fingers (row 0 near the wrist). Equal taxel pitch with the 14 taxels across
# (~5 mm): 8 taxels ~ 36 mm. Centre at y = -80 mm chosen against 13 photo-labelled grasps (pose/eval).
PALM_LENGTH_RANGE = (-0.062, -0.098)
PALM_WIDTH_RANGE = (-0.036, 0.030)     # across the palm, index side to little side
# The palm's raw values are a 14 x 8 grid (logged as 8 x 14); see data_io.palm_grid.
PALM_FIX_LAYOUT = True
PALM_FLIP_ALONG = False                 # direction of the 8-taxel axis (fingers <-> wrist), set from photo labels

# --- Fitting --------------------------------------------------------------
N_ORIENTATION_STARTS = 6
N_REFINE = 6                   # best distinct candidates refined against the dense hand surface
NORMAL_WEIGHT = 0.004          # metres-equivalent weight of normal mismatch
PENETRATION_WEIGHT = 20.0
MAX_PENETRATION_MM = 5.0     # allowed overlap with the hand (skin compression); more = rejected
PRIOR_WEIGHT = 0.005           # weak pull of object centre toward contact centroid
GOOD_FIT_RMS_MM = 2.0
CONTACT_WEIGHT_FLOOR = 1.0

# Rough grasp type, as a hint for the fit. Several grasps fit the contacts equally well
# (e.g. three fingertips in a row), so this tells the fit which family to prefer.
#   None           -> no preference
#   "along_palm"   -> sensor parallel to the fingers (hangs below curled fingertips)
#   "across_palm"  -> sensor lies across the palm, fingers wrapped around it
#   "out_of_palm"  -> sensor points away from the palm
# Override per run with --axis-hint. Per-session hints go in SESSION_AXIS_HINTS.
AXIS_HINT = None
AXIS_HINT_WEIGHT = 0.03        # strength of the hint (soft: data can still win)
SESSION_AXIS_HINTS = {
    # Fill in from your photos: "along_palm" or "across_palm" per session
    "20261016_204513": "along_palm",     # photo: hangs below curled fingertips
}
