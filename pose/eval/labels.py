"""Photo labels for 13 real grasps (front-view photos from the session videos, 2026-10-17).

Frame: hand_base_link (URDF), as used by pose/. +z = little-finger side, -z = thumb side,
-y = toward the fingertips ("up" in the photos), +y = toward the wrist, -x = out of the palm.
end_dir: direction of the sensor's CAD +y end, read from the photo (in-palm-plane only):
  VEGABAR 29: yellow electrical end (the other end has thread, hex and green O-ring)
  VEGABAR 38: yellow head (the other end is the steel shaft with the white cap)
Uncertainty of a photo reading: roughly +-15 deg.
"""
import numpy as np
LABELS = {
    "20261016_204315": ("vegabar29", [0, -0.09, -1.0], "across; O-ring end out on the little side"),
    "20261016_204344": ("vegabar29", [0, 0.0, 1.0], "across, level; yellow end out on the little side"),
    "20261016_204402": ("vegabar29", [0, 0.42, 0.91], "across, tilted ~25 deg; yellow end little side, lower"),
    "20261016_204429": ("vegabar29", [0, -0.87, -0.5], "diagonal ~30 deg; O-ring end down past the palm"),
    "20261016_204457": ("vegabar29", [0, -1.0, 0.0], "along, vertical; O-ring end down"),
    "20261016_204513": ("vegabar29", [0, 1.0, 0.0], "along, vertical; yellow end down"),
    "20261016_204524": ("vegabar29", [0, -0.71, -0.71], "diagonal ~45 deg; O-ring end low on the little side"),
    "20261016_204558": ("vegabar38", [0, 0.0, -1.0], "across; head at the thumb (thumb on it)"),
    "20261016_204609": ("vegabar38", [0, -1.0, 0.0], "along, vertical; head up"),
    "20261016_204619": ("vegabar38", [0, 0.0, 1.0], "across, level; head out on the little side"),
    "20261016_204627": ("vegabar38", [0, 0.5, -0.87], "diagonal ~30 deg; head low on the thumb side"),
    "20261016_204637": ("vegabar38", [0, 0.0, -1.0], "across; head at the thumb"),
    "20261016_204649": ("vegabar38", [0, -0.97, 0.26], "along, ~15 deg; head up"),
}
def end_dir(sess):
    v = np.array(LABELS[sess][1], float)
    return v / np.linalg.norm(v)
