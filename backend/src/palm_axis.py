"""The line along which a held object presses into the palm, measured from the palm's 8x14 taxels.

A cylinder lying across the palm leaves an elongated strip of pressure; the strip's main axis (second
moments of the pressure) is the object's axis line in the palm plane. Pure geometry, no training: on the 37
photo-labelled left-hand grasps it gets the axis line within 30 deg as often as the learned direction model,
and unlike that model it also gets the grasps along the fingers right. It can't tell the object's two ends
apart (the strip is near-symmetric), so the end still comes from the learned model.

Input: the palm group's 112 offset-corrected taxel values in data order, as hardware/config.py holds them
(tactile["Palm"].ravel()) and as tactile_raw.csv logs them (palm_r0_c0 ... palm_r7_c13). Data point k is
physical row 7 - k % 8, column k // 8 (the palm is wired column-major, bottom to top).
"""
import numpy as np

CONTACT = 50          # taxel value above the zero offset that counts as contact (GUI / summary threshold)
MIN_ELONGATION = 5.0  # strip length^2 / width^2 below this: a blob, not a line - no reliable axis
MIN_TOTAL = 5000.0    # summed pressure below this: barely touching the palm - no reliable axis


def palm_line(values) -> tuple[float, float, float]:
    """(line_deg, elongation, total). line_deg in (-90, 90] in the hand's palm plane before calibration
    (0 = across the palm, +-90 = along the fingers); nan if the palm is not touched."""
    v = np.asarray(values, float)
    if v.size != 112:
        return float("nan"), 0.0, 0.0
    g = np.zeros((8, 14))
    k = np.arange(112)
    g[7 - k % 8, k // 8] = v
    g[g < CONTACT] = 0.0
    tot = g.sum()
    if tot <= 0:
        return float("nan"), 0.0, 0.0
    r, c = np.mgrid[0:8, 0:14]
    mr, mc = (g * r).sum() / tot, (g * c).sum() / tot
    m20 = (g * (c - mc) ** 2).sum() / tot
    m02 = (g * (r - mr) ** 2).sum() / tot
    m11 = (g * (c - mc) * (r - mr)).sum() / tot
    ev = np.linalg.eigvalsh([[m20, m11], [m11, m02]])
    # grid columns run across the palm and rows up it, mirrored against the hand frame: hence the minus
    line = -0.5 * np.degrees(np.arctan2(2 * m11, m20 - m02))
    return float(line), float(ev[1] / max(ev[0], 1e-6)), float(tot)


def palm_point(values, center_mm, pitch_mm):
    """Where the pressure strip lies on the palm, in the hand's palm plane: (y, z) in mm of the right-hand
    model (y towards the thumb, z towards the fingertips), or None if the palm is barely touched.

    The strip runs along the object's axis, so this is a point the object's axis passes over. Grid
    orientation, from the data: higher columns lie towards the little finger (the strip's column centroid
    moves opposite to index-vs-little finger contact, Spearman -0.50 over 48 grasps), and the angle
    mapping (see palm_line) then puts higher rows towards the fingertips. center_mm: the sensor's centre,
    pitch_mm: taxel spacing (hand manual p.26 and config.PALM_SENSOR_*)."""
    v = np.asarray(values, float)
    if v.size != 112:
        return None
    g = np.zeros((8, 14))
    k = np.arange(112)
    g[7 - k % 8, k // 8] = v
    g[g < CONTACT] = 0.0
    tot = g.sum()
    if tot < MIN_TOTAL:
        return None
    r, c = np.mgrid[0:8, 0:14]
    mr, mc = (g * r).sum() / tot, (g * c).sum() / tot
    return (center_mm[0] - (mc - 6.5) * pitch_mm, center_mm[1] + (mr - 3.5) * pitch_mm)


def wrap90(a):
    return (np.asarray(a, float) + 90.0) % 180.0 - 90.0


def reliable(elongation: float, total: float) -> bool:
    return elongation >= MIN_ELONGATION and total >= MIN_TOTAL


def fuse(model_deg: float, line_deg: float, elongation: float, total: float, bias_deg: float) -> float:
    """Direction of the object's end (deg, 0 = thumb, 90 = fingertips): the palm's axis line (calibrated by
    bias_deg) when the palm shows a clear strip, pointed whichever way is closer to the model's direction;
    otherwise the model's direction."""
    if not np.isfinite(line_deg) or not reliable(elongation, total):
        return model_deg
    line = float(wrap90(line_deg + bias_deg))
    return line if abs((line - model_deg + 180) % 360 - 180) <= 90 else line + 180.0
