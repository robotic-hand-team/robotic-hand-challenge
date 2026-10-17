"""Load one logged grasp session and pick the steady holding window."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config as C
import pandas as pd


@dataclass
class Session:
    name: str
    meta: dict
    actuators: pd.DataFrame
    tactile: pd.DataFrame          # raw taxel columns
    groups: list                   # sensor_groups from meta.json

    def region_grid(self, frames, group):
        """Mean taxel grid (rows x cols) of one region over the given frames."""
        key = group["name"].lower().replace(" ", "_")
        cols = [f"{key}_r{r}_c{c}" for r in range(group["rows"]) for c in range(group["cols"])]
        vals = self.tactile.loc[frames, cols].to_numpy(float).mean(0)
        if group["name"] == "Palm" and C.PALM_FIX_LAYOUT:
            return palm_grid(vals)
        return vals.reshape(group["rows"], group["cols"])


def palm_grid(vals):
    """The logger labels the palm as 8 x 14 (r0_c0, r0_c1, ...), but the values are really a 14 x 8
    grid: read that way, pressure patterns are coherent; read as 8 x 14 they form repeating diagonal
    stripes. The 14-taxel direction runs across the palm, row 0 at the little finger (the heavy
    VEGABAR 38 head presses rows 0-4 when it lies on the little-finger side, rows 9-13 at the thumb).
    Returned as 8 x 14 (along the fingers x across the palm, thumb side first) to match
    hand_model._palm_patch, which spans PALM_LENGTH_RANGE x PALM_WIDTH_RANGE."""
    g = np.asarray(vals, float).reshape(14, 8)[::-1, :].T     # -> 8 x 14, column 0 = thumb side
    return g[::-1, :] if C.PALM_FLIP_ALONG else g


def load_session(folder: Path) -> Session:
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    act = pd.read_csv(folder / "actuators.csv")
    tac = pd.read_csv(folder / "tactile_raw.csv")
    return Session(folder.name, meta, act, tac, meta["sensor_groups"])


def hold_window(s: Session, closed_below=600, frac_of_peak=0.7):
    """Frames where the fingers are closed and pressure is near its peak.

    Returns row indices. Empty if the session never grasps anything.
    """
    fingers = ["little", "ring", "middle", "index"]
    closed = (s.actuators[[f"angle_act_{f}" for f in fingers]].min(axis=1) < closed_below)
    total = s.tactile.iloc[:, 3:].sum(axis=1)
    if total[closed].empty or total[closed].max() < 5000:
        return np.array([], dtype=int)
    good = closed & (total >= frac_of_peak * total[closed].max())
    idx = np.flatnonzero(good.to_numpy())
    if len(idx) > 5:                      # skip the first frames while fingers settle
        idx = idx[len(idx) // 5:]
    return idx


def joint_raw(s: Session, frames):
    """Median raw actuator angle per DOF over the window."""
    return {d: float(s.actuators.loc[frames, f"angle_act_{d}"].median())
            for d in s.meta["dof"]}
