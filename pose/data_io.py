"""Load one logged grasp session and pick the steady holding window."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
        return vals.reshape(group["rows"], group["cols"])


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
