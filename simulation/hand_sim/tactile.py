"""Simulated tactile sensors matching the real hand's 17 register groups (hand_manual.pdf 2.6.20).

TactileModel derives each patch's position and size from the live mesh at startup (see
config.SENSOR_GROUPS), then every step distributes nearby contact forces onto the patch's grid,
converts the result to raw 0-4095 units, and computes the same per-group summary statistics as
models/dataset/*/tactile_summary.csv.
"""

import mujoco
import numpy as np

from . import config as C

# Contact force -> raw unit conversion. Picked so a ~1 N fingertip contact (typical light touch)
# reads a few hundred raw units, similar in scale to models/dataset; not calibrated against hardware.
NEWTONS_TO_RAW = 900.0
BLUR_SIGMA_CELLS = 0.9  # contact force spreads onto neighbouring taxels (real sensors aren't points)
NOISE_STD_RAW = 3.0


def _body_local_points(model, body_id, group=2):
    """All geom surface points (mesh vertices or primitive-geom corners) in body frame."""
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id or model.geom_group[g] != group:
            continue
        gtype = mujoco.mjtGeom(model.geom_type[g])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        rot = rot.reshape(3, 3)
        size = model.geom_size[g]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            start, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            local = model.mesh_vert[start : start + n]
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            local = np.array([[x, y, z] for x in (-size[0], size[0]) for y in (-size[0], size[0]) for z in (-size[1], size[1])])
        else:  # box (and a crude box fallback for anything else)
            local = np.array([[x, y, z] for x in (-size[0], size[0]) for y in (-size[1], size[1]) for z in (-size[2], size[2])])
        pts.append(local @ rot.T + model.geom_pos[g])
    return np.concatenate(pts)


def _fit_patch(points, row_axis, col_axis, normal_axis, row_frac):
    row_proj, col_proj, nrm_proj = points @ row_axis, points @ col_axis, points @ normal_axis
    row_lo, row_hi = row_proj.min(), row_proj.max()
    f0, f1 = row_frac
    row_center = row_lo + (f0 + f1) / 2 * (row_hi - row_lo)
    row_extent = (f1 - f0) * (row_hi - row_lo)
    c0, c1 = C.COL_FRAC
    col_lo, col_hi = col_proj.min(), col_proj.max()
    col_center = col_lo + (c0 + c1) / 2 * (col_hi - col_lo)
    col_extent = (c1 - c0) * (col_hi - col_lo)
    normal_pos = np.percentile(nrm_proj, 90)  # outer surface, robust to thin mesh edges/outliers
    center_local = row_center * row_axis + col_center * col_axis + normal_pos * normal_axis
    return center_local, max(row_extent, 1e-4), max(col_extent, 1e-4)


class TactileModel:
    def __init__(self, model: mujoco.MjModel, rng: np.random.Generator | None = None):
        self.m = model
        self.rng = rng or np.random.default_rng()
        self.groups = []  # list of dicts, one per config.SENSOR_GROUPS entry
        body_points = {}
        for name, addr, rows, cols, body_name, axes, row_frac in C.SENSOR_GROUPS:
            body_id = model.body(body_name).id
            if body_id not in body_points:
                body_points[body_id] = _body_local_points(model, body_id)
            row_axis, col_axis, normal_axis = (np.array(axes[k], dtype=float) for k in ("row", "col", "normal"))
            center, row_extent, col_extent = _fit_patch(body_points[body_id], row_axis, col_axis, normal_axis, row_frac)
            self.groups.append({
                "name": name, "addr": addr, "rows": rows, "cols": cols, "body_id": body_id,
                "center_local": center, "row_axis": row_axis, "col_axis": col_axis,
                "row_extent": row_extent, "col_extent": col_extent,
                "offset": self.rng.uniform(0, 20, (rows, cols)),  # per-taxel baseline, like meta.json
            })
        self.by_body = {}
        for g in self.groups:
            self.by_body.setdefault(g["body_id"], []).append(g)
        self._f6 = np.zeros(6)

    def step(self, data: mujoco.MjData) -> dict:
        """Returns {group_name: (rows, cols) float array of raw units in [0, TACTILE_RAW_MAX]}."""
        force_n = {g["name"]: np.zeros((g["rows"], g["cols"])) for g in self.groups}
        for c_id in range(data.ncon):
            con = data.contact[c_id]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            mujoco.mj_contactForce(self.m, data, c_id, self._f6)
            normal_force = float(self._f6[0])
            if normal_force <= 0:
                continue
            for body_id in (b1, b2):
                for g in self.by_body.get(body_id, []):
                    xmat = data.xmat[body_id].reshape(3, 3)
                    local = xmat.T @ (con.pos - data.xpos[body_id])
                    u = np.dot(local - g["center_local"], g["row_axis"])
                    v = np.dot(local - g["center_local"], g["col_axis"])
                    if abs(u) > g["row_extent"] / 2 or abs(v) > g["col_extent"] / 2:
                        continue
                    ri = (u / g["row_extent"] + 0.5) * g["rows"]
                    ci = (v / g["col_extent"] + 0.5) * g["cols"]
                    _splat(force_n[g["name"]], ri, ci, normal_force)

        raw = {}
        for g in self.groups:
            vals = force_n[g["name"]] * NEWTONS_TO_RAW + g["offset"]
            vals += self.rng.normal(0, NOISE_STD_RAW, vals.shape)
            raw[g["name"]] = np.clip(vals, 0, C.TACTILE_RAW_MAX)
        return raw

    @staticmethod
    def summary(raw: dict) -> dict:
        """Per-group sum/max/contacts/centroid/force, matching models/dataset/*/tactile_summary.csv."""
        out = {}
        for name, grid in raw.items():
            active = grid > C.TACTILE_CONTACT_THRESHOLD
            n = int(active.sum())
            row = {"sum": round(float(grid.sum())), "max": round(float(grid.max())), "contacts": n}
            if n >= 2:
                ys, xs = np.where(active)
                row["cy"], row["cx"] = round(float(ys.mean()), 2), round(float(xs.mean()), 2)
                row["force"] = round(float(grid[active].mean()), 1)
            else:
                row["cy"], row["cx"], row["force"] = "", "", 0.0
            out[name] = row
        return out


def _splat(grid: np.ndarray, ri: float, ci: float, value: float):
    """Spread `value` onto grid around the fractional cell (ri, ci) with a small Gaussian blur."""
    rows, cols = grid.shape
    r0, c0 = int(np.floor(ri)), int(np.floor(ci))
    s = BLUR_SIGMA_CELLS
    weights = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < rows and 0 <= c < cols:
                w = np.exp(-((r + 0.5 - ri) ** 2 + (c + 0.5 - ci) ** 2) / (2 * s * s))
                weights.append((r, c, w))
    total = sum(w for *_, w in weights) or 1.0
    for r, c, w in weights:
        grid[r, c] += value * w / total
