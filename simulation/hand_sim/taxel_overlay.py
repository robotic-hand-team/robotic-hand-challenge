"""Real-time tactile heat map drawn into the MuJoCo viewer's user scene (viewer.user_scn).

Every simulated taxel (config.SENSOR_GROUPS) becomes a small flat tile on the hand, coloured with
the same green -> yellow -> orange -> red ramp as hardware/tools/hand_gui.py. Values are
baseline-corrected per taxel, so an untouched sensor reads ~0 (faint green) and contact lights up
opaque. The tiles are real geoms drawn with the 3D scene: rotate/zoom and they stay glued to the hand.

Patch geometry is fitted from the *visible* hand mesh (geom group 1) rather than the collision mesh
the tactile model senses on, because collision meshes are simplified primitives that sit up to ~3 mm
inside the drawn hand -- tiles fitted to them would be hidden.

Usage inside a passive viewer loop:
    overlay = TaxelOverlay(world)
    ...
    overlay.update(viewer.user_scn, world.tactile.step(world.d))  # each frame
"""

import mujoco
import numpy as np

from . import config as C
from .tactile import _body_local_points, _fit_patch

VISUAL_GROUP = 1           # hand mesh as drawn (group 2 is the hidden collision geometry)
SURFACE_OFFSET_M = 0.002   # lift tiles off the surface so they do not z-fight the mesh
TILE_HALF_THICK_M = 0.0012
IDLE_ALPHA = 0.20
ACTIVE_ALPHA = 0.95
ACTIVE_VALUE = C.TACTILE_CONTACT_THRESHOLD  # >= this counts as real contact

# Same ramp as hand_gui.draw(); thresholds are fractions of the raw range.
_COLOR_STEPS = (
    (0.00, (0.00, 0.67, 0.00)),  # green
    (0.20, (0.80, 0.80, 0.00)),  # yellow
    (0.50, (1.00, 0.53, 0.00)),  # orange
    (0.80, (1.00, 0.00, 0.00)),  # red
)


def _color(fraction: float, active: bool) -> np.ndarray:
    rgb = _COLOR_STEPS[0][1]
    for threshold, color in _COLOR_STEPS:
        if fraction >= threshold:
            rgb = color
    return np.array([*rgb, ACTIVE_ALPHA if active else IDLE_ALPHA])


class TaxelOverlay:
    def __init__(self, world):
        self.data = world.d
        model = world.m
        body_points = {}
        self.groups = []
        # world.tactile.groups is in the same order as C.SENSOR_GROUPS; reuse its per-taxel offsets.
        for (name, _, rows, cols, body_name, axes, row_frac), sensed in zip(C.SENSOR_GROUPS,
                                                                           world.tactile.groups):
            body_id = model.body(body_name).id
            if body_id not in body_points:
                body_points[body_id] = _body_local_points(model, body_id, VISUAL_GROUP)
            row_axis = np.array(axes["row"], dtype=float)
            col_axis = np.array(axes["col"], dtype=float)
            normal = np.array(axes["normal"], dtype=float)
            center, row_extent, col_extent = _fit_patch(body_points[body_id], row_axis, col_axis,
                                                        normal, row_frac)
            # Column-major rotation: local x=row, y=col, z=normal (outward).
            mat = np.stack([row_axis, col_axis, normal], axis=1).flatten()

            row_f = (np.arange(rows) + 0.5) / rows - 0.5
            col_f = (np.arange(cols) + 0.5) / cols - 0.5
            rr, cc = np.meshgrid(row_f, col_f, indexing="ij")
            local = (center + rr[..., None] * row_axis * row_extent
                     + cc[..., None] * col_axis * col_extent
                     + normal * SURFACE_OFFSET_M).reshape(-1, 3)

            self.groups.append({
                "name": name,
                "body_id": body_id,
                "offset": sensed["offset"].reshape(-1),
                "local": local,
                "mat": mat,
                "size": np.array([row_extent / rows / 2, col_extent / cols / 2, TILE_HALF_THICK_M]),
            })

    @property
    def n_taxels(self) -> int:
        return sum(len(g["local"]) for g in self.groups)

    def update(self, scene, raw: dict, stride: int = 1, clear: bool = True):
        """Rebuild the overlay from TactileModel.step() output. Call under the viewer lock.

        clear: reset scene.ngeom first (the viewer's user_scn); False appends to the current scene.
        """
        d, geoms = self.data, scene.geoms
        n = 0 if clear else scene.ngeom
        capacity = len(geoms)
        for g in self.groups:
            values = np.clip(raw[g["name"]].reshape(-1) - g["offset"], 0.0, None)
            if stride > 1:
                values = values[::stride]
            xmat = d.xmat[g["body_id"]].reshape(3, 3)
            world_pos = g["local"][::stride] @ xmat.T + d.xpos[g["body_id"]]

            for i in range(len(values)):
                if n >= capacity:
                    scene.ngeom = n
                    return
                value = values[i]
                mujoco.mjv_initGeom(geoms[n], mujoco.mjtGeom.mjGEOM_BOX, g["size"],
                                    np.ascontiguousarray(world_pos[i]), g["mat"],
                                    _color(value / C.TACTILE_RAW_MAX, value >= ACTIVE_VALUE))
                n += 1
        scene.ngeom = n
