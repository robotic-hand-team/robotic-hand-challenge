"""Real-time thermal-imaging tactile heat map for the MuJoCo viewer (viewer.set_images).

Renders the hand as a thermal camera would: a smooth, continuous field over the whole hand -- one
Gaussian splat per taxel, weighted by its pressure -- mapped through a thermal colormap and masked
to the hand's silhouette. Nothing is drawn as a grid of squares, so neighbouring sensors blend into
gradients.

    inactive hand            -> deep blue / purple
    light contact            -> green / yellow transition bands
    hard contact, saturation -> orange / red with a yellow-white hot core

Sensor placement reuses the layout in hardware/tools/hand_gui.py (so the robot's real tactile layout
and the simulation line up), and the silhouette + line art come from frontend/public/hand.png.

Cost is kept off the physics loop: the field is splatted only around taxels that are actually
loaded, the colormap and shading run at panel resolution, and the tactile sample is the only part
taken under the viewer lock.

Usage inside a passive viewer loop:
    panel = TactilePanel(world, height=300)
    viewer.set_images([(panel.rect(viewer), panel.render(world.tactile.step(world.d)))])
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import config as C

CANVAS_W, CANVAS_H = 700, 900   # layout basis, matches hardware/tools/hand_gui.py
PIXEL = 6                       # taxel pitch on the layout canvas
PHOTO = Path(__file__).resolve().parents[2] / "frontend" / "public" / "hand.png"

SUPERSAMPLE = 2                 # splat field is this much finer than the panel, then downscaled
GLOW_TAXELS = 5.0               # splat radius, in taxel pitches: blends neighbouring sensors
SPLAT_FLOOR = 12.0              # raw counts below this are idle and not splatted at all
SIGNAL_DECAY = 1.0              # counts that map to the top of the ramp, in units of TACTILE_RAW_MAX
FIELD_GAMMA = 0.75              # <1 lifts light contact into the visible bands (the peak is capped)
INK_THRESHOLD = 205             # gray level below which hand.png counts as line art
SEAL_RADIUS = 3                 # close gaps of this many px before filling, then open again
ARTWORK_GAIN = 0.75             # how strongly the line art darkens the glow (0 = invisible)

# name, rows, cols, body key, offset relative to the body key anchor (copied from hand_gui.py)
GROUPS = [
    ("little_tip", 3, 3, "Little", (8, 10)),
    ("little_nail", 12, 8, "Little", (-2, 40)),
    ("little_pad", 10, 8, "Little", (4, 190)),
    ("ring_tip", 3, 3, "Ring", (-9, -20)),
    ("ring_nail", 12, 8, "Ring", (-23, 30)),
    ("ring_pad", 10, 8, "Ring", (-18, 200)),
    ("middle_tip", 3, 3, "Middle", (-21, -25)),
    ("middle_nail", 12, 8, "Middle", (-39, 22)),
    ("middle_pad", 10, 8, "Middle", (-35, 210)),
    ("index_tip", 3, 3, "Index", (-38, -19)),
    ("index_nail", 12, 8, "Index", (-49, 17)),
    ("index_pad", 10, 8, "Index", (-54, 200)),
    ("thumb_tip", 3, 3, "Thumb", (185, 140)),
    ("thumb_nail", 12, 8, "Thumb", (145, 170)),
    ("thumb_mid", 3, 3, "Thumb", (90, 240)),
    ("thumb_pad", 12, 8, "Thumb", (-30, 240)),
    ("palm", 8, 14, "Palm", (-15, 0)),
]

ANCHORS = {                     # copied from hand_gui.py
    "Little": (150, 120),
    "Ring": (230, 110),
    "Middle": (310, 100),
    "Index": (390, 110),
    "Thumb": (420, 360),
    "Palm": (240, 520),
}

# Thermal camera ramp: cold blue/purple -> cyan -> green -> yellow -> orange -> red hot.
_THERMAL_STOPS = (
    (0.00, (0.11, 0.09, 0.44)),
    (0.15, (0.16, 0.15, 0.72)),
    (0.30, (0.09, 0.42, 0.92)),
    (0.44, (0.08, 0.78, 0.76)),
    (0.56, (0.33, 0.90, 0.34)),
    (0.68, (0.84, 0.95, 0.14)),
    (0.78, (1.00, 0.86, 0.10)),
    (0.87, (1.00, 0.56, 0.05)),
    (0.94, (0.95, 0.24, 0.05)),
    (1.00, (1.00, 0.78, 0.32)),
)


def thermal_lut(n: int = 256) -> np.ndarray:
    """(n, 3) uint8 lookup table for the thermal ramp."""
    t = np.linspace(0.0, 1.0, n)
    stops = np.array([s[0] for s in _THERMAL_STOPS])
    channels = [np.interp(t, stops, [c[k] for _, c in _THERMAL_STOPS]) for k in range(3)]
    return (np.stack(channels, axis=1).clip(0.0, 1.0) * 255.0).astype(np.uint8)


class TactilePanel:
    """Builds the thermal image; call render() once per sensor sample."""

    def __init__(self, world, height: int = 300, gain: float = 1.0):
        self.height = int(height)
        self.width = int(round(CANVAS_W * self.height / CANVAS_H))
        self.gain = float(gain)
        self._fh, self._fw = self.height * SUPERSAMPLE, self.width * SUPERSAMPLE
        self._lut = thermal_lut()
        self._mask, self._shade = self._artwork()
        # tactile model groups are in config.SENSOR_GROUPS order: reuse their baseline offsets
        self._offsets = {g["name"]: np.asarray(g["offset"], dtype=np.float32).reshape(-1)
                         for g in world.tactile.groups}
        # taxel pixel positions on the (supersampled) splat field
        sx, sy = self._fw / CANVAS_W, self._fh / CANVAS_H
        self._taxels = []
        for name, rows, cols, key, (rx, ry) in GROUPS:
            x0, y0 = ANCHORS[key][0] + rx, ANCHORS[key][1] + ry
            xs = np.clip(((x0 + (np.arange(cols) + 0.5) * PIXEL) * sx).astype(np.int32), 0, self._fw - 1)
            ys = np.clip(((y0 + (np.arange(rows) + 0.5) * PIXEL) * sy).astype(np.int32), 0, self._fh - 1)
            xx, yy = np.meshgrid(xs, ys)
            self._taxels.append((name, xx.ravel(), yy.ravel()))
        # one small Gaussian splat per taxel, peak 1.0 at its centre
        radius = max(1, int(round(GLOW_TAXELS * PIXEL * sy)))
        taps = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        self._dy = np.array([t[0] for t in taps], dtype=np.int32)
        self._dx = np.array([t[1] for t in taps], dtype=np.int32)
        sigma = 0.5 * radius
        d2 = (self._dy ** 2 + self._dx ** 2).astype(np.float32)
        self._splat = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)

    def render(self, raw: dict) -> np.ndarray:
        """Thermal RGB uint8 image (height, width, 3) for viewer.set_images()."""
        field = self._splat_field(raw)
        if (field.shape[0], field.shape[1]) != (self.height, self.width):
            field = np.asarray(Image.fromarray(field, mode="F").resize((self.width, self.height),
                                                                       Image.BILINEAR))
        norm = np.clip(self.gain * field / (SIGNAL_DECAY * C.TACTILE_RAW_MAX), 0.0, 1.0) ** FIELD_GAMMA
        scaled = (norm * 255.0).astype(np.uint8)
        rgb = self._lut[scaled].astype(np.float32) * self._shade[..., None]
        rgb[~self._mask] = 0.0
        return rgb.astype(np.uint8)

    def _splat_field(self, raw: dict) -> np.ndarray:
        """Blend one Gaussian splat per loaded taxel and keep the strongest value per pixel.

        Taking the max (rather than summing) means a taxel's peak survives as exactly its own
        reading and falls off as a smooth Gaussian, so contacts spread out into gradients instead of
        piling up into flat, blown-out blobs. Only pixels next to a loaded sensor are touched.
        """
        field = np.zeros(self._fh * self._fw, dtype=np.float32)
        for name, xx, yy in self._taxels:
            values = np.clip(raw[name].reshape(-1) - self._offsets[name], 0.0, None).astype(np.float32)
            keep = values > SPLAT_FLOOR
            if not keep.any():
                continue
            values, xx, yy = values[keep], xx[keep], yy[keep]
            rows = np.clip(yy[:, None] + self._dy[None, :], 0, self._fh - 1)
            cols = np.clip(xx[:, None] + self._dx[None, :], 0, self._fw - 1)
            np.maximum.at(field, (rows * self._fw + cols).ravel(),
                          (values[:, None] * self._splat[None, :]).ravel())
        return field.reshape(self._fh, self._fw)

    def _artwork(self):
        """Hand silhouette mask and line-art shading, both at panel resolution."""
        if not PHOTO.exists():
            ones = np.ones((self.height, self.width), dtype=np.float32)
            return ones > 0, ones
        gray = Image.open(PHOTO).convert("L").resize((CANVAS_W, CANVAS_H))
        # ink = line art. Dilate it to seal gaps in the outline, fill the enclosed hand, then erode
        # back by the same amount -- a plain fill leaks into the palm where the outline has a break.
        size = 2 * SEAL_RADIUS + 1
        ink = gray.point(lambda v: 0 if v < INK_THRESHOLD else 255).filter(ImageFilter.MinFilter(size))
        flooded = ink.convert("RGB")
        for seed in ((0, 0), (CANVAS_W - 1, 0), (0, CANVAS_H - 1), (CANVAS_W - 1, CANVAS_H - 1)):
            ImageDraw.floodfill(flooded, seed, (255, 0, 0), thresh=0)
        outside = np.all(np.asarray(flooded) == (255, 0, 0), axis=2)
        hand = Image.fromarray((~outside).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(size))
        mask = hand.resize((self.width, self.height))
        shade = gray.resize((self.width, self.height))
        g = np.asarray(shade, dtype=np.float32) / 255.0
        return np.asarray(mask) > 127, ARTWORK_GAIN + (1.0 - ARTWORK_GAIN) * g

    def rect(self, viewer, margin: int = 12):
        """MjrRect for the panel, `margin` px inside the viewer's 3D viewport (bottom-left).

        Anchoring to the viewport keeps the panel clear of MuJoCo's own left UI panel, which shares
        the window's bottom-left corner. The viewport moves when the window is resized, so call this
        each time the panel is refreshed.
        """
        import mujoco
        vp = viewer.viewport
        left = int(getattr(vp, "left", 0) or 0)
        bottom = int(getattr(vp, "bottom", 0) or 0)
        return mujoco.MjrRect(left + margin, bottom + margin, self.width, self.height)
