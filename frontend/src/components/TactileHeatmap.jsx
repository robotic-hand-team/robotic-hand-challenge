import { useEffect, useRef } from "react";

// Per-group tactile heat map on the hand drawing. Layout (anchors, offsets, taxel pitch) is copied from
// hardware/config.py; frames only carry one summed value per sensor group, so each group is one soft patch.
// The patches are merged into one smooth scalar field, coloured cool-to-warm, and drawn with thin contour
// lines. Levels ease toward each new frame, so contact fades in and out instead of flickering.
const PITCH = 6;
const ANCHORS = { Little: [150, 120], Ring: [230, 110], Middle: [310, 100], Index: [390, 110], Thumb: [420, 360], Palm: [240, 520] };
// name, rows, cols, anchor, [dx, dy]
const GROUPS = [
  ["little_tip", 3, 3, "Little", 8, 10], ["little_nail", 12, 8, "Little", -2, 40], ["little_pad", 10, 8, "Little", 4, 190],
  ["ring_tip", 3, 3, "Ring", -9, -20], ["ring_nail", 12, 8, "Ring", -23, 30], ["ring_pad", 10, 8, "Ring", -18, 200],
  ["middle_tip", 3, 3, "Middle", -21, -25], ["middle_nail", 12, 8, "Middle", -39, 22], ["middle_pad", 10, 8, "Middle", -35, 210],
  ["index_tip", 3, 3, "Index", -38, -19], ["index_nail", 12, 8, "Index", -49, 17], ["index_pad", 10, 8, "Index", -54, 200],
  ["thumb_tip", 3, 3, "Thumb", 185, 140], ["thumb_nail", 12, 8, "Thumb", 145, 170], ["thumb_mid", 3, 3, "Thumb", 90, 240],
  ["thumb_pad", 12, 8, "Thumb", -30, 240], ["palm", 8, 14, "Palm", -15, 0],
].map(([name, rows, cols, key, dx, dy]) => {
  const w = cols * PITCH, h = rows * PITCH, x0 = ANCHORS[key][0] + dx, y0 = ANCHORS[key][1] + dy;
  // An elliptical blob inscribed in the patch; small tips stay tight, the big palm gets a wide soft halo.
  return { name, n: rows * cols, cx: x0 + w / 2, cy: y0 + h / 2, rx: w / 2, ry: h / 2, glow: Math.min(26, Math.max(10, Math.min(w, h) * 0.45)) };
});

const VB_W = 700, VB_H = 900;
const CELL = 3;                            // field resolution in viewBox units; the browser interpolates the upscale
const GW = Math.round(VB_W / CELL), GH = Math.round(VB_H / CELL);
const FULL_SCALE = 1500;                   // mean counts per taxel that read as "hot" (VEGABAR 29 ~1500, VEGABAR 38 saturates at ~4094)
const MIN_LEVEL = 0.03;
const BANDS = 6;                           // contour lines every 1/6 of the scale
const EASE = 0.22;                         // fraction of the gap closed per animation frame

// Moreland cool-to-warm diverging scale: blue -> light grey -> red.
const COOL_WARM = [[59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249], [221, 221, 221], [245, 196, 173], [244, 154, 123], [222, 96, 77], [180, 4, 38]];
const LUT = Array.from({ length: 256 }, (_, i) => {
  const t = (i / 255) * (COOL_WARM.length - 1), k = Math.min(Math.floor(t), COOL_WARM.length - 2), f = t - k;
  return COOL_WARM[k].map((v, c) => Math.round(v + (COOL_WARM[k + 1][c] - v) * f));
});
const COOL_WARM_CSS = `linear-gradient(90deg, ${COOL_WARM.map(([r, g, b], i) => `rgb(${r},${g},${b}) ${(i / (COOL_WARM.length - 1)) * 100}%`).join(", ")})`;

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const targetLevel = (g, tactile) => {
  const sum = tactile?.[g.name];
  if (!sum) return 0;
  const level = Math.min(1, Math.sqrt(sum / g.n / FULL_SCALE));
  return level < MIN_LEVEL ? 0 : level;
};

function paint(canvas, levels) {
  const field = new Float32Array(GW * GH);
  for (const g of GROUPS) {
    const level = levels[g.name];
    if (level < 0.005) continue;
    const reach = 2.2 * g.glow;
    const ix0 = Math.max(0, Math.floor((g.cx - g.rx - reach) / CELL)), ix1 = Math.min(GW - 1, Math.ceil((g.cx + g.rx + reach) / CELL));
    const iy0 = Math.max(0, Math.floor((g.cy - g.ry - reach) / CELL)), iy1 = Math.min(GH - 1, Math.ceil((g.cy + g.ry + reach) / CELL));
    for (let iy = iy0; iy <= iy1; iy++) {
      const y = (iy + 0.5) * CELL;
      for (let ix = ix0; ix <= ix1; ix++) {
        const x = (ix + 0.5) * CELL;
        // Distance outside the ellipse (0 inside), then a smooth gaussian fall-off.
        const e = Math.hypot((x - g.cx) / g.rx, (y - g.cy) / g.ry);
        const d = e <= 1 ? 0 : ((e - 1) * Math.min(g.rx, g.ry)) / g.glow;
        const v = level * Math.exp(-d * d);
        if (v > field[iy * GW + ix]) field[iy * GW + ix] = v; // max, not sum: a patch's peak stays its own reading
      }
    }
  }
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(GW, GH);
  for (let i = 0; i < field.length; i++) {
    const v = field[i];
    if (v < 0.01) continue;
    const [r, g, b] = LUT[Math.round(Math.min(1, v) * 255)];
    // Contour lines: darken a thin sliver at the start of every band.
    const band = v * BANDS, line = band > 1 && band - Math.floor(band) < 0.07 ? 0.78 : 1;
    img.data.set([r * line, g * line, b * line, Math.round(255 * 0.94 * smooth(0.01, 0.1, v))], i * 4);
  }
  ctx.putImageData(img, 0, 0);
}

export default function TactileHeatmap({ tactile }) {
  const canvas = useRef(null);
  const levels = useRef(Object.fromEntries(GROUPS.map((g) => [g.name, 0])));
  const raf = useRef(0);

  useEffect(() => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const target = Object.fromEntries(GROUPS.map((g) => [g.name, targetLevel(g, tactile)]));
    cancelAnimationFrame(raf.current);
    const step = () => {
      let moving = false;
      for (const g of GROUPS) {
        const cur = levels.current[g.name], gap = target[g.name] - cur;
        if (reduce || Math.abs(gap) < 0.004) levels.current[g.name] = target[g.name];
        else { levels.current[g.name] = cur + gap * EASE; moving = true; }
      }
      if (canvas.current) paint(canvas.current, levels.current);
      if (moving) raf.current = requestAnimationFrame(step);
    };
    step();
    return () => cancelAnimationFrame(raf.current);
  }, [tactile]);

  return (
    <div className="heatmap">
      <div className="heat-stage">
        <canvas ref={canvas} width={GW} height={GH} className="heat-canvas" role="img" aria-label="Tactile heat map of the hand" />
        {/* multiply drops the drawing's white background and keeps its lines on top of the heat */}
        <img className="heat-hand" src="/hand.png" alt="" />
        {!tactile && <span className="heatmap-empty">Waiting for tactile data</span>}
      </div>
      <div className="heat-legend" aria-hidden="true">
        <span>Light</span>
        <span className="heat-ramp" style={{ background: COOL_WARM_CSS }} />
        <span>Firm</span>
      </div>
    </div>
  );
}
