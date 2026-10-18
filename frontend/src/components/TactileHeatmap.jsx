import { useEffect, useRef, useState } from "react";

// Thermal-camera style tactile heat map. Layout (anchors, offsets, taxel pitch) is copied from
// hardware/config.py; frames only carry one summed value per sensor group, so each group is one soft blob.
// The blobs are merged into a smooth field and laid over the hand silhouette (taken from hand.png), which
// glows cool at the edges and a little warmer in the middle. Contact adds heat on top: blue -> green ->
// yellow -> red. Levels ease toward each new frame, so contact fades in and out instead of flickering.
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
  // An elliptical blob inscribed in the patch with a wide, soft halo; small tips stay tighter than the palm.
  return { name, n: rows * cols, cx: x0 + w / 2, cy: y0 + h / 2, rx: w / 2, ry: h / 2, glow: Math.min(34, Math.max(14, Math.min(w, h) * 0.7)) };
});

const VB_W = 700, VB_H = 900;
const CELL = 2;                            // field resolution in viewBox units; the browser scales the canvas smoothly
const GW = VB_W / CELL, GH = VB_H / CELL;
const INK_THRESHOLD = 205;                 // gray level below which hand.png counts as line art (same as the sim panel)
const SEAL_RADIUS = 3;                     // px; close gaps in the outline before filling (same as the sim panel)

// Auto gain: a group's mean counts per taxel are scaled against the current peak group, so the light grip of
// VEGABAR 29 (peak ~80 counts) and the hard one of VEGABAR 38 (peak ~2000) both use the whole colour range.
// The peak is clamped so sensor noise isn't blown up into a hot spot, and eased so the scale doesn't flicker.
const GAIN_MIN = 150, GAIN_MAX = 1500;     // mean counts per taxel
const MIN_LEVEL = 0.03;
const EASE = 0.22;                         // fraction of the gap closed per animation frame
const BASE_EDGE = 0.05, BASE_CORE = 0.27;  // resting "temperature" of the hand: cool edge, warmer core
const PLATEAU = 0.45;                      // fraction of a patch's radius that stays at full level before the fall-off
const SNAP_DEPTH = 0.55;                   // how far inside the hand (0 edge .. 1 deep) a patch centre must be
const T_MAX = 0.96;                       // hottest colour used: red. The pale yellow-white end would draw bullseyes

// Thermal palette: deep indigo -> blue -> cyan -> green -> yellow -> orange -> red -> hot yellow-white.
const THERMAL = [
  [0.0, [14, 10, 60]], [0.12, [44, 22, 150]], [0.25, [28, 76, 224]], [0.38, [0, 172, 222]], [0.5, [34, 200, 118]],
  [0.62, [196, 230, 44]], [0.72, [255, 208, 22]], [0.84, [255, 118, 16]], [0.93, [232, 34, 30]], [1.0, [255, 238, 170]],
];
const LUT = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let k = 1;
  while (k < THERMAL.length - 1 && t > THERMAL[k][0]) k++;
  const [t0, c0] = THERMAL[k - 1], [t1, c1] = THERMAL[k], f = (t - t0) / (t1 - t0);
  for (let c = 0; c < 3; c++) LUT[i * 3 + c] = Math.round(c0[c] + (c1[c] - c0[c]) * f);
}
const THERMAL_CSS = `linear-gradient(90deg, ${THERMAL.slice(1).map(([t, [r, g, b]]) => `rgb(${r},${g},${b}) ${Math.round(t * 100)}%`).join(", ")})`;

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const meanOf = (g, tactile) => (tactile?.[g.name] ?? 0) / g.n;
const toLevel = (mean, gain) => {
  const level = Math.min(1, Math.sqrt(mean / gain));
  return level < MIN_LEVEL ? 0 : level;
};

// ---- Hand silhouette ---------------------------------------------------------------------------------
function boxBlur(src, w, h, r) {
  const k = 2 * r + 1, tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  const at = (i, n) => Math.min(n - 1, Math.max(0, i));
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let x = -r; x <= r; x++) acc += src[row + at(x, w)];
    for (let x = 0; x < w; x++) { tmp[row + x] = acc / k; acc += src[row + at(x + r + 1, w)] - src[row + at(x - r, w)]; }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) acc += tmp[at(y, h) * w + x];
    for (let y = 0; y < h; y++) { out[y * w + x] = acc / k; acc += tmp[at(y + r + 1, h) * w + x] - tmp[at(y - r, h) * w + x]; }
  }
  return out;
}

function dilate(src, w, h, r) { // square window max on a 0/1 mask
  const tmp = new Uint8Array(src.length), out = new Uint8Array(src.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = 0;
    for (let d = -r; d <= r && !m; d++) { const xx = x + d; if (xx >= 0 && xx < w && src[y * w + xx]) m = 1; }
    tmp[y * w + x] = m;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = 0;
    for (let d = -r; d <= r && !m; d++) { const yy = y + d; if (yy >= 0 && yy < h && tmp[yy * w + x]) m = 1; }
    out[y * w + x] = m;
  }
  return out;
}

// Same recipe as simulation/hand_sim/tactile_panel.py: ink = line art, dilate it to seal gaps, flood-fill the
// outside from the corners, everything else is hand, then dilate the hand back. A plain fill would leak into the
// palm wherever the drawn outline has a break.
function buildHand(img) {
  const cv = document.createElement("canvas");
  cv.width = GW; cv.height = GH;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, GW, GH);
  const s = Math.min(GW / img.naturalWidth, GH / img.naturalHeight); // "contain", like the old SVG <image>
  const dw = img.naturalWidth * s, dh = img.naturalHeight * s;
  ctx.drawImage(img, (GW - dw) / 2, (GH - dh) / 2, dw, dh);
  const px = ctx.getImageData(0, 0, GW, GH).data;
  const ink = new Uint8Array(GW * GH);
  for (let i = 0; i < ink.length; i++) ink[i] = (px[i * 4] * 0.299 + px[i * 4 + 1] * 0.587 + px[i * 4 + 2] * 0.114) < INK_THRESHOLD ? 1 : 0;
  const sealed = dilate(ink, GW, GH, SEAL_RADIUS);

  const outside = new Uint8Array(GW * GH), stack = [];
  const push = (x, y) => { const i = y * GW + x; if (!sealed[i] && !outside[i]) { outside[i] = 1; stack.push(i); } };
  push(0, 0); push(GW - 1, 0); push(0, GH - 1); push(GW - 1, GH - 1);
  while (stack.length) {
    const i = stack.pop(), x = i % GW, y = (i - x) / GW;
    if (x > 0) push(x - 1, y);
    if (x < GW - 1) push(x + 1, y);
    if (y > 0) push(x, y - 1);
    if (y < GH - 1) push(x, y + 1);
  }
  const inside = new Uint8Array(GW * GH);
  for (let i = 0; i < inside.length; i++) inside[i] = outside[i] ? 0 : 1;
  const mask = Float32Array.from(dilate(inside, GW, GH, SEAL_RADIUS));

  const edge = boxBlur(boxBlur(mask, GW, GH, 2), GW, GH, 2);    // soft, anti-aliased silhouette
  const core = boxBlur(boxBlur(mask, GW, GH, 14), GW, GH, 14);  // how deep inside the hand a pixel is
  const alpha = new Float32Array(mask.length), base = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) {
    alpha[i] = edge[i];
    base[i] = BASE_EDGE + (BASE_CORE - BASE_EDGE) * smooth(0.45, 0.95, core[i]);
  }
  return { alpha, base, core };
}

// The layout in GROUPS is a hand-tuned approximation, so a few patches (fingertips) sit just off the drawn
// hand. Heat is clipped to the silhouette, so move any patch whose centre is not well inside the hand to the
// nearest point that is; otherwise its reading would be clipped away.
function snapGroups(hand) {
  const R = 70; // search radius in cells
  for (const g of GROUPS) {
    const gx = Math.round(g.cx / CELL), gy = Math.round(g.cy / CELL);
    if (hand.core[gy * GW + gx] >= SNAP_DEPTH) continue;
    // Nearest point that is deep enough; if the finger is too thin for that, the deepest point nearby.
    let best = -1, bestD = Infinity, deep = -1, deepScore = -Infinity;
    for (let y = Math.max(0, gy - R); y <= Math.min(GH - 1, gy + R); y++) {
      for (let x = Math.max(0, gx - R); x <= Math.min(GW - 1, gx + R); x++) {
        const c = hand.core[y * GW + x], d = (x - gx) ** 2 + (y - gy) ** 2;
        if (c >= SNAP_DEPTH && d < bestD) { bestD = d; best = y * GW + x; }
        const score = c - 0.004 * Math.sqrt(d);
        if (score > deepScore) { deepScore = score; deep = y * GW + x; }
      }
    }
    if (best < 0) best = deep;
    if (best >= 0) { g.cx = ((best % GW) + 0.5) * CELL; g.cy = (Math.floor(best / GW) + 0.5) * CELL; }
  }
}

let handPromise;
const loadHand = () => (handPromise ??= new Promise((resolve) => {
  const img = new Image();
  img.onload = () => { const h = buildHand(img); snapGroups(h); resolve(h); };
  img.onerror = () => resolve(null);
  img.src = "/hand.png";
}));

// ---- Painting ----------------------------------------------------------------------------------------
function paint(canvas, levels, hand) {
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
        // Flat only in the inner 45% of the ellipse, then a smooth gaussian fall-off: a peak, not a plateau with a rim.
        const e = Math.hypot((x - g.cx) / g.rx, (y - g.cy) / g.ry);
        const d = e <= PLATEAU ? 0 : ((e - PLATEAU) * Math.min(g.rx, g.ry)) / (0.9 * g.glow);
        const v = level * Math.exp(-d * d);
        if (v > field[iy * GW + ix]) field[iy * GW + ix] = v; // max, not sum: a patch's peak stays its own reading
      }
    }
  }
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(GW, GH);
  const out = img.data;
  for (let i = 0; i < field.length; i++) {
    const c = field[i], a = hand ? hand.alpha[i] : 0;
    const heat = smooth(0.02, 0.12, c);
    const alpha = hand ? a : heat * 0.95;          // heat stays inside the hand outline (no silhouette: fall back to contact only)
    if (alpha < 0.004) continue;
    const t0 = hand ? hand.base[i] : 0;
    const t = Math.min(T_MAX, t0 + (1 - t0) * c);
    const k = Math.round(t * 255) * 3;
    out[i * 4] = LUT[k]; out[i * 4 + 1] = LUT[k + 1]; out[i * 4 + 2] = LUT[k + 2]; out[i * 4 + 3] = Math.round(alpha * 255);
  }
  ctx.putImageData(img, 0, 0);
}

export default function TactileHeatmap({ tactile }) {
  const canvas = useRef(null);
  const levels = useRef(Object.fromEntries(GROUPS.map((g) => [g.name, 0])));
  const gain = useRef(GAIN_MIN);
  const handRef = useRef(null);
  const [handReady, setHandReady] = useState(false);
  const raf = useRef(0);

  useEffect(() => {
    let alive = true;
    loadHand().then((h) => { if (alive) { handRef.current = h; setHandReady(true); } });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const means = Object.fromEntries(GROUPS.map((g) => [g.name, meanOf(g, tactile)]));
    const gainTarget = Math.min(GAIN_MAX, Math.max(GAIN_MIN, ...Object.values(means)));
    cancelAnimationFrame(raf.current);
    const step = () => {
      let moving = false;
      const gGap = gainTarget - gain.current;
      if (reduce || Math.abs(gGap) < 1) gain.current = gainTarget;
      else { gain.current += gGap * EASE * 0.5; moving = true; }
      for (const g of GROUPS) {
        const target = toLevel(means[g.name], gain.current), cur = levels.current[g.name], gap = target - cur;
        if (reduce || Math.abs(gap) < 0.004) levels.current[g.name] = target;
        else { levels.current[g.name] = cur + gap * EASE; moving = true; }
      }
      if (canvas.current) paint(canvas.current, levels.current, handRef.current);
      if (moving) raf.current = requestAnimationFrame(step);
    };
    step();
    return () => cancelAnimationFrame(raf.current);
  }, [tactile, handReady]);

  return (
    <div className="heatmap">
      <div className="heat-stage">
        <canvas ref={canvas} width={GW} height={GH} className="heat-canvas" role="img" aria-label="Tactile heat map of the hand" />
        {/* the drawing, inverted to light lines and screened over the glow, adds the fine anatomical detail */}
        <img className="heat-hand" src="/hand.png" alt="" />
        {!tactile && <span className="heatmap-empty">Waiting for tactile data</span>}
      </div>
      <div className="heat-legend" aria-hidden="true">
        <span>Light</span>
        <span className="heat-ramp" style={{ background: THERMAL_CSS }} />
        <span>Firm</span>
      </div>
    </div>
  );
}
