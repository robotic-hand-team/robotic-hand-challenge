// Per-group tactile heat map on the hand drawing. Layout (anchors, offsets, taxel pitch) is copied from
// hardware/config.py; frames only carry one summed value per sensor group, so each group is one glowing patch.
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
];

const FULL_SCALE = 1500; // mean counts per taxel that read as "hot" (VEGABAR 29 ~1500, VEGABAR 38 saturates at ~4094)
const STOPS = [[0, [28, 23, 112]], [0.3, [23, 107, 235]], [0.5, [20, 199, 194]], [0.68, [214, 242, 36]], [0.82, [255, 204, 0]], [0.93, [242, 61, 13]], [1, [255, 199, 82]]];

function ramp(t) {
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
      const f = (t - t0) / (t1 - t0);
      return `rgb(${c0.map((v, k) => Math.round(v + (c1[k] - v) * f)).join(",")})`;
    }
  }
  return `rgb(${STOPS.at(-1)[1].join(",")})`;
}

export default function TactileHeatmap({ tactile }) {
  return (
    <div className="heatmap">
      <svg viewBox="0 0 700 900" role="img" aria-label="Tactile heat map of the hand">
        <defs><filter id="hm-blur" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="7" /></filter></defs>
        <image href="/hand.png" x="0" y="0" width="700" height="900" opacity="0.45" />
        <g filter="url(#hm-blur)">
          {GROUPS.map(([name, rows, cols, key, dx, dy]) => {
            const sum = tactile?.[name];
            if (!sum) return null;
            const level = Math.min(1, Math.sqrt(sum / (rows * cols) / FULL_SCALE));
            if (level < 0.04) return null;
            return (
              <rect key={name} x={ANCHORS[key][0] + dx} y={ANCHORS[key][1] + dy} width={cols * PITCH} height={rows * PITCH}
                    rx="6" fill={ramp(level)} />
            );
          })}
        </g>
      </svg>
      {!tactile && <span className="heatmap-empty">Waiting for tactile data</span>}
    </div>
  );
}
