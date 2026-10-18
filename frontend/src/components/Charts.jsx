// Three time-series charts under the twin: finger forces, classifier confidence, slip velocity.
// `frames` is the whole replay session, or a rolling window of live frames; `cursor` marks the frame on screen.
const W = 300, H = 150, PAD = { l: 34, r: 8, t: 8, b: 18 };
const FINGERS = [
  ["Little", "#282828"], ["Ring", "#1f6fd1"], ["Middle", "#0f9b8e"],
  ["Index", "#b8860b"], ["Thumb bend", "#c62828"], ["Thumb rot", "#7b4fc4"],
];
const nice = (max) => {
  if (max <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(max)), m = max / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
};

function Chart({ series, n, cursor, yMax, yTicks, unit = "", threshold, empty }) {
  if (!n || empty) return <div className="chart-empty">{empty || "Waiting for data"}</div>;
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
  const x = (i) => PAD.l + (n > 1 ? (i / (n - 1)) * iw : 0);
  const y = (v) => PAD.t + ih - (Math.min(Math.max(v, 0), yMax) / yMax) * ih;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img">
      {yTicks.map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--border)" />
          <text x={PAD.l - 4} y={y(t) + 3} textAnchor="end" className="chart-tick">{t}{unit}</text>
        </g>
      ))}
      {threshold != null && (
        <line x1={PAD.l} x2={W - PAD.r} y1={y(threshold)} y2={y(threshold)} stroke="var(--danger)" strokeDasharray="4 3" />
      )}
      {series.map((s) => (
        <polyline key={s.name} fill="none" stroke={s.color} strokeWidth="1.6" strokeLinejoin="round"
                  points={s.values.map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`)).filter(Boolean).join(" ")} />
      ))}
      {cursor != null && cursor < n && (
        <line x1={x(cursor)} x2={x(cursor)} y1={PAD.t} y2={PAD.t + ih} stroke="var(--yellow)" strokeWidth="2" />
      )}
    </svg>
  );
}

function ChartCard({ title, sub, legend, children }) {
  return (
    <section className="card chart-card" aria-label={title}>
      <div className="chart-head">
        <span className="placeholder-title">{title}</span>
        {sub && <span className="chart-sub">{sub}</span>}
      </div>
      {children}
      {legend && (
        <ul className="chart-legend">
          {legend.map(([name, color]) => <li key={name}><span style={{ background: color }} />{name}</li>)}
        </ul>
      )}
    </section>
  );
}

export default function Charts({ frames, cursor, live }) {
  const n = frames.length;
  const cur = live ? n - 1 : cursor;

  const forceSeries = FINGERS.map(([name, color], k) => ({ name, color, values: frames.map((f) => f.force_g?.[k] ?? null) }));
  const forceMax = nice(Math.max(0, ...forceSeries.flatMap((s) => s.values.filter((v) => v != null))));

  // p_sensor_b is the probability of Sensor B; show the probability of whichever label currently leads.
  const conf = frames.map((f) => {
    const p = f.kpi?.p_sensor_b;
    return p == null ? null : Math.max(p, 1 - p);
  });

  const speed = frames.map((f) => f.kpi?.speed_mm_s ?? null);
  const hasSpeed = speed.some((v) => v != null);
  const speedMax = nice(Math.max(25, ...speed.filter((v) => v != null)));

  return (
    <>
      <ChartCard title="Finger forces" sub="grams" legend={FINGERS}>
        <Chart n={n} cursor={cur} yMax={forceMax} yTicks={[0, forceMax / 2, forceMax]} series={forceSeries}
               empty={!forceSeries.some((s) => s.values.some((v) => v != null)) && "No force data"} />
      </ChartCard>
      <ChartCard title="Classifier confidence" sub="leading label">
        <Chart n={n} cursor={cur} yMax={1} yTicks={[0, 0.5, 1]} threshold={0.5}
               series={[{ name: "confidence", color: "#b8860b", values: conf }]}
               empty={!conf.some((v) => v != null) && "Waiting for contact frames"} />
      </ChartCard>
      <ChartCard title="Slip velocity" sub="mm/s · red line = slip limit">
        <Chart n={n} cursor={cur} yMax={speedMax} yTicks={[0, speedMax / 2, speedMax]} threshold={20}
               series={[{ name: "speed", color: "#c62828", values: speed }]}
               empty={!hasSpeed && "No speed estimate (needs sim ground truth)"} />
      </ChartCard>
    </>
  );
}
