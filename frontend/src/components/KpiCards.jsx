import { SENSOR_INFO } from "../api.js";

const pct = (x) => `${Math.round(x * 100)}%`;
const fmt = (x, d = 1) => (x == null ? "–" : Number(x).toFixed(d));

function SensorCard({ estimate, truth }) {
  const info = estimate ? SENSOR_INFO[estimate.label] : null;
  const correct = estimate && truth ? estimate.label === truth : null;
  return (
    <section className="card kpi kpi-accent" aria-label="Estimated sensor">
      <span className="kpi-label">Estimated sensor</span>
      <span className="kpi-value">{info ? info.name : "–"}</span>
      <span className="kpi-sub">{info ? `${info.model} · confidence ${pct(estimate.confidence)}` : "Waiting for contact frames"}</span>
      <div className="meter" aria-hidden="true"><span style={{ width: info ? pct(estimate.confidence) : 0 }} /></div>
      {correct != null && (
        <span className={`badge ${correct ? "badge-ok" : "badge-danger"}`}>
          <span className="dot" />{correct ? "Matches sim ground truth" : `Ground truth: Sensor ${truth}`}
        </span>
      )}
    </section>
  );
}

function PositionCard({ kpi }) {
  const p = kpi?.pos_mm ?? [null, null, null];
  return (
    <section className="card kpi" aria-label="Object position">
      <span className="kpi-label">Object position · hand frame</span>
      <div className="xyz">
        {["x", "y", "z"].map((ax, i) => (
          <div key={ax}><span className="ax">{ax} (mm)</span><span className="v">{fmt(p[i])}</span></div>
        ))}
      </div>
      <span className="kpi-sub">Relative to the hand root · palm +x, fingers +z</span>
    </section>
  );
}

function OrientationCard({ kpi }) {
  const a = kpi?.axis;
  return (
    <section className="card kpi" aria-label="Object orientation">
      <span className="kpi-label">Orientation</span>
      <span className="kpi-value">{fmt(kpi?.tilt_deg)}<span className="unit">° tilt</span></span>
      <span className="kpi-sub">
        Long axis {a ? `[${a.map((v) => fmt(v, 2)).join(", ")}]` : "–"} · vs. nominal grasp
      </span>
    </section>
  );
}

function GripCard({ kpi, contacts, phase }) {
  const slipping = kpi?.slipping;
  return (
    <section className={`card kpi ${slipping ? "kpi-danger" : ""}`} aria-label="Grip and slip">
      <span className="kpi-label">Grip &amp; slip</span>
      <span className={`badge ${slipping ? "badge-danger" : phase === "hold" ? "badge-ok" : ""}`}>
        <span className="dot" />{slipping ? "Slip detected" : phase === "hold" ? "Stable hold" : phase ?? "–"}
      </span>
      <span className="kpi-value">{kpi ? kpi.grip_force_g : "–"}<span className="unit">g grip</span></span>
      <span className="kpi-sub">{fmt(kpi?.speed_mm_s)} mm/s object speed · {contacts ?? "–"} taxels in contact</span>
    </section>
  );
}

export default function KpiCards({ frame, session }) {
  const kpi = frame?.kpi;
  return (
    <div className="kpis">
      <SensorCard estimate={kpi?.sensor} truth={session?.truth} />
      <PositionCard kpi={kpi} />
      <OrientationCard kpi={kpi} />
      <GripCard kpi={kpi} contacts={frame?.contacts} phase={frame?.phase} />
    </div>
  );
}
