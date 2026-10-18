import { useState } from "react";
import { SENSOR_INFO } from "../api.js";

const fmt = (x, d = 1) => (x == null ? "–" : Number(x).toFixed(d));

function HeldObjectCard({ estimate }) {
  const info = estimate ? SENSOR_INFO[estimate.label] : null;
  const [open, setOpen] = useState(false);
  return (
    <section className="card kpi held" aria-label="Held object">
      <div className="held-img">
        {info ? <img src={info.image} alt={info.model} /> : <span className="held-empty" aria-hidden="true" />}
      </div>
      <div className="held-text">
        <span className="held-name">{info ? info.model : "No object"}</span>
        {info && (
          <span className="held-blurb">
            {info.blurb}{" "}
            <button type="button" className="read-more" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
              {open ? "Close" : "Read more"} <span aria-hidden="true">→</span>
            </button>
          </span>
        )}
      </div>
      {info && open && <div className="held-more" role="note">{info.more}</div>}
    </section>
  );
}

function PositionCard({ kpi, live }) {
  const p = kpi?.pos_mm ?? [null, null, null];
  return (
    <section className="card kpi" aria-label="Object position">
      <span className="kpi-label">Object position · hand frame</span>
      <div className="xyz">
        {["x", "y", "z"].map((ax, i) => (
          <div key={ax}><span className="ax">{ax} (mm)</span><span className="v">{fmt(p[i])}</span></div>
        ))}
      </div>
      <span className="kpi-sub">
        {live
          ? "Approximate: default grasp position, shifted ≤ 20 mm by the tactile model · palm +x, fingers +z"
          : "Relative to the hand root · palm +x, fingers +z"}
      </span>
    </section>
  );
}

function OrientationCard({ kpi }) {
  const a = kpi?.axis;
  const nominal = kpi?.orientation === "nominal";
  return (
    <section className="card kpi" aria-label="Object orientation">
      <span className="kpi-label">Orientation</span>
      {nominal ? (
        <span className="kpi-value">Nominal</span>
      ) : (
        <span className="kpi-value">{fmt(kpi?.tilt_deg)}<span className="unit">° tilt</span></span>
      )}
      <span className="kpi-sub">
        Long axis {a ? `[${a.map((v) => fmt(v, 2)).join(", ")}]` : "–"} ·{" "}
        {nominal ? "default grasp, no live orientation estimate yet" : "vs. nominal grasp"}
      </span>
    </section>
  );
}

function GripCard({ kpi, contacts, phase }) {
  const slipping = kpi?.slipping;
  const holding = phase === "hold" || phase === "HOLD";
  const live = kpi && kpi.speed_mm_s == null;
  return (
    <section className={`card kpi ${slipping ? "kpi-danger" : ""}`} aria-label="Grip and slip">
      <span className="kpi-label">Grip &amp; slip</span>
      <span className={`badge ${slipping ? "badge-danger" : holding ? "badge-ok" : ""}`}>
        <span className="dot" />{slipping ? "Slip detected" : holding ? "Stable hold" : phase ?? "–"}
      </span>
      <span className="kpi-value">{kpi?.grip_force_g ?? "–"}<span className="unit">g grip</span></span>
      <span className="kpi-sub">
        {live ? kpi.slip_text || "Slip detector idle" : `${fmt(kpi?.speed_mm_s)} mm/s object speed`}
        {" · "}{contacts ?? "–"} taxels in contact
      </span>
    </section>
  );
}

export default function KpiCards({ frame, session, live, noTactile }) {
  const kpi = frame?.kpi;
  return (
    <div className="kpis">
      <HeldObjectCard estimate={kpi?.sensor} />
      <PositionCard kpi={kpi} live={live} />
      <OrientationCard kpi={kpi} />
      <GripCard kpi={kpi} contacts={frame?.contacts} phase={frame?.phase} />
    </div>
  );
}
