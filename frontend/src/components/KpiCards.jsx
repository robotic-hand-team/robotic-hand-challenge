import { useEffect, useRef, useState } from "react";
import { SENSOR_INFO } from "../api.js";

const fmt = (x, d = 1) => (x == null ? "–" : Number(x).toFixed(d));
const TENTATIVE_BELOW = 0.7; // mean probability of the leading label; 0.5 is a coin flip

// The challenge's main answer: which sensor the hand holds, how sure the classifier is, and (in replay)
// estimate = {label, confidence, agreement, n_votes} | null.
function HeldObjectCard({ estimate, noTactile }) {
  const info = estimate ? SENSOR_INFO[estimate.label] : null;
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => { if (!info) setOpen(false); }, [info]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    const onClick = (e) => !ref.current?.contains(e.target) && setOpen(false);
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onClick);
    return () => { document.removeEventListener("keydown", onKey); document.removeEventListener("pointerdown", onClick); };
  }, [open]);

  if (!info) {
    return (
      <section className="card kpi held held-unknown" aria-label="Held object" aria-live="polite">
        <span className="kpi-label">Holding</span>
        <div className="held-row">
          <div className="held-img"><span className="held-empty" aria-hidden="true">?</span></div>
          <div className="held-text">
            <span className="held-name">Unidentified</span>
            <span className="held-blurb">
              {noTactile
                ? "No tactile data in this stream, so the classifier cannot vote."
                : "Waiting for contact."}
            </span>
          </div>
        </div>
        <div className="conf">
          <div className="conf-head"><span>Confidence</span><span className="conf-pct">–</span></div>
          <div className="conf-bar conf-bar-idle" aria-hidden="true" />
        </div>
      </section>
    );
  }

  const pct = Math.round(estimate.confidence * 100);
  const tentative = estimate.confidence < TENTATIVE_BELOW;
  return (
    <section ref={ref} className="card kpi held kpi-accent" aria-label="Held object" aria-live="polite">
      <div className="held-top">
        <span className="kpi-label">Holding</span>
        <span className={`badge ${tentative ? "" : "badge-ok"}`}>{tentative ? "Tentative" : info.name}</span>
      </div>
      <div className="held-row">
        <div className="held-img"><img src={info.image} alt="" /></div>
        <div className="held-text">
          <span className="held-name">{info.model}</span>
          <span className="held-blurb">{info.blurb}</span>
          <button type="button" className="read-more" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            {open ? "Show less" : "Read more"}
          </button>
        </div>
      </div>
      <div className="conf">
        <div className="conf-head">
          <span>Confidence</span>
          <span className="conf-pct">{pct}%</span>
        </div>
        <div className="conf-bar" role="meter" aria-label="Classifier confidence" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
          <span className={tentative ? "conf-fill-weak" : ""} style={{ width: `${pct}%` }} />
        </div>
      </div>
      {open && <div className="held-more" role="note">{info.more}</div>}
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
          ? "Tactile estimate (physics model, about ±25 mm) · palm +x, fingers +z"
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
      <HeldObjectCard estimate={kpi?.sensor} noTactile={noTactile} />
      <PositionCard kpi={kpi} live={live} />
      <OrientationCard kpi={kpi} />
      <GripCard kpi={kpi} contacts={frame?.contacts} phase={frame?.phase} />
    </div>
  );
}
