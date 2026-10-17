import { useEffect, useMemo, useState } from "react";
import { fetchSession, fetchSessions } from "./api.js";
import KpiCards from "./components/KpiCards.jsx";
import TwinView from "./components/TwinView.jsx";

const CHART_SLOTS = ["Tactile pressure over time", "Finger forces", "Classifier confidence", "Slip velocity"];

function usePlayback(session) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const n = session?.frames.length ?? 0;

  useEffect(() => setIndex(0), [session]);
  useEffect(() => {
    if (!playing || n === 0) return;
    const id = setInterval(() => setIndex((i) => (i + 1) % n), (session.dt ?? 0.05) * 1000);
    return () => clearInterval(id);
  }, [playing, n, session]);

  return { index: Math.min(index, Math.max(n - 1, 0)), setIndex, playing, setPlaying, n };
}

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [filter, setFilter] = useState("all");
  const [sessionId, setSessionId] = useState(null);
  const [session, setSession] = useState(null);
  const [error, setError] = useState(null);
  const play = usePlayback(session);

  useEffect(() => {
    fetchSessions()
      .then((s) => { setSessions(s); setSessionId(s[0]?.id ?? null); })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let stale = false;
    fetchSession(sessionId)
      .then((s) => !stale && (setSession(s), setError(null)))
      .catch((e) => !stale && setError(e.message));
    return () => { stale = true; };
  }, [sessionId]);

  const shown = useMemo(
    () => sessions.filter((s) => filter === "all" || s.label === filter),
    [sessions, filter]
  );
  const frame = session?.frames[play.index];
  const random = () => shown.length && setSessionId(shown[Math.floor(Math.random() * shown.length)].id);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-title">Tactile Digital Twin</div>
            <div className="brand-sub">VEGA challenge · Inspire RH56DFTP · pose &amp; sensor estimation</div>
          </div>
        </div>
        <div className="header-spacer" />
        <div className="header-controls">
          <div className="seg" role="group" aria-label="Filter sessions by sensor">
            {[["all", "All"], ["A", "Sensor A"], ["B", "Sensor B"]].map(([k, label]) => (
              <button key={k} aria-pressed={filter === k} onClick={() => setFilter(k)}>{label}</button>
            ))}
          </div>
          <select
            className="select"
            aria-label="Session"
            value={sessionId ?? ""}
            onChange={(e) => setSessionId(e.target.value)}
          >
            {sessionId && !shown.some((s) => s.id === sessionId) && <option value={sessionId}>{sessionId}</option>}
            {shown.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} · {s.label ? `Sensor ${s.label}` : s.object}{s.success === false ? " · failed grasp" : ""}
              </option>
            ))}
          </select>
          <button className="btn btn-primary" onClick={random}>Random session</button>
        </div>
      </header>

      <main className="main">
        <KpiCards frame={frame} session={session} />

        <section className="card twin" aria-label="Digital twin">
          <TwinView frame={frame} session={session} estimate={frame?.kpi.sensor} />
          <div className="timeline">
            <button
              className="btn"
              aria-label={play.playing ? "Pause" : "Play"}
              onClick={() => play.setPlaying((p) => !p)}
            >
              {play.playing ? "❚❚" : "▶"}
            </button>
            <input
              className="scrubber"
              type="range"
              min={0}
              max={Math.max(play.n - 1, 0)}
              value={play.index}
              aria-label="Frame"
              onChange={(e) => { play.setPlaying(false); play.setIndex(Number(e.target.value)); }}
            />
            <span className={`badge phase ${frame?.phase === "hold" ? "badge-ok" : ""}`}>{frame?.phase ?? "–"}</span>
            <span className="time">
              {frame ? `${frame.t.toFixed(2)} s · ${play.index + 1}/${play.n}` : "–"}
            </span>
          </div>
          {error && <div className="timeline error" role="alert">Backend: {error}</div>}
        </section>

        <div className="charts">
          {CHART_SLOTS.map((title) => (
            <section key={title} className="card placeholder" aria-label={title}>
              <span className="placeholder-title">{title}</span>
              <div className="placeholder-body">Visualization placeholder</div>
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}
