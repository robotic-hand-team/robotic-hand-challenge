import { useEffect, useMemo, useRef, useState } from "react";
import { fetchSession, fetchSessions } from "./api.js";
import Charts from "./components/Charts.jsx";
import KpiCards from "./components/KpiCards.jsx";
import TactileHeatmap from "./components/TactileHeatmap.jsx";
import TwinView from "./components/TwinView.jsx";
import useLive from "./useLive.js";

const LIVE_WINDOW = 300;  // frames kept for the live charts

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

function ReplayBar({ play, frame }) {
  return (
    <div className="timeline">
      <button className="btn" aria-label={play.playing ? "Pause" : "Play"} onClick={() => play.setPlaying((p) => !p)}>
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
      <span className="time">{frame ? `${frame.t.toFixed(2)} s · ${play.index + 1}/${play.n}` : "–"}</span>
    </div>
  );
}

function LiveBar({ live }) {
  const { status, connected, socketOpen, frame } = live;
  return (
    <div className="timeline live-bar">
      <span className={`badge ${connected ? "badge-ok" : "badge-danger"}`}>
        <span className={`dot ${connected ? "pulse" : ""}`} />
        {connected ? "Live" : socketOpen ? "No hand data" : "Backend offline"}
      </span>
      <span className="live-meta">
        {connected
          ? `${status.rate_hz.toFixed(0)} Hz · ${status.source} · ${status.hand_side} hand` +
            (status.has_tactile ? "" : " · no tactile data in broadcast")
          : `Start hardware/config.py — listening on udp://${status?.udp ?? "0.0.0.0:5005"}`}
      </span>
      <span className="header-spacer" />
      {connected && frame?.frozen && <span className="badge badge-danger"><span className="dot" />Motor data frozen</span>}
      <span className={`badge phase ${frame?.phase === "HOLD" ? "badge-ok" : ""}`}>{frame?.phase ?? "–"}</span>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("replay");
  const [sessions, setSessions] = useState([]);
  const [filter, setFilter] = useState("all");
  const [sessionId, setSessionId] = useState(null);
  const [session, setSession] = useState(null);
  const [error, setError] = useState(null);
  const play = usePlayback(session);
  const live = useLive(mode === "live");

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
  const random = () => shown.length && setSessionId(shown[Math.floor(Math.random() * shown.length)].id);

  const isLive = mode === "live";
  const [liveHistory, setLiveHistory] = useState([]);
  const lastLive = useRef(null);
  useEffect(() => {
    if (!isLive || !live.frame || live.frame === lastLive.current) return;
    lastLive.current = live.frame;
    setLiveHistory((h) => [...h.slice(-(LIVE_WINDOW - 1)), live.frame]);
  }, [isLive, live.frame]);
  const chartFrames = isLive ? liveHistory : session?.frames ?? [];
  const frame = isLive ? live.frame : session?.frames[play.index];
  const twinSession = isLive
    ? { object: live.frame?.object ?? null, mirrored: (live.status?.hand_side ?? "left") === "left" }
    : session && { object: session.object, mirrored: false };
  const caption = isLive
    ? live.connected && `${live.status.hand_side} hand · live`
    : session && <>{session.hand_side} hand · sim session <strong>{session.id}</strong></>;
  const message = isLive && !live.frame
    ? live.socketOpen ? "Waiting for the hand — start hardware/config.py" : "Connecting to the backend…"
    : null;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <img className="brand-logo" src="/logo_vega_bfl.png" alt="VEGA × Black Forest Labs" />
          <div className="brand-title">Tactile Digital Twin</div>
        </div>
        <div className="header-spacer" />
        <div className="header-controls">
          <div className="seg" role="group" aria-label="Data source">
            {[["replay", "Replay"], ["live", "Live hand"]].map(([k, label]) => (
              <button key={k} aria-pressed={mode === k} onClick={() => setMode(k)}>{label}</button>
            ))}
          </div>
          {!isLive && (
            <>
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
            </>
          )}
        </div>
      </header>

      <main className="main">
        <KpiCards frame={frame} session={isLive ? null : session} live={isLive} noTactile={isLive && frame && !frame.has_tactile} />

        <section className="card twin" aria-label="Digital twin">
          <TwinView
            frame={frame}
            session={twinSession}
            estimate={frame?.kpi.sensor}
            caption={caption}
            message={message}
          />
          {isLive ? <LiveBar live={live} /> : <ReplayBar play={play} frame={frame} />}
          {error && !isLive && <div className="timeline error" role="alert">Backend: {error}</div>}
        </section>

        <div className="charts">
          <section className="card heat-card" aria-label="Tactile heat map">
            <span className="placeholder-title">Tactile heat map</span>
            <TactileHeatmap tactile={frame?.tactile} />
          </section>
          <Charts frames={chartFrames} cursor={play.index} live={isLive} />
        </div>
      </main>
    </div>
  );
}
