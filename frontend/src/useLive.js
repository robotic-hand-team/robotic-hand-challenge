import { useEffect, useState } from "react";
import { liveSocketUrl } from "./api.js";

// Live hand frames from the backend WebSocket (/api/live). Reconnects while enabled.
export default function useLive(enabled) {
  const [frame, setFrame] = useState(null);
  const [status, setStatus] = useState(null);
  const [socketOpen, setSocketOpen] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let ws, retry, raf = 0, pending = null, closed = false;
    // Render only the newest message, at most once per display frame. Setting state for every
    // message made them queue up whenever rendering was slower than the stream, so the twin
    // fell further and further behind the real hand; now older messages are simply dropped.
    const flush = () => {
      raf = 0;
      const msg = pending;
      pending = null;
      if (!msg) return;
      setStatus(msg.status);
      if (msg.frame) setFrame(msg.frame);
    };
    const connect = () => {
      ws = new WebSocket(liveSocketUrl());
      ws.onopen = () => setSocketOpen(true);
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        // a status-only message must not discard a frame that is still waiting to be shown
        pending = { status: msg.status, frame: msg.type === "frame" ? msg.frame : pending?.frame ?? null };
        if (!raf) raf = requestAnimationFrame(flush);
      };
      ws.onclose = () => {
        setSocketOpen(false);
        if (!closed) retry = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => { closed = true; clearTimeout(retry); cancelAnimationFrame(raf); ws?.close(); };
  }, [enabled]);

  // The last frame stays on screen when the stream stops; `connected` says whether it is current.
  return { frame, status, socketOpen, connected: Boolean(socketOpen && status?.connected) };
}
