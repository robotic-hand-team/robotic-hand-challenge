import { useEffect, useState } from "react";
import { liveSocketUrl } from "./api.js";

// Live hand frames from the backend WebSocket (/api/live). Reconnects while enabled.
export default function useLive(enabled) {
  const [frame, setFrame] = useState(null);
  const [status, setStatus] = useState(null);
  const [socketOpen, setSocketOpen] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let ws, retry, closed = false;
    const connect = () => {
      ws = new WebSocket(liveSocketUrl());
      ws.onopen = () => setSocketOpen(true);
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        setStatus(msg.status);
        if (msg.type === "frame") setFrame(msg.frame);
      };
      ws.onclose = () => {
        setSocketOpen(false);
        if (!closed) retry = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => { closed = true; clearTimeout(retry); ws?.close(); };
  }, [enabled]);

  // The last frame stays on screen when the stream stops; `connected` says whether it is current.
  return { frame, status, socketOpen, connected: Boolean(socketOpen && status?.connected) };
}
