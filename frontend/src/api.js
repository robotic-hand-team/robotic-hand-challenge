// Backend API (proxied to FastAPI on :8000 by vite.config.js).
async function get(path) {
  const res = await fetch(`/api${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for /api${path}`);
  return res.json();
}

export const fetchSessions = () => get("/sessions");
export const fetchSession = (id) => get(`/sessions/${encodeURIComponent(id)}`);
export const TWIN_GLB_URL = "/api/twin.glb";
export const liveSocketUrl = () =>
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/api/live`;

// Sensor labels -> simulation object ids (vegabar_29 = Sensor A, vegabar_38 = Sensor B).
export const LABEL_TO_OBJECT = { A: "vegabar_29", B: "vegabar_38" };
export const SENSOR_INFO = {
  A: { name: "Sensor A", model: "VEGABAR 29" },
  B: { name: "Sensor B", model: "VEGABAR 38" },
};
