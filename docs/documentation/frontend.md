# `frontend/`: web dashboard ("Tactile Digital Twin")

Backend: [backend.md](backend.md)

This is a React + Three.js dashboard. It shows the hand and the held object in 3D, together with
the tactile readings and the model estimates. It gets all its data from the [backend](../../backend/)
through `/api`.

## Using the dashboard

Open **http://localhost:3000** once it's running. In the header:

- **Replay / Live hand** switches the data source.
  - **Replay** plays a recorded session. Filter by **All / Sensor A / Sensor B**, pick one from the drop-down list (failed simulated grasps are marked), or press **Random session**. The bar under the twin plays, pauses and scrubs.
  - **Live hand** follows the real hand, or `hardware/replay_to_twin.py`, through the backend's WebSocket. The bar shows the connection status.

The page shows:

| Area | Component | Contents |
|---|---|---|
| 3D twin | `components/TwinView.jsx` | Hand and object in 3D, using the meshes from `/api/twin.glb` and the body poses per frame |
| KPI cards | `components/KpiCards.jsx` | **Holding** (Sensor A/B with product image and confidence bar, marked tentative below 70%), **Object position** (hand frame), **Orientation**, **Grip & slip** |
| Tactile heat map | `components/TactileHeatmap.jsx` | Thermal-camera-style heat map clipped to the hand outline (`public/hand.png`), with a layout copied from `hardware/config.py` |
| Charts | `components/Charts.jsx` | Finger forces, classifier confidence and slip velocity over time, with a cursor at the current frame |

## Files

| File | What it is |
|---|---|
| `src/main.jsx` | Entry point |
| `src/App.jsx` | Layout, replay/live switch, session picker, playback |
| `src/api.js` | REST helpers (`/api/sessions`, `/api/twin.glb`), the live WebSocket URL, sensor info |
| `src/useLive.js` | React hook for the live WebSocket `/api/live`; reconnects automatically |
| `src/theme.css` | Styles |
| `src/websocket.js` | Empty placeholder |
| `public/` | `hand.png` (also used by the simulation viewer), VEGABAR product images, VEGA logos. `inspire_hand.gltf` is an empty placeholder |
| `index.html`, `vite.config.js` | Vite setup. The dev server runs on port 3000 and forwards `/api` (including WebSockets) to `http://127.0.0.1:8000` |
| `package.json`, `package-lock.json` | React 18.3, three 0.169, @react-three/fiber + drei, Vite 5 |
| `Dockerfile`, `nginx.conf.template` | Production build with Node 22, served by nginx on port 3000. nginx forwards `/api` to `BACKEND_ADDR` (default `backend:8000`) |

## Running

**With Docker** (from the repo root; starts the backend as well):

```bash
docker compose up --build
```

**Without Docker** (needs Node.js 22; start the backend first):

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Other commands: `npm run build` creates a production build in `dist/`, and `npm run preview` serves that build.

To run only the frontend container against a backend on your PC:

```bash
docker build -t robotics-frontend ./frontend
docker run -p 3000:3000 -e BACKEND_ADDR=host.docker.internal:8000 robotics-frontend
```

## Notes

- The dev server targets `127.0.0.1` rather than `localhost` on purpose. Node resolves `localhost` to IPv6 `::1` first, while uvicorn listens on IPv4 only.
- On Windows, `fs.strict` is off in `vite.config.js` because differences in drive-letter case (`c:\` vs `C:\`) made Vite reject its own cache.
- If the twin stays empty, check that the backend answers at http://localhost:8000/api/health.
