# Tactile Robotics Hand System — Hold My Sensors

**Black Forest Hackathon 2026 · Smart Automation Edition**

We estimate **which sensor** an Inspire RH56DFTP robot hand is holding (VEGABAR 29 or VEGABAR 38), **where it sits and how it is tilted** in the hand, and **whether it slips**, using only the hand's tactile pads and joint registers. A web dashboard shows the result as a live digital twin.

## Team: Hold My Sensors

| Member | GitHub |
| :--- | :--- |
| **Sami Adam Moughli** | [GitHub](https://github.com/r00tnomade) |
| **Sai Sharan Medepalli** | [GitHub](https://github.com/EnshrinedSharan) |
| **Prasanth Vallam Ramesh** | [GitHub](https://github.com/) |
| **Lars Pernickel** | [GitHub](https://github.com/LarsPernickel/) |
| **Nitinshan Fredrick William Amalraj** | [GitHub](https://github.com/) |

## Quick start

```bash
docker compose up --build
```

Then open <http://localhost:3000>. The dashboard replays recorded grasp sessions from `models/dataset_sim`: a 3D twin of the hand and the object, the estimated sensor with its confidence, the object's position and tilt, grip force, slip, and a thermal-style tactile heat map.

## What runs where

| Part | Needs | Docs |
| --- | --- | --- |
| Dashboard (`backend/` FastAPI + MuJoCo twin, `frontend/` React + three.js) | Docker only | [backend](docs/documentation/backend.md) · [frontend](docs/documentation/frontend.md) |
| Sensor classifier and pose / slip models (`classifier/`, `final_model/`, `pose/`, `matcher/`) | Python 3.11 | [classifier](docs/documentation/classifier.md) · [final_model](docs/documentation/final_model.md) · [pose](docs/documentation/pose.md) · [matcher](docs/documentation/matcher.md) |
| Grasp simulation and dataset generator (`simulation/hand_sim`) | Python 3.11, MuJoCo; the 3D viewer needs a display | [simulation](docs/documentation/simulation.md) |
| Live hand (`hardware/`) | **The real Inspire hand** on Modbus TCP (`192.168.123.211:6000`) | [hardware](docs/documentation/hardware.md) |
| Recorded data (`models/dataset`, `models/dataset_sim`) | — | [models](docs/documentation/models.md) |

**Live mode:** `hardware/config.py` reads the real hand and sends its frames over UDP to port 5005, which the backend container publishes. Without the hand, the dashboard's "Live hand" tab waits for data. Replay works everywhere.

## What the automatic build checks

The root `docker-compose.yml` is the build. `docker compose build` installs the pinned Python packages and builds the frontend. It needs no secrets, no `.env` and no hardware. The CI workflow [`judge-build.yml`](.github/workflows/judge-build.yml) runs that build on a fresh Ubuntu machine. It then starts the stack and checks that the backend health endpoint, the session list, a session replay, the frontend page and the 3D twin model all respond.

What a plain Linux build cannot cover: talking to the physical hand, and the interactive MuJoCo viewer. Both need hardware or a display.

## Local development

```bash
python3.11 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python -m uvicorn backend.src.main:app --port 8000      # backend
cd frontend && npm ci && npm run dev                               # frontend on http://localhost:3000
cd simulation && ../venv/bin/python -m hand_sim.test_smoke         # simulation smoke tests
```

## Licence note

The Inspire hand 3D assets come from dex-urdf and are licensed CC BY-NC-SA 4.0.
