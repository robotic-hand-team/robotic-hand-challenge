# `hardware/`: controlling and recording the real hand

This folder holds everything that talks to the physical **Inspire RH56DFTP** hand. It drives the
hand, records grasp sessions to disk and streams the live hand state to other programs.

## Files

| File | What it does |
|---|---|
| `config.py` | **The main hand application** (the name is historical): GUI, grasp logic, logging and UDP broadcast |
| `plot_logs.py` | Plots a session that `config.py` recorded: joints, forces, tactile heat maps |
| `replay_to_twin.py` | Replays a recorded session as if it were the live hand, over UDP |
| `hand.png` | Line drawing of the hand, used as the background of tactile heat maps |
| `requirements.txt` | Pinned packages: pymodbus 3.16.1, numpy, pandas, pillow. `plot_logs.py` also needs matplotlib |
| `logs/` | Created at runtime and git-ignored. One folder per recorded session |

## `config.py`: the hand GUI

```bash
python hardware/config.py
```

The hand has to be reachable on the network. The connection settings are at the top of the file:

```python
IP = "192.168.123.210"     # the team's LEFT hand (the right hand is out of service)
PORT = 6000                # Modbus TCP, device id 1
```

### What it does

- **Controls the hand** with the **Open**, **Close**, **Wave hi**, **Step + / Step −**, **Auto hold**, **Grip force**, **Save position** and **Clear errors** controls.
- **Close runs a staged grasp.** The fingers close one after another, the thumb rotates in slowly, and the thumb can optionally probe for contact. The force limits rise to the full grip force in the last stage.
- **Tracks the hand's state** through `IDLE → CLOSING → CONTACT → HOLD → RELEASING → OPENED`, which the GUI shows.
- **Has safety guards:**
  - A thumb-rotation guard stops the thumb when it pushes the object instead of gripping it.
  - No writes are sent while a finger is pressing hard, because that froze the motor controller in tests.
  - The GUI warns about high motor temperature.
- **Detects slip** during HOLD, from the contact centre shifting, the pressure dropping or a finger being pushed open. The result is shown in the GUI.
- **Calibrates the tactile sensors** at startup: it averages the first samples as a zero offset and subtracts it from every later reading.
- **Logs sessions.** **● Start logging** writes `logs/<YYYYmmdd_HHMMSS>/` with `actuators.csv`, `tactile_raw.csv`, `tactile_summary.csv` and `meta.json`, the same format as `models/dataset/`.
- **Broadcasts live state** as one JSON datagram per frame (about 40 Hz) to `127.0.0.1:5005` (`BROADCAST_HOST` / `BROADCAST_PORT`). The dashboard backend (`backend/`) listens on that port, so starting the dashboard (`docker compose up --build`) shows the grasp live at http://localhost:3000. With Docker Desktop, keep `127.0.0.1` or use this PC's address; `255.255.255.255` doesn't reach the container.

### Main settings (constants at the top of `config.py`)

| Constant | Meaning |
|---|---|
| `SAMPLE_RATE_HZ` | Target acquisition and logging rate (50 Hz, limited by Modbus speed) |
| `LOG_AUTOSTART` | Start logging automatically after calibration |
| `FORCE_LIMIT` | Grip force for the fingers, 0–3000 g |
| `SPEED_FINGERS`, `SPEED_THUMB` | Movement speeds, 0–1000 (1000 = full stroke in about 0.6 s) |
| `SLIP_*` | Slip detection thresholds |
| `BROADCAST_HOST`, `BROADCAST_PORT` | Where the live UDP stream goes |

## `plot_logs.py`

```bash
python hardware/plot_logs.py                              # newest session in hardware/logs
python hardware/plot_logs.py hardware/logs/<session>      # a specific session
python hardware/plot_logs.py --frame 20                   # also show tactile heat maps of frame 20
python hardware/plot_logs.py --save                       # save PNGs into the session folder
```

## `replay_to_twin.py`

This sends a recorded session over UDP exactly as `config.py` would, so you can test the live dashboard or a digital twin without the hand.

```bash
python hardware/replay_to_twin.py                         # newest session in models/dataset
python hardware/replay_to_twin.py 20261016_204513 --loop  # one session, repeated
python hardware/replay_to_twin.py --speed 2 --no-tactile  # faster, joints only
```

The other options are `--host` / `--port` (default `127.0.0.1:5005`) and `--hand left|right`.

## Hardware facts worth knowing

- The source of truth is the manual, [docs/internal/hand_manual.pdf](../internal/hand_manual.pdf): the register table is on p.16–17 and the tactile layout on p.26–27.
- **DOF order:** `little, ring, middle, index, thumb_bend, thumb_rot`.
- **Angles:** 0–1000, where 1000 = open. **Forces:** grams.
- **Register reads:** keep them at 120 registers or fewer per request; see `safe_read()`.
- **pymodbus 3.10 or newer:** the code uses `device_id=`, not the older `slave=`.
- **Palm tactile data** (8×14) arrives column-major, bottom-to-top.
- **Status bytes:** `0` releasing, `1` closing or holding, `2` stopped by the force limit, `255` not initialised.
