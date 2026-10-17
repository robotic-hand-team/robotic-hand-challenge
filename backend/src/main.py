"""Dashboard backend: serves the digital twin geometry, replayable sessions and the live hand.

Run from the repo root:
    python -m uvicorn backend.src.main:app --port 8000
Live hand: start hardware/config.py; the backend receives its UDP broadcast on port 5005.
"""
import asyncio
import threading
from functools import lru_cache

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import config as C
from .live import LivePipeline, start_udp
from .sessions import list_sessions, load_session
from .twin import Twin

app = FastAPI(title="VEGA Tactile Twin")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

GLB = C.CACHE_DIR / f"twin_{C.HAND_SIDE}.glb"


@lru_cache(maxsize=1)
def twin() -> Twin:
    t = Twin()
    if not GLB.exists():
        t.export_glb(GLB)
    return t


_lock = threading.Lock()  # one MjData is shared, and FastAPI runs sync endpoints in parallel threads


@lru_cache(maxsize=32)
def _session(session_id: str) -> dict:
    return load_session(session_id, twin())


def session(session_id: str) -> dict:
    with _lock:
        return _session(session_id)


live: LivePipeline | None = None


@app.on_event("startup")
async def warm_up():
    global live
    twin()
    list_sessions()
    live = LivePipeline(twin())
    try:
        await start_udp(live)
    except OSError as e:
        print(f"[live] cannot listen on udp://{C.LIVE_UDP_HOST}:{C.LIVE_UDP_PORT}: {e}")


@app.get("/api/live/status")
def live_status():
    return live.status()


@app.websocket("/api/live")
async def live_ws(ws: WebSocket):
    """Pushes every new live frame (about 20 Hz), plus a status message every second."""
    await ws.accept()
    last_seq, last_status = None, 0.0
    loop = asyncio.get_running_loop()
    try:
        while True:
            frame = live.latest
            if frame is not None and frame["seq"] != last_seq:
                last_seq = frame["seq"]
                await ws.send_json({"type": "frame", "frame": frame, "status": live.status()})
            elif loop.time() - last_status > 1.0:
                last_status = loop.time()
                await ws.send_json({"type": "status", "status": live.status()})
            await asyncio.sleep(0.02)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/api/health")
def health():
    return {"ok": True, "dataset": str(C.DATASET_DIR), "sessions": len(list_sessions())}


@app.get("/api/sessions")
def sessions():
    return list_sessions()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in {s["id"] for s in list_sessions()}:
        raise HTTPException(404, f"unknown session {session_id}")
    return session(session_id)


@app.get("/api/twin.glb")
def twin_glb():
    twin()
    return FileResponse(GLB, media_type="model/gltf-binary")
