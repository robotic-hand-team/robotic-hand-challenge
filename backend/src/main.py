"""Dashboard backend: serves the digital twin geometry and replayable sessions.

Run from the repo root:
    python -m uvicorn backend.src.main:app --reload --port 8000
"""
import threading
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import config as C
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


@app.on_event("startup")
def warm_up():
    twin()
    list_sessions()


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
