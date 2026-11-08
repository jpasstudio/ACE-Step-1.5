"""OneFlow wrapper for ACE-Step's API with shared-GPU lifecycle control.

ACE-Step stays socket-activated through the configured systemd socket, but the model is not allowed to
pin RTX VRAM forever. Activity is published to a local state file (never polled
through the socket, which would wake a cold service) and an idle safety timer
releases the process after ACESTEP_ONEFLOW_IDLE_RELEASE_SECONDS.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from acestep.api_server import app

from oneflow_reference_paths import LegacyVoiceReferenceMiddleware, reference_pool

REF_DIR = reference_pool()
app.add_middleware(LegacyVoiceReferenceMiddleware)
ALLOWED_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
MAX_REF_BYTES = 30 * 1024 * 1024
STATE_PATH = Path(os.environ.get(
    "ACESTEP_ONEFLOW_STATE_PATH",
    str(Path.home() / ".cache" / "acestep" / "oneflow-gpu-state.json"),
))
IDLE_RELEASE_SECONDS = max(0, int(os.environ.get("ACESTEP_ONEFLOW_IDLE_RELEASE_SECONDS", "600")))
EXPLICIT_RELEASE_EXIT = os.environ.get(
    "ACESTEP_ONEFLOW_EXPLICIT_RELEASE_EXIT",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
PASSIVE_PATHS = {"/health", "/v1/stats", "/v1/models", "/v1/oneflow/gpu_state", "/v1/release_gpu"}

_active_requests = 0
_last_activity_ms = int(time.time() * 1000)
_idle_handle = None
_loop = None
_release_pending = False


def _write_state(*, resident: bool = True, reason: str | None = None) -> None:
    now_ms = int(time.time() * 1000)
    payload = {
        "pid": os.getpid(),
        "resident": resident,
        "busy": bool(_active_requests),
        "active_requests": _active_requests,
        "release_pending": _release_pending,
        "last_activity_ms": _last_activity_ms,
        "idle_for_ms": max(0, now_ms - _last_activity_ms),
        "idle_release_ms": IDLE_RELEASE_SECONDS * 1000,
        "updated_at_ms": now_ms,
        "status": "BUSY" if _active_requests else ("WARM" if resident else "COLD"),
    }
    if reason:
        payload["reason"] = reason
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _cancel_idle_timer() -> None:
    global _idle_handle
    if _idle_handle is not None:
        _idle_handle.cancel()
        _idle_handle = None


def _release_for_idle() -> None:
    global _release_pending
    if _active_requests:
        _schedule_idle_timer()
        return
    _release_pending = True
    _write_state(resident=False, reason="idle_timeout")
    os._exit(0)


def _schedule_idle_timer() -> None:
    global _idle_handle
    _cancel_idle_timer()
    if IDLE_RELEASE_SECONDS <= 0 or _loop is None or _active_requests:
        return
    _idle_handle = _loop.call_later(IDLE_RELEASE_SECONDS, _release_for_idle)


@app.on_event("startup")
async def _oneflow_lifecycle_startup():
    global _loop, _last_activity_ms
    _loop = asyncio.get_running_loop()
    _last_activity_ms = int(time.time() * 1000)
    _write_state(resident=True, reason="startup")
    _schedule_idle_timer()


@app.on_event("shutdown")
async def _oneflow_lifecycle_shutdown():
    _cancel_idle_timer()
    _write_state(resident=False, reason="shutdown")


@app.middleware("http")
async def _oneflow_activity_guard(request: Request, call_next):
    global _active_requests, _last_activity_ms
    if request.url.path in PASSIVE_PATHS:
        return await call_next(request)
    if _release_pending:
        return JSONResponse(
            status_code=503,
            content={"detail": "ACE-Step release in progress; retry after socket restart"},
            headers={"Retry-After": "2"},
        )
    _cancel_idle_timer()
    _active_requests += 1
    _last_activity_ms = int(time.time() * 1000)
    _write_state(resident=True)
    try:
        return await call_next(request)
    finally:
        _active_requests = max(0, _active_requests - 1)
        _last_activity_ms = int(time.time() * 1000)
        _write_state(resident=True)
        if _active_requests == 0:
            _schedule_idle_timer()


@app.get("/v1/oneflow/gpu_state")
async def oneflow_gpu_state():
    _write_state(resident=True)
    return {
        "data": json.loads(STATE_PATH.read_text(encoding="utf-8")),
        "code": 200,
        "error": None,
        "extra": None,
    }


@app.post("/v1/release_gpu")
async def release_gpu():
    global _release_pending
    _cancel_idle_timer()
    if not EXPLICIT_RELEASE_EXIT:
        # A retry may already have been admitted while the prior worker's
        # finally-block is still releasing its lease. Keep the socket process
        # alive in this governed mode so that stale release cannot kill the
        # newly admitted task and erase the in-memory result registry.
        _write_state(resident=True, reason="explicit_release_deferred")
        return {
            "data": {"status": "deferred", "models_initialized": True},
            "code": 200,
            "error": None,
            "extra": {"exit_disabled": True},
        }
    if _active_requests:
        _write_state(resident=True, reason="explicit_release_busy")
        return {
            "data": {"status": "busy", "models_initialized": True},
            "code": 409,
            "error": "active requests still running",
            "extra": {"active_requests": _active_requests},
        }
    _release_pending = True
    _write_state(resident=False, reason="explicit_release")
    loop = asyncio.get_running_loop()
    loop.call_later(0.75, lambda: os._exit(0))
    return {
        "data": {"status": "released", "models_initialized": False},
        "code": 200,
        "error": None,
        "extra": None,
    }


@app.post("/v1/upload_reference")
async def upload_reference(file: UploadFile = File(...)):
    """Persist a voice-reference clip on this host for later generation."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        ext = ".mp3"
    dest = REF_DIR / f"{uuid.uuid4().hex}{ext}"
    size = 0
    try:
        with dest.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_REF_BYTES:
                    raise HTTPException(status_code=413, detail="reference clip too large (max 30MB)")
                fh.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    if size < 1024:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded audio is empty")
    return {
        "data": {"path": str(dest), "bytes": size},
        "code": 200,
        "error": None,
        "extra": None,
    }
