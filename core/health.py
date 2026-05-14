"""M05/M06 — FastAPI /health endpoint (async lifespan).

Mounts automatically when imported by main.py.
"""
from __future__ import annotations

import time
from typing import Any

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False

_START_TIME: float = time.monotonic()


def build_health_app() -> Any:  # returns FastAPI | None
    """Return a FastAPI sub-application exposing /health."""
    if not _FASTAPI_AVAILABLE:
        return None  # pragma: no cover

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        app.state.start_time = time.monotonic()
        yield

    app = FastAPI(title="Argus Health", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        uptime = round(time.monotonic() - _START_TIME, 1)
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "uptime_seconds": uptime,
                "version": _get_version(),
            },
        )

    @app.get("/health/live")
    async def liveness() -> JSONResponse:
        return JSONResponse(status_code=200, content={"alive": True})

    @app.get("/health/ready")
    async def readiness() -> JSONResponse:
        return JSONResponse(status_code=200, content={"ready": True})

    return app


def _get_version() -> str:
    try:
        from version import __version__
        return __version__
    except Exception:
        return "unknown"


# Module-level singleton used by main.py
health_app = build_health_app()
