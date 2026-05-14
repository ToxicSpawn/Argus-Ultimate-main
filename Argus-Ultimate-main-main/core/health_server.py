"""Package: core.health_server

Standalone uvicorn runner for the Argus health endpoint.

Usage (from main.py or any async entrypoint)::

    from core.health_server import start_health_server

    async def main():
        health_task = await start_health_server(host="0.0.0.0", port=8765)
        # health_task is a running asyncio.Task — cancel it on shutdown

Or as a background thread (non-async callers)::

    from core.health_server import start_health_server_thread
    start_health_server_thread(port=8765)  # daemon thread, safe to leave running

Configuration (environment variables)::

    ARGUS_HEALTH_HOST   default 0.0.0.0
    ARGUS_HEALTH_PORT   default 8765
    ARGUS_HEALTH_ENABLED  default true (set to false to disable entirely)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_HOST = os.getenv("ARGUS_HEALTH_HOST", "0.0.0.0")
_DEFAULT_PORT = int(os.getenv("ARGUS_HEALTH_PORT", "8765"))
_HEALTH_ENABLED = os.getenv("ARGUS_HEALTH_ENABLED", "true").lower() not in {"false", "0", "no"}


async def start_health_server(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    *,
    log_level: str = "warning",
) -> Optional[asyncio.Task]:
    """Start the health server as a background asyncio Task.

    Returns the Task so callers can cancel it on shutdown.
    Returns None if health server is disabled or uvicorn/fastapi unavailable.
    """
    if not _HEALTH_ENABLED:
        logger.info("Health server disabled (ARGUS_HEALTH_ENABLED=false)")
        return None

    try:
        import uvicorn
        from core.health import get_health_app
    except ImportError as exc:
        logger.warning("Health server unavailable (%s) — skipping.", exc)
        return None

    app = get_health_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve(), name="argus-health-server")
    logger.info("Health server started on http://%s:%d/health", host, port)
    return task


def start_health_server_thread(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
) -> Optional[threading.Thread]:
    """Start the health server in a background daemon thread (for non-async callers).

    Returns the Thread object, or None if disabled/unavailable.
    """
    if not _HEALTH_ENABLED:
        return None

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(start_health_server(host=host, port=port))
        finally:
            loop.close()

    t = threading.Thread(target=_run, name="argus-health-thread", daemon=True)
    t.start()
    logger.info("Health server thread started (port=%d)", port)
    return t
