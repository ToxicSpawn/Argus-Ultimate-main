"""FastAPI health + readiness + liveness endpoints — Push 62.

Endpoints::

    GET /health        Full SystemHealth JSON (all components)
    GET /health/ready  200 if HEALTHY or DEGRADED; 503 if UNHEALTHY
    GET /health/live   Always 200 — liveness probe

Mount::

    app.include_router(health_router(registry))

Prometheus::

    argus_health_status{component}   0=HEALTHY 1=DEGRADED 2=UNHEALTHY
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Gauge
    _GAUGE = Gauge(
        "argus_health_status",
        "Component health status (0=healthy 1=degraded 2=unhealthy)",
        ["component"],
    )
except Exception:
    _GAUGE = None


def health_router(registry=None):
    """Return a FastAPI APIRouter with /health, /health/ready, /health/live."""
    try:
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.warning("health_router: FastAPI not installed")
        return None

    from core.health.health_models import HealthStatus
    from core.health.health_registry import HealthRegistry

    _registry: HealthRegistry = registry or HealthRegistry()
    router = APIRouter(tags=["health"])

    @router.get("/health")
    async def health_full():
        """Full system health report."""
        system = await _registry.run_checks()
        _update_prometheus(system)
        return JSONResponse(
            content=system.to_dict(),
            status_code=200 if system.is_ready else 503,
        )

    @router.get("/health/ready")
    async def health_ready():
        """Kubernetes readiness probe."""
        system = await _registry.run_checks()
        if system.is_ready:
            return JSONResponse({"status": "ready"}, status_code=200)
        return JSONResponse(
            {"status": "not_ready", "overall": system.overall.label},
            status_code=503,
        )

    @router.get("/health/live")
    async def health_live():
        """Kubernetes liveness probe — always 200."""
        return JSONResponse({"status": "alive"}, status_code=200)

    return router


def _update_prometheus(system) -> None:
    if _GAUGE is None:
        return
    try:
        for name, comp in system.components.items():
            _GAUGE.labels(component=name).set(comp.status.value)
    except Exception:  # noqa: BLE001
        pass
