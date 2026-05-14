"""Built-in health check factories — Push 62.

Each factory returns an async callable::

    async () -> ComponentHealth

Usage::

    registry.register_check("disk", disk_check("/data", min_free_mb=500))
    registry.register_check("memory", memory_check(max_pct=90.0))
    registry.register_check("event_loop", event_loop_check())
    registry.register_check("exchange", exchange_check(connector))
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Optional

from core.health.health_models import ComponentHealth, HealthStatus


# ---------------------------------------------------------------------------
# Infrastructure checks
# ---------------------------------------------------------------------------

def disk_check(
    path: str = "/",
    min_free_mb: float = 500.0,
    name: str = "disk",
) -> Callable:
    """Check available disk space."""
    async def _check() -> ComponentHealth:
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            free_mb = free / (1024 ** 2)
            status = HealthStatus.HEALTHY if free_mb >= min_free_mb else HealthStatus.UNHEALTHY
            return ComponentHealth(
                name=name,
                status=status,
                message=f"{free_mb:.0f} MB free on {path}",
                extra={"free_mb": round(free_mb, 1), "total_mb": round(total / 1024 ** 2, 1)},
            )
        except Exception as exc:
            return ComponentHealth(name=name, status=HealthStatus.UNHEALTHY, message=str(exc))
    return _check


def memory_check(
    max_pct: float = 90.0,
    name: str = "memory",
) -> Callable:
    """Check system memory usage percentage."""
    async def _check() -> ComponentHealth:
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            pct = vm.percent
            status = HealthStatus.HEALTHY if pct < max_pct else HealthStatus.DEGRADED
            return ComponentHealth(
                name=name,
                status=status,
                message=f"{pct:.1f}% used",
                extra={"used_pct": pct, "available_mb": round(vm.available / 1024 ** 2, 1)},
            )
        except ImportError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.DEGRADED,
                message="psutil not installed",
            )
        except Exception as exc:
            return ComponentHealth(name=name, status=HealthStatus.UNHEALTHY, message=str(exc))
    return _check


def event_loop_check(name: str = "event_loop") -> Callable:
    """Verify the asyncio event loop is running and responsive."""
    async def _check() -> ComponentHealth:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running() and not loop.is_closed():
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Event loop running",
                )
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Event loop not running",
            )
        except Exception as exc:
            return ComponentHealth(name=name, status=HealthStatus.UNHEALTHY, message=str(exc))
    return _check


# ---------------------------------------------------------------------------
# Component checks (use optional injected objects)
# ---------------------------------------------------------------------------

def exchange_check(connector: Any, name: str = "exchange") -> Callable:
    """Check exchange connector health."""
    async def _check() -> ComponentHealth:
        try:
            if hasattr(connector, "is_connected") and not connector.is_connected:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Exchange connector disconnected",
                )
            if hasattr(connector, "ping"):
                await connector.ping()
            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                message="Exchange connected",
            )
        except Exception as exc:
            return ComponentHealth(name=name, status=HealthStatus.UNHEALTHY, message=str(exc))
    return _check


def pnl_tracker_check(tracker: Any, name: str = "pnl_tracker") -> Callable:
    """Check PnLTracker liveness."""
    async def _check() -> ComponentHealth:
        try:
            equity = tracker.equity
            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                message=f"equity=${equity:,.2f}",
                extra={"equity": equity},
            )
        except Exception as exc:
            return ComponentHealth(name=name, status=HealthStatus.UNHEALTHY, message=str(exc))
    return _check


def risk_manager_check(rm: Any, name: str = "risk_manager") -> Callable:
    """Check RiskManager halt state."""
    async def _check() -> ComponentHealth:
        try:
            halted = getattr(rm, "halted", False)
            status = HealthStatus.UNHEALTHY if halted else HealthStatus.HEALTHY
            msg = "HALTED" if halted else "Risk OK"
            return ComponentHealth(name=name, status=status, message=msg)
        except Exception as exc:
            return ComponentHealth(name=name, status=HealthStatus.UNHEALTHY, message=str(exc))
    return _check
