"""HealthRegistry — runs and aggregates health checks — Push 62."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Dict, Optional

from core.health.health_models import ComponentHealth, HealthStatus, SystemHealth

logger = logging.getLogger(__name__)

CheckFn = Callable[[], Awaitable[ComponentHealth]]


class HealthRegistry:
    """Registry of named async health check functions.

    Parameters
    ----------
    version : str
        Application version tag for SystemHealth output.
    env : str
        Environment label (production / paper / backtest).
    start_time : float
        Process start time (time.time()). Used for uptime_s.
    timeout : float
        Per-check timeout in seconds (default 5.0).
    """

    def __init__(
        self,
        version: str = "7.8.0",
        env: str = "production",
        start_time: Optional[float] = None,
        timeout: float = 5.0,
    ) -> None:
        self._checks: Dict[str, CheckFn] = {}
        self._version = version
        self._env = env
        self._start_time = start_time or time.time()
        self._timeout = timeout
        self._last_result: Optional[SystemHealth] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_check(self, name: str, fn: CheckFn) -> None:
        self._checks[name] = fn
        logger.debug("HealthRegistry: registered check '%s'", name)

    def unregister_check(self, name: str) -> bool:
        return self._checks.pop(name, None) is not None

    @property
    def check_names(self):
        return list(self._checks.keys())

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run_checks(self) -> SystemHealth:
        """Run all registered checks concurrently.

        Returns a SystemHealth with overall = worst component status.
        """
        if not self._checks:
            return SystemHealth(
                overall=HealthStatus.HEALTHY,
                components={},
                uptime_s=time.time() - self._start_time,
                version=self._version,
                env=self._env,
            )

        tasks = {name: asyncio.create_task(self._run_one(name, fn))
                 for name, fn in self._checks.items()}
        results: Dict[str, ComponentHealth] = {}
        for name, task in tasks.items():
            try:
                results[name] = await asyncio.wait_for(task, timeout=self._timeout)
            except asyncio.TimeoutError:
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check timed out after {self._timeout}s",
                )
            except Exception as exc:  # noqa: BLE001
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(exc),
                )

        worst = max((ch.status for ch in results.values()), default=HealthStatus.HEALTHY)
        system = SystemHealth(
            overall=worst,
            components=results,
            uptime_s=time.time() - self._start_time,
            version=self._version,
            env=self._env,
        )
        self._last_result = system
        logger.debug(
            "HealthRegistry: %d checks, overall=%s",
            len(results), worst.label,
        )
        return system

    @staticmethod
    async def _run_one(name: str, fn: CheckFn) -> ComponentHealth:
        t0 = time.perf_counter()
        try:
            result = await fn()
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result
        except Exception as exc:  # noqa: BLE001
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    @property
    def last_result(self) -> Optional[SystemHealth]:
        return self._last_result

    # ------------------------------------------------------------------
    # Metrics (for backwards compatibility with monitoring.update_metrics)
    # ------------------------------------------------------------------

    async def update_metrics(self, metrics: Dict) -> None:
        """Store metrics dict for external consumers (e.g. Grafana pushgateway).

        This is a no-op stub that stores metrics in-memory for retrieval
        via last_metrics property.
        """
        self._last_metrics = metrics
        logger.debug("HealthRegistry: updated metrics with %d keys", len(metrics))

    @property
    def last_metrics(self) -> Optional[Dict]:
        """Return the last metrics dict passed to update_metrics."""
        return getattr(self, "_last_metrics", None)
