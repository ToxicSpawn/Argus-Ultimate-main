"""TradingEngine — base lifecycle for all Argus trading loops.

Extracted from unified_trading_system.py.
Provides: startup, main loop tick, graceful shutdown, health check.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TradingEngine(ABC):
    """
    Abstract base for all Argus trading engine implementations.

    Subclass and implement:
    - on_startup() — called once before main loop
    - on_tick() — called every loop iteration
    - on_shutdown() — called on graceful exit
    """

    def __init__(self, config: Any, *, loop_interval_s: float = 1.0) -> None:
        self.config = config
        self.loop_interval_s = float(loop_interval_s)
        self._running = False
        self._tick_count = 0
        self._start_time: Optional[float] = None
        self._last_tick_latency_ms: float = 0.0
        self._errors: int = 0
        logger.info("%s initialised | interval=%.1fs", self.__class__.__name__, self.loop_interval_s)

    @abstractmethod
    async def on_startup(self) -> None:
        """One-time initialisation: connect exchanges, warm up models."""

    @abstractmethod
    async def on_tick(self) -> None:
        """Called every loop_interval_s. Core trading logic goes here."""

    @abstractmethod
    async def on_shutdown(self) -> None:
        """Graceful shutdown: cancel orders, flush state, disconnect."""

    async def health_check(self) -> Dict[str, Any]:
        """Return a health snapshot for Prometheus / dashboard polling."""
        uptime_s = time.time() - self._start_time if self._start_time else 0
        return {
            "engine": self.__class__.__name__,
            "running": self._running,
            "uptime_s": round(uptime_s, 1),
            "ticks": self._tick_count,
            "errors": self._errors,
            "last_tick_latency_ms": round(self._last_tick_latency_ms, 3),
        }

    async def run(self) -> None:
        """Start the engine. Blocks until stopped or SIGINT/SIGTERM."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        self._running = True
        self._start_time = time.time()

        try:
            await self.on_startup()
            logger.info("%s started", self.__class__.__name__)

            while self._running:
                t0 = time.perf_counter()
                try:
                    await self.on_tick()
                    self._tick_count += 1
                except Exception as exc:
                    self._errors += 1
                    logger.exception("%s tick error (total=%d): %s", self.__class__.__name__, self._errors, exc)

                elapsed = time.perf_counter() - t0
                self._last_tick_latency_ms = elapsed * 1000.0
                sleep_s = max(0.0, self.loop_interval_s - elapsed)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

        finally:
            logger.info("%s shutting down after %d ticks", self.__class__.__name__, self._tick_count)
            try:
                await self.on_shutdown()
            except Exception as exc:
                logger.exception("Shutdown error: %s", exc)
            self._running = False

    def _request_stop(self) -> None:
        logger.info("%s stop requested", self.__class__.__name__)
        self._running = False

    def stop(self) -> None:
        """Programmatic stop (e.g. from risk circuit breaker)."""
        self._request_stop()
