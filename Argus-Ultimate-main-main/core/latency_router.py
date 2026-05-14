"""
Latency Router — measure per-exchange latency and route orders to fastest venue.

Pings each exchange every N seconds, maintains rolling latency statistics
(p50/p95/p99), and exposes ``get_fastest_venue(symbol)`` for the execution
layer to auto-route orders to the exchange with lowest round-trip time.

Features:
  - Background async ping loop (configurable interval, default 30s)
  - Rolling window of latency samples per exchange (default 100)
  - p50/p95/p99 statistics
  - Degradation warnings when latency exceeds 2x rolling median
  - ``get_fastest_venue(symbol)`` returns best exchange for a given pair
  - Thread-safe sample collection

Usage:
    router = LatencyRouter(exchange_manager=em)
    await router.start()
    venue = router.get_fastest_venue("BTC/USD")
    report = router.get_latency_report()
    await router.stop()
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _LatencySamples:
    """Rolling window of latency measurements for one exchange."""
    exchange: str
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_ping_time: float = 0.0
    last_latency_ms: float = 0.0
    degraded: bool = False
    consecutive_failures: int = 0

    def record(self, latency_ms: float) -> None:
        self.samples.append(latency_ms)
        self.last_latency_ms = latency_ms
        self.last_ping_time = time.monotonic()
        self.consecutive_failures = 0

    def percentile(self, pct: float) -> float:
        if not self.samples:
            return 0.0
        return float(np.percentile(list(self.samples), pct))

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def median(self) -> float:
        return self.p50

    @property
    def sample_count(self) -> int:
        return len(self.samples)


class LatencyRouter:
    """
    Latency-aware order routing engine.

    Measures round-trip time to each exchange and routes orders to the
    fastest venue that supports the requested symbol.

    Parameters
    ----------
    exchange_manager
        Object with ``exchanges`` dict and ``get_ticker(symbol, exchange)``
        async method for pinging.
    ping_interval_s
        Seconds between ping rounds (default 30).
    degradation_threshold
        Warn when latency exceeds this multiple of rolling median (default 2.0).
    venue_symbols
        Optional mapping of venue -> list of supported symbols.
        If not provided, all venues are assumed to support all symbols.
    """

    def __init__(
        self,
        exchange_manager: Any = None,
        ping_interval_s: float = 30.0,
        degradation_threshold: float = 2.0,
        venue_symbols: Optional[Dict[str, List[str]]] = None,
        window_size: int = 100,
    ) -> None:
        self._exchange_manager = exchange_manager
        self._ping_interval_s = ping_interval_s
        self._degradation_threshold = degradation_threshold
        self._venue_symbols = venue_symbols or {}
        self._window_size = window_size

        self._samples: Dict[str, _LatencySamples] = {}
        self._ping_task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background ping loop."""
        if self._running:
            return
        self._running = True
        # Initial ping
        await self._ping_all()
        self._ping_task = asyncio.create_task(self._ping_loop())
        logger.info("LatencyRouter: started (interval=%.0fs)", self._ping_interval_s)

    async def stop(self) -> None:
        """Stop the ping loop."""
        self._running = False
        if self._ping_task is not None:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        logger.info("LatencyRouter: stopped")

    # ------------------------------------------------------------------
    # Manual sample recording (for callers who measure their own latency)
    # ------------------------------------------------------------------

    def record_latency(self, exchange: str, latency_ms: float) -> None:
        """Manually record a latency sample for *exchange*."""
        entry = self._get_or_create(exchange)
        entry.record(latency_ms)
        self._check_degradation(exchange, entry)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def get_fastest_venue(self, symbol: str) -> Optional[str]:
        """
        Return the venue with lowest median latency that supports *symbol*.

        Returns None if no latency data is available.
        """
        candidates = []
        for exchange, entry in self._samples.items():
            if entry.sample_count == 0:
                continue
            # Check symbol support if venue_symbols is configured
            if self._venue_symbols:
                supported = self._venue_symbols.get(exchange, [])
                if supported and symbol not in supported:
                    continue
            candidates.append((exchange, entry.median))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def get_venue_ranking(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all venues ranked by median latency, optionally filtered by symbol."""
        ranking = []
        for exchange, entry in self._samples.items():
            if entry.sample_count == 0:
                continue
            if symbol and self._venue_symbols:
                supported = self._venue_symbols.get(exchange, [])
                if supported and symbol not in supported:
                    continue
            ranking.append({
                "exchange": exchange,
                "median_ms": entry.median,
                "p95_ms": entry.p95,
                "p99_ms": entry.p99,
                "last_ms": entry.last_latency_ms,
                "samples": entry.sample_count,
                "degraded": entry.degraded,
            })
        ranking.sort(key=lambda x: x["median_ms"])
        return ranking

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_latency_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Return p50/p95/p99 latency per exchange.

        Returns dict mapping exchange name to latency stats.
        """
        report: Dict[str, Dict[str, Any]] = {}
        for exchange, entry in self._samples.items():
            report[exchange] = {
                "p50_ms": entry.p50,
                "p95_ms": entry.p95,
                "p99_ms": entry.p99,
                "last_ms": entry.last_latency_ms,
                "samples": entry.sample_count,
                "degraded": entry.degraded,
                "consecutive_failures": entry.consecutive_failures,
            }
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, exchange: str) -> _LatencySamples:
        if exchange not in self._samples:
            self._samples[exchange] = _LatencySamples(
                exchange=exchange,
                samples=deque(maxlen=self._window_size),
            )
        return self._samples[exchange]

    def _check_degradation(self, exchange: str, entry: _LatencySamples) -> None:
        """Check if current latency indicates degradation."""
        if entry.sample_count < 5:
            return
        median = entry.median
        if median <= 0:
            return
        ratio = entry.last_latency_ms / median
        if ratio >= self._degradation_threshold:
            if not entry.degraded:
                entry.degraded = True
                logger.warning(
                    "LatencyRouter: %s DEGRADED — %.1fms (%.1fx median %.1fms)",
                    exchange, entry.last_latency_ms, ratio, median,
                )
        else:
            if entry.degraded:
                logger.info("LatencyRouter: %s recovered (%.1fms)", exchange, entry.last_latency_ms)
            entry.degraded = False

    async def _ping_all(self) -> None:
        """Ping all known exchanges and record latency."""
        if self._exchange_manager is None:
            return

        exchanges = getattr(self._exchange_manager, "active_exchanges", [])
        if not exchanges:
            exchanges = list(getattr(self._exchange_manager, "exchanges", {}).keys())

        for exchange in exchanges:
            entry = self._get_or_create(exchange)
            try:
                t0 = time.perf_counter()
                # Use get_ticker as a lightweight ping
                if hasattr(self._exchange_manager, "get_ticker"):
                    await self._exchange_manager.get_ticker("BTC/USD", exchange=exchange)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                entry.record(latency_ms)
                self._check_degradation(exchange, entry)
                logger.debug("LatencyRouter: %s ping %.1fms", exchange, latency_ms)
            except Exception as exc:
                entry.consecutive_failures += 1
                if entry.consecutive_failures >= 3:
                    entry.degraded = True
                logger.debug("LatencyRouter: %s ping failed: %s", exchange, exc)

    async def _ping_loop(self) -> None:
        """Background loop that pings all exchanges periodically."""
        try:
            while self._running:
                await asyncio.sleep(self._ping_interval_s)
                if not self._running:
                    break
                await self._ping_all()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("LatencyRouter ping loop error: %s", exc)
