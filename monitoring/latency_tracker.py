"""
Latency Tracker — measures and alerts on trading system latency.

Tracks:
  - Signal-to-order latency (time from signal generation to order submission)
  - Order-to-fill latency (time from order submission to fill confirmation)
  - Exchange API round-trip latency
  - Strategy cycle time (full loop duration)

Alerts when latency exceeds thresholds (e.g., order submission > 500ms).

Usage:
    tracker = LatencyTracker()
    with tracker.measure("signal_to_order"):
        submit_order()
    stats = tracker.get_stats("signal_to_order")
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Deque, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """A single latency observation."""

    operation: str
    latency_ms: float
    timestamp: float  # Unix epoch (time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Aggregated statistics for one operation type."""

    operation: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    alert_count: int


def _compute_percentile(values: list[float], pct: float) -> float:
    """
    Compute a single percentile from a sorted list of floats.

    Lazy-imports numpy for accuracy; falls back to a linear-interpolation
    implementation if numpy is unavailable.
    """
    if not values:
        return 0.0
    try:
        import numpy as np  # type: ignore[import]

        return float(np.percentile(values, pct))
    except ImportError:
        # Pure-Python fallback: linear interpolation (same as numpy default).
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        rank = pct / 100.0 * (n - 1)
        lo = int(rank)
        hi = lo + 1
        if hi >= n:
            return sorted_vals[-1]
        frac = rank - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class LatencyTracker:
    """
    Thread-safe latency measurement and alerting facility.

    One deque per named operation holds the most recent measurements.
    Statistics (mean, p50, p95, p99, max) are computed on demand from the
    deque contents.

    Alert callbacks receive ``(operation: str, latency_ms: float,
    threshold_ms: float)`` and run synchronously in the calling thread.
    """

    THRESHOLDS: Dict[str, float] = {
        "signal_to_order": 500.0,   # ms
        "order_to_fill": 5000.0,    # ms
        "api_round_trip": 200.0,    # ms
        "strategy_cycle": 1000.0,   # ms
    }

    def __init__(
        self,
        max_history: int = 1000,
        alert_callback: Optional[Callable[[str, float, float], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_history:
            Maximum number of measurements to keep per operation.
            Older entries are evicted automatically (deque with maxlen).
        alert_callback:
            Optional callable invoked whenever a measurement exceeds its
            operation threshold.  Signature:
                callback(operation, latency_ms, threshold_ms) -> None
        """
        self._max_history = max_history
        self._alert_callback = alert_callback
        self._lock = threading.Lock()
        # history: operation name → deque of LatencyMeasurement
        self._history: Dict[str, Deque[LatencyMeasurement]] = {}
        # alert counter per operation
        self._alert_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Context manager: measure a block of code
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def measure(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        """
        Context manager that records the wall-clock time of the enclosed block.

        Example::

            with tracker.measure("signal_to_order"):
                submit_order(signal)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1_000.0
            self.record(operation, elapsed_ms, metadata=metadata)

    # ------------------------------------------------------------------
    # Direct recording
    # ------------------------------------------------------------------

    def record(
        self,
        operation: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a single latency measurement and fire an alert if needed.

        Parameters
        ----------
        operation:
            Logical name (e.g. "signal_to_order", "api_round_trip").
        latency_ms:
            Elapsed time in milliseconds.
        metadata:
            Arbitrary key-value pairs attached to the measurement for
            downstream diagnostics (e.g. {"symbol": "BTC/AUD"}).
        """
        measurement = LatencyMeasurement(
            operation=operation,
            latency_ms=latency_ms,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with self._lock:
            if operation not in self._history:
                self._history[operation] = deque(maxlen=self._max_history)
                self._alert_counts[operation] = 0
            self._history[operation].append(measurement)

            threshold = self.THRESHOLDS.get(operation)
            if threshold is not None and latency_ms > threshold:
                self._alert_counts[operation] += 1
                alert_count = self._alert_counts[operation]

        # Fire callback outside the lock to avoid deadlocks.
        if threshold is not None and latency_ms > threshold:
            logger.warning(
                "Latency alert: operation='%s' latency=%.1f ms threshold=%.1f ms (alert #%d)",
                operation,
                latency_ms,
                threshold,
                alert_count,
            )
            if self._alert_callback is not None:
                try:
                    self._alert_callback(operation, latency_ms, threshold)
                except Exception:
                    logger.exception(
                        "LatencyTracker alert_callback raised for operation='%s'",
                        operation,
                    )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, operation: str) -> LatencyStats:
        """
        Compute and return aggregated statistics for one operation.

        Returns a zeroed LatencyStats if no measurements have been recorded.
        """
        with self._lock:
            history = list(self._history.get(operation, []))
            alert_count = self._alert_counts.get(operation, 0)

        if not history:
            return LatencyStats(
                operation=operation,
                count=0,
                mean_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
                alert_count=0,
            )

        values = [m.latency_ms for m in history]
        count = len(values)
        mean_ms = sum(values) / count
        max_ms = max(values)
        p50_ms = _compute_percentile(values, 50.0)
        p95_ms = _compute_percentile(values, 95.0)
        p99_ms = _compute_percentile(values, 99.0)

        return LatencyStats(
            operation=operation,
            count=count,
            mean_ms=mean_ms,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            max_ms=max_ms,
            alert_count=alert_count,
        )

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Return LatencyStats for every operation that has been recorded."""
        with self._lock:
            operations = list(self._history.keys())
        return {op: self.get_stats(op) for op in operations}

    # ------------------------------------------------------------------
    # Monitoring dashboard snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a serialisable dict suitable for pushing to a monitoring
        dashboard (e.g. Grafana, the ARGUS health endpoint).

        Structure::

            {
                "operations": {
                    "signal_to_order": {
                        "count": 42,
                        "mean_ms": 123.4,
                        "p50_ms": 110.2,
                        "p95_ms": 490.1,
                        "p99_ms": 498.7,
                        "max_ms": 502.3,
                        "alert_count": 1,
                        "threshold_ms": 500.0
                    },
                    ...
                },
                "total_alerts": 3,
                "timestamp": 1741600000.0
            }
        """
        all_stats = self.get_all_stats()
        ops: Dict[str, dict] = {}
        total_alerts = 0
        for op, stats in all_stats.items():
            ops[op] = {
                "count": stats.count,
                "mean_ms": round(stats.mean_ms, 3),
                "p50_ms": round(stats.p50_ms, 3),
                "p95_ms": round(stats.p95_ms, 3),
                "p99_ms": round(stats.p99_ms, 3),
                "max_ms": round(stats.max_ms, 3),
                "alert_count": stats.alert_count,
                "threshold_ms": self.THRESHOLDS.get(op),
            }
            total_alerts += stats.alert_count
        return {
            "operations": ops,
            "total_alerts": total_alerts,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def set_threshold(self, operation: str, threshold_ms: float) -> None:
        """
        Override or add an alert threshold for an operation at runtime.

        Changes are reflected immediately for all subsequent recordings.
        """
        self.THRESHOLDS = {**self.THRESHOLDS, operation: threshold_ms}
        logger.info(
            "LatencyTracker threshold updated: operation='%s' threshold=%.1f ms",
            operation,
            threshold_ms,
        )

    def clear_history(self, operation: Optional[str] = None) -> None:
        """
        Clear recorded measurements.

        If ``operation`` is given, clears only that operation's history.
        Otherwise clears all history.  Alert counts are also reset.
        """
        with self._lock:
            if operation is not None:
                self._history.pop(operation, None)
                self._alert_counts.pop(operation, None)
            else:
                self._history.clear()
                self._alert_counts.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_tracker: Optional[LatencyTracker] = None
_global_lock = threading.Lock()


def get_tracker(
    max_history: int = 1000,
    alert_callback: Optional[Callable[[str, float, float], None]] = None,
) -> LatencyTracker:
    """
    Return the module-level LatencyTracker singleton, creating it if needed.

    Subsequent calls with different arguments do NOT reconfigure the existing
    singleton; pass arguments only on the first call (or create your own
    LatencyTracker directly).
    """
    global _global_tracker
    if _global_tracker is None:
        with _global_lock:
            if _global_tracker is None:
                _global_tracker = LatencyTracker(
                    max_history=max_history,
                    alert_callback=alert_callback,
                )
    return _global_tracker
