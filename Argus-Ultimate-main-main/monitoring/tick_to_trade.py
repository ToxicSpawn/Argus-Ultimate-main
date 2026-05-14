"""
Tick-to-Trade Latency Monitor — measure every stage from market data tick
through signal generation to order acknowledgment.

Tracks each stage independently and computes p50/p95/p99 statistics.
Logs a WARNING when total tick-to-trade exceeds a configurable threshold
(default 500ms).

Stages:
  1. tick_received     — timestamp when market data tick arrives
  2. signal_generated  — timestamp when strategy produces signal
  3. risk_checked      — timestamp after risk gate approval
  4. order_submitted   — timestamp when order is sent to exchange
  5. order_acknowledged — timestamp when exchange confirms receipt

Usage:
    monitor = TickToTradeMonitor(threshold_ms=500.0)
    monitor.record_tick_to_trade({
        "tick_received": 1710000000.001,
        "signal_generated": 1710000000.050,
        "risk_checked": 1710000000.055,
        "order_submitted": 1710000000.060,
        "order_acknowledged": 1710000000.120,
    })
    stats = monitor.get_stats()
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Standard pipeline stages
STAGES = [
    "tick_received",
    "signal_generated",
    "risk_checked",
    "order_submitted",
    "order_acknowledged",
]

# Derived intervals between consecutive stages
INTERVALS = [
    ("tick_to_signal", "tick_received", "signal_generated"),
    ("signal_to_risk", "signal_generated", "risk_checked"),
    ("risk_to_submit", "risk_checked", "order_submitted"),
    ("submit_to_ack", "order_submitted", "order_acknowledged"),
    ("total", "tick_received", "order_acknowledged"),
]


@dataclass
class _IntervalSamples:
    """Rolling window of interval measurements in milliseconds."""
    name: str
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=500))

    def record(self, ms: float) -> None:
        self.samples.append(ms)

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
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean(list(self.samples)))

    @property
    def last(self) -> float:
        if not self.samples:
            return 0.0
        return self.samples[-1]


class TickToTradeMonitor:
    """
    Measures and tracks tick-to-trade latency across every pipeline stage.

    Parameters
    ----------
    threshold_ms
        Log WARNING when total tick-to-trade exceeds this value.
    window_size
        Number of samples to keep per interval.
    """

    __slots__ = (
        "_threshold_ms",
        "_window_size",
        "_intervals",
        "_total_records",
        "_threshold_breaches",
    )

    def __init__(
        self,
        threshold_ms: float = 500.0,
        window_size: int = 500,
    ) -> None:
        self._threshold_ms = threshold_ms
        self._window_size = window_size
        self._intervals: Dict[str, _IntervalSamples] = {}
        self._total_records: int = 0
        self._threshold_breaches: int = 0

        # Pre-allocate interval trackers
        for name, _, _ in INTERVALS:
            self._intervals[name] = _IntervalSamples(
                name=name,
                samples=deque(maxlen=window_size),
            )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_tick_to_trade(self, stages: Dict[str, float]) -> None:
        """
        Record a complete tick-to-trade measurement.

        Parameters
        ----------
        stages
            Dict mapping stage names to timestamps (float, seconds since epoch
            or monotonic — must be consistent within a single record).
        """
        self._total_records += 1

        for interval_name, start_stage, end_stage in INTERVALS:
            start_ts = stages.get(start_stage)
            end_ts = stages.get(end_stage)
            if start_ts is not None and end_ts is not None:
                delta_ms = (end_ts - start_ts) * 1000.0
                self._intervals[interval_name].record(delta_ms)

        # Check threshold on total
        total_entry = self._intervals.get("total")
        if total_entry and total_entry.count > 0:
            last_total = total_entry.last
            if last_total > self._threshold_ms:
                self._threshold_breaches += 1
                logger.warning(
                    "TickToTrade: SLOW — total %.1fms > threshold %.1fms "
                    "(tick_to_signal=%.1fms, signal_to_risk=%.1fms, "
                    "risk_to_submit=%.1fms, submit_to_ack=%.1fms)",
                    last_total,
                    self._threshold_ms,
                    self._intervals.get("tick_to_signal", _IntervalSamples("")).last,
                    self._intervals.get("signal_to_risk", _IntervalSamples("")).last,
                    self._intervals.get("risk_to_submit", _IntervalSamples("")).last,
                    self._intervals.get("submit_to_ack", _IntervalSamples("")).last,
                )

    def record_stage(self, interval_name: str, duration_ms: float) -> None:
        """Record a single stage duration directly (convenience method)."""
        entry = self._intervals.get(interval_name)
        if entry is None:
            entry = _IntervalSamples(
                name=interval_name,
                samples=deque(maxlen=self._window_size),
            )
            self._intervals[interval_name] = entry
        entry.record(duration_ms)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return p50/p95/p99 for each interval and total.
        """
        result: Dict[str, Any] = {
            "total_records": self._total_records,
            "threshold_ms": self._threshold_ms,
            "threshold_breaches": self._threshold_breaches,
        }
        for name, entry in self._intervals.items():
            result[name] = {
                "p50_ms": entry.p50,
                "p95_ms": entry.p95,
                "p99_ms": entry.p99,
                "mean_ms": entry.mean,
                "last_ms": entry.last,
                "samples": entry.count,
            }
        return result

    def get_summary_line(self) -> str:
        """One-line summary for logging."""
        total = self._intervals.get("total")
        if total is None or total.count == 0:
            return "TickToTrade: no data"
        return (
            f"TickToTrade: p50={total.p50:.1f}ms p95={total.p95:.1f}ms "
            f"p99={total.p99:.1f}ms (n={total.count}, breaches={self._threshold_breaches})"
        )

    @property
    def threshold_ms(self) -> float:
        return self._threshold_ms

    @threshold_ms.setter
    def threshold_ms(self, value: float) -> None:
        self._threshold_ms = value
