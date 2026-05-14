"""
SLA / Uptime Tracker -- records system availability and computes SLA metrics.

Tracks uptime ticks (heartbeats), downtime incidents, and provides:
  - Uptime percentage over a configurable window
  - Mean Time Between Failures (MTBF)
  - Mean Time To Recovery (MTTR)
  - Full SLA report

All data is kept in-memory (resets on process restart).  For persistent
tracking, use TimescaleDB via the storage_factory layer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DowntimeEvent:
    """Single downtime incident."""
    start_ts: float
    duration_seconds: float
    reason: str


class SLATracker:
    """Track system uptime and compute SLA metrics."""

    def __init__(self) -> None:
        self._uptime_ticks: List[float] = []  # timestamps of heartbeats
        self._downtime_events: List[DowntimeEvent] = []
        self._first_tick: Optional[float] = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_uptime_tick(self) -> None:
        """Called each heartbeat cycle to record the system is alive."""
        now = time.time()
        if self._first_tick is None:
            self._first_tick = now
        self._uptime_ticks.append(now)

    def record_downtime(self, duration_seconds: float, reason: str) -> None:
        """Log a downtime incident.

        Args:
            duration_seconds: How long the system was down.
            reason:           Human-readable explanation.
        """
        now = time.time()
        self._downtime_events.append(
            DowntimeEvent(
                start_ts=now - duration_seconds,
                duration_seconds=duration_seconds,
                reason=reason,
            )
        )
        logger.warning(
            "Downtime recorded: %.1fs -- %s", duration_seconds, reason,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_uptime_pct(self, hours: float = 24) -> float:
        """Return uptime percentage over the last *hours*.

        Uptime is ``1 - (total_downtime / window)`` clamped to ``[0, 100]``.
        If no data has been recorded, returns ``100.0`` (optimistic default).
        """
        if not self._uptime_ticks:
            return 100.0

        now = time.time()
        window_seconds = hours * 3600
        window_start = now - window_seconds

        total_downtime = sum(
            evt.duration_seconds
            for evt in self._downtime_events
            if evt.start_ts >= window_start
        )

        uptime_pct = max(0.0, min(100.0, (1.0 - total_downtime / window_seconds) * 100))
        return round(uptime_pct, 4)

    def get_mtbf(self) -> float:
        """Mean Time Between Failures in hours.

        MTBF = total_uptime / number_of_failures.
        Returns ``float('inf')`` if no failures recorded, ``0.0`` if no data.
        """
        if not self._uptime_ticks:
            return 0.0
        if not self._downtime_events:
            return float("inf")

        total_uptime_s = self._uptime_ticks[-1] - (self._first_tick or self._uptime_ticks[0])
        total_downtime_s = sum(e.duration_seconds for e in self._downtime_events)
        effective_uptime_s = max(0.0, total_uptime_s - total_downtime_s)

        n_failures = len(self._downtime_events)
        return (effective_uptime_s / n_failures) / 3600 if n_failures else float("inf")

    def get_mttr(self) -> float:
        """Mean Time To Recovery in hours.

        MTTR = average downtime duration across all incidents.
        Returns ``0.0`` if no failures recorded.
        """
        if not self._downtime_events:
            return 0.0
        total = sum(e.duration_seconds for e in self._downtime_events)
        return (total / len(self._downtime_events)) / 3600

    def get_report(self) -> Dict:
        """Full SLA report dictionary.

        Keys: ``uptime_pct_24h``, ``uptime_pct_7d``, ``mtbf_hours``,
        ``mttr_hours``, ``incident_count``, ``total_downtime_seconds``,
        ``total_ticks``.
        """
        return {
            "uptime_pct_24h": self.get_uptime_pct(24),
            "uptime_pct_7d": self.get_uptime_pct(168),
            "mtbf_hours": self.get_mtbf(),
            "mttr_hours": self.get_mttr(),
            "incident_count": len(self._downtime_events),
            "total_downtime_seconds": sum(
                e.duration_seconds for e in self._downtime_events
            ),
            "total_ticks": len(self._uptime_ticks),
        }
