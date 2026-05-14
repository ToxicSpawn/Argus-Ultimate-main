"""Venue latency monitoring with spike detection."""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencySnapshot:
    venue: str
    avg_ms: float
    p95_ms: float
    baseline_ms: float
    is_elevated: bool
    multiplier: float
    reason: str


class VenueLatencyMonitor:
    """Tracks per-venue latency and detects spikes relative to baseline."""

    def __init__(self, spike_multiplier: float = 2.0, baseline_window: int = 100) -> None:
        self._spike_multiplier = spike_multiplier
        self._baseline_window = baseline_window
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=baseline_window)
        )

    def record_latency(self, venue: str, latency_ms: float) -> None:
        """Record a latency measurement for a venue."""
        self._history[venue].append(latency_ms)

    def check(self, venue: str) -> LatencySnapshot:
        """Compute latency snapshot for a venue."""
        buf = self._history.get(venue)
        if not buf:
            return LatencySnapshot(
                venue=venue,
                avg_ms=0.0,
                p95_ms=0.0,
                baseline_ms=0.0,
                is_elevated=False,
                multiplier=0.0,
                reason="no data",
            )
        values = list(buf)
        n = len(values)
        avg = sum(values) / n
        sorted_vals = sorted(values)
        p95_idx = min(int(n * 0.95), n - 1)
        p95 = sorted_vals[p95_idx]
        baseline = sum(sorted_vals) / n  # baseline = mean over full window
        mult = avg / baseline if baseline > 0 else 0.0
        elevated = avg > baseline * self._spike_multiplier
        reason = (
            f"avg={avg:.1f}ms > baseline={baseline:.1f}ms x {self._spike_multiplier}"
            if elevated
            else "ok"
        )
        if elevated:
            logger.warning(
                "Latency elevated for %s: avg=%.1fms, baseline=%.1fms, multiplier=%.2f",
                venue, avg, baseline, mult,
            )
        return LatencySnapshot(
            venue=venue,
            avg_ms=round(avg, 2),
            p95_ms=round(p95, 2),
            baseline_ms=round(baseline, 2),
            is_elevated=elevated,
            multiplier=round(mult, 2),
            reason=reason,
        )

    def is_elevated(self, venue: str) -> bool:
        """Return True if current latency exceeds baseline x multiplier."""
        return self.check(venue).is_elevated

    def get_recommended_venue(self, venues: List[str]) -> str:
        """Return the venue with the lowest average latency."""
        best_venue = venues[0] if venues else ""
        best_avg = float("inf")
        for v in venues:
            buf = self._history.get(v)
            if not buf:
                continue
            avg = sum(buf) / len(buf)
            if avg < best_avg:
                best_avg = avg
                best_venue = v
        return best_venue

    def get_stats(self) -> Dict:
        """Return stats for all tracked venues."""
        stats: Dict[str, object] = {}
        for venue in self._history:
            snap = self.check(venue)
            stats[venue] = {
                "avg_ms": snap.avg_ms,
                "p95_ms": snap.p95_ms,
                "baseline_ms": snap.baseline_ms,
                "is_elevated": snap.is_elevated,
                "multiplier": snap.multiplier,
            }
        return stats
