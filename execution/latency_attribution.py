"""
Latency monitoring and attribution: data -> signal -> order -> fill.

Per-component latency and P99 for observability and tuning.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LatencySpan:
    name: str
    start_ts: float
    end_ts: float
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000.0


class LatencyTracker:
    """Track spans per component; compute p50/p99 and expose for Grafana."""

    def __init__(self, max_samples: int = 10_000) -> None:
        self.max_samples = max_samples
        self._spans: Dict[str, deque] = {}
        self._current: Dict[str, float] = {}

    def start(self, component: str) -> None:
        self._current[component] = time.perf_counter()

    def end(self, component: str, extra: Optional[Dict[str, Any]] = None) -> Optional[LatencySpan]:
        if component not in self._current:
            return None
        start = self._current.pop(component, None)
        if start is None:
            return None
        end = time.perf_counter()
        span = LatencySpan(component, start, end, extra or {})
        if component not in self._spans:
            self._spans[component] = deque(maxlen=self.max_samples)
        self._spans[component].append(span.duration_ms)
        return span

    def get_stats(self, component: str) -> Dict[str, float]:
        """Return count, p50, p95, p99 for component."""
        if component not in self._spans or not self._spans[component]:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        arr = sorted(self._spans[component])
        n = len(arr)
        return {
            "count": n,
            "p50_ms": arr[int(n * 0.50)] if n else 0.0,
            "p95_ms": arr[int(n * 0.95)] if n else 0.0,
            "p99_ms": arr[int(n * 0.99)] if n else 0.0,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        return {k: self.get_stats(k) for k in list(self._spans.keys())}


# Global tracker for process
_global_tracker: Optional[LatencyTracker] = None


def get_latency_tracker() -> LatencyTracker:
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LatencyTracker()
    return _global_tracker
