from __future__ import annotations

from collections import Counter


class MetricsBridge:
    def __init__(self) -> None:
        self.counters = Counter()

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value
