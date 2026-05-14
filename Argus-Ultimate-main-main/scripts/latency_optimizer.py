"""Latency measurement and execution-route selection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
import statistics


@dataclass
class LatencyReport:
    route: str
    p50_ms: float
    p95_ms: float
    samples: int
    recommended_route: str


class LatencyOptimizer:
    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, route: str):
        start = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - start) * 1000
            self.samples.setdefault(route, []).append(elapsed_ms)

    def record(self, route: str, latency_ms: float) -> None:
        self.samples.setdefault(route, []).append(latency_ms)

    def recommend(self) -> str:
        if not self.samples:
            return "default"
        return min(self.samples, key=lambda route: statistics.median(self.samples[route]))

    def report(self, route: str) -> LatencyReport:
        values = sorted(self.samples.get(route, []))
        if not values:
            return LatencyReport(route, 0.0, 0.0, 0, self.recommend())
        p50 = statistics.median(values)
        p95 = values[min(int(len(values) * 0.95), len(values) - 1)]
        return LatencyReport(route, float(p50), float(p95), len(values), self.recommend())


def _demo() -> None:
    opt = LatencyOptimizer()
    for value in [4.1, 5.0, 4.4, 6.2]:
        opt.record("rest", value)
    for value in [1.1, 1.4, 1.2, 1.8]:
        opt.record("websocket", value)
    print("Latency optimizer ready")
    print(opt.report("websocket"))


if __name__ == "__main__":
    _demo()
