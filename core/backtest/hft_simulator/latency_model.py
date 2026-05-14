"""Simple network and processing latency model."""
# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class LatencySample:
    outbound_us: float
    exchange_us: float
    processing_us: float
    inbound_us: float

    @property
    def total_us(self) -> float:
        return float(self.outbound_us + self.exchange_us + self.processing_us + self.inbound_us)


@dataclass
class LatencyModel:
    base_network_us: float = 150.0
    network_jitter_us: float = 35.0
    processing_us: float = 25.0
    exchange_matching_us: float = 20.0
    queue_penalty_us: float = 1.2
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample(self, queue_ahead: float = 0.0) -> LatencySample:
        outbound = self._one_way_network_us()
        inbound = self._one_way_network_us()
        processing = float(max(self.processing_us + self._rng.normal(0.0, self.processing_us * 0.15), 1.0))
        exchange = float(max(self.exchange_matching_us + np.log1p(max(queue_ahead, 0.0)) * self.queue_penalty_us, 1.0))
        return LatencySample(
            outbound_us=outbound,
            exchange_us=exchange,
            processing_us=processing,
            inbound_us=inbound,
        )

    def estimate_order_arrival_ns(self, submit_timestamp_ns: int, queue_ahead: float = 0.0) -> int:
        return int(submit_timestamp_ns + self.sample(queue_ahead).outbound_us * 1_000.0)

    def round_trip_us(self, queue_ahead: float = 0.0) -> float:
        return self.sample(queue_ahead).total_us

    def _one_way_network_us(self) -> float:
        return float(max(self.base_network_us + abs(self._rng.normal(0.0, self.network_jitter_us)), 1.0))
