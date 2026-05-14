"""Practical temporary and permanent market impact estimates."""
# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


@dataclass(slots=True)
class MarketImpactEstimate:
    temporary_bps: float
    permanent_bps: float

    @property
    def total_bps(self) -> float:
        return float(self.temporary_bps + self.permanent_bps)


@dataclass
class MarketImpactModel:
    temporary_coefficient: float = 18.0
    permanent_coefficient: float = 4.0
    participation_exponent: float = 0.6
    kyle_lambda: float = 0.0

    def calibrate_kyle_lambda(self, signed_volume: Iterable[float], mid_price_returns: Iterable[float]) -> float:
        volume = np.asarray(list(signed_volume), dtype=float)
        returns = np.asarray(list(mid_price_returns), dtype=float)
        if volume.size == 0 or returns.size == 0 or volume.size != returns.size:
            raise ValueError("signed_volume and mid_price_returns must be aligned and non-empty")
        denominator = float(np.dot(volume, volume))
        self.kyle_lambda = 0.0 if denominator <= 0 else float(np.dot(volume, returns) / denominator)
        return self.kyle_lambda

    def estimate_impact(
        self,
        order_quantity: float,
        visible_depth: float,
        daily_volume: float,
        volatility: float = 0.0,
    ) -> MarketImpactEstimate:
        participation = self._participation(order_quantity, visible_depth)
        adv_participation = self._participation(order_quantity, daily_volume)
        temporary = self.temporary_coefficient * (participation ** self.participation_exponent) * (1.0 + max(volatility, 0.0))
        permanent = self.permanent_coefficient * adv_participation + abs(self.kyle_lambda) * adv_participation * 10_000.0
        return MarketImpactEstimate(temporary_bps=float(temporary), permanent_bps=float(permanent))

    def apply_to_price(
        self,
        reference_price: float,
        side: str,
        order_quantity: float,
        visible_depth: float,
        daily_volume: float,
        volatility: float = 0.0,
    ) -> tuple[float, MarketImpactEstimate]:
        impact = self.estimate_impact(order_quantity, visible_depth, daily_volume, volatility)
        signed_bps = impact.total_bps if side == "buy" else -impact.total_bps
        impacted_price = reference_price * (1.0 + signed_bps / 10_000.0)
        return float(impacted_price), impact

    @staticmethod
    def _participation(order_quantity: float, reference_volume: float) -> float:
        if reference_volume <= 0:
            return 1.0
        return float(np.clip(abs(order_quantity) / reference_volume, 0.0, 25.0))
