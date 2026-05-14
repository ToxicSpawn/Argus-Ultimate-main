from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RegimeResult:
    regime: str
    trend_state: str
    vol_state: str
    score: float


class DeterministicRegimeClassifier:
    """Simple, deterministic trend/range + volatility classifier."""

    def __init__(
        self,
        *,
        trend_threshold: float = 0.004,
        high_vol_pct: float = 2.0,
        low_vol_pct: float = 0.8,
    ) -> None:
        self.trend_threshold = float(trend_threshold)
        self.high_vol_pct = float(high_vol_pct)
        self.low_vol_pct = float(low_vol_pct)

    def classify(self, features: Dict[str, float]) -> RegimeResult:
        trend = float(features.get("trend_slope", 0.0) or 0.0)
        vol = float(features.get("volatility_pct", 0.0) or 0.0)

        if abs(trend) >= self.trend_threshold:
            trend_state = "trend_up" if trend > 0 else "trend_down"
        else:
            trend_state = "range"

        if vol >= self.high_vol_pct:
            vol_state = "high_vol"
        elif vol <= self.low_vol_pct:
            vol_state = "low_vol"
        else:
            vol_state = "mid_vol"

        regime = f"{trend_state}:{vol_state}"
        score = abs(trend) + vol / 100.0
        return RegimeResult(regime=regime, trend_state=trend_state, vol_state=vol_state, score=float(score))
