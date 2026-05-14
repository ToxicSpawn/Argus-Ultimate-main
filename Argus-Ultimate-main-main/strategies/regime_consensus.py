"""
RegimeAwareConsensus — extends SignalConsensus with per-regime strategy
weight multipliers so that strategy influence is dynamically scaled
depending on the detected market regime.

Regimes
-------
TREND_UP  : momentum ×1.5, mean_reversion ×0.5
RANGE     : mean_reversion ×1.5, breakout ×0.3
HIGH_VOL  : scalping disabled (weight ×0.0)
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional


class MarketRegime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOL = "HIGH_VOL"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Per-regime multiplier tables
# Keys are strategy *category* labels that calling code assigns to each
# strategy instance via the `category` attribute.
# ---------------------------------------------------------------------------
REGIME_MULTIPLIERS: Dict[MarketRegime, Dict[str, float]] = {
    MarketRegime.TREND_UP: {
        "momentum": 1.5,
        "mean_reversion": 0.5,
        "breakout": 1.2,
        "scalping": 1.0,
    },
    MarketRegime.TREND_DOWN: {
        "momentum": 1.2,
        "mean_reversion": 0.6,
        "breakout": 0.8,
        "scalping": 1.0,
    },
    MarketRegime.RANGE: {
        "momentum": 0.7,
        "mean_reversion": 1.5,
        "breakout": 0.3,
        "scalping": 1.1,
    },
    MarketRegime.HIGH_VOL: {
        "momentum": 0.9,
        "mean_reversion": 0.9,
        "breakout": 0.8,
        "scalping": 0.0,  # disabled
    },
    MarketRegime.UNKNOWN: {
        "momentum": 1.0,
        "mean_reversion": 1.0,
        "breakout": 1.0,
        "scalping": 1.0,
    },
}


class StrategySignal:
    """Lightweight container for a single strategy vote."""

    def __init__(
        self,
        name: str,
        signal: float,          # -1.0 … +1.0
        base_weight: float = 1.0,
        category: str = "momentum",
    ) -> None:
        self.name = name
        self.signal = signal
        self.base_weight = base_weight
        self.category = category


class SignalConsensus:
    """Base consensus: weighted average of strategy signals."""

    def __init__(self, strategies: Optional[List[StrategySignal]] = None) -> None:
        self.strategies: List[StrategySignal] = strategies or []

    def add_strategy(self, strategy: StrategySignal) -> None:
        self.strategies.append(strategy)

    def compute(self) -> float:
        """Return weighted-average signal in [-1.0, 1.0]."""
        if not self.strategies:
            return 0.0
        total_weight = sum(s.base_weight for s in self.strategies)
        if total_weight == 0.0:
            return 0.0
        weighted_sum = sum(s.signal * s.base_weight for s in self.strategies)
        return weighted_sum / total_weight


class RegimeAwareConsensus(SignalConsensus):
    """
    Extends SignalConsensus by applying per-regime multipliers to each
    strategy's base_weight before computing the consensus signal.
    """

    def __init__(
        self,
        strategies: Optional[List[StrategySignal]] = None,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        custom_multipliers: Optional[Dict[MarketRegime, Dict[str, float]]] = None,
    ) -> None:
        super().__init__(strategies)
        self.regime = regime
        self._multiplier_table = custom_multipliers or REGIME_MULTIPLIERS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_regime(self, regime: MarketRegime) -> None:
        self.regime = regime

    def effective_weight(self, strategy: StrategySignal) -> float:
        """Return base_weight scaled by the regime multiplier for its category."""
        table = self._multiplier_table.get(self.regime, {})
        multiplier = table.get(strategy.category, 1.0)
        return strategy.base_weight * multiplier

    def compute(self) -> float:  # type: ignore[override]
        """Regime-adjusted weighted consensus signal."""
        if not self.strategies:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for s in self.strategies:
            w = self.effective_weight(s)
            total_weight += w
            weighted_sum += s.signal * w

        if total_weight == 0.0:
            return 0.0

        raw = weighted_sum / total_weight
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, raw))

    def summary(self) -> Dict[str, object]:
        """Return a diagnostic dict for logging / monitoring."""
        return {
            "regime": self.regime.value,
            "consensus_signal": self.compute(),
            "strategies": [
                {
                    "name": s.name,
                    "signal": s.signal,
                    "base_weight": s.base_weight,
                    "effective_weight": self.effective_weight(s),
                    "category": s.category,
                }
                for s in self.strategies
            ],
        }
