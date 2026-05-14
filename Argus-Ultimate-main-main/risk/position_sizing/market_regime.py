"""
Market Regime Position Sizing
Scales position size based on detected market regime.
"""
from __future__ import annotations

from typing import Any, Dict


class MarketRegimeSizer:
    """Regime-aware position sizing with per-regime scale factors."""

    REGIME_SCALES = {
        "trend_up": 1.2,
        "trend_down": 0.6,
        "range": 0.8,
        "high_vol": 0.4,
        "low_vol": 1.0,
        "crash": 0.1,
        "recovery": 0.7,
        "unknown": 0.5,
    }

    def __init__(
        self,
        regime_scales: Dict[str, float] = None,
        max_position_pct: float = 0.15,
        regime_transition_smoothing: float = 0.3,
    ):
        self.regime_scales = dict(self.REGIME_SCALES)
        if regime_scales:
            self.regime_scales.update(regime_scales)
        self.max_position_pct = float(max_position_pct)
        self.smoothing = float(regime_transition_smoothing)
        self._prev_scale: float = 1.0
        self._prev_regime: str = "unknown"

    def _get_regime_scale(self, regime: str) -> float:
        raw = self.regime_scales.get(regime, self.regime_scales.get("unknown", 0.5))
        if regime != self._prev_regime:
            smoothed = self._prev_scale * (1 - self.smoothing) + raw * self.smoothing
            self._prev_regime = regime
            self._prev_scale = smoothed
            return smoothed
        self._prev_scale = raw
        return raw

    def calculate(
        self,
        capital: float,
        risk_per_trade: float,
        confidence: float = 1.0,
        regime: str = "unknown",
        regime_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        cap = max(float(capital), 1.0)
        regime_scale = self._get_regime_scale(regime)
        regime_conf = max(0.0, min(float(regime_confidence), 1.0))
        effective_scale = 1.0 + (regime_scale - 1.0) * regime_conf
        effective_scale = max(effective_scale, 0.05)
        base_size = cap * float(risk_per_trade) * float(confidence)
        adjusted_size = base_size * effective_scale
        max_size = cap * self.max_position_pct
        adjusted_size = min(adjusted_size, max_size)
        return {
            "position_size": adjusted_size,
            "pct_of_capital": (adjusted_size / cap) * 100,
            "regime": regime,
            "regime_scale": regime_scale,
            "regime_confidence": regime_conf,
            "effective_scale": effective_scale,
            "method": "market_regime",
        }
