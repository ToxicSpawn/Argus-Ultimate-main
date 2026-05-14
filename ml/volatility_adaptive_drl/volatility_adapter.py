"""Risk adaptation utilities for volatility-aware control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class VolatilityAdapterConfig:
    base_risk_aversion: float = 1.0
    volatility_floor: float = 1e-4
    target_volatility: float = 0.02
    min_position_scale: float = 0.1
    max_position_scale: float = 1.0
    crisis_scale_multiplier: float = 0.35
    regime_risk_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.4,
            "crisis": 2.0,
        }
    )


class VolatilityAdapter:
    def __init__(self, config: VolatilityAdapterConfig | None = None) -> None:
        self.config = config or VolatilityAdapterConfig()

    def risk_aversion(self, volatility: float, regime: str) -> float:
        vol = max(float(volatility), self.config.volatility_floor)
        multiplier = self.config.regime_risk_multipliers.get(str(regime), 1.0)
        return float(self.config.base_risk_aversion * multiplier * (vol / self.config.target_volatility))

    def position_scale(self, volatility: float, regime: str) -> float:
        vol = max(float(volatility), self.config.volatility_floor)
        inverse_scale = self.config.target_volatility / vol
        scale = float(np.clip(inverse_scale, self.config.min_position_scale, self.config.max_position_scale))
        if str(regime) == "crisis":
            scale *= self.config.crisis_scale_multiplier
        return float(np.clip(scale, self.config.min_position_scale, self.config.max_position_scale))

    def adapt_position(self, action: np.ndarray | float, volatility: float, regime: str) -> np.ndarray:
        raw_action = np.asarray(action, dtype=np.float32)
        scaled = raw_action * self.position_scale(volatility=volatility, regime=regime)
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)

    def shape_reward(self, reward: float, volatility: float, turnover: float, regime: str) -> float:
        penalty = self.risk_aversion(volatility=volatility, regime=regime) * float(turnover) * 0.01
        return float(reward - penalty)
