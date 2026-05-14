'''
Position Sizing Strategies Module
21 different position sizing implementations
'''

# Re-export core classes from parent module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the parent module's position_sizing.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class SizingMethod(str, Enum):
    """Position sizing methods."""
    KELLY = "kelly"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    FIXED_FRACTIONAL = "fixed_fractional"
    DYNAMIC = "dynamic"


@dataclass
class SizingConfig:
    """Configuration for position sizing."""
    method: SizingMethod = SizingMethod.DYNAMIC
    kelly_fraction: float = 0.25
    min_win_rate: float = 0.40
    min_win_loss_ratio: float = 1.0
    fixed_risk_pct: float = 0.01
    target_risk_pct: float = 0.02
    atr_multiplier: float = 2.0
    max_position_pct: float = 0.10
    min_position_pct: float = 0.01
    max_position_value_aud: float = 25000.0
    regime_scaling: bool = True
    high_vol_scale: float = 0.5
    range_scale: float = 0.8
    trend_scale: float = 1.0
    use_confidence_scaling: bool = True
    min_confidence_scale: float = 0.5


class PositionSizer:
    """Full-featured position sizer with Kelly, volatility, regime and confidence support."""

    def __init__(self, config=None):
        self.config = config or SizingConfig()
        self._total_trades = 0
        self._winning_trades = 0
        self._total_win_pnl = 0.0
        self._total_loss_pnl = 0.0

    def record_trade(self, pnl: float):
        """Record a completed trade for Kelly history."""
        self._total_trades += 1
        if pnl > 0:
            self._winning_trades += 1
            self._total_win_pnl += pnl
        else:
            self._total_loss_pnl += abs(pnl)

    def _kelly_edge(self) -> float:
        """Compute Kelly fraction from trade history."""
        if self._total_trades < 5:
            return 0.0
        win_rate = self._winning_trades / self._total_trades
        if self._total_loss_pnl == 0:
            return 0.0
        avg_win = self._total_win_pnl / max(self._winning_trades, 1)
        avg_loss = self._total_loss_pnl / max(self._total_trades - self._winning_trades, 1)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        edge = win_rate - (1.0 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0.0
        return max(edge, 0.0)

    def calculate_position_size(self, capital: float, entry_price: float, stop_loss: float,
                                confidence: float = 1.0, regime=None, risk_level=None,
                                volatility: float = 0.0, **kwargs) -> "PositionSizeResult":
        from core.types import PositionSizeResult, MarketRegime, RiskLevel

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return PositionSizeResult(
                quantity=0, notional_aud=0, risk_amount=0,
                risk_pct=0, method="zero_stop", reasoning="Zero stop distance",
            )

        cfg = self.config
        method = cfg.method

        # --- Base sizing by method ---
        if method == SizingMethod.FIXED_FRACTIONAL:
            risk_pct = cfg.fixed_risk_pct
            risk_amount = capital * risk_pct
            quantity = risk_amount / stop_distance
            method_name = "fixed_fractional"

        elif method == SizingMethod.KELLY:
            kelly_edge = self._kelly_edge()
            risk_pct = kelly_edge * cfg.kelly_fraction
            risk_pct = min(risk_pct, cfg.max_position_pct)
            risk_amount = capital * risk_pct
            quantity = risk_amount / stop_distance
            method_name = "kelly"

        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            risk_pct = cfg.target_risk_pct
            risk_amount = capital * risk_pct
            if volatility > 0:
                effective_stop = max(volatility * cfg.atr_multiplier, stop_distance)
            else:
                effective_stop = stop_distance
            quantity = risk_amount / effective_stop
            method_name = "volatility_adjusted"

        else:  # DYNAMIC
            risk_pct = cfg.target_risk_pct
            risk_amount = capital * risk_pct
            quantity = risk_amount / stop_distance
            method_name = "dynamic"

        # --- Confidence scaling ---
        if cfg.use_confidence_scaling and confidence < 1.0:
            scale = cfg.min_confidence_scale + (1.0 - cfg.min_confidence_scale) * confidence
            quantity *= scale

        # --- Regime scaling ---
        if cfg.regime_scaling and regime is not None:
            regime_mult = 1.0
            if regime == MarketRegime.HIGH_VOL:
                regime_mult = cfg.high_vol_scale
            elif regime == MarketRegime.RANGE:
                regime_mult = cfg.range_scale
            elif regime in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN):
                regime_mult = cfg.trend_scale
            quantity *= regime_mult

        # --- Risk level scaling ---
        if risk_level is not None:
            risk_mult = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 0.75,
                RiskLevel.HIGH: 0.5,
                RiskLevel.CRITICAL: 0.25,
            }.get(risk_level, 1.0)
            quantity *= risk_mult

        # --- Position limits ---
        notional = quantity * entry_price
        max_notional = capital * cfg.max_position_pct
        if cfg.max_position_value_aud:
            max_notional = min(max_notional, cfg.max_position_value_aud)
        if notional > max_notional:
            quantity = max_notional / entry_price

        notional = quantity * entry_price
        actual_risk = quantity * stop_distance
        actual_risk_pct = actual_risk / capital if capital > 0 else 0.0

        return PositionSizeResult(
            quantity=quantity,
            notional_aud=notional,
            risk_amount=actual_risk,
            risk_pct=actual_risk_pct,
            method=method_name,
            reasoning=f"{method_name} sizing (conf={confidence:.2f})",
        )


# --- Convenience functions ---

def kelly_position_size(capital: float, win_rate: float, win_loss_ratio: float,
                        kelly_fraction: float = 0.25) -> float:
    """Calculate Kelly-optimal position size as a dollar amount."""
    if win_loss_ratio <= 0:
        return 0.0
    edge = win_rate - (1.0 - win_rate) / win_loss_ratio
    if edge <= 0:
        return 0.0
    return capital * edge * kelly_fraction


def volatility_adjusted_position_size(capital: float, risk_pct: float, entry_price: float,
                                      stop_price: float, volatility: float = 0.0,
                                      atr_multiplier: float = 2.0) -> float:
    """Calculate volatility-adjusted position quantity."""
    risk_amount = capital * risk_pct
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return 0.0
    if volatility > 0:
        effective_stop = max(volatility * atr_multiplier, stop_distance)
    else:
        effective_stop = stop_distance
    return risk_amount / effective_stop


# Also try to import library sizers (but don't fail if missing)
try:
    from .confidence_weighted import ConfidenceWeightedSizer
except Exception:
    ConfidenceWeightedSizer = None
try:
    from .kelly_criterion import KellyCriterionSizer
except Exception:
    KellyCriterionSizer = None
try:
    from .volatility_adjusted import VolatilityAdjustedSizer
except Exception:
    VolatilityAdjustedSizer = None


__all__ = [
    "PositionSizer",
    "SizingConfig",
    "SizingMethod",
    "ConfidenceWeightedSizer",
    "KellyCriterionSizer",
    "VolatilityAdjustedSizer",
    "kelly_position_size",
    "volatility_adjusted_position_size",
]
