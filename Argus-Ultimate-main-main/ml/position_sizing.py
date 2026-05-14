"""
Uncertainty-Aware Position Sizing — scales positions based on prediction uncertainty.

Features:
  - Kelly criterion with uncertainty adjustment
  - Confidence-weighted position sizing
  - Maximum position limits per confidence level
  - Integration with uncertainty quantifier outputs

Usage:
    sizing = UncertaintyPositionSizer(
        base_position_pct=0.10,
        min_position_pct=0.01,
        max_position_pct=0.20,
    )
    
    # Get position size adjusted by uncertainty
    position = sizing.compute(
        prediction_confidence=0.8,
        uncertainty=0.15,
        base_equity=10000,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingResult:
    """Result from position sizing calculation."""
    position_pct: float      # Position as fraction of equity [0, 1]
    position_usd: float      # Position size in USD
    confidence: float        # Adjusted confidence used
    uncertainty: float       # Uncertainty level used
    kelly_fraction: float    # Raw Kelly fraction before limits
    adjustment_factor: float # Multiplier applied due to uncertainty
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_pct": round(self.position_pct, 4),
            "position_usd": round(self.position_usd, 2),
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "adjustment_factor": round(self.adjustment_factor, 4),
        }


class UncertaintyPositionSizer:
    """
    Position sizer that accounts for prediction uncertainty.
    
    Uses modified Kelly criterion:
    - Base Kelly: f = (p * b - q) / b
    - Uncertainty adjustment: f_adj = f * (1 - uncertainty)
    - Confidence adjustment: f_adj *= confidence
    
    Args:
        base_position_pct: Base position size as fraction of equity
        min_position_pct: Minimum position size
        max_position_pct: Maximum position size
        kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
        uncertainty_scale: How much uncertainty reduces position (0-1)
    """
    
    def __init__(
        self,
        base_position_pct: float = 0.10,
        min_position_pct: float = 0.01,
        max_position_pct: float = 0.20,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        uncertainty_scale: float = 0.5,
    ):
        self.base_position_pct = base_position_pct
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.uncertainty_scale = uncertainty_scale
    
    def compute(
        self,
        prediction_confidence: float,
        uncertainty: float,
        base_equity: float,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> PositionSizingResult:
        """
        Compute position size adjusted for uncertainty.
        
        Args:
            prediction_confidence: Model confidence [0, 1]
            uncertainty: Prediction uncertainty [0, 1]
            base_equity: Current equity in USD
            win_rate: Optional historical win rate for Kelly
            win_loss_ratio: Optional win/loss ratio for Kelly
            
        Returns:
            PositionSizingResult with computed position sizes
        """
        # Clamp inputs
        confidence = np.clip(prediction_confidence, 0.0, 1.0)
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
        
        # Compute Kelly fraction if we have win rate data
        if win_rate is not None and win_loss_ratio is not None:
            kelly = self._compute_kelly(win_rate, win_loss_ratio)
            kelly *= self.kelly_fraction  # Half-Kelly or custom
        else:
            # Use base position as fallback
            kelly = self.base_position_pct
        
        # Uncertainty adjustment
        # Higher uncertainty = smaller position
        uncertainty_factor = 1.0 - (uncertainty * self.uncertainty_scale)
        
        # Confidence adjustment
        # Lower confidence = smaller position
        confidence_factor = confidence
        
        # Combined adjustment
        adjustment = uncertainty_factor * confidence_factor
        
        # Apply adjustment to Kelly
        adjusted_position = kelly * adjustment
        
        # Apply limits
        final_position = np.clip(
            adjusted_position,
            self.min_position_pct,
            self.max_position_pct,
        )
        
        # Compute USD position
        position_usd = final_position * base_equity
        
        return PositionSizingResult(
            position_pct=float(final_position),
            position_usd=float(position_usd),
            confidence=confidence,
            uncertainty=uncertainty,
            kelly_fraction=float(kelly),
            adjustment_factor=float(adjustment),
        )
    
    def compute_for_regime(
        self,
        regime: str,
        confidence: float,
        uncertainty: float,
        base_equity: float,
    ) -> PositionSizingResult:
        """
        Compute position size with regime-specific adjustments.
        
        Args:
            regime: Current market regime
            confidence: Prediction confidence
            uncertainty: Prediction uncertainty
            base_equity: Current equity
            
        Returns:
            PositionSizingResult with regime-adjusted position
        """
        # Regime-specific multipliers
        regime_multipliers = {
            "TREND_UP": 1.0,
            "TREND_DOWN": 0.8,
            "RANGING": 0.6,
            "VOLATILE": 0.4,
            "CRISIS": 0.2,
        }
        
        multiplier = regime_multipliers.get(regime, 0.5)
        
        # Compute base position
        result = self.compute(confidence, uncertainty, base_equity)
        
        # Apply regime multiplier
        adjusted_pct = result.position_pct * multiplier
        adjusted_pct = max(self.min_position_pct, min(adjusted_pct, self.max_position_pct))
        
        return PositionSizingResult(
            position_pct=adjusted_pct,
            position_usd=adjusted_pct * base_equity,
            confidence=confidence,
            uncertainty=uncertainty,
            kelly_fraction=result.kelly_fraction,
            adjustment_factor=result.adjustment_factor * multiplier,
        )
    
    def _compute_kelly(
        self,
        win_rate: float,
        win_loss_ratio: float,
    ) -> float:
        """
        Compute Kelly criterion fraction.
        
        f* = (p * b - q) / b
        where:
            p = win probability
            q = 1 - p
            b = win/loss ratio
        """
        p = np.clip(win_rate, 0.01, 0.99)
        q = 1.0 - p
        b = max(win_loss_ratio, 0.01)
        
        kelly = (p * b - q) / b
        
        # Kelly can be negative (don't bet)
        return max(0.0, kelly)


class ConfidenceScaler:
    """Scale various trading parameters by confidence."""
    
    @staticmethod
    def scale_position(base_position: float, confidence: float) -> float:
        """Scale position size by confidence."""
        return base_position * np.clip(confidence, 0.0, 1.0)
    
    @staticmethod
    def scale_stop_loss(base_sl: float, confidence: float) -> float:
        """Scale stop loss by confidence (lower confidence = tighter stop)."""
        # Higher confidence = wider stop (more room to breathe)
        # Lower confidence = tighter stop (cut losses faster)
        return base_sl * (0.5 + 0.5 * np.clip(confidence, 0.0, 1.0))
    
    @staticmethod
    def scale_take_profit(base_tp: float, confidence: float) -> float:
        """Scale take profit by confidence."""
        # Higher confidence = wider target
        return base_tp * (0.5 + 0.5 * np.clip(confidence, 0.0, 1.0))
    
    @staticmethod
    def compute_risk_per_trade(
        base_risk_pct: float,
        confidence: float,
        uncertainty: float,
    ) -> float:
        """
        Compute risk per trade adjusted for confidence and uncertainty.
        
        Returns:
            Risk as percentage of equity
        """
        # Lower confidence + higher uncertainty = lower risk
        adjustment = np.clip(confidence, 0.0, 1.0) * (1.0 - np.clip(uncertainty, 0.0, 1.0))
        return base_risk_pct * adjustment
