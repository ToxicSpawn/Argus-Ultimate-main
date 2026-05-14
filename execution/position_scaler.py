"""
Position Scaling v2.0
=====================
Automatically scales positions based on performance in Argus Ultimate.

Provides:
- Winner scaling (add to winning positions)
- Loser scaling (reduce losing positions)
- Volatility-adjusted sizing
- Performance-based allocation
- Risk-parity scaling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ScalingMethod(Enum):
    """Position scaling methods."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    VOLATILITY_PARITY = "volatility_parity"
    RISK_PARITY = "risk_parity"
    MOMENTUM = "momentum"


@dataclass
class PositionPerformance:
    """Performance metrics for a position/strategy."""
    symbol: str
    entry_price: float
    current_price: float
    size: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    holding_time_hours: float
    max_favorable: float  # Max favorable excursion
    max_adverse: float  # Max adverse excursion
    entry_volatility: float
    current_volatility: float


@dataclass
class ScalingDecision:
    """Position scaling decision."""
    symbol: str
    action: str  # "scale_up", "scale_down", "hold", "close"
    current_size: float
    target_size: float
    size_change_pct: float
    reason: str
    confidence: float
    risk_score: float
    timestamp: datetime


class PerformanceAnalyzer:
    """
    Analyzes position performance for scaling decisions.
    """
    
    def __init__(self, lookback_trades: int = 50) -> None:
        """
        Initialize performance analyzer.
        
        Args:
            lookback_trades: Number of trades to analyze
        """
        self.lookback_trades = lookback_trades
        self._trade_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_trade(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        holding_hours: float,
        max_favorable: float,
        max_adverse: float
    ) -> None:
        """Record a completed trade."""
        if symbol not in self._trade_history:
            self._trade_history[symbol] = []
        
        self._trade_history[symbol].append({
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_hours": holding_hours,
            "max_favorable": max_favorable,
            "max_adverse": max_adverse,
            "timestamp": datetime.now()
        })
        
        # Keep only recent trades
        if len(self._trade_history[symbol]) > self.lookback_trades:
            self._trade_history[symbol] = self._trade_history[symbol][-self.lookback_trades:]
    
    def get_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance metrics for symbol."""
        if symbol not in self._trade_history:
            return {"status": "no_data"}
        
        trades = self._trade_history[symbol]
        if not trades:
            return {"status": "no_data"}
        
        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Calculate metrics
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        # Sharpe-like metric
        if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
            sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Consistency (lower std = more consistent)
        consistency = 1.0 / (1.0 + np.std(pnl_pcts)) if len(pnl_pcts) > 1 else 0.5
        
        return {
            "n_trades": len(trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "consistency": consistency,
            "total_pnl": sum(pnls),
            "recent_trend": self._calculate_trend(pnls[-10:] if len(pnls) >= 10 else pnls)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (-1 to 1)."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Normalize slope
        if np.std(values) > 0:
            return np.clip(slope / np.std(values), -1, 1)
        return 0.0


class KellyCalculator:
    """
    Kelly criterion calculator for optimal position sizing.
    """
    
    @staticmethod
    def kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25  # Fractional Kelly for safety
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Kelly = W - (1-W)/R
        Where W = win rate, R = avg_win/avg_loss
        
        Args:
            win_rate: Win rate (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
            fraction: Fraction of Kelly to use (0-1)
            
        Returns:
            Optimal fraction of capital to risk
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply fractional Kelly for safety
        fractional_kelly = kelly * fraction
        
        # Ensure non-negative and reasonable
        return max(0.0, min(0.25, fractional_kelly))


class VolatilityScaler:
    """
    Scales positions based on volatility.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annualized
        min_multiplier: float = 0.25,
        max_multiplier: float = 3.0
    ) -> None:
        """
        Initialize volatility scaler.
        
        Args:
            target_volatility: Target portfolio volatility
            min_multiplier: Minimum position multiplier
            max_multiplier: Maximum position multiplier
        """
        self.target_volatility = target_volatility
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
    
    def calculate_multiplier(
        self,
        current_volatility: float,
        asset_beta: float = 1.0
    ) -> float:
        """
        Calculate position size multiplier based on volatility.
        
        Lower volatility = larger positions
        Higher volatility = smaller positions
        """
        if current_volatility <= 0:
            return self.max_multiplier
        
        # Inverse volatility scaling
        vol_ratio = self.target_volatility / current_volatility
        
        # Adjust for beta
        beta_adjusted = vol_ratio / max(0.1, asset_beta)
        
        # Apply limits
        multiplier = np.clip(beta_adjusted, self.min_multiplier, self.max_multiplier)
        
        return float(multiplier)
    
    def calculate_volatility_target_size(
        self,
        base_size: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size for volatility targeting.
        """
        multiplier = self.calculate_multiplier(current_volatility)
        return base_size * multiplier


class PositionScaler:
    """
    Main position scaler for Argus.
    
    Automatically scales positions based on performance and conditions.
    """
    
    def __init__(
        self,
        base_risk_pct: float = 0.02,  # 2% risk per trade
        max_position_pct: float = 0.20,  # Max 20% per position
        scale_up_threshold: float = 0.05,  # 5% profit to scale up
        scale_down_threshold: float = -0.03,  # -3% loss to scale down
        max_scale: float = 2.0,  # Max 2x base size
        min_scale: float = 0.25  # Min 0.25x base size
    ) -> None:
        """
        Initialize position scaler.
        
        Args:
            base_risk_pct: Base risk percentage
            max_position_pct: Maximum position percentage
            scale_up_threshold: Profit threshold to scale up
            scale_down_threshold: Loss threshold to scale down
            max_scale: Maximum scale factor
            min_scale: Minimum scale factor
        """
        self.base_risk_pct = base_risk_pct
        self.max_position_pct = max_position_pct
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.max_scale = max_scale
        self.min_scale = min_scale
        
        self.performance_analyzer = PerformanceAnalyzer()
        self.kelly_calculator = KellyCalculator()
        self.volatility_scaler = VolatilityScaler()
        
        self._position_scales: Dict[str, float] = {}
        self._scaling_history: List[ScalingDecision] = []
    
    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        volatility: Optional[float] = None,
        use_kelly: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Trading symbol
            account_balance: Account balance
            entry_price: Entry price
            stop_loss_price: Stop loss price
            volatility: Current volatility
            use_kelly: Use Kelly criterion
            
        Returns:
            Position sizing details
        """
        # Base risk amount
        risk_amount = account_balance * self.base_risk_pct
        
        # Price risk per unit
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            price_risk = entry_price * 0.02  # Default 2% stop
        
        # Base position size
        base_size = risk_amount / price_risk
        
        # Get performance-based scaling
        perf = self.performance_analyzer.get_performance(symbol)
        perf_multiplier = 1.0
        
        if perf.get("status") != "no_data" and use_kelly:
            # Use Kelly for scaling
            kelly_frac = self.kelly_calculator.kelly_fraction(
                win_rate=perf.get("win_rate", 0.5),
                avg_win=abs(perf.get("avg_win", 1)),
                avg_loss=abs(perf.get("avg_loss", 1))
            )
            
            if kelly_frac > 0:
                perf_multiplier = kelly_frac / self.base_risk_pct
        
        # Volatility adjustment
        vol_multiplier = 1.0
        if volatility is not None:
            vol_multiplier = self.volatility_scaler.calculate_multiplier(volatility)
        
        # Get current scale
        current_scale = self._position_scales.get(symbol, 1.0)
        
        # Calculate final size
        adjusted_size = base_size * current_scale * perf_multiplier * vol_multiplier
        
        # Cap at max position percentage
        max_size = account_balance * self.max_position_pct / entry_price
        adjusted_size = min(adjusted_size, max_size)
        
        return {
            "symbol": symbol,
            "base_size": base_size,
            "current_scale": current_scale,
            "performance_multiplier": perf_multiplier,
            "volatility_multiplier": vol_multiplier,
            "final_size": adjusted_size,
            "position_value": adjusted_size * entry_price,
            "risk_amount": adjusted_size * price_risk,
            "risk_pct": (adjusted_size * price_risk) / account_balance * 100
        }
    
    def update_and_get_scaling(
        self,
        symbol: str,
        current_pnl_pct: float,
        current_volatility: float
    ) -> ScalingDecision:
        """
        Update performance and get scaling decision.
        
        Args:
            symbol: Trading symbol
            current_pnl_pct: Current unrealized PnL percentage
            current_volatility: Current volatility
            
        Returns:
            ScalingDecision
        """
        current_scale = self._position_scales.get(symbol, 1.0)
        
        # Determine action
        if current_pnl_pct >= self.scale_up_threshold * 100:
            # Scale up
            new_scale = min(current_scale * 1.25, self.max_scale)
            action = "scale_up"
            reason = f"Profit {current_pnl_pct:.1f}% exceeds threshold"
            confidence = min(1.0, current_pnl_pct / 10)
        
        elif current_pnl_pct <= self.scale_down_threshold * 100:
            # Scale down
            new_scale = max(current_scale * 0.75, self.min_scale)
            action = "scale_down"
            reason = f"Loss {current_pnl_pct:.1f}% exceeds threshold"
            confidence = min(1.0, abs(current_pnl_pct) / 10)
        
        elif current_volatility > 0.1:
            # High volatility - reduce
            new_scale = max(current_scale * 0.9, self.min_scale)
            action = "scale_down"
            reason = f"High volatility: {current_volatility:.1%}"
            confidence = 0.5
        
        else:
            # Hold
            new_scale = current_scale
            action = "hold"
            reason = "Within normal parameters"
            confidence = 0.5
        
        # Update scale
        self._position_scales[symbol] = new_scale
        
        decision = ScalingDecision(
            symbol=symbol,
            action=action,
            current_size=0.0,  # Would be filled in real implementation
            target_size=0.0,
            size_change_pct=(new_scale - current_scale) / current_scale * 100 if current_scale > 0 else 0,
            reason=reason,
            confidence=confidence,
            risk_score=1.0 - confidence,
            timestamp=datetime.now()
        )
        
        self._scaling_history.append(decision)
        
        return decision
    
    def get_scale(self, symbol: str) -> float:
        """Get current scale for symbol."""
        return self._position_scales.get(symbol, 1.0)
    
    def get_all_scales(self) -> Dict[str, float]:
        """Get all position scales."""
        return self._position_scales.copy()
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        if not self._scaling_history:
            return {"total_decisions": 0}
        
        recent = self._scaling_history[-100:]
        
        scale_ups = [d for d in recent if d.action == "scale_up"]
        scale_downs = [d for d in recent if d.action == "scale_down"]
        holds = [d for d in recent if d.action == "hold"]
        
        return {
            "total_decisions": len(self._scaling_history),
            "recent_decisions": len(recent),
            "scale_ups": len(scale_ups),
            "scale_downs": len(scale_downs),
            "holds": len(holds),
            "active_positions": len(self._position_scales),
            "avg_scale": np.mean(list(self._position_scales.values())) if self._position_scales else 1.0
        }
