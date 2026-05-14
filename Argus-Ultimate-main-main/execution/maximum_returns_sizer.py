"""
Maximum Returns Position Sizer
==============================
Optimizes position sizing for 70-150% monthly returns.

Uses:
- Kelly Criterion (half-Kelly for safety)
- Volatility-adjusted sizing
- Correlation-aware allocation
- Dynamic leverage
- Compounding optimization
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing result."""
    symbol: str
    size_usd: float
    size_pct: float
    leverage: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_amount: float
    expected_value: float
    kelly_fraction: float


class MaximumReturnsSizer:
    """
    Maximum Returns Position Sizer
    ===============================
    Aggressive position sizing for maximum returns.
    """
    
    def __init__(
        self,
        capital: float = 1000.0,
        max_leverage: float = 10.0,
        max_risk_per_trade: float = 0.05,  # 5% risk per trade
        max_portfolio_risk: float = 0.50,   # 50% total portfolio risk
        kelly_fraction: float = 0.5         # Half-Kelly
    ):
        self.capital = capital
        self.max_leverage = max_leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.kelly_fraction = kelly_fraction
        
        self.open_positions: Dict[str, PositionSize] = {}
        self.total_risk = 0.0
        
    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Kelly % = (bp - q) / b
        where:
            b = odds (avg_win / avg_loss)
            p = win probability
            q = loss probability (1 - p)
        """
        if avg_loss == 0 or avg_win == 0:
            return 0
        
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply Kelly fraction (half-Kelly for safety)
        kelly = kelly * self.kelly_fraction
        
        # Apply confidence multiplier
        kelly *= confidence
        
        # Clamp to reasonable range
        kelly = max(0, min(kelly, 0.25))  # Max 25% per trade
        
        return kelly
    
    def calculate_volatility_adjusted_size(
        self,
        symbol: str,
        volatility: float,
        target_risk: float = 0.02
    ) -> float:
        """
        Calculate position size adjusted for volatility.
        
        Higher volatility = smaller position
        """
        if volatility <= 0:
            return 0
        
        # Inverse volatility weighting
        vol_adjustment = target_risk / volatility
        
        # Cap adjustment
        vol_adjustment = min(vol_adjustment, 2.0)  # Max 2x
        
        position_pct = config.max_position_size_pct * vol_adjustment
        
        return min(position_pct, 0.30)  # Cap at 30%
    
    def calculate_optimal_leverage(
        self,
        volatility: float,
        sharpe_ratio: float,
        max_drawdown: float
    ) -> float:
        """Calculate optimal leverage based on Kelly and risk."""
        if volatility <= 0:
            return 1.0
        
        # Kelly optimal leverage
        kelly_leverage = sharpe_ratio / volatility if volatility > 0 else 1.0
        
        # Adjust for drawdown tolerance
        if max_drawdown > 0.3:
            kelly_leverage *= 0.5
        elif max_drawdown > 0.2:
            kelly_leverage *= 0.75
        
        # Apply limits
        kelly_leverage = max(1.0, min(kelly_leverage, self.max_leverage))
        
        return kelly_leverage
    
    def calculate_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        win_rate: float = 0.55,
        avg_win_pct: float = 0.03,
        avg_loss_pct: float = 0.015,
        volatility: float = 0.02,
        confidence: float = 0.6,
        sharpe: float = 2.0
    ) -> PositionSize:
        """Calculate optimal position size for maximum returns."""
        
        # Kelly position size
        kelly_pct = self.calculate_kelly_size(
            win_rate, avg_win_pct, avg_loss_pct, confidence
        )
        
        # Volatility adjustment
        vol_pct = self.calculate_volatility_adjusted_size(symbol, volatility)
        
        # Use the more conservative of Kelly and volatility
        position_pct = min(kelly_pct, vol_pct)
        
        # Ensure minimum position for meaningful returns
        position_pct = max(position_pct, 0.05)  # Minimum 5%
        
        # Calculate optimal leverage
        leverage = self.calculate_optimal_leverage(volatility, self.backtest_sharpe, 0.15)
        
        # Position value with leverage
        position_value = self.capital * position_pct * leverage
        
        # Risk calculation
        stop_distance = abs(entry_price - stop_loss) / entry_price
        risk_amount = position_value * stop_distance
        
        # Expected value
        win_prob = win_rate
        lose_prob = 1 - win_prob
        expected_value = (win_prob * avg_win_pct - lose_prob * avg_loss_pct) * position_value
        
        return PositionSize(
            symbol=symbol,
            size_usd=position_value,
            size_pct=position_pct,
            leverage=leverage,
            stop_loss_pct=stop_distance * 100,
            take_profit_pct=abs(take_profit - entry_price) / entry_price * 100,
            risk_amount=risk_amount,
            expected_value=expected_value,
            kelly_fraction=kelly_pct
        )
    
    def calculate_compounding_schedule(
        self,
        initial_capital: float,
        monthly_target: float = 1.0,  # 100% monthly
        days: int = 30
    ) -> List[Dict[str, float]]:
        """Calculate compounding schedule to hit target."""
        daily_target = (1 + monthly_target) ** (1/days) - 1
        
        schedule = []
        capital = initial_capital
        
        for day in range(days):
            daily_profit = capital * daily_target
            reinvest = daily_profit * 0.8  # Reinvest 80%
            withdrawal = daily_profit * 0.2  # Withdraw 20%
            
            schedule.append({
                "day": day + 1,
                "start_capital": capital,
                "target_profit": daily_profit,
                "reinvest": reinvest,
                "withdrawal": withdrawal,
                "end_capital": capital + reinvest
            })
            
            capital += reinvest
        
        return schedule


class DynamicLeverageManager:
    """
    Dynamic Leverage Manager
    ========================
    Adjusts leverage based on market conditions.
    """
    
    def __init__(
        self,
        max_leverage: float = 10.0,
        base_leverage: float = 3.0
    ):
        self.max_leverage = max_leverage
        self.base_leverage = base_leverage
        
    def calculate_leverage(
        self,
        volatility: float,
        trend_strength: float,
        win_rate_recent: float,
        drawdown: float
    ) -> float:
        """Calculate dynamic leverage."""
        leverage = self.base_leverage
        
        # Increase leverage in trending markets
        if abs(trend_strength) > 0.5:
            leverage *= 1.2
        
        # Increase leverage when winning
        if win_rate_recent > 0.6:
            leverage *= 1.15
        
        # Decrease leverage in high volatility
        if volatility > 0.03:
            leverage *= 0.7
        elif volatility > 0.05:
            leverage *= 0.5
        
        # Decrease leverage during drawdown
        if drawdown > 0.15:
            leverage *= 0.6
        elif drawdown > 0.10:
            leverage *= 0.8
        
        # Apply limits
        leverage = max(1.0, min(leverage, self.max_leverage))
        
        return leverage


class ProfitOptimizer:
    """
    Profit Optimizer
    ================
    Optimizes for maximum risk-adjusted returns.
    """
    
    def __init__(self, target_monthly: float = 1.0):
        self.target_monthly = target_monthly
        self.daily_target = (1 + target_monthly) ** (1/30) - 1
        
    def calculate_required_trades(
        self,
        avg_trade_size_pct: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float
    ) -> int:
        """Calculate number of trades needed to hit target."""
        expected_value = (win_rate * avg_win_pct - (1 - win_rate) * abs(avg_loss_pct))
        
        if expected_value <= 0:
            return float('inf')
        
        daily_trades_needed = self.daily_target / (expected_value * avg_trade_size_pct)
        
        return int(np.ceil(daily_trades_needed))
    
    def optimize_strategy_mix(
        self,
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Optimize allocation across strategies."""
        # Sort by Sharpe ratio
        sorted_strats = sorted(strategies, key=lambda s: s.get("sharpe", 0), reverse=True)
        
        # Allocate based on Sharpe
        total_sharpe = sum(s.get("sharpe", 1) for s in sorted_strats)
        
        allocations = {}
        for strat in sorted_strats:
            name = strat["name"]
            sharpe = strat.get("sharpe", 1)
            allocations[name] = sharpe / total_sharpe
        
        return allocations


# Export
__all__ = [
    "MaximumReturnsSizer",
    "DynamicLeverageManager",
    "ProfitOptimizer",
    "PositionSize"
]
