"""
Portfolio Compounding v2.0
===========================
Automatic profit reinvestment and position scaling for Argus Ultimate.

Provides:
- Automatic profit compounding
- Position scaling on winners
- Risk-adjusted reinvestment
- Drawdown-aware compounding
- Tax-efficient profit taking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CompoundingAction(Enum):
    """Compounding actions."""
    REINVEST = "reinvest"
    WITHDRAW = "withdraw"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"


@dataclass
class TradeResult:
    """Result of a completed trade."""
    symbol: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    holding_time_hours: float
    timestamp: datetime
    strategy: str


@dataclass
class CompoundingRule:
    """Rule for profit compounding."""
    min_profit_pct: float  # Minimum profit to trigger compounding
    reinvest_pct: float  # Percentage of profit to reinvest
    max_position_pct: float  # Maximum position size as % of portfolio
    scale_up_threshold: float  # Profit % to trigger scale-up
    scale_up_multiplier: float  # Multiplier for scale-up
    drawdown_pause_threshold: float  # Pause compounding if drawdown > this


class CompoundingStrategy:
    """
    Base compounding strategy.
    """
    
    def __init__(
        self,
        base_reinvest_pct: float = 0.5,  # Reinvest 50% of profits
        max_reinvest_pct: float = 0.8,  # Max 80% reinvestment
        min_reinvest_amount: float = 10.0  # Min $10 to reinvest
    ) -> None:
        """
        Initialize compounding strategy.
        
        Args:
            base_reinvest_pct: Base percentage of profits to reinvest
            max_reinvest_pct: Maximum reinvestment percentage
            min_reinvest_amount: Minimum amount to reinvest
        """
        self.base_reinvest_pct = base_reinvest_pct
        self.max_reinvest_pct = max_reinvest_pct
        self.min_reinvest_amount = min_reinvest_amount
    
    def calculate_reinvestment(
        self,
        profit: float,
        portfolio_value: float,
        current_drawdown: float,
        winning_streak: int = 0
    ) -> Dict[str, float]:
        """
        Calculate reinvestment amount.
        
        Returns dict with reinvestment details.
        """
        if profit <= 0:
            return {
                "reinvest_amount": 0.0,
                "withdraw_amount": 0.0,
                "reinvest_pct": 0.0,
                "action": "no_profit"
            }
        
        # Adjust reinvestment based on performance
        adjusted_pct = self.base_reinvest_pct
        
        # Increase reinvestment during winning streaks
        if winning_streak >= 3:
            adjusted_pct = min(self.max_reinvest_pct, adjusted_pct * 1.2)
        
        # Decrease reinvestment during drawdowns
        if current_drawdown > 0.1:
            adjusted_pct = max(0.2, adjusted_pct * 0.7)
        
        # Calculate amounts
        reinvest_amount = profit * adjusted_pct
        withdraw_amount = profit - reinvest_amount
        
        # Check minimum
        if reinvest_amount < self.min_reinvest_amount:
            reinvest_amount = 0.0
            withdraw_amount = profit
        
        return {
            "reinvest_amount": reinvest_amount,
            "withdraw_amount": withdraw_amount,
            "reinvest_pct": adjusted_pct,
            "action": "reinvest" if reinvest_amount > 0 else "withdraw"
        }


class PositionScaler:
    """
    Scales positions based on performance.
    """
    
    def __init__(
        self,
        scale_up_threshold: float = 0.05,  # 5% profit
        scale_up_factor: float = 1.25,  # Increase size by 25%
        scale_down_threshold: float = -0.03,  # -3% loss
        scale_down_factor: float = 0.75,  # Decrease size by 25%
        max_scale: float = 3.0,  # Max 3x base size
        min_scale: float = 0.25  # Min 0.25x base size
    ) -> None:
        """
        Initialize position scaler.
        
        Args:
            scale_up_threshold: Profit threshold to scale up
            scale_up_factor: Multiplier when scaling up
            scale_down_threshold: Loss threshold to scale down
            scale_down_factor: Multiplier when scaling down
            max_scale: Maximum scale factor
            min_scale: Minimum scale factor
        """
        self.scale_up_threshold = scale_up_threshold
        self.scale_up_factor = scale_up_factor
        self.scale_down_threshold = scale_down_threshold
        self.scale_down_factor = scale_down_factor
        self.max_scale = max_scale
        self.min_scale = min_scale
        
        self._strategy_scales: Dict[str, float] = {}
        self._strategy_pnl: Dict[str, List[float]] = {}
    
    def update_strategy(self, strategy: str, pnl_pct: float) -> float:
        """
        Update strategy performance and return new scale factor.
        
        Args:
            strategy: Strategy name
            pnl_pct: Last trade PnL percentage
            
        Returns:
            New scale factor
        """
        if strategy not in self._strategy_scales:
            self._strategy_scales[strategy] = 1.0
            self._strategy_pnl[strategy] = []
        
        self._strategy_pnl[strategy].append(pnl_pct)
        
        # Keep last 50 trades
        if len(self._strategy_pnl[strategy]) > 50:
            self._strategy_pnl[strategy] = self._strategy_pnl[strategy][-50:]
        
        current_scale = self._strategy_scales[strategy]
        
        # Adjust scale based on recent performance
        if pnl_pct >= self.scale_up_threshold:
            new_scale = current_scale * self.scale_up_factor
        elif pnl_pct <= self.scale_down_threshold:
            new_scale = current_scale * self.scale_down_factor
        else:
            new_scale = current_scale
        
        # Apply limits
        new_scale = np.clip(new_scale, self.min_scale, self.max_scale)
        
        self._strategy_scales[strategy] = new_scale
        
        return new_scale
    
    def get_scale(self, strategy: str) -> float:
        """Get current scale for strategy."""
        return self._strategy_scales.get(strategy, 1.0)
    
    def get_strategy_stats(self, strategy: str) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        if strategy not in self._strategy_pnl:
            return {"status": "no_data"}
        
        pnls = self._strategy_pnl[strategy]
        
        if not pnls:
            return {"status": "no_data"}
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        return {
            "total_trades": len(pnls),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "total_pnl_pct": sum(pnls),
            "current_scale": self._strategy_scales.get(strategy, 1.0)
        }


class DrawdownTracker:
    """
    Tracks drawdown and pauses compounding during severe drawdowns.
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.20,  # 20% max drawdown
        pause_threshold: float = 0.15,  # Pause at 15% drawdown
        recovery_threshold: float = 0.05  # Resume at 5% drawdown
    ) -> None:
        """
        Initialize drawdown tracker.
        
        Args:
            max_drawdown: Maximum allowed drawdown
            pause_threshold: Drawdown threshold to pause compounding
            recovery_threshold: Drawdown threshold to resume
        """
        self.max_drawdown = max_drawdown
        self.pause_threshold = pause_threshold
        self.recovery_threshold = recovery_threshold
        
        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._is_paused: bool = False
        self._drawdown_history: List[float] = []
    
    def update(self, portfolio_value: float) -> Dict[str, Any]:
        """
        Update with current portfolio value.
        
        Returns drawdown status.
        """
        self._current_value = portfolio_value
        
        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            if self._is_paused and self._current_drawdown() < self.recovery_threshold:
                self._is_paused = False
                logger.info("Compounding resumed - drawdown recovered")
        
        # Calculate current drawdown
        current_dd = self._current_drawdown()
        self._drawdown_history.append(current_dd)
        
        # Check pause threshold
        if current_dd >= self.pause_threshold and not self._is_paused:
            self._is_paused = True
            logger.warning(
                "Compounding paused - drawdown %.1f%% exceeds threshold %.1f%%",
                current_dd * 100, self.pause_threshold * 100
            )
        
        # Check max drawdown (emergency)
        if current_dd >= self.max_drawdown:
            logger.critical(
                "MAX DRAWDOWN REACHED: %.1f%% - consider stopping trading",
                current_dd * 100
            )
        
        return {
            "peak_value": self._peak_value,
            "current_value": self._current_value,
            "current_drawdown": current_dd,
            "is_paused": self._is_paused,
            "max_drawdown_allowed": self.max_drawdown
        }
    
    def _current_drawdown(self) -> float:
        """Calculate current drawdown."""
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value
    
    @property
    def is_paused(self) -> bool:
        """Check if compounding is paused."""
        return self._is_paused
    
    @property
    def current_drawdown(self) -> float:
        """Get current drawdown."""
        return self._current_drawdown()


class PortfolioCompounding:
    """
    Main portfolio compounding system for Argus.
    
    Automatically reinvests profits and scales positions.
    """
    
    def __init__(
        self,
        initial_capital: float,
        reinvest_pct: float = 0.5,
        max_drawdown: float = 0.20,
        enable_auto_compound: bool = True
    ) -> None:
        """
        Initialize portfolio compounding.
        
        Args:
            initial_capital: Initial capital
            reinvest_pct: Percentage of profits to reinvest
            max_drawdown: Maximum drawdown before pausing
            enable_auto_compound: Whether to auto-compound
        """
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.enable_auto_compound = enable_auto_compound
        
        self.compounding_strategy = CompoundingStrategy(
            base_reinvest_pct=reinvest_pct
        )
        self.position_scaler = PositionScaler()
        self.drawdown_tracker = DrawdownTracker(max_drawdown=max_drawdown)
        
        self._trade_history: List[TradeResult] = []
        self._total_withdrawn: float = 0.0
        self._total_reinvested: float = 0.0
        self._winning_streak: int = 0
        self._losing_streak: int = 0
        
        logger.info(
            "PortfolioCompounding initialized: capital=$%.2f, reinvest=%.0f%%",
            initial_capital, reinvest_pct * 100
        )
    
    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        size: float,
        strategy: str = "default",
        holding_hours: float = 0.0
    ) -> Dict[str, Any]:
        """
        Record a completed trade and calculate compounding.
        
        Returns compounding decision.
        """
        # Calculate PnL
        pnl = (exit_price - entry_price) * size
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        
        # Create trade result
        trade = TradeResult(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_time_hours=holding_hours,
            timestamp=datetime.now(),
            strategy=strategy
        )
        
        self._trade_history.append(trade)
        
        # Update portfolio value
        self.current_value += pnl
        
        # Update drawdown tracker
        dd_status = self.drawdown_tracker.update(self.current_value)
        
        # Update winning/losing streak
        if pnl > 0:
            self._winning_streak += 1
            self._losing_streak = 0
        else:
            self._losing_streak += 1
            self._winning_streak = 0
        
        # Update position scaler
        new_scale = self.position_scaler.update_strategy(strategy, pnl_pct)
        
        # Calculate compounding
        compounding = {"action": "none", "reinvest_amount": 0.0}
        
        if self.enable_auto_compound and not self.drawdown_tracker.is_paused:
            compounding = self.compounding_strategy.calculate_reinvestment(
                profit=max(0, pnl),
                portfolio_value=self.current_value,
                current_drawdown=dd_status["current_drawdown"],
                winning_streak=self._winning_streak
            )
            
            if compounding["action"] == "reinvest":
                self._total_reinvested += compounding["reinvest_amount"]
                self.current_value += compounding["reinvest_amount"]
        
        return {
            "trade": {
                "symbol": symbol,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "strategy": strategy
            },
            "compounding": compounding,
            "position_scale": new_scale,
            "drawdown_status": dd_status,
            "winning_streak": self._winning_streak,
            "portfolio_value": self.current_value
        }
    
    def calculate_position_size(
        self,
        base_size: float,
        strategy: str,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate position size with scaling and compounding.
        
        Args:
            base_size: Base position size
            strategy: Strategy name
            volatility: Optional volatility for adjustment
            
        Returns:
            Adjusted position size
        """
        if not self.enable_auto_compound:
            return base_size
        
        # Get strategy scale
        scale = self.position_scaler.get_scale(strategy)
        
        # Adjust for drawdown
        dd = self.drawdown_tracker.current_drawdown
        if dd > 0.1:
            dd_multiplier = max(0.5, 1.0 - dd)
        else:
            dd_multiplier = 1.0
        
        # Adjust for volatility if provided
        vol_multiplier = 1.0
        if volatility is not None:
            # Reduce size in high volatility
            if volatility > 0.1:
                vol_multiplier = 0.7
            elif volatility > 0.05:
                vol_multiplier = 0.85
        
        # Calculate final size
        adjusted_size = base_size * scale * dd_multiplier * vol_multiplier
        
        # Cap at portfolio percentage
        max_size = self.current_value * 0.20  # Max 20% per position
        adjusted_size = min(adjusted_size, max_size)
        
        return adjusted_size
    
    def get_compounding_stats(self) -> Dict[str, Any]:
        """Get compounding statistics."""
        total_return = (self.current_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate winning/losing trades
        winning_trades = [t for t in self._trade_history if t.pnl > 0]
        losing_trades = [t for t in self._trade_history if t.pnl < 0]
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "initial_capital": self.initial_capital,
            "current_value": self.current_value,
            "total_return_pct": total_return,
            "total_withdrawn": self._total_withdrawn,
            "total_reinvested": self._total_reinvested,
            "total_trades": len(self._trade_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self._trade_history) * 100 if self._trade_history else 0,
            "profit_factor": profit_factor,
            "current_drawdown": self.drawdown_tracker.current_drawdown * 100,
            "compounding_paused": self.drawdown_tracker.is_paused,
            "winning_streak": self._winning_streak,
            "losing_streak": self._losing_streak
        }
    
    def withdraw(self, amount: float) -> bool:
        """
        Withdraw profits from portfolio.
        
        Returns True if successful.
        """
        # Can only withdraw profits, not initial capital
        available_profit = self.current_value - self.initial_capital
        
        if amount > available_profit:
            logger.warning(
                "Withdrawal exceeds available profit: $%.2f > $%.2f",
                amount, available_profit
            )
            return False
        
        self.current_value -= amount
        self._total_withdrawn += amount
        
        logger.info("Withdrawal: $%.2f, New value: $%.2f", amount, self.current_value)
        return True
    
    def get_growth_projection(
        self,
        avg_monthly_return: float = 0.05,
        months: int = 12
    ) -> Dict[str, float]:
        """
        Project portfolio growth with compounding.
        
        Args:
            avg_monthly_return: Expected monthly return (e.g., 0.05 = 5%)
            months: Number of months to project
            
        Returns:
            Projection details
        """
        values = [self.current_value]
        
        for m in range(1, months + 1):
            new_value = values[-1] * (1 + avg_monthly_return)
            values.append(new_value)
        
        return {
            "starting_value": self.current_value,
            "monthly_return_pct": avg_monthly_return * 100,
            "months": months,
            "projected_value": values[-1],
            "projected_profit": values[-1] - self.current_value,
            "projected_return_pct": (values[-1] - self.current_value) / self.current_value * 100,
            "monthly_values": values
        }
