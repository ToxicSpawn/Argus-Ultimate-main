"""
RISK ADAPTATION SYSTEM
======================
Dynamically adjusts risk based on market conditions.

Key Principles:
1. Protect capital in bad conditions
2. Maximize returns in good conditions
3. Never fight the market
4. Adapt faster than the market changes
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Dynamic risk limits."""
    max_position_pct: float = 25.0      # Max % of capital per position
    max_daily_loss_pct: float = 10.0    # Max daily loss %
    max_drawdown_pct: float = 20.0      # Max total drawdown %
    max_correlation: float = 0.7        # Max portfolio correlation
    max_leverage: float = 2.0           # Max leverage
    stop_loss_pct: float = 2.0          # Default stop loss %
    take_profit_pct: float = 4.0        # Default take profit %
    max_open_positions: int = 10        # Max simultaneous positions
    min_liquidity_score: float = 0.3    # Min liquidity to trade


@dataclass
class RiskState:
    """Current risk state."""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0
    portfolio_var: float = 0.0
    portfolio_beta: float = 1.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    avg_win_loss_ratio: float = 1.0
    consecutive_losses: int = 0
    last_update: float = field(default_factory=time.time)


class RiskAdaptationSystem:
    """
    Adapts risk parameters based on market conditions and performance.
    
    Adapts:
    - Position sizes
    - Stop losses
    - Take profits
    - Leverage limits
    - Position counts
    """
    
    def __init__(self, base_capital: float = 10000):
        self.base_capital = base_capital
        self.current_capital = base_capital
        self.peak_capital = base_capital
        
        self.base_limits = RiskLimits()
        self.current_limits = RiskLimits()
        self.risk_state = RiskState()
        
        # Performance tracking
        self.trade_history: deque = deque(maxlen=1000)
        self.daily_pnl_history: deque = deque(maxlen=30)
        
        # Adaptation factors
        self.volatility_factor = 1.0
        self.performance_factor = 1.0
        self.drawdown_factor = 1.0
        
        logger.info(f"RiskAdaptationSystem initialized: capital=${base_capital}")
    
    async def update_risk_state(
        self,
        positions: Dict[str, Dict],
        daily_pnl: float,
        portfolio_value: float,
    ) -> RiskState:
        """Update risk state from current positions."""
        self.current_capital = portfolio_value
        self.peak_capital = max(self.peak_capital, portfolio_value)
        
        # Calculate drawdown
        current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
        
        # Update state
        self.risk_state = RiskState(
            current_drawdown=current_drawdown,
            daily_pnl=daily_pnl,
            open_positions=len(positions),
            last_update=time.time(),
        )
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        return self.risk_state
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from trade history."""
        if len(self.trade_history) < 10:
            return
        
        trades = list(self.trade_history)
        profits = [t.get("pnl", 0) for t in trades]
        
        # Win rate
        wins = sum(1 for p in profits if p > 0)
        self.risk_state.win_rate = wins / len(profits) if profits else 0.5
        
        # Avg win/loss ratio
        win_profits = [p for p in profits if p > 0]
        loss_profits = [p for p in profits if p < 0]
        
        avg_win = np.mean(win_profits) if win_profits else 0
        avg_loss = abs(np.mean(loss_profits)) if loss_profits else 1
        
        self.risk_state.avg_win_loss_ratio = avg_win / (avg_loss + 1e-10)
        
        # Sharpe ratio (simplified)
        if len(profits) > 1:
            returns = np.array(profits) / self.base_capital
            self.risk_state.sharpe_ratio = float(
                np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            )
        
        # Consecutive losses
        consecutive = 0
        for p in reversed(profits):
            if p < 0:
                consecutive += 1
            else:
                break
        self.risk_state.consecutive_losses = consecutive
    
    async def adapt_to_market(
        self,
        market_condition: str,
        volatility: float,
        trend_strength: float,
    ) -> RiskLimits:
        """Adapt risk limits to market conditions."""
        # Start with base limits
        self.current_limits = RiskLimits(
            max_position_pct=self.base_limits.max_position_pct,
            max_daily_loss_pct=self.base_limits.max_daily_loss_pct,
            max_drawdown_pct=self.base_limits.max_drawdown_pct,
            max_correlation=self.base_limits.max_correlation,
            max_leverage=self.base_limits.max_leverage,
            stop_loss_pct=self.base_limits.stop_loss_pct,
            take_profit_pct=self.base_limits.take_profit_pct,
            max_open_positions=self.base_limits.max_open_positions,
            min_liquidity_score=self.base_limits.min_liquidity_score,
        )
        
        # Adjust based on market condition
        condition_multipliers = {
            "bull_strong": {"position": 1.2, "risk": 1.0, "stop": 1.0},
            "bull_weak": {"position": 0.8, "risk": 0.8, "stop": 1.2},
            "sideways": {"position": 0.6, "risk": 0.7, "stop": 0.8},
            "bear_weak": {"position": 0.5, "risk": 0.6, "stop": 1.3},
            "bear_strong": {"position": 0.3, "risk": 0.4, "stop": 1.5},
            "high_vol": {"position": 0.4, "risk": 0.5, "stop": 1.5},
            "low_liq": {"position": 0.3, "risk": 0.4, "stop": 1.2},
            "crash": {"position": 0.1, "risk": 0.2, "stop": 2.0},
            "pump": {"position": 0.6, "risk": 0.6, "stop": 1.0},
        }
        
        multipliers = condition_multipliers.get(market_condition, {"position": 0.5, "risk": 0.5, "stop": 1.0})
        
        # Apply condition multipliers
        self.current_limits.max_position_pct *= multipliers["position"]
        self.current_limits.max_daily_loss_pct *= multipliers["risk"]
        self.current_limits.stop_loss_pct *= multipliers["stop"]
        self.current_limits.take_profit_pct *= multipliers["stop"]
        
        # Adjust for volatility
        vol_factor = 1.0 + volatility * 5
        self.current_limits.stop_loss_pct *= vol_factor
        self.current_limits.max_position_pct /= vol_factor
        
        # Adjust for performance
        if self.risk_state.consecutive_losses >= 3:
            # Reduce risk after consecutive losses
            loss_factor = 0.7 ** (self.risk_state.consecutive_losses - 2)
            self.current_limits.max_position_pct *= loss_factor
            self.current_limits.max_daily_loss_pct *= loss_factor
        
        if self.risk_state.win_rate < 0.4:
            # Reduce risk with poor win rate
            self.current_limits.max_position_pct *= 0.7
            self.current_limits.max_open_positions = max(3, self.current_limits.max_open_positions - 2)
        
        if self.risk_state.sharpe_ratio < 0:
            # Negative Sharpe = reduce risk
            self.current_limits.max_position_pct *= 0.6
        
        # Adjust for drawdown
        if self.risk_state.current_drawdown > 0.1:
            # 10%+ drawdown - reduce risk
            drawdown_factor = max(0.3, 1.0 - self.risk_state.current_drawdown)
            self.current_limits.max_position_pct *= drawdown_factor
            self.current_limits.max_daily_loss_pct *= drawdown_factor
        
        # Ensure minimum limits
        self.current_limits.max_position_pct = max(1.0, self.current_limits.max_position_pct)
        self.current_limits.max_daily_loss_pct = max(1.0, self.current_limits.max_daily_loss_pct)
        self.current_limits.stop_loss_pct = max(0.5, self.current_limits.stop_loss_pct)
        
        logger.debug(
            f"Risk adapted to {market_condition}: "
            f"pos={self.current_limits.max_position_pct:.1f}%, "
            f"sl={self.current_limits.stop_loss_pct:.1f}%, "
            f"tp={self.current_limits.take_profit_pct:.1f}%"
        )
        
        return self.current_limits
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Calculate position size based on risk limits."""
        # Risk per trade (fixed fraction of capital)
        risk_per_trade = self.current_capital * (self.current_limits.max_position_pct / 100)
        
        # Adjust for confidence
        risk_per_trade *= confidence
        
        # Calculate position size from stop loss
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            return {"error": "Stop loss equals entry price"}
        
        position_size = risk_per_trade / risk_per_unit
        
        # Calculate notional value
        notional = position_size * entry_price
        
        # Check against limits
        notional_pct = (notional / self.current_capital) * 100
        
        if notional_pct > self.current_limits.max_position_pct:
            # Reduce size to fit limits
            position_size = (self.current_capital * self.current_limits.max_position_pct / 100) / entry_price
            notional = position_size * entry_price
        
        return {
            "symbol": symbol,
            "position_size": position_size,
            "notional": notional,
            "notional_pct": notional_pct,
            "risk_amount": risk_per_unit * position_size,
            "risk_pct": (risk_per_unit * position_size / self.current_capital) * 100,
        }
    
    def should_stop_trading(self) -> Tuple[bool, str]:
        """Determine if we should stop trading."""
        # Daily loss limit
        daily_loss_pct = abs(self.risk_state.daily_pnl) / self.base_capital * 100
        if self.risk_state.daily_pnl < 0 and daily_loss_pct >= self.current_limits.max_daily_loss_pct:
            return True, f"Daily loss limit reached: {daily_loss_pct:.1f}%"
        
        # Max drawdown
        if self.risk_state.current_drawdown >= self.current_limits.max_drawdown_pct / 100:
            return True, f"Max drawdown reached: {self.risk_state.current_drawdown*100:.1f}%"
        
        # Too many consecutive losses
        if self.risk_state.consecutive_losses >= 5:
            return True, f"Too many consecutive losses: {self.risk_state.consecutive_losses}"
        
        # Capital below threshold
        if self.current_capital < self.base_capital * 0.5:
            return True, f"Capital below 50%: ${self.current_capital:.2f}"
        
        return False, "OK"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary."""
        return {
            "capital": {
                "current": self.current_capital,
                "peak": self.peak_capital,
                "base": self.base_capital,
            },
            "state": {
                "drawdown": self.risk_state.current_drawdown,
                "daily_pnl": self.risk_state.daily_pnl,
                "open_positions": self.risk_state.open_positions,
                "win_rate": self.risk_state.win_rate,
                "sharpe": self.risk_state.sharpe_ratio,
                "consecutive_losses": self.risk_state.consecutive_losses,
            },
            "limits": {
                "max_position_pct": self.current_limits.max_position_pct,
                "max_daily_loss_pct": self.current_limits.max_daily_loss_pct,
                "stop_loss_pct": self.current_limits.stop_loss_pct,
                "take_profit_pct": self.current_limits.take_profit_pct,
                "max_open_positions": self.current_limits.max_open_positions,
            },
            "should_stop": self.should_stop_trading()[0],
            "stop_reason": self.should_stop_trading()[1],
        }
    
    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade."""
        self.trade_history.append(trade)
        
        if trade.get("pnl", 0) < 0:
            logger.debug(f"Loss trade recorded: ${trade.get('pnl', 0):.2f}")
        else:
            logger.debug(f"Win trade recorded: ${trade.get('pnl', 0):.2f}")


def get_risk_adaptation(capital: float = 10000) -> RiskAdaptationSystem:
    """Get risk adaptation system instance."""
    return RiskAdaptationSystem(base_capital=capital)
