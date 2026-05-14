"""
Learning Risk Manager
=====================
Risk system that learns and adapts constantly with the market.

Key Features:
- Adaptive position sizing based on volatility and recent performance
- Learning stop loss optimizer
- Dynamic drawdown limits based on market regime
- Risk parameters learned from every trade
- Real-time risk adjustment based on market conditions
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskLearningConfig:
    """Configuration for risk learning."""
    # Position sizing bounds
    min_position_pct: float = 0.01  # 1% minimum
    max_position_pct: float = 0.25  # 25% maximum
    base_position_pct: float = 0.10  # 10% base
    
    # Stop loss bounds
    min_stop_loss_pct: float = 0.005  # 0.5%
    max_stop_loss_pct: float = 0.05   # 5%
    base_stop_loss_pct: float = 0.02  # 2%
    
    # Drawdown limits
    min_drawdown_limit: float = 0.05   # 5%
    max_drawdown_limit: float = 0.25   # 25%
    base_drawdown_limit: float = 0.15  # 15%
    
    # Risk per trade bounds
    min_risk_per_trade: float = 0.005  # 0.5%
    max_risk_per_trade: float = 0.03   # 3%
    base_risk_per_trade: float = 0.01  # 1%
    
    # Learning rates
    position_learning_rate: float = 0.1
    stop_loss_learning_rate: float = 0.05
    drawdown_learning_rate: float = 0.02
    
    # Windows for learning
    performance_window: int = 100
    volatility_window: int = 50


@dataclass
class TradeOutcome:
    """Record of a trade outcome for learning."""
    timestamp: float
    entry_price: float
    exit_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_pct: float
    regime: str
    volatility: float
    duration_seconds: float
    was_stopped_out: bool
    was_take_profit: bool


class VolatilityRiskScaler:
    """Scales risk parameters based on current volatility."""
    
    def __init__(self, window: int = 50):
        self.window = window
        self.returns: Deque[float] = deque(maxlen=window)
        self.last_volatility: float = 0.5  # Normalized 0-1
    
    def update(self, price: float, prev_price: float) -> None:
        """Update with new price data."""
        if prev_price > 0:
            ret = abs(price - prev_price) / prev_price
            self.returns.append(ret)
    
    def get_volatility_score(self) -> float:
        """Get normalized volatility score (0-1)."""
        if len(self.returns) < 10:
            return 0.5
        
        current_vol = np.std(list(self.returns)[-20:]) if len(self.returns) >= 20 else np.std(list(self.returns))
        historical_vol = np.std(list(self.returns))
        
        if historical_vol == 0:
            return 0.5
        
        # Normalize: current vol / historical vol, capped at 2.0
        ratio = min(current_vol / historical_vol, 2.0)
        self.last_volatility = ratio / 2.0  # Scale to 0-1
        return self.last_volatility
    
    def get_risk_multiplier(self) -> float:
        """Get risk multiplier based on volatility.
        
        Low vol = higher risk (more opportunity)
        High vol = lower risk (protect capital)
        """
        vol = self.get_volatility_score()
        
        # Inverse relationship: low vol = high multiplier
        if vol < 0.25:  # Very low volatility
            return 1.5  # Increase risk
        elif vol < 0.5:  # Low volatility
            return 1.2
        elif vol < 0.75:  # Normal volatility
            return 1.0
        else:  # High volatility
            return 0.7  # Reduce risk


class StopLossOptimizer:
    """Learns optimal stop loss distances based on market conditions."""
    
    def __init__(self, config: RiskLearningConfig):
        self.config = config
        
        # Learned optimal stop losses by regime
        self.regime_stops: Dict[str, float] = {
            "trending": config.base_stop_loss_pct,
            "ranging": config.base_stop_loss_pct * 0.7,  # Tighter in ranging
            "high_vol": config.base_stop_loss_pct * 1.5,  # Wider in high vol
            "low_vol": config.base_stop_loss_pct * 0.8,
        }
        
        # Trade outcomes for learning
        self.outcomes: Deque[TradeOutcome] = deque(maxlen=config.performance_window)
    
    def record_trade(self, outcome: TradeOutcome) -> None:
        """Record a trade outcome for learning."""
        self.outcomes.append(outcome)
    
    def get_optimal_stop(self, regime: str, volatility: float) -> float:
        """Get learned optimal stop loss percentage."""
        base_stop = self.regime_stops.get(regime, self.config.base_stop_loss_pct)
        
        # Adjust for volatility
        vol_multiplier = 0.5 + volatility  # Higher vol = wider stop
        adjusted_stop = base_stop * vol_multiplier
        
        return np.clip(adjusted_stop, self.config.min_stop_loss_pct, self.config.max_stop_loss_pct)
    
    def learn_from_outcomes(self) -> Dict[str, float]:
        """Learn optimal stops from trade outcomes."""
        if len(self.outcomes) < 20:
            return {"updates": 0}
        
        updates = 0
        
        # Group outcomes by regime
        regime_outcomes: Dict[str, List[TradeOutcome]] = {}
        for outcome in self.outcomes:
            if outcome.regime not in regime_outcomes:
                regime_outcomes[outcome.regime] = []
            regime_outcomes[outcome.regime].append(outcome)
        
        # Learn optimal stop for each regime
        for regime, outcomes in regime_outcomes.items():
            if len(outcomes) < 5:
                continue
            
            # Analyze stopped out trades vs winners
            stopped_trades = [o for o in outcomes if o.was_stopped_out]
            winning_trades = [o for o in outcomes if o.pnl > 0 and not o.was_stopped_out]
            
            if not stopped_trades:
                continue
            
            # Calculate average stop distance for stopped trades
            avg_stopped_distance = np.mean([
                abs(o.entry_price - o.stop_loss) / o.entry_price 
                for o in stopped_trades
            ])
            
            # If many trades stopped out, widen stops
            stop_rate = len(stopped_trades) / len(outcomes)
            
            if stop_rate > 0.6:  # Too many stopped out
                # Widen stops
                new_stop = self.regime_stops.get(regime, self.config.base_stop_loss_pct) * 1.1
                updates += 1
            elif stop_rate < 0.2:  # Very few stopped out
                # Tighten stops for better R:R
                new_stop = self.regime_stops.get(regime, self.config.base_stop_loss_pct) * 0.95
                updates += 1
            else:
                continue
            
            # Apply learning rate
            old_stop = self.regime_stops.get(regime, self.config.base_stop_loss_pct)
            learned_stop = old_stop * (1 - self.config.stop_loss_learning_rate) + new_stop * self.config.stop_loss_learning_rate
            
            # Clip to bounds
            self.regime_stops[regime] = np.clip(
                learned_stop, 
                self.config.min_stop_loss_pct, 
                self.config.max_stop_loss_pct
            )
        
        return {"updates": updates, "regime_stops": dict(self.regime_stops)}


class PositionSizingLearner:
    """Learns optimal position sizes based on performance."""
    
    def __init__(self, config: RiskLearningConfig):
        self.config = config
        
        # Learned position multipliers by regime
        self.regime_multipliers: Dict[str, float] = {
            "trending": 1.2,
            "ranging": 0.8,
            "high_vol": 0.6,
            "low_vol": 1.0,
        }
        
        # Performance tracking
        self.recent_pnls: Deque[float] = deque(maxlen=config.performance_window)
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        
        # Current learned base position
        self.learned_position_pct: float = config.base_position_pct
    
    def record_trade(self, pnl: float, pnl_pct: float) -> None:
        """Record trade outcome."""
        self.recent_pnls.append(pnl_pct)
        
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
    
    def get_position_multiplier(self, regime: str, confidence: float) -> float:
        """Get position size multiplier based on regime and performance."""
        base_multiplier = self.regime_multipliers.get(regime, 1.0)
        
        # Adjust for consecutive results
        if self.consecutive_losses >= 3:
            # Reduce after consecutive losses
            loss_adjustment = 0.8 ** (self.consecutive_losses - 2)
        elif self.consecutive_wins >= 3:
            # Slightly increase after consecutive wins (hot hand)
            loss_adjustment = 1.1
        else:
            loss_adjustment = 1.0
        
        # Adjust for recent performance
        if len(self.recent_pnls) >= 20:
            recent_avg = np.mean(list(self.recent_pnls)[-20:])
            if recent_avg < -0.005:  # Losing money recently
                perf_adjustment = 0.8
            elif recent_avg > 0.01:  # Winning recently
                perf_adjustment = 1.1
            else:
                perf_adjustment = 1.0
        else:
            perf_adjustment = 1.0
        
        return base_multiplier * loss_adjustment * perf_adjustment * confidence
    
    def learn_from_performance(self) -> Dict[str, float]:
        """Learn optimal position sizes from recent performance."""
        if len(self.recent_pnls) < 30:
            return {"updates": 0}
        
        recent_pnls = list(self.recent_pnls)
        avg_return = np.mean(recent_pnls)
        win_rate = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
        sharpe = np.mean(recent_pnls) / (np.std(recent_pnls) + 1e-8)
        
        # Adjust base position based on Sharpe
        if sharpe > 1.0:  # Good risk-adjusted returns
            target_position = min(
                self.learned_position_pct * 1.1,
                self.config.max_position_pct
            )
        elif sharpe < 0:  # Poor risk-adjusted returns
            target_position = max(
                self.learned_position_pct * 0.9,
                self.config.min_position_pct
            )
        else:
            target_position = self.learned_position_pct
        
        # Apply learning rate
        self.learned_position_pct = (
            self.learned_position_pct * (1 - self.config.position_learning_rate) +
            target_position * self.config.position_learning_rate
        )
        
        return {
            "updates": 1,
            "learned_position_pct": self.learned_position_pct,
            "sharpe": sharpe,
            "win_rate": win_rate,
        }


class LearningRiskManager:
    """
    Risk manager that learns and adapts to market conditions in real-time.
    
    Key Learning Areas:
    1. Position sizing - learns optimal sizes based on volatility and performance
    2. Stop losses - learns optimal stop distances by regime
    3. Drawdown limits - adapts based on recent drawdown patterns
    4. Risk per trade - adjusts based on win rate and Sharpe
    """
    
    def __init__(self, capital: float, config: Optional[RiskLearningConfig] = None):
        self.capital = capital
        self.initial_capital = capital
        self.peak_capital = capital
        self.config = config or RiskLearningConfig()
        
        # Sub-components
        self.volatility_scaler = VolatilityRiskScaler(window=self.config.volatility_window)
        self.stop_loss_optimizer = StopLossOptimizer(self.config)
        self.position_sizer = PositionSizingLearner(self.config)
        
        # Current state
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.trade_count: int = 0
        
        # Learned limits (adapt over time)
        self.learned_drawdown_limit: float = self.config.base_drawdown_limit
        self.learned_risk_per_trade: float = self.config.base_risk_per_trade
        
        # Trade history for learning
        self.trade_history: Deque[TradeOutcome] = deque(maxlen=self.config.performance_window)
        
        # Performance metrics
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0.0
        
        logger.info(f"LearningRiskManager initialized: capital=${capital:,.2f}")
        logger.info(f"  Base position: {self.config.base_position_pct*100:.1f}%")
        logger.info(f"  Base stop loss: {self.config.base_stop_loss_pct*100:.2f}%")
        logger.info(f"  Base drawdown limit: {self.config.base_drawdown_limit*100:.1f}%")
    
    def update_market_data(self, price: float, prev_price: float, regime: str) -> None:
        """Update with new market data for learning."""
        self.volatility_scaler.update(price, prev_price)
    
    def can_trade(self, regime: str) -> Tuple[bool, str]:
        """Check if trading is allowed based on learned risk limits."""
        # Check drawdown
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if current_drawdown > self.learned_drawdown_limit:
            return False, f"Drawdown limit reached: {current_drawdown:.1%} > {self.learned_drawdown_limit:.1%}"
        
        # Check consecutive losses (learned threshold)
        max_consecutive = 5 if regime == "trending" else 3
        if self.consecutive_losses >= max_consecutive:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        # Check daily loss limit
        daily_loss_limit = self.capital * 0.05  # 5% daily limit
        if self.daily_pnl < -daily_loss_limit:
            return False, f"Daily loss limit: ${self.daily_pnl:,.2f}"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        regime: str,
        confidence: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate optimal position size based on learned parameters."""
        volatility = self.volatility_scaler.get_volatility_score()
        vol_multiplier = self.volatility_scaler.get_risk_multiplier()
        
        # Get learned position multiplier
        regime_multiplier = self.position_sizer.get_position_multiplier(regime, confidence)
        
        # Calculate base position
        base_position = self.capital * self.position_sizer.learned_position_pct
        
        # Apply all multipliers
        position_size = base_position * regime_multiplier * vol_multiplier * signal_strength
        
        # Cap at risk per trade limit
        max_risk_amount = self.capital * self.learned_risk_per_trade
        max_position = max_risk_amount / max(self.config.min_stop_loss_pct, 0.005)
        
        # Final bounds
        min_position = self.capital * self.config.min_position_pct
        max_position = min(max_position, self.capital * self.config.max_position_pct)
        
        return np.clip(position_size, min_position, max_position)
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        regime: str,
    ) -> float:
        """Calculate learned optimal stop loss."""
        volatility = self.volatility_scaler.get_volatility_score()
        stop_pct = self.stop_loss_optimizer.get_optimal_stop(regime, volatility)
        
        if side == "buy":
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
        min_rr: float = 2.0,
    ) -> float:
        """Calculate take profit with minimum risk:reward ratio."""
        risk = abs(entry_price - stop_loss)
        min_reward = risk * min_rr
        
        if side == "buy":
            return entry_price + min_reward
        else:
            return entry_price - min_reward
    
    def record_trade(self, outcome: TradeOutcome) -> None:
        """Record a trade outcome for learning."""
        self.trade_history.append(outcome)
        self.trade_count += 1
        self.total_pnl += outcome.pnl
        
        # Update daily PnL
        self.daily_pnl += outcome.pnl
        
        # Update peak capital
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Update consecutive losses
        if outcome.pnl > 0:
            self.consecutive_losses = 0
            self.winning_trades += 1
        else:
            self.consecutive_losses += 1
            self.losing_trades += 1
        
        # Feed to sub-learners
        self.stop_loss_optimizer.record_trade(outcome)
        self.position_sizer.record_trade(outcome.pnl, outcome.pnl_pct)
    
    def update_capital(self, new_capital: float) -> None:
        """Update current capital."""
        self.capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
    
    def learn(self) -> Dict[str, Any]:
        """Run learning cycle to update all risk parameters."""
        results = {}
        
        # Learn optimal stops
        stop_results = self.stop_loss_optimizer.learn_from_outcomes()
        results["stop_learning"] = stop_results
        
        # Learn position sizing
        pos_results = self.position_sizer.learn_from_performance()
        results["position_learning"] = pos_results
        
        # Learn drawdown limit (adaptive based on recovery ability)
        if len(self.trade_history) >= 50:
            drawdown_result = self._learn_drawdown_limit()
            results["drawdown_learning"] = drawdown_result
        
        # Learn risk per trade (based on Sharpe)
        if len(self.trade_history) >= 30:
            risk_result = self._learn_risk_per_trade()
            results["risk_per_trade_learning"] = risk_result
        
        return results
    
    def _learn_drawdown_limit(self) -> Dict[str, float]:
        """Learn optimal drawdown limit based on recovery patterns."""
        if len(self.trade_history) < 50:
            return {"updates": 0}
        
        # Analyze drawdown patterns
        pnls = [t.pnl_pct for t in self.trade_history]
        
        # Calculate max drawdown in recent trades
        cumulative = np.cumprod(1 + np.array(pnls))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(min(drawdowns))
        
        # If we're consistently not hitting drawdown limit, we can be more aggressive
        if max_drawdown < self.learned_drawdown_limit * 0.5:
            # We're being too conservative
            new_limit = max(
                self.learned_drawdown_limit * 0.9,
                self.config.min_drawdown_limit
            )
        elif max_drawdown > self.learned_drawdown_limit * 0.8:
            # Getting close to limit, be more conservative
            new_limit = min(
                self.learned_drawdown_limit * 1.1,
                self.config.max_drawdown_limit
            )
        else:
            return {"updates": 0}
        
        # Apply learning rate
        self.learned_drawdown_limit = (
            self.learned_drawdown_limit * (1 - self.config.drawdown_learning_rate) +
            new_limit * self.config.drawdown_learning_rate
        )
        
        return {
            "updates": 1,
            "learned_drawdown_limit": self.learned_drawdown_limit,
            "max_drawdown_observed": max_drawdown,
        }
    
    def _learn_risk_per_trade(self) -> Dict[str, float]:
        """Learn optimal risk per trade based on Sharpe ratio."""
        if len(self.trade_history) < 30:
            return {"updates": 0}
        
        pnls = [t.pnl_pct for t in self.trade_history]
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8)
        
        # Adjust risk per trade based on Sharpe
        if sharpe > 1.0:
            # Good risk-adjusted returns, can increase risk
            new_risk = min(
                self.learned_risk_per_trade * 1.05,
                self.config.max_risk_per_trade
            )
        elif sharpe < 0:
            # Poor risk-adjusted returns, decrease risk
            new_risk = max(
                self.learned_risk_per_trade * 0.95,
                self.config.min_risk_per_trade
            )
        else:
            return {"updates": 0}
        
        # Apply slowly (low learning rate for risk)
        self.learned_risk_per_trade = (
            self.learned_risk_per_trade * 0.95 +
            new_risk * 0.05
        )
        
        return {
            "updates": 1,
            "learned_risk_per_trade": self.learned_risk_per_trade,
            "sharpe": sharpe,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current risk learning statistics."""
        win_rate = self.winning_trades / max(self.trade_count, 1)
        
        return {
            "capital": self.capital,
            "peak_capital": self.peak_capital,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_rate": win_rate,
            "consecutive_losses": self.consecutive_losses,
            "current_drawdown": (self.peak_capital - self.capital) / self.peak_capital,
            "learned_position_pct": self.position_sizer.learned_position_pct,
            "learned_stop_losses": dict(self.stop_loss_optimizer.regime_stops),
            "learned_drawdown_limit": self.learned_drawdown_limit,
            "learned_risk_per_trade": self.learned_risk_per_trade,
            "volatility_score": self.volatility_scaler.get_volatility_score(),
            "regime_multipliers": dict(self.position_sizer.regime_multipliers),
        }
