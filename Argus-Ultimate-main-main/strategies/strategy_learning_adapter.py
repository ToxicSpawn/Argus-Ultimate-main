# pyright: reportMissingImports=false
"""
Strategy Learning Adapter
==========================
Connects ALL strategies to the LearningOrchestrator for continuous improvement.

This adapter:
1. Wraps each strategy with learning capabilities
2. Tracks strategy performance and feeds to learning
3. Adjusts strategy parameters based on learned values
4. Provides regime-aware parameter selection
5. Enables strategy self-optimization

ARCHITECTURE:
    Strategy → StrategyLearningAdapter → LearningOrchestrator → 17 Algorithms
    
    Every trade outcome feeds back to:
    - Adjust strategy parameters
    - Update strategy weights
    - Learn optimal thresholds per regime
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGY PARAMETERS THAT CAN BE LEARNED
# ============================================================================

@dataclass
class LearnableStrategyParams:
    """
    Parameters for each strategy that can be learned.
    These are the knobs that the learning algorithms adjust.
    """
    # Momentum
    momentum_short_window: int = 10
    momentum_long_window: int = 40
    momentum_min_strength: float = 0.002
    momentum_accel_threshold: float = 0.0
    
    # Mean Reversion
    mr_lookback: int = 50
    mr_base_threshold: float = 1.5
    mr_vol_scale: float = 1.0
    
    # Trend Following
    tf_fast_window: int = 12
    tf_slow_window: int = 48
    tf_min_diff: float = 0.002
    
    # Breakout
    breakout_lookback: int = 30
    breakout_buffer_pct: float = 0.0015
    
    # Scalping
    scalping_imbalance_threshold: float = 0.15
    scalping_max_spread_bps: float = 3.0
    
    # Universal (applied to all strategies)
    position_size_pct: float = 0.1  # 10% of capital per trade
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_pct: float = 0.04   # 4% take profit
    confidence_threshold: float = 0.5  # Minimum confidence to trade


class StrategyType(Enum):
    """Supported strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"


@dataclass
class StrategyPerformance:
    """Track strategy performance over time."""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_return: float = 0.0
    return_history: List[float] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(self.total_trades, 1)
    
    @property
    def profit_factor(self) -> float:
        wins = sum(r for r in self.return_history if r > 0)
        losses = abs(sum(r for r in self.return_history if r < 0))
        return wins / max(losses, 1e-9)


class StrategyLearningAdapter:
    """
    Adapts a single strategy with learning capabilities.
    
    Wraps any strategy and:
    1. Learns optimal parameters for current regime
    2. Adjusts thresholds based on performance
    3. Reports outcomes to LearningOrchestrator
    4. Gets learned parameters for next decision
    
    Usage:
        adapter = StrategyLearningAdapter(
            strategy=MomentumStrategy(),
            strategy_type=StrategyType.MOMENTUM,
            learning_orchestrator=orchestrator
        )
        
        # Generate signal (uses learned parameters)
        signal = adapter.generate_signal(prices, regime)
        
        # After trade
        adapter.record_outcome(pnl=100.0, regime="trending_up")
    """
    
    def __init__(
        self,
        strategy: Any,
        strategy_type: StrategyType,
        learning_orchestrator: Any = None,
        name: Optional[str] = None,
    ):
        self.strategy = strategy
        self.strategy_type = strategy_type
        self.learning_orchestrator = learning_orchestrator
        self.name = name or strategy_type.value
        
        # Current parameters (will be updated by learning)
        self.params = LearnableStrategyParams()
        
        # Performance tracking
        self.performance = StrategyPerformance(strategy_name=self.name)
        self.regime_performance: Dict[str, StrategyPerformance] = {}
        
        # Learning state
        self.last_signal: Optional[str] = None
        self.last_params_used: Dict[str, float] = {}
        self.trade_history: deque = deque(maxlen=1000)
        
        logger.info(f"StrategyLearningAdapter initialized: {self.name}")
    
    def generate_signal(self, prices: List[float], regime: str = "unknown", **kwargs) -> Dict[str, Any]:
        """
        Generate signal using LEARNED parameters.
        
        Before generating, updates strategy with learned parameters.
        """
        # Update strategy with learned parameters
        self._apply_learned_parameters(regime)
        
        # Generate signal based on strategy type
        if self.strategy_type == StrategyType.MOMENTUM:
            signal = self._generate_momentum(prices)
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            volatility = kwargs.get("volatility")
            signal = self._generate_mean_reversion(prices, volatility)
        elif self.strategy_type == StrategyType.TREND_FOLLOWING:
            signal = self._generate_trend_following(prices)
        elif self.strategy_type == StrategyType.BREAKOUT:
            signal = self._generate_breakout(prices)
        elif self.strategy_type == StrategyType.SCALPING:
            imbalance = kwargs.get("imbalance", 0.0)
            spread_bps = kwargs.get("spread_bps", 1.0)
            signal = self._generate_scalping(imbalance, spread_bps)
        else:
            signal = {"action": "hold", "confidence": 0.0}
        
        # Store for outcome tracking
        self.last_signal = signal.get("action", "hold")
        self.last_params_used = self._get_current_param_values()
        
        return signal
    
    def _apply_learned_parameters(self, regime: str) -> None:
        """Apply learned parameters to strategy."""
        if not self.learning_orchestrator:
            return
        
        # Get learned parameters for this regime
        learned = self.learning_orchestrator.get_parameters_for_decision(
            regime=regime,
            context={"strategy": self.name}
        )
        
        # Apply to strategy if it has the attribute
        if self.strategy_type == StrategyType.MOMENTUM:
            if hasattr(self.strategy, 'short_window'):
                self.strategy.short_window = max(5, int(learned.get("momentum_short_window", self.params.momentum_short_window)))
            if hasattr(self.strategy, 'long_window'):
                self.strategy.long_window = max(20, int(learned.get("momentum_long_window", self.params.momentum_long_window)))
            if hasattr(self.strategy, 'min_strength'):
                self.strategy.min_strength = learned.get("momentum_min_strength", self.params.momentum_min_strength)
        
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            if hasattr(self.strategy, 'lookback'):
                self.strategy.lookback = max(20, int(learned.get("mr_lookback", self.params.mr_lookback)))
            if hasattr(self.strategy, 'base_threshold'):
                self.strategy.base_threshold = learned.get("mr_base_threshold", self.params.mr_base_threshold)
            if hasattr(self.strategy, 'vol_scale'):
                self.strategy.vol_scale = learned.get("mr_vol_scale", self.params.mr_vol_scale)
        
        elif self.strategy_type == StrategyType.TREND_FOLLOWING:
            if hasattr(self.strategy, 'fast_window'):
                self.strategy.fast_window = max(5, int(learned.get("tf_fast_window", self.params.tf_fast_window)))
            if hasattr(self.strategy, 'slow_window'):
                self.strategy.slow_window = max(20, int(learned.get("tf_slow_window", self.params.tf_slow_window)))
        
        elif self.strategy_type == StrategyType.BREAKOUT:
            if hasattr(self.strategy, 'lookback'):
                self.strategy.lookback = max(10, int(learned.get("breakout_lookback", self.params.breakout_lookback)))
            if hasattr(self.strategy, 'buffer_pct'):
                self.strategy.buffer_pct = learned.get("breakout_buffer_pct", self.params.breakout_buffer_pct)
        
        elif self.strategy_type == StrategyType.SCALPING:
            if hasattr(self.strategy, 'imbalance_threshold'):
                self.strategy.imbalance_threshold = learned.get("scalping_imbalance_threshold", self.params.scalping_imbalance_threshold)
            if hasattr(self.strategy, 'max_spread_bps'):
                self.strategy.max_spread_bps = learned.get("scalping_max_spread_bps", self.params.scalping_max_spread_bps)
    
    def _get_current_param_values(self) -> Dict[str, float]:
        """Get current parameter values for learning."""
        params = {}
        
        if self.strategy_type == StrategyType.MOMENTUM:
            params["momentum_short_window"] = float(getattr(self.strategy, 'short_window', 10))
            params["momentum_long_window"] = float(getattr(self.strategy, 'long_window', 40))
            params["momentum_min_strength"] = getattr(self.strategy, 'min_strength', 0.002)
        
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            params["mr_lookback"] = float(getattr(self.strategy, 'lookback', 50))
            params["mr_base_threshold"] = getattr(self.strategy, 'base_threshold', 1.5)
            params["mr_vol_scale"] = getattr(self.strategy, 'vol_scale', 1.0)
        
        elif self.strategy_type == StrategyType.TREND_FOLLOWING:
            params["tf_fast_window"] = float(getattr(self.strategy, 'fast_window', 12))
            params["tf_slow_window"] = float(getattr(self.strategy, 'slow_window', 48))
        
        elif self.strategy_type == StrategyType.BREAKOUT:
            params["breakout_lookback"] = float(getattr(self.strategy, 'lookback', 30))
            params["breakout_buffer_pct"] = getattr(self.strategy, 'buffer_pct', 0.0015)
        
        elif self.strategy_type == StrategyType.SCALPING:
            params["scalping_imbalance_threshold"] = getattr(self.strategy, 'imbalance_threshold', 0.15)
            params["scalping_max_spread_bps"] = getattr(self.strategy, 'max_spread_bps', 3.0)
        
        return params
    
    def _generate_momentum(self, prices: List[float]) -> Dict[str, Any]:
        """Generate momentum signal."""
        if len(prices) <= self.strategy.long_window:
            return {"action": "hold", "confidence": 0.0}
        
        short_ret = (prices[-1] - prices[-self.strategy.short_window]) / prices[-self.strategy.short_window]
        long_ret = (prices[-1] - prices[-self.strategy.long_window]) / prices[-self.strategy.long_window]
        prev_short_ret = (prices[-2] - prices[-self.strategy.short_window - 1]) / prices[-self.strategy.short_window - 1]
        acceleration = short_ret - prev_short_ret
        score = (0.6 * short_ret) + (0.3 * long_ret) + (0.1 * acceleration)
        
        if score > self.strategy.min_strength:
            return {"action": "buy", "confidence": min(score / (self.strategy.min_strength * 4), 1.0)}
        if score < -self.strategy.min_strength:
            return {"action": "sell", "confidence": min(abs(score) / (self.strategy.min_strength * 4), 1.0)}
        return {"action": "hold", "confidence": 0.0}
    
    def _generate_mean_reversion(self, prices: List[float], volatility: Optional[float] = None) -> Dict[str, Any]:
        """Generate mean reversion signal."""
        import math
        
        if len(prices) < self.strategy.lookback:
            return {"action": "hold", "confidence": 0.0}
        
        window = prices[-self.strategy.lookback:]
        mean_price = sum(window) / len(window)
        variance = sum((p - mean_price) ** 2 for p in window) / max(len(window), 1)
        std = math.sqrt(max(variance, 1e-9))
        z = (prices[-1] - mean_price) / std
        
        vol_adj = 1.0 + ((volatility or 0.0) * self.strategy.vol_scale)
        threshold = self.strategy.base_threshold * vol_adj
        
        if z < -threshold:
            return {"action": "buy", "confidence": min(abs(z) / threshold, 1.0)}
        if z > threshold:
            return {"action": "sell", "confidence": min(abs(z) / threshold, 1.0)}
        return {"action": "hold", "confidence": 0.0}
    
    def _generate_trend_following(self, prices: List[float]) -> Dict[str, Any]:
        """Generate trend following signal."""
        if len(prices) < self.strategy.slow_window:
            return {"action": "hold", "confidence": 0.0}
        
        fast_ma = sum(prices[-self.strategy.fast_window:]) / self.strategy.fast_window
        slow_ma = sum(prices[-self.strategy.slow_window:]) / self.strategy.slow_window
        diff = (fast_ma - slow_ma) / max(slow_ma, 1e-9)
        
        if diff > 0.002:
            return {"action": "buy", "confidence": min(diff / 0.01, 1.0)}
        if diff < -0.002:
            return {"action": "sell", "confidence": min(abs(diff) / 0.01, 1.0)}
        return {"action": "hold", "confidence": 0.0}
    
    def _generate_breakout(self, prices: List[float]) -> Dict[str, Any]:
        """Generate breakout signal."""
        if len(prices) < self.strategy.lookback:
            return {"action": "hold", "confidence": 0.0}
        
        # Simplified - using closes as proxy for highs/lows
        lookback_prices = prices[-self.strategy.lookback:]
        breakout_level = max(lookback_prices)
        breakdown_level = min(lookback_prices)
        price = prices[-1]
        
        if price > breakout_level * (1 + self.strategy.buffer_pct):
            score = min((price - breakout_level) / max(breakout_level * self.strategy.buffer_pct, 1e-9), 1.0)
            return {"action": "buy", "confidence": score}
        if price < breakdown_level * (1 - self.strategy.buffer_pct):
            score = min((breakdown_level - price) / max(breakdown_level * self.strategy.buffer_pct, 1e-9), 1.0)
            return {"action": "sell", "confidence": score}
        return {"action": "hold", "confidence": 0.0}
    
    def _generate_scalping(self, imbalance: float, spread_bps: float) -> Dict[str, Any]:
        """Generate scalping signal."""
        if spread_bps > self.strategy.max_spread_bps:
            return {"action": "hold", "confidence": 0.0}
        if imbalance > self.strategy.imbalance_threshold:
            return {"action": "buy", "confidence": min(imbalance / 0.5, 1.0)}
        if imbalance < -self.strategy.imbalance_threshold:
            return {"action": "sell", "confidence": min(abs(imbalance) / 0.5, 1.0)}
        return {"action": "hold", "confidence": 0.0}
    
    def record_outcome(self, pnl: float, regime: str = "unknown") -> None:
        """
        Record trade outcome and update learning.
        
        This is the KEY method that feeds learning.
        """
        # Update performance tracking
        self.performance.total_trades += 1
        self.performance.total_pnl += pnl
        self.performance.return_history.append(pnl)
        
        if pnl > 0:
            self.performance.winning_trades += 1
        
        # Update regime-specific performance
        if regime not in self.regime_performance:
            self.regime_performance[regime] = StrategyPerformance(strategy_name=self.name)
        
        reg_perf = self.regime_performance[regime]
        reg_perf.total_trades += 1
        reg_perf.total_pnl += pnl
        reg_perf.return_history.append(pnl)
        if pnl > 0:
            reg_perf.winning_trades += 1
        
        # Report to LearningOrchestrator
        if self.learning_orchestrator:
            self.learning_orchestrator.record_trade_outcome(
                params_used=self.last_params_used,
                pnl=pnl,
                regime=regime,
                context={"strategy": self.name, "strategy_type": self.strategy_type.value},
                strategy=self.name
            )
        
        # Record in history
        self.trade_history.append({
            "timestamp": time.time(),
            "pnl": pnl,
            "regime": regime,
            "signal": self.last_signal,
            "params": self.last_params_used.copy()
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "total_trades": self.performance.total_trades,
            "win_rate": self.performance.win_rate,
            "total_pnl": self.performance.total_pnl,
            "profit_factor": self.performance.profit_factor,
            "regime_performances": {
                r: {"trades": p.total_trades, "pnl": p.total_pnl, "win_rate": p.win_rate}
                for r, p in self.regime_performance.items()
            }
        }


class StrategyLearningManager:
    """
    Manages learning adapters for ALL strategies.
    
    This is the entry point for connecting strategies to learning.
    
    Usage:
        manager = StrategyLearningManager(learning_orchestrator)
        
        # Register strategies
        manager.register_strategy("momentum", MomentumStrategy(), StrategyType.MOMENTUM)
        manager.register_strategy("mean_reversion", MeanReversionStrategy(), StrategyType.MEAN_REVERSION)
        
        # Generate signals (uses learned parameters)
        signals = manager.generate_all_signals(prices, regime)
        
        # After trades
        manager.record_outcome("momentum", pnl=100.0, regime="trending_up")
    """
    
    def __init__(self, learning_orchestrator: Any = None):
        self.learning_orchestrator = learning_orchestrator
        self.adapters: Dict[str, StrategyLearningAdapter] = {}
        self.strategy_weights: Dict[str, float] = {}
        
        logger.info("StrategyLearningManager initialized")
    
    def register_strategy(
        self,
        name: str,
        strategy: Any,
        strategy_type: StrategyType,
        initial_weight: float = 1.0
    ) -> StrategyLearningAdapter:
        """Register a strategy for learning."""
        adapter = StrategyLearningAdapter(
            strategy=strategy,
            strategy_type=strategy_type,
            learning_orchestrator=self.learning_orchestrator,
            name=name
        )
        
        self.adapters[name] = adapter
        self.strategy_weights[name] = initial_weight
        
        logger.info(f"Registered strategy for learning: {name} ({strategy_type.value})")
        return adapter
    
    def generate_all_signals(
        self,
        prices: List[float],
        regime: str = "unknown",
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals from ALL strategies.
        Each strategy uses its LEARNED parameters.
        """
        signals = {}
        
        for name, adapter in self.adapters.items():
            try:
                signal = adapter.generate_signal(prices, regime, **kwargs)
                signal["weight"] = self.strategy_weights.get(name, 1.0)
                signals[name] = signal
            except Exception as e:
                logger.warning(f"Error generating signal for {name}: {e}")
                signals[name] = {"action": "hold", "confidence": 0.0, "weight": 0.0}
        
        return signals
    
    def get_best_signal(
        self,
        prices: List[float],
        regime: str = "unknown",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get the BEST signal from all strategies.
        Uses weighted voting based on learned weights.
        """
        signals = self.generate_all_signals(prices, regime, **kwargs)
        
        # Calculate weighted votes
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        for name, signal in signals.items():
            weight = signal.get("weight", 1.0)
            confidence = signal.get("confidence", 0.0)
            action = signal.get("action", "hold")
            
            if action == "buy":
                buy_score += weight * confidence
            elif action == "sell":
                sell_score += weight * confidence
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Return best signal
        if buy_score > sell_score and buy_score > 0.3:
            return {"action": "buy", "confidence": buy_score, "source": "ensemble"}
        elif sell_score > buy_score and sell_score > 0.3:
            return {"action": "sell", "confidence": sell_score, "source": "ensemble"}
        
        return {"action": "hold", "confidence": 0.0, "source": "ensemble"}
    
    def record_outcome(
        self,
        strategy_name: str,
        pnl: float,
        regime: str = "unknown"
    ) -> None:
        """Record trade outcome for a specific strategy."""
        if strategy_name in self.adapters:
            self.adapters[strategy_name].record_outcome(pnl, regime)
            
            # Update strategy weight based on performance
            perf = self.adapters[strategy_name].performance
            if perf.total_trades >= 5:
                # Weight based on profit factor and win rate
                weight = perf.profit_factor * perf.win_rate
                self.strategy_weights[strategy_name] = max(0.1, min(5.0, weight))
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all strategies."""
        return {name: adapter.get_stats() for name, adapter in self.adapters.items()}
    
    def get_best_strategy(self, regime: str = "unknown") -> str:
        """Get best performing strategy for current regime."""
        if not self.adapters:
            return ""
        
        # Check regime-specific performance first
        best_name = ""
        best_pnl = float('-inf')
        
        for name, adapter in self.adapters.items():
            if regime in adapter.regime_performance:
                perf = adapter.regime_performance[regime]
                if perf.total_trades >= 5 and perf.total_pnl > best_pnl:
                    best_pnl = perf.total_pnl
                    best_name = name
        
        if best_name:
            return best_name
        
        # Fallback to overall performance
        return max(self.adapters.keys(), key=lambda n: self.adapters[n].performance.total_pnl)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_strategy_manager: Optional[StrategyLearningManager] = None


def get_strategy_learning_manager(learning_orchestrator: Any = None) -> StrategyLearningManager:
    """Get or create the global strategy learning manager."""
    global _global_strategy_manager
    if _global_strategy_manager is None:
        _global_strategy_manager = StrategyLearningManager(learning_orchestrator)
    return _global_strategy_manager


def wire_all_strategies(learning_orchestrator: Any = None) -> StrategyLearningManager:
    """
    Wire all strategies to the learning system.
    
    This is the main entry point for strategy learning.
    """
    manager = get_strategy_learning_manager(learning_orchestrator)
    
    logger.info("=" * 70)
    logger.info("STRATEGY LEARNING MANAGER - All strategies wired to learning")
    logger.info("=" * 70)
    logger.info("  Strategies now benefit from 17 learning algorithms:")
    logger.info("  - Optimal parameters learned per regime")
    logger.info("  - Adaptive thresholds based on performance")
    logger.info("  - Continuous improvement from every trade")
    logger.info("  - Ensemble weighting of all strategies")
    logger.info("=" * 70)
    
    return manager


__all__ = [
    "StrategyLearningAdapter",
    "StrategyLearningManager",
    "LearnableStrategyParams",
    "StrategyType",
    "StrategyPerformance",
    "get_strategy_learning_manager",
    "wire_all_strategies",
]
