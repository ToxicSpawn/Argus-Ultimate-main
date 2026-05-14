"""
STRATEGY ADAPTATION SYSTEM
===========================
Selects and weights strategies based on market conditions.

Key Principles:
1. Different strategies work in different conditions
2. Adapt strategy mix dynamically
3. Never use a strategy that doesn't fit the market
4. Rotate strategies before conditions change
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
class StrategyPerformance:
    """Strategy performance tracking."""
    name: str
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_hold_time: float = 0.0
    last_used: float = 0.0
    condition_performance: Dict[str, float] = field(default_factory=dict)


class StrategyAdaptationSystem:
    """
    Adapts strategy selection and weighting based on market conditions.
    
    Strategies:
    - Trend Following (works in trending markets)
    - Mean Reversion (works in sideways markets)
    - Momentum (works in strong trends)
    - Breakout (works before big moves)
    - Grid Trading (works in range-bound)
    - Scalping (works in liquid markets)
    - Swing Trading (works in moderate trends)
    """
    
    def __init__(self):
        self.strategies: Dict[str, StrategyPerformance] = {}
        self.current_weights: Dict[str, float] = {}
        self.active_strategies: List[str] = []
        
        # Initialize all strategies
        self._init_strategies()
        
        # Market condition -> strategy weights
        self.condition_strategy_map = self._build_condition_map()
        
        logger.info(f"StrategyAdaptationSystem initialized: {len(self.strategies)} strategies")
    
    def _init_strategies(self):
        """Initialize strategy performance tracking."""
        strategy_names = [
            "trend_following",
            "mean_reversion",
            "momentum",
            "breakout",
            "grid_trading",
            "scalping",
            "swing_trading",
            "statistical_arb",
            "pairs_trading",
            "volatility_trading",
        ]
        
        for name in strategy_names:
            self.strategies[name] = StrategyPerformance(name=name)
    
    def _build_condition_map(self) -> Dict[str, Dict[str, float]]:
        """Build market condition to strategy weight mapping."""
        return {
            "bull_strong": {
                "trend_following": 0.35,
                "momentum": 0.30,
                "breakout": 0.15,
                "swing_trading": 0.15,
                "mean_reversion": 0.05,
            },
            "bull_weak": {
                "swing_trading": 0.30,
                "mean_reversion": 0.25,
                "trend_following": 0.20,
                "momentum": 0.15,
                "breakout": 0.10,
            },
            "sideways": {
                "mean_reversion": 0.35,
                "grid_trading": 0.30,
                "swing_trading": 0.20,
                "statistical_arb": 0.15,
            },
            "bear_weak": {
                "swing_trading": 0.30,
                "mean_reversion": 0.25,
                "trend_following": 0.20,  # Short-side
                "volatility_trading": 0.15,
                "statistical_arb": 0.10,
            },
            "bear_strong": {
                "trend_following": 0.35,  # Short-side
                "momentum": 0.25,         # Short-side
                "volatility_trading": 0.20,
                "swing_trading": 0.20,
            },
            "high_vol": {
                "volatility_trading": 0.35,
                "breakout": 0.25,
                "scalping": 0.20,
                "momentum": 0.20,
            },
            "low_liq": {
                "mean_reversion": 0.35,
                "grid_trading": 0.30,
                "swing_trading": 0.25,
                "statistical_arb": 0.10,
            },
            "crash": {
                "volatility_trading": 0.40,
                "mean_reversion": 0.30,  # Buy the dip
                "trend_following": 0.30,  # Short-side
            },
            "pump": {
                "momentum": 0.35,
                "trend_following": 0.30,
                "breakout": 0.25,
                "scalping": 0.10,
            },
        }
    
    async def adapt_to_condition(
        self,
        condition: str,
        volatility: float,
        volume_ratio: float,
    ) -> Dict[str, float]:
        """Adapt strategy weights to market condition."""
        # Get base weights for condition
        base_weights = self.condition_strategy_map.get(condition, {
            "mean_reversion": 0.4,
            "trend_following": 0.3,
            "swing_trading": 0.3,
        })
        
        # Adjust based on historical performance
        adjusted_weights = {}
        for strategy, weight in base_weights.items():
            perf = self.strategies.get(strategy)
            if perf and perf.trades > 10:
                # Adjust weight based on performance in this condition
                cond_perf = perf.condition_performance.get(condition, 0.5)
                perf_factor = 0.5 + cond_perf  # 0.5 to 1.5
                
                # Adjust for overall Sharpe
                sharpe_factor = max(0.5, min(1.5, 1.0 + perf.sharpe * 0.2))
                
                adjusted_weights[strategy] = weight * perf_factor * sharpe_factor
            else:
                adjusted_weights[strategy] = weight
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        # Update active strategies
        self.current_weights = adjusted_weights
        self.active_strategies = [s for s, w in adjusted_weights.items() if w > 0.05]
        
        logger.debug(f"Strategy weights adapted to {condition}: {len(self.active_strategies)} active")
        
        return adjusted_weights
    
    async def record_strategy_result(
        self,
        strategy: str,
        pnl: float,
        hold_time: float,
        condition: str,
    ):
        """Record strategy performance."""
        if strategy not in self.strategies:
            return
        
        perf = self.strategies[strategy]
        perf.trades += 1
        perf.total_pnl += pnl
        perf.avg_hold_time = (perf.avg_hold_time * (perf.trades - 1) + hold_time) / perf.trades
        
        if pnl > 0:
            perf.wins += 1
        
        # Update condition-specific performance
        if condition not in perf.condition_performance:
            perf.condition_performance[condition] = 0.5
        
        # Exponential moving average
        win_score = 1.0 if pnl > 0 else 0.0
        perf.condition_performance[condition] = (
            perf.condition_performance[condition] * 0.9 + win_score * 0.1
        )
        
        perf.last_used = time.time()
    
    def get_strategy_signal(
        self,
        strategy: str,
        market_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get signal from a strategy."""
        weight = self.current_weights.get(strategy, 0)
        
        if weight < 0.05:
            return None  # Strategy not active
        
        # Strategy-specific signal generation
        signal_generators = {
            "trend_following": self._trend_following_signal,
            "mean_reversion": self._mean_reversion_signal,
            "momentum": self._momentum_signal,
            "breakout": self._breakout_signal,
            "grid_trading": self._grid_trading_signal,
            "scalping": self._scalping_signal,
            "swing_trading": self._swing_trading_signal,
        }
        
        generator = signal_generators.get(strategy)
        if generator:
            signal = generator(market_data)
            signal["strategy"] = strategy
            signal["weight"] = weight
            return signal
        
        return None
    
    def _trend_following_signal(self, data: Dict) -> Dict[str, Any]:
        """Trend following signal."""
        prices = data.get("prices", [])
        if len(prices) < 50:
            return {"action": "hold", "confidence": 0}
        
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        
        if sma_20 > sma_50 * 1.02:
            return {"action": "buy", "confidence": 0.7}
        elif sma_20 < sma_50 * 0.98:
            return {"action": "sell", "confidence": 0.7}
        
        return {"action": "hold", "confidence": 0.3}
    
    def _mean_reversion_signal(self, data: Dict) -> Dict[str, Any]:
        """Mean reversion signal."""
        prices = data.get("prices", [])
        if len(prices) < 20:
            return {"action": "hold", "confidence": 0}
        
        current = prices[-1]
        mean = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        
        if std == 0:
            return {"action": "hold", "confidence": 0}
        
        z_score = (current - mean) / std
        
        if z_score < -2:
            return {"action": "buy", "confidence": 0.8}  # Oversold
        elif z_score > 2:
            return {"action": "sell", "confidence": 0.8}  # Overbought
        
        return {"action": "hold", "confidence": 0.3}
    
    def _momentum_signal(self, data: Dict) -> Dict[str, Any]:
        """Momentum signal."""
        prices = data.get("prices", [])
        if len(prices) < 10:
            return {"action": "hold", "confidence": 0}
        
        # Rate of change
        roc = (prices[-1] - prices[-10]) / prices[-10]
        
        if roc > 0.05:
            return {"action": "buy", "confidence": 0.7}
        elif roc < -0.05:
            return {"action": "sell", "confidence": 0.7}
        
        return {"action": "hold", "confidence": 0.3}
    
    def _breakout_signal(self, data: Dict) -> Dict[str, Any]:
        """Breakout signal."""
        prices = data.get("prices", [])
        if len(prices) < 20:
            return {"action": "hold", "confidence": 0}
        
        high_20 = max(prices[-20:])
        low_20 = min(prices[-20:])
        current = prices[-1]
        
        if current > high_20 * 1.01:
            return {"action": "buy", "confidence": 0.75}
        elif current < low_20 * 0.99:
            return {"action": "sell", "confidence": 0.75}
        
        return {"action": "hold", "confidence": 0.3}
    
    def _grid_trading_signal(self, data: Dict) -> Dict[str, Any]:
        """Grid trading signal."""
        prices = data.get("prices", [])
        if len(prices) < 10:
            return {"action": "hold", "confidence": 0}
        
        current = prices[-1]
        recent_high = max(prices[-10:])
        recent_low = min(prices[-10:])
        
        # Buy near low, sell near high
        if current < recent_low * 1.02:
            return {"action": "buy", "confidence": 0.6}
        elif current > recent_high * 0.98:
            return {"action": "sell", "confidence": 0.6}
        
        return {"action": "hold", "confidence": 0.4}
    
    def _scalping_signal(self, data: Dict) -> Dict[str, Any]:
        """Scalping signal."""
        prices = data.get("prices", [])
        if len(prices) < 5:
            return {"action": "hold", "confidence": 0}
        
        # Quick momentum
        short_roc = (prices[-1] - prices[-3]) / prices[-3]
        
        if short_roc > 0.005:
            return {"action": "buy", "confidence": 0.5}
        elif short_roc < -0.005:
            return {"action": "sell", "confidence": 0.5}
        
        return {"action": "hold", "confidence": 0.3}
    
    def _swing_trading_signal(self, data: Dict) -> Dict[str, Any]:
        """Swing trading signal."""
        prices = data.get("prices", [])
        if len(prices) < 30:
            return {"action": "hold", "confidence": 0}
        
        # Support/resistance
        recent_prices = prices[-30:]
        support = min(recent_prices)
        resistance = max(recent_prices)
        current = prices[-1]
        
        range_size = resistance - support
        if range_size == 0:
            return {"action": "hold", "confidence": 0}
        
        position_in_range = (current - support) / range_size
        
        if position_in_range < 0.2:
            return {"action": "buy", "confidence": 0.7}
        elif position_in_range > 0.8:
            return {"action": "sell", "confidence": 0.7}
        
        return {"action": "hold", "confidence": 0.4}
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get strategy adaptation summary."""
        return {
            "active_strategies": self.active_strategies,
            "current_weights": self.current_weights,
            "total_strategies": len(self.strategies),
            "performance": {
                name: {
                    "trades": perf.trades,
                    "win_rate": perf.wins / perf.trades if perf.trades > 0 else 0,
                    "total_pnl": perf.total_pnl,
                    "sharpe": perf.sharpe,
                }
                for name, perf in self.strategies.items()
                if perf.trades > 0
            },
        }


def get_strategy_adaptation() -> StrategyAdaptationSystem:
    """Get strategy adaptation system instance."""
    return StrategyAdaptationSystem()
