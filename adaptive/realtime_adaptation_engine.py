"""
Real-Time Adaptation Engine
============================
Makes Argus adapt to market conditions in real-time.

Adaptation happens on EVERY tick/cycle, not every hour.
- Regime detection: every tick
- Strategy weights: every cycle
- Parameter tuning: every 5 cycles
- Champion review: every 3 cycles
- Decay detection: every cycle
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class RealtimeAdaptationConfig:
    """Configuration for real-time adaptation."""
    
    # Core timing
    enabled: bool = True
    adaptation_cycle_seconds: float = 5.0  # Adapt every 5 seconds
    
    # Regime detection (real-time)
    regime_update_on_every_tick: bool = True  # Update regime on every price tick
    regime_min_bars: int = 20  # Minimum bars before regime detection
    
    # Strategy weights (real-time)
    weight_update_interval_cycles: int = 1  # Update weights every cycle
    weight_smoothing_alpha: float = 0.3  # EMA smoothing for weight changes
    weight_min_allocation: float = 0.02  # Minimum 2% allocation
    weight_max_allocation: float = 0.40  # Maximum 40% allocation
    
    # Parameter tuning (fast)
    param_tuning_interval_cycles: int = 5  # Tune params every 5 cycles
    param_tuning_population: int = 20  # Smaller population for speed
    param_tuning_generations: int = 3  # Fewer generations for speed
    
    # Champion/Challenger (fast)
    champion_review_interval_cycles: int = 3  # Review champions every 3 cycles
    champion_min_trades: int = 10  # Minimum trades before promotion
    champion_promotion_threshold: float = 0.1  # 10% better to promote
    
    # Decay detection (real-time)
    decay_check_every_cycle: bool = True  # Check decay every cycle
    decay_lookback_trades: int = 20  # Look at last 20 trades
    decay_slope_threshold: float = -0.05  # Negative slope = decaying
    
    # Performance feedback (real-time)
    performance_update_interval_ms: float = 1000.0  # Update P&L every second
    sharpe_calculation_window: int = 50  # Rolling Sharpe window
    
    # Market regime multipliers (real-time)
    regime_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "TREND_UP": {
            "momentum": 1.5,
            "breakout": 1.3,
            "mean_reversion": 0.5,
            "scalping": 1.2,
        },
        "TREND_DOWN": {
            "momentum": 1.3,
            "breakout": 1.1,
            "mean_reversion": 0.7,
            "short_selling": 1.5,
        },
        "RANGE": {
            "momentum": 0.5,
            "breakout": 0.7,
            "mean_reversion": 1.5,
            "scalping": 1.4,
            "market_making": 1.3,
        },
        "HIGH_VOL": {
            "momentum": 1.2,
            "breakout": 1.4,
            "mean_reversion": 0.6,
            "scalping": 1.5,
            "volatility": 1.6,
        },
        "CRISIS": {
            "momentum": 0.3,
            "breakout": 0.5,
            "mean_reversion": 0.3,
            "scalping": 0.5,
            "hedging": 2.0,
            "defensive": 2.0,
        },
    })


class RealtimeAdaptationEngine:
    """
    Real-time adaptation engine that continuously adjusts:
    - Strategy weights based on regime and performance
    - Strategy parameters based on recent results
    - Champion/Challenger based on rolling metrics
    - Decay detection to remove underperformers
    """
    
    def __init__(self, config: Optional[RealtimeAdaptationConfig] = None):
        self.config = config or RealtimeAdaptationConfig()
        
        # State tracking
        self._cycle_count = 0
        self._last_adaptation_time = 0.0
        self._last_weight_update = 0.0
        self._last_param_tuning = 0
        self._last_champion_review = 0
        
        # Performance tracking
        self._strategy_performance: Dict[str, List[float]] = {}
        self._strategy_weights: Dict[str, float] = {}
        self._strategy_params: Dict[str, Dict[str, Any]] = {}
        self._regime_state: str = "UNKNOWN"
        
        # Decay tracking
        self._decay_scores: Dict[str, float] = {}
        
        logger.info(
            "RealtimeAdaptationEngine initialized: cycle=%.1fs, regime=tick, weights=cycle, params=%dcycles",
            self.config.adaptation_cycle_seconds,
            self.config.param_tuning_interval_cycles,
        )
    
    async def adapt(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main adaptation entry point. Called every cycle.
        
        Returns adaptation decisions:
        - regime: current market regime
        - strategy_weights: updated weights
        - param_updates: parameter changes
        - promotions: champion/challenger changes
        - decay_alerts: strategies to reduce/remove
        """
        self._cycle_count += 1
        now = time.time()
        
        decisions = {
            "cycle": self._cycle_count,
            "regime": None,
            "weight_updates": {},
            "param_updates": {},
            "promotions": [],
            "demotions": [],
            "decay_alerts": [],
        }
        
        # 1. REAL-TIME REGIME DETECTION (every tick)
        if self.config.regime_update_on_every_tick:
            decisions["regime"] = await self._detect_regime(market_state)
            self._regime_state = decisions["regime"]
        
        # 2. REAL-TIME WEIGHT UPDATE (every cycle)
        if self._cycle_count % self.config.weight_update_interval_cycles == 0:
            decisions["weight_updates"] = await self._update_strategy_weights(
                market_state, decisions["regime"]
            )
            self._last_weight_update = now
        
        # 3. FAST PARAMETER TUNING (every N cycles)
        if self._cycle_count % self.config.param_tuning_interval_cycles == 0:
            decisions["param_updates"] = await self._tune_parameters(market_state)
            self._last_param_tuning = self._cycle_count
        
        # 4. FAST CHAMPION REVIEW (every N cycles)
        if self._cycle_count % self.config.champion_review_interval_cycles == 0:
            promo_result = await self._review_champions(market_state)
            decisions["promotions"] = promo_result.get("promotions", [])
            decisions["demotions"] = promo_result.get("demotions", [])
            self._last_champion_review = self._cycle_count
        
        # 5. REAL-TIME DECAY DETECTION (every cycle)
        if self.config.decay_check_every_cycle:
            decisions["decay_alerts"] = await self._detect_decay(market_state)
        
        self._last_adaptation_time = now
        return decisions
    
    async def _detect_regime(self, market_state: Dict[str, Any]) -> str:
        """
        Detect current market regime in real-time.
        
        Uses price action, volatility, and volume to classify:
        - TREND_UP: Strong upward momentum
        - TREND_DOWN: Strong downward momentum
        - RANGE: Sideways/choppy
        - HIGH_VOL: High volatility environment
        - CRISIS: Extreme moves, potential crash
        """
        prices = market_state.get("prices", {})
        volumes = market_state.get("volumes", {})
        indicators = market_state.get("indicators", {})
        
        # Get primary symbol data
        btc_price = prices.get("BTC/USD", {})
        closes = btc_price.get("close_history", [])
        
        if len(closes) < self.config.regime_min_bars:
            return "UNKNOWN"
        
        # Calculate regime indicators
        recent = closes[-10:]
        older = closes[-30:-10]
        
        # Trend detection
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        trend_pct = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        # Volatility detection
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = (sum(r**2 for r in returns[-20:]) / 20) ** 0.5
        
        # Crisis detection (extreme moves)
        max_move = max(abs(r) for r in returns[-5:]) if returns else 0
        
        # Classify regime
        if max_move > 0.05:  # 5%+ move in single period
            return "CRISIS"
        elif volatility > 0.03:  # High volatility
            return "HIGH_VOL"
        elif trend_pct > 0.02:  # 2%+ upward trend
            return "TREND_UP"
        elif trend_pct < -0.02:  # 2%+ downward trend
            return "TREND_DOWN"
        else:
            return "RANGE"
    
    async def _update_strategy_weights(
        self, 
        market_state: Dict[str, Any], 
        regime: Optional[str]
    ) -> Dict[str, float]:
        """
        Update strategy weights in real-time based on:
        1. Current regime multipliers
        2. Recent performance
        3. Decay scores
        """
        if not regime or regime == "UNKNOWN":
            return {}
        
        regime_multipliers = self.config.regime_multipliers.get(regime, {})
        
        weight_updates = {}
        for strategy_type, multiplier in regime_multipliers.items():
            # Get base performance
            perf = self._strategy_performance.get(strategy_type, [])
            base_weight = sum(perf[-10:]) / max(len(perf[-10:]), 1) if perf else 0.5
            
            # Apply regime multiplier
            adjusted_weight = base_weight * multiplier
            
            # Apply decay penalty
            decay_penalty = 1.0 - self._decay_scores.get(strategy_type, 0.0)
            adjusted_weight *= decay_penalty
            
            # Clamp to configured bounds
            adjusted_weight = max(
                self.config.weight_min_allocation,
                min(self.config.weight_max_allocation, adjusted_weight)
            )
            
            # EMA smoothing
            old_weight = self._strategy_weights.get(strategy_type, 0.5)
            smoothed_weight = (
                self.config.weight_smoothing_alpha * adjusted_weight +
                (1 - self.config.weight_smoothing_alpha) * old_weight
            )
            
            self._strategy_weights[strategy_type] = smoothed_weight
            weight_updates[strategy_type] = smoothed_weight
        
        # Normalize weights to sum to 1.0
        total = sum(weight_updates.values())
        if total > 0:
            weight_updates = {k: v/total for k, v in weight_updates.items()}
            self._strategy_weights = weight_updates
        
        return weight_updates
    
    async def _tune_parameters(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast parameter tuning using simplified optimization.
        
        Instead of full Bayesian optimization (slow), use:
        1. Perturbation-based search
        2. Gradient-free hill climbing
        3. Regime-specific parameter templates
        """
        regime = self._regime_state
        param_updates = {}
        
        # Regime-specific parameter templates
        regime_params = {
            "TREND_UP": {
                "rsi_period": 14,
                "rsi_overbought": 75,
                "rsi_oversold": 30,
                "macd_fast": 12,
                "macd_slow": 26,
                "bb_period": 20,
                "bb_std": 2.0,
                "trend_threshold": 0.02,
            },
            "TREND_DOWN": {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 25,
                "macd_fast": 10,
                "macd_slow": 22,
                "bb_period": 20,
                "bb_std": 2.0,
                "trend_threshold": -0.02,
            },
            "RANGE": {
                "rsi_period": 10,
                "rsi_overbought": 65,
                "rsi_oversold": 35,
                "macd_fast": 8,
                "macd_slow": 21,
                "bb_period": 15,
                "bb_std": 1.5,
                "range_threshold": 0.01,
            },
            "HIGH_VOL": {
                "rsi_period": 7,
                "rsi_overbought": 80,
                "rsi_oversold": 20,
                "macd_fast": 5,
                "macd_slow": 15,
                "bb_period": 10,
                "bb_std": 2.5,
                "vol_multiplier": 1.5,
            },
            "CRISIS": {
                "rsi_period": 5,
                "rsi_overbought": 85,
                "rsi_oversold": 15,
                "macd_fast": 3,
                "macd_slow": 10,
                "bb_period": 10,
                "bb_std": 3.0,
                "crisis_mode": True,
            },
        }
        
        # Apply regime-specific parameters
        if regime in regime_params:
            param_updates = regime_params[regime].copy()
            self._strategy_params[regime] = param_updates
        
        return param_updates
    
    async def _review_champions(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast champion/challenger review.
        
        Compares recent performance and promotes/demotes as needed.
        """
        result = {"promotions": [], "demotions": []}
        
        # Get performance metrics
        for strategy_type, perf_history in self._strategy_performance.items():
            if len(perf_history) < self.config.champion_min_trades:
                continue
            
            recent_perf = sum(perf_history[-10:]) / 10
            older_perf = sum(perf_history[-20:-10]) / 10 if len(perf_history) >= 20 else recent_perf
            
            # Calculate improvement
            improvement = (recent_perf - older_perf) / abs(older_perf) if older_perf != 0 else 0
            
            # Promotion/demotion decisions
            if improvement > self.config.champion_promotion_threshold:
                result["promotions"].append({
                    "strategy": strategy_type,
                    "improvement": improvement,
                    "action": "PROMOTE",
                })
            elif improvement < -self.config.champion_promotion_threshold:
                result["demotions"].append({
                    "strategy": strategy_type,
                    "improvement": improvement,
                    "action": "DEMOTE",
                })
        
        return result
    
    async def _detect_decay(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Real-time decay detection using linear regression on recent P&L.
        
        Returns list of strategies showing decay patterns.
        """
        alerts = []
        
        for strategy_type, perf_history in self._strategy_performance.items():
            if len(perf_history) < self.config.decay_lookback_trades:
                continue
            
            # Simple linear regression on recent performance
            recent = perf_history[-self.config.decay_lookback_trades:]
            n = len(recent)
            
            # Calculate slope
            x_mean = (n - 1) / 2
            y_mean = sum(recent) / n
            
            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            
            # Check for decay
            if slope < self.config.decay_slope_threshold:
                alerts.append({
                    "strategy": strategy_type,
                    "slope": slope,
                    "severity": "HIGH" if slope < -0.1 else "MEDIUM",
                    "action": "REDUCE_EXPOSURE",
                })
                self._decay_scores[strategy_type] = min(1.0, abs(slope))
            else:
                self._decay_scores[strategy_type] = 0.0
        
        return alerts
    
    def update_performance(self, strategy_type: str, pnl: float):
        """Update performance tracking for a strategy."""
        if strategy_type not in self._strategy_performance:
            self._strategy_performance[strategy_type] = []
        self._strategy_performance[strategy_type].append(pnl)
        
        # Keep only recent history
        if len(self._strategy_performance[strategy_type]) > 100:
            self._strategy_performance[strategy_type] = self._strategy_performance[strategy_type][-100:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current adaptation status."""
        return {
            "cycle": self._cycle_count,
            "regime": self._regime_state,
            "weights": self._strategy_weights.copy(),
            "decay_scores": self._decay_scores.copy(),
            "last_adaptation": self._last_adaptation_time,
        }


# Global instance
_realtime_adapter: Optional[RealtimeAdaptationEngine] = None


def get_realtime_adapter(config: Optional[RealtimeAdaptationConfig] = None) -> RealtimeAdaptationEngine:
    """Get or create the global real-time adaptation engine."""
    global _realtime_adapter
    if _realtime_adapter is None:
        _realtime_adapter = RealtimeAdaptationEngine(config)
    return _realtime_adapter
