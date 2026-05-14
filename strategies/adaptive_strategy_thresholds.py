"""
Adaptive Strategy Thresholds
============================
Learns and adapts strategy signal thresholds based on market conditions.

Problem: Original thresholds too strict (1-3% moves required)
Solution: Adaptive thresholds that learn from market volatility

This module provides:
1. AdaptiveThresholdLearner - Learns optimal thresholds per regime
2. MarketAdaptiveStrategies - Strategies with adaptive thresholds
3. Integration with existing StrategyEngine
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholds."""
    # Trend signal thresholds (SMA difference)
    trend_threshold_min: float = 0.002  # 0.2% (was 1%)
    trend_threshold_max: float = 0.02   # 2%
    trend_threshold_default: float = 0.005  # 0.5%
    
    # Momentum thresholds (rate of change)
    momentum_threshold_min: float = 0.005  # 0.5% (was 3%)
    momentum_threshold_max: float = 0.03   # 3%
    momentum_threshold_default: float = 0.01  # 1%
    
    # Mean reversion thresholds (z-score)
    reversion_threshold_min: float = 1.0  # 1 std dev (was 2)
    reversion_threshold_max: float = 2.5
    reversion_threshold_default: float = 1.5
    
    # Breakout thresholds (% above/below range)
    breakout_threshold_min: float = 0.002  # 0.2% (was 1%)
    breakout_threshold_max: float = 0.015  # 1.5%
    breakout_threshold_default: float = 0.005  # 0.5%
    
    # Learning rate
    learning_rate: float = 0.1
    performance_window: int = 100
    
    # Continuous learning settings
    continuous_learning_interval: float = 0.5  # Learn every 0.5 seconds
    min_outcomes_per_cycle: int = 5  # Minimum outcomes before learning


class AdaptiveThresholdLearner:
    """Learns optimal signal thresholds based on market conditions and performance."""
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()
        
        # Current learned thresholds
        self.thresholds: Dict[str, Dict[str, float]] = {
            "trending": {
                "trend": 0.004,      # 0.4% - tighter in trends
                "momentum": 0.008,   # 0.8%
                "reversion": 1.8,    # Higher z-score in trends
                "breakout": 0.004,   # 0.4%
            },
            "ranging": {
                "trend": 0.008,      # 0.8% - looser (trends rare)
                "momentum": 0.015,   # 1.5%
                "reversion": 1.2,    # Lower z-score in ranges (mean reversion works)
                "breakout": 0.003,   # 0.3% - catch breakouts early
            },
            "high_vol": {
                "trend": 0.006,      # 0.6%
                "momentum": 0.012,   # 1.2%
                "reversion": 1.5,
                "breakout": 0.005,   # 0.5%
            },
            "low_vol": {
                "trend": 0.003,      # 0.3% - very tight
                "momentum": 0.006,   # 0.6%
                "reversion": 1.3,
                "breakout": 0.002,   # 0.2% - catch small breakouts
            },
        }
        
        # Performance tracking per regime
        self.signal_outcomes: Dict[str, List[Tuple[float, bool]]] = {
            "trending": [],
            "ranging": [],
            "high_vol": [],
            "low_vol": [],
        }
        
        # Statistics
        self.signals_generated: int = 0
        self.signals_winning: int = 0
        self.threshold_adjustments: int = 0
        
        # Continuous learning state
        self._continuous_learning_enabled: bool = False
        self._continuous_learning_task: Optional[asyncio.Task] = None
        self._last_learning_time: float = 0.0
        self._learning_cycle_count: int = 0
        self._total_learning_time_ms: float = 0.0
    
    def get_threshold(self, regime: str, signal_type: str) -> float:
        """Get learned threshold for signal type in regime."""
        regime_thresholds = self.thresholds.get(regime, self.thresholds["ranging"])
        return regime_thresholds.get(signal_type, 0.01)
    
    def record_signal_outcome(
        self,
        regime: str,
        signal_type: str,
        signal_strength: float,
        was_profitable: bool,
    ) -> None:
        """Record outcome of a signal for learning."""
        regime = self._normalize_regime(regime)
        
        if regime not in self.signal_outcomes:
            self.signal_outcomes[regime] = []
        
        self.signal_outcomes[regime].append((signal_strength, was_profitable))
        self.signals_generated += 1
        
        if was_profitable:
            self.signals_winning += 1
        
        # Keep window limited
        if len(self.signal_outcomes[regime]) > self.config.performance_window:
            self.signal_outcomes[regime] = self.signal_outcomes[regime][-self.config.performance_window:]
    
    def learn_thresholds(self) -> Dict[str, Any]:
        """Learn optimal thresholds from recent outcomes."""
        results = {"regimes_updated": [], "threshold_changes": {}}
        
        for regime, outcomes in self.signal_outcomes.items():
            if len(outcomes) < 20:
                continue
            
            # Analyze winning vs losing signal strengths
            winning_strengths = [s for s, w in outcomes if w]
            losing_strengths = [s for s, w in outcomes if not w]
            
            if not winning_strengths or not losing_strengths:
                continue
            
            avg_win = np.mean(winning_strengths)
            avg_loss = np.mean(losing_strengths)
            
            # Optimal threshold is between winning and losing averages
            optimal_threshold = (avg_win + avg_loss) / 2
            
            # Update thresholds based on signal type
            regime_thresholds = self.thresholds.get(regime, {})
            
            for signal_type in ["trend", "momentum", "breakout"]:
                old_threshold = regime_thresholds.get(signal_type, 0.01)
                
                # Map optimal to signal type scale
                if signal_type == "trend":
                    mapped = np.clip(optimal_threshold * 0.01, 
                                     self.config.trend_threshold_min,
                                     self.config.trend_threshold_max)
                elif signal_type == "momentum":
                    mapped = np.clip(optimal_threshold * 0.02,
                                     self.config.momentum_threshold_min,
                                     self.config.momentum_threshold_max)
                else:  # breakout
                    mapped = np.clip(optimal_threshold * 0.01,
                                     self.config.breakout_threshold_min,
                                     self.config.breakout_threshold_max)
                
                # Apply learning rate
                new_threshold = old_threshold * (1 - self.config.learning_rate) + mapped * self.config.learning_rate
                
                if abs(new_threshold - old_threshold) > 0.0001:
                    regime_thresholds[signal_type] = new_threshold
                    self.thresholds[regime] = regime_thresholds
                    self.threshold_adjustments += 1
                    
                    if regime not in results["threshold_changes"]:
                        results["threshold_changes"][regime] = {}
                    results["threshold_changes"][regime][signal_type] = {
                        "old": old_threshold,
                        "new": new_threshold,
                    }
            
            results["regimes_updated"].append(regime)
        
        return results
    
    def _normalize_regime(self, regime: str) -> str:
        """Normalize regime string to standard categories."""
        regime_lower = regime.lower()
        
        # Check low_vol first (before high_vol) since both contain "volatility"
        if "low_vol" in regime_lower:
            return "low_vol"
        elif "high_vol" in regime_lower:
            return "high_vol"
        elif "trend" in regime_lower and ("up" in regime_lower or "strong" in regime_lower):
            return "trending"
        elif "trend" in regime_lower and ("down" in regime_lower or "weak" in regime_lower):
            return "trending"
        elif "range" in regime_lower or "accumulation" in regime_lower or "distribution" in regime_lower:
            return "ranging"
        elif "volatility" in regime_lower:
            return "high_vol"  # Default volatility is high
        else:
            return "ranging"  # Default
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        win_rate = self.signals_winning / max(self.signals_generated, 1)
        
        return {
            "signals_generated": self.signals_generated,
            "signals_winning": self.signals_winning,
            "win_rate": win_rate,
            "threshold_adjustments": self.threshold_adjustments,
            "current_thresholds": dict(self.thresholds),
            "continuous_learning_enabled": self._continuous_learning_enabled,
            "learning_cycle_count": self._learning_cycle_count,
            "avg_learning_time_ms": self._total_learning_time_ms / max(self._learning_cycle_count, 1),
        }
    
    def should_learn(self) -> bool:
        """Check if it's time to run a learning cycle."""
        if not self._continuous_learning_enabled:
            return False
        
        time_since_last = time.time() - self._last_learning_time
        return time_since_last >= self.config.continuous_learning_interval
    
    async def start_continuous_learning(self) -> None:
        """Start continuous learning in background."""
        self._continuous_learning_enabled = True
        self._last_learning_time = time.time()
        logger.info(f"Adaptive Thresholds: Continuous learning STARTED ({self.config.continuous_learning_interval}s intervals)")
    
    def stop_continuous_learning(self) -> None:
        """Stop continuous learning."""
        self._continuous_learning_enabled = False
        logger.info(f"Adaptive Thresholds: Continuous learning STOPPED ({self._learning_cycle_count} cycles)")
    
    def run_learning_cycle(self) -> Dict[str, Any]:
        """Run a single learning cycle (called from continuous learning loop)."""
        if not self.should_learn():
            return {"skipped": True, "reason": "not_time_yet"}
        
        start_time = time.time()
        
        # Run the learning
        result = self.learn_thresholds()
        
        # Update timing
        elapsed_ms = (time.time() - start_time) * 1000
        self._last_learning_time = time.time()
        self._learning_cycle_count += 1
        self._total_learning_time_ms += elapsed_ms
        
        result["cycle_number"] = self._learning_cycle_count
        result["learning_time_ms"] = elapsed_ms
        result["skipped"] = False
        
        if result.get("regimes_updated"):
            logger.debug(
                f"Adaptive Thresholds Learning Cycle #{self._learning_cycle_count}: "
                f"Updated {len(result['regimes_updated'])} regimes in {elapsed_ms:.1f}ms"
            )
        
        return result
    
    def record_signal_event(
        self,
        regime: str,
        signal_type: str,
        signal_strength: float,
    ) -> None:
        """Record signal generation event (for tracking signal frequency)."""
        # This can be used to track how often signals are generated
        # and adjust thresholds based on signal frequency
        pass  # Future: track signal frequency per regime


class MarketAdaptiveStrategies:
    """
    Strategy signal generation with adaptive thresholds.
    
    Replaces the fixed-threshold strategies with adaptive versions
    that learn optimal thresholds from market conditions.
    """
    
    def __init__(self, threshold_learner: Optional[AdaptiveThresholdLearner] = None):
        self.learner = threshold_learner or AdaptiveThresholdLearner()
        self._last_signals: Dict[str, Dict] = {}
    
    def get_trend_signal(
        self,
        prices: List[float],
        regime: str,
        sma_short: int = 10,
        sma_long: int = 20,
    ) -> Optional[Dict]:
        """Generate trend signal with adaptive threshold."""
        if len(prices) < sma_long:
            return None
        
        sma_short_val = np.mean(prices[-sma_short:])
        sma_long_val = np.mean(prices[-sma_long:])
        
        # Calculate percentage difference
        diff = (sma_short_val - sma_long_val) / sma_long_val
        abs_diff = abs(diff)
        
        # Get adaptive threshold
        threshold = self.learner.get_threshold(regime, "trend")
        
        if abs_diff > threshold:
            action = "buy" if diff > 0 else "sell"
            # Confidence scales with how much we exceed threshold
            confidence = min(0.85, 0.5 + (abs_diff - threshold) / threshold * 0.2)
            
            return {
                "action": action,
                "confidence": confidence,
                "signal_type": "trend",
                "strength": abs_diff,
                "threshold_used": threshold,
            }
        
        return None
    
    def get_momentum_signal(
        self,
        prices: List[float],
        regime: str,
        lookback: int = 10,
    ) -> Optional[Dict]:
        """Generate momentum signal with adaptive threshold."""
        if len(prices) < lookback + 1:
            return None
        
        # Rate of change
        roc = (prices[-1] - prices[-lookback]) / prices[-lookback]
        abs_roc = abs(roc)
        
        # Get adaptive threshold
        threshold = self.learner.get_threshold(regime, "momentum")
        
        if abs_roc > threshold:
            action = "buy" if roc > 0 else "sell"
            confidence = min(0.85, 0.5 + (abs_roc - threshold) / threshold * 0.2)
            
            return {
                "action": action,
                "confidence": confidence,
                "signal_type": "momentum",
                "strength": abs_roc,
                "threshold_used": threshold,
            }
        
        return None
    
    def get_mean_reversion_signal(
        self,
        prices: List[float],
        regime: str,
        lookback: int = 20,
    ) -> Optional[Dict]:
        """Generate mean reversion signal with adaptive threshold."""
        if len(prices) < lookback:
            return None
        
        mean = np.mean(prices[-lookback:])
        std = np.std(prices[-lookback:])
        
        if std == 0:
            return None
        
        # Z-score
        z = (prices[-1] - mean) / std
        abs_z = abs(z)
        
        # Get adaptive threshold
        threshold = self.learner.get_threshold(regime, "reversion")
        
        if abs_z > threshold:
            # Mean reversion is contrarian
            action = "buy" if z < 0 else "sell"
            confidence = min(0.85, 0.5 + (abs_z - threshold) / threshold * 0.15)
            
            return {
                "action": action,
                "confidence": confidence,
                "signal_type": "mean_reversion",
                "strength": abs_z,
                "threshold_used": threshold,
            }
        
        return None
    
    def get_breakout_signal(
        self,
        prices: List[float],
        regime: str,
        lookback: int = 20,
    ) -> Optional[Dict]:
        """Generate breakout signal with adaptive threshold."""
        if len(prices) < lookback:
            return None
        
        high = max(prices[-lookback:])
        low = min(prices[-lookback:])
        current = prices[-1]
        
        # Calculate breakout magnitude
        if current > high:
            breakout = (current - high) / high
            action = "buy"
        elif current < low:
            breakout = (low - current) / low
            action = "sell"
        else:
            return None
        
        # Get adaptive threshold
        threshold = self.learner.get_threshold(regime, "breakout")
        
        if breakout > threshold:
            confidence = min(0.85, 0.55 + (breakout - threshold) / threshold * 0.2)
            
            return {
                "action": action,
                "confidence": confidence,
                "signal_type": "breakout",
                "strength": breakout,
                "threshold_used": threshold,
            }
        
        return None
    
    def get_all_signals(
        self,
        prices: List[float],
        regime: str,
    ) -> List[Dict]:
        """Get all adaptive signals for given prices and regime."""
        signals = []
        
        # Trend
        trend = self.get_trend_signal(prices, regime)
        if trend:
            trend["strategy"] = "trend"
            signals.append(trend)
        
        # Momentum
        momentum = self.get_momentum_signal(prices, regime)
        if momentum:
            momentum["strategy"] = "momentum"
            signals.append(momentum)
        
        # Mean Reversion
        reversion = self.get_mean_reversion_signal(prices, regime)
        if reversion:
            reversion["strategy"] = "mean_reversion"
            signals.append(reversion)
        
        # Breakout
        breakout = self.get_breakout_signal(prices, regime)
        if breakout:
            breakout["strategy"] = "breakout"
            signals.append(breakout)
        
        return signals
    
    def record_outcome(self, regime: str, signal_type: str, strength: float, profitable: bool) -> None:
        """Record signal outcome for learning."""
        self.learner.record_signal_outcome(regime, signal_type, strength, profitable)
    
    def learn(self) -> Dict[str, Any]:
        """Run learning cycle."""
        return self.learner.learn_thresholds()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.learner.get_stats()


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_adaptive_strategies: Optional[MarketAdaptiveStrategies] = None


def get_adaptive_strategies() -> MarketAdaptiveStrategies:
    """Get or create singleton adaptive strategies instance."""
    global _adaptive_strategies
    if _adaptive_strategies is None:
        _adaptive_strategies = MarketAdaptiveStrategies()
    return _adaptive_strategies


def reset_adaptive_strategies() -> None:
    """Reset singleton (for testing)."""
    global _adaptive_strategies
    _adaptive_strategies = None
