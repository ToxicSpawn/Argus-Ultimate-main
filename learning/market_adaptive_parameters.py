"""
Market Adaptive Parameters
==========================
Unified system that learns and adapts ALL trading parameters every 0.5 seconds.

Parameters learned:
1. Signal Filter Thresholds - per regime, based on pass rates and trade outcomes
2. Strategy Confidence Floors - boost/reduce based on signal quality
3. Threshold Configuration - trend/momentum/reversion/breakout thresholds
4. Filter Settings - max trades/hour, min time between trades
5. Risk Parameters - position sizing, stop losses

Learning every 0.5 seconds based on:
- Recent trade outcomes (win/loss, PnL)
- Signal quality metrics
- Market regime (volatility, trend strength)
- Overtrading prevention needs
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
class MarketState:
    """Current market state for learning."""
    regime: str
    volatility: float  # 0-1
    trend_strength: float  # -1 to 1
    signal_frequency: float  # signals per second
    avg_signal_confidence: float  # 0-1
    timestamp: float = 0.0


@dataclass
class LearningConfig:
    """Configuration for adaptive parameter learning."""
    # Learning interval
    learning_interval: float = 0.5  # Learn every 0.5 seconds
    
    # Performance windows
    trade_window: int = 50  # Last 50 trades for learning
    signal_window: int = 200  # Last 200 signals for learning
    
    # Learning rates (how fast parameters change)
    filter_threshold_rate: float = 0.2  # 20% adjustment per cycle
    confidence_rate: float = 0.15  # 15% adjustment per cycle
    strategy_threshold_rate: float = 0.25  # 25% adjustment per cycle
    
    # Targets
    target_win_rate: float = 0.55  # Aim for 55% win rate
    target_trades_per_hour: float = 30  # Aim for 30 trades/hour
    target_pass_rate: float = 0.25  # 25% of signals should pass
    min_pass_rate: float = 0.10  # Minimum 10% pass rate
    max_pass_rate: float = 0.50  # Maximum 50% pass rate


class PerformanceTracker:
    """Tracks recent performance for learning."""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.trades: Deque[Dict] = deque(maxlen=config.trade_window)
        self.signals: Deque[Dict] = deque(maxlen=config.signal_window)
        self.trade_times: Deque[float] = deque(maxlen=100)
        
    def record_trade(self, trade: Dict) -> None:
        """Record a completed trade."""
        self.trades.append(trade)
        self.trade_times.append(time.time())
    
    def record_signal(self, signal: Dict, passed: bool) -> None:
        """Record a signal and whether it passed filter."""
        self.signals.append({
            "signal": signal,
            "passed": passed,
            "time": time.time(),
        })
    
    def get_win_rate(self) -> float:
        """Get recent win rate (0-1)."""
        if not self.trades:
            return 0.5  # Default assumption
        
        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return wins / len(self.trades)
    
    def get_avg_pnl(self) -> float:
        """Get average PnL per trade."""
        if not self.trades:
            return 0.0
        
        pnls = [t.get("pnl", 0) for t in self.trades]
        return np.mean(pnls)
    
    def get_profit_factor(self) -> float:
        """Get profit factor (gross profit / gross loss)."""
        if not self.trades:
            return 1.0
        
        profits = [t["pnl"] for t in self.trades if t.get("pnl", 0) > 0]
        losses = [abs(t["pnl"]) for t in self.trades if t.get("pnl", 0) < 0]
        
        total_profit = sum(profits) if profits else 0.0
        total_loss = sum(losses) if losses else 0.001
        
        return total_profit / total_loss
    
    def get_trades_per_hour(self) -> float:
        """Get current trade frequency."""
        if len(self.trade_times) < 2:
            return 0.0
        
        now = time.time()
        recent = [t for t in self.trade_times if now - t < 3600]
        return len(recent)
    
    def get_pass_rate(self) -> float:
        """Get signal pass rate."""
        if not self.signals:
            return 0.25  # Default
        
        passed = sum(1 for s in self.signals if s.get("passed", False))
        return passed / len(self.signals)
    
    def get_avg_confidence(self) -> float:
        """Get average signal confidence."""
        if not self.signals:
            return 0.5
        
        confidences = [s["signal"].get("confidence", 0.5) for s in self.signals]
        return np.mean(confidences)


class FilterThresholdLearner:
    """Learns optimal filter thresholds per regime."""
    
    def __init__(self):
        # Current thresholds by regime
        self.thresholds: Dict[str, float] = {
            "trending_up": 0.10,
            "trending_down": 0.12,
            "ranging": 0.15,
            "ranging_tight": 0.12,
            "high_volatility": 0.18,
            "low_volatility": 0.08,
            "default": 0.12,
        }
        
        # Bounds
        self.min_threshold = 0.02
        self.max_threshold = 0.40
        
        # Learning state
        self.adjustments: int = 0
        self.last_adjustment: Dict[str, float] = {}
    
    def learn(self, 
              regime: str, 
              win_rate: float, 
              pass_rate: float,
              profit_factor: float,
              target_win_rate: float,
              target_pass_rate: float) -> float:
        """Learn and return new threshold for regime."""
        
        old_threshold = self.thresholds.get(regime, self.thresholds["default"])
        
        # Calculate adjustments
        adjustment = 0.0
        
        # Win rate too low → raise threshold (be more selective)
        if win_rate < target_win_rate * 0.9:
            adjustment += 0.03
        # Win rate high → lower threshold (take more trades)
        elif win_rate > target_win_rate * 1.1:
            adjustment -= 0.02
        
        # Pass rate too low → lower threshold (more trades pass)
        if pass_rate < 0.10:
            adjustment -= 0.04
        # Pass rate too high → raise threshold (fewer trades)
        elif pass_rate > 0.40:
            adjustment += 0.03
        
        # Profit factor
        if profit_factor < 1.0:
            adjustment += 0.02  # Losing money, be more selective
        elif profit_factor > 2.0:
            adjustment -= 0.01  # Doing well, can be more aggressive
        
        # Apply adjustment
        new_threshold = old_threshold + adjustment
        new_threshold = np.clip(new_threshold, self.min_threshold, self.max_threshold)
        
        if abs(new_threshold - old_threshold) > 0.001:
            self.thresholds[regime] = new_threshold
            self.adjustments += 1
            self.last_adjustment[regime] = adjustment
            
            logger.debug(
                f"Filter threshold learned [{regime}]: "
                f"{old_threshold:.3f} → {new_threshold:.3f} "
                f"(adj={adjustment:+.3f}, WR={win_rate:.0%}, PR={pass_rate:.0%})"
            )
        
        return new_threshold
    
    def get_threshold(self, regime: str) -> float:
        """Get current threshold for regime."""
        return self.thresholds.get(regime, self.thresholds["default"])


class ConfidenceLearner:
    """Learns optimal confidence floors."""
    
    def __init__(self):
        # Current confidence floors by signal type
        self.floors: Dict[str, float] = {
            "trend": 0.45,
            "momentum": 0.45,
            "mean_reversion": 0.45,
            "breakout": 0.50,
        }
        
        # Confidence multipliers by regime
        self.regime_multipliers: Dict[str, float] = {
            "trending_up": 1.0,
            "trending_down": 1.0,
            "ranging": 0.9,  # Lower confidence needed in ranges
            "ranging_tight": 0.85,
            "high_volatility": 1.1,  # Higher confidence in high vol
            "low_volatility": 0.85,
        }
        
        self.adjustments: int = 0
    
    def learn(self,
              regime: str,
              signal_type: str,
              win_rate: float,
              avg_pnl: float,
              target_win_rate: float) -> float:
        """Learn optimal confidence floor."""
        
        old_floor = self.floors.get(signal_type, 0.45)
        
        # Adjustment based on performance
        adjustment = 0.0
        
        # Win rate analysis
        if win_rate < target_win_rate * 0.8:
            # Much lower than target, raise confidence floor
            adjustment += 0.05
        elif win_rate < target_win_rate:
            adjustment += 0.02
        elif win_rate > target_win_rate * 1.2:
            # Much higher, can lower floor
            adjustment -= 0.03
        elif win_rate > target_win_rate:
            adjustment -= 0.01
        
        # Apply
        new_floor = old_floor + adjustment
        new_floor = np.clip(new_floor, 0.25, 0.70)
        
        if abs(new_floor - old_floor) > 0.001:
            self.floors[signal_type] = new_floor
            self.adjustments += 1
            
            logger.debug(
                f"Confidence floor learned [{signal_type}]: "
                f"{old_floor:.3f} → {new_floor:.3f} "
                f"(adj={adjustment:+.3f}, WR={win_rate:.0%})"
            )
        
        return new_floor
    
    def get_adjusted_confidence(self, 
                                 base_confidence: float,
                                 signal_type: str,
                                 regime: str) -> float:
        """Get confidence adjusted by learned floors and regime."""
        floor = self.floors.get(signal_type, 0.45)
        multiplier = self.regime_multipliers.get(regime, 1.0)
        
        # Ensure confidence meets floor, then apply regime multiplier
        adjusted = max(base_confidence, floor) * multiplier
        return np.clip(adjusted, 0.20, 0.90)


class StrategyThresholdLearner:
    """Learns optimal strategy signal thresholds."""
    
    def __init__(self):
        # Current thresholds by regime
        self.thresholds: Dict[str, Dict[str, float]] = {
            "trending": {
                "trend": 0.002,
                "momentum": 0.004,
                "reversion": 0.6,
                "breakout": 0.002,
            },
            "ranging": {
                "trend": 0.004,
                "momentum": 0.008,
                "reversion": 0.4,
                "breakout": 0.001,
            },
            "high_vol": {
                "trend": 0.003,
                "momentum": 0.006,
                "reversion": 0.5,
                "breakout": 0.002,
            },
            "low_vol": {
                "trend": 0.001,
                "momentum": 0.003,
                "reversion": 0.3,
                "breakout": 0.001,
            },
        }
        
        self.adjustments: int = 0
    
    def learn(self,
              regime: str,
              signal_type: str,
              signal_frequency: float,
              win_rate: float,
              target_frequency: float) -> float:
        """Learn optimal strategy threshold."""
        
        # Normalize regime
        regime_norm = self._normalize_regime(regime)
        
        old_threshold = self.thresholds.get(regime_norm, {}).get(signal_type, 0.01)
        
        # Adjustment
        adjustment = 0.0
        
        # Too few signals → lower threshold
        if signal_frequency < target_frequency * 0.5:
            adjustment -= 0.001
        elif signal_frequency < target_frequency * 0.8:
            adjustment -= 0.0005
        
        # Too many signals → raise threshold
        if signal_frequency > target_frequency * 1.5:
            adjustment += 0.001
        elif signal_frequency > target_frequency * 1.2:
            adjustment += 0.0005
        
        # Win rate feedback
        if win_rate < 0.4:
            adjustment += 0.0005  # Be more selective
        elif win_rate > 0.6:
            adjustment -= 0.0003  # Can be more aggressive
        
        # Apply
        new_threshold = old_threshold + adjustment
        
        # Bounds by type
        bounds = {
            "trend": (0.0005, 0.01),
            "momentum": (0.001, 0.02),
            "reversion": (0.2, 2.0),
            "breakout": (0.0005, 0.01),
        }
        min_t, max_t = bounds.get(signal_type, (0.001, 0.01))
        new_threshold = np.clip(new_threshold, min_t, max_t)
        
        if abs(new_threshold - old_threshold) > 0.0001:
            if regime_norm not in self.thresholds:
                self.thresholds[regime_norm] = {}
            self.thresholds[regime_norm][signal_type] = new_threshold
            self.adjustments += 1
            
            logger.debug(
                f"Strategy threshold learned [{regime_norm}/{signal_type}]: "
                f"{old_threshold:.4f} → {new_threshold:.4f} "
                f"(adj={adjustment:+.4f}, freq={signal_frequency:.1f}/hr)"
            )
        
        return new_threshold
    
    def get_threshold(self, regime: str, signal_type: str) -> float:
        """Get current threshold."""
        regime_norm = self._normalize_regime(regime)
        return self.thresholds.get(regime_norm, {}).get(signal_type, 0.01)
    
    def _normalize_regime(self, regime: str) -> str:
        """Normalize regime string."""
        regime_lower = regime.lower()
        if "low_vol" in regime_lower:
            return "low_vol"
        elif "high_vol" in regime_lower:
            return "high_vol"
        elif "trend" in regime_lower:
            return "trending"
        else:
            return "ranging"


class FilterSettingsLearner:
    """Learns optimal filter settings (max trades/hour, min time between)."""
    
    def __init__(self):
        self.max_trades_per_hour: float = 36.0
        self.min_time_between_trades: float = 5.0
        
        self.min_trades_per_hour = 12.0
        self.max_trades_per_hour_limit = 72.0
        self.min_time_min = 2.0
        self.min_time_max = 15.0
        
        self.adjustments: int = 0
    
    def learn(self,
              trades_per_hour: float,
              win_rate: float,
              profit_factor: float,
              target_trades_per_hour: float) -> Tuple[float, float]:
        """Learn optimal filter settings."""
        
        old_max_trades = self.max_trades_per_hour
        old_min_time = self.min_time_between_trades
        
        # Adjust max trades based on performance
        if win_rate > 0.6 and profit_factor > 1.5:
            # Doing well, allow more trades
            self.max_trades_per_hour *= 1.1
        elif win_rate < 0.4 or profit_factor < 0.8:
            # Doing poorly, reduce trades
            self.max_trades_per_hour *= 0.9
        
        # Adjust based on frequency vs target
        if trades_per_hour < target_trades_per_hour * 0.7:
            self.max_trades_per_hour *= 1.05
            self.min_time_between_trades *= 0.95
        elif trades_per_hour > target_trades_per_hour * 1.3:
            self.max_trades_per_hour *= 0.95
            self.min_time_between_trades *= 1.05
        
        # Apply bounds
        self.max_trades_per_hour = np.clip(
            self.max_trades_per_hour,
            self.min_trades_per_hour,
            self.max_trades_per_hour_limit
        )
        self.min_time_between_trades = np.clip(
            self.min_time_between_trades,
            self.min_time_min,
            self.min_time_max
        )
        
        if (abs(self.max_trades_per_hour - old_max_trades) > 0.5 or
            abs(self.min_time_between_trades - old_min_time) > 0.2):
            self.adjustments += 1
            
            logger.debug(
                f"Filter settings learned: "
                f"max_trades={self.max_trades_per_hour:.0f}/hr, "
                f"min_time={self.min_time_between_trades:.1f}s"
            )
        
        return self.max_trades_per_hour, self.min_time_between_trades


class MarketAdaptiveParameters:
    """
    Unified system that learns ALL trading parameters every 0.5 seconds.
    
    This is the central nervous system of Argus's learning capability.
    Every 0.5 seconds, it:
    1. Analyzes recent performance
    2. Adjusts signal filter thresholds
    3. Adjusts strategy confidence floors
    4. Adjusts strategy signal thresholds
    5. Adjusts filter settings (max trades, min time)
    """
    
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        
        # Sub-learners
        self.performance = PerformanceTracker(self.config)
        self.filter_thresholds = FilterThresholdLearner()
        self.confidence = ConfidenceLearner()
        self.strategy_thresholds = StrategyThresholdLearner()
        self.filter_settings = FilterSettingsLearner()
        
        # State
        self.current_regime: str = "ranging"
        self.last_learning_time: float = time.time()
        self.learning_cycles: int = 0
        self.total_learning_ms: float = 0.0
        
        # Signal frequency tracking
        self.signals_per_second: Deque[float] = deque(maxlen=30)
        self.last_signal_count: int = 0
        self.last_signal_check: float = time.time()
        
        logger.info(
            f"MarketAdaptiveParameters initialized: "
            f"learning every {self.config.learning_interval}s"
        )
    
    def record_trade(self, trade: Dict) -> None:
        """Record a completed trade."""
        self.performance.record_trade(trade)
    
    def record_signal(self, signal: Dict, passed: bool) -> None:
        """Record a signal outcome."""
        self.performance.record_signal(signal, passed)
    
    def update_regime(self, regime: str) -> None:
        """Update current market regime."""
        self.current_regime = regime
    
    def should_learn(self) -> bool:
        """Check if it's time to learn."""
        return time.time() - self.last_learning_time >= self.config.learning_interval
    
    def learn(self) -> Dict[str, Any]:
        """
        Run learning cycle - adjusts ALL parameters.
        
        Returns dict of all learned parameters.
        """
        if not self.should_learn():
            return {"learned": False, "reason": "not_time_yet"}
        
        start_time = time.time()
        
        # Get performance metrics
        win_rate = self.performance.get_win_rate()
        avg_pnl = self.performance.get_avg_pnl()
        profit_factor = self.performance.get_profit_factor()
        trades_per_hour = self.performance.get_trades_per_hour()
        pass_rate = self.performance.get_pass_rate()
        avg_confidence = self.performance.get_avg_confidence()
        
        # Calculate signal frequency
        now = time.time()
        signal_frequency = len(self.performance.signals) / max(now - self.performance.signals[0]["time"], 1) * 3600 if self.performance.signals else 0
        
        results = {
            "learned": True,
            "cycle": self.learning_cycles + 1,
            "metrics": {
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "profit_factor": profit_factor,
                "trades_per_hour": trades_per_hour,
                "pass_rate": pass_rate,
                "avg_confidence": avg_confidence,
                "signal_frequency": signal_frequency,
            },
            "parameters": {},
        }
        
        # 1. Learn filter thresholds
        filter_threshold = self.filter_thresholds.learn(
            regime=self.current_regime,
            win_rate=win_rate,
            pass_rate=pass_rate,
            profit_factor=profit_factor,
            target_win_rate=self.config.target_win_rate,
            target_pass_rate=self.config.target_pass_rate,
        )
        results["parameters"]["filter_threshold"] = filter_threshold
        
        # 2. Learn confidence floors for each signal type
        confidence_floors = {}
        for signal_type in ["trend", "momentum", "mean_reversion", "breakout"]:
            floor = self.confidence.learn(
                regime=self.current_regime,
                signal_type=signal_type,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                target_win_rate=self.config.target_win_rate,
            )
            confidence_floors[signal_type] = floor
        results["parameters"]["confidence_floors"] = confidence_floors
        
        # 3. Learn strategy thresholds
        strategy_thresholds = {}
        for signal_type in ["trend", "momentum", "reversion", "breakout"]:
            threshold = self.strategy_thresholds.learn(
                regime=self.current_regime,
                signal_type=signal_type,
                signal_frequency=signal_frequency,
                win_rate=win_rate,
                target_frequency=self.config.target_trades_per_hour,
            )
            strategy_thresholds[signal_type] = threshold
        results["parameters"]["strategy_thresholds"] = strategy_thresholds
        
        # 4. Learn filter settings
        max_trades, min_time = self.filter_settings.learn(
            trades_per_hour=trades_per_hour,
            win_rate=win_rate,
            profit_factor=profit_factor,
            target_trades_per_hour=self.config.target_trades_per_hour,
        )
        results["parameters"]["max_trades_per_hour"] = max_trades
        results["parameters"]["min_time_between_trades"] = min_time
        
        # Update timing
        elapsed_ms = (time.time() - start_time) * 1000
        self.last_learning_time = time.time()
        self.learning_cycles += 1
        self.total_learning_ms += elapsed_ms
        
        results["learning_time_ms"] = elapsed_ms
        
        if self.learning_cycles % 10 == 0:
            logger.info(
                f"Learning cycle #{self.learning_cycles}: "
                f"WR={win_rate:.0%}, PF={profit_factor:.2f}, "
                f"TPH={trades_per_hour:.0f}, "
                f"Filter={filter_threshold:.3f}, "
                f"Time={elapsed_ms:.1f}ms"
            )
        
        return results
    
    def get_filter_threshold(self, regime: Optional[str] = None) -> float:
        """Get current filter threshold for regime."""
        return self.filter_thresholds.get_threshold(regime or self.current_regime)
    
    def get_confidence_floor(self, signal_type: str) -> float:
        """Get current confidence floor for signal type."""
        return self.confidence.floors.get(signal_type, 0.45)
    
    def get_adjusted_confidence(self, 
                                  base_confidence: float,
                                  signal_type: str) -> float:
        """Get confidence adjusted by learned parameters."""
        return self.confidence.get_adjusted_confidence(
            base_confidence, signal_type, self.current_regime
        )
    
    def get_strategy_threshold(self, signal_type: str) -> float:
        """Get current strategy threshold."""
        return self.strategy_thresholds.get_threshold(self.current_regime, signal_type)
    
    def get_filter_settings(self) -> Tuple[float, float]:
        """Get current filter settings (max_trades_per_hour, min_time_between)."""
        return self.filter_settings.max_trades_per_hour, self.filter_settings.min_time_between_trades
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        avg_cycle_ms = self.total_learning_ms / max(self.learning_cycles, 1)
        
        return {
            "learning_cycles": self.learning_cycles,
            "avg_cycle_time_ms": avg_cycle_ms,
            "current_regime": self.current_regime,
            "filter_thresholds": dict(self.filter_thresholds.thresholds),
            "confidence_floors": dict(self.confidence.floors),
            "strategy_thresholds": {
                regime: dict(thresholds)
                for regime, thresholds in self.strategy_thresholds.thresholds.items()
            },
            "max_trades_per_hour": self.filter_settings.max_trades_per_hour,
            "min_time_between_trades": self.filter_settings.min_time_between_trades,
            "win_rate": self.performance.get_win_rate(),
            "profit_factor": self.performance.get_profit_factor(),
            "trades_per_hour": self.performance.get_trades_per_hour(),
            "pass_rate": self.performance.get_pass_rate(),
        }


# Singleton
_adaptive_params: Optional[MarketAdaptiveParameters] = None


def get_adaptive_params() -> MarketAdaptiveParameters:
    """Get or create singleton adaptive parameters instance."""
    global _adaptive_params
    if _adaptive_params is None:
        _adaptive_params = MarketAdaptiveParameters()
    return _adaptive_params


def reset_adaptive_params() -> None:
    """Reset singleton (for testing)."""
    global _adaptive_params
    _adaptive_params = None
