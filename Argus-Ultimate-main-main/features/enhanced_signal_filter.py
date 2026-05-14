"""
Enhanced Signal Filter v2
=========================
More aggressive signal filtering for higher trade frequency while maintaining quality.

Key improvements from v1:
1. Lower base thresholds (0.35 vs 0.50)
2. Faster learning rate (0.15 vs 0.05)
3. Dynamic threshold adjustment based on recent win rate
4. Signal momentum scoring (favor signals in trending conditions)
5. Confluence bonus (multiple agreeing signals get boosted)
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
class EnhancedFilterConfig:
    """Configuration for enhanced signal filtering."""
    
    # Base confidence thresholds by regime (LOWER = more trades)
    base_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "trending_up": 0.15,       # Very low - catch trends early
        "trending_down": 0.18,     # Low for downtrends
        "ranging": 0.22,           # Moderate for ranging
        "ranging_tight": 0.20,     # Slightly lower
        "high_volatility": 0.25,   # Slightly higher for safety
        "low_volatility": 0.15,    # Lowest - catch breakouts
        "default": 0.20,
    })
    
    # Learning parameters (FASTER learning)
    learning_rate: float = 0.15      # How fast thresholds adjust (was 0.05)
    performance_window: int = 30     # Trades to look back (was 50)
    min_trades_to_learn: int = 5     # Minimum trades before adjusting
    
    # Profit factor targets
    target_profit_factor: float = 1.8  # Aim for this
    min_acceptable_pf: float = 1.2     # Below this = raise threshold
    
    # Overtrading prevention
    max_trades_per_hour: int = 12      # More aggressive than before (was 6)
    min_time_between_trades: float = 60  # 1 minute (was 5 minutes)
    
    # Confluence settings
    require_confluence: bool = False   # Don't require multiple signals
    confluence_bonus: float = 0.15     # Boost for agreeing signals
    
    # Signal momentum
    momentum_window: int = 10          # Cycles to check for momentum
    momentum_threshold: float = 0.6    # Signal direction consistency


class EnhancedSignalFilter:
    """
    Enhanced signal filter with faster learning and more aggressive thresholds.
    """
    
    def __init__(self, config: Optional[EnhancedFilterConfig] = None):
        self.config = config or EnhancedFilterConfig()
        
        # Current thresholds per regime
        self.thresholds: Dict[str, float] = dict(self.config.base_thresholds)
        
        # Performance tracking
        self.trade_history: Deque[Dict] = deque(maxlen=100)
        self.regime_performance: Dict[str, List[Dict]] = {
            "trending_up": [],
            "trending_down": [],
            "ranging": [],
            "ranging_tight": [],
            "high_volatility": [],
            "low_volatility": [],
        }
        
        # Signal tracking for momentum
        self.recent_signals: Deque[Dict] = deque(maxlen=self.config.momentum_window)
        
        # Overtrading prevention
        self.last_trade_time: float = 0
        self.trades_last_hour: Deque[float] = deque(maxlen=100)
        
        # Statistics
        self.total_signals: int = 0
        self.passed_signals: int = 0
        self.filtered_signals: int = 0
        self.threshold_adjustments: int = 0
    
    def should_allow_trade(self) -> bool:
        """Check if trading is allowed (overtrading prevention)."""
        now = time.time()
        
        # Check time since last trade
        if now - self.last_trade_time < self.config.min_time_between_trades:
            return False
        
        # Check trades per hour
        one_hour_ago = now - 3600
        recent_trades = [t for t in self.trades_last_hour if t > one_hour_ago]
        if len(recent_trades) >= self.config.max_trades_per_hour:
            return False
        
        return True
    
    def record_trade_time(self) -> None:
        """Record that a trade was executed."""
        now = time.time()
        self.last_trade_time = now
        self.trades_last_hour.append(now)
    
    def filter_signal(
        self,
        signal: Dict[str, Any],
        regime: str,
        volatility: float = 0.5,
        all_signals: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Filter a signal with enhanced logic.
        
        Returns:
            {
                "should_trade": bool,
                "confidence": float,
                "quality": float,
                "reasoning": str,
            }
        """
        self.total_signals += 1
        
        # Normalize regime
        regime_norm = self._normalize_regime(regime)
        
        # Get base threshold for this regime
        base_threshold = self.thresholds.get(regime_norm, self.config.base_thresholds["default"])
        
        # Adjust threshold for volatility
        vol_adjustment = (volatility - 0.5) * 0.1  # Higher vol = slightly higher threshold
        threshold = base_threshold + vol_adjustment
        
        # Get signal confidence
        confidence = signal.get("confidence", 0.0)
        
        # Calculate signal quality
        quality = self._calculate_quality(signal, regime_norm, volatility)
        
        # Adjust confidence by quality
        adjusted_confidence = confidence * quality
        
        # Check confluence bonus
        if all_signals and len(all_signals) > 1:
            confluence = self._calculate_confluence(signal, all_signals)
            if confluence > 0.5:
                adjusted_confidence += self.config.confluence_bonus * confluence
        
        # Check overtrading
        if not self.should_allow_trade():
            self.filtered_signals += 1
            return {
                "should_trade": False,
                "confidence": adjusted_confidence,
                "quality": quality,
                "reasoning": f"Overtrading prevention (max {self.config.max_trades_per_hour}/hr)",
            }
        
        # Final threshold check
        if adjusted_confidence >= threshold:
            self.passed_signals += 1
            self.recent_signals.append({
                "signal": signal,
                "regime": regime_norm,
                "confidence": adjusted_confidence,
                "time": time.time(),
            })
            return {
                "should_trade": True,
                "confidence": adjusted_confidence,
                "quality": quality,
                "reasoning": f"Passed: {adjusted_confidence:.2f} >= {threshold:.2f}",
            }
        else:
            self.filtered_signals += 1
            return {
                "should_trade": False,
                "confidence": adjusted_confidence,
                "quality": quality,
                "reasoning": f"Below threshold: {adjusted_confidence:.2f} < {threshold:.2f}",
            }
    
    def _calculate_quality(
        self,
        signal: Dict[str, Any],
        regime: str,
        volatility: float,
    ) -> float:
        """Calculate signal quality score (0.8 - 1.2)."""
        quality = 1.0
        
        # Signal type reliability
        signal_type = signal.get("signal_type", "unknown")
        type_quality = {
            "trend": 1.1,
            "momentum": 1.05,
            "mean_reversion": 0.95,
            "breakout": 1.0,
            "unknown": 0.9,
        }
        quality *= type_quality.get(signal_type, 0.9)
        
        # Regime appropriateness
        regime_quality = {
            ("trend", "trending_up"): 1.2,
            ("trend", "trending_down"): 1.2,
            ("momentum", "trending_up"): 1.1,
            ("momentum", "trending_down"): 1.1,
            ("mean_reversion", "ranging"): 1.15,
            ("mean_reversion", "ranging_tight"): 1.15,
            ("breakout", "ranging"): 1.1,
            ("breakout", "low_volatility"): 1.2,
        }
        quality *= regime_quality.get((signal_type, regime), 1.0)
        
        # Volatility adjustment
        if volatility > 0.7 and signal_type != "breakout":
            quality *= 0.9  # Reduce quality in high vol except breakouts
        
        return np.clip(quality, 0.8, 1.2)
    
    def _calculate_confluence(
        self,
        signal: Dict[str, Any],
        all_signals: List[Dict],
    ) -> float:
        """Calculate how many signals agree with this signal."""
        action = signal.get("action", "")
        agreeing = sum(1 for s in all_signals if s.get("action") == action)
        total = len(all_signals)
        
        return agreeing / total if total > 0 else 0.0
    
    def record_trade_result(
        self,
        pnl: float,
        confidence: float,
        regime: str,
        signal_type: str,
    ) -> None:
        """Record trade result for learning."""
        regime_norm = self._normalize_regime(regime)
        
        trade_data = {
            "pnl": pnl,
            "confidence": confidence,
            "regime": regime_norm,
            "signal_type": signal_type,
            "profitable": pnl > 0,
            "time": time.time(),
        }
        
        self.trade_history.append(trade_data)
        self.regime_performance[regime_norm].append(trade_data)
        
        # Trigger learning
        self._learn_from_trade(trade_data)
    
    def _learn_from_trade(self, trade: Dict) -> None:
        """Learn from trade outcome and adjust thresholds."""
        regime = trade["regime"]
        perf = self.regime_performance.get(regime, [])
        
        if len(perf) < self.config.min_trades_to_learn:
            return
        
        # Get recent performance
        recent = perf[-self.config.performance_window:]
        
        # Calculate metrics
        profits = [t["pnl"] for t in recent if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in recent if t["pnl"] <= 0]
        
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0.001
        
        profit_factor = total_profit / total_loss
        win_rate = sum(1 for t in recent if t["pnl"] > 0) / len(recent)
        
        # Calculate adjustment
        adjustment = 0.0
        
        if profit_factor < self.config.min_acceptable_pf:
            # Below acceptable - raise threshold (be more selective)
            adjustment = +0.02
        elif profit_factor > self.config.target_profit_factor and win_rate > 0.4:
            # Above target - lower threshold (take more similar signals)
            adjustment = -0.015
        elif win_rate > 0.6:
            # High win rate - can afford to lower threshold slightly
            adjustment = -0.01
        elif win_rate < 0.3:
            # Low win rate - raise threshold
            adjustment = +0.015
        
        if adjustment != 0:
            old_threshold = self.thresholds.get(regime, 0.35)
            new_threshold = np.clip(
                old_threshold + adjustment,
                0.12,  # Minimum threshold (very aggressive)
                0.50,  # Maximum threshold
            )
            
            if abs(new_threshold - old_threshold) > 0.001:
                self.thresholds[regime] = new_threshold
                self.threshold_adjustments += 1
                logger.debug(
                    f"Threshold adjusted for {regime}: "
                    f"{old_threshold:.3f} -> {new_threshold:.3f} "
                    f"(PF={profit_factor:.2f}, WR={win_rate:.0%})"
                )
    
    def _normalize_regime(self, regime: str) -> str:
        """Normalize regime string."""
        regime_lower = regime.lower()
        
        if "trend" in regime_lower and "up" in regime_lower:
            return "trending_up"
        elif "trend" in regime_lower and "down" in regime_lower:
            return "trending_down"
        elif "range" in regime_lower and "tight" in regime_lower:
            return "ranging_tight"
        elif "range" in regime_lower or "accumulation" in regime_lower:
            return "ranging"
        elif "high_vol" in regime_lower:
            return "high_volatility"
        elif "low_vol" in regime_lower:
            return "low_volatility"
        else:
            return "ranging"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        pass_rate = self.passed_signals / max(self.total_signals, 1)
        
        recent_trades = list(self.trade_history)[-20:]
        win_rate = sum(1 for t in recent_trades if t["profitable"]) / max(len(recent_trades), 1)
        
        return {
            "total_signals": self.total_signals,
            "passed_signals": self.passed_signals,
            "filtered_signals": self.filtered_signals,
            "pass_rate": pass_rate,
            "threshold_adjustments": self.threshold_adjustments,
            "current_thresholds": dict(self.thresholds),
            "recent_win_rate": win_rate,
            "trades_last_hour": len([t for t in self.trades_last_hour if time.time() - t < 3600]),
        }


# Singleton
_enhanced_filter: Optional[EnhancedSignalFilter] = None


def get_enhanced_filter() -> EnhancedSignalFilter:
    """Get or create singleton enhanced filter."""
    global _enhanced_filter
    if _enhanced_filter is None:
        _enhanced_filter = EnhancedSignalFilter()
    return _enhanced_filter


def reset_enhanced_filter() -> None:
    """Reset singleton (for testing)."""
    global _enhanced_filter
    _enhanced_filter = None
