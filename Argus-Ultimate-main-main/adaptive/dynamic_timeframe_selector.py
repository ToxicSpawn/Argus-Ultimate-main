"""
Dynamic Timeframe Selector

Automatically selects the optimal trading timeframe based on:
- Current market volatility
- Recent signal quality per timeframe
- Trend strength
- Time of day (session overlaps)

Supports: 5m, 15m, 30m, 1h, 4h, 1d
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported trading timeframes."""
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    
    @property
    def minutes(self) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        return mapping[self.value]


@dataclass
class TimeframePerformance:
    """Performance metrics for a timeframe."""
    timeframe: Timeframe
    signal_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    avg_signal_strength: float = 0.0
    last_used: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return self.win_count / max(self.signal_count, 1)
    
    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / max(self.signal_count, 1)


@dataclass
class VolatilityRegime:
    """Volatility classification."""
    level: str  # "low", "normal", "high", "extreme"
    atr_pct: float
    recommended_timeframes: List[Timeframe]


class DynamicTimeframeSelector:
    """
    Dynamically selects optimal trading timeframe.
    
    Adapts to market conditions:
    - High volatility → shorter timeframes (catch moves early)
    - Low volatility → longer timeframes (avoid noise)
    - Trending → medium timeframes (ride the trend)
    - Ranging → shorter timeframes (mean reversion)
    """
    
    # Volatility thresholds (ATR as % of price)
    VOLATILITY_THRESHOLDS = {
        "low": 0.01,      # < 1% daily ATR
        "normal": 0.025,   # 1-2.5%
        "high": 0.05,      # 2.5-5%
        "extreme": 0.10,   # > 5%
    }
    
    # Recommended timeframes per volatility regime
    RECOMMENDATIONS = {
        "low": [Timeframe.H4, Timeframe.D1],
        "normal": [Timeframe.H1, Timeframe.H4],
        "high": [Timeframe.M15, Timeframe.M30, Timeframe.H1],
        "extreme": [Timeframe.M5, Timeframe.M15],
    }
    
    def __init__(
        self,
        default_timeframe: Timeframe = Timeframe.H1,
        evaluation_period_minutes: int = 60,
        min_samples_for_switch: int = 5,
        switch_cooldown_minutes: int = 30,
    ):
        self.default_timeframe = default_timeframe
        self.evaluation_period = evaluation_period_minutes
        self.min_samples = min_samples_for_switch
        self.switch_cooldown = switch_cooldown_minutes
        
        self.current_timeframe = default_timeframe
        self.last_switch_time: float = 0
        
        # Track performance per timeframe
        self.performance: Dict[Timeframe, TimeframePerformance] = {
            tf: TimeframePerformance(timeframe=tf)
            for tf in Timeframe
        }
        
        # Recent volatility history
        self.volatility_history: deque = deque(maxlen=100)
        
        # Recent signal history with timeframe tags
        self.signal_history: deque = deque(maxlen=500)
        
        logger.info(
            "DynamicTimeframeSelector initialized: default=%s, cooldown=%dmin",
            default_timeframe.value, switch_cooldown_minutes,
        )
    
    def update_volatility(self, atr_pct: float) -> None:
        """Update with current ATR-based volatility."""
        self.volatility_history.append({
            "atr_pct": atr_pct,
            "timestamp": time.time(),
        })
    
    def record_signal(
        self,
        timeframe: Timeframe,
        pnl_pct: float,
        signal_strength: float,
    ) -> None:
        """Record signal outcome for timeframe evaluation."""
        perf = self.performance[timeframe]
        perf.signal_count += 1
        perf.total_pnl += pnl_pct
        perf.avg_signal_strength = (
            (perf.avg_signal_strength * (perf.signal_count - 1) + signal_strength)
            / perf.signal_count
        )
        
        if pnl_pct > 0:
            perf.win_count += 1
        
        perf.last_used = time.time()
        
        self.signal_history.append({
            "timeframe": timeframe,
            "pnl_pct": pnl_pct,
            "strength": signal_strength,
            "timestamp": time.time(),
        })
    
    def get_volatility_regime(self) -> VolatilityRegime:
        """Classify current volatility regime."""
        if not self.volatility_history:
            return VolatilityRegime("normal", 0.025, [Timeframe.H1])
        
        recent_atr = np.mean([v["atr_pct"] for v in list(self.volatility_history)[-10:]])
        
        if recent_atr < self.VOLATILITY_THRESHOLDS["low"]:
            level = "low"
        elif recent_atr < self.VOLATILITY_THRESHOLDS["normal"]:
            level = "normal"
        elif recent_atr < self.VOLATILITY_THRESHOLDS["high"]:
            level = "high"
        else:
            level = "extreme"
        
        return VolatilityRegime(
            level=level,
            atr_pct=recent_atr,
            recommended_timeframes=self.RECOMMENDATIONS[level],
        )
    
    def select_timeframe(self) -> Timeframe:
        """
        Select optimal timeframe based on current conditions.
        
        Factors:
        1. Current volatility regime
        2. Recent performance per timeframe
        3. Time since last switch (avoid whipsaw)
        """
        # Check cooldown
        time_since_switch = time.time() - self.last_switch_time
        if time_since_switch < self.switch_cooldown * 60:
            return self.current_timeframe
        
        # Get volatility regime
        vol_regime = self.get_volatility_regime()
        
        # Score each timeframe
        scores: Dict[Timeframe, float] = {}
        
        for tf in Timeframe:
            score = 0.0
            
            # 1. Volatility match (40% weight)
            if tf in vol_regime.recommended_timeframes:
                score += 40.0
            elif abs(Timeframe.H1.value) in [t.value for t in vol_regime.recommended_timeframes]:
                score += 20.0  # Partial credit for nearby timeframe
            
            # 2. Historical performance (40% weight)
            perf = self.performance[tf]
            if perf.signal_count >= self.min_samples:
                # Sharpe-like score
                if perf.total_pnl != 0 and perf.signal_count > 0:
                    win_rate_score = perf.win_rate * 20
                    pnl_score = min(20, perf.avg_pnl * 1000)  # Scale up small values
                    score += win_rate_score + pnl_score
            
            # 3. Recent usage penalty (20% weight) - prefer variety
            if tf == self.current_timeframe:
                score -= 10.0  # Slight penalty for staying same (encourage exploration)
            
            scores[tf] = score
        
        # Select best timeframe
        best_tf = max(scores, key=scores.get)
        best_score = scores[best_tf]
        
        # Only switch if improvement is significant (>10 points)
        current_score = scores.get(self.current_timeframe, 0)
        if best_tf != self.current_timeframe and best_score > current_score + 10:
            logger.info(
                "Timeframe switch: %s → %s (score: %.1f → %.1f, vol=%s)",
                self.current_timeframe.value, best_tf.value,
                current_score, best_score, vol_regime.level,
            )
            self.current_timeframe = best_tf
            self.last_switch_time = time.time()
        
        return self.current_timeframe
    
    def get_candle_interval(self) -> str:
        """Get the ccxt-compatible interval string for current timeframe."""
        return self.current_timeframe.value
    
    def get_analysis_period_bars(self, timeframe: Optional[Timeframe] = None) -> int:
        """Get recommended number of bars for analysis."""
        tf = timeframe or self.current_timeframe
        
        # More bars for longer timeframes
        bars_per_tf = {
            Timeframe.M5: 200,
            Timeframe.M15: 150,
            Timeframe.M30: 120,
            Timeframe.H1: 100,
            Timeframe.H4: 80,
            Timeframe.D1: 50,
        }
        return bars_per_tf.get(tf, 100)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current selector status."""
        vol_regime = self.get_volatility_regime()
        
        return {
            "current_timeframe": self.current_timeframe.value,
            "volatility_regime": vol_regime.level,
            "volatility_atr_pct": round(vol_regime.atr_pct * 100, 2),
            "recommended_timeframes": [tf.value for tf in vol_regime.recommended_timeframes],
            "timeframe_performance": {
                tf.value: {
                    "signals": perf.signal_count,
                    "win_rate": round(perf.win_rate * 100, 1),
                    "avg_pnl": round(perf.avg_pnl * 100, 4),
                }
                for tf, perf in self.performance.items()
                if perf.signal_count > 0
            },
            "last_switch": self.last_switch_time,
        }


class TimeframeAdaptiveIndicator:
    """
    Calculates indicators using the optimal timeframe.
    
    Automatically adjusts indicator parameters based on selected timeframe.
    """
    
    # Parameter adjustments per timeframe
    TIMEFRAME_MULTIPLIERS = {
        Timeframe.M5: 0.5,
        Timeframe.M15: 0.75,
        Timeframe.M30: 1.0,
        Timeframe.H1: 1.0,
        Timeframe.H4: 1.5,
        Timeframe.D1: 2.0,
    }
    
    def __init__(self, selector: DynamicTimeframeSelector):
        self.selector = selector
    
    def adjusted_period(self, base_period: int) -> int:
        """Adjust indicator period for current timeframe."""
        multiplier = self.TIMEFRAME_MULTIPLIERS.get(
            self.selector.current_timeframe, 1.0
        )
        return max(5, int(base_period * multiplier))
    
    def rsi_period(self) -> int:
        """Get RSI period adjusted for timeframe."""
        return self.adjusted_period(14)
    
    def macd_params(self) -> Tuple[int, int, int]:
        """Get MACD parameters adjusted for timeframe."""
        base_fast, base_slow, base_signal = 12, 26, 9
        mult = self.TIMEFRAME_MULTIPLIERS.get(self.selector.current_timeframe, 1.0)
        return (
            max(5, int(base_fast * mult)),
            max(10, int(base_slow * mult)),
            max(3, int(base_signal * mult)),
        )
    
    def bb_period(self) -> int:
        """Get Bollinger Band period adjusted for timeframe."""
        return self.adjusted_period(20)
    
    def atr_period(self) -> int:
        """Get ATR period adjusted for timeframe."""
        return self.adjusted_period(14)
