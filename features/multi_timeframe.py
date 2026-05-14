"""
Multi-Timeframe Analysis Module
================================
Analyzes signals across multiple timeframes for confluence.

Key insights:
- 5m signal + 15m signal + 1h trend = high conviction
- Higher timeframe context prevents counter-trend trades
- Lower timeframe provides precise entry timing

Timeframe hierarchy:
- 4h: Strategic (key levels, major trends)
- 1h: Tactical (trend direction, support/resistance)
- 15m: Operational (trade direction, entry zones)
- 5m: Execution (entry timing, fine-tuning)
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
class TimeframeConfig:
    """Configuration for multi-timeframe analysis."""
    
    # Timeframes to analyze (in minutes)
    timeframes: List[int] = field(default_factory=lambda: [5, 15, 60, 240])
    
    # Timeframe names
    timeframe_names: Dict[int, str] = field(default_factory=lambda: {
        5: "5m",
        15: "15m",
        60: "1h",
        240: "4h",
    })
    
    # Confluence requirements
    min_aligned_timeframes: int = 3  # At least 3 timeframes must agree
    require_higher_tf: bool = True   # Higher TF must not contradict
    
    # Trend detection
    trend_ma_period: int = 20
    trend_lookback: int = 100


@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe."""
    timeframe: int
    timeframe_name: str
    action: str  # "buy", "sell", "hold"
    confidence: float
    trend_direction: str  # "up", "down", "neutral"
    key_levels: List[float]
    timestamp: float


class TimeframeAnalyzer:
    """Analyzes a single timeframe."""
    
    def __init__(self, timeframe: int, name: str, config: TimeframeConfig):
        self.timeframe = timeframe
        self.name = name
        self.config = config
        self._prices: Deque[float] = deque(maxlen=config.trend_lookback)
        self._trend_direction: str = "neutral"
        self._key_levels: List[float] = []
    
    def update(self, prices: List[float]) -> None:
        """Update with new price data."""
        for price in prices:
            self._prices.append(price)
        
        if len(self._prices) >= self.config.trend_ma_period:
            self._detect_trend()
            self._find_key_levels()
    
    def _detect_trend(self) -> None:
        """Detect trend direction using moving average."""
        prices = list(self._prices)
        
        if len(prices) < self.config.trend_ma_period:
            self._trend_direction = "neutral"
            return
        
        ma = np.mean(prices[-self.config.trend_ma_period:])
        current_price = prices[-1]
        
        # Also check slope of MA
        if len(prices) >= self.config.trend_ma_period + 5:
            ma_prev = np.mean(prices[-(self.config.trend_ma_period + 5):-5])
            ma_slope = (ma - ma_prev) / 5
        else:
            ma_slope = 0
        
        # Determine trend
        price_vs_ma = (current_price - ma) / ma
        
        if price_vs_ma > 0.005 and ma_slope > 0:
            self._trend_direction = "up"
        elif price_vs_ma < -0.005 and ma_slope < 0:
            self._trend_direction = "down"
        else:
            self._trend_direction = "neutral"
    
    def _find_key_levels(self) -> None:
        """Find key support/resistance levels."""
        prices = list(self._prices)
        
        if len(prices) < 20:
            self._key_levels = []
            return
        
        # Find local maxima and minima
        levels = []
        
        # Simple pivot detection
        for i in range(2, len(prices) - 2):
            # Resistance (local max)
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                levels.append(prices[i])
            
            # Support (local min)
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                levels.append(prices[i])
        
        # Keep unique levels (within 0.5% of each other)
        unique_levels = []
        for level in levels:
            is_unique = True
            for existing in unique_levels:
                if abs(level - existing) / existing < 0.005:
                    is_unique = False
                    break
            if is_unique:
                unique_levels.append(level)
        
        self._key_levels = sorted(unique_levels)[-5:]  # Keep 5 most recent
    
    def get_signal(self) -> TimeframeSignal:
        """Get signal from this timeframe."""
        # Base action on trend direction
        if self._trend_direction == "up":
            action = "buy"
            confidence = 0.7
        elif self._trend_direction == "down":
            action = "sell"
            confidence = 0.7
        else:
            action = "hold"
            confidence = 0.0
        
        return TimeframeSignal(
            timeframe=self.timeframe,
            timeframe_name=self.name,
            action=action,
            confidence=confidence,
            trend_direction=self._trend_direction,
            key_levels=self._key_levels.copy(),
            timestamp=time.time(),
        )
    
    def get_features(self) -> Dict[str, float]:
        """Get features for this timeframe."""
        trend_encoding = {"up": 1.0, "neutral": 0.0, "down": -1.0}
        
        return {
            f"trend_{self.name}": trend_encoding.get(self._trend_direction, 0.0),
            f"key_levels_{self.name}": float(len(self._key_levels)),
        }


class MultiTimeframeAnalyzer:
    """
    Combines signals from multiple timeframes.
    
    Strategy:
    1. Higher timeframes (4h, 1h) provide trend context
    2. Middle timeframe (15m) provides trade direction
    3. Lower timeframe (5m) provides entry timing
    
    Confluence: All timeframes agreeing = high confidence
    Conflict: Higher TF contradicts = no trade
    """
    
    def __init__(self, config: Optional[TimeframeConfig] = None):
        self.config = config or TimeframeConfig()
        
        # Create analyzers for each timeframe
        self._analyzers: Dict[int, TimeframeAnalyzer] = {}
        for tf in self.config.timeframes:
            name = self.config.timeframe_names.get(tf, f"{tf}m")
            self._analyzers[tf] = TimeframeAnalyzer(tf, name, self.config)
        
        # Statistics
        self._confluence_signals: int = 0
        self._conflict_signals: int = 0
        self._updates: int = 0
    
    def update(self, timeframe_data: Dict[int, List[float]]) -> None:
        """
        Update all timeframes with new data.
        
        Args:
            timeframe_data: Dict of {timeframe_minutes: [prices]}
        """
        self._updates += 1
        
        for tf, prices in timeframe_data.items():
            if tf in self._analyzers:
                self._analyzers[tf].update(prices)
    
    def get_confluence_signal(self) -> Dict[str, Any]:
        """
        Get confluence signal from all timeframes.
        
        Returns:
            - action: "buy", "sell", or "hold"
            - confidence: 0.0 to 1.0
            - aligned_timeframes: number of timeframes that agree
            - conflicting_timeframes: list of conflicting TFs
            - higher_tf_support: whether higher TF supports the trade
        """
        signals = []
        for tf in sorted(self.config.timeframes, reverse=True):  # Higher first
            if tf in self._analyzers:
                signal = self._analyzers[tf].get_signal()
                signals.append(signal)
        
        if not signals:
            return {
                "action": "hold",
                "confidence": 0.0,
                "aligned_timeframes": 0,
                "conflicting_timeframes": [],
                "higher_tf_support": False,
                "reasoning": "No timeframe data"
            }
        
        # Count buy and sell signals
        buy_count = sum(1 for s in signals if s.action == "buy")
        sell_count = sum(1 for s in signals if s.action == "sell")
        
        # Check higher timeframe support
        higher_tfs = [s for s in signals if s.timeframe >= 60]  # 1h and 4h
        higher_tf_aligned = False
        
        if buy_count > sell_count:
            aligned_count = buy_count
            dominant_action = "buy"
            conflicting = [s for s in signals if s.action == "sell"]
            
            # Check if higher TF supports buy
            higher_tf_support = all(s.action in ["buy", "hold"] for s in higher_tfs)
        elif sell_count > buy_count:
            aligned_count = sell_count
            dominant_action = "sell"
            conflicting = [s for s in signals if s.action == "buy"]
            
            # Check if higher TF supports sell
            higher_tf_support = all(s.action in ["sell", "hold"] for s in higher_tfs)
        else:
            return {
                "action": "hold",
                "confidence": 0.0,
                "aligned_timeframes": 0,
                "conflicting_timeframes": [s.timeframe_name for s in signals],
                "higher_tf_support": False,
                "reasoning": "Mixed signals, no clear direction"
            }
        
        # Calculate confidence based on alignment
        alignment_ratio = aligned_count / len(signals)
        
        # Check minimum alignment
        if aligned_count < self.config.min_aligned_timeframes:
            self._conflict_signals += 1
            return {
                "action": "hold",
                "confidence": 0.0,
                "aligned_timeframes": aligned_count,
                "conflicting_timeframes": [s.timeframe_name for s in conflicting],
                "higher_tf_support": higher_tf_support,
                "reasoning": f"Only {aligned_count} timeframes aligned (need {self.config.min_aligned_timeframes})"
            }
        
        # Check higher TF support (if required)
        if self.config.require_higher_tf and not higher_tf_support:
            self._conflict_signals += 1
            return {
                "action": "hold",
                "confidence": 0.0,
                "aligned_timeframes": aligned_count,
                "conflicting_timeframes": [s.timeframe_name for s in conflicting],
                "higher_tf_support": False,
                "reasoning": "Higher timeframe contradicts the signal"
            }
        
        # Calculate final confidence
        base_confidence = alignment_ratio * 0.8
        
        # Bonus for higher TF support
        if higher_tf_support:
            base_confidence += 0.15
        
        # Bonus for all timeframes aligned
        if aligned_count == len(signals):
            base_confidence += 0.05
        
        confidence = min(base_confidence, 0.95)
        
        self._confluence_signals += 1
        
        return {
            "action": dominant_action,
            "confidence": confidence,
            "aligned_timeframes": aligned_count,
            "conflicting_timeframes": [s.timeframe_name for s in conflicting],
            "higher_tf_support": higher_tf_support,
            "reasoning": f"{aligned_count}/{len(signals)} timeframes aligned, higher TF support: {higher_tf_support}",
            "signal_type": "multi_timeframe_confluence",
        }
    
    def get_features(self) -> Dict[str, float]:
        """Get multi-timeframe features for learning."""
        features = {}
        
        for analyzer in self._analyzers.values():
            features.update(analyzer.get_features())
        
        # Add confluence features
        confluence = self.get_confluence_signal()
        
        # Encode confluence
        confluence_encoding = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
        features["mtf_confluence"] = confluence_encoding.get(confluence["action"], 0.0)
        features["mtf_confidence"] = confluence["confidence"]
        features["mtf_alignment"] = confluence["aligned_timeframes"] / max(len(self.config.timeframes), 1)
        features["mtf_higher_tf_support"] = 1.0 if confluence["higher_tf_support"] else 0.0
        
        return features
    
    def get_key_levels(self) -> List[float]:
        """Get combined key levels from all timeframes."""
        all_levels = []
        
        for analyzer in self._analyzers.values():
            signal = analyzer.get_signal()
            all_levels.extend(signal.key_levels)
        
        # Cluster similar levels
        if not all_levels:
            return []
        
        all_levels.sort()
        clustered = []
        
        for level in all_levels:
            is_clustered = False
            for i, cluster_level in enumerate(clustered):
                if abs(level - cluster_level) / cluster_level < 0.005:
                    # Merge into cluster
                    clustered[i] = (clustered[i] + level) / 2
                    is_clustered = True
                    break
            
            if not is_clustered:
                clustered.append(level)
        
        return sorted(clustered)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "updates": self._updates,
            "confluence_signals": self._confluence_signals,
            "conflict_signals": self._conflict_signals,
            "confluence_rate": self._confluence_signals / max(self._updates, 1),
            "timeframes_tracked": len(self._analyzers),
        }


__all__ = [
    "TimeframeConfig",
    "TimeframeSignal",
    "TimeframeAnalyzer",
    "MultiTimeframeAnalyzer",
]
