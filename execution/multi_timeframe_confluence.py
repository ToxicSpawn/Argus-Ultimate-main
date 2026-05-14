"""
Multi-Timeframe Confluence Analyzer for Argus Ultimate.

Analyzes trading signals across multiple timeframes to identify high-probability
setups where multiple timeframes align. Increases win rate by filtering out
false signals and scaling into high-confluence opportunities.

Key Features:
- Parallel analysis across 6+ timeframes
- Weighted confluence scoring
- Trend alignment detection
- Signal strength filtering
- Dynamic position sizing based on confluence
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Standard trading timeframes."""
    M1 = "1m"      # 1 minute
    M5 = "5m"      # 5 minutes
    M15 = "15m"    # 15 minutes
    H1 = "1h"      # 1 hour
    H4 = "4h"      # 4 hours
    D1 = "1d"      # 1 day


class TrendDirection(Enum):
    """Trend direction states."""
    STRONG_UP = 2
    UP = 1
    NEUTRAL = 0
    DOWN = -1
    STRONG_DOWN = -2


class ConfluenceLevel(Enum):
    """Confluence quality levels."""
    ELITE = 5       # All timeframes aligned - maximum confidence
    HIGH = 4        # Strong alignment across major timeframes
    MODERATE = 3    # Good alignment with minor conflicts
    LOW = 2         # Weak alignment, proceed with caution
    NONE = 1        # Conflicting signals - stay out


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: Timeframe
    direction: TrendDirection
    strength: float  # 0.0 to 1.0
    rsi: float
    macd_signal: float  # -1 to 1
    volume_ratio: float  # vs average
    support_distance: float  # % to nearest support
    resistance_distance: float  # % to nearest resistance
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def bullish_score(self) -> float:
        """Calculate bullish bias score."""
        score = 0.0
        if self.direction in (TrendDirection.UP, TrendDirection.STRONG_UP):
            score += 0.3
            if self.direction == TrendDirection.STRONG_UP:
                score += 0.1
        if self.rsi < 30:
            score += 0.2  # Oversold
        elif self.rsi < 50:
            score += 0.1
        if self.macd_signal > 0:
            score += 0.2
        if self.volume_ratio > 1.5:
            score += 0.15
        if self.support_distance < 0.02:  # Near support
            score += 0.05
        return min(score, 1.0)
    
    @property
    def bearish_score(self) -> float:
        """Calculate bearish bias score."""
        score = 0.0
        if self.direction in (TrendDirection.DOWN, TrendDirection.STRONG_DOWN):
            score += 0.3
            if self.direction == TrendDirection.STRONG_DOWN:
                score += 0.1
        if self.rsi > 70:
            score += 0.2  # Overbought
        elif self.rsi > 50:
            score += 0.1
        if self.macd_signal < 0:
            score += 0.2
        if self.volume_ratio > 1.5:
            score += 0.15
        if self.resistance_distance < 0.02:  # Near resistance
            score += 0.05
        return min(score, 1.0)


@dataclass
class ConfluenceResult:
    """Result of multi-timeframe confluence analysis."""
    symbol: str
    timestamp: datetime
    timeframe_signals: Dict[Timeframe, TimeframeSignal]
    
    # Confluence scores
    bullish_confluence: float  # 0.0 to 1.0
    bearish_confluence: float  # 0.0 to 1.0
    confluence_level: ConfluenceLevel
    
    # Trend alignment
    trend_alignment: float  # -1.0 to 1.0 (all bearish to all bullish)
    dominant_trend: TrendDirection
    
    # Signal quality
    signal_strength: float  # Combined signal strength
    confidence: float  # Statistical confidence in signal
    
    # Position sizing recommendation
    recommended_size_multiplier: float  # 0.5 to 2.0
    max_leverage_recommendation: float  # Based on confluence
    
    # Timeframe breakdown
    timeframe_agreement: Dict[str, bool]  # Which timeframes agree
    conflicting_timeframes: List[Timeframe]
    
    @property
    def is_bullish(self) -> bool:
        """Check if overall signal is bullish."""
        return self.bullish_confluence > self.bearish_confluence
    
    @property
    def is_bearish(self) -> bool:
        """Check if overall signal is bearish."""
        return self.bearish_confluence > self.bullish_confluence
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal meets minimum confluence threshold."""
        return self.confluence_level.value >= ConfluenceLevel.MODERATE.value
    
    @property
    def quality_score(self) -> float:
        """Overall quality score for ranking signals."""
        return self.signal_strength * self.confidence * (self.confluence_level.value / 5.0)


class MultiTimeframeConfluence:
    """
    Multi-Timeframe Confluence Analyzer.
    
    Analyzes price action, indicators, and trends across multiple timeframes
    to identify high-probability trading setups where multiple timeframes align.
    """
    
    # Timeframe weights (higher timeframes have more influence)
    TIMEFRAME_WEIGHTS = {
        Timeframe.M1: 0.05,
        Timeframe.M5: 0.10,
        Timeframe.M15: 0.15,
        Timeframe.H1: 0.25,
        Timeframe.H4: 0.30,
        Timeframe.D1: 0.15,
    }
    
    # Minimum confluence thresholds
    CONFLUENCE_THRESHOLDS = {
        ConfluenceLevel.ELITE: 0.85,
        ConfluenceLevel.HIGH: 0.70,
        ConfluenceLevel.MODERATE: 0.55,
        ConfluenceLevel.LOW: 0.40,
        ConfluenceLevel.NONE: 0.0,
    }
    
    def __init__(
        self,
        min_timeframes: int = 4,
        require_higher_tf: bool = True,
        min_confluence: float = 0.55,
        enable_dynamic_sizing: bool = True,
    ):
        """
        Initialize Multi-Timeframe Confluence Analyzer.
        
        Args:
            min_timeframes: Minimum number of timeframes to analyze
            require_higher_tf: Require H4/D1 alignment for high confidence
            min_confluence: Minimum confluence score to generate signal
            enable_dynamic_sizing: Enable position sizing based on confluence
        """
        self.min_timeframes = min_timeframes
        self.require_higher_tf = require_higher_tf
        self.min_confluence = min_confluence
        self.enable_dynamic_sizing = enable_dynamic_sizing
        
        # Signal history for pattern detection
        self.signal_history: Dict[str, deque] = {}
        self.max_history = 100
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_filtered = 0
        self.confluence_stats: Dict[ConfluenceLevel, int] = {
            level: 0 for level in ConfluenceLevel
        }
        
        logger.info(
            f"MultiTimeframeConfluence initialized: "
            f"min_timeframes={min_timeframes}, "
            f"require_higher_tf={require_higher_tf}, "
            f"min_confluence={min_confluence}"
        )
    
    async def analyze(
        self,
        symbol: str,
        timeframe_data: Dict[Timeframe, Dict[str, Any]],
    ) -> ConfluenceResult:
        """
        Analyze multi-timeframe confluence for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe_data: Dict mapping timeframes to OHLCV + indicator data
            
        Returns:
            ConfluenceResult with analysis and recommendations
        """
        timeframe_signals = {}
        
        # Analyze each timeframe
        for tf, data in timeframe_data.items():
            try:
                signal = self._analyze_timeframe(tf, data)
                timeframe_signals[tf] = signal
            except Exception as e:
                logger.warning(f"Error analyzing {tf.value} for {symbol}: {e}")
                continue
        
        # Check minimum timeframes
        if len(timeframe_signals) < self.min_timeframes:
            logger.debug(
                f"Insufficient timeframes for {symbol}: "
                f"{len(timeframe_signals)} < {self.min_timeframes}"
            )
            return self._create_empty_result(symbol, timeframe_signals)
        
        # Calculate confluence scores
        bullish_confluence, bearish_confluence = self._calculate_confluence(
            timeframe_signals
        )
        
        # Determine confluence level
        confluence_level = self._determine_confluence_level(
            bullish_confluence, bearish_confluence
        )
        
        # Calculate trend alignment
        trend_alignment = self._calculate_trend_alignment(timeframe_signals)
        dominant_trend = self._get_dominant_trend(timeframe_signals)
        
        # Calculate signal strength and confidence
        signal_strength = self._calculate_signal_strength(timeframe_signals)
        confidence = self._calculate_confidence(
            timeframe_signals, bullish_confluence, bearish_confluence
        )
        
        # Calculate position sizing
        size_multiplier, leverage_rec = self._calculate_position_sizing(
            confluence_level, signal_strength, confidence
        )
        
        # Analyze timeframe agreement
        agreement, conflicts = self._analyze_timeframe_agreement(
            timeframe_signals, bullish_confluence, bearish_confluence
        )
        
        # Create result
        result = ConfluenceResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            timeframe_signals=timeframe_signals,
            bullish_confluence=bullish_confluence,
            bearish_confluence=bearish_confluence,
            confluence_level=confluence_level,
            trend_alignment=trend_alignment,
            dominant_trend=dominant_trend,
            signal_strength=signal_strength,
            confidence=confidence,
            recommended_size_multiplier=size_multiplier,
            max_leverage_recommendation=leverage_rec,
            timeframe_agreement=agreement,
            conflicting_timeframes=conflicts,
        )
        
        # Update statistics
        self._update_stats(result)
        
        # Store in history
        self._update_history(symbol, result)
        
        return result
    
    def _analyze_timeframe(
        self, tf: Timeframe, data: Dict[str, Any]
    ) -> TimeframeSignal:
        """Analyze a single timeframe's data."""
        # Extract OHLCV data
        close_prices = data.get("close", [])
        high_prices = data.get("high", [])
        low_prices = data.get("low", [])
        volumes = data.get("volume", [])
        
        if not close_prices or len(close_prices) < 20:
            raise ValueError(f"Insufficient data for {tf.value}")
        
        # Calculate indicators
        rsi = self._calculate_rsi(close_prices, period=14)
        macd_signal = self._calculate_macd_signal(close_prices)
        volume_ratio = self._calculate_volume_ratio(volumes)
        
        # Determine trend direction
        direction = self._determine_trend(close_prices, high_prices, low_prices)
        
        # Calculate trend strength
        strength = self._calculate_trend_strength(close_prices, direction)
        
        # Calculate support/resistance distances
        support_dist, resistance_dist = self._calculate_sr_distances(
            close_prices[-1], high_prices, low_prices
        )
        
        return TimeframeSignal(
            timeframe=tf,
            direction=direction,
            strength=strength,
            rsi=rsi,
            macd_signal=macd_signal,
            volume_ratio=volume_ratio,
            support_distance=support_dist,
            resistance_distance=resistance_dist,
        )
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        # Handle edge case: no losses means RSI = 100 (extremely bullish)
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _calculate_macd_signal(self, prices: List[float]) -> float:
        """Calculate MACD signal (-1 to 1)."""
        if len(prices) < 26:
            return 0.0
        
        # Simplified MACD
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        macd = ema12 - ema26
        
        # Normalize to -1 to 1
        price_range = max(prices[-26:]) - min(prices[-26:])
        if price_range > 0:
            return np.clip(macd / price_range, -1.0, 1.0)
        return 0.0
    
    def _calculate_volume_ratio(self, volumes: List[float]) -> float:
        """Calculate volume ratio vs average."""
        if len(volumes) < 20:
            return 1.0
        
        current = volumes[-1] if volumes[-1] > 0 else 1.0
        avg = np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1.0
        
        return current / avg
    
    def _determine_trend(
        self,
        close: List[float],
        high: List[float],
        low: List[float],
    ) -> TrendDirection:
        """Determine trend direction from price action."""
        if len(close) < 20:
            return TrendDirection.NEUTRAL
        
        # Use multiple indicators
        short_ma = np.mean(close[-5:])
        medium_ma = np.mean(close[-15:])
        long_ma = np.mean(close[-20:])
        
        # Price position
        recent_high = max(high[-20:])
        recent_low = min(low[-20:])
        price_position = (close[-1] - recent_low) / (recent_high - recent_low + 1e-10)
        
        # Calculate trend score
        score = 0
        if short_ma > medium_ma:
            score += 1
        if medium_ma > long_ma:
            score += 1
        if close[-1] > short_ma:
            score += 1
        if price_position > 0.6:
            score += 1
        if close[-1] > close[-5]:
            score += 1
        
        # Map score to direction
        if score >= 4:
            return TrendDirection.STRONG_UP
        elif score >= 3:
            return TrendDirection.UP
        elif score <= 1:
            return TrendDirection.STRONG_DOWN
        elif score <= 2:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL
    
    def _calculate_trend_strength(
        self, prices: List[float], direction: TrendDirection
    ) -> float:
        """Calculate trend strength (0 to 1)."""
        if len(prices) < 10:
            return 0.0
        
        # Use ADX-like calculation
        highs = prices  # Simplified
        lows = prices
        
        # Calculate directional movement
        up_moves = []
        down_moves = []
        for i in range(1, min(14, len(prices))):
            up = prices[i] - prices[i-1] if prices[i] > prices[i-1] else 0
            down = prices[i-1] - prices[i] if prices[i-1] > prices[i] else 0
            up_moves.append(up)
            down_moves.append(down)
        
        avg_up = np.mean(up_moves) if up_moves else 0
        avg_down = np.mean(down_moves) if down_moves else 0
        
        # Normalize to 0-1
        total = avg_up + avg_down
        if total > 0:
            if direction in (TrendDirection.UP, TrendDirection.STRONG_UP):
                return min(avg_up / total * 2, 1.0)
            else:
                return min(avg_down / total * 2, 1.0)
        return 0.0
    
    def _calculate_sr_distances(
        self,
        current_price: float,
        highs: List[float],
        lows: List[float],
    ) -> Tuple[float, float]:
        """Calculate distance to nearest support and resistance."""
        # Find recent swing highs and lows
        recent_highs = sorted(set(highs[-20:]))[-3:]  # Top 3 resistance levels
        recent_lows = sorted(set(lows[-20:]))[:3]  # Bottom 3 support levels
        
        # Find nearest resistance above
        resistance_dist = float('inf')
        for level in recent_highs:
            if level > current_price:
                dist = (level - current_price) / current_price
                resistance_dist = min(resistance_dist, dist)
        
        # Find nearest support below
        support_dist = float('inf')
        for level in recent_lows:
            if level < current_price:
                dist = (current_price - level) / current_price
                support_dist = min(support_dist, dist)
        
        # Default to large distance if none found
        if resistance_dist == float('inf'):
            resistance_dist = 0.1  # 10%
        if support_dist == float('inf'):
            support_dist = 0.1
        
        return support_dist, resistance_dist
    
    def _calculate_confluence(
        self, signals: Dict[Timeframe, TimeframeSignal]
    ) -> Tuple[float, float]:
        """Calculate bullish and bearish confluence scores."""
        weighted_bullish = 0.0
        weighted_bearish = 0.0
        total_weight = 0.0
        
        for tf, signal in signals.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.1)
            weighted_bullish += signal.bullish_score * weight
            weighted_bearish += signal.bearish_score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_bullish /= total_weight
            weighted_bearish /= total_weight
        
        return weighted_bullish, weighted_bearish
    
    def _determine_confluence_level(
        self, bullish: float, bearish: float
    ) -> ConfluenceLevel:
        """Determine the confluence level from scores."""
        max_confluence = max(bullish, bearish)
        
        for level in [ConfluenceLevel.ELITE, ConfluenceLevel.HIGH, 
                      ConfluenceLevel.MODERATE, ConfluenceLevel.LOW]:
            if max_confluence >= self.CONFLUENCE_THRESHOLDS[level]:
                return level
        
        return ConfluenceLevel.NONE
    
    def _calculate_trend_alignment(
        self, signals: Dict[Timeframe, TimeframeSignal]
    ) -> float:
        """Calculate how aligned all timeframes are (-1 to 1)."""
        if not signals:
            return 0.0
        
        # Weighted average of trend directions
        weighted_direction = 0.0
        total_weight = 0.0
        
        for tf, signal in signals.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.1)
            weighted_direction += signal.direction.value * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_direction /= total_weight
        
        # Normalize to -1 to 1
        return weighted_direction / 2.0
    
    def _get_dominant_trend(
        self, signals: Dict[Timeframe, TimeframeSignal]
    ) -> TrendDirection:
        """Get the dominant trend across timeframes."""
        trend_votes = {
            TrendDirection.STRONG_UP: 0,
            TrendDirection.UP: 0,
            TrendDirection.NEUTRAL: 0,
            TrendDirection.DOWN: 0,
            TrendDirection.STRONG_DOWN: 0,
        }
        
        for tf, signal in signals.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.1)
            trend_votes[signal.direction] += weight
        
        return max(trend_votes.items(), key=lambda x: x[1])[0]
    
    def _calculate_signal_strength(
        self, signals: Dict[Timeframe, TimeframeSignal]
    ) -> float:
        """Calculate overall signal strength."""
        if not signals:
            return 0.0
        
        strengths = [s.strength for s in signals.values()]
        return float(np.mean(strengths))
    
    def _calculate_confidence(
        self,
        signals: Dict[Timeframe, TimeframeSignal],
        bullish: float,
        bearish: float,
    ) -> float:
        """Calculate statistical confidence in the signal."""
        if not signals:
            return 0.0
        
        # Factor 1: Confluence strength
        confluence_strength = max(bullish, bearish)
        
        # Factor 2: Timeframe agreement
        agreement_ratio = self._calculate_agreement_ratio(signals)
        
        # Factor 3: Higher timeframe alignment
        higher_tf_bonus = 0.0
        if self.require_higher_tf:
            h4_signal = signals.get(Timeframe.H4)
            d1_signal = signals.get(Timeframe.D1)
            if h4_signal and d1_signal:
                if h4_signal.direction == d1_signal.direction:
                    higher_tf_bonus = 0.15
        
        # Combine factors
        confidence = (
            confluence_strength * 0.5 +
            agreement_ratio * 0.35 +
            higher_tf_bonus
        )
        
        return min(confidence, 1.0)
    
    def _calculate_agreement_ratio(
        self, signals: Dict[Timeframe, TimeframeSignal]
    ) -> float:
        """Calculate ratio of timeframes that agree on direction."""
        if len(signals) < 2:
            return 0.5
        
        bullish_count = sum(
            1 for s in signals.values()
            if s.direction in (TrendDirection.UP, TrendDirection.STRONG_UP)
        )
        bearish_count = sum(
            1 for s in signals.values()
            if s.direction in (TrendDirection.DOWN, TrendDirection.STRONG_DOWN)
        )
        
        total = len(signals)
        max_agreement = max(bullish_count, bearish_count)
        
        return max_agreement / total
    
    def _calculate_position_sizing(
        self,
        confluence_level: ConfluenceLevel,
        signal_strength: float,
        confidence: float,
    ) -> Tuple[float, float]:
        """Calculate position sizing multiplier and max leverage."""
        # Base multiplier from confluence level
        multipliers = {
            ConfluenceLevel.ELITE: 1.5,
            ConfluenceLevel.HIGH: 1.25,
            ConfluenceLevel.MODERATE: 1.0,
            ConfluenceLevel.LOW: 0.75,
            ConfluenceLevel.NONE: 0.0,
        }
        
        base_multiplier = multipliers.get(confluence_level, 0.5)
        
        # Adjust by signal strength and confidence
        adjusted_multiplier = base_multiplier * signal_strength * confidence
        
        # Clamp to reasonable range
        size_multiplier = np.clip(adjusted_multiplier, 0.0, 2.0)
        
        # Calculate max leverage recommendation
        leverage_map = {
            ConfluenceLevel.ELITE: 10.0,
            ConfluenceLevel.HIGH: 5.0,
            ConfluenceLevel.MODERATE: 3.0,
            ConfluenceLevel.LOW: 1.5,
            ConfluenceLevel.NONE: 1.0,
        }
        
        max_leverage = leverage_map.get(confluence_level, 1.0)
        max_leverage *= confidence  # Reduce if low confidence
        
        return float(size_multiplier), float(max_leverage)
    
    def _analyze_timeframe_agreement(
        self,
        signals: Dict[Timeframe, TimeframeSignal],
        bullish: float,
        bearish: float,
    ) -> Tuple[Dict[str, bool], List[Timeframe]]:
        """Analyze which timeframes agree and which conflict."""
        agreement = {}
        conflicts = []
        
        # Determine overall direction
        overall_bullish = bullish > bearish
        
        for tf, signal in signals.items():
            tf_bullish = signal.direction in (TrendDirection.UP, TrendDirection.STRONG_UP)
            tf_bearish = signal.direction in (TrendDirection.DOWN, TrendDirection.STRONG_DOWN)
            
            agrees = (overall_bullish and tf_bullish) or (not overall_bullish and tf_bearish)
            agreement[tf.value] = agrees
            
            if not agrees and signal.direction != TrendDirection.NEUTRAL:
                conflicts.append(tf)
        
        return agreement, conflicts
    
    def _create_empty_result(
        self, symbol: str, signals: Dict[Timeframe, TimeframeSignal]
    ) -> ConfluenceResult:
        """Create an empty/neutral result when insufficient data."""
        return ConfluenceResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            timeframe_signals=signals,
            bullish_confluence=0.0,
            bearish_confluence=0.0,
            confluence_level=ConfluenceLevel.NONE,
            trend_alignment=0.0,
            dominant_trend=TrendDirection.NEUTRAL,
            signal_strength=0.0,
            confidence=0.0,
            recommended_size_multiplier=0.0,
            max_leverage_recommendation=1.0,
            timeframe_agreement={},
            conflicting_timeframes=[],
        )
    
    def _update_stats(self, result: ConfluenceResult):
        """Update internal statistics."""
        self.confluence_stats[result.confluence_level] += 1
        
        if result.is_actionable:
            self.signals_generated += 1
        else:
            self.signals_filtered += 1
    
    def _update_history(self, symbol: str, result: ConfluenceResult):
        """Update signal history for pattern detection."""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = deque(maxlen=self.max_history)
        
        self.signal_history[symbol].append({
            "timestamp": result.timestamp,
            "bullish": result.bullish_confluence,
            "bearish": result.bearish_confluence,
            "level": result.confluence_level,
            "actionable": result.is_actionable,
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        total = self.signals_generated + self.signals_filtered
        
        return {
            "total_analyses": total,
            "actionable_signals": self.signals_generated,
            "filtered_signals": self.signals_filtered,
            "action_rate": self.signals_generated / max(total, 1),
            "confluence_distribution": {
                level.name: count
                for level, count in self.confluence_stats.items()
            },
            "tracked_symbols": len(self.signal_history),
        }
    
    def get_signal_history(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Get recent signal history for a symbol."""
        if symbol not in self.signal_history:
            return []
        
        return list(self.signal_history[symbol])[-limit:]
    
    async def batch_analyze(
        self,
        symbols_data: Dict[str, Dict[Timeframe, Dict[str, Any]]],
    ) -> Dict[str, ConfluenceResult]:
        """
        Analyze multiple symbols in parallel.
        
        Args:
            symbols_data: Dict mapping symbols to timeframe data
            
        Returns:
            Dict mapping symbols to ConfluenceResults
        """
        tasks = [
            self.analyze(symbol, tf_data)
            for symbol, tf_data in symbols_data.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for symbol, result in zip(symbols_data.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing {symbol}: {result}")
                output[symbol] = self._create_empty_result(symbol, {})
            else:
                output[symbol] = result
        
        return output
    
    def filter_signals(
        self,
        results: Dict[str, ConfluenceResult],
        min_level: ConfluenceLevel = ConfluenceLevel.MODERATE,
        min_confidence: float = 0.5,
    ) -> Dict[str, ConfluenceResult]:
        """Filter results to only include high-quality signals."""
        filtered = {}
        
        for symbol, result in results.items():
            if (result.confluence_level.value >= min_level.value and
                result.confidence >= min_confidence):
                filtered[symbol] = result
        
        return filtered
    
    def rank_signals(
        self, results: Dict[str, ConfluenceResult]
    ) -> List[Tuple[str, ConfluenceResult]]:
        """Rank signals by quality score."""
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].quality_score,
            reverse=True,
        )
        return ranked


# Factory function for easy integration
def create_confluence_analyzer(
    min_timeframes: int = 4,
    require_higher_tf: bool = True,
    min_confluence: float = 0.55,
) -> MultiTimeframeConfluence:
    """Create a configured Multi-Timeframe Confluence analyzer."""
    return MultiTimeframeConfluence(
        min_timeframes=min_timeframes,
        require_higher_tf=require_higher_tf,
        min_confluence=min_confluence,
        enable_dynamic_sizing=True,
    )
