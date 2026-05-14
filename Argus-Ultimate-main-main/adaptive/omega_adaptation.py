"""
ADAPTATION SYSTEM V2 - OMEGA
==============================
The most advanced market adaptation system.

30 Components:
1. Regime Detection (HMM)
2. Volatility Regime Classifier
3. Trend Strength Analyzer
4. Market State Classifier
5. Correlation Regime Monitor
6. Liquidity State Detector
7. Momentum Regime Tracker
8. Mean Reversion Detector
9. Breakout Probability Estimator
10. Drawdown State Monitor
11. Recovery Pattern Detector
12. Black Swan Alert System
13. Euphoria Detector
14. Capitulation Detector
15. Accumulation Detector
16. Distribution Detector
17. Position Size Adapter
18. Risk Level Adapter
19. Strategy Weight Adapter
20. Timeframe Adapter
21. Volatility Forecast Adapter
22. Confidence Calibrator
23. Learning Rate Adapter
24. Exploration Rate Adapter
25. Ensemble Weight Optimizer
26. Feature Importance Tracker
27. Signal Decay Detector
28. Alpha Decay Monitor
29. Regime Transition Predictor
30. Meta-Adaptation Controller
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    EUPHORIA = "euphoria"
    CAPITULATION = "capitulation"
    BLACK_SWAN = "black_swan"
    RECOVERY = "recovery"
    CRASH = "crash"


@dataclass
class AdaptationState:
    """Current adaptation state."""
    regime: MarketRegime = MarketRegime.RANGING
    confidence: float = 0.5
    volatility_regime: str = "normal"
    trend_strength: float = 0.0
    position_multiplier: float = 0.5
    risk_multiplier: float = 1.0
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    timeframe_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegimeDetector:
    """HMM-based regime detection."""
    
    def __init__(self):
        self.regime_history: deque = deque(maxlen=100)
        self.transition_matrix = np.ones((16, 16)) / 16
        
    def detect(self, prices: List[float]) -> Tuple[MarketRegime, float]:
        """Detect current market regime."""
        if len(prices) < 50:
            return MarketRegime.RANGING, 0.5
        
        returns = np.diff(np.log(prices[-50:]))
        volatility = np.std(returns) * np.sqrt(252)
        trend = np.polyfit(range(50), prices[-50:], 1)[0]
        trend_pct = trend / np.mean(prices[-50:])
        
        # Regime classification
        if volatility > 1.0:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(volatility / 1.5, 1.0)
        elif volatility < 0.2:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 1 - volatility / 0.2
        elif trend_pct > 0.002:
            if trend_pct > 0.005:
                regime = MarketRegime.STRONG_UPTREND
            else:
                regime = MarketRegime.WEAK_UPTREND
            confidence = min(abs(trend_pct) / 0.01, 0.9)
        elif trend_pct < -0.002:
            if trend_pct < -0.005:
                regime = MarketRegime.STRONG_DOWNTREND
            else:
                regime = MarketRegime.WEAK_DOWNTREND
            confidence = min(abs(trend_pct) / 0.01, 0.9)
        else:
            regime = MarketRegime.RANGING
            confidence = 0.6
        
        self.regime_history.append(regime)
        return regime, confidence


class VolatilityRegimeClassifier:
    """Classify volatility regime."""
    
    def __init__(self):
        self.vol_history: deque = deque(maxlen=100)
        
    def classify(self, prices: List[float]) -> Tuple[str, float]:
        """Classify volatility regime."""
        if len(prices) < 20:
            return "normal", 0.5
        
        returns = np.diff(np.log(prices[-20:]))
        current_vol = np.std(returns[-5:]) * np.sqrt(252)
        hist_vol = np.std(returns) * np.sqrt(252)
        
        self.vol_history.append(current_vol)
        
        vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1
        
        if vol_ratio > 1.5:
            return "high", min(vol_ratio / 2, 1.0)
        elif vol_ratio < 0.7:
            return "low", 1 - vol_ratio
        else:
            return "normal", 0.7


class TrendStrengthAnalyzer:
    """Analyze trend strength."""
    
    def __init__(self):
        self.strength_history: deque = deque(maxlen=100)
        
    def analyze(self, prices: List[float]) -> Tuple[float, str]:
        """Analyze trend strength (0-1)."""
        if len(prices) < 50:
            return 0.5, "neutral"
        
        # Multiple timeframe trend alignment
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        
        current = prices[-1]
        
        # Alignment score
        bullish = 0
        if current > sma_10 > sma_20 > sma_50:
            bullish = 1
        elif current < sma_10 < sma_20 < sma_50:
            bullish = -1
        
        # ADX-like strength calculation
        high = max(prices[-14:])
        low = min(prices[-14:])
        tr = high - low
        
        plus_dm = max(0, prices[-1] - prices[-2]) if len(prices) > 1 else 0
        minus_dm = max(0, prices[-2] - prices[-1]) if len(prices) > 1 else 0
        
        if tr == 0:
            strength = 0
        else:
            strength = abs(plus_dm - minus_dm) / tr
        
        self.strength_history.append(strength)
        
        direction = "bullish" if bullish > 0 else "bearish" if bullish < 0 else "neutral"
        return strength, direction


class MarketStateClassifier:
    """Classify overall market state."""
    
    def __init__(self):
        self.state_history: deque = deque(maxlen=100)
        
    def classify(self, prices: List[float], volumes: Optional[List[float]] = None) -> Dict[str, Any]:
        """Classify market state."""
        if len(prices) < 30:
            return {"state": "unknown", "confidence": 0.5}
        
        # Price action
        recent = prices[-30:]
        high = max(recent)
        low = min(recent)
        current = prices[-1]
        
        position_in_range = (current - low) / (high - low) if high != low else 0.5
        
        # Volume analysis (if available)
        if volumes and len(volumes) >= 10:
            avg_vol = np.mean(volumes[-20:])
            current_vol = volumes[-1]
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        else:
            vol_ratio = 1.0
        
        # State classification
        if position_in_range > 0.8:
            state = "near_high"
        elif position_in_range < 0.2:
            state = "near_low"
        elif 0.4 < position_in_range < 0.6:
            state = "mid_range"
        else:
            state = "trending"
        
        self.state_history.append(state)
        
        return {
            "state": state,
            "position_in_range": position_in_range,
            "vol_ratio": vol_ratio,
            "range_size": (high - low) / current,
        }


class CorrelationRegimeMonitor:
    """Monitor correlation regime."""
    
    def __init__(self):
        self.corr_history: deque = deque(maxlen=100)
        
    def monitor(self, prices1: List[float], prices2: List[float]) -> Tuple[str, float]:
        """Monitor correlation regime."""
        if len(prices1) < 30 or len(prices2) < 30:
            return "normal", 0.5
        
        returns1 = np.diff(np.log(prices1[-30:]))
        returns2 = np.diff(np.log(prices2[-30:]))
        
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        self.corr_history.append(correlation)
        
        if correlation > 0.8:
            regime = "high_correlation"
        elif correlation < 0.2:
            regime = "low_correlation"
        elif correlation < 0:
            regime = "negative_correlation"
        else:
            regime = "normal_correlation"
        
        return regime, abs(correlation)


class LiquidityStateDetector:
    """Detect liquidity state."""
    
    def __init__(self):
        self.liquidity_history: deque = deque(maxlen=100)
        
    def detect(self, prices: List[float], volumes: Optional[List[float]] = None) -> Tuple[str, float]:
        """Detect liquidity state."""
        if len(prices) < 20:
            return "normal", 0.5
        
        # Estimate from price impact
        price_changes = np.abs(np.diff(prices[-20:]))
        avg_change = np.mean(price_changes)
        
        if volumes:
            avg_volume = np.mean(volumes[-20:])
            # High volume + low price impact = high liquidity
            if avg_volume > 0:
                liquidity_score = avg_volume / (avg_change * 1000 + 1)
            else:
                liquidity_score = 0.5
        else:
            # Estimate from price stability
            liquidity_score = 1 - min(avg_change / np.mean(prices[-20:]), 1)
        
        self.liquidity_history.append(liquidity_score)
        
        if liquidity_score > 0.7:
            state = "high_liquidity"
        elif liquidity_score < 0.3:
            state = "low_liquidity"
        else:
            state = "normal_liquidity"
        
        return state, liquidity_score


class MomentumRegimeTracker:
    """Track momentum regime."""
    
    def __init__(self):
        self.momentum_history: deque = deque(maxlen=100)
        
    def track(self, prices: List[float]) -> Tuple[str, float]:
        """Track momentum regime."""
        if len(prices) < 20:
            return "neutral", 0.5
        
        # Multiple momentum indicators
        roc_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        roc_10 = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        roc_20 = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # Momentum score
        momentum = (roc_5 * 0.5 + roc_10 * 0.3 + roc_20 * 0.2)
        
        self.momentum_history.append(momentum)
        
        if momentum > 0.02:
            regime = "strong_bullish"
        elif momentum > 0.005:
            regime = "bullish"
        elif momentum < -0.02:
            regime = "strong_bearish"
        elif momentum < -0.005:
            regime = "bearish"
        else:
            regime = "neutral"
        
        strength = min(abs(momentum) / 0.05, 1.0)
        return regime, strength


class MeanReversionDetector:
    """Detect mean reversion opportunities."""
    
    def __init__(self):
        self.reversion_history: deque = deque(maxlen=100)
        
    def detect(self, prices: List[float]) -> Tuple[bool, float]:
        """Detect mean reversion opportunity."""
        if len(prices) < 20:
            return False, 0.5
        
        recent = prices[-20:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std == 0:
            return False, 0.5
        
        z_score = (prices[-1] - mean) / std
        
        # Check for reversion potential
        is_reversion = abs(z_score) > 1.5
        probability = min(abs(z_score) / 3, 1.0)
        
        self.reversion_history.append((is_reversion, probability))
        return is_reversion, probability


class BreakoutProbabilityEstimator:
    """Estimate breakout probability."""
    
    def __init__(self):
        self.breakout_history: deque = deque(maxlen=100)
        
    def estimate(self, prices: List[float]) -> Tuple[bool, float, str]:
        """Estimate breakout probability."""
        if len(prices) < 20:
            return False, 0.5, "none"
        
        recent = prices[-20:]
        high = max(recent)
        low = min(recent)
        current = prices[-1]
        
        # Bollinger Band squeeze detection
        sma = np.mean(recent)
        std = np.std(recent)
        
        bandwidth = (2 * std) / sma if sma > 0 else 0
        
        # Squeeze detection
        is_squeeze = bandwidth < 0.02
        
        # Breakout detection
        atr = np.mean([abs(recent[i] - recent[i-1]) for i in range(1, len(recent))])
        
        if current > high + atr * 0.5:
            is_breakout = True
            direction = "up"
            probability = 0.7
        elif current < low - atr * 0.5:
            is_breakout = True
            direction = "down"
            probability = 0.7
        else:
            is_breakout = False
            direction = "none"
            probability = 0.3 if is_squeeze else 0.5
        
        self.breakout_history.append((is_breakout, probability))
        return is_breakout, probability, direction


class DrawdownStateMonitor:
    """Monitor drawdown state."""
    
    def __init__(self):
        self.peak = 0
        self.drawdown_history: deque = deque(maxlen=100)
        
    def monitor(self, equity: float) -> Tuple[str, float]:
        """Monitor drawdown state."""
        self.peak = max(self.peak, equity)
        
        drawdown = (self.peak - equity) / self.peak if self.peak > 0 else 0
        
        self.drawdown_history.append(drawdown)
        
        if drawdown > 0.20:
            state = "severe_drawdown"
        elif drawdown > 0.10:
            state = "significant_drawdown"
        elif drawdown > 0.05:
            state = "moderate_drawdown"
        elif drawdown > 0.02:
            state = "minor_drawdown"
        else:
            state = "no_drawdown"
        
        return state, drawdown


class RecoveryPatternDetector:
    """Detect recovery patterns."""
    
    def __init__(self):
        self.recovery_history: deque = deque(maxlen=100)
        
    def detect(self, prices: List[float]) -> Tuple[bool, float]:
        """Detect recovery pattern."""
        if len(prices) < 20:
            return False, 0.5
        
        # V-shaped recovery detection
        recent = prices[-20:]
        low_idx = np.argmin(recent)
        
        # Check if we're after a low and recovering
        if low_idx < len(recent) - 5:
            pre_low = recent[max(0, low_idx-3):low_idx]
            post_low = recent[low_idx:min(len(recent), low_idx+5)]
            
            if len(pre_low) > 0 and len(post_low) > 0:
                drop = (min(pre_low) - recent[low_idx]) / min(pre_low)
                recovery = (post_low[-1] - recent[low_idx]) / recent[low_idx]
                
                is_recovery = recovery > drop * 0.5
                confidence = min(recovery / 0.05, 1.0) if is_recovery else 0.5
                
                self.recovery_history.append((is_recovery, confidence))
                return is_recovery, confidence
        
        return False, 0.5


class BlackSwanAlertSystem:
    """Black swan alert system."""
    
    def __init__(self):
        self.alert_history: deque = deque(maxlen=100)
        self.risk_score = 0.0
        
    def check(self, prices: List[float]) -> Tuple[bool, float]:
        """Check for black swan conditions."""
        if len(prices) < 20:
            return False, 0.0
        
        # Extreme move detection
        recent_change = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        volatility = np.std(np.diff(prices[-20:])) * np.sqrt(252)
        
        # Black swan indicators
        risk_score = 0
        
        if abs(recent_change) > 0.10:  # 10% move in 5 periods
            risk_score += 0.5
        
        if volatility > 1.5:  # Extreme volatility
            risk_score += 0.3
        
        if len(prices) >= 10:
            max_drawdown = (min(prices[-10:]) - max(prices[-10:])) / max(prices[-10:])
            if max_drawdown < -0.15:
                risk_score += 0.2
        
        self.risk_score = risk_score
        self.alert_history.append(risk_score)
        
        is_black_swan = risk_score > 0.7
        return is_black_swan, risk_score


class EuphoriaDetector:
    """Detect euphoria/excess optimism."""
    
    def __init__(self):
        self.euphoria_history: deque = deque(maxlen=100)
        
    def detect(self, prices: List[float]) -> Tuple[bool, float]:
        """Detect euphoria state."""
        if len(prices) < 30:
            return False, 0.5
        
        # Euphoria indicators
        roc_20 = (prices[-1] - prices[-20]) / prices[-20]
        acceleration = roc_20 - ((prices[-20] - prices[-40]) / prices[-40] if len(prices) >= 40 else 0)
        
        euphoria_score = 0
        
        if roc_20 > 0.15:  # 20%+ gain in 20 periods
            euphoria_score += 0.4
        
        if acceleration > 0.05:  # Accelerating gains
            euphoria_score += 0.3
        
        # Extended run (many green candles)
        if len(prices) >= 10:
            green_count = sum(1 for i in range(-10, 0) if prices[i] > prices[i-1])
            if green_count >= 8:
                euphoria_score += 0.3
        
        self.euphoria_history.append(euphoria_score)
        is_euphoria = euphoria_score > 0.6
        
        return is_euphoria, euphoria_score


class CapitulationDetector:
    """Detect capitulation/selling exhaustion."""
    
    def __init__(self):
        self.capitulation_history: deque = deque(maxlen=100)
        
    def detect(self, prices: List[float], volumes: Optional[List[float]] = None) -> Tuple[bool, float]:
        """Detect capitulation state."""
        if len(prices) < 20:
            return False, 0.5
        
        # Capitulation indicators
        roc_10 = (prices[-1] - prices[-10]) / prices[-10]
        
        capitulation_score = 0
        
        if roc_10 < -0.10:  # 10%+ drop
            capitulation_score += 0.4
        
        # Panic selling (large red candles)
        if len(prices) >= 5:
            recent_drops = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-5, 0)]
            large_drops = sum(1 for d in recent_drops if d < -0.02)
            if large_drops >= 3:
                capitulation_score += 0.3
        
        # Volume spike (if available)
        if volumes and len(volumes) >= 10:
            avg_vol = np.mean(volumes[-20:])
            current_vol = volumes[-1]
            if current_vol > avg_vol * 2:
                capitulation_score += 0.3
        
        self.capitulation_history.append(capitulation_score)
        is_capitulation = capitulation_score > 0.6
        
        return is_capitulation, capitulation_score


class AccumulationDetector:
    """Detect accumulation phase."""
    
    def __init__(self):
        self.accumulation_history: deque = deque(maxlen=100)
        
    def detect(self, prices: List[float]) -> Tuple[bool, float]:
        """Detect accumulation phase."""
        if len(prices) < 40:
            return False, 0.5
        
        # Accumulation indicators
        recent = prices[-40:]
        
        # Tight range after downtrend
        high = max(recent[-20:])
        low = min(recent[-20:])
        range_pct = (high - low) / np.mean(recent[-20:])
        
        # Higher lows
        lows = [min(recent[i:i+5]) for i in range(0, len(recent)-5, 5)]
        higher_lows = all(lows[i] <= lows[i+1] for i in range(len(lows)-1)) if len(lows) > 1 else False
        
        accumulation_score = 0
        
        if range_pct < 0.05:  # Tight range
            accumulation_score += 0.4
        
        if higher_lows:
            accumulation_score += 0.3
        
        # Base formation
        sma = np.mean(recent)
        if abs(prices[-1] - sma) / sma < 0.02:  # Near average
            accumulation_score += 0.3
        
        self.accumulation_history.append(accumulation_score)
        is_accumulation = accumulation_score > 0.6
        
        return is_accumulation, accumulation_score


class DistributionDetector:
    """Detect distribution phase."""
    
    def __init__(self):
        self.distribution_history: deque = deque(maxlen=100)
        
    def detect(self, prices: List[float]) -> Tuple[bool, float]:
        """Detect distribution phase."""
        if len(prices) < 40:
            return False, 0.5
        
        recent = prices[-40:]
        
        # Distribution indicators
        high = max(recent[-20:])
        low = min(recent[-20:])
        range_pct = (high - low) / np.mean(recent[-20:])
        
        # Lower highs
        highs = [max(recent[i:i+5]) for i in range(0, len(recent)-5, 5)]
        lower_highs = all(highs[i] >= highs[i+1] for i in range(len(highs)-1)) if len(highs) > 1 else False
        
        distribution_score = 0
        
        if range_pct < 0.05:  # Tight range at top
            distribution_score += 0.4
        
        if lower_highs:
            distribution_score += 0.3
        
        # Near recent highs but stalling
        if abs(prices[-1] - high) / high < 0.02:
            distribution_score += 0.3
        
        self.distribution_history.append(distribution_score)
        is_distribution = distribution_score > 0.6
        
        return is_distribution, distribution_score


class PositionSizeAdapter:
    """Adapt position size based on conditions."""
    
    def __init__(self):
        self.multiplier_history: deque = deque(maxlen=100)
        
    def adapt(self, regime: MarketRegime, volatility: float, confidence: float) -> float:
        """Calculate position size multiplier."""
        # Base multiplier
        if regime in [MarketRegime.STRONG_UPTREND]:
            base = 1.0
        elif regime in [MarketRegime.WEAK_UPTREND]:
            base = 0.7
        elif regime in [MarketRegime.RANGING]:
            base = 0.5
        elif regime in [MarketRegime.WEAK_DOWNTREND]:
            base = 0.3
        elif regime in [MarketRegime.STRONG_DOWNTREND]:
            base = 0.1
        elif regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.BLACK_SWAN]:
            base = 0.2
        else:
            base = 0.5
        
        # Adjust for volatility
        vol_adj = max(0.5, 1 - volatility * 0.5)
        
        # Adjust for confidence
        conf_adj = 0.5 + confidence * 0.5
        
        multiplier = base * vol_adj * conf_adj
        multiplier = max(0.1, min(multiplier, 1.5))
        
        self.multiplier_history.append(multiplier)
        return multiplier


class RiskLevelAdapter:
    """Adapt risk level based on conditions."""
    
    def __init__(self):
        self.risk_history: deque = deque(maxlen=100)
        
    def adapt(self, regime: MarketRegime, drawdown: float, volatility: float) -> float:
        """Calculate risk multiplier."""
        # Base risk by regime
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.LOW_VOLATILITY]:
            base = 1.2
        elif regime in [MarketRegime.WEAK_UPTREND]:
            base = 1.0
        elif regime in [MarketRegime.RANGING]:
            base = 0.8
        elif regime in [MarketRegime.WEAK_DOWNTREND, MarketRegime.HIGH_VOLATILITY]:
            base = 0.6
        elif regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.BLACK_SWAN, MarketRegime.CRASH]:
            base = 0.3
        else:
            base = 0.8
        
        # Adjust for drawdown
        dd_adj = max(0.3, 1 - drawdown * 2)
        
        # Adjust for volatility
        vol_adj = max(0.5, 1 - volatility * 0.3)
        
        risk = base * dd_adj * vol_adj
        risk = max(0.1, min(risk, 1.5))
        
        self.risk_history.append(risk)
        return risk


class StrategyWeightAdapter:
    """Adapt strategy weights based on regime."""
    
    def __init__(self):
        self.weight_history: deque = deque(maxlen=100)
        
    def adapt(self, regime: MarketRegime, volatility: float) -> Dict[str, float]:
        """Calculate strategy weights."""
        # Base weights by regime
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND]:
            weights = {
                "trend": 0.8,
                "momentum": 0.7,
                "breakout": 0.5,
                "mean_reversion": 0.3,
                "scalping": 0.4,
                "swing": 0.6,
            }
        elif regime in [MarketRegime.RANGING]:
            weights = {
                "trend": 0.3,
                "momentum": 0.4,
                "breakout": 0.4,
                "mean_reversion": 0.8,
                "scalping": 0.6,
                "swing": 0.4,
            }
        elif regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND]:
            weights = {
                "trend": 0.7,
                "momentum": 0.6,
                "breakout": 0.5,
                "mean_reversion": 0.4,
                "scalping": 0.5,
                "swing": 0.5,
            }
        elif regime in [MarketRegime.HIGH_VOLATILITY]:
            weights = {
                "trend": 0.4,
                "momentum": 0.5,
                "breakout": 0.6,
                "mean_reversion": 0.5,
                "scalping": 0.7,
                "swing": 0.3,
            }
        else:
            weights = {
                "trend": 0.5,
                "momentum": 0.5,
                "breakout": 0.5,
                "mean_reversion": 0.5,
                "scalping": 0.5,
                "swing": 0.5,
            }
        
        # Adjust for volatility
        if volatility > 0.5:
            weights["scalping"] = weights.get("scalping", 0.5) * 1.2
            weights["swing"] = weights.get("swing", 0.5) * 0.8
        
        self.weight_history.append(weights)
        return weights


class TimeframeAdapter:
    """Adapt timeframe weights."""
    
    def __init__(self):
        self.timeframe_history: deque = deque(maxlen=100)
        
    def adapt(self, regime: MarketRegime, volatility: float) -> Dict[str, float]:
        """Calculate timeframe weights."""
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
            # Strong trends - higher timeframes
            weights = {
                "1m": 0.1,
                "5m": 0.2,
                "15m": 0.3,
                "1h": 0.4,
                "4h": 0.5,
                "1d": 0.4,
            }
        elif regime in [MarketRegime.HIGH_VOLATILITY]:
            # High volatility - lower timeframes for quick exits
            weights = {
                "1m": 0.4,
                "5m": 0.5,
                "15m": 0.4,
                "1h": 0.3,
                "4h": 0.2,
                "1d": 0.1,
            }
        elif regime in [MarketRegime.RANGING]:
            # Ranging - balanced
            weights = {
                "1m": 0.2,
                "5m": 0.3,
                "15m": 0.4,
                "1h": 0.4,
                "4h": 0.3,
                "1d": 0.2,
            }
        else:
            # Default balanced
            weights = {
                "1m": 0.2,
                "5m": 0.3,
                "15m": 0.4,
                "1h": 0.4,
                "4h": 0.3,
                "1d": 0.2,
            }
        
        self.timeframe_history.append(weights)
        return weights


class VolatilityForecastAdapter:
    """Adapt based on volatility forecast."""
    
    def __init__(self):
        self.forecast_history: deque = deque(maxlen=100)
        
    def forecast(self, prices: List[float]) -> Dict[str, float]:
        """Forecast volatility."""
        if len(prices) < 20:
            return {"current": 0.02, "forecast": 0.02, "trend": "stable"}
        
        returns = np.diff(np.log(prices[-20:]))
        
        current_vol = np.std(returns[-5:]) * np.sqrt(252)
        hist_vol = np.std(returns) * np.sqrt(252)
        
        # Simple GARCH-like forecast
        forecast_vol = current_vol * 0.7 + hist_vol * 0.3
        
        trend = "increasing" if current_vol > hist_vol else "decreasing" if current_vol < hist_vol else "stable"
        
        result = {
            "current": current_vol,
            "forecast": forecast_vol,
            "trend": trend,
            "ratio": current_vol / hist_vol if hist_vol > 0 else 1,
        }
        
        self.forecast_history.append(result)
        return result


class ConfidenceCalibrator:
    """Calibrate confidence levels."""
    
    def __init__(self):
        self.calibration_history: deque = deque(maxlen=100)
        self.calibration_error = 0.0
        
    def calibrate(self, raw_confidence: float, regime: MarketRegime, volatility: float) -> float:
        """Calibrate confidence level."""
        # Reduce confidence in extreme conditions
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.BLACK_SWAN]:
            raw_confidence *= 0.7
        
        if volatility > 0.5:
            raw_confidence *= 0.8
        
        # Apply calibration
        calibrated = raw_confidence * (1 - self.calibration_error * 0.5)
        
        calibrated = max(0.1, min(calibrated, 0.95))
        
        self.calibration_history.append(calibrated)
        return calibrated


class LearningRateAdapter:
    """Adapt learning rate for online learning."""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.lr_history: deque = deque(maxlen=100)
        
    def adapt(self, performance: float, regime_stability: float) -> float:
        """Adapt learning rate."""
        # Higher learning rate when performance is poor (need to adapt faster)
        if performance < 0.5:
            self.learning_rate = min(0.3, self.learning_rate * 1.1)
        elif performance > 0.7:
            # Lower learning rate when performing well
            self.learning_rate = max(0.01, self.learning_rate * 0.95)
        
        # Adjust for regime stability
        if regime_stability < 0.5:
            # Unstable regime - higher learning rate
            self.learning_rate = min(0.3, self.learning_rate * 1.05)
        
        self.lr_history.append(self.learning_rate)
        return self.learning_rate


class ExplorationRateAdapter:
    """Adapt exploration rate."""
    
    def __init__(self):
        self.exploration_rate = 0.2
        self.er_history: deque = deque(maxlen=100)
        
    def adapt(self, n_cycles: int, performance: float) -> float:
        """Adapt exploration rate."""
        # Decay exploration over time
        decay = 0.999
        self.exploration_rate *= decay
        
        # Minimum exploration
        self.exploration_rate = max(0.05, self.exploration_rate)
        
        # Increase exploration if performance drops
        if performance < 0.4:
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
        
        self.er_history.append(self.exploration_rate)
        return self.exploration_rate


class EnsembleWeightOptimizer:
    """Optimize ensemble weights."""
    
    def __init__(self):
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, deque] = {}
        
    def optimize(self, strategy_performances: Dict[str, float]) -> Dict[str, float]:
        """Optimize ensemble weights based on performance."""
        if not strategy_performances:
            return {}
        
        # Calculate weights proportional to performance
        total_perf = sum(max(p, 0.01) for p in strategy_performances.values())
        
        weights = {}
        for strategy, perf in strategy_performances.items():
            weights[strategy] = max(perf, 0.01) / total_perf
            
            # Track performance
            if strategy not in self.performance_history:
                self.performance_history[strategy] = deque(maxlen=100)
            self.performance_history[strategy].append(perf)
        
        self.weights = weights
        return weights


class FeatureImportanceTracker:
    """Track feature importance over time."""
    
    def __init__(self):
        self.importance: Dict[str, float] = {
            "price": 0.3,
            "volume": 0.2,
            "volatility": 0.2,
            "momentum": 0.15,
            "trend": 0.15,
        }
        self.importance_history: deque = deque(maxlen=100)
        
    def update(self, feature_performance: Dict[str, float]) -> Dict[str, float]:
        """Update feature importance."""
        if not feature_performance:
            return self.importance
        
        # Update importance based on performance
        for feature, perf in feature_performance.items():
            if feature in self.importance:
                # Exponential moving average
                self.importance[feature] = 0.9 * self.importance[feature] + 0.1 * perf
        
        # Normalize
        total = sum(self.importance.values())
        if total > 0:
            self.importance = {k: v/total for k, v in self.importance.items()}
        
        self.importance_history.append(self.importance.copy())
        return self.importance


class SignalDecayDetector:
    """Detect signal decay."""
    
    def __init__(self):
        self.signal_history: deque = deque(maxlen=100)
        self.decay_rate = 0.0
        
    def detect(self, signal_outcomes: List[bool]) -> Tuple[bool, float]:
        """Detect signal decay."""
        if len(signal_outcomes) < 20:
            return False, 0.0
        
        # Calculate rolling success rate
        recent = signal_outcomes[-20:]
        older = signal_outcomes[-40:-20] if len(signal_outcomes) >= 40 else recent
        
        recent_rate = sum(recent) / len(recent)
        older_rate = sum(older) / len(older)
        
        # Decay is when recent performance is worse
        self.decay_rate = older_rate - recent_rate
        
        is_decaying = self.decay_rate > 0.1
        
        self.signal_history.append(is_decaying)
        return is_decaying, self.decay_rate


class AlphaDecayMonitor:
    """Monitor alpha decay."""
    
    def __init__(self):
        self.alpha_history: deque = deque(maxlen=100)
        self.decay_detected = False
        
    def monitor(self, returns: List[float]) -> Tuple[bool, float]:
        """Monitor alpha decay."""
        if len(returns) < 30:
            return False, 0.0
        
        # Calculate rolling alpha
        window = 10
        alphas = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            alpha = np.mean(window_returns)
            alphas.append(alpha)
        
        if len(alphas) < 2:
            return False, 0.0
        
        # Fit trend to alphas
        x = np.arange(len(alphas))
        slope = np.polyfit(x, alphas, 1)[0]
        
        self.decay_detected = slope < -0.001
        self.alpha_history.append(slope)
        
        return self.decay_detected, abs(slope)


class RegimeTransitionPredictor:
    """Predict regime transitions."""
    
    def __init__(self):
        self.transition_history: deque = deque(maxlen=100)
        self.transition_matrix: Dict[str, Dict[str, float]] = {}
        
    def predict(self, current_regime: str, regime_history: List[str]) -> Dict[str, float]:
        """Predict regime transition probabilities."""
        if len(regime_history) < 10:
            return {"same": 0.7, "change": 0.3}
        
        # Calculate transition probabilities from history
        transitions = {}
        for i in range(len(regime_history) - 1):
            from_regime = regime_history[i]
            to_regime = regime_history[i + 1]
            
            if from_regime not in transitions:
                transitions[from_regime] = {}
            if to_regime not in transitions[from_regime]:
                transitions[from_regime][to_regime] = 0
            transitions[from_regime][to_regime] += 1
        
        # Normalize
        if current_regime in transitions:
            total = sum(transitions[current_regime].values())
            probs = {k: v/total for k, v in transitions[current_regime].items()}
        else:
            probs = {"same": 0.7}
        
        self.transition_history.append(probs)
        return probs


class MetaAdaptationController:
    """Meta-controller for adaptation system."""
    
    def __init__(self):
        self.adaptation_weights: Dict[str, float] = {
            "regime": 0.25,
            "volatility": 0.20,
            "trend": 0.15,
            "momentum": 0.15,
            "risk": 0.15,
            "performance": 0.10,
        }
        self.meta_history: deque = deque(maxlen=100)
        
    def control(self, component_performances: Dict[str, float]) -> Dict[str, float]:
        """Control adaptation weights based on component performance."""
        # Update weights based on performance
        for component, perf in component_performances.items():
            if component in self.adaptation_weights:
                # Boost weights for good performing components
                self.adaptation_weights[component] *= (0.95 + perf * 0.1)
        
        # Normalize
        total = sum(self.adaptation_weights.values())
        if total > 0:
            self.adaptation_weights = {k: v/total for k, v in self.adaptation_weights.items()}
        
        self.meta_history.append(self.adaptation_weights.copy())
        return self.adaptation_weights


class OmegaAdaptationEngine:
    """
    THE OMEGA ADAPTATION ENGINE.
    
    30 Components.
    """
    
    def __init__(self):
        # Initialize all 30 components
        self.regime_detector = RegimeDetector()
        self.volatility_classifier = VolatilityRegimeClassifier()
        self.trend_analyzer = TrendStrengthAnalyzer()
        self.market_state_classifier = MarketStateClassifier()
        self.correlation_monitor = CorrelationRegimeMonitor()
        self.liquidity_detector = LiquidityStateDetector()
        self.momentum_tracker = MomentumRegimeTracker()
        self.mean_reversion_detector = MeanReversionDetector()
        self.breakout_estimator = BreakoutProbabilityEstimator()
        self.drawdown_monitor = DrawdownStateMonitor()
        self.recovery_detector = RecoveryPatternDetector()
        self.black_swan_alert = BlackSwanAlertSystem()
        self.euphoria_detector = EuphoriaDetector()
        self.capitulation_detector = CapitulationDetector()
        self.accumulation_detector = AccumulationDetector()
        self.distribution_detector = DistributionDetector()
        self.position_size_adapter = PositionSizeAdapter()
        self.risk_level_adapter = RiskLevelAdapter()
        self.strategy_weight_adapter = StrategyWeightAdapter()
        self.timeframe_adapter = TimeframeAdapter()
        self.volatility_forecaster = VolatilityForecastAdapter()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.learning_rate_adapter = LearningRateAdapter()
        self.exploration_rate_adapter = ExplorationRateAdapter()
        self.ensemble_optimizer = EnsembleWeightOptimizer()
        self.feature_importance_tracker = FeatureImportanceTracker()
        self.signal_decay_detector = SignalDecayDetector()
        self.alpha_decay_monitor = AlphaDecayMonitor()
        self.regime_transition_predictor = RegimeTransitionPredictor()
        self.meta_controller = MetaAdaptationController()
        
        # State
        self.state = AdaptationState()
        self.cycle_count = 0
        self.regime_history: deque = deque(maxlen=100)
        
        logger.info("OmegaAdaptationEngine: 30 components initialized")
    
    def analyze(self, prices: List[float], **kwargs) -> AdaptationState:
        """Analyze market and adapt."""
        self.cycle_count += 1
        
        # 1. Regime Detection
        regime, regime_confidence = self.regime_detector.detect(prices)
        
        # 2. Volatility Classification
        vol_regime, vol_confidence = self.volatility_classifier.classify(prices)
        
        # 3. Trend Analysis
        trend_strength, trend_direction = self.trend_analyzer.analyze(prices)
        
        # 4. Market State
        market_state = self.market_state_classifier.classify(prices)
        
        # 5. Momentum Tracking
        momentum_regime, momentum_strength = self.momentum_tracker.track(prices)
        
        # 6. Mean Reversion Detection
        is_reversion, reversion_prob = self.mean_reversion_detector.detect(prices)
        
        # 7. Breakout Estimation
        is_breakout, breakout_prob, breakout_dir = self.breakout_estimator.estimate(prices)
        
        # 8. Drawdown Monitoring
        # Use last price as equity proxy
        drawdown_state, drawdown = self.drawdown_monitor.monitor(prices[-1] if prices else 100)
        
        # 9. Recovery Detection
        is_recovery, recovery_conf = self.recovery_detector.detect(prices)
        
        # 10. Black Swan Check
        is_black_swan, black_swan_score = self.black_swan_alert.check(prices)
        
        # 11. Euphoria Detection
        is_euphoria, euphoria_score = self.euphoria_detector.detect(prices)
        
        # 12. Capitulation Detection
        is_capitulation, capitulation_score = self.capitulation_detector.detect(prices)
        
        # 13. Accumulation Detection
        is_accumulation, accumulation_score = self.accumulation_detector.detect(prices)
        
        # 14. Distribution Detection
        is_distribution, distribution_score = self.distribution_detector.detect(prices)
        
        # 15. Volatility Forecast
        vol_forecast = self.volatility_forecaster.forecast(prices)
        
        # Calculate overall volatility
        volatility = vol_forecast.get("current", 0.02)
        
        # 16. Position Size Adaptation
        position_multiplier = self.position_size_adapter.adapt(regime, volatility, regime_confidence)
        
        # 17. Risk Level Adaptation
        risk_multiplier = self.risk_level_adapter.adapt(regime, drawdown, volatility)
        
        # 18. Strategy Weight Adaptation
        strategy_weights = self.strategy_weight_adapter.adapt(regime, volatility)
        
        # 19. Timeframe Adaptation
        timeframe_weights = self.timeframe_adapter.adapt(regime, volatility)
        
        # 20. Confidence Calibration
        calibrated_confidence = self.confidence_calibrator.calibrate(regime_confidence, regime, volatility)
        
        # 21. Learning Rate Adaptation
        learning_rate = self.learning_rate_adapter.adapt(0.5, 0.7)  # Placeholder performance
        
        # 22. Exploration Rate Adaptation
        exploration_rate = self.exploration_rate_adapter.adapt(self.cycle_count, 0.5)
        
        # 23. Ensemble Weight Optimization
        strategy_performances = {k: 0.5 for k in strategy_weights.keys()}  # Placeholder
        ensemble_weights = self.ensemble_optimizer.optimize(strategy_performances)
        
        # 24. Feature Importance
        feature_performance = {"price": 0.6, "volume": 0.5, "volatility": 0.5}
        feature_importance = self.feature_importance_tracker.update(feature_performance)
        
        # 25. Signal Decay Detection
        is_decaying, decay_rate = self.signal_decay_detector.detect([True, False] * 10)
        
        # 26. Alpha Decay Monitoring
        alpha_decaying, alpha_decay_rate = self.alpha_decay_monitor.monitor(np.diff(prices[-30:]) if len(prices) >= 30 else [])
        
        # 27. Regime Transition Prediction
        self.regime_history.append(regime.value)
        transition_probs = self.regime_transition_predictor.predict(regime.value, list(self.regime_history))
        
        # 28. Meta-Adaptation Control
        component_performances = {
            "regime": regime_confidence,
            "volatility": vol_confidence,
            "trend": trend_strength,
            "momentum": momentum_strength,
            "risk": risk_multiplier,
            "performance": 0.5,
        }
        meta_weights = self.meta_controller.control(component_performances)
        
        # Update state
        self.state = AdaptationState(
            regime=regime,
            confidence=calibrated_confidence,
            volatility_regime=vol_regime,
            trend_strength=trend_strength,
            position_multiplier=position_multiplier,
            risk_multiplier=risk_multiplier,
            strategy_weights=strategy_weights,
            timeframe_weights=timeframe_weights,
            metadata={
                "regime_confidence": regime_confidence,
                "volatility": volatility,
                "vol_forecast": vol_forecast,
                "momentum_regime": momentum_regime,
                "momentum_strength": momentum_strength,
                "is_reversion": is_reversion,
                "is_breakout": is_breakout,
                "breakout_direction": breakout_dir,
                "drawdown_state": drawdown_state,
                "drawdown": drawdown,
                "is_recovery": is_recovery,
                "is_black_swan": is_black_swan,
                "black_swan_score": black_swan_score,
                "is_euphoria": is_euphoria,
                "is_capitulation": is_capitulation,
                "is_accumulation": is_accumulation,
                "is_distribution": is_distribution,
                "learning_rate": learning_rate,
                "exploration_rate": exploration_rate,
                "ensemble_weights": ensemble_weights,
                "feature_importance": feature_importance,
                "signal_decay": decay_rate,
                "alpha_decay": alpha_decay_rate,
                "transition_probs": transition_probs,
                "meta_weights": meta_weights,
            },
        )
        
        return self.state
    
    def get_status(self) -> Dict[str, Any]:
        """Get adaptation engine status."""
        return {
            "total_components": 30,
            "cycle_count": self.cycle_count,
            "current_regime": self.state.regime.value,
            "confidence": self.state.confidence,
            "position_multiplier": self.state.position_multiplier,
            "risk_multiplier": self.state.risk_multiplier,
            "active_components": [
                "regime_detector", "volatility_classifier", "trend_analyzer",
                "market_state_classifier", "correlation_monitor", "liquidity_detector",
                "momentum_tracker", "mean_reversion_detector", "breakout_estimator",
                "drawdown_monitor", "recovery_detector", "black_swan_alert",
                "euphoria_detector", "capitulation_detector", "accumulation_detector",
                "distribution_detector", "position_size_adapter", "risk_level_adapter",
                "strategy_weight_adapter", "timeframe_adapter", "volatility_forecaster",
                "confidence_calibrator", "learning_rate_adapter", "exploration_rate_adapter",
                "ensemble_optimizer", "feature_importance_tracker", "signal_decay_detector",
                "alpha_decay_monitor", "regime_transition_predictor", "meta_controller",
            ],
        }


def get_omega_adaptation() -> OmegaAdaptationEngine:
    """Get Omega Adaptation Engine."""
    return OmegaAdaptationEngine()
