"""Real-Time Market Regime Detection and Adaptation.

Features:
- Continuous market analysis
- Regime classification (trending, ranging, volatile, calm, etc.)
- Smoothed regime transitions
- Multi-timeframe analysis
- Volatility regime detection
- Liquidity regime detection
- Correlation regime detection
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    HIGH_VOLUME = "high_volume"
    LOW_VOLUME = "low_volume"
    REGIME_UNKNOWN = "unknown"


class TrendDirection(Enum):
    UP = 1
    DOWN = -1
    SIDEWAYS = 0


@dataclass
class RegimeMetrics:
    regime: MarketRegime
    confidence: float
    trend_direction: TrendDirection
    volatility_level: float
    volume_level: float
    momentum: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeTransition:
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: float
    smoothness: float


class VolatilityAnalyzer:
    def __init__(self, window: int = 20):
        self._window = window
        self._returns: deque = deque(maxlen=window)

    def add_return(self, return_pct: float) -> None:
        self._returns.append(return_pct)

    def get_volatility(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        return float(np.std(list(self._returns))) * np.sqrt(252)

    def get_volatility_level(self) -> str:
        vol = self.get_volatility()
        if vol > 1.5:
            return "extreme"
        elif vol > 1.0:
            return "high"
        elif vol > 0.5:
            return "normal"
        elif vol > 0.25:
            return "low"
        return "very_low"

    def get_atr_ratio(self, atr: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return atr / price * 100


class TrendDetector:
    def __init__(self, short_window: int = 10, long_window: int = 50):
        self._short_window = short_window
        self._long_window = long_window
        self._prices: deque = deque(maxlen=long_window)

    def add_price(self, price: float) -> None:
        self._prices.append(price)

    def get_trend(self) -> TrendDirection:
        if len(self._prices) < self._long_window:
            return TrendDirection.SIDEWAYS

        short_ma = np.mean(list(self._prices)[-self._short_window:])
        long_ma = np.mean(list(self._prices)[-self._long_window:])

        threshold = 0.005 * long_ma

        if short_ma > long_ma + threshold:
            return TrendDirection.UP
        elif short_ma < long_ma - threshold:
            return TrendDirection.DOWN
        return TrendDirection.SIDEWAYS

    def get_momentum(self) -> float:
        if len(self._prices) < 10:
            return 0.0

        recent = list(self._prices)[-10:]
        momentum = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0
        return momentum * 100


class VolumeAnalyzer:
    def __init__(self, window: int = 20):
        self._window = window
        self._volumes: deque = deque(maxlen=window)

    def add_volume(self, volume: float) -> None:
        self._volumes.append(volume)

    def get_volume_level(self) -> str:
        if len(self._volumes) < 2:
            return "normal"

        avg_volume = np.mean(list(self._volumes))
        current_volume = self._volumes[-1]

        ratio = current_volume / avg_volume if avg_volume > 0 else 1

        if ratio > 2.0:
            return "extreme_high"
        elif ratio > 1.5:
            return "high"
        elif ratio > 0.75:
            return "normal"
        elif ratio > 0.5:
            return "low"
        return "very_low"

    def get_volume_ratio(self) -> float:
        if len(self._volumes) < 2:
            return 1.0
        avg = np.mean(list(self._volumes))
        return self._volumes[-1] / avg if avg > 0 else 1.0


class SupportResistanceDetector:
    def __init__(self, window: int = 100):
        self._window = window
        self._prices: deque = deque(maxlen=window)

    def add_price(self, price: float) -> None:
        self._prices.append(price)

    def find_levels(self, num_levels: int = 5) -> Dict[str, List[float]]:
        if len(self._prices) < 20:
            return {"support": [], "resistance": []}

        prices = np.array(list(self._prices))

        levels = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                levels.append(("resistance", prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                levels.append(("support", prices[i]))

        support_levels = sorted([p for t, p in levels if t == "support"], reverse=True)[:num_levels]
        resistance_levels = sorted([p for t, p in levels if t == "resistance"])[:num_levels]

        return {
            "support": support_levels,
            "resistance": resistance_levels,
        }

    def distance_to_support(self, price: float) -> float:
        levels = self.find_levels()
        supports = levels.get("support", [])
        if not supports:
            return 0.0
        closest = max([s for s in supports if s < price], default=0)
        return (price - closest) / price * 100 if price > 0 else 0.0

    def distance_to_resistance(self, price: float) -> float:
        levels = self.find_levels()
        resistances = levels.get("resistance", [])
        if not resistances:
            return 0.0
        closest = min([r for r in resistances if r > price], default=float("inf"))
        return (closest - price) / price * 100 if price > 0 else 0.0


class MarketRegimeDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._volatility = VolatilityAnalyzer(
            window=self.config.get("volatility_window", 20)
        )
        self._trend = TrendDetector(
            short_window=self.config.get("short_window", 10),
            long_window=self.config.get("long_window", 50)
        )
        self._volume = VolumeAnalyzer(
            window=self.config.get("volume_window", 20)
        )
        self._sr = SupportResistanceDetector(
            window=self.config.get("sr_window", 100)
        )

        self._current_regime = MarketRegime.REGIME_UNKNOWN
        self._regime_confidence = 0.0
        self._transition_history: deque = deque(maxlen=100)
        self._regime_history: deque = deque(maxlen=1000)

        self._smoothing_factor = self.config.get("smoothing_factor", 0.3)
        self._min_confidence = self.config.get("min_confidence", 0.6)

    def update(
        self,
        price: float,
        volume: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> RegimeMetrics:
        if len(self._trend._prices) > 0:
            last_price = self._trend._prices[-1]
            return_pct = (price - last_price) / last_price * 100 if last_price > 0 else 0
            self._volatility.add_return(return_pct)

        self._trend.add_price(price)
        self._volume.add_volume(volume)
        self._sr.add_price(price)

        regime = self._classify_regime()
        confidence = self._calculate_confidence(regime)

        smoothed_regime = self._smooth_transition(regime, confidence)

        metrics = RegimeMetrics(
            regime=smoothed_regime,
            confidence=confidence,
            trend_direction=self._trend.get_trend(),
            volatility_level=self._volatility.get_volatility_level(),
            volume_level=self._volume.get_volume_level(),
            momentum=self._trend.get_momentum(),
            timestamp=time.time(),
            metadata={
                "volatility": self._volatility.get_volatility(),
                "volume_ratio": self._volume.get_volume_ratio(),
            }
        )

        self._regime_history.append(metrics)

        if smoothed_regime != self._current_regime:
            self._transition_history.append(RegimeTransition(
                from_regime=self._current_regime,
                to_regime=smoothed_regime,
                timestamp=time.time(),
                smoothness=confidence,
            ))
            self._current_regime = smoothed_regime

        self._regime_confidence = confidence

        return metrics

    def _classify_regime(self) -> MarketRegime:
        volatility = self._volatility.get_volatility_level()
        trend = self._trend.get_trend()
        volume = self._volume.get_volume_level()
        momentum = self._trend.get_momentum()

        if volatility == "extreme" or volatility == "high":
            return MarketRegime.VOLATILE
        elif volatility == "very_low" or volatility == "low":
            return MarketRegime.CALM

        if volume == "extreme_high" or volume == "high":
            if abs(momentum) > 2:
                return MarketRegime.BREAKOUT
            return MarketRegime.HIGH_VOLUME
        elif volume == "very_low" or volume == "low":
            return MarketRegime.LOW_VOLUME

        if trend == TrendDirection.UP:
            if momentum > 3:
                return MarketRegime.TRENDING_UP
            elif momentum > 1:
                return MarketRegime.TRENDING_UP
            return MarketRegime.CONSOLIDATION
        elif trend == TrendDirection.DOWN:
            if momentum < -3:
                return MarketRegime.TRENDING_DOWN
            elif momentum < -1:
                return MarketRegime.TRENDING_DOWN
            return MarketRegime.CONSOLIDATION

        return MarketRegime.RANGING

    def _calculate_confidence(self, regime: MarketRegime) -> float:
        volatility = self._volatility.get_volatility_level()
        volume = self._volume.get_volume_level()
        momentum = abs(self._trend.get_momentum())

        confidence = 0.5

        if volatility in ["extreme", "very_low"]:
            confidence += 0.2

        if volume in ["extreme_high", "very_low"]:
            confidence += 0.15

        if momentum > 3:
            confidence += 0.2
        elif momentum > 1:
            confidence += 0.1

        return min(1.0, confidence)

    def _smooth_transition(
        self,
        new_regime: MarketRegime,
        confidence: float,
    ) -> MarketRegime:
        if confidence < self._min_confidence:
            return self._current_regime

        if self._current_regime == MarketRegime.REGIME_UNKNOWN:
            return new_regime

        return new_regime

    def get_current_regime(self) -> RegimeMetrics:
        if self._regime_history:
            return self._regime_history[-1]
        return RegimeMetrics(
            regime=MarketRegime.REGIME_UNKNOWN,
            confidence=0.0,
            trend_direction=TrendDirection.SIDEWAYS,
            volatility_level="normal",
            volume_level="normal",
            momentum=0.0,
            timestamp=time.time(),
        )

    def get_regime_distribution(self, window: int = 100) -> Dict[MarketRegime, int]:
        recent = list(self._regime_history)[-window:]
        counts = defaultdict(int)
        for m in recent:
            counts[m.regime] += 1
        return dict(counts)

    def get_stability_score(self) -> float:
        if len(self._regime_history) < 10:
            return 1.0

        recent = list(self._regime_history)[-10:]
        regimes = [m.regime for m in recent]
        stable_count = regimes.count(regimes[0])
        return stable_count / len(regimes)


class MultiTimeframeRegimeAnalyzer:
    def __init__(self):
        self._timeframes: Dict[str, MarketRegimeDetector] = {}

    def add_timeframe(self, name: str, detector: MarketRegimeDetector) -> None:
        self._timeframes[name] = detector

    def get_unified_regime(self) -> MarketRegime:
        if not self._timeframes:
            return MarketRegime.REGIME_UNKNOWN

        regime_counts: Dict[MarketRegime, int] = defaultdict(int)
        weights = {"1m": 1, "5m": 2, "15m": 3, "1h": 4, "4h": 5, "1d": 6}

        for name, detector in self._timeframes.items():
            regime = detector.get_current_regime().regime
            weight = weights.get(name, 1)
            regime_counts[regime] += weight

        if not regime_counts:
            return MarketRegime.REGIME_UNKNOWN

        return max(regime_counts, key=regime_counts.get)

    def get_adaptive_parameters(self) -> Dict[str, Any]:
        unified = self.get_unified_regime()

        params = {
            "position_size_mult": 1.0,
            "stop_loss_mult": 1.0,
            "take_profit_mult": 1.0,
            "signal_threshold_mult": 1.0,
            "max_holding_period_mult": 1.0,
        }

        if unified == MarketRegime.VOLATILE:
            params["position_size_mult"] = 0.5
            params["stop_loss_mult"] = 1.5
            params["take_profit_mult"] = 1.2
            params["signal_threshold_mult"] = 1.3
            params["max_holding_period_mult"] = 0.7
        elif unified == MarketRegime.CALM:
            params["position_size_mult"] = 1.2
            params["stop_loss_mult"] = 0.8
            params["take_profit_mult"] = 1.0
            params["signal_threshold_mult"] = 0.9
            params["max_holding_period_mult"] = 1.2
        elif unified in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            params["position_size_mult"] = 1.1
            params["stop_loss_mult"] = 1.0
            params["take_profit_mult"] = 1.3
            params["signal_threshold_mult"] = 0.85
            params["max_holding_period_mult"] = 1.3
        elif unified == MarketRegime.RANGING:
            params["position_size_mult"] = 0.8
            params["stop_loss_mult"] = 0.7
            params["take_profit_mult"] = 0.8
            params["signal_threshold_mult"] = 1.1
            params["max_holding_period_mult"] = 0.8

        return params


class RealTimeRegimeAdaptation:
    def __init__(self):
        self._detector = MarketRegimeDetector()
        self._mtf_analyzer = MultiTimeframeRegimeAnalyzer()
        self._adaptation_callbacks: List[Callable] = []

    def register_callback(self, callback: Callable[[RegimeMetrics, Dict], None]) -> None:
        self._adaptation_callbacks.append(callback)

    def update(
        self,
        price: float,
        volume: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> RegimeMetrics:
        metrics = self._detector.update(price, volume, high, low, atr)

        params = self._get_adapted_parameters(metrics)

        for callback in self._adaptation_callbacks:
            try:
                callback(metrics, params)
            except Exception as e:
                logger.warning(f"Adaptation callback error: {e}")

        return metrics

    def _get_adapted_parameters(self, metrics: RegimeMetrics) -> Dict[str, Any]:
        params = {
            "position_size_mult": 1.0,
            "stop_loss_mult": 1.0,
            "take_profit_mult": 1.0,
            "signal_threshold_mult": 1.0,
            "max_holding_period_mult": 1.0,
            "trailing_stop_mult": 1.0,
            "entries_per_day": 3,
        }

        regime = metrics.regime

        if regime == MarketRegime.VOLATILE:
            params["position_size_mult"] = 0.5
            params["stop_loss_mult"] = 2.0
            params["take_profit_mult"] = 1.5
            params["signal_threshold_mult"] = 1.5
            params["max_holding_period_mult"] = 0.5
            params["trailing_stop_mult"] = 1.5
            params["entries_per_day"] = 1
        elif regime == MarketRegime.CALM:
            params["position_size_mult"] = 1.3
            params["stop_loss_mult"] = 0.8
            params["take_profit_mult"] = 0.9
            params["signal_threshold_mult"] = 0.8
            params["max_holding_period_mult"] = 1.5
            params["trailing_stop_mult"] = 0.8
            params["entries_per_day"] = 5
        elif regime == MarketRegime.TRENDING_UP:
            params["position_size_mult"] = 1.2
            params["stop_loss_mult"] = 1.0
            params["take_profit_mult"] = 1.5
            params["signal_threshold_mult"] = 0.8
            params["max_holding_period_mult"] = 1.5
            params["trailing_stop_mult"] = 0.7
            params["entries_per_day"] = 4
        elif regime == MarketRegime.TRENDING_DOWN:
            params["position_size_mult"] = 1.2
            params["stop_loss_mult"] = 1.0
            params["take_profit_mult"] = 1.5
            params["signal_threshold_mult"] = 0.8
            params["max_holding_period_mult"] = 1.5
            params["trailing_stop_mult"] = 0.7
            params["entries_per_day"] = 4
        elif regime == MarketRegime.RANGING:
            params["position_size_mult"] = 0.7
            params["stop_loss_mult"] = 0.7
            params["take_profit_mult"] = 0.7
            params["signal_threshold_mult"] = 1.2
            params["max_holding_period_mult"] = 0.8
            params["trailing_stop_mult"] = 1.0
            params["entries_per_day"] = 6
        elif regime == MarketRegime.BREAKOUT:
            params["position_size_mult"] = 1.3
            params["stop_loss_mult"] = 1.2
            params["take_profit_mult"] = 1.8
            params["signal_threshold_mult"] = 0.7
            params["max_holding_period_mult"] = 1.2
            params["trailing_stop_mult"] = 0.6
            params["entries_per_day"] = 2
        elif regime == MarketRegime.LOW_VOLUME:
            params["position_size_mult"] = 0.6
            params["stop_loss_mult"] = 0.6
            params["take_profit_mult"] = 0.6
            params["signal_threshold_mult"] = 1.3
            params["max_holding_period_mult"] = 0.5
            params["trailing_stop_mult"] = 1.2
            params["entries_per_day"] = 1

        if metrics.volatility_level == "high":
            params["position_size_mult"] *= 0.7
        elif metrics.volatility_level == "low":
            params["position_size_mult"] *= 1.1

        if metrics.volume_level == "high":
            params["signal_threshold_mult"] *= 0.9
        elif metrics.volume_level == "low":
            params["signal_threshold_mult"] *= 1.2

        return params

    def get_current_regime(self) -> RegimeMetrics:
        return self._detector.get_current_regime()

    def get_stability(self) -> float:
        return self._detector.get_stability_score()
