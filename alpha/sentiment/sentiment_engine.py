"""
Sentiment Engine
Multi-source sentiment aggregation for alpha generation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SentimentReading:
    source: str
    score: float  # -1 (fear) to 1 (greed)
    confidence: float  # 0 to 1
    timestamp: float = 0.0


class FearGreedIndex:
    """Composite fear/greed index from multiple signals."""

    def __init__(self, lookback: int = 100, ewm_span: int = 14):
        self.lookback = int(lookback)
        self.ewm_span = int(ewm_span)
        self._readings: Dict[str, deque] = {}
        self._source_weights: Dict[str, float] = {
            "funding_rate": 0.25,
            "volume_profile": 0.20,
            "price_momentum": 0.20,
            "volatility": 0.15,
            "open_interest": 0.10,
            "social": 0.10,
        }

    def update(self, reading: SentimentReading) -> None:
        source = reading.source
        if source not in self._readings:
            self._readings[source] = deque(maxlen=self.lookback)
        self._readings[source].append(reading)

    def fear_greed(self, symbol: str = "") -> float:
        if not self._readings:
            return 0.0
        weighted_sum = 0.0
        total_weight = 0.0
        for source, readings in self._readings.items():
            if not readings:
                continue
            weight = self._source_weights.get(source, 0.1)
            scores = [r.score * r.confidence for r in list(readings)[-self.ewm_span:]]
            if not scores:
                continue
            if len(scores) >= 3:
                alpha = 2.0 / (min(self.ewm_span, len(scores)) + 1)
                ewm = scores[0]
                for s in scores[1:]:
                    ewm = alpha * s + (1 - alpha) * ewm
                avg_score = ewm
            else:
                avg_score = float(np.mean(scores))
            weighted_sum += avg_score * weight
            total_weight += weight
        return float(np.clip(weighted_sum / max(total_weight, 1e-12), -1.0, 1.0))

    def get_score(self) -> float:
        return self.fear_greed()


class SentimentFromFunding:
    """Derive sentiment signal from funding rates."""

    def __init__(self, extreme_threshold: float = 0.0005):
        self.extreme_threshold = float(extreme_threshold)

    def compute(self, funding_rate: float) -> SentimentReading:
        rate = float(funding_rate)
        if abs(rate) < self.extreme_threshold * 0.5:
            score = 0.0
            confidence = 0.3
        else:
            score = -float(np.clip(rate / self.extreme_threshold, -1.0, 1.0))
            confidence = min(abs(rate) / self.extreme_threshold, 1.0)
        return SentimentReading(source="funding_rate", score=score, confidence=confidence)


class SentimentFromVolatility:
    """Derive fear/greed from volatility regime."""

    def __init__(self, lookback: int = 30):
        self._vols: deque = deque(maxlen=lookback)

    def compute(self, current_vol: float) -> SentimentReading:
        self._vols.append(float(current_vol))
        if len(self._vols) < 5:
            return SentimentReading(source="volatility", score=0.0, confidence=0.2)
        median_vol = float(np.median(list(self._vols)))
        ratio = current_vol / max(median_vol, 1e-12)
        if ratio > 1.5:
            score = -min((ratio - 1.0) / 2.0, 1.0)
        elif ratio < 0.7:
            score = min((1.0 - ratio) / 0.5, 1.0)
        else:
            score = 0.0
        confidence = min(abs(ratio - 1.0), 1.0)
        return SentimentReading(source="volatility", score=score, confidence=confidence)


class SentimentFromMomentum:
    """Derive sentiment from price momentum."""

    def __init__(self, fast_period: int = 7, slow_period: int = 30):
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)
        self._returns: deque = deque(maxlen=slow_period * 2)

    def compute(self, ret: float) -> SentimentReading:
        self._returns.append(float(ret))
        if len(self._returns) < self.fast_period:
            return SentimentReading(source="price_momentum", score=0.0, confidence=0.1)
        rets = list(self._returns)
        fast_mom = float(np.mean(rets[-self.fast_period:]))
        slow_mom = float(np.mean(rets[-self.slow_period:])) if len(rets) >= self.slow_period else fast_mom
        combined = (fast_mom * 0.6 + slow_mom * 0.4)
        score = float(np.clip(combined * 100, -1.0, 1.0))
        confidence = min(len(self._returns) / self.slow_period, 1.0)
        return SentimentReading(source="price_momentum", score=score, confidence=confidence)


class SentimentEngine:
    """Unified sentiment engine combining all sources."""

    def __init__(self):
        self.fear_greed = FearGreedIndex()
        self.from_funding = SentimentFromFunding()
        self.from_volatility = SentimentFromVolatility()
        self.from_momentum = SentimentFromMomentum()

    def on_funding_rate(self, rate: float) -> SentimentReading:
        reading = self.from_funding.compute(rate)
        self.fear_greed.update(reading)
        return reading

    def on_volatility(self, vol: float) -> SentimentReading:
        reading = self.from_volatility.compute(vol)
        self.fear_greed.update(reading)
        return reading

    def on_return(self, ret: float) -> SentimentReading:
        reading = self.from_momentum.compute(ret)
        self.fear_greed.update(reading)
        return reading

    def get_composite_sentiment(self) -> Dict[str, Any]:
        score = self.fear_greed.get_score()
        return {
            "fear_greed_score": score,
            "label": "extreme_fear" if score < -0.6 else (
                "fear" if score < -0.2 else (
                    "neutral" if score < 0.2 else (
                        "greed" if score < 0.6 else "extreme_greed"
                    )
                )
            ),
        }
