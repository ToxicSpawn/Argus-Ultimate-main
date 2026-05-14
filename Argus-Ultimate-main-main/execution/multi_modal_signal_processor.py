"""Multi-Modal Signal Processing Module.

Combines signals from multiple sources:
- Technical analysis (price patterns, indicators)
- Sentiment analysis (news, social media)
- Macro indicators (rates, currencies)
- Alternative data (on-chain, satellite)
- Quantum-enhanced signals

Features:
- Signal fusion with weighted ensemble
- Confidence calibration
- Cross-modal validation
- Signal decay modeling
"""

from __future__ import annotations

import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    ALTERNATIVE = "alternative"
    QUANTUM = "quantum"
    FUNDAMENTAL = "fundamental"
    ONCHAIN = "onchain"


class SignalDirection(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class SignalInput:
    source: SignalSource
    symbol: str
    direction: SignalDirection
    confidence: float
    strength: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    lookback_bars: int = 0


@dataclass
class FusedSignal:
    symbol: str
    direction: SignalDirection
    confidence: float
    strength: float
    sources: List[SignalSource]
    source_weights: Dict[SignalSource, float]
    timestamp: float
    decay_factor: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalFusionEngine:
    def __init__(
        self,
        source_weights: Optional[Dict[SignalSource, float]] = None,
        min_source_agreement: int = 2,
        confidence_threshold: float = 0.5,
        decay_half_life_bars: int = 10,
    ):
        self._source_weights = source_weights or {
            SignalSource.TECHNICAL: 0.35,
            SignalSource.SENTIMENT: 0.15,
            SignalSource.MACRO: 0.15,
            SignalSource.ALTERNATIVE: 0.10,
            SignalSource.QUANTUM: 0.15,
            SignalSource.FUNDAMENTAL: 0.05,
            SignalSource.ONCHAIN: 0.05,
        }
        self._min_agreement = min_source_agreement
        self._confidence_threshold = confidence_threshold
        self._decay_half_life = decay_half_life_bars
        
        self._signal_history: deque = deque(maxlen=1000)
        self._source_accuracy: Dict[SignalSource, Dict[str, float]] = {}
        self._cross_validation_enabled = True

    def fuse_signals(
        self,
        signals: List[SignalInput],
        symbol: str,
        current_bar: int = 0,
    ) -> Optional[FusedSignal]:
        if not signals:
            return None
        
        grouped = self._group_by_direction(signals)
        
        total_weight = 0.0
        weighted_direction = 0.0
        
        for direction, source_signals in grouped.items():
            dir_weight = sum(
                self._get_source_weight(s.source) * s.confidence * s.strength
                for s in source_signals
            )
            weighted_direction += direction.value * dir_weight
            total_weight += dir_weight
        
        if total_weight <= 0:
            return None
        
        normalized_direction = weighted_direction / total_weight
        
        if normalized_direction > 0.3:
            direction = SignalDirection.BULLISH
        elif normalized_direction < -0.3:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL
        
        unique_sources = set(s.source for s in signals)
        
        if len(unique_sources) < self._min_agreement:
            return None
        
        avg_confidence = np.mean([s.confidence for s in signals])
        
        if direction != SignalDirection.NEUTRAL:
            boost = min(0.2, 0.05 * (len(unique_sources) - 1))
            avg_confidence = min(1.0, avg_confidence + boost)
        
        avg_strength = np.mean([s.strength for s in signals])
        
        decay_factor = self._calculate_decay(current_bar, signals)
        
        source_weights = {
            s.source: self._get_source_weight(s.source)
            for s in signals
        }
        
        fused = FusedSignal(
            symbol=symbol,
            direction=direction,
            confidence=avg_confidence,
            strength=avg_strength,
            sources=list(unique_sources),
            source_weights=source_weights,
            timestamp=time.time(),
            decay_factor=decay_factor,
            metadata={
                "raw_signals": len(signals),
                "unique_sources": len(unique_sources),
                "normalized_direction": normalized_direction,
            }
        )
        
        self._signal_history.append(fused)
        return fused

    def _group_by_direction(
        self,
        signals: List[SignalInput]
    ) -> Dict[SignalDirection, List[SignalInput]]:
        grouped: Dict[SignalDirection, List[SignalInput]] = {
            SignalDirection.BULLISH: [],
            SignalDirection.BEARISH: [],
            SignalDirection.NEUTRAL: [],
        }
        for s in signals:
            grouped[s.direction].append(s)
        return grouped

    def _get_source_weight(self, source: SignalSource) -> float:
        base_weight = self._source_weights.get(source, 0.1)
        
        if source in self._source_accuracy:
            acc = self._source_accuracy[source].get("accuracy", 0.5)
            accuracy_multiplier = 0.5 + (acc * 0.5)
            return base_weight * accuracy_multiplier
        
        return base_weight

    def _calculate_decay(
        self,
        current_bar: int,
        signals: List[SignalInput]
    ) -> float:
        if not signals:
            return 1.0
        
        avg_lookback = np.mean([s.lookback_bars for s in signals])
        bars_since_signal = current_bar - avg_lookback
        
        decay = np.power(0.5, bars_since_signal / self._decay_half_life)
        return max(0.1, decay)

    def update_accuracy(
        self,
        source: SignalSource,
        symbol: str,
        was_correct: bool,
    ) -> None:
        if source not in self._source_accuracy:
            self._source_accuracy[source] = {"wins": 0, "total": 0, "accuracy": 0.5}
        
        acc_data = self._source_accuracy[source]
        acc_data["total"] += 1
        if was_correct:
            acc_data["wins"] += 1
        acc_data["accuracy"] = acc_data["wins"] / acc_data["total"]
        
        logger.debug(
            f"Source {source.value} accuracy for {symbol}: {acc_data['accuracy']:.2%}"
        )

    def cross_validate(
        self,
        signal: FusedSignal,
        validation_signals: List[SignalInput],
    ) -> float:
        if not validation_signals:
            return signal.confidence
        
        validation_dir = np.mean([
            s.direction.value * s.confidence for s in validation_signals
        ])
        
        signal_dir = signal.direction.value
        
        cross_validation_score = 1.0 - min(1.0, abs(validation_dir - signal_dir) / 2)
        
        adjusted_confidence = (
            signal.confidence * 0.7 +
            cross_validation_score * 0.3
        )
        
        return adjusted_confidence


class MultiModalSignalProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._fusion_engine = SignalFusionEngine()
        
        self._technical_handler: Optional[Callable] = None
        self._sentiment_handler: Optional[Callable] = None
        self._macro_handler: Optional[Callable] = None
        self._alternative_handler: Optional[Callable] = None
        self._quantum_handler: Optional[Callable] = None
        
        self._signal_transformers: Dict[SignalSource, Callable] = {}
        
        self._processors = {
            SignalSource.TECHNICAL: self._process_technical,
            SignalSource.SENTIMENT: self._process_sentiment,
            SignalSource.MACRO: self._process_macro,
            SignalSource.ALTERNATIVE: self._process_alternative,
            SignalSource.QUANTUM: self._process_quantum,
        }

    def register_handler(
        self,
        source: SignalSource,
        handler: Callable[[str], List[SignalInput]],
    ) -> None:
        self._processors[source] = handler

    def register_transformer(
        self,
        source: SignalSource,
        transformer: Callable[[Any], SignalInput],
    ) -> None:
        self._signal_transformers[source] = transformer

    async def process(
        self,
        symbol: str,
        current_bar: int = 0,
    ) -> Optional[FusedSignal]:
        signals = []
        
        for source, processor in self._processors.items():
            try:
                source_signals = processor(symbol)
                if source_signals:
                    signals.extend(source_signals)
            except Exception as e:
                logger.warning(f"Error processing {source.value}: {e}")
        
        if not signals:
            return None
        
        fused = self._fusion_engine.fuse_signals(
            signals,
            symbol=symbol,
            current_bar=current_bar,
        )
        
        return fused

    def _process_technical(self, symbol: str) -> List[SignalInput]:
        return []

    def _process_sentiment(self, symbol: str) -> List[SignalInput]:
        return []

    def _process_macro(self, symbol: str) -> List[SignalInput]:
        return []

    def _process_alternative(self, symbol: str) -> List[SignalInput]:
        return []

    def _process_quantum(self, symbol: str) -> List[SignalInput]:
        return []

    def add_signal(
        self,
        source: SignalSource,
        symbol: str,
        direction: SignalDirection,
        confidence: float,
        strength: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> SignalInput:
        signal = SignalInput(
            source=source,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            strength=strength,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        
        if source in self._signal_transformers:
            signal = self._signal_transformers[source](signal)
        
        return signal


class SignalAggregator:
    def __init__(self, window_bars: int = 20):
        self._window = window_bars
        self._signal_buffer: Dict[str, deque] = {}

    def add_signal(self, signal: FusedSignal) -> None:
        if signal.symbol not in self._signal_buffer:
            self._signal_buffer[signal.symbol] = deque(maxlen=self._window)
        self._signal_buffer[signal.symbol].append(signal)

    def get_aggregated(
        self,
        symbol: str,
    ) -> Optional[FusedSignal]:
        if symbol not in self._signal_buffer:
            return None
        
        signals = list(self._signal_buffer[symbol])
        if not signals:
            return None
        
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_strength = np.mean([s.strength for s in signals])
        
        directions = [s.direction for s in signals]
        direction = max(set(directions), key=directions.count)
        
        all_sources = set()
        for s in signals:
            all_sources.update(s.sources)
        
        return FusedSignal(
            symbol=symbol,
            direction=direction,
            confidence=avg_confidence,
            strength=avg_strength,
            sources=list(all_sources),
            source_weights={},
            timestamp=time.time(),
            metadata={"aggregated_signals": len(signals)}
        )
