"""
Signal Fusion — combines multiple signal sources into unified trading signals.

Features:
  - Weighted fusion of technical, sentiment, regime, and orderbook signals
  - Signal quality scoring
  - Conflict resolution between opposing signals
  - Signal decay over time

Usage:
    fusion = SignalFusion()
    
    signal = fusion.combine(
        technical={"direction": 0.8, "strength": 0.7},
        sentiment={"score": 0.6, "confidence": 0.8},
        regime={"type": "TREND_UP", "confidence": 0.9},
        orderbook={"imbalance": 0.3, "depth_ratio": 1.2},
    )
    # signal.direction, signal.strength, signal.confidence
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalSource:
    """Individual signal source."""
    name: str
    direction: float     # -1 to +1 (bearish to bullish)
    strength: float      # 0 to 1
    confidence: float    # 0 to 1
    timestamp: float     # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Age of signal in seconds."""
        return time.time() - self.timestamp
    
    def decayed_strength(self, half_life_seconds: float = 300.0) -> float:
        """Strength adjusted for signal age."""
        age = self.age_seconds
        decay = np.exp(-0.693 * age / half_life_seconds)  # ln(2) / half_life
        return self.strength * decay


@dataclass
class FusedSignal:
    """Result from signal fusion."""
    direction: float          # -1 to +1
    strength: float           # 0 to 1
    confidence: float         # 0 to 1
    action: str               # "buy", "sell", "hold"
    sources_used: int
    source_contributions: Dict[str, float]  # Weight of each source
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": round(self.direction, 4),
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "action": self.action,
            "sources_used": self.sources_used,
            "source_contributions": self.source_contributions,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class SignalFusion:
    """
    Fuses multiple signal sources into a unified trading signal.
    
    Features:
    - Configurable weights per source type
    - Signal quality scoring
    - Conflict detection and resolution
    - Temporal decay of old signals
    
    Args:
        weights: Dict mapping source names to weights
        conflict_threshold: Direction difference threshold for conflict detection
        signal_half_life: Half-life for signal decay in seconds
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        conflict_threshold: float = 0.5,
        signal_half_life: float = 300.0,
    ):
        self.weights = weights or {
            "technical": 0.35,
            "sentiment": 0.15,
            "regime": 0.30,
            "orderbook": 0.20,
        }
        self.conflict_threshold = conflict_threshold
        self.signal_half_life = signal_half_life
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def combine(
        self,
        technical: Optional[Dict[str, float]] = None,
        sentiment: Optional[Dict[str, float]] = None,
        regime: Optional[Dict[str, Any]] = None,
        orderbook: Optional[Dict[str, float]] = None,
        custom_signals: Optional[List[SignalSource]] = None,
    ) -> FusedSignal:
        """
        Combine multiple signal sources.
        
        Args:
            technical: Technical analysis signal (direction, strength, confidence)
            sentiment: Sentiment signal (score, confidence)
            regime: Regime signal (type, confidence)
            orderbook: Orderbook signal (imbalance, depth_ratio)
            custom_signals: Additional custom SignalSource objects
            
        Returns:
            FusedSignal with combined direction, strength, confidence
        """
        sources: List[SignalSource] = []
        now = time.time()
        
        # Process technical signal
        if technical:
            sources.append(SignalSource(
                name="technical",
                direction=technical.get("direction", 0.0),
                strength=technical.get("strength", 0.5),
                confidence=technical.get("confidence", 0.5),
                timestamp=now,
                metadata=technical,
            ))
        
        # Process sentiment signal
        if sentiment:
            score = sentiment.get("score", 0.0)
            sources.append(SignalSource(
                name="sentiment",
                direction=np.clip(score, -1.0, 1.0),
                strength=abs(score),
                confidence=sentiment.get("confidence", 0.5),
                timestamp=now,
                metadata=sentiment,
            ))
        
        # Process regime signal
        if regime:
            regime_type = regime.get("type", "RANGING")
            regime_direction = self._regime_to_direction(regime_type)
            sources.append(SignalSource(
                name="regime",
                direction=regime_direction,
                strength=0.8,  # Regime signals are usually strong
                confidence=regime.get("confidence", 0.5),
                timestamp=now,
                metadata=regime,
            ))
        
        # Process orderbook signal
        if orderbook:
            imbalance = orderbook.get("imbalance", 0.0)
            sources.append(SignalSource(
                name="orderbook",
                direction=np.clip(imbalance, -1.0, 1.0),
                strength=min(abs(imbalance) * 2, 1.0),
                confidence=min(orderbook.get("depth_ratio", 1.0) / 2, 1.0),
                timestamp=now,
                metadata=orderbook,
            ))
        
        # Add custom signals
        if custom_signals:
            sources.extend(custom_signals)
        
        if not sources:
            return FusedSignal(
                direction=0.0,
                strength=0.0,
                confidence=0.0,
                action="hold",
                sources_used=0,
                source_contributions={},
                timestamp=datetime.now(timezone.utc),
            )
        
        # Check for conflicts
        has_conflict, conflict_info = self._detect_conflicts(sources)
        
        # Compute fused signal
        fused_direction = 0.0
        fused_strength = 0.0
        fused_confidence = 0.0
        contributions: Dict[str, float] = {}
        total_weight = 0.0
        
        for source in sources:
            weight = self.weights.get(source.name, 0.1)
            
            # Apply temporal decay
            decayed_strength = source.decayed_strength(self.signal_half_life)
            
            # Weighted contributions
            fused_direction += source.direction * weight * source.confidence
            fused_strength += decayed_strength * weight
            fused_confidence += source.confidence * weight
            contributions[source.name] = weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            fused_direction /= total_weight
            fused_strength /= total_weight
            fused_confidence /= total_weight
        
        # Conflict penalty
        if has_conflict:
            fused_confidence *= 0.7  # Reduce confidence on conflict
            logger.warning("Signal conflict detected: %s", conflict_info)
        
        # Determine action
        action = self._direction_to_action(fused_direction, fused_strength)
        
        return FusedSignal(
            direction=float(np.clip(fused_direction, -1.0, 1.0)),
            strength=float(np.clip(fused_strength, 0.0, 1.0)),
            confidence=float(np.clip(fused_confidence, 0.0, 1.0)),
            action=action,
            sources_used=len(sources),
            source_contributions=contributions,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "has_conflict": has_conflict,
                "conflict_info": conflict_info if has_conflict else None,
            },
        )
    
    def _regime_to_direction(self, regime: str) -> float:
        """Convert regime type to direction signal."""
        regime_map = {
            "TREND_UP": 0.8,
            "TREND_DOWN": -0.8,
            "RANGING": 0.0,
            "VOLATILE": 0.0,  # Neutral - too unpredictable
            "CRISIS": -0.5,   # Slightly bearish (defensive)
        }
        return regime_map.get(regime, 0.0)
    
    def _direction_to_action(self, direction: float, strength: float) -> str:
        """Convert direction and strength to action."""
        threshold = 0.15
        
        if abs(direction) < threshold or strength < 0.2:
            return "hold"
        elif direction > 0:
            return "buy"
        else:
            return "sell"
    
    def _detect_conflicts(
        self,
        sources: List[SignalSource],
    ) -> Tuple[bool, Optional[str]]:
        """Detect conflicts between signal sources."""
        if len(sources) < 2:
            return False, None
        
        # Check for opposing strong signals
        bullish_sources = [s for s in sources if s.direction > 0.3 and s.confidence > 0.5]
        bearish_sources = [s for s in sources if s.direction < -0.3 and s.confidence > 0.5]
        
        if bullish_sources and bearish_sources:
            # Check if directions differ significantly
            avg_bull = np.mean([s.direction for s in bullish_sources])
            avg_bear = np.mean([s.direction for s in bearish_sources])
            
            if abs(avg_bull - avg_bear) > self.conflict_threshold:
                return True, f"Bullish ({len(bullish_sources)}) vs Bearish ({len(bearish_sources)})"
        
        return False, None


class SignalQualityScorer:
    """Score the quality of trading signals."""
    
    @staticmethod
    def score(
        signal: FusedSignal,
        historical_accuracy: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Score signal quality.
        
        Returns dict with:
        - overall_score: 0-100
        - components: breakdown of scoring factors
        - recommendation: "strong", "moderate", "weak", "skip"
        """
        components = {}
        
        # Confidence component (40%)
        components["confidence"] = signal.confidence * 100
        
        # Strength component (30%)
        components["strength"] = signal.strength * 100
        
        # Source diversity (20%)
        diversity = min(signal.sources_used / 4, 1.0) * 100
        components["diversity"] = diversity
        
        # Historical accuracy (10%)
        if historical_accuracy is not None:
            components["accuracy"] = historical_accuracy * 100
        else:
            components["accuracy"] = 50.0  # Neutral
        
        # Weighted overall score
        overall = (
            components["confidence"] * 0.40 +
            components["strength"] * 0.30 +
            components["diversity"] * 0.20 +
            components["accuracy"] * 0.10
        )
        
        # Recommendation
        if overall >= 70:
            recommendation = "strong"
        elif overall >= 50:
            recommendation = "moderate"
        elif overall >= 30:
            recommendation = "weak"
        else:
            recommendation = "skip"
        
        return {
            "overall_score": round(overall, 1),
            "components": {k: round(v, 1) for k, v in components.items()},
            "recommendation": recommendation,
        }
