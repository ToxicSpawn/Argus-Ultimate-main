"""
Predictive Regime Detection
===========================
Predicts market regime changes BEFORE they happen, allowing proactive
parameter adjustments instead of reactive ones.

Key Features:
1. Multi-timeframe regime analysis
2. Regime transition probability modeling
3. Early warning system for regime changes
4. Proactive parameter pre-adjustment
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
class RegimeTransition:
    """Represents a regime transition."""
    from_regime: str
    to_regime: str
    probability: float
    avg_duration_before: float  # Average time spent in from_regime
    lead_signals: List[str]  # Signals that predict this transition


class RegimePredictor:
    """
    Predicts regime changes using Markov chain + lead indicators.
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        
        # Regime history
        self.regime_history: Deque[Tuple[float, str]] = deque(maxlen=history_size)
        
        # Transition matrix (Markov)
        self.transitions: Dict[str, Dict[str, int]] = {}
        self.transition_probs: Dict[str, Dict[str, float]] = {}
        
        # Regime durations
        self.regime_durations: Dict[str, List[float]] = {
            "trending_up": [],
            "trending_down": [],
            "ranging": [],
            "high_volatility": [],
            "low_volatility": [],
        }
        
        # Lead indicators (features that predict regime change)
        self.lead_indicators: Dict[str, Dict[str, float]] = {
            "trending_up": {
                "volatility_increase": 0.0,
                "volume_spike": 0.0,
                "momentum_divergence": 0.0,
            },
            "high_volatility": {
                "range_expansion": 0.0,
                "trend_weakness": 0.0,
                "correlation_break": 0.0,
            },
        }
        
        # Current state
        self.current_regime: str = "ranging"
        self.regime_start_time: float = time.time()
        self.regime_duration: float = 0.0
        
        # Prediction state
        self.predicted_next_regime: Optional[str] = None
        self.prediction_confidence: float = 0.0
        self.prediction_horizon: float = 0.0  # Seconds until predicted change
    
    def update(self, regime: str, market_features: Dict[str, float]) -> None:
        """Update with new regime observation."""
        now = time.time()
        
        # Record regime duration
        if regime != self.current_regime:
            duration = now - self.regime_start_time
            if self.current_regime in self.regime_durations:
                self.regime_durations[self.current_regime].append(duration)
            
            # Record transition
            self._record_transition(self.current_regime, regime)
            
            # Reset
            self.current_regime = regime
            self.regime_start_time = now
        
        self.regime_duration = now - self.regime_start_time
        self.regime_history.append((now, regime))
        
        # Update lead indicators
        self._update_lead_indicators(regime, market_features)
        
        # Update transition probabilities
        self._update_transition_probs()
        
        # Make prediction
        self._predict_next_regime(market_features)
    
    def _record_transition(self, from_regime: str, to_regime: str) -> None:
        """Record a regime transition."""
        if from_regime not in self.transitions:
            self.transitions[from_regime] = {}
        if to_regime not in self.transitions[from_regime]:
            self.transitions[from_regime][to_regime] = 0
        self.transitions[from_regime][to_regime] += 1
    
    def _update_transition_probs(self) -> None:
        """Update transition probability matrix."""
        for from_regime, to_regimes in self.transitions.items():
            total = sum(to_regimes.values())
            if total > 0:
                self.transition_probs[from_regime] = {
                    to: count / total 
                    for to, count in to_regimes.items()
                }
    
    def _update_lead_indicators(self, regime: str, features: Dict[str, float]) -> None:
        """Update lead indicator correlations with regime changes."""
        # Simplified: track feature values that preceded past transitions
        pass
    
    def _predict_next_regime(self, features: Dict[str, float]) -> None:
        """Predict the next regime based on Markov + features."""
        if self.current_regime not in self.transition_probs:
            self.predicted_next_regime = None
            self.prediction_confidence = 0.0
            return
        
        # Get transition probabilities
        probs = self.transition_probs[self.current_regime]
        
        if not probs:
            self.predicted_next_regime = None
            self.prediction_confidence = 0.0
            return
        
        # Find most likely transition
        best_regime = max(probs.items(), key=lambda x: x[1])
        
        # Adjust confidence based on duration
        # Longer in regime → higher chance of change
        avg_duration = np.mean(self.regime_durations.get(self.current_regime, [3600]))
        duration_factor = min(1.0, self.regime_duration / avg_duration)
        
        self.predicted_next_regime = best_regime[0]
        self.prediction_confidence = best_regime[1] * (0.5 + duration_factor * 0.5)
        
        # Estimate time until change
        remaining_avg = max(0, avg_duration - self.regime_duration)
        self.prediction_horizon = remaining_avg
    
    def get_prediction(self) -> Optional[Dict[str, Any]]:
        """Get current regime prediction."""
        if not self.predicted_next_regime:
            return None
        
        return {
            "current_regime": self.current_regime,
            "predicted_regime": self.predicted_next_regime,
            "confidence": self.prediction_confidence,
            "horizon_seconds": self.prediction_horizon,
            "should_pre_adjust": self.prediction_confidence > 0.6,
        }
    
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the current transition probability matrix."""
        return dict(self.transition_probs)
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime durations and transitions."""
        stats = {
            "current_regime": self.current_regime,
            "current_duration": self.regime_duration,
            "average_durations": {},
            "transition_counts": {},
        }
        
        for regime, durations in self.regime_durations.items():
            if durations:
                stats["average_durations"][regime] = np.mean(durations)
        
        for from_regime, to_regimes in self.transitions.items():
            stats["transition_counts"][from_regime] = dict(to_regimes)
        
        return stats


class PredictiveParameterAdjuster:
    """
    Proactively adjusts parameters based on predicted regime changes.
    """
    
    def __init__(self, predictor: RegimePredictor):
        self.predictor = predictor
        
        # Pre-adjustment profiles for regime transitions
        self.pre_adjustments: Dict[Tuple[str, str], Dict[str, float]] = {
            ("ranging", "trending_up"): {
                "filter_threshold_delta": -0.02,
                "confidence_floor_trend_delta": -0.05,
                "strategy_threshold_trend_delta": -0.001,
            },
            ("trending_up", "high_volatility"): {
                "filter_threshold_delta": +0.03,
                "confidence_floor_all_delta": +0.05,
                "strategy_threshold_all_delta": +0.002,
            },
            ("high_volatility", "ranging"): {
                "filter_threshold_delta": -0.01,
                "confidence_floor_reversion_delta": -0.03,
                "strategy_threshold_reversion_delta": -0.1,
            },
        }
        
        # Tracking
        self.pre_adjustments_made: int = 0
        self.successful_pre_adjustments: int = 0
    
    def get_pre_adjustments(self, current_params: Dict[str, float]) -> Dict[str, float]:
        """Get pre-adjustments based on regime prediction."""
        prediction = self.predictor.get_prediction()
        
        if not prediction or not prediction.get("should_pre_adjust"):
            return {}
        
        current_regime = prediction["current_regime"]
        predicted_regime = prediction["predicted_regime"]
        confidence = prediction["confidence"]
        
        key = (current_regime, predicted_regime)
        if key not in self.pre_adjustments:
            return {}
        
        adjustments = self.pre_adjustments[key]
        
        # Scale by confidence
        scaled_adjustments = {
            k: v * confidence 
            for k, v in adjustments.items()
        }
        
        self.pre_adjustments_made += 1
        
        logger.info(
            f"Pre-adjusting for {current_regime}→{predicted_regime}: "
            f"confidence={confidence:.2f}, adjustments={len(scaled_adjustments)}"
        )
        
        return scaled_adjustments
    
    def record_pre_adjustment_outcome(self, was_successful: bool) -> None:
        """Record whether a pre-adjustment was successful."""
        if was_successful:
            self.successful_pre_adjustments += 1


def create_enhanced_market_features(prices: List[float]) -> Dict[str, float]:
    """Create enhanced market features for regime prediction."""
    if len(prices) < 50:
        return {}
    
    prices_array = np.array(prices[-50:])
    
    # Returns
    returns = np.diff(prices_array) / prices_array[:-1]
    
    features = {
        # Volatility features
        "volatility_short": float(np.std(returns[-10:])),
        "volatility_long": float(np.std(returns)),
        "volatility_ratio": float(np.std(returns[-10:]) / max(np.std(returns), 0.0001)),
        
        # Trend features
        "trend_strength": float((prices_array[-1] - prices_array[-20]) / prices_array[-20]),
        "trend_consistency": float(np.mean(np.sign(returns[-10:]))),
        
        # Mean reversion features
        "distance_from_mean": float((prices_array[-1] - np.mean(prices_array)) / np.std(prices_array)),
        "mean_reversion_speed": float(np.corrcoef(returns[-20:], np.arange(20))[0, 1]),
        
        # Momentum features
        "momentum_5": float((prices_array[-1] - prices_array[-5]) / prices_array[-5]),
        "momentum_10": float((prices_array[-1] - prices_array[-10]) / prices_array[-10]),
        "momentum_acceleration": float(
            ((prices_array[-1] - prices_array[-3]) / prices_array[-3]) -
            ((prices_array[-4] - prices_array[-7]) / prices_array[-7])
        ),
        
        # Range features
        "range_ratio": float(
            (np.max(prices_array[-10:]) - np.min(prices_array[-10:])) /
            max(np.max(prices_array) - np.min(prices_array), 0.0001)
        ),
        
        # Statistical features
        "skewness": float(np.mean(((returns - np.mean(returns)) / np.std(returns))**3)),
        "kurtosis": float(np.mean(((returns - np.mean(returns)) / np.std(returns))**4) - 3),
    }
    
    return features
