"""
Regime Prediction Model for ARGUS.

Predicts the NEXT market regime before it happens using:
1. Markov transition matrix (base rate from observed transitions)
2. Feature-based adjustment (high vol + negative momentum -> CRISIS)
3. Time-in-regime factor (longer in current regime -> higher transition probability)

Usage:
    predictor = RegimePredictor(lookback_periods=50)
    predictor.update("TRENDING_UP", {"volatility": 0.02, "momentum": 0.5})
    prediction = predictor.predict_next()
    # {'predicted_regime': 'HIGH_VOL', 'confidence': 0.73, ...}
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known regimes
# ---------------------------------------------------------------------------
KNOWN_REGIMES = [
    "TRENDING_UP",
    "TRENDING_DOWN",
    "MEAN_REVERTING",
    "HIGH_VOL",
    "LOW_VOL",
    "CRISIS",
    "BREAKOUT",
    "UNKNOWN",
]

# Feature-based regime signals: (feature_condition, likely_regime, weight)
_FEATURE_RULES = [
    # High vol + negative momentum -> CRISIS
    (lambda f: f.get("volatility", 0) > 0.05 and f.get("momentum", 0) < -0.3, "CRISIS", 0.4),
    # Low vol + low momentum -> LOW_VOL
    (lambda f: f.get("volatility", 0) < 0.01 and abs(f.get("momentum", 0)) < 0.1, "LOW_VOL", 0.3),
    # Positive momentum + moderate vol -> TRENDING_UP
    (lambda f: f.get("momentum", 0) > 0.3 and f.get("volatility", 0) < 0.04, "TRENDING_UP", 0.3),
    # Negative momentum + moderate vol -> TRENDING_DOWN
    (lambda f: f.get("momentum", 0) < -0.2 and f.get("volatility", 0) < 0.04, "TRENDING_DOWN", 0.3),
    # Vol spike -> HIGH_VOL
    (lambda f: f.get("volatility", 0) > 0.04, "HIGH_VOL", 0.25),
    # Volume surge + vol expansion -> BREAKOUT
    (lambda f: f.get("volume_ratio", 1) > 2.0 and f.get("volatility", 0) > 0.03, "BREAKOUT", 0.3),
]


# ---------------------------------------------------------------------------
# RegimePredictor
# ---------------------------------------------------------------------------


class RegimePredictor:
    """
    Predicts the next market regime using Markov chains and feature signals.

    Parameters
    ----------
    lookback_periods : int
        Number of regime observations to retain (default 50).
    """

    def __init__(self, lookback_periods: int = 50) -> None:
        self._lookback = max(10, int(lookback_periods))

        # History: list of (timestamp, regime, features)
        self._regime_history: List[Tuple[float, str, dict]] = []

        # Transition counts: regime_from -> {regime_to: count}
        self._transition_matrix: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Time tracking
        self._current_regime: Optional[str] = None
        self._regime_start_time: float = 0.0

    # ── Update ────────────────────────────────────────────────────────────

    def update(self, current_regime: str, features: Optional[dict] = None) -> None:
        """
        Record current regime with market features.

        Parameters
        ----------
        current_regime : str
            The currently detected regime label.
        features : dict, optional
            Market features: volatility, momentum, correlation, volume, volume_ratio.
        """
        now = time.time()
        feat = dict(features) if features else {}

        # Record transition
        if self._current_regime is not None and current_regime != self._current_regime:
            self._transition_matrix[self._current_regime][current_regime] += 1
            self._regime_start_time = now

        if self._current_regime is None:
            self._regime_start_time = now

        self._current_regime = current_regime
        self._regime_history.append((now, current_regime, feat))

        # Trim history
        if len(self._regime_history) > self._lookback * 2:
            self._regime_history = self._regime_history[-self._lookback:]

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_next(self) -> dict:
        """
        Predict next regime using:
        1. Markov transition matrix (base rate)
        2. Feature-based adjustment
        3. Time-in-regime factor

        Returns
        -------
        dict with keys:
            predicted_regime, confidence, expected_transition_hours, features_driving
        """
        if not self._regime_history:
            return {
                "predicted_regime": "UNKNOWN",
                "confidence": 0.0,
                "expected_transition_hours": float("inf"),
                "features_driving": [],
            }

        current = self._current_regime or "UNKNOWN"
        now = time.time()

        # 1. Markov transition probabilities
        markov_probs = self._get_transition_probs(current)

        # 2. Feature-based adjustment
        latest_features = self._regime_history[-1][2] if self._regime_history else {}
        feature_probs, feature_signals = self._feature_adjustment(latest_features)

        # 3. Time-in-regime factor
        time_in_regime_hours = (now - self._regime_start_time) / 3600.0
        # Longer in regime -> higher probability of transition
        # Use sigmoid: p_transition = 1 / (1 + exp(-0.1 * (hours - 12)))
        transition_urgency = 1.0 / (1.0 + np.exp(-0.1 * (time_in_regime_hours - 12.0)))

        # Combine: 50% Markov, 30% features, 20% time-based
        combined: Dict[str, float] = {}
        all_regimes = set(list(markov_probs.keys()) + list(feature_probs.keys()) + KNOWN_REGIMES)

        for regime in all_regimes:
            m = markov_probs.get(regime, 0.0)
            f = feature_probs.get(regime, 0.0)
            combined[regime] = 0.50 * m + 0.30 * f + 0.20 * (
                transition_urgency if regime != current else (1.0 - transition_urgency)
            ) / max(len(all_regimes) - 1, 1)

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        # Find most likely next regime (excluding current if transition is likely)
        if transition_urgency > 0.5:
            # Exclude staying in current regime
            candidates = {k: v for k, v in combined.items() if k != current}
        else:
            candidates = combined

        if not candidates:
            candidates = combined

        predicted = max(candidates, key=candidates.get) if candidates else "UNKNOWN"
        confidence = candidates.get(predicted, 0.0)

        # Expected transition time based on historical durations
        expected_hours = self._estimate_transition_time(current)

        return {
            "predicted_regime": predicted,
            "confidence": round(float(confidence), 4),
            "expected_transition_hours": round(float(expected_hours), 2),
            "features_driving": feature_signals,
        }

    # ── Transition matrix ─────────────────────────────────────────────────

    def get_transition_matrix(self) -> dict:
        """Return full transition probability matrix."""
        result = {}
        for from_regime, transitions in self._transition_matrix.items():
            total = sum(transitions.values())
            if total > 0:
                result[from_regime] = {
                    to_regime: round(count / total, 4)
                    for to_regime, count in transitions.items()
                }
            else:
                result[from_regime] = {}
        return result

    # ── Pre-transition signals ────────────────────────────────────────────

    def get_pre_transition_signals(self) -> List[str]:
        """
        Return list of warning signals that a regime change is imminent.
        """
        signals = []
        if not self._regime_history:
            return signals

        current = self._current_regime or "UNKNOWN"
        now = time.time()
        hours_in_regime = (now - self._regime_start_time) / 3600.0

        # Time-based signal
        avg_duration = self._estimate_transition_time(current)
        if avg_duration > 0 and hours_in_regime > avg_duration * 0.8:
            signals.append(
                f"Time-in-regime ({hours_in_regime:.1f}h) approaching "
                f"historical average ({avg_duration:.1f}h)"
            )

        # Feature-based signals
        latest_features = self._regime_history[-1][2] if self._regime_history else {}
        for rule_fn, target_regime, _ in _FEATURE_RULES:
            try:
                if rule_fn(latest_features) and target_regime != current:
                    signals.append(
                        f"Feature conditions suggest transition to {target_regime}"
                    )
            except Exception as _e:
                logger.debug("regime_predictor error: %s", _e)

        # Volatility acceleration
        if len(self._regime_history) >= 5:
            recent_vols = [
                entry[2].get("volatility", 0)
                for entry in self._regime_history[-5:]
                if "volatility" in entry[2]
            ]
            if len(recent_vols) >= 3:
                vol_change = recent_vols[-1] - recent_vols[0]
                if vol_change > 0.02:
                    signals.append(
                        f"Volatility accelerating: +{vol_change:.4f} over last 5 observations"
                    )

        return signals

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_transition_probs(self, from_regime: str) -> Dict[str, float]:
        """Get transition probabilities from a given regime."""
        transitions = self._transition_matrix.get(from_regime, {})
        total = sum(transitions.values())
        if total == 0:
            # Uniform prior
            return {r: 1.0 / len(KNOWN_REGIMES) for r in KNOWN_REGIMES}
        return {regime: count / total for regime, count in transitions.items()}

    def _feature_adjustment(self, features: dict) -> Tuple[Dict[str, float], List[str]]:
        """
        Compute regime probabilities from feature rules.
        Returns (probabilities, list_of_triggered_signal_names).
        """
        scores: Dict[str, float] = defaultdict(float)
        triggered: List[str] = []

        for rule_fn, target_regime, weight in _FEATURE_RULES:
            try:
                if rule_fn(features):
                    scores[target_regime] += weight
                    triggered.append(f"{target_regime} ({weight:.2f})")
            except Exception as _e:
                logger.debug("regime_predictor error: %s", _e)

        total = sum(scores.values())
        if total > 0:
            probs = {k: v / total for k, v in scores.items()}
        else:
            probs = {}

        return probs, triggered

    def _estimate_transition_time(self, from_regime: str) -> float:
        """Estimate average hours before transitioning out of a regime."""
        durations = []
        i = 0
        while i < len(self._regime_history) - 1:
            ts, regime, _ = self._regime_history[i]
            if regime == from_regime:
                # Find when it changed
                j = i + 1
                while j < len(self._regime_history) and self._regime_history[j][1] == regime:
                    j += 1
                if j < len(self._regime_history):
                    duration_h = (self._regime_history[j][0] - ts) / 3600.0
                    durations.append(duration_h)
                i = j
            else:
                i += 1

        if durations:
            return float(np.mean(durations))
        return 24.0  # default 24h if no data

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for dashboard/logging."""
        prediction = self.predict_next()
        return {
            "current_regime": self._current_regime,
            "history_length": len(self._regime_history),
            "transition_count": sum(
                sum(v.values()) for v in self._transition_matrix.values()
            ),
            "prediction": prediction,
            "pre_transition_signals": self.get_pre_transition_signals(),
        }
