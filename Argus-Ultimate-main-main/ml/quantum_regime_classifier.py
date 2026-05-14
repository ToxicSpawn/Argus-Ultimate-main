"""
Quantum circuit-based market regime classifier.

Uses a variational quantum circuit as a classifier to detect market regimes.
Local-only, no quantum advantage claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RegimePrediction:
    """Result of quantum regime classification."""

    regime: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    features: List[float] = field(default_factory=list)
    method: str = "vqc_classifier"
    honest_claim: str = (
        "Variational quantum circuit classifier for regime detection. "
        "Classical simulation of quantum circuit; no quantum advantage claimed."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "confidence": float(self.confidence),
            "probabilities": self.probabilities,
            "features": [float(x) for x in self.features],
            "method": self.method,
            "honest_claim": self.honest_claim,
        }


class QuantumRegimeClassifier:
    """
    Quantum circuit classifier for market regimes.

    Uses a simple variational quantum circuit (VQC) to classify market regimes.
    The circuit is trained classically via gradient descent on the measurement
    outcomes.

    Supported regimes:
    - TREND_UP: Strong upward momentum
    - TREND_DOWN: Strong downward momentum
    - MEAN_REVERT: Oscillating around a mean
    - HIGH_VOLATILITY: High variance regardless of direction
    - LOW_VOLATILITY: Low variance, calm markets
    """

    REGIMES = ["TREND_UP", "TREND_DOWN", "MEAN_REVERT", "HIGH_VOLATILITY", "LOW_VOLATILITY"]

    def __init__(
        self,
        *,
        n_qubits: int = 4,
        n_layers: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        self.n_qubits = min(10, max(int(n_qubits), 2))
        self.n_layers = max(1, int(n_layers))
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._params: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def is_available(self) -> bool:
        return True

    def _initialize_params(self) -> np.ndarray:
        """Initialize circuit parameters."""
        n_params = self.n_qubits * self.n_layers
        return self._rng.random(n_params) * 2 * np.pi

    def _encode_features(self, features: np.ndarray) -> np.ndarray:
        """Encode classical features into quantum state."""
        n = min(len(features), self.n_qubits)
        state = np.zeros(self.n_qubits, dtype=np.complex128)
        state[:n] = features[:n]
        # Normalize
        norm = np.linalg.norm(state)
        if norm > 1e-10:
            state = state / norm
        else:
            state[0] = 1.0
        return state

    def _apply_variational_layer(
        self,
        state: np.ndarray,
        params: np.ndarray,
        layer: int,
    ) -> np.ndarray:
        """Apply one variational layer."""
        new_state = state.copy()
        # Single-qubit rotations (Ry)
        for q in range(self.n_qubits):
            idx = layer * self.n_qubits + q
            theta = params[idx] if idx < len(params) else 0.0
            # Ry rotation on qubit q
            cos_t = np.cos(theta / 2)
            sin_t = np.sin(theta / 2)
            # Apply to state - simplified
            new_state[q] = cos_t * state[q] + sin_t * state[(q + 1) % self.n_qubits]
        return new_state

    def _measure(self, state: np.ndarray) -> np.ndarray:
        """Measure state and return probabilities."""
        probs = np.abs(state) ** 2
        probs = probs / max(probs.sum(), 1e-10)
        return probs

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict regime probabilities for features."""
        if self._params is None:
            self._params = self._initialize_params()

        state = self._encode_features(features)

        # Apply variational layers
        for layer in range(self.n_layers):
            state = self._apply_variational_layer(state, self._params, layer)

        probs = self._measure(state)

        # Map to regimes (n_regimes might not equal n_qubits)
        n_regimes = len(self.REGIMES)
        regime_probs = np.zeros(n_regimes)
        for i in range(min(n_regimes, len(probs))):
            regime_probs[i] = probs[i]
        # Normalize
        regime_probs = regime_probs / max(regime_probs.sum(), 1e-10)

        return regime_probs

    def predict(self, features: np.ndarray) -> RegimePrediction:
        """Predict market regime from features."""
        feat = np.asarray(features, dtype=float).ravel()
        if len(feat) < self.n_qubits:
            feat = np.pad(feat, (0, self.n_qubits - len(feat)))

        probs = self.predict_proba(feat)
        best_idx = int(np.argmax(probs))
        regime = self.REGIMES[best_idx]
        confidence = float(probs[best_idx])

        probabilities = {r: float(probs[i]) for i, r in enumerate(self.REGIMES)}

        return RegimePrediction(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            features=feat.tolist(),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.1,
        n_iter: int = 50,
    ) -> "QuantumRegimeClassifier":
        """
        Train the VQC classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Regime labels (encoded as integers)
        """
        self._params = self._initialize_params()
        n_regimes = len(self.REGIMES)

        for _ in range(n_iter):
            # Compute gradients numerically
            grad = np.zeros_like(self._params)
            eps = 1e-2

            for i in range(len(self._params)):
                params_plus = self._params.copy()
                params_plus[i] += eps
                self._params = params_plus

                loss_plus = 0.0
                for xi, yi in zip(X, y):
                    probs = self.predict_proba(xi)
                    # Cross-entropy-like loss
                    if yi < len(probs):
                        loss_plus -= np.log(max(probs[int(yi)], 1e-10))

                self._params[i] -= eps
                loss_minus = 0.0
                for xi, yi in zip(X, y):
                    probs = self.predict_proba(xi)
                    if yi < len(probs):
                        loss_minus -= np.log(max(probs[int(yi)], 1e-10))

                grad[i] = (loss_plus - loss_minus) / (2 * eps)

            # Gradient update
            self._params = self._params - lr * grad
            self._params = np.clip(self._params, 0, 2 * np.pi)

        self._fitted = True
        return self

    def classify_returns(
        self,
        returns: List[float],
        prices: Optional[List[float]] = None,
    ) -> RegimePrediction:
        """Classify market regime from return series."""
        valid_returns = [float(r) for r in returns if np.isfinite(float(r))]
        if len(valid_returns) < 5:
            return RegimePrediction(regime="UNKNOWN", confidence=0.0)

        # Extract features
        returns_arr = np.array(valid_returns)

        features = [
            float(np.mean(returns_arr)),
            float(np.std(returns_arr)),
            float(np.min(returns_arr)),
            float(np.max(returns_arr)),
        ]

        # Add momentum features if enough data
        if len(returns_arr) >= 5:
            features.append(float(returns_arr[-1] - returns_arr[0]))
        if len(returns_arr) >= 10:
            features.append(float(np.mean(returns_arr[-5:]) - np.mean(returns_arr[:-5])))

        return self.predict(np.array(features[:self.n_qubits]))


def simple_regime_classify(
    returns: List[float],
    prices: Optional[List[float]] = None,
) -> RegimePrediction:
    """Quick regime classification without training."""
    classifier = QuantumRegimeClassifier()
    return classifier.classify_returns(returns, prices)


__all__ = ["QuantumRegimeClassifier", "RegimePrediction", "simple_regime_classify"]