"""Tests for quantum regime classifier."""

from __future__ import annotations

import numpy as np


def test_regime_classifier_initialization():
    from ml.quantum_regime_classifier import QuantumRegimeClassifier

    clf = QuantumRegimeClassifier(n_qubits=4, n_layers=2, seed=42)

    assert clf.n_qubits == 4
    assert clf.n_layers == 2
    assert clf.is_available


def test_regime_prediction():
    from ml.quantum_regime_classifier import QuantumRegimeClassifier, RegimePrediction

    clf = QuantumRegimeClassifier(n_qubits=4, seed=42)

    features = np.array([0.01, 0.02, -0.01, 0.03])
    result = clf.predict(features)

    assert isinstance(result, RegimePrediction)
    assert result.regime in clf.REGIMES + ["UNKNOWN"]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == len(clf.REGIMES)


def test_classify_returns():
    from ml.quantum_regime_classifier import QuantumRegimeClassifier

    clf = QuantumRegimeClassifier(n_qubits=4, seed=42)

    returns = [0.01, 0.02, -0.01, 0.03, 0.02, -0.02, 0.01, -0.01, 0.02, 0.01]
    result = clf.classify_returns(returns)

    assert result.regime in clf.REGIMES + ["UNKNOWN"]
    assert result.confidence >= 0.0


def test_vqc_fit():
    from ml.quantum_regime_classifier import QuantumRegimeClassifier

    clf = QuantumRegimeClassifier(n_qubits=4, n_layers=2, seed=42)

    # Simple training data: regime labels
    X = np.array([
        [0.01, 0.02, 0.01, 0.02],
        [0.02, 0.03, 0.02, 0.03],
        [-0.01, -0.02, -0.01, -0.02],
        [-0.02, -0.03, -0.02, -0.03],
        [0.001, 0.002, 0.001, 0.002],
        [-0.001, -0.002, -0.001, -0.002],
    ])
    y = np.array([0, 0, 1, 1, 2, 2])  # TREND_UP, TREND_DOWN, LOW_VOLATILITY

    clf.fit(X, y, lr=0.1, n_iter=10)

    assert clf._fitted
    assert clf._params is not None


def test_regime_prediction_short_returns():
    from ml.quantum_regime_classifier import QuantumRegimeClassifier

    clf = QuantumRegimeClassifier(n_qubits=4)

    result = clf.classify_returns([0.01])

    assert result.regime == "UNKNOWN"
    assert result.confidence == 0.0


def test_regime_serialization():
    from ml.quantum_regime_classifier import RegimePrediction

    pred = RegimePrediction(
        regime="TREND_UP",
        confidence=0.8,
        probabilities={"TREND_UP": 0.8, "TREND_DOWN": 0.1, "MEAN_REVERT": 0.1},
        features=[0.01, 0.02, 0.03, 0.04],
    )

    payload = pred.to_dict()

    assert payload["regime"] == "TREND_UP"
    assert payload["confidence"] == 0.8
    assert "no quantum advantage claimed" in payload["honest_claim"]
    assert "vqc_classifier" in payload["method"]


def test_simple_regime_classify():
    from ml.quantum_regime_classifier import simple_regime_classify, QuantumRegimeClassifier
    REGIMES = QuantumRegimeClassifier.REGIMES

    returns = [0.01, 0.02, 0.03, 0.02, 0.01]
    result = simple_regime_classify(returns)

    assert result.regime in REGIMES + ["UNKNOWN"]
    assert result.confidence >= 0.0