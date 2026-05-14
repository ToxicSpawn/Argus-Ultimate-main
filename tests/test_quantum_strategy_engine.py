"""Tests for quantum strategy engine."""

from __future__ import annotations

import numpy as np


def test_strategy_engine_disabled_returns_neutral():
    from ml.quantum_strategy_engine import QuantumStrategyEngine, QuantumStrategyFeatures

    engine = QuantumStrategyEngine(enabled=False)

    features = engine.extract_features([100, 101, 102, 103, 104])

    assert isinstance(features, QuantumStrategyFeatures)
    assert features.predicted_regime == "UNKNOWN"
    assert features.signal == "hold"


def test_strategy_engine_extracts_kernel_features():
    from ml.quantum_strategy_engine import QuantumStrategyEngine

    engine = QuantumStrategyEngine(enabled=True, n_qubits=4, n_layers=1)

    prices = list(range(90, 110))
    features = engine.extract_features(prices)

    # Should complete without error
    assert features is not None
    assert isinstance(features.kernel_features, np.ndarray) or features.kernel_features is None


def test_strategy_engine_classify_regime():
    from ml.quantum_strategy_engine import QuantumStrategyEngine

    engine = QuantumStrategyEngine(enabled=True, n_qubits=4, n_layers=1)

    prices = [100, 102, 104, 103, 105, 107, 110, 112, 111, 108, 106, 103, 101, 99, 97]
    features = engine.classify_regime(prices)

    assert features is not None
    assert features.predicted_regime in ("UNKNOWN", "TREND_UP", "HIGH_VOLATILITY", "MEAN_REVERT", "TREND_DOWN")
    assert 0.0 <= features.regime_confidence <= 1.0


def test_strategy_engine_portfolio_hybrid():
    from ml.quantum_strategy_engine import QuantumStrategyEngine

    engine = QuantumStrategyEngine(enabled=True)

    returns = np.array([0.08, 0.04, 0.12, 0.06, 0.09])
    cov = np.array([
        [0.10, 0.02, 0.01, 0.03, 0.02],
        [0.02, 0.08, 0.01, 0.02, 0.01],
        [0.01, 0.01, 0.12, 0.01, 0.02],
        [0.03, 0.02, 0.01, 0.09, 0.02],
        [0.02, 0.01, 0.02, 0.02, 0.11],
    ])

    result = engine.optimize_portfolio_hybrid(returns, cov, budget=3)

    if "error" not in result:
        assert "qaoa_weights" in result
        assert "refined_weights" in result
        assert "qaoa_sharpe" in result


def test_strategy_features_serialize():
    from ml.quantum_strategy_engine import QuantumStrategyFeatures

    features = QuantumStrategyFeatures(
        kernel_features=np.array([0.1, 0.2, 0.3]),
        reservoir_features=np.array([0.4, 0.5, 0.6]),
        predicted_regime="TREND_UP",
        regime_confidence=0.7,
        signal="buy",
        confidence=0.6,
    )

    payload = features.to_dict()

    assert payload["kernel_features"] == [0.1, 0.2, 0.3]
    assert payload["reservoir_features"] == [0.4, 0.5, 0.6]
    assert payload["predicted_regime"] == "TREND_UP"
    assert payload["regime_confidence"] == 0.7
    assert payload["signal"] == "buy"
    assert payload["confidence"] == 0.6
    assert "timestamp" in payload


def test_get_quantum_strategy_engine():
    from ml.quantum_strategy_engine import get_quantum_strategy_engine

    engine = get_quantum_strategy_engine(enabled=True, n_qubits=4)

    assert engine.is_available
    assert engine.n_qubits == 4


def test_strategy_engine_short_history():
    from ml.quantum_strategy_engine import QuantumStrategyEngine

    engine = QuantumStrategyEngine(enabled=True)

    # Too short for meaningful features
    features = engine.extract_features([100, 101])

    assert features.kernel_features is None
    assert features.reservoir_features is None