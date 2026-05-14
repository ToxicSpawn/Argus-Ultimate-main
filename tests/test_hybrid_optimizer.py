"""Tests for hybrid quantum-classical portfolio optimization."""

from __future__ import annotations

import numpy as np


def test_hybrid_refiner_basic():
    from ml.hybrid_optimizer import QAOARefiner, HybridOptimizationResult

    refiner = QAOARefiner(use_scipy=False)

    returns = np.array([0.08, 0.04, 0.12])
    cov = np.array([
        [0.10, 0.02, 0.01],
        [0.02, 0.08, 0.01],
        [0.01, 0.01, 0.12],
    ])

    qaoa_weights = np.array([0.6, 0.25, 0.15])
    result = refiner.refine(qaoa_weights, returns, cov)

    assert isinstance(result, HybridOptimizationResult)
    assert len(result.qaoa_weights) == 3
    assert len(result.refined_weights) == 3
    assert abs(result.qaoa_weights.sum() - 1.0) < 0.01
    assert abs(result.refined_weights.sum() - 1.0) < 0.01
    assert result.qaoa_sharpe >= 0.0
    assert result.refined_sharpe >= 0.0
    assert result.improvement >= -1.0
    assert result.iterations >= 0
    assert "qaoa_scipy_refinement" in result.method


def test_hybrid_refiner_serializes():
    from ml.hybrid_optimizer import QAOARefiner

    refiner = QAOARefiner(use_scipy=False)

    returns = np.array([0.08, 0.04, 0.12])
    cov = np.array([
        [0.10, 0.02, 0.01],
        [0.02, 0.08, 0.01],
        [0.01, 0.01, 0.12],
    ])

    result = refiner.refine(np.array([0.4, 0.3, 0.3]), returns, cov)
    payload = result.to_dict()

    assert "qaoa_weights" in payload
    assert "refined_weights" in payload
    assert "qaoa_sharpe" in payload
    assert "refined_sharpe" in payload
    assert "improvement" in payload
    assert "iterations" in payload
    assert "convergence_history" in payload
    assert "No quantum speedup" in payload["honest_claim"]


def test_hybrid_convenience_function():
    from ml.hybrid_optimizer import hybrid_portfolio_optimize

    returns = np.array([0.08, 0.04, 0.12])
    cov = np.array([
        [0.10, 0.02, 0.01],
        [0.02, 0.08, 0.01],
        [0.01, 0.01, 0.12],
    ])

    result = hybrid_portfolio_optimize(returns, cov)

    assert result.refined_sharpe >= result.qaoa_sharpe - 0.01
    assert result.to_dict()["method"] in ("qaoa_scipy_refinement", "gradient_descent")


def test_canonical_optimize_hybrid():
    from quantum import get_quantum_facade

    returns = np.array([0.08, 0.04, 0.12])
    cov = np.array([
        [0.10, 0.02, 0.01],
        [0.02, 0.08, 0.01],
        [0.01, 0.01, 0.12],
    ])

    facade = get_quantum_facade(simulation_backend="statevector", noise_model="ideal")
    result = facade.optimize_hybrid(
        returns,
        cov,
        budget=2,
        n_layers=1,
        max_assets=3,
    )

    assert "qaoa_weights" in result
    assert "refined_weights" in result
    assert "qaoa_sharpe" in result
    assert "refined_sharpe" in result
    assert "improvement" in result
    assert "simulation_backend" in result
    assert result["quantum_metadata"]["capability"] == "hybrid_qaoa_refinement"
    assert "hybrid" in result["quantum_metadata"]["honest_claim"].lower()
    assert result["simulation_backend"] == "statevector"
    assert result["noise_model"] == "ideal"


def test_hybrid_fallback_on_error():
    from ml.hybrid_optimizer import QAOARefiner, HybridOptimizationResult

    refiner = QAOARefiner(use_scipy=False)

    # Empty arrays - should not crash
    returns = np.array([])
    cov = np.array([]).reshape(0, 0)

    result = refiner.refine(np.array([]), returns, cov)

    assert isinstance(result, HybridOptimizationResult)
    assert "qaoa_scipy_refinement" in result.method