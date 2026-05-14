"""Tests for quantum hyperparameter optimizer."""

from __future__ import annotations


def test_discrete_optimization():
    from ml.quantum_hyperopt import QuantumHyperOptimizer

    optimizer = QuantumHyperOptimizer(max_evals=20, seed=42)

    param_grid = {
        "lr": [0.01, 0.1, 0.5],
        "layers": [1, 2, 3],
    }

    def objective(params):
        # Simple objective: reward lower lr and more layers
        return params["lr"] * 0.1 + params["layers"] * 0.5

    result = optimizer.optimize(param_grid, objective, maximize=True)

    assert result.best_params is not None
    assert "lr" in result.best_params
    assert "layers" in result.best_params
    assert len(result.search_history) > 0
    assert "qaoa_hyperopt" in result.method


def test_continuous_optimization():
    from ml.quantum_hyperopt import QuantumHyperOptimizer

    optimizer = QuantumHyperOptimizer(max_evals=30, seed=42)

    param_ranges = {
        "threshold": (0.0, 1.0),
        "decay": (0.9, 1.0),
    }

    def objective(params):
        return params["threshold"] * params["decay"]

    result = optimizer.optimize_continuous(param_ranges, objective, maximize=True)

    assert result.best_params is not None
    assert 0.0 <= result.best_params["threshold"] <= 1.0
    assert 0.9 <= result.best_params["decay"] <= 1.0


def test_quick_optimize():
    from ml.quantum_hyperopt import quick_optimize

    param_grid = {"k": [1, 2, 3], "p": [0.5, 1.0]}

    def objective(params):
        return params["k"] + params["p"]

    result = quick_optimize(param_grid, objective, maximize=True)

    assert result.best_score >= 0.0
    assert "k" in result.best_params


def test_hyperopt_serialization():
    from ml.quantum_hyperopt import QuantumHyperOptimizer

    optimizer = QuantumHyperOptimizer(max_evals=10, seed=42)

    param_grid = {"a": [1, 2]}

    def objective(params):
        return params["a"]

    result = optimizer.optimize(param_grid, objective)
    payload = result.to_dict()

    assert "best_params" in payload
    assert "best_score" in payload
    assert "search_history" in payload
    assert "no quantum speedup claimed" in payload["honest_claim"]


def test_fallback_on_empty_combinations():
    from ml.quantum_hyperopt import QuantumHyperOptimizer

    optimizer = QuantumHyperOptimizer(max_evals=5)

    # Empty grid - should not crash
    result = optimizer.optimize({}, lambda x: 0.0)

    assert result.best_params == {}
    assert result.best_score == 0.0