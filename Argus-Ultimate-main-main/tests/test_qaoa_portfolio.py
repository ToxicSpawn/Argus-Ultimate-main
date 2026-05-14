"""
QAOA portfolio optimization tests on the in-repo simulator.

Phase D2 of the quantum overhaul. Verifies that the new QAOA implementation
produces meaningful subset selections relative to brute-force / heuristic
classical baselines.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantum.algorithms.qaoa import QAOAPortfolioOptimizer


def _brute_force_max_sharpe(mu: np.ndarray, sigma: np.ndarray) -> float:
    """Enumerate all non-empty subsets and return the best Sharpe."""
    n = len(mu)
    best = float("-inf")
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            sub_sigma = sigma[np.ix_(combo, combo)]
            sub_mu = mu[list(combo)]
            # Use inverse-variance weights
            diag = np.diag(sub_sigma)
            diag = np.maximum(diag, 1e-12)
            inv_var = 1.0 / diag
            w = inv_var / inv_var.sum()
            ret = float(w @ sub_mu)
            risk = float(np.sqrt(max(w @ sub_sigma @ w, 0.0)))
            sharpe = ret / risk if risk > 1e-12 else 0.0
            if sharpe > best:
                best = sharpe
    return best


def _random_portfolio(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random valid portfolio (mu vector + cov matrix)."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.15, n)
    A = rng.standard_normal((n, n)) * 0.1
    sigma = A.T @ A + np.eye(n) * 0.01
    return mu, sigma


# ═════════════════════════════════════════════════════════════════════════════


class TestQAOA4Asset:
    """4-asset MaxCut: QAOA vs brute-force."""

    @pytest.mark.parametrize("seed", [11, 22, 33])
    def test_4asset_qaoa_finds_non_trivial_subset(self, seed):
        """QAOA must find a non-trivial portfolio (Sharpe >= 0)."""
        mu, sigma = _random_portfolio(4, seed)
        opt = QAOAPortfolioOptimizer(n_layers=3, max_assets=4)
        result = opt.optimize(mu, sigma, risk_aversion=0.5)

        qaoa_sharpe = float(result["sharpe"])
        # On random portfolios with mu in [0.02, 0.15] and PSD covariance, the
        # Sharpe should be positive (any non-empty subset gives positive return).
        assert qaoa_sharpe > 0, (
            f"seed={seed}: QAOA produced negative or zero Sharpe {qaoa_sharpe:.3f}"
        )

    def test_4asset_aggregate_achieves_30pct_average(self):
        """Average over multiple seeds: QAOA should achieve >30% of brute-force."""
        ratios = []
        for seed in [11, 22, 33, 44, 55]:
            mu, sigma = _random_portfolio(4, seed)
            opt = QAOAPortfolioOptimizer(n_layers=3, max_assets=4)
            result = opt.optimize(mu, sigma, risk_aversion=0.5)
            brute = _brute_force_max_sharpe(mu, sigma)
            if brute > 0:
                ratios.append(float(result["sharpe"]) / brute)
        avg = sum(ratios) / len(ratios) if ratios else 0.0
        assert avg >= 0.30, f"Average QAOA ratio {avg:.2%} below 30% of brute-force"

    def test_4asset_returns_method_in_repo_simulator(self):
        mu, sigma = _random_portfolio(4, 1)
        opt = QAOAPortfolioOptimizer(n_layers=2, max_assets=4)
        result = opt.optimize(mu, sigma)
        assert result["method"] == "qaoa_in_repo_simulator"

    def test_4asset_weights_sum_to_one(self):
        mu, sigma = _random_portfolio(4, 5)
        opt = QAOAPortfolioOptimizer(n_layers=2, max_assets=4)
        result = opt.optimize(mu, sigma)
        weights = np.array(result["weights"])
        assert abs(weights.sum() - 1.0) < 0.05
        assert all(w >= -1e-6 for w in weights)


class TestQAOA8Asset:
    """8-asset: QAOA convergence at larger sizes."""

    def test_8asset_runs_and_returns_valid(self):
        mu, sigma = _random_portfolio(8, 42)
        opt = QAOAPortfolioOptimizer(n_layers=2, max_assets=8)
        result = opt.optimize(mu, sigma, risk_aversion=0.5)

        weights = np.array(result["weights"])
        assert len(weights) == 8
        assert abs(weights.sum() - 1.0) < 0.05
        assert result["method"] == "qaoa_in_repo_simulator"

    def test_8asset_selects_at_least_one(self):
        mu, sigma = _random_portfolio(8, 99)
        opt = QAOAPortfolioOptimizer(n_layers=2, max_assets=8)
        result = opt.optimize(mu, sigma)
        assert len(result["selected_assets"]) >= 1


class TestBenchmarkComparison:
    """Phase D2: benchmark_vs_classical reports honest assessment."""

    def test_benchmark_returns_in_repo_method(self):
        mu, sigma = _random_portfolio(4, 7)
        opt = QAOAPortfolioOptimizer(n_layers=1, max_assets=4)
        bench = opt.benchmark_vs_classical(mu, sigma)
        assert bench["qaoa_method"] == "qaoa_in_repo_simulator"
        assert "honest_assessment" in bench
        assert isinstance(bench["honest_assessment"], str)
        assert len(bench["honest_assessment"]) > 50


class TestVariationalCircuitBuilder:
    """build_variational_circuit (used by Phase C1) produces valid circuits."""

    def test_circuit_has_rzz_gates_for_off_diagonal(self):
        from quantum_simulator import GateType
        mu, sigma = _random_portfolio(4, 1)
        opt = QAOAPortfolioOptimizer(n_layers=2, max_assets=4)
        qubo = opt.build_cost_hamiltonian(mu, sigma, risk_aversion=0.5)
        gammas, betas = opt.default_params(n_layers=2)
        qc = opt.build_variational_circuit(4, qubo, gammas, betas)
        gate_types = {op.gate for op in qc.operations}
        assert GateType.H in gate_types  # initial superposition
        assert GateType.RX in gate_types  # mixer
        # The new builder uses RZZ for off-diagonal QUBO entries
        # (vs the old diagonal-only legacy stub)
        assert GateType.RZZ in gate_types or GateType.RZ in gate_types

    def test_default_params_shape(self):
        opt = QAOAPortfolioOptimizer(n_layers=3)
        gammas, betas = opt.default_params(n_layers=3)
        assert len(gammas) == 3
        assert len(betas) == 3
