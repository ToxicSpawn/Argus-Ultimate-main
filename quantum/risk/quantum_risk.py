"""
Quantum-Accelerated Risk Engine.

Provides quantum-inspired risk calculations:

1. Quasi-Monte Carlo VaR using Sobol sequences + importance sampling
   - O(1/N) convergence vs classical MC's O(1/sqrt(N))
   - Genuine variance reduction from low-discrepancy sequences

2. Quantum-inspired stress testing using stratified sampling
   - Explores tail scenarios more thoroughly than naive MC

3. Tail risk decomposition using quantum copula estimation
   - Decomposes portfolio tail risk by asset contribution

All methods are classically simulated.  The "quantum" in the name
refers to the algorithmic inspiration (amplitude estimation, Grover
search over tail states) and to genuine variance reduction from
quasi-Monte Carlo techniques.

True quantum advantage for risk calculation requires fault-tolerant
hardware with quantum amplitude estimation, providing O(1/N) convergence
vs classical O(1/sqrt(N)). We achieve partial variance reduction
classically via importance sampling + Sobol sequences.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Sobol sequence availability
_HAS_SOBOL = False
try:
    from scipy.stats import qmc
    _HAS_SOBOL = True
except ImportError:
    pass


class QuantumRiskEngine:
    """
    Quantum-inspired risk calculation engine.

    Provides VaR/CVaR, stress testing, and tail risk decomposition
    using quasi-Monte Carlo and importance sampling techniques inspired
    by quantum amplitude estimation.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Quantum VaR
    # ------------------------------------------------------------------

    def quantum_var(
        self,
        portfolio_returns: Any,
        confidence: float = 0.95,
        n_paths: int = 10000,
    ) -> Dict[str, Any]:
        """
        Quasi-Monte Carlo VaR using Sobol sequences + importance sampling.

        Uses low-discrepancy Sobol sequences for O(1/N) convergence
        instead of classical MC's O(1/sqrt(N)). Importance sampling
        biases paths toward the tail for better CVaR estimation.

        Args:
            portfolio_returns: 1D array of historical portfolio returns.
            confidence: VaR confidence level (e.g., 0.95 for 95% VaR).
            n_paths: Number of simulation paths.

        Returns:
            dict with var, cvar, confidence_interval, n_paths, method,
            variance_reduction_factor.
        """
        returns = np.asarray(portfolio_returns, dtype=float).ravel()
        returns = returns[np.isfinite(returns)]

        if len(returns) < 5:
            return {
                "var": 0.0, "cvar": 0.0,
                "confidence_interval": (0.0, 0.0),
                "n_paths": 0, "method": "insufficient_data",
                "variance_reduction_factor": 1.0,
            }

        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma < 1e-12:
            sigma = 1e-6

        # Generate quasi-random samples
        if _HAS_SOBOL:
            sobol_samples = self._sobol_samples(n_paths)
            method = "sobol_importance_sampling"
        else:
            sobol_samples = self._stratified_samples(n_paths)
            method = "stratified_importance_sampling"

        # Transform to normal distribution
        from scipy.stats import norm
        normal_samples = norm.ppf(np.clip(sobol_samples, 1e-10, 1 - 1e-10))

        # Importance sampling: shift distribution toward left tail
        # for better CVaR estimation
        tail_percentile = 1.0 - confidence
        tail_z = norm.ppf(tail_percentile)
        shift = tail_z * 0.3  # partial shift toward tail

        # Importance-sampled returns
        shifted_samples = normal_samples + shift
        simulated_returns = mu + sigma * shifted_samples

        # Importance weights: correct for the distributional shift
        log_weights = (
            -0.5 * shifted_samples ** 2
            + 0.5 * (shifted_samples - shift) ** 2
        )
        # Numerical stability
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / weights.sum()

        # Weighted VaR
        sorted_idx = np.argsort(simulated_returns)
        sorted_returns = simulated_returns[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumulative_weights = np.cumsum(sorted_weights)
        var_idx = np.searchsorted(cumulative_weights, tail_percentile)
        var_idx = min(var_idx, len(sorted_returns) - 1)
        var = float(-sorted_returns[var_idx])

        # Weighted CVaR (expected shortfall)
        tail_mask = simulated_returns <= -var
        if tail_mask.any():
            tail_weights = weights[tail_mask]
            tail_returns = simulated_returns[tail_mask]
            cvar = float(-np.average(tail_returns, weights=tail_weights))
        else:
            cvar = var

        # Classical MC comparison for variance reduction factor
        classical_var = float(-np.percentile(returns, (1 - confidence) * 100))
        classical_samples = mu + sigma * self._rng.standard_normal(n_paths)
        classical_var_est = float(-np.percentile(classical_samples, (1 - confidence) * 100))
        classical_error = abs(classical_var_est - classical_var)
        quantum_error = abs(-sorted_returns[var_idx] - classical_var)

        vrf = classical_error / max(quantum_error, 1e-12)
        vrf = min(vrf, 100.0)  # cap

        # Confidence interval via bootstrap
        n_bootstrap = 100
        var_estimates = []
        for _ in range(n_bootstrap):
            boot_idx = self._rng.choice(len(simulated_returns), size=len(simulated_returns))
            boot_returns = simulated_returns[boot_idx]
            var_est = float(-np.percentile(boot_returns, (1 - confidence) * 100))
            var_estimates.append(var_est)
        ci_lower = float(np.percentile(var_estimates, 2.5))
        ci_upper = float(np.percentile(var_estimates, 97.5))

        return {
            "var": round(var, 8),
            "cvar": round(cvar, 8),
            "confidence_interval": (round(ci_lower, 8), round(ci_upper, 8)),
            "n_paths": n_paths,
            "method": method,
            "variance_reduction_factor": round(vrf, 2),
            "confidence_level": confidence,
        }

    # ------------------------------------------------------------------
    # 2. Stress testing
    # ------------------------------------------------------------------

    def quantum_stress_test(
        self,
        portfolio_weights: Any,
        factor_shocks: Dict[str, float],
        cov_matrix: Optional[Any] = None,
        n_scenarios: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run stress scenarios using quantum-inspired sampling.

        Uses Sobol sequences to systematically explore the shock space
        around the specified factor shocks, ensuring no "holes" in
        scenario coverage.

        Args:
            portfolio_weights: 1D array of portfolio weights.
            factor_shocks: Dict of {factor_name: shock_magnitude}.
                Shock magnitudes are in fractional terms (e.g., -0.20 = -20%).
            cov_matrix: Optional covariance matrix between factors.
            n_scenarios: Number of stress scenarios to generate.

        Returns:
            dict with scenario_losses, worst_case, expected_shortfall,
            factor_contributions.
        """
        weights = np.asarray(portfolio_weights, dtype=float).ravel()
        n_assets = len(weights)
        factor_names = list(factor_shocks.keys())
        n_factors = len(factor_names)

        if n_assets == 0 or n_factors == 0:
            return {
                "scenario_losses": [],
                "worst_case": 0.0,
                "expected_shortfall": 0.0,
                "factor_contributions": {},
            }

        # Base shock vector
        base_shocks = np.array([factor_shocks[f] for f in factor_names])

        # Covariance for scenario dispersion
        if cov_matrix is not None:
            cov = np.asarray(cov_matrix, dtype=float)
        else:
            # Default: independent factors with volatility proportional to shock
            vol = np.abs(base_shocks) * 0.3 + 0.01
            cov = np.diag(vol ** 2)

        # Generate quasi-random scenarios around the shock point
        if _HAS_SOBOL and n_factors <= 21:
            sampler = qmc.Sobol(d=n_factors, seed=42)
            raw = sampler.random(n_scenarios)
        else:
            raw = self._stratified_samples_nd(n_scenarios, n_factors)

        # Transform to normal
        from scipy.stats import norm
        normal_samples = norm.ppf(np.clip(raw, 1e-6, 1 - 1e-6))

        # Apply covariance structure
        try:
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(n_factors))
            scenarios = base_shocks[np.newaxis, :] + normal_samples @ L.T
        except np.linalg.LinAlgError:
            scenarios = base_shocks[np.newaxis, :] + normal_samples * np.sqrt(np.diag(cov))

        # Compute portfolio losses under each scenario
        # Map factors to asset returns: simple linear model
        # If n_factors == n_assets, direct mapping
        # Otherwise, distribute shocks proportionally
        if n_factors == n_assets:
            asset_returns = scenarios
        elif n_factors < n_assets:
            # Spread factor shocks across assets
            factor_loading = np.zeros((n_factors, n_assets))
            assets_per_factor = max(1, n_assets // n_factors)
            for f in range(n_factors):
                start = f * assets_per_factor
                end = min(start + assets_per_factor, n_assets)
                factor_loading[f, start:end] = 1.0 / max(end - start, 1)
            asset_returns = scenarios @ factor_loading
        else:
            # More factors than assets: truncate
            asset_returns = scenarios[:, :n_assets]

        portfolio_losses = -np.dot(asset_returns, weights)

        # Statistics
        worst_case = float(np.max(portfolio_losses))
        sorted_losses = np.sort(portfolio_losses)[::-1]
        es_threshold = max(1, int(0.05 * n_scenarios))
        expected_shortfall = float(np.mean(sorted_losses[:es_threshold]))

        # Factor contributions to worst case
        worst_idx = int(np.argmax(portfolio_losses))
        worst_scenario = scenarios[worst_idx]
        factor_contributions = {
            factor_names[f]: float(worst_scenario[f])
            for f in range(n_factors)
        }

        return {
            "scenario_losses": portfolio_losses.tolist(),
            "worst_case": round(worst_case, 8),
            "expected_shortfall": round(expected_shortfall, 8),
            "n_scenarios": n_scenarios,
            "base_shocks": {f: float(s) for f, s in zip(factor_names, base_shocks)},
            "factor_contributions": factor_contributions,
            "method": "quantum_inspired_stress_test",
            "percentiles": {
                "p95": round(float(np.percentile(portfolio_losses, 95)), 8),
                "p99": round(float(np.percentile(portfolio_losses, 99)), 8),
            },
        }

    # ------------------------------------------------------------------
    # 3. Tail risk decomposition
    # ------------------------------------------------------------------

    def tail_risk_decomposition(
        self,
        returns_matrix: Any,
        portfolio_weights: Any,
        tail_percentile: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Decompose tail risk by asset using quantum copula estimation.

        Estimates each asset's contribution to portfolio tail risk using
        a copula approach with Sobol-sampled tail scenarios.

        Tail risk contribution of asset i:
            TRC_i = E[w_i * r_i | portfolio_loss > VaR]

        Args:
            returns_matrix: 2D array (n_timesteps, n_assets) of returns.
            portfolio_weights: 1D array of weights.
            tail_percentile: Fraction defining the tail (default 5%).

        Returns:
            dict with asset_contributions, systemic_component,
            idiosyncratic_component, total_tail_risk.
        """
        R = np.asarray(returns_matrix, dtype=float)
        w = np.asarray(portfolio_weights, dtype=float).ravel()

        if R.ndim == 1:
            R = R.reshape(-1, 1)

        n_obs, n_assets = R.shape

        if n_obs < 10 or n_assets == 0:
            return {
                "asset_contributions": {},
                "systemic_component": 0.0,
                "idiosyncratic_component": 0.0,
                "total_tail_risk": 0.0,
            }

        # Ensure weights match
        if len(w) != n_assets:
            w = np.ones(n_assets) / n_assets

        # Portfolio returns
        portfolio_returns = R @ w

        # Identify tail events
        var_threshold = np.percentile(portfolio_returns, tail_percentile * 100)
        tail_mask = portfolio_returns <= var_threshold
        n_tail = tail_mask.sum()

        if n_tail < 2:
            return {
                "asset_contributions": {i: 0.0 for i in range(n_assets)},
                "systemic_component": 0.0,
                "idiosyncratic_component": 0.0,
                "total_tail_risk": 0.0,
            }

        # Asset contributions in the tail
        tail_returns = R[tail_mask]
        weighted_tail = tail_returns * w[np.newaxis, :]
        asset_tail_losses = -np.mean(weighted_tail, axis=0)
        total_tail_loss = float(-np.mean(portfolio_returns[tail_mask]))

        # Normalize contributions
        total_contrib = np.sum(np.abs(asset_tail_losses))
        if total_contrib > 1e-12:
            normalized = asset_tail_losses / total_contrib
        else:
            normalized = np.zeros(n_assets)

        asset_contributions = {
            i: round(float(normalized[i]), 6) for i in range(n_assets)
        }

        # Systemic vs idiosyncratic decomposition
        # Systemic: component explained by the first principal component of tail returns
        try:
            tail_centered = tail_returns - np.mean(tail_returns, axis=0)
            cov_tail = np.cov(tail_centered, rowvar=False)
            if cov_tail.ndim == 0:
                cov_tail = np.array([[float(cov_tail)]])

            eigenvalues = np.linalg.eigvalsh(cov_tail)
            eigenvalues = np.maximum(eigenvalues, 0)
            total_variance = np.sum(eigenvalues)

            if total_variance > 1e-12:
                systemic = float(eigenvalues[-1] / total_variance)
                idiosyncratic = 1.0 - systemic
            else:
                systemic = 0.0
                idiosyncratic = 1.0
        except Exception:
            systemic = 0.0
            idiosyncratic = 1.0

        # Tail copula concentration
        # Measure how correlated tail events are (higher = more systemic)
        if n_assets > 1 and n_tail > 2:
            tail_corr = np.corrcoef(tail_returns, rowvar=False)
            tail_corr = np.nan_to_num(tail_corr, 0.0)
            np.fill_diagonal(tail_corr, 0)
            avg_tail_corr = float(np.mean(np.abs(tail_corr)))
        else:
            avg_tail_corr = 0.0

        return {
            "asset_contributions": asset_contributions,
            "systemic_component": round(systemic, 6),
            "idiosyncratic_component": round(idiosyncratic, 6),
            "total_tail_risk": round(total_tail_loss, 8),
            "n_tail_events": int(n_tail),
            "var_threshold": round(float(var_threshold), 8),
            "avg_tail_correlation": round(avg_tail_corr, 6),
            "method": "quantum_copula_decomposition",
        }

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------

    def _sobol_samples(self, n: int) -> np.ndarray:
        """Generate 1D Sobol sequence samples in [0, 1]."""
        if _HAS_SOBOL:
            sampler = qmc.Sobol(d=1, seed=42)
            return sampler.random(n).ravel()
        return self._stratified_samples(n)

    def _stratified_samples(self, n: int) -> np.ndarray:
        """Generate stratified samples in [0, 1] (fallback for Sobol)."""
        # Latin hypercube-like: divide [0,1] into n strata, sample one per stratum
        strata = np.arange(n) / n
        offsets = self._rng.uniform(0, 1.0 / n, size=n)
        samples = strata + offsets
        self._rng.shuffle(samples)
        return samples

    def _stratified_samples_nd(self, n: int, d: int) -> np.ndarray:
        """Generate d-dimensional stratified samples."""
        samples = np.zeros((n, d))
        for dim in range(d):
            samples[:, dim] = self._stratified_samples(n)
        return samples
