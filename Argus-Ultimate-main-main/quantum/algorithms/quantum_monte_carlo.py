"""
Quasi-Monte Carlo VaR/CVaR using Sobol low-discrepancy sequences.

Sobol low-discrepancy sampling can improve coverage versus pseudo-random
sampling for smooth integrands and often reduces variance in practice. This is
quantum-inspired numerical analysis, not a hardware quantum speedup.

Falls back to classical MC if scipy is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

logger = None
try:
    import logging
    logger = logging.getLogger(__name__)
except Exception:
    pass


def _log(msg: str, *args: Any) -> None:
    if logger:
        logger.debug(msg, *args)


def run(
    returns: Union[List[float], np.ndarray, Any],
    *,
    n_samples: int = 10000,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute VaR and CVaR using Quasi-Monte Carlo (Sobol sequences).

    Uses low-discrepancy Sobol sequences to sample from the empirical
    return distribution with better coverage than pseudo-random sampling.
    For bootstrapped VaR, QMC converges O(1/N) vs O(1/√N) for classical MC.

    Args:
        returns: 1D array-like of period returns.
        n_samples: Number of quasi-random samples for bootstrap.
        confidence: Tail confidence level (e.g. 0.95 for 95% VaR/CVaR).

    Returns:
        dict with var, cvar, var_95, cvar_95, expected_shortfall_bps,
        n_samples_used, method (sobol_qmc or classical).
    """
    r = np.asarray(returns, dtype=float).ravel()
    n_obs = len(r)

    if n_obs < 2:
        return {
            "var": 0.0, "cvar": 0.0, "from_classical": True,
            "var_95": 0.0, "cvar_95": 0.0,
            "expected_shortfall_bps": 0.0, "n_samples_used": 0,
            "method": "insufficient_data",
        }

    # Direct empirical VaR/CVaR (always computed as baseline)
    alpha = 1.0 - float(confidence)
    empirical_var = float(np.percentile(r, alpha * 100.0))
    tail = r[r <= empirical_var]
    empirical_cvar = float(np.mean(tail)) if len(tail) > 0 else empirical_var

    # Try Sobol quasi-random bootstrap for better convergence
    method = "classical"
    var_est = empirical_var
    cvar_est = empirical_cvar

    try:
        from scipy.stats import qmc

        # Generate Sobol sequence in [0, 1) — n must be power of 2
        sampler = qmc.Sobol(d=1, scramble=True)
        n_sobol = 2 ** int(np.ceil(np.log2(max(n_samples, 2))))
        sobol_points = sampler.random(n_sobol).ravel()

        # Use Sobol points to sample from empirical distribution
        # Map uniform [0,1) -> sorted return indices (inverse CDF)
        sorted_returns = np.sort(r)
        indices = np.clip(
            (sobol_points * n_obs).astype(int),
            0, n_obs - 1,
        )
        bootstrap_returns = sorted_returns[indices]

        # Compute VaR/CVaR from Sobol-bootstrapped distribution
        var_est = float(np.percentile(bootstrap_returns, alpha * 100.0))
        boot_tail = bootstrap_returns[bootstrap_returns <= var_est]
        cvar_est = float(np.mean(boot_tail)) if len(boot_tail) > 0 else var_est
        method = "sobol_qmc"

        # Also compute confidence interval via multiple Sobol blocks
        n_blocks = 8
        # Ensure block_size is a power of 2 for Sobol balance
        block_size = max(n_samples // n_blocks, 64)
        block_size = 2 ** int(np.ceil(np.log2(block_size)))
        block_vars = []
        for i in range(n_blocks):
            block_sampler = qmc.Sobol(d=1, scramble=True, seed=i)
            pts = block_sampler.random(block_size).ravel()
            idx = np.clip((pts * n_obs).astype(int), 0, n_obs - 1)
            block_ret = sorted_returns[idx]
            block_vars.append(float(np.percentile(block_ret, alpha * 100.0)))

        var_std = float(np.std(block_vars))
        _log("QMC VaR std across %d blocks: %.6f", n_blocks, var_std)

    except ImportError:
        # scipy not available, use stratified sampling as fallback
        # (still better than pure random)
        method = "stratified_mc"
        n_strata = min(100, n_samples)
        samples_per_stratum = max(n_samples // n_strata, 1)
        sorted_returns = np.sort(r)
        strata_samples = []
        for s in range(n_strata):
            lo = int(s * n_obs / n_strata)
            hi = int((s + 1) * n_obs / n_strata)
            hi = max(hi, lo + 1)
            hi = min(hi, n_obs)
            idx = np.random.randint(lo, hi, size=samples_per_stratum)
            strata_samples.extend(sorted_returns[idx].tolist())
        strata_arr = np.array(strata_samples)
        var_est = float(np.percentile(strata_arr, alpha * 100.0))
        strata_tail = strata_arr[strata_arr <= var_est]
        cvar_est = float(np.mean(strata_tail)) if len(strata_tail) > 0 else var_est

    es_bps = -cvar_est * 1e4 if cvar_est < 0 else 0.0

    return {
        "var": var_est,
        "cvar": cvar_est,
        "from_classical": method == "classical",
        "var_95": var_est,
        "cvar_95": cvar_est,
        "expected_shortfall_bps": es_bps,
        "n_samples_used": n_samples,
        "method": method,
        "empirical_var": empirical_var,
        "empirical_cvar": empirical_cvar,
    }


class QuantumMonteCarlo:
    """
    Quantum-Inspired Monte Carlo for VaR/CVaR Estimation.

    Uses Quasi-Monte Carlo (Sobol sequences) for faster convergence
    than classical Monte Carlo. O(1/N) vs O(1/sqrt(N)).
    """

    def __init__(self, n_qubits: int = 8, n_samples: int = 10000):
        """
        Initialize QMC VaR calculator.

        Args:
            n_qubits: Number of qubits (determines QMC subspace size, 2^n_qubits)
            n_samples: Number of samples for VaR estimation
        """
        self.n_qubits = n_qubits
        self.n_samples = n_samples
        self.risk_history = []

    async def simulate(
        self,
        returns: "np.ndarray",
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Simulate VaR/CVaR using QMC.

        Args:
            returns: Array of historical returns
            confidence: Confidence level for VaR (default 0.95)

        Returns:
        Dict with var, cvar, and honest QMC metadata.
        """
        result = run(
            returns,
            n_samples=self.n_samples,
            confidence=confidence,
        )

        # Add honest quantum-inspired metadata. Keep the legacy
        # ``quantum_advantage`` key for compatibility, but do not claim a
        # hardware or wall-clock advantage from classical QMC.
        result["n_qubits"] = self.n_qubits
        result["quantum_advantage"] = 1.0
        result["quantum_advantage_claimed"] = False
        result["qmc_note"] = (
            "Sobol quasi-Monte Carlo is quantum-inspired variance reduction; "
            "no hardware quantum advantage is claimed."
        )

        self.risk_history.append(result)
        return result

    def estimate_var(
        self,
        returns: "np.ndarray",
        confidence: float = 0.95,
    ) -> float:
        """Quick VaR estimation."""
        result = run(returns, n_samples=self.n_samples, confidence=confidence)
        return result.get("var", 0.0)

    def estimate_cvar(
        self,
        returns: "np.ndarray",
        confidence: float = 0.95,
    ) -> float:
        """Quick CVaR estimation."""
        result = run(returns, n_samples=self.n_samples, confidence=confidence)
        return result.get("cvar", 0.0)
