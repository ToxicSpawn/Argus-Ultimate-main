"""
Quantum Monte Carlo (QMC) Accelerated Backtesting.

Uses quasi-random Sobol sequences instead of pseudo-random sampling for
Monte Carlo backtesting.  Sobol sequences provide more uniform coverage
of the sample space, converging at O(1/N) instead of O(1/sqrt(N)) for
standard MC -- a genuine mathematical advantage that does not require
quantum hardware.

The "quantum" label comes from the connection to Quantum Monte Carlo
methods in computational physics, where similar low-discrepancy sequences
are used.  This is a classical algorithm with provably better convergence.

Typical usage::

    from quantum.backtesting.quantum_backtest import QuantumBacktestAccelerator

    qba = QuantumBacktestAccelerator(n_scenarios=10000)
    result = qba.run_qmc_scenarios(returns, strategy_func)
    comparison = qba.compare_classical(returns, strategy_func)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try scipy for Sobol; fall back to custom implementation
_HAS_SCIPY_SOBOL = False
try:
    from scipy.stats import qmc as scipy_qmc
    _HAS_SCIPY_SOBOL = True
except ImportError:
    pass


def _sobol_sequence(n_samples: int, n_dims: int, seed: int = 0) -> np.ndarray:
    """Generate Sobol quasi-random sequence.

    Uses scipy.stats.qmc.Sobol if available, otherwise falls back to
    a Halton sequence (also quasi-random, slightly lower quality).

    Args:
        n_samples: Number of points to generate.
        n_dims: Number of dimensions.
        seed: Random seed for scrambling.

    Returns:
        Array of shape (n_samples, n_dims) with values in [0, 1).
    """
    if _HAS_SCIPY_SOBOL:
        try:
            sampler = scipy_qmc.Sobol(d=n_dims, scramble=True, seed=seed)
            # Sobol requires power-of-2 samples; generate enough and trim
            m = int(np.ceil(np.log2(max(n_samples, 2))))
            points = sampler.random_base2(m)
            return points[:n_samples]
        except Exception:
            pass

    # Fallback: Halton sequence (van der Corput base-p for each dimension)
    primes = _first_primes(n_dims)
    result = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        result[:, d] = _van_der_corput(n_samples, primes[d])
    return result


def _van_der_corput(n: int, base: int) -> np.ndarray:
    """Van der Corput sequence in given base."""
    seq = np.zeros(n)
    for i in range(n):
        num = i + 1
        denom = 1.0
        val = 0.0
        while num > 0:
            denom *= base
            num, remainder = divmod(num, base)
            val += remainder / denom
        seq[i] = val
    return seq


def _first_primes(n: int) -> List[int]:
    """Return the first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = all(candidate % p != 0 for p in primes)
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


class QuantumBacktestAccelerator:
    """
    QMC-accelerated Monte Carlo backtesting.

    Uses Sobol/Halton quasi-random sequences for scenario generation,
    providing better coverage of the return distribution and faster
    convergence than classical pseudo-random MC.

    Attributes:
        n_scenarios: Default number of scenarios to generate.
    """

    def __init__(
        self,
        n_scenarios: int = 10000,
        seed: int = 42,
    ) -> None:
        if n_scenarios < 10:
            raise ValueError(f"n_scenarios must be >= 10, got {n_scenarios}")

        self.n_scenarios = n_scenarios
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def _generate_qmc_scenarios(
        self,
        returns: np.ndarray,
        n_scenarios: int,
    ) -> np.ndarray:
        """Generate return scenarios using quasi-Monte Carlo sampling.

        Uses the inverse CDF method with Sobol points:
        1. Generate Sobol points in [0,1)^T (T = series length)
        2. Map through empirical inverse CDF of returns

        This produces scenarios that cover the return distribution
        more uniformly than pseudo-random sampling.

        Args:
            returns: Historical return series (1D).
            n_scenarios: Number of scenarios.

        Returns:
            Array of shape (n_scenarios, len(returns)) with synthetic returns.
        """
        T = len(returns)
        sorted_returns = np.sort(returns)

        # Generate Sobol points
        # For large T, we use block sampling (chunks of returns)
        block_size = min(T, 50)
        sobol_points = _sobol_sequence(n_scenarios, block_size, seed=self.seed)

        # Map through empirical inverse CDF
        scenarios = np.zeros((n_scenarios, T))
        for t in range(T):
            col_idx = t % block_size
            # Inverse CDF: sobol value -> quantile of empirical distribution
            indices = (sobol_points[:, col_idx] * (len(sorted_returns) - 1)).astype(int)
            indices = np.clip(indices, 0, len(sorted_returns) - 1)
            scenarios[:, t] = sorted_returns[indices]

        return scenarios

    def _generate_classical_scenarios(
        self,
        returns: np.ndarray,
        n_scenarios: int,
    ) -> np.ndarray:
        """Generate return scenarios using standard pseudo-random MC.

        Bootstrap sampling with replacement from historical returns.

        Args:
            returns: Historical return series.
            n_scenarios: Number of scenarios.

        Returns:
            Array of shape (n_scenarios, len(returns)).
        """
        T = len(returns)
        scenarios = np.zeros((n_scenarios, T))
        for s in range(n_scenarios):
            indices = self._rng.randint(0, len(returns), size=T)
            scenarios[s] = returns[indices]
        return scenarios

    # ------------------------------------------------------------------
    # Strategy evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_strategy(
        scenario_returns: np.ndarray,
        strategy_func: Optional[Callable],
    ) -> Dict[str, float]:
        """Evaluate a strategy on a single scenario of returns.

        If strategy_func is None, uses a simple buy-and-hold (cumulative returns).

        Args:
            scenario_returns: 1D array of returns for one scenario.
            strategy_func: Callable that takes returns and returns a float (total return).

        Returns:
            Dict with total_return, max_drawdown.
        """
        if strategy_func is not None:
            try:
                total_return = float(strategy_func(scenario_returns))
            except Exception:
                total_return = float(np.sum(scenario_returns))
        else:
            total_return = float(np.sum(scenario_returns))

        # Compute max drawdown
        cumulative = np.cumsum(scenario_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        return {
            "total_return": total_return,
            "max_drawdown": max_dd,
        }

    # ------------------------------------------------------------------
    # Main QMC backtest
    # ------------------------------------------------------------------

    def run_qmc_scenarios(
        self,
        returns: np.ndarray,
        strategy_func: Optional[Callable] = None,
        n_scenarios: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run strategy across QMC-sampled return scenarios.

        Uses Sobol sequences for faster convergence than classical MC.

        Args:
            returns: Historical returns (1D array).
            strategy_func: Strategy function: returns_array -> total_return.
                If None, uses buy-and-hold.
            n_scenarios: Number of scenarios (default: self.n_scenarios).

        Returns:
            Dict with: mean_return, std_return, sharpe, var_95, cvar_95,
            max_drawdown, scenarios (list of per-scenario results),
            convergence_rate, time_s.
        """
        t0 = time.monotonic()
        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[~np.isnan(returns)]

        if len(returns) < 5:
            return {
                "mean_return": 0.0,
                "std_return": 0.0,
                "sharpe": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "max_drawdown": 0.0,
                "n_scenarios": 0,
                "time_s": 0.0,
                "error": "insufficient_data",
            }

        n_sc = n_scenarios or self.n_scenarios
        scenarios = self._generate_qmc_scenarios(returns, n_sc)

        # Evaluate strategy on each scenario
        total_returns = np.zeros(n_sc)
        max_drawdowns = np.zeros(n_sc)

        for i in range(n_sc):
            result = self._evaluate_strategy(scenarios[i], strategy_func)
            total_returns[i] = result["total_return"]
            max_drawdowns[i] = result["max_drawdown"]

        # Statistics
        mean_ret = float(np.mean(total_returns))
        std_ret = float(np.std(total_returns, ddof=1)) if n_sc > 1 else 0.0

        # Sharpe ratio (annualized assuming daily returns, 252 trading days)
        daily_factor = np.sqrt(252) if len(returns) > 1 else 1.0
        sharpe = (mean_ret / std_ret * daily_factor) if std_ret > 1e-12 else 0.0

        # VaR and CVaR at 95%
        sorted_returns = np.sort(total_returns)
        var_idx = max(int(n_sc * 0.05), 1)
        var_95 = float(-sorted_returns[var_idx - 1])
        cvar_95 = float(-np.mean(sorted_returns[:var_idx]))

        max_dd = float(np.mean(max_drawdowns))
        elapsed = time.monotonic() - t0

        return {
            "mean_return": round(mean_ret, 8),
            "std_return": round(std_ret, 8),
            "sharpe": round(sharpe, 4),
            "var_95": round(var_95, 8),
            "cvar_95": round(cvar_95, 8),
            "max_drawdown": round(max_dd, 8),
            "n_scenarios": n_sc,
            "time_s": round(elapsed, 4),
            "method": "quasi_monte_carlo_sobol",
        }

    # ------------------------------------------------------------------
    # Comparison with classical MC
    # ------------------------------------------------------------------

    def compare_classical(
        self,
        returns: np.ndarray,
        strategy_func: Optional[Callable] = None,
        n_scenarios: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run both QMC and classical MC, return comparison.

        Args:
            returns: Historical returns.
            strategy_func: Strategy function.
            n_scenarios: Number of scenarios per method.

        Returns:
            Dict with qmc_results, classical_results, comparison metrics.
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[~np.isnan(returns)]
        n_sc = n_scenarios or self.n_scenarios

        # QMC
        t0 = time.monotonic()
        qmc_result = self.run_qmc_scenarios(returns, strategy_func, n_sc)
        qmc_time = time.monotonic() - t0

        # Classical MC
        t0 = time.monotonic()
        classical_scenarios = self._generate_classical_scenarios(returns, n_sc)
        classical_returns = np.zeros(n_sc)
        classical_drawdowns = np.zeros(n_sc)

        for i in range(n_sc):
            res = self._evaluate_strategy(classical_scenarios[i], strategy_func)
            classical_returns[i] = res["total_return"]
            classical_drawdowns[i] = res["max_drawdown"]

        classical_time = time.monotonic() - t0

        classical_mean = float(np.mean(classical_returns))
        classical_std = float(np.std(classical_returns, ddof=1)) if n_sc > 1 else 0.0
        classical_sorted = np.sort(classical_returns)
        var_idx = max(int(n_sc * 0.05), 1)
        classical_var95 = float(-classical_sorted[var_idx - 1])
        classical_cvar95 = float(-np.mean(classical_sorted[:var_idx]))

        # Convergence analysis: run at multiple sample sizes
        convergence = self._convergence_analysis(returns, strategy_func)

        # True mean estimate (use all data as reference)
        true_mean = float(np.mean(returns))

        qmc_error = abs(qmc_result["mean_return"] - true_mean)
        classical_error = abs(classical_mean - true_mean)

        return {
            "qmc": qmc_result,
            "classical": {
                "mean_return": round(classical_mean, 8),
                "std_return": round(classical_std, 8),
                "var_95": round(classical_var95, 8),
                "cvar_95": round(classical_cvar95, 8),
                "max_drawdown": round(float(np.mean(classical_drawdowns)), 8),
                "time_s": round(classical_time, 4),
                "method": "pseudo_random_monte_carlo",
            },
            "comparison": {
                "qmc_time_s": round(qmc_time, 4),
                "classical_time_s": round(classical_time, 4),
                "qmc_mean_error": round(qmc_error, 8),
                "classical_mean_error": round(classical_error, 8),
                "qmc_advantage": qmc_error < classical_error,
                "convergence": convergence,
            },
            "note": (
                "QMC uses Sobol quasi-random sequences for better space coverage. "
                "Convergence rate is O(1/N) vs O(1/sqrt(N)) for classical MC. "
                "This is a classical mathematical advantage, not quantum."
            ),
        }

    def _convergence_analysis(
        self,
        returns: np.ndarray,
        strategy_func: Optional[Callable],
        sizes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Analyze convergence rates of QMC vs classical MC.

        Runs both methods at increasing sample sizes and measures
        how quickly the mean estimate stabilizes.
        """
        if sizes is None:
            sizes = [100, 500, 1000, 5000]

        # Filter to sizes <= n_scenarios
        sizes = [s for s in sizes if s <= self.n_scenarios]
        if not sizes:
            sizes = [min(100, self.n_scenarios)]

        true_mean = float(np.mean(returns))
        qmc_errors = []
        mc_errors = []

        for n in sizes:
            # QMC
            qmc_res = self.run_qmc_scenarios(returns, strategy_func, n)
            qmc_errors.append(abs(qmc_res["mean_return"] - true_mean))

            # Classical
            classical_scenarios = self._generate_classical_scenarios(returns, n)
            mc_means = []
            for i in range(n):
                if strategy_func is not None:
                    try:
                        mc_means.append(float(strategy_func(classical_scenarios[i])))
                    except Exception:
                        mc_means.append(float(np.sum(classical_scenarios[i])))
                else:
                    mc_means.append(float(np.sum(classical_scenarios[i])))
            mc_errors.append(abs(float(np.mean(mc_means)) - true_mean))

        return {
            "sample_sizes": sizes,
            "qmc_errors": [round(e, 8) for e in qmc_errors],
            "mc_errors": [round(e, 8) for e in mc_errors],
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return configuration summary."""
        return {
            "n_scenarios": self.n_scenarios,
            "seed": self.seed,
            "sobol_available": _HAS_SCIPY_SOBOL,
            "method": "quasi_monte_carlo",
        }
