"""Monte Carlo simulation engine for portfolio risk analysis.

Provides GBM, historical bootstrap, and copula-based path simulation
together with VaR/CVaR computation, drawdown analysis, and scenario
generation. Designed to integrate with the Argus risk stack via
dependency injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import cholesky

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations."""

    n_simulations: int = 10000
    time_horizon_days: int = 252
    confidence_level: float = 0.95
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if self.time_horizon_days <= 0:
            raise ValueError("time_horizon_days must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1 (exclusive)")


@dataclass
class RiskMetrics:
    """Aggregated risk metrics from Monte Carlo simulation."""

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown_mean: float
    max_drawdown_std: float
    sharpe_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "var_95": float(self.var_95),
            "var_99": float(self.var_99),
            "cvar_95": float(self.cvar_95),
            "cvar_99": float(self.cvar_99),
            "max_drawdown_mean": float(self.max_drawdown_mean),
            "max_drawdown_std": float(self.max_drawdown_std),
            "sharpe_mean": float(np.mean(self.sharpe_distribution)) if self.sharpe_distribution.size else 0.0,
            "sharpe_std": float(np.std(self.sharpe_distribution)) if self.sharpe_distribution.size else 0.0,
        }


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------


class MonteCarloEngine:
    """Monte Carlo simulation engine for risk analysis.

    Supports Geometric Brownian Motion, historical bootstrap, and
    Gaussian/t-copula path generation for multi-asset portfolios.
    """

    _MIN_OBSERVATIONS = 10
    _EPSILON = 1e-12

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    # -- Simulation methods --------------------------------------------------

    def simulate_gbm(
        self,
        returns: np.ndarray,
        n_paths: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate paths via Geometric Brownian Motion.

        Parameters
        ----------
        returns : np.ndarray
            Historical returns (1-D for single asset, 2-D for multi-asset).
        n_paths : int, optional
            Number of simulated paths (defaults to config n_simulations).
        horizon : int, optional
            Number of time steps per path (defaults to config time_horizon_days).

        Returns
        -------
        np.ndarray
            Shape (n_paths, horizon) for single asset or
            (n_paths, horizon, n_assets) for multi-asset.
        """
        n_paths = n_paths or self.config.n_simulations
        horizon = horizon or self.config.time_horizon_days

        arr = self._normalize_returns(returns)
        if arr.ndim == 1:
            return self._gbm_single(arr, n_paths, horizon)
        return self._gbm_multi(arr, n_paths, horizon)

    def _gbm_single(self, returns: np.ndarray, n_paths: int, horizon: int) -> np.ndarray:
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma < self._EPSILON:
            logger.warning("GBM: near-zero volatility; returning flat paths")
            return np.zeros((n_paths, horizon), dtype=float)

        brownian = self._rng.standard_normal((n_paths, horizon))
        log_returns = (mu - 0.5 * sigma ** 2) + sigma * brownian
        price_paths = np.exp(np.cumsum(log_returns, axis=1))
        return price_paths

    def _gbm_multi(self, returns: np.ndarray, n_paths: int, horizon: int) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = np.mean(returns, axis=0)
        cov = np.cov(returns, rowvar=False)
        cov = self._regularize_matrix(cov)

        try:
            L = cholesky(cov, lower=True)
        except Exception:
            logger.warning("GBM multi: Cholesky failed; using diagonal covariance")
            L = np.diag(np.sqrt(np.diag(cov)))

        brownian = self._rng.standard_normal((n_paths, horizon, n_assets))
        correlated = brownian @ L.T

        log_returns = (mu - 0.5 * np.diag(cov)) + correlated
        price_paths = np.exp(np.cumsum(log_returns, axis=1))
        return price_paths

    def simulate_historical(
        self,
        returns: np.ndarray,
        n_paths: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate paths via historical bootstrap (sampling with replacement).

        Parameters
        ----------
        returns : np.ndarray
            Historical returns (1-D or 2-D).
        n_paths : int, optional
            Number of simulated paths.
        horizon : int, optional
            Number of time steps per path.

        Returns
        -------
        np.ndarray
            Shape (n_paths, horizon) or (n_paths, horizon, n_assets).
        """
        n_paths = n_paths or self.config.n_simulations
        horizon = horizon or self.config.time_horizon_days

        arr = self._normalize_returns(returns)
        if arr.size < self._MIN_OBSERVATIONS:
            logger.warning("Historical bootstrap: insufficient data (%d observations)", arr.shape[0])
            return np.zeros((n_paths, horizon) + (arr.shape[1:] if arr.ndim > 1 else ()), dtype=float)

        n_obs = arr.shape[0]
        indices = self._rng.integers(0, n_obs, size=(n_paths, horizon))

        if arr.ndim == 1:
            sampled = arr[indices]
        else:
            sampled = arr[indices]

        price_paths = np.exp(np.cumsum(sampled, axis=1))
        return price_paths

    def simulate_copula(
        self,
        returns: np.ndarray,
        n_paths: Optional[int] = None,
        horizon: Optional[int] = None,
        copula_type: str = "gaussian",
    ) -> np.ndarray:
        """Simulate correlated paths using a copula model.

        Parameters
        ----------
        returns : np.ndarray
            Historical multi-asset returns (2-D, shape (n_obs, n_assets)).
        n_paths : int, optional
            Number of simulated paths.
        horizon : int, optional
            Number of time steps per path.
        copula_type : str
            "gaussian" or "t" (Student-t copula).

        Returns
        -------
        np.ndarray
            Shape (n_paths, horizon, n_assets).
        """
        n_paths = n_paths or self.config.n_simulations
        horizon = horizon or self.config.time_horizon_days

        arr = self._normalize_returns(returns)
        if arr.ndim != 2:
            raise ValueError("Copula simulation requires 2-D returns (n_obs, n_assets)")

        n_assets = arr.shape[1]
        if n_assets < 2:
            logger.warning("Copula: single asset; falling back to GBM")
            return self.simulate_gbm(arr, n_paths, horizon)

        marginal_params = self._fit_marginals(arr)
        corr_matrix = np.corrcoef(arr, rowvar=False)
        corr_matrix = self._regularize_matrix(corr_matrix)

        if copula_type == "gaussian":
            uniform_samples = self._sample_gaussian_copula(corr_matrix, n_paths, horizon)
        elif copula_type == "t":
            uniform_samples = self._sample_t_copula(corr_matrix, n_paths, horizon)
        else:
            raise ValueError(f"Unsupported copula type: {copula_type}")

        simulated_returns = self._inverse_transform(uniform_samples, marginal_params)
        price_paths = np.exp(np.cumsum(simulated_returns, axis=1))
        return price_paths

    def _sample_gaussian_copula(
        self, corr_matrix: np.ndarray, n_paths: int, horizon: int
    ) -> np.ndarray:
        n_assets = corr_matrix.shape[0]
        try:
            L = cholesky(corr_matrix, lower=True)
        except Exception:
            L = np.eye(n_assets)

        Z = self._rng.standard_normal((n_paths, horizon, n_assets))
        correlated = Z @ L.T
        uniform = stats.norm.cdf(correlated)
        return uniform

    def _sample_t_copula(
        self, corr_matrix: np.ndarray, n_paths: int, horizon: int, df: float = 5.0
    ) -> np.ndarray:
        n_assets = corr_matrix.shape[0]
        try:
            L = cholesky(corr_matrix, lower=True)
        except Exception:
            L = np.eye(n_assets)

        Z = self._rng.standard_normal((n_paths, horizon, n_assets))
        chi2 = self._rng.chisquare(df, size=(n_paths, horizon, 1))
        t_samples = Z / np.sqrt(chi2 / df)
        correlated = t_samples @ L.T
        uniform = stats.t.cdf(correlated, df=df)
        return uniform

    def _fit_marginals(self, returns: np.ndarray) -> List[Dict[str, float]]:
        n_assets = returns.shape[1]
        params: List[Dict[str, float]] = []
        for i in range(n_assets):
            col = returns[:, i]
            params.append({
                "loc": float(np.mean(col)),
                "scale": max(float(np.std(col, ddof=1)), self._EPSILON),
            })
        return params

    def _inverse_transform(
        self, uniform: np.ndarray, marginal_params: List[Dict[str, float]]
    ) -> np.ndarray:
        n_assets = len(marginal_params)
        simulated = np.empty_like(uniform)
        for i in range(n_assets):
            loc = marginal_params[i]["loc"]
            scale = marginal_params[i]["scale"]
            simulated[:, :, i] = stats.norm.ppf(uniform[:, :, i]) * scale + loc
        return simulated

    # -- Risk metric computation ---------------------------------------------

    def compute_var(self, paths: np.ndarray, confidence: Optional[float] = None) -> float:
        """Compute Value at Risk from simulated paths.

        Parameters
        ----------
        paths : np.ndarray
            Simulated price paths, shape (n_paths, horizon) or
            (n_paths, horizon, n_assets).
        confidence : float, optional
            Confidence level (defaults to config confidence_level).

        Returns
        -------
        float
            VaR as a positive percentage loss.
        """
        confidence = confidence or self.config.confidence_level
        terminal_returns = self._terminal_returns(paths)
        if terminal_returns.size == 0:
            return 0.0

        percentile = (1.0 - confidence) * 100.0
        var_value = -float(np.percentile(terminal_returns, percentile))
        return max(0.0, var_value)

    def compute_cvar(self, paths: np.ndarray, confidence: Optional[float] = None) -> float:
        """Compute Conditional VaR (Expected Shortfall).

        Parameters
        ----------
        paths : np.ndarray
            Simulated price paths.
        confidence : float, optional
            Confidence level.

        Returns
        -------
        float
            CVaR as a positive percentage loss.
        """
        confidence = confidence or self.config.confidence_level
        terminal_returns = self._terminal_returns(paths)
        if terminal_returns.size == 0:
            return 0.0

        var_value = self.compute_var(paths, confidence)
        var_threshold = -var_value
        tail_losses = terminal_returns[terminal_returns <= var_threshold]
        if tail_losses.size == 0:
            return var_value
        return max(0.0, -float(np.mean(tail_losses)))

    def compute_max_drawdown(self, paths: np.ndarray) -> np.ndarray:
        """Compute max drawdown distribution across all paths.

        Parameters
        ----------
        paths : np.ndarray
            Simulated price paths.

        Returns
        -------
        np.ndarray
            Max drawdown per path, shape (n_paths,).
        """
        if paths.ndim == 3:
            paths = paths.mean(axis=2)

        n_paths = paths.shape[0]
        drawdowns = np.empty(n_paths, dtype=float)

        for i in range(n_paths):
            path = paths[i]
            if path.size == 0:
                drawdowns[i] = 0.0
                continue
            running_max = np.maximum.accumulate(path)
            dd = np.where(running_max > 0, (running_max - path) / running_max, 0.0)
            drawdowns[i] = float(np.max(dd))

        return drawdowns

    # -- Internal helpers ----------------------------------------------------

    def _terminal_returns(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 3:
            paths = paths.mean(axis=2)
        if paths.shape[1] == 0:
            return np.array([], dtype=float)
        initial = paths[:, 0:1]
        initial = np.where(initial > 0, initial, 1.0)
        return (paths[:, -1] / initial.flatten()) - 1.0

    def _normalize_returns(self, returns: np.ndarray) -> np.ndarray:
        arr = np.asarray(returns, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.isfinite(arr)]
        else:
            mask = np.all(np.isfinite(arr), axis=1)
            arr = arr[mask]
        return arr

    def _regularize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        jitter = np.eye(n, dtype=float) * 1e-8
        return matrix + jitter


# ---------------------------------------------------------------------------
# Scenario Generator
# ---------------------------------------------------------------------------


class ScenarioGenerator:
    """Generates stress scenarios and shock simulations for risk analysis."""

    _EPSILON = 1e-12

    def __init__(self, random_seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_seed)

    def generate_stress_scenarios(
        self,
        base_returns: np.ndarray,
        n_scenarios: int = 100,
    ) -> np.ndarray:
        """Generate stress scenarios by perturbing base return distribution.

        Parameters
        ----------
        base_returns : np.ndarray
            Historical returns (1-D or 2-D).
        n_scenarios : int
            Number of stress scenarios to generate.

        Returns
        -------
        np.ndarray
            Shape (n_scenarios, n_timesteps) or (n_scenarios, n_timesteps, n_assets).
        """
        arr = self._normalize(base_returns)
        if arr.size < 10:
            logger.warning("Stress scenarios: insufficient base data")
            return np.zeros((n_scenarios, 1) + (arr.shape[1:] if arr.ndim > 1 else ()), dtype=float)

        mu = np.mean(arr, axis=0)
        sigma = np.std(arr, axis=0, ddof=1)
        sigma = np.maximum(sigma, self._EPSILON)

        if arr.ndim == 1:
            scenarios = self._rng.normal(
                loc=mu - 2.0 * sigma,
                scale=sigma * 1.5,
                size=(n_scenarios, arr.shape[0]),
            )
        else:
            cov = np.cov(arr, rowvar=False)
            cov = self._regularize(cov)
            stressed_mu = mu - 2.0 * sigma
            scenarios = np.empty((n_scenarios, arr.shape[0], arr.shape[1]), dtype=float)
            for i in range(n_scenarios):
                scenarios[i] = self._rng.multivariate_normal(stressed_mu, cov, size=arr.shape[0])

        return scenarios

    def apply_shock(
        self,
        returns: np.ndarray,
        shock_params: Dict[str, Any],
    ) -> np.ndarray:
        """Apply a deterministic shock to returns.

        Parameters
        ----------
        returns : np.ndarray
            Returns to shock.
        shock_params : dict
            Supported keys:
            - "magnitude": float or np.ndarray — shock size (negative for downside)
            - "volatility_multiplier": float — scale volatility
            - "correlation_shock": float — increase cross-asset correlation

        Returns
        -------
        np.ndarray
            Shocked returns with same shape as input.
        """
        arr = self._normalize(returns).copy()

        magnitude = shock_params.get("magnitude", -0.1)
        vol_multiplier = shock_params.get("volatility_multiplier", 1.0)
        correlation_shock = shock_params.get("correlation_shock", 0.0)

        arr = arr * vol_multiplier
        arr = arr + magnitude

        if arr.ndim == 2 and correlation_shock > 0:
            n_assets = arr.shape[1]
            common_factor = self._rng.normal(0, correlation_shock, size=(arr.shape[0], 1))
            arr = arr + common_factor * np.ones((1, n_assets))

        return arr

    def generate_correlated_shocks(
        self,
        n_assets: int,
        correlation_matrix: np.ndarray,
        n_scenarios: int = 100,
    ) -> np.ndarray:
        """Generate correlated shock vectors across assets.

        Parameters
        ----------
        n_assets : int
            Number of assets.
        correlation_matrix : np.ndarray
            Asset correlation matrix (n_assets x n_assets).
        n_scenarios : int
            Number of shock scenarios.

        Returns
        -------
        np.ndarray
            Shape (n_scenarios, n_assets).
        """
        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"correlation_matrix must be ({n_assets}, {n_assets}), "
                f"got {correlation_matrix.shape}"
            )

        corr = self._regularize(correlation_matrix)
        try:
            L = cholesky(corr, lower=True)
        except Exception:
            L = np.eye(n_assets)

        Z = self._rng.standard_normal((n_scenarios, n_assets))
        correlated = Z @ L.T
        return correlated

    # -- Internal helpers ----------------------------------------------------

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            return a[np.isfinite(a)]
        mask = np.all(np.isfinite(a), axis=1)
        return a[mask]

    def _regularize(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        jitter = np.eye(n, dtype=float) * 1e-8
        return matrix + jitter


# ---------------------------------------------------------------------------
# Portfolio Simulator
# ---------------------------------------------------------------------------


class PortfolioSimulator:
    """Simulates portfolio-level outcomes from Monte Carlo paths."""

    _EPSILON = 1e-12
    _RISK_FREE_RATE = 0.02

    def __init__(self, random_seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_seed)

    def simulate_portfolio(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        n_paths: int = 10000,
    ) -> np.ndarray:
        """Simulate portfolio value paths given asset weights and returns.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights, shape (n_assets,). Should sum to 1.
        returns : np.ndarray
            Historical asset returns, shape (n_obs, n_assets).
        n_paths : int
            Number of simulated paths.

        Returns
        -------
        np.ndarray
            Portfolio price paths, shape (n_paths, horizon).
        """
        arr = np.asarray(returns, dtype=float)
        w = np.asarray(weights, dtype=float)

        if arr.ndim != 2:
            raise ValueError("returns must be 2-D (n_obs, n_assets)")
        if w.shape[0] != arr.shape[1]:
            raise ValueError(
                f"weights length ({w.shape[0]}) must match number of assets ({arr.shape[1]})"
            )

        n_assets = arr.shape[1]
        mu = np.mean(arr, axis=0)
        cov = np.cov(arr, rowvar=False)
        cov = self._regularize(cov)

        try:
            L = cholesky(cov, lower=True)
        except Exception:
            L = np.diag(np.sqrt(np.maximum(np.diag(cov), self._EPSILON)))

        horizon = arr.shape[0]
        brownian = self._rng.standard_normal((n_paths, horizon, n_assets))
        correlated = brownian @ L.T

        log_returns = (mu - 0.5 * np.diag(cov)) + correlated
        asset_paths = np.exp(np.cumsum(log_returns, axis=1))

        portfolio_paths = asset_paths @ w
        return portfolio_paths

    def compute_risk_metrics(self, paths: np.ndarray) -> RiskMetrics:
        """Compute comprehensive risk metrics from portfolio paths.

        Parameters
        ----------
        paths : np.ndarray
            Portfolio price paths, shape (n_paths, horizon).

        Returns
        -------
        RiskMetrics
        """
        engine = MonteCarloEngine()

        var_95 = engine.compute_var(paths, confidence=0.95)
        var_99 = engine.compute_var(paths, confidence=0.99)
        cvar_95 = engine.compute_cvar(paths, confidence=0.95)
        cvar_99 = engine.compute_cvar(paths, confidence=0.99)

        drawdowns = engine.compute_max_drawdown(paths)
        max_dd_mean = float(np.mean(drawdowns))
        max_dd_std = float(np.std(drawdowns))

        sharpe = self._sharpe_distribution(paths)

        metrics = RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown_mean=max_dd_mean,
            max_drawdown_std=max_dd_std,
            sharpe_distribution=sharpe,
        )

        logger.info(
            "Portfolio risk metrics: VaR95=%.4f VaR99=%.4f CVaR95=%.4f CVaR99=%.4f "
            "MaxDD_mean=%.4f MaxDD_std=%.4f",
            var_95, var_99, cvar_95, cvar_99, max_dd_mean, max_dd_std,
        )
        return metrics

    def compute_drawdown_distribution(self, paths: np.ndarray) -> np.ndarray:
        """Compute full drawdown distribution across all paths and timesteps.

        Parameters
        ----------
        paths : np.ndarray
            Portfolio price paths, shape (n_paths, horizon).

        Returns
        -------
        np.ndarray
            Drawdown values per path per timestep, shape (n_paths, horizon).
        """
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = np.where(running_max > 0, (running_max - paths) / running_max, 0.0)
        return drawdowns

    def compute_time_to_recovery(
        self,
        paths: np.ndarray,
        target_return: float = 0.0,
    ) -> np.ndarray:
        """Compute time (in steps) for each path to recover to target return.

        Parameters
        ----------
        paths : np.ndarray
            Portfolio price paths, shape (n_paths, horizon).
        target_return : float
            Target return level (0.0 = return to initial value).

        Returns
        -------
        np.ndarray
            Recovery time per path, shape (n_paths,). Paths that never
            recover are set to -1.
        """
        n_paths = paths.shape[0]
        initial_values = paths[:, 0]
        target_values = initial_values * (1.0 + target_return)

        recovery_times = np.full(n_paths, -1, dtype=float)

        for i in range(n_paths):
            path = paths[i]
            target = target_values[i]
            for t in range(1, path.size):
                if path[t] >= target:
                    recovery_times[i] = float(t)
                    break

        return recovery_times

    # -- Internal helpers ----------------------------------------------------

    def _sharpe_distribution(self, paths: np.ndarray) -> np.ndarray:
        n_paths = paths.shape[0]
        sharpe_ratios = np.empty(n_paths, dtype=float)

        for i in range(n_paths):
            path = paths[i]
            if path.size < 2:
                sharpe_ratios[i] = 0.0
                continue
            period_returns = np.diff(path) / path[:-1]
            mean_ret = float(np.mean(period_returns))
            std_ret = float(np.std(period_returns))
            if std_ret < self._EPSILON:
                sharpe_ratios[i] = 0.0
            else:
                sharpe_ratios[i] = (mean_ret - self._RISK_FREE_RATE / 252.0) / std_ret

        return sharpe_ratios

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            return a[np.isfinite(a)]
        mask = np.all(np.isfinite(a), axis=1)
        return a[mask]

    def _regularize(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        jitter = np.eye(n, dtype=float) * 1e-8
        return matrix + jitter


__all__ = [
    "MonteCarloEngine",
    "PortfolioSimulator",
    "RiskMetrics",
    "ScenarioGenerator",
    "SimulationConfig",
]
