"""
Monte Carlo Simulation — path generation for risk and strategy analysis.

Capabilities:
  - Geometric Brownian Motion (GBM) paths
  - Bootstrap resampling from historical returns
  - Regime-switching simulation (bull/bear/volatile)
  - Confidence intervals for equity curve projections
  - VaR/CVaR estimation from simulated paths
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MCConfig:
    n_paths: int = 1000
    n_steps: int = 252          # trading days
    dt: float = 1 / 252        # step size in years
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
    random_seed: Optional[int] = None


@dataclass
class MCResult:
    paths: np.ndarray              # shape: (n_paths, n_steps+1) — price levels
    terminal_values: np.ndarray    # shape: (n_paths,) — final portfolio value
    percentiles: Dict[float, float]  # confidence → terminal value
    var_95: float                  # 5th percentile loss from initial
    cvar_95: float                 # mean of worst 5% paths
    max_drawdown_median: float     # median max drawdown across paths
    prob_ruin: float               # fraction of paths below 50% of start
    initial_value: float


def _max_drawdown(path: np.ndarray) -> float:
    """Compute max drawdown of a single equity path."""
    peak = path[0]
    max_dd = 0.0
    for v in path:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


class MonteCarlo:
    """
    Monte Carlo path simulator for portfolio equity curves.

    Usage::

        mc = MonteCarlo(MCConfig(n_paths=1000, n_steps=252))
        result = mc.simulate_gbm(initial_value=1000, mu=0.15, sigma=0.60)
        print(f"95% VaR: {result.var_95:.1f}")
        print(f"Median final: {result.percentiles[0.50]:.1f}")
    """

    def __init__(self, config: Optional[MCConfig] = None) -> None:
        self._cfg = config or MCConfig()
        if self._cfg.random_seed is not None:
            np.random.seed(self._cfg.random_seed)

    # ------------------------------------------------------------------
    def simulate_gbm(
        self,
        initial_value: float,
        mu: float,       # annualised drift (e.g. 0.15 = 15%)
        sigma: float,    # annualised volatility (e.g. 0.60 = 60%)
    ) -> MCResult:
        """Geometric Brownian Motion simulation."""
        cfg = self._cfg
        n, m, dt = cfg.n_paths, cfg.n_steps, cfg.dt

        # Drift-diffusion increments
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt)
        Z = np.random.standard_normal((n, m))
        log_returns = drift + diffusion * Z
        # Cumulative sum → log price paths
        log_paths = np.cumsum(log_returns, axis=1)
        # Prepend 0 (initial) and exponentiate
        log_paths = np.hstack([np.zeros((n, 1)), log_paths])
        paths = initial_value * np.exp(log_paths)

        return self._compute_result(paths, initial_value)

    def simulate_bootstrap(
        self,
        initial_value: float,
        historical_returns: Sequence[float],
        block_size: int = 5,
    ) -> MCResult:
        """
        Block bootstrap resampling from historical returns.
        Preserves autocorrelation structure better than IID resampling.
        """
        cfg = self._cfg
        n, m = cfg.n_paths, cfg.n_steps
        hist = np.array(historical_returns)
        T = len(hist)
        if T < block_size:
            raise ValueError(f"Need at least {block_size} historical returns")

        paths = np.zeros((n, m + 1))
        paths[:, 0] = initial_value

        for i in range(n):
            val = initial_value
            step = 0
            while step < m:
                # Random block start
                start = np.random.randint(0, T - block_size)
                block = hist[start: start + block_size]
                for r in block:
                    if step >= m:
                        break
                    val = val * math.exp(r)
                    paths[i, step + 1] = val
                    step += 1

        return self._compute_result(paths, initial_value)

    def simulate_regime_switching(
        self,
        initial_value: float,
        regimes: Dict[str, Tuple[float, float]],  # name → (mu, sigma)
        transition_matrix: Optional[np.ndarray] = None,
        initial_regime: int = 0,
    ) -> MCResult:
        """
        Markov regime-switching simulation.

        Args:
            regimes: dict of regime_name → (annual_mu, annual_sigma)
            transition_matrix: (n_regimes × n_regimes) row-stochastic
            initial_regime: starting regime index
        """
        cfg = self._cfg
        n, m, dt = cfg.n_paths, cfg.n_steps, cfg.dt

        regime_list = list(regimes.values())
        n_reg = len(regime_list)

        if transition_matrix is None:
            # Default: 95% stay in same regime, 5% spread to others
            T = np.full((n_reg, n_reg), 0.05 / max(1, n_reg - 1))
            np.fill_diagonal(T, 0.95)
        else:
            T = np.array(transition_matrix)

        cumT = np.cumsum(T, axis=1)
        paths = np.zeros((n, m + 1))
        paths[:, 0] = initial_value

        for i in range(n):
            regime = initial_regime
            val = initial_value
            for step in range(m):
                mu, sigma = regime_list[regime]
                drift = (mu - 0.5 * sigma ** 2) * dt
                diffusion = sigma * math.sqrt(dt) * np.random.standard_normal()
                val = val * math.exp(drift + diffusion)
                paths[i, step + 1] = val
                # Transition
                u = np.random.random()
                for next_reg in range(n_reg):
                    if u < cumT[regime, next_reg]:
                        regime = next_reg
                        break

        return self._compute_result(paths, initial_value)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_result(paths: np.ndarray, initial_value: float) -> MCResult:
        terminal = paths[:, -1]
        pct_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
        pcts = {p: float(np.percentile(terminal, p * 100)) for p in pct_levels}

        # VaR (5th pct loss)
        var_95 = float(initial_value - np.percentile(terminal, 5))

        # CVaR (mean of worst 5%)
        worst_5 = terminal[terminal <= np.percentile(terminal, 5)]
        cvar_95 = float(initial_value - np.mean(worst_5)) if len(worst_5) > 0 else var_95

        # Median max drawdown
        drawdowns = np.array([_max_drawdown(paths[i]) for i in range(len(paths))])
        med_dd = float(np.median(drawdowns))

        # Probability of ruin (< 50% of start)
        ruin_level = initial_value * 0.5
        prob_ruin = float(np.mean(terminal < ruin_level))

        return MCResult(
            paths=paths,
            terminal_values=terminal,
            percentiles=pcts,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown_median=med_dd,
            prob_ruin=prob_ruin,
            initial_value=initial_value,
        )

    def summary(self, result: MCResult) -> Dict[str, float]:
        """Human-readable summary dict."""
        return {
            "initial_value": result.initial_value,
            "median_terminal": result.percentiles.get(0.50, 0.0),
            "p5_terminal": result.percentiles.get(0.05, 0.0),
            "p95_terminal": result.percentiles.get(0.95, 0.0),
            "var_95_loss": result.var_95,
            "cvar_95_loss": result.cvar_95,
            "median_max_drawdown_pct": result.max_drawdown_median * 100,
            "prob_ruin_pct": result.prob_ruin * 100,
        }
