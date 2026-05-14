"""
Monte Carlo Risk Simulation
Simulates portfolio paths to estimate tail risk and drawdown distribution.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class MonteCarlo:
    """Monte Carlo simulation for portfolio risk assessment."""

    def __init__(
        self,
        n_simulations: int = 5000,
        horizon_days: int = 30,
        confidence: float = 0.95,
        block_size: int = 5,
    ):
        self.n_simulations = int(n_simulations)
        self.horizon_days = int(horizon_days)
        self.confidence = float(confidence)
        self.block_size = int(block_size)

    def _bootstrap_paths(self, returns: np.ndarray) -> np.ndarray:
        n = len(returns)
        if n < self.block_size:
            indices = np.random.randint(0, n, size=(self.n_simulations, self.horizon_days))
            return returns[indices]
        n_blocks = max(self.horizon_days // self.block_size + 1, 1)
        max_start = n - self.block_size
        if max_start <= 0:
            indices = np.random.randint(0, n, size=(self.n_simulations, self.horizon_days))
            return returns[indices]
        paths = np.zeros((self.n_simulations, self.horizon_days))
        for sim in range(self.n_simulations):
            samples = []
            for _ in range(n_blocks):
                start = np.random.randint(0, max_start + 1)
                samples.extend(returns[start:start + self.block_size].tolist())
            paths[sim, :] = samples[:self.horizon_days]
        return paths

    def _compute_max_drawdown(self, cum_returns: np.ndarray) -> float:
        peak = np.maximum.accumulate(cum_returns)
        dd = (peak - cum_returns) / np.maximum(peak, 1e-12)
        return float(np.max(dd))

    def calculate(
        self,
        returns: Any = None,
        portfolio_value: float = 10000.0,
    ) -> Dict[str, Any]:
        if returns is None or len(returns) < 10:
            return {
                "mc_var": 0.0, "mc_cvar": 0.0, "median_return": 0.0,
                "worst_case": 0.0, "best_case": 0.0, "max_dd_median": 0.0,
                "max_dd_95": 0.0, "metric": "monte_carlo", "confidence": self.confidence,
            }

        rets = np.array(returns, dtype=float)
        paths = self._bootstrap_paths(rets)
        cum_paths = np.cumprod(1.0 + paths, axis=1)
        terminal_values = cum_paths[:, -1]
        terminal_returns = terminal_values - 1.0

        pct = (1.0 - self.confidence) * 100
        mc_var = -float(np.percentile(terminal_returns, pct))
        tail = terminal_returns[terminal_returns <= np.percentile(terminal_returns, pct)]
        mc_cvar = -float(np.mean(tail)) if len(tail) > 0 else mc_var

        max_dds = np.array([self._compute_max_drawdown(cum_paths[i]) for i in range(self.n_simulations)])

        pv = float(portfolio_value)

        return {
            "mc_var": mc_var,
            "mc_var_dollar": mc_var * pv,
            "mc_cvar": mc_cvar,
            "mc_cvar_dollar": mc_cvar * pv,
            "median_return": float(np.median(terminal_returns)),
            "mean_return": float(np.mean(terminal_returns)),
            "worst_case": float(np.min(terminal_returns)),
            "best_case": float(np.max(terminal_returns)),
            "worst_case_dollar": float(np.min(terminal_returns)) * pv,
            "max_dd_median": float(np.median(max_dds)),
            "max_dd_95": float(np.percentile(max_dds, 95)),
            "max_dd_worst": float(np.max(max_dds)),
            "n_simulations": self.n_simulations,
            "horizon_days": self.horizon_days,
            "confidence": self.confidence,
            "metric": "monte_carlo",
        }
