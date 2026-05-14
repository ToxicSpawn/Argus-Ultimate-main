"""
Multi-period portfolio optimization via QAOA dynamic programming.

The single-period Markowitz problem (max E[r] - λ Var[r] s.t. Σ w = 1) is
extended to T time periods, where the portfolio weights are adjusted at each
period subject to transaction-cost penalties.

The state space at each period is the discrete set of possible portfolios;
QAOA solves each period's transition optimization, and dynamic programming
chains the periods.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from quantum.algorithms.qaoa import QAOAPortfolioOptimizer


def multi_period_portfolio_qaoa(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    n_periods: int = 4,
    transaction_cost_bps: float = 5.0,
    risk_aversion: float = 0.5,
    n_layers: int = 2,
) -> Dict[str, Any]:
    """
    Solve multi-period portfolio optimization via QAOA-DP.

    For each period, QAOA selects an asset subset that maximizes the
    risk-adjusted return minus transaction costs from the previous holding.
    DP chains the periods.

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected per-period returns of shape (n_assets,).
    covariance : np.ndarray
        Covariance matrix (n_assets, n_assets).
    n_periods : int
        Number of trading periods.
    transaction_cost_bps : float
        Round-trip transaction cost in bps.

    Returns
    -------
    Dict[str, Any]
        ``{"weights_per_period", "total_return", "method"}``
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    n_assets = len(mu)

    # Run QAOA for each period (DP-style: each period uses the previous
    # period's holding as the starting point for transaction-cost calculation)
    weights_per_period: List[np.ndarray] = []
    prev_weights = np.zeros(n_assets)
    period_returns = []

    for t in range(n_periods):
        opt = QAOAPortfolioOptimizer(n_layers=n_layers, max_assets=n_assets)
        # Build a modified return vector: mu - transaction_cost for assets we
        # don't currently hold (proxy for the cost of opening a new position)
        tc = transaction_cost_bps / 10000.0
        cost_penalty = np.where(prev_weights < 0.01, tc, 0.0)
        adjusted_mu = mu - cost_penalty

        result = opt.optimize(adjusted_mu, sigma, risk_aversion=risk_aversion)
        weights_t = np.array(result["weights"])
        weights_per_period.append(weights_t)
        period_returns.append(float(weights_t @ mu))
        prev_weights = weights_t

    return {
        "weights_per_period": [w.tolist() for w in weights_per_period],
        "period_returns": period_returns,
        "total_return": float(np.sum(period_returns)),
        "n_periods": n_periods,
        "transaction_cost_bps": transaction_cost_bps,
        "method": "multi_period_qaoa_dp",
    }
