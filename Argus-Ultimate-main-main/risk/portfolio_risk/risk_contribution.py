"""
Risk Contribution Analysis
Decomposes portfolio risk into per-asset contributions using marginal risk.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class RiskContribution:
    """Compute each asset's contribution to total portfolio risk."""

    def __init__(self, lookback: int = 60, min_observations: int = 20):
        self.lookback = int(lookback)
        self.min_observations = int(min_observations)

    def _covariance_matrix(self, returns_dict: Dict[str, List[float]]) -> tuple:
        symbols = list(returns_dict.keys())
        n = len(symbols)
        if n == 0:
            return np.array([[]]), symbols
        min_len = min(len(returns_dict[s]) for s in symbols)
        if min_len < self.min_observations:
            return np.eye(n) * 0.01, symbols
        data = np.array([list(returns_dict[s])[-min_len:] for s in symbols])
        cov = np.cov(data)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        return cov, symbols

    def calculate(
        self,
        returns_dict: Dict[str, List[float]] = None,
        weights: Dict[str, float] = None,
        portfolio_value: float = 0.0,
    ) -> Dict[str, Any]:
        if not returns_dict or len(returns_dict) == 0:
            return {
                "contributions": {},
                "total_risk": 0.0,
                "concentration": 0.0,
                "metric": "risk_contribution",
            }

        cov, symbols = self._covariance_matrix(returns_dict)
        n = len(symbols)

        if weights is None:
            w = np.ones(n) / max(n, 1)
        else:
            w = np.array([weights.get(s, 1.0 / n) for s in symbols])
            w_sum = np.sum(np.abs(w))
            if w_sum > 0:
                w = w / w_sum

        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0)))

        marginal_risk = cov @ w
        if port_vol > 1e-12:
            marginal_risk = marginal_risk / port_vol

        risk_contrib = w * marginal_risk
        total_contrib = float(np.sum(np.abs(risk_contrib)))

        contributions: Dict[str, Dict[str, float]] = {}
        for i, sym in enumerate(symbols):
            rc = float(risk_contrib[i])
            pct = (rc / total_contrib * 100) if total_contrib > 1e-12 else 0.0
            contributions[sym] = {
                "risk_contribution": rc,
                "pct_of_total": pct,
                "weight": float(w[i]),
                "marginal_risk": float(marginal_risk[i]),
            }

        hhi = float(np.sum((np.abs(risk_contrib) / max(total_contrib, 1e-12)) ** 2)) if total_contrib > 0 else 0
        concentration = hhi * n if n > 0 else 0

        annualized_vol = port_vol * np.sqrt(365 * 24)

        return {
            "contributions": contributions,
            "total_risk": port_vol,
            "annualized_vol": annualized_vol,
            "portfolio_variance": port_var,
            "concentration_hhi": hhi,
            "concentration_normalized": concentration,
            "n_assets": n,
            "risk_dollar": annualized_vol * float(portfolio_value) if portfolio_value > 0 else 0.0,
            "metric": "risk_contribution",
        }
