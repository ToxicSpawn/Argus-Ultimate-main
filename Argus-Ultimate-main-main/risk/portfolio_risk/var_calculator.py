"""
Value at Risk Calculator
Historical VaR, Parametric VaR, and Conditional VaR (Expected Shortfall).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np


class VarCalculator:
    """Multi-method Value at Risk computation."""

    def __init__(
        self,
        confidence: float = 0.95,
        lookback: int = 252,
        horizon_days: int = 1,
    ):
        self.confidence = float(confidence)
        self.lookback = int(lookback)
        self.horizon_days = int(horizon_days)
        self._portfolio_returns: deque = deque(maxlen=lookback)

    def update_return(self, ret: float) -> None:
        self._portfolio_returns.append(float(ret))

    def _historical_var(self, returns: np.ndarray) -> float:
        if len(returns) < 10:
            return 0.0
        percentile = (1.0 - self.confidence) * 100
        var = -float(np.percentile(returns, percentile))
        return var * np.sqrt(self.horizon_days)

    def _parametric_var(self, returns: np.ndarray) -> float:
        if len(returns) < 10:
            return 0.0
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(self.confidence, 1.645)
        var = -(mu - z * sigma)
        return var * np.sqrt(self.horizon_days)

    def _conditional_var(self, returns: np.ndarray) -> float:
        if len(returns) < 10:
            return 0.0
        percentile = (1.0 - self.confidence) * 100
        threshold = np.percentile(returns, percentile)
        tail = returns[returns <= threshold]
        if len(tail) == 0:
            return self._historical_var(returns)
        cvar = -float(np.mean(tail))
        return cvar * np.sqrt(self.horizon_days)

    def _cornish_fisher_var(self, returns: np.ndarray) -> float:
        if len(returns) < 30:
            return self._parametric_var(returns)
        mu = np.mean(returns)
        sigma = np.std(returns)
        skew = float(np.mean(((returns - mu) / max(sigma, 1e-12)) ** 3))
        kurt = float(np.mean(((returns - mu) / max(sigma, 1e-12)) ** 4)) - 3.0
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(self.confidence, 1.645)
        z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
        var = -(float(mu) - z_cf * float(sigma))
        return var * np.sqrt(self.horizon_days)

    def calculate(self, returns: Any = None, portfolio_value: float = 0.0) -> Dict[str, Any]:
        if returns is not None:
            rets = np.array(returns, dtype=float)
        elif self._portfolio_returns:
            rets = np.array(list(self._portfolio_returns), dtype=float)
        else:
            return {
                "historical_var": 0.0, "parametric_var": 0.0,
                "conditional_var": 0.0, "cornish_fisher_var": 0.0,
                "var_pct": 0.0, "var_dollar": 0.0,
                "confidence": self.confidence, "metric": "var_calculator",
            }

        h_var = self._historical_var(rets)
        p_var = self._parametric_var(rets)
        c_var = self._conditional_var(rets)
        cf_var = self._cornish_fisher_var(rets)
        best_var = max(h_var, cf_var)

        return {
            "historical_var": h_var,
            "parametric_var": p_var,
            "conditional_var": c_var,
            "cornish_fisher_var": cf_var,
            "var_pct": best_var,
            "var_dollar": best_var * float(portfolio_value) if portfolio_value > 0 else 0.0,
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "n_observations": len(rets),
            "metric": "var_calculator",
        }
