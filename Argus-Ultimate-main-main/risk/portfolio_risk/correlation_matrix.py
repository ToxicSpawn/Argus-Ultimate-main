"""
Correlation Matrix Manager
Tracks rolling correlations between assets with decay weighting.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class CorrelationMatrix:
    """Rolling correlation matrix with exponential decay weighting."""

    def __init__(
        self,
        lookback: int = 60,
        ewm_halflife: int = 20,
        min_observations: int = 20,
    ):
        self.lookback = int(lookback)
        self.ewm_halflife = int(ewm_halflife)
        self.min_observations = int(min_observations)
        self._returns: Dict[str, deque] = {}

    def update_returns(self, symbol: str, ret: float) -> None:
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self.lookback)
        self._returns[symbol].append(float(ret))

    def _ewm_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        n = len(a)
        if n < self.min_observations:
            return 0.0
        decay = 0.5 ** (1.0 / max(self.ewm_halflife, 1))
        weights = np.array([decay ** i for i in range(n - 1, -1, -1)])
        weights /= weights.sum()
        wa = np.sum(weights * a)
        wb = np.sum(weights * b)
        cov = np.sum(weights * (a - wa) * (b - wb))
        var_a = np.sum(weights * (a - wa) ** 2)
        var_b = np.sum(weights * (b - wb) ** 2)
        denom = np.sqrt(max(var_a, 1e-15) * max(var_b, 1e-15))
        return float(cov / denom)

    def get_correlation(self, sym_a: str, sym_b: str) -> float:
        if sym_a == sym_b:
            return 1.0
        if sym_a not in self._returns or sym_b not in self._returns:
            return 0.0
        a = list(self._returns[sym_a])
        b = list(self._returns[sym_b])
        n = min(len(a), len(b))
        if n < self.min_observations:
            return 0.0
        return self._ewm_correlation(np.array(a[-n:]), np.array(b[-n:]))

    def calculate(self, symbols: List[str] = None, returns: Any = None) -> Dict[str, Any]:
        if symbols is None:
            symbols = list(self._returns.keys())
        n = len(symbols)
        if n == 0:
            return {"matrix": {}, "avg_correlation": 0.0, "max_correlation": 0.0, "metric": "correlation_matrix"}

        matrix: Dict[str, Dict[str, float]] = {}
        all_corrs: List[float] = []

        for i, sym_a in enumerate(symbols):
            matrix[sym_a] = {}
            for j, sym_b in enumerate(symbols):
                if i == j:
                    matrix[sym_a][sym_b] = 1.0
                elif j < i:
                    matrix[sym_a][sym_b] = matrix[sym_b][sym_a]
                else:
                    corr = self.get_correlation(sym_a, sym_b)
                    matrix[sym_a][sym_b] = corr
                    all_corrs.append(abs(corr))

        avg_corr = float(np.mean(all_corrs)) if all_corrs else 0.0
        max_corr = float(np.max(all_corrs)) if all_corrs else 0.0

        diversification = 1.0 - avg_corr if avg_corr < 1.0 else 0.0

        return {
            "matrix": matrix,
            "symbols": symbols,
            "avg_correlation": avg_corr,
            "max_correlation": max_corr,
            "diversification_score": diversification,
            "n_assets": n,
            "metric": "correlation_matrix",
        }
