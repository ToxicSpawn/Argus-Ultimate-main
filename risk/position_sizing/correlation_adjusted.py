"""
Correlation-Adjusted Position Sizing
Reduces position size when new trade is correlated with existing portfolio exposure.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np


class CorrelationAdjustedSizer:
    """Scale position size down when correlated with existing positions."""

    def __init__(
        self,
        max_portfolio_correlation: float = 0.7,
        concentration_penalty: float = 0.5,
        lookback: int = 100,
        max_position_pct: float = 0.15,
    ):
        self.max_portfolio_correlation = float(max_portfolio_correlation)
        self.concentration_penalty = float(concentration_penalty)
        self.lookback = int(lookback)
        self.max_position_pct = float(max_position_pct)
        self._return_series: Dict[str, deque] = {}

    def update_returns(self, symbol: str, ret: float) -> None:
        if symbol not in self._return_series:
            self._return_series[symbol] = deque(maxlen=self.lookback)
        self._return_series[symbol].append(float(ret))

    def _pair_correlation(self, sym_a: str, sym_b: str) -> float:
        if sym_a not in self._return_series or sym_b not in self._return_series:
            return 0.0
        a = list(self._return_series[sym_a])
        b = list(self._return_series[sym_b])
        n = min(len(a), len(b))
        if n < 20:
            return 0.0
        a_arr = np.array(a[-n:])
        b_arr = np.array(b[-n:])
        if np.std(a_arr) < 1e-12 or np.std(b_arr) < 1e-12:
            return 0.0
        return float(np.corrcoef(a_arr, b_arr)[0, 1])

    def _portfolio_correlation(self, symbol: str, existing_positions: List[str]) -> float:
        if not existing_positions:
            return 0.0
        corrs = []
        for pos_sym in existing_positions:
            if pos_sym != symbol:
                corrs.append(abs(self._pair_correlation(symbol, pos_sym)))
        return float(np.mean(corrs)) if corrs else 0.0

    def _concentration_factor(self, existing_positions: List[str], max_positions: int = 10) -> float:
        n = len(existing_positions)
        if n == 0:
            return 1.0
        ratio = n / max(max_positions, 1)
        return max(1.0 - ratio * self.concentration_penalty, 0.2)

    def calculate(
        self,
        capital: float,
        risk_per_trade: float,
        confidence: float = 1.0,
        symbol: str = "",
        existing_positions: List[str] = None,
    ) -> Dict[str, Any]:
        cap = max(float(capital), 1.0)
        existing = existing_positions or []
        avg_corr = self._portfolio_correlation(symbol, existing)
        corr_scale = 1.0
        if avg_corr > self.max_portfolio_correlation:
            excess = avg_corr - self.max_portfolio_correlation
            corr_scale = max(1.0 - excess * 3.0, 0.1)
        conc_scale = self._concentration_factor(existing)
        base_size = cap * float(risk_per_trade) * float(confidence)
        adjusted_size = base_size * corr_scale * conc_scale
        max_size = cap * self.max_position_pct
        adjusted_size = min(adjusted_size, max_size)
        return {
            "position_size": adjusted_size,
            "pct_of_capital": (adjusted_size / cap) * 100,
            "correlation_scale": corr_scale,
            "concentration_scale": conc_scale,
            "avg_portfolio_correlation": avg_corr,
            "num_existing_positions": len(existing),
            "method": "correlation_adjusted",
        }
