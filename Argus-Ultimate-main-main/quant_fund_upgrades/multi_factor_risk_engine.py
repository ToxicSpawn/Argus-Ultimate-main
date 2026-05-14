"""
Multi-factor risk engine – portfolio risk and correlation (advisory).

Used by the unified system to compute portfolio risk and optional correlation matrix.
Stub implementation: maintains price history and returns simple risk metrics;
replace with full factor model for production quant use.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MultiFactorRiskEngine:
    """
    Multi-factor portfolio risk: price history, optional correlation, VaR-style metrics.
    - update_price(symbol, price): append price to history.
    - calculate_portfolio_risk(current_positions, current_prices, portfolio_value): async, returns risk object.
    """

    def __init__(self) -> None:
        self._prices: Dict[str, List[float]] = {}
        self._max_history = 500

    def update_price(self, symbol: str, price: float) -> None:
        """Append price to symbol history (for volatility/correlation)."""
        if symbol not in self._prices:
            self._prices[symbol] = []
        hist = self._prices[symbol]
        hist.append(float(price))
        if len(hist) > self._max_history:
            hist.pop(0)

    async def calculate_portfolio_risk(
        self,
        *,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
    ) -> Any:
        """
        Compute portfolio risk metrics (advisory). Returns object with at least
        correlation_matrix (optional), var, cvar, volatility, etc.
        """
        # Stub: return a simple namespace so existing code that reads risk.correlation_matrix doesn't break
        try:
            import numpy as np
            symbols = [s for s in current_positions if current_positions.get(s, 0) != 0 or s in current_prices]
            n = len(symbols)
            if n == 0:
                return _RiskResult(correlation_matrix=np.eye(1), var=0.0, cvar=0.0, volatility=0.0)
            # Build simple correlation from returns if we have history
            returns_list = []
            for s in symbols:
                hist = self._prices.get(s, [])
                if len(hist) < 2:
                    returns_list.append([0.0])
                    continue
                arr = np.array(hist[-100:], dtype=float)
                ret = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
                returns_list.append(ret.tolist())
            max_len = max(len(r) for r in returns_list)
            if max_len < 2:
                corr = np.eye(n)
            else:
                # Align to shortest series
                min_len = min(len(r) for r in returns_list)
                mat = np.array([r[-min_len:] for r in returns_list], dtype=float)
                if mat.shape[0] >= 2 and mat.shape[1] >= 2:
                    corr = np.corrcoef(mat)
                    if corr is None or not np.all(np.isfinite(corr)):
                        corr = np.eye(n)
                else:
                    corr = np.eye(n)
            # Simple VaR/CVaR from position PnL distribution (stub)
            var = 0.0
            cvar = 0.0
            vol = 0.0
            if portfolio_value > 0 and returns_list:
                all_ret = np.concatenate([np.array(r) for r in returns_list if len(r) > 0])
                if len(all_ret) > 10:
                    var = float(np.percentile(all_ret, 5))
                    cvar = float(np.mean(all_ret[all_ret <= var]))
                    vol = float(np.std(all_ret))
            return _RiskResult(correlation_matrix=corr, var=var, cvar=cvar, volatility=vol)
        except Exception as e:
            logger.debug("MultiFactorRiskEngine calculate_portfolio_risk: %s", e)
            import numpy as np
            return _RiskResult(correlation_matrix=np.eye(1), var=0.0, cvar=0.0, volatility=0.0)


class _RiskResult:
    """Simple result holder for risk metrics."""
    def __init__(
        self,
        *,
        correlation_matrix: Any,
        var: float = 0.0,
        cvar: float = 0.0,
        volatility: float = 0.0,
    ) -> None:
        self.correlation_matrix = correlation_matrix
        self.var = var
        self.cvar = cvar
        self.volatility = volatility
