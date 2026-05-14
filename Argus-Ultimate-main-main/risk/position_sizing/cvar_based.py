"""
CVaR-based position sizing.

Uses Conditional Value at Risk (Expected Shortfall) from recent returns
to scale position sizes inversely with tail risk.  When the tail is fat
(high CVaR) the sizer shrinks; when the tail is thin it grows — up to
a configurable cap.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CvarBasedSizer:
    """CVaR-aware position sizer.

    Parameters
    ----------
    confidence_level : VaR/CVaR confidence (0.95 = 95%)
    max_cvar_pct : if CVaR exceeds this pct of capital, scale to zero
    lookback : number of returns to keep in the rolling window
    """

    confidence_level: float = 0.95
    max_cvar_pct: float = 0.08  # 8% CVaR cap
    lookback: int = 252
    _returns: deque = field(default_factory=lambda: deque(maxlen=252), repr=False)

    def __post_init__(self) -> None:
        self._returns = deque(maxlen=max(30, int(self.lookback)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_return(self, ret: float) -> None:
        """Feed a single-period return (e.g. per-bar % change)."""
        self._returns.append(float(ret))

    def add_returns(self, rets: List[float]) -> None:
        """Feed a batch of historical returns."""
        for r in rets:
            self._returns.append(float(r))

    def current_cvar(self) -> float:
        """Compute CVaR (Expected Shortfall) from the rolling window."""
        if len(self._returns) < 10:
            return 0.0
        arr = np.array(list(self._returns), dtype=float)
        percentile = (1.0 - self.confidence_level) * 100.0
        threshold = np.percentile(arr, percentile)
        tail = arr[arr <= threshold]
        if len(tail) == 0:
            return 0.0
        return -float(np.mean(tail))

    def calculate(
        self,
        capital: float,
        risk_per_trade: float,
        confidence: float = 1.0,
        returns: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Compute CVaR-scaled position size.

        Parameters
        ----------
        capital : current account equity
        risk_per_trade : base fraction to risk (e.g. 0.02 = 2%)
        confidence : signal confidence [0, 1]
        returns : optional explicit returns list (overrides rolling window)
        """
        cap = max(float(capital), 1.0)
        rpt = max(0.0, float(risk_per_trade))
        conf = max(0.0, min(1.0, float(confidence)))

        # Seed the rolling window if external returns provided
        if returns is not None and len(returns) > 0:
            self.add_returns(returns)

        cvar = self.current_cvar()
        max_cvar = max(self.max_cvar_pct, 1e-9)

        # Scale factor: 1.0 when CVaR is zero, 0.0 when CVaR >= max_cvar_pct
        if cvar <= 0:
            cvar_scale = 1.0
        else:
            cvar_scale = max(0.0, 1.0 - (cvar / max_cvar))

        base_size = cap * rpt * conf
        adjusted_size = base_size * cvar_scale

        # Hard cap: never exceed 15% of capital
        adjusted_size = min(adjusted_size, cap * 0.15)

        # Sanity
        if math.isnan(adjusted_size) or math.isinf(adjusted_size):
            adjusted_size = cap * rpt * 0.5

        return {
            "position_size": float(adjusted_size),
            "pct_of_capital": (adjusted_size / cap) * 100.0,
            "cvar": float(cvar),
            "cvar_scale": float(cvar_scale),
            "method": "cvar_based",
        }


# Backwards-compat name used by `risk.position_sizing.__init__`
CVarBasedSizer = CvarBasedSizer

__all__ = ["CVarBasedSizer", "CvarBasedSizer"]
