"""Push 66 — CVaR (Conditional Value at Risk) circuit breaker.

Replaces simple max-drawdown halt with Expected Shortfall:
  CVaR_alpha = E[loss | loss > VaR_alpha]

This measures the expected loss in the worst alpha% of scenarios,
which is more robust than peak-to-trough drawdown.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass
class CVaRSnapshot:
    cvar_95: float
    cvar_99: float
    var_95: float
    var_99: float
    n_samples: int
    halted: bool
    breach_level: str | None  # "CVaR_95" | "CVaR_99" | None


class CVaRHalt:
    """CVaR-based circuit breaker for Argus RiskManager.

    Args:
        max_cvar_95: Max acceptable CVaR at 95% confidence (e.g. 0.05 = 5%)
        max_cvar_99: Max acceptable CVaR at 99% confidence (e.g. 0.10 = 10%)
        window:      Rolling window of daily returns to compute CVaR
        min_samples: Minimum samples before breaker can fire
    """

    def __init__(
        self,
        max_cvar_95: float = 0.05,
        max_cvar_99: float = 0.10,
        window: int = 60,
        min_samples: int = 20,
    ):
        self.max_cvar_95 = max_cvar_95
        self.max_cvar_99 = max_cvar_99
        self.window = window
        self.min_samples = min_samples
        self._returns: Deque[float] = deque(maxlen=window)
        self._halted = False
        self._breach_level: str | None = None

    def record_return(self, daily_return: float) -> None:
        """Record a daily (or session) return value."""
        self._returns.append(daily_return)

    def evaluate(self) -> CVaRSnapshot:
        """Compute CVaR and check breaker. Returns snapshot."""
        n = len(self._returns)
        if n < self.min_samples:
            return CVaRSnapshot(0.0, 0.0, 0.0, 0.0, n, False, None)

        arr = np.array(self._returns)
        losses = -arr  # flip: positive loss = bad

        var_95 = float(np.percentile(losses, 95))
        var_99 = float(np.percentile(losses, 99))
        cvar_95 = float(losses[losses >= var_95].mean()) if any(losses >= var_95) else var_95
        cvar_99 = float(losses[losses >= var_99].mean()) if any(losses >= var_99) else var_99

        breach = None
        if cvar_99 > self.max_cvar_99:
            breach = "CVaR_99"
            self._halted = True
        elif cvar_95 > self.max_cvar_95:
            breach = "CVaR_95"
            self._halted = True

        self._breach_level = breach
        return CVaRSnapshot(
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            var_95=var_95,
            var_99=var_99,
            n_samples=n,
            halted=self._halted,
            breach_level=breach,
        )

    def reset(self) -> None:
        """Manually reset the halt (requires operator intervention)."""
        self._halted = False
        self._breach_level = None
        self._returns.clear()

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def breach_level(self) -> str | None:
        return self._breach_level
