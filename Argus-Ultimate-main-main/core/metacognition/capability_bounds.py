"""
Capability bounds: "Which regimes / symbols am I strong or weak in?"

Tracks rolling Sharpe ratio per regime, per symbol, per strategy so that
ARGUS can identify its own blind spots (e.g., "I lose money in CRISIS
regime and should not size up there").

The trading loop consumes this via:
- If current regime Sharpe < threshold → reduce size
- If current strategy Sharpe < threshold → deselect it
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# CapabilityBounds
# ═════════════════════════════════════════════════════════════════════════════


class CapabilityBounds:
    """
    Rolling performance tracker per (regime, symbol, strategy) dimension.

    Parameters
    ----------
    window : int
        Number of fills to keep per bucket.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = int(window)
        self._by_regime: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self._by_symbol: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self._by_strategy: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def record(
        self,
        *,
        regime: str,
        symbol: str,
        strategy: str,
        pnl: float,
    ) -> None:
        self._by_regime[str(regime)].append(float(pnl))
        self._by_symbol[str(symbol)].append(float(pnl))
        self._by_strategy[str(strategy)].append(float(pnl))

    @staticmethod
    def _sharpe(pnls: List[float]) -> float:
        """Rolling Sharpe (mean / std), annualized if needed."""
        if len(pnls) < 2:
            return 0.0
        arr = np.array(pnls, dtype=float)
        m = float(np.mean(arr))
        s = float(np.std(arr))
        if s < 1e-9:
            return 0.0
        return m / s

    def sharpe_by_regime(self, regime: str) -> float:
        return self._sharpe(list(self._by_regime.get(regime, [])))

    def sharpe_by_symbol(self, symbol: str) -> float:
        return self._sharpe(list(self._by_symbol.get(symbol, [])))

    def sharpe_by_strategy(self, strategy: str) -> float:
        return self._sharpe(list(self._by_strategy.get(strategy, [])))

    def strong_regimes(self, threshold: float = 0.5) -> List[str]:
        """Regimes where rolling Sharpe > threshold."""
        return sorted(
            [r for r, pnls in self._by_regime.items()
             if self._sharpe(list(pnls)) > threshold]
        )

    def weak_regimes(self, threshold: float = 0.0) -> List[str]:
        """Regimes where rolling Sharpe <= threshold (ARGUS blind spots)."""
        return sorted(
            [r for r, pnls in self._by_regime.items()
             if self._sharpe(list(pnls)) <= threshold
             and len(pnls) >= 5]
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "by_regime": {
                r: {
                    "sharpe": self._sharpe(list(pnls)),
                    "n": len(pnls),
                }
                for r, pnls in self._by_regime.items()
            },
            "by_symbol": {
                s: {
                    "sharpe": self._sharpe(list(pnls)),
                    "n": len(pnls),
                }
                for s, pnls in self._by_symbol.items()
            },
            "by_strategy": {
                st: {
                    "sharpe": self._sharpe(list(pnls)),
                    "n": len(pnls),
                }
                for st, pnls in self._by_strategy.items()
            },
            "strong_regimes": self.strong_regimes(),
            "weak_regimes": self.weak_regimes(),
        }
