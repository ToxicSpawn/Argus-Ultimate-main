"""Push 66 — Correlation-aware portfolio risk monitor.

When held positions are highly correlated (e.g. BTC + ETH + SOL),
they behave as a single leveraged bet. This monitor:
  1. Tracks rolling return correlation between symbol pairs
  2. Fires a warning when avg correlation > warn_threshold
  3. Halts new positions when avg correlation > halt_threshold
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CorrelationSnapshot:
    pairs: Dict[str, float]       # {"BTC-ETH": 0.87, ...}
    avg_correlation: float
    max_correlation: float
    warning: bool
    halt_new_positions: bool


class CorrelationMonitor:
    """Rolling correlation monitor for multi-symbol portfolios."""

    def __init__(
        self,
        window: int = 60,
        warn_threshold: float = 0.75,
        halt_threshold: float = 0.85,
        min_samples: int = 20,
    ):
        self.window = window
        self.warn_threshold = warn_threshold
        self.halt_threshold = halt_threshold
        self.min_samples = min_samples
        self._returns: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def record_return(self, symbol: str, ret: float) -> None:
        self._returns[symbol].append(ret)

    def evaluate(self, active_symbols: List[str]) -> CorrelationSnapshot:
        """Compute pairwise correlations for currently held symbols."""
        syms = [s for s in active_symbols if len(self._returns[s]) >= self.min_samples]
        if len(syms) < 2:
            return CorrelationSnapshot({}, 0.0, 0.0, False, False)

        pairs: Dict[str, float] = {}
        all_corrs: List[float] = []

        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a = np.array(list(self._returns[syms[i]]))
                b = np.array(list(self._returns[syms[j]]))
                n = min(len(a), len(b))
                corr = float(np.corrcoef(a[-n:], b[-n:])[0, 1])
                if not np.isnan(corr):
                    key = f"{syms[i]}-{syms[j]}"
                    pairs[key] = corr
                    all_corrs.append(corr)

        if not all_corrs:
            return CorrelationSnapshot({}, 0.0, 0.0, False, False)

        avg_corr = float(np.mean(all_corrs))
        max_corr = float(np.max(all_corrs))
        return CorrelationSnapshot(
            pairs=pairs,
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            warning=avg_corr >= self.warn_threshold,
            halt_new_positions=avg_corr >= self.halt_threshold,
        )
