"""
Implementation shortfall tracker: rolling average IS by strategy and by symbol.
Used to down-weight or gate strategies/symbols with consistently bad execution.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Module-level singleton so execution and audit can share state
_tracker: Optional["ISTracker"] = None


def get_is_tracker() -> "ISTracker":
    global _tracker
    if _tracker is None:
        _tracker = ISTracker()
    return _tracker


class ISTracker:
    """Rolling average implementation shortfall (bps) by strategy and by symbol."""

    def __init__(self, max_per_key: int = 100):
        self.max_per_key = max(10, int(max_per_key))
        self._by_strategy: Dict[str, deque] = {}
        self._by_symbol: Dict[str, deque] = {}

    def record(self, is_bps: float, strategy: Optional[str] = None, symbol: Optional[str] = None) -> None:
        if strategy:
            if strategy not in self._by_strategy:
                self._by_strategy[strategy] = deque(maxlen=self.max_per_key)
            self._by_strategy[strategy].append(float(is_bps))
        if symbol:
            if symbol not in self._by_symbol:
                self._by_symbol[symbol] = deque(maxlen=self.max_per_key)
            self._by_symbol[symbol].append(float(is_bps))

    def get_avg_is_bps(self, strategy: Optional[str] = None, symbol: Optional[str] = None) -> Optional[float]:
        """Return rolling average IS in bps. Positive = cost (worse execution)."""
        if strategy and strategy in self._by_strategy and self._by_strategy[strategy]:
            d = self._by_strategy[strategy]
            return float(sum(d) / len(d))
        if symbol and symbol in self._by_symbol and self._by_symbol[symbol]:
            d = self._by_symbol[symbol]
            return float(sum(d) / len(d))
        return None

    def should_gate(self, strategy: Optional[str], symbol: Optional[str], max_avg_is_bps: float) -> bool:
        """Return True if avg IS (positive = cost) exceeds max_avg_is_bps (gate threshold)."""
        if max_avg_is_bps <= 0:
            return False
        avg = self.get_avg_is_bps(strategy=strategy, symbol=symbol)
        if avg is None:
            return False
        return float(avg) > float(max_avg_is_bps)
