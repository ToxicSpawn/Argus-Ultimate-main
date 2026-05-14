"""ExposureTracker — per-symbol and total notional exposure — Push 55."""
from __future__ import annotations

import threading
from typing import Dict, List


class ExposureTracker:
    """Tracks open notional exposure per symbol and in aggregate.

    All operations are thread-safe.

    Parameters
    ----------
    max_total_notional : float
        Hard cap on total open notional (default 0 = unlimited).
    """

    def __init__(self, max_total_notional: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._positions: Dict[str, float] = {}  # symbol -> notional
        self._max_total = max_total_notional

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, symbol: str, notional: float) -> None:
        """Register a new open position."""
        with self._lock:
            self._positions[symbol] = self._positions.get(symbol, 0.0) + notional

    def remove(self, symbol: str) -> float:
        """Remove a symbol's exposure. Returns removed notional."""
        with self._lock:
            return self._positions.pop(symbol, 0.0)

    def update(self, symbol: str, notional: float) -> None:
        """Overwrite notional for a symbol."""
        with self._lock:
            if notional <= 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = notional

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def total_notional(self) -> float:
        with self._lock:
            return sum(self._positions.values())

    def symbol_notional(self, symbol: str) -> float:
        with self._lock:
            return self._positions.get(symbol, 0.0)

    @property
    def open_count(self) -> int:
        with self._lock:
            return len(self._positions)

    @property
    def symbols(self) -> List[str]:
        with self._lock:
            return list(self._positions.keys())

    def utilisation(self, equity: float) -> float:
        """Total exposure as a fraction of equity (0.0–n)."""
        if equity <= 0:
            return 0.0
        return self.total_notional / equity

    def would_exceed(self, additional_notional: float) -> bool:
        """True if adding this notional would breach max_total_notional."""
        if self._max_total <= 0:
            return False
        with self._lock:
            return (sum(self._positions.values()) + additional_notional) > self._max_total

    def to_dict(self) -> dict:
        with self._lock:
            return dict(self._positions)
