"""Running max-drawdown calculator — Push 54."""
from __future__ import annotations


class RunningDrawdown:
    """Tracks running high-water mark and maximum drawdown.

    Call ``update(equity)`` after each equity change.
    Drawdown is expressed as a fraction (0.0 → 1.0).

    Example::

        dd = RunningDrawdown()
        dd.update(1000)
        dd.update(900)
        print(dd.current_dd)  # 0.1  (10% drawdown)
        print(dd.max_dd)      # 0.1
    """

    __slots__ = ("_hwm", "_current_dd", "_max_dd")

    def __init__(self) -> None:
        self._hwm: float = float("-inf")
        self._current_dd: float = 0.0
        self._max_dd: float = 0.0

    def update(self, equity: float) -> float:
        """Update with new equity value. Returns current drawdown fraction."""
        if equity > self._hwm:
            self._hwm = equity
        if self._hwm > 0:
            self._current_dd = max(0.0, (self._hwm - equity) / self._hwm)
        else:
            self._current_dd = 0.0
        if self._current_dd > self._max_dd:
            self._max_dd = self._current_dd
        return self._current_dd

    def reset(self) -> None:
        self._hwm = float("-inf")
        self._current_dd = 0.0
        self._max_dd = 0.0

    @property
    def hwm(self) -> float:
        """Current high-water mark."""
        return self._hwm

    @property
    def current_dd(self) -> float:
        """Current drawdown as a fraction (0.0–1.0)."""
        return self._current_dd

    @property
    def max_dd(self) -> float:
        """Maximum observed drawdown as a fraction (0.0–1.0)."""
        return self._max_dd
