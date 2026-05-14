"""Spread monitoring with width detection and size scaling."""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

_EXTREME_MULTIPLIER = 5.0


@dataclass(frozen=True)
class SpreadSnapshot:
    symbol: str
    current_bps: float
    baseline_bps: float
    is_wide: bool
    multiplier: float
    reason: str


class SpreadMonitor:
    """Tracks per-symbol bid-ask spreads and detects widening."""

    def __init__(self, wide_multiplier: float = 2.0, baseline_window: int = 100) -> None:
        self._wide_multiplier = wide_multiplier
        self._baseline_window = baseline_window
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=baseline_window)
        )

    def record_spread(self, symbol: str, spread_bps: float) -> None:
        """Record a spread measurement in basis points."""
        self._history[symbol].append(spread_bps)

    def check(self, symbol: str) -> SpreadSnapshot:
        """Compute spread snapshot for a symbol."""
        buf = self._history.get(symbol)
        if not buf:
            return SpreadSnapshot(
                symbol=symbol,
                current_bps=0.0,
                baseline_bps=0.0,
                is_wide=False,
                multiplier=0.0,
                reason="no data",
            )
        values = list(buf)
        current = values[-1]
        baseline = sum(values) / len(values)
        mult = current / baseline if baseline > 0 else 0.0
        wide = current > baseline * self._wide_multiplier
        reason = (
            f"current={current:.1f}bps > baseline={baseline:.1f}bps x {self._wide_multiplier}"
            if wide
            else "ok"
        )
        if wide:
            logger.warning(
                "Spread wide for %s: current=%.1fbps, baseline=%.1fbps, multiplier=%.2f",
                symbol, current, baseline, mult,
            )
        return SpreadSnapshot(
            symbol=symbol,
            current_bps=round(current, 2),
            baseline_bps=round(baseline, 2),
            is_wide=wide,
            multiplier=round(mult, 2),
            reason=reason,
        )

    def is_wide(self, symbol: str) -> bool:
        """Return True if current spread exceeds baseline x multiplier."""
        return self.check(symbol).is_wide

    def get_size_multiplier(self, symbol: str) -> float:
        """Return position size multiplier: 1.0 normal, 0.5 wide, 0.0 extreme."""
        snap = self.check(symbol)
        if not snap.is_wide:
            return 1.0
        if snap.multiplier >= _EXTREME_MULTIPLIER:
            return 0.0
        return 0.5

    def get_stats(self) -> Dict:
        """Return stats for all tracked symbols."""
        stats: Dict[str, object] = {}
        for symbol in self._history:
            snap = self.check(symbol)
            stats[symbol] = {
                "current_bps": snap.current_bps,
                "baseline_bps": snap.baseline_bps,
                "is_wide": snap.is_wide,
                "multiplier": snap.multiplier,
                "size_multiplier": self.get_size_multiplier(symbol),
            }
        return stats
