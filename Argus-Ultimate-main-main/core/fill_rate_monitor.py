"""Fill rate monitoring per venue/symbol pair."""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillRateSnapshot:
    venue: str
    symbol: str
    total_orders: int
    filled_orders: int
    fill_rate: float
    is_degraded: bool
    reason: str


class FillRateMonitor:
    """Tracks order fill rates per venue/symbol and flags degradation."""

    def __init__(self, min_fill_rate: float = 0.30, lookback: int = 100) -> None:
        self._min_fill_rate = min_fill_rate
        self._lookback = lookback
        self._history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=lookback)
        )

    def record_order(self, venue: str, symbol: str, was_filled: bool) -> None:
        """Record an order outcome (filled or not)."""
        self._history[(venue, symbol)].append(was_filled)

    def get_fill_rate(self, venue: str, symbol: str) -> FillRateSnapshot:
        """Compute fill rate snapshot for a venue/symbol pair."""
        key = (venue, symbol)
        buf = self._history.get(key)
        if not buf:
            return FillRateSnapshot(
                venue=venue,
                symbol=symbol,
                total_orders=0,
                filled_orders=0,
                fill_rate=1.0,
                is_degraded=False,
                reason="no data",
            )
        total = len(buf)
        filled = sum(buf)
        rate = filled / total if total > 0 else 1.0
        degraded = rate < self._min_fill_rate
        reason = (
            f"fill_rate={rate:.2f} < threshold={self._min_fill_rate:.2f}"
            if degraded
            else "ok"
        )
        if degraded:
            logger.warning(
                "Fill rate degraded for %s/%s: %.2f (%d/%d)",
                venue, symbol, rate, filled, total,
            )
        return FillRateSnapshot(
            venue=venue,
            symbol=symbol,
            total_orders=total,
            filled_orders=filled,
            fill_rate=rate,
            is_degraded=degraded,
            reason=reason,
        )

    def is_degraded(self, venue: str, symbol: str) -> bool:
        """Return True if fill rate is below threshold."""
        return self.get_fill_rate(venue, symbol).is_degraded

    def get_stats(self) -> Dict:
        """Return stats for all tracked venue/symbol pairs."""
        stats: Dict[str, object] = {}
        for (venue, symbol), buf in self._history.items():
            total = len(buf)
            filled = sum(buf)
            rate = filled / total if total > 0 else 1.0
            stats[f"{venue}/{symbol}"] = {
                "total_orders": total,
                "filled_orders": filled,
                "fill_rate": round(rate, 4),
                "is_degraded": rate < self._min_fill_rate,
            }
        return stats
