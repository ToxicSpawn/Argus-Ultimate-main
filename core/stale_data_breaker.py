from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaleDataCheck:
    symbol: str
    data_age_seconds: float
    threshold_seconds: float
    is_stale: bool
    reason: str


class StaleDataBreaker:
    """Circuit breaker that detects stale market data per symbol."""

    def __init__(self, threshold_seconds: float = 30.0) -> None:
        self._threshold = threshold_seconds
        self._last_update: Dict[str, float] = {}
        self._check_count = 0
        self._stale_count = 0

    def update(self, symbol: str, timestamp: float) -> None:
        """Record the latest data timestamp for a symbol."""
        self._last_update[symbol] = timestamp

    def check(self, symbol: str) -> StaleDataCheck:
        """Check whether data for *symbol* is stale."""
        self._check_count += 1
        now = time.time()

        if symbol not in self._last_update:
            self._stale_count += 1
            return StaleDataCheck(
                symbol=symbol,
                data_age_seconds=float("inf"),
                threshold_seconds=self._threshold,
                is_stale=True,
                reason=f"No data ever received for {symbol}",
            )

        age = now - self._last_update[symbol]
        is_stale = age > self._threshold
        if is_stale:
            self._stale_count += 1
            reason = (
                f"{symbol} data is {age:.1f}s old, "
                f"exceeds {self._threshold:.1f}s threshold"
            )
            logger.warning("Stale data detected: %s", reason)
        else:
            reason = f"{symbol} data is {age:.1f}s old, within threshold"

        return StaleDataCheck(
            symbol=symbol,
            data_age_seconds=age,
            threshold_seconds=self._threshold,
            is_stale=is_stale,
            reason=reason,
        )

    def check_all(self) -> List[StaleDataCheck]:
        """Check all tracked symbols and return a list of results."""
        return [self.check(sym) for sym in sorted(self._last_update)]

    def assert_fresh(self, symbol: str) -> None:
        """Raise RuntimeError if data for *symbol* is stale."""
        result = self.check(symbol)
        if result.is_stale:
            raise RuntimeError(result.reason)

    def get_stats(self) -> Dict:
        return {
            "threshold_seconds": self._threshold,
            "tracked_symbols": len(self._last_update),
            "total_checks": self._check_count,
            "stale_checks": self._stale_count,
        }
