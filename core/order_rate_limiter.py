from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitCheck:
    venue: str
    orders_in_window: int
    max_orders: int
    allowed: bool
    reason: str


class OrderRateLimiter:
    """Per-venue sliding-window order rate limiter."""

    def __init__(
        self, max_orders_per_minute: int = 10, window_seconds: float = 60.0
    ) -> None:
        self._max_orders = max_orders_per_minute
        self._window = window_seconds
        self._timestamps: Dict[str, Deque[float]] = defaultdict(deque)
        self._total_recorded = 0
        self._total_rejected = 0

    def _prune(self, venue: str) -> None:
        cutoff = time.time() - self._window
        ts = self._timestamps[venue]
        while ts and ts[0] < cutoff:
            ts.popleft()

    def record_order(self, venue: str) -> None:
        """Record that an order was sent to *venue*."""
        self._timestamps[venue].append(time.time())
        self._total_recorded += 1

    def check(self, venue: str) -> RateLimitCheck:
        """Check whether *venue* is under the rate limit."""
        self._prune(venue)
        count = len(self._timestamps[venue])
        allowed = count < self._max_orders

        if allowed:
            reason = (
                f"{venue}: {count}/{self._max_orders} orders in "
                f"last {self._window:.0f}s — allowed"
            )
        else:
            self._total_rejected += 1
            reason = (
                f"{venue}: {count}/{self._max_orders} orders in "
                f"last {self._window:.0f}s — rate limit reached"
            )
            logger.warning("Rate limit hit: %s", reason)

        return RateLimitCheck(
            venue=venue,
            orders_in_window=count,
            max_orders=self._max_orders,
            allowed=allowed,
            reason=reason,
        )

    def assert_allowed(self, venue: str) -> None:
        """Raise RuntimeError if *venue* rate limit is exceeded."""
        result = self.check(venue)
        if not result.allowed:
            raise RuntimeError(result.reason)

    def get_stats(self) -> Dict:
        return {
            "max_orders_per_minute": self._max_orders,
            "window_seconds": self._window,
            "tracked_venues": len(self._timestamps),
            "total_recorded": self._total_recorded,
            "total_rejected": self._total_rejected,
        }
