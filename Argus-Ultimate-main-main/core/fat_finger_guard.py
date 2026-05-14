from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FatFingerCheck:
    order_value: float
    portfolio_value: float
    pct_of_portfolio: float
    max_pct: float
    allowed: bool
    reason: str


class FatFingerGuard:
    """Rejects orders that are too large relative to portfolio value."""

    def __init__(self, max_order_pct: float = 0.25) -> None:
        self._max_pct = max_order_pct
        self._check_count = 0
        self._reject_count = 0

    def check(
        self, quantity: float, price: float, portfolio_value: float
    ) -> FatFingerCheck:
        """Check whether an order is within the fat-finger limit."""
        self._check_count += 1
        order_value = abs(quantity * price)

        if portfolio_value <= 0:
            self._reject_count += 1
            return FatFingerCheck(
                order_value=order_value,
                portfolio_value=portfolio_value,
                pct_of_portfolio=float("inf"),
                max_pct=self._max_pct,
                allowed=False,
                reason="Portfolio value is zero or negative — order blocked",
            )

        pct = order_value / portfolio_value
        allowed = pct <= self._max_pct

        if allowed:
            reason = (
                f"Order {order_value:.2f} is {pct:.1%} of portfolio "
                f"{portfolio_value:.2f} — within {self._max_pct:.0%} limit"
            )
        else:
            self._reject_count += 1
            reason = (
                f"FAT FINGER — order {order_value:.2f} is {pct:.1%} of portfolio "
                f"{portfolio_value:.2f}, exceeds {self._max_pct:.0%} limit"
            )
            logger.warning("Fat finger rejected: %s", reason)

        return FatFingerCheck(
            order_value=order_value,
            portfolio_value=portfolio_value,
            pct_of_portfolio=pct,
            max_pct=self._max_pct,
            allowed=allowed,
            reason=reason,
        )

    def assert_allowed(
        self, quantity: float, price: float, portfolio_value: float
    ) -> None:
        """Raise RuntimeError if order exceeds the fat-finger limit."""
        result = self.check(quantity, price, portfolio_value)
        if not result.allowed:
            raise RuntimeError(result.reason)

    def get_stats(self) -> Dict:
        return {
            "max_order_pct": self._max_pct,
            "total_checks": self._check_count,
            "total_rejects": self._reject_count,
        }
