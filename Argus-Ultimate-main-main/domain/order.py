"""
Typed Order dataclass — represents an order sent (or to be sent) to an exchange.

Uses Decimal for monetary values to avoid floating-point precision issues in
accounting and fee calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class Order:
    """An order to be placed or already placed on an exchange."""

    symbol: str
    """Trading pair, e.g. 'BTC/USD'."""

    side: Literal["buy", "sell"]
    """Direction of the trade."""

    quantity: Decimal
    """Order size in base currency (always positive)."""

    order_type: Literal["market", "limit", "stop_limit"]
    """Order type."""

    price: Decimal | None = None
    """Limit price — required for 'limit' and 'stop_limit' types."""

    stop_price: Decimal | None = None
    """Stop trigger price — required for 'stop_limit' orders."""

    client_order_id: str = ""
    """Optional idempotency key; set before submission, carry through fills."""

    strategy_id: str = ""
    """Source strategy that generated this order (for attribution)."""

    signal_timestamp: float = 0.0
    """Unix epoch seconds of the originating Signal (latency tracking)."""

    def validate(self) -> None:
        """Raise ValueError if the order is malformed."""
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be > 0, got {self.quantity}")
        if self.order_type in ("limit", "stop_limit") and self.price is None:
            raise ValueError(f"Limit/stop_limit orders require a price, got None")
        if self.order_type == "stop_limit" and self.stop_price is None:
            raise ValueError("stop_limit orders require stop_price")
