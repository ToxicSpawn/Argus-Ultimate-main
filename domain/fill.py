"""
Typed Fill dataclass — represents the execution report returned by an exchange
after an order is (partially) filled.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class Fill:
    """Execution report for a filled or partially-filled order."""

    exchange_order_id: str
    """Exchange-assigned order ID."""

    client_order_id: str
    """Our idempotency key (mirrors Order.client_order_id)."""

    symbol: str
    """Trading pair, e.g. 'BTC/USD'."""

    side: Literal["buy", "sell"]
    """Direction of the fill."""

    filled_quantity: Decimal
    """Amount filled in base currency (may be < order quantity for partials)."""

    fill_price: Decimal
    """Average fill price."""

    fee: Decimal
    """Total fee charged (in quote currency)."""

    fee_currency: str
    """Currency in which the fee was charged (usually USD)."""

    timestamp: float
    """Unix epoch seconds of the fill."""

    is_partial: bool = False
    """True if this fill only covers part of the order quantity."""

    strategy_id: str = ""
    """Strategy attribution — propagated from the originating Order."""

    @property
    def notional_value(self) -> Decimal:
        """Gross notional: filled_quantity × fill_price."""
        return self.filled_quantity * self.fill_price

    @property
    def net_proceeds(self) -> Decimal:
        """Net proceeds after fee (positive for sells, negative for buys)."""
        if self.side == "sell":
            return self.notional_value - self.fee
        return -(self.notional_value + self.fee)
