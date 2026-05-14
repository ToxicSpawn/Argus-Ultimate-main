from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedOrder:
    order_id: str
    symbol: str
    status: str
    filled: float
    average_price: float | None
