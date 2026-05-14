from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VenueOrderResult:
    success: bool
    venue_order_id: str | None
    reason: str
    raw: dict[str, Any] | None = None


class VenueAdapter:
    def submit_limit_order(self, *, symbol: str, side: str, quantity: float, price: float) -> VenueOrderResult:
        raise NotImplementedError

    def fetch_order(self, *, venue_order_id: str, symbol: str) -> dict[str, Any]:
        raise NotImplementedError
