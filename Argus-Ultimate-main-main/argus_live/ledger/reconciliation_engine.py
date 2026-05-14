from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReconciliationResult:
    matched: bool
    reason: str
    venue_order_id: str | None
    venue_filled_qty: float
    venue_avg_price: float | None


def reconcile_fill(*, intent_id: str, venue_order_id: str, internal_qty: float, internal_price: float, venue_qty: float, venue_price: float) -> ReconciliationResult:
    qty_match = abs(internal_qty - venue_qty) < 1e-9
    price_match = abs(internal_price - venue_price) < 1e-9
    if qty_match and price_match:
        return ReconciliationResult(True, "reconciliation matched", venue_order_id, venue_qty, venue_price)
    return ReconciliationResult(False, "reconciliation mismatch", venue_order_id, venue_qty, venue_price)
