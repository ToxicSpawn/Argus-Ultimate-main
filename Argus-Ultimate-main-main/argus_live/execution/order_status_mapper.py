from __future__ import annotations

from argus_live.execution.normalized_order import NormalizedOrder


def normalize_order(raw: dict) -> NormalizedOrder:
    return NormalizedOrder(
        order_id=str(raw.get("id", "unknown")),
        symbol=str(raw.get("symbol", "unknown")),
        status=str(raw.get("status", "unknown")),
        filled=float(raw.get("filled", 0.0) or 0.0),
        average_price=(None if raw.get("average") is None else float(raw.get("average"))),
    )
