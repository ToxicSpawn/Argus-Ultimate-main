from __future__ import annotations

from typing import Optional

from argus_live.execution.intent_models import ExecutionIntent


def build_intent(*, symbol: str, side: str, quantity: float, strategy_id: str, manifest_hash: str, order_type: str = "limit", limit_price: Optional[float] = None) -> ExecutionIntent:
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if order_type == "limit" and limit_price is None:
        raise ValueError("limit orders require limit_price")
    return ExecutionIntent.new(symbol=symbol, side=side, order_type=order_type, quantity=quantity, strategy_id=strategy_id, manifest_hash=manifest_hash, limit_price=limit_price)
