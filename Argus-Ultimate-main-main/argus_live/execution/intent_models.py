from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import uuid

IntentSide = Literal["buy", "sell"]
IntentType = Literal["limit", "market"]


@dataclass(frozen=True)
class ExecutionIntent:
    intent_id: str
    symbol: str
    side: IntentSide
    order_type: IntentType
    quantity: float
    limit_price: float | None
    strategy_id: str
    manifest_hash: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def new(*, symbol: str, side: IntentSide, order_type: IntentType, quantity: float, strategy_id: str, manifest_hash: str, limit_price: float | None = None) -> "ExecutionIntent":
        return ExecutionIntent(str(uuid.uuid4()), symbol, side, order_type, quantity, limit_price, strategy_id, manifest_hash)
