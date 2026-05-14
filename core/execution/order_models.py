"""Order and Fill dataclasses — Push 56."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Fill:
    """Represents a single execution fill."""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    price: float = 0.0
    qty: float = 0.0
    fee: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    venue: str = ""

    @property
    def notional(self) -> float:
        return self.price * self.qty


@dataclass
class Order:
    """Represents a trading order with full lifecycle tracking."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    status: OrderStatus = OrderStatus.PENDING
    fills: List[Fill] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    venue: str = ""
    client_ref: str = ""
    fee_bps: float = 2.0

    # ------------------------------------------------------------------
    # Fill aggregation
    # ------------------------------------------------------------------

    @property
    def filled_qty(self) -> float:
        return sum(f.qty for f in self.fills)

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def avg_fill_price(self) -> float:
        total_qty = self.filled_qty
        if total_qty == 0:
            return 0.0
        return sum(f.price * f.qty for f in self.fills) / total_qty

    @property
    def total_fees(self) -> float:
        return sum(f.fee for f in self.fills)

    @property
    def is_complete(self) -> bool:
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}

    @property
    def is_active(self) -> bool:
        return self.status in {OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL}

    def add_fill(self, fill: Fill) -> None:
        self.fills.append(fill)
        self.updated_at = datetime.now(timezone.utc)
        if self.remaining_qty <= 1e-9:
            self.status = OrderStatus.FILLED
        elif self.filled_qty > 0:
            self.status = OrderStatus.PARTIAL

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "remaining_qty": self.remaining_qty,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status.value,
            "venue": self.venue,
            "total_fees": self.total_fees,
            "created_at": self.created_at.isoformat(),
        }
