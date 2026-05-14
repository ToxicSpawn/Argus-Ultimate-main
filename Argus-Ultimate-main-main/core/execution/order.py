"""Push 77 — Order, Fill, Position dataclasses and enums.

Order lifecycle:
  NEW → SUBMITTED → PARTIALLY_FILLED → FILLED
                 ↘ CANCELLED
                 ↘ REJECTED

Fill tracks each partial or full execution with fee.
Position tracks net exposure per symbol.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    NEW              = "NEW"
    SUBMITTED        = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED           = "FILLED"
    CANCELLED        = "CANCELLED"
    REJECTED         = "REJECTED"


class PositionSide(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"


@dataclass
class Fill:
    order_id:   str
    symbol:     str
    side:       OrderSide
    qty:        float
    price:      float
    fee:        float        = 0.0
    fee_asset:  str          = "USDT"
    timestamp:  float        = field(default_factory=time.time)
    trade_id:   str          = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def net_proceeds(self) -> float:
        if self.side == OrderSide.BUY:
            return -(self.notional + self.fee)
        return self.notional - self.fee


@dataclass
class Order:
    symbol:      str
    side:        OrderSide
    order_type:  OrderType
    qty:         float
    price:       Optional[float]  = None    # None for MARKET
    stop_price:  Optional[float]  = None
    strategy_id: str              = "unknown"
    signal_strength: float        = 0.5
    order_id:    str              = field(default_factory=lambda: str(uuid.uuid4()))
    client_id:   str              = ""
    status:      OrderStatus      = OrderStatus.NEW
    filled_qty:  float            = 0.0
    avg_fill_price: float         = 0.0
    fills:       List[Fill]       = field(default_factory=list)
    created_at:  float            = field(default_factory=time.time)
    updated_at:  float            = field(default_factory=time.time)
    exchange_id: str              = ""      # exchange-assigned order ID

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.SUBMITTED,
                               OrderStatus.PARTIALLY_FILLED)

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                               OrderStatus.REJECTED)

    def apply_fill(self, fill: Fill) -> None:
        """Update order state from a fill event."""
        self.fills.append(fill)
        self.filled_qty    += fill.qty
        total_value         = sum(f.qty * f.price for f in self.fills)
        total_qty           = sum(f.qty for f in self.fills)
        self.avg_fill_price = total_value / total_qty if total_qty > 0 else 0.0
        self.updated_at     = time.time()
        if self.remaining_qty <= 1e-9:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def to_dict(self) -> dict:
        return {
            "order_id":    self.order_id,
            "symbol":      self.symbol,
            "side":        self.side.value,
            "type":        self.order_type.value,
            "qty":         self.qty,
            "price":       self.price,
            "status":      self.status.value,
            "filled_qty":  self.filled_qty,
            "avg_price":   self.avg_fill_price,
            "strategy_id": self.strategy_id,
        }


@dataclass
class Position:
    symbol:       str
    side:         PositionSide = PositionSide.FLAT
    qty:          float        = 0.0
    avg_entry:    float        = 0.0
    realised_pnl: float        = 0.0
    unrealised_pnl: float      = 0.0
    opened_at:    float        = field(default_factory=time.time)

    def update_unrealised(self, current_price: float) -> None:
        if self.side == PositionSide.LONG:
            self.unrealised_pnl = self.qty * (current_price - self.avg_entry)
        elif self.side == PositionSide.SHORT:
            self.unrealised_pnl = self.qty * (self.avg_entry - current_price)
        else:
            self.unrealised_pnl = 0.0

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.avg_entry

    @property
    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT or abs(self.qty) < 1e-9

    def to_dict(self) -> dict:
        return {
            "symbol":        self.symbol,
            "side":          self.side.value,
            "qty":           round(self.qty, 8),
            "avg_entry":     round(self.avg_entry, 4),
            "realised_pnl":  round(self.realised_pnl, 4),
            "unrealised_pnl": round(self.unrealised_pnl, 4),
            "notional":      round(self.notional, 2),
        }
