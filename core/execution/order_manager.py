"""Push 77 — OrderManager: order lifecycle + position tracking.

Responsibilities:
  - Accept orders from ExecutionEngine
  - Enforce pre-trade risk checks (exposure limits, max open orders)
  - Track open orders and positions per symbol
  - Process fills → update positions + realised PnL
  - Maintain order history ring-buffer (500 orders)
  - Thread-safe via asyncio.Lock
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from core.execution.order import (
    Fill, Order, OrderSide, OrderStatus, Position, PositionSide,
)


class OrderManager:
    """Manages order lifecycle and positions.

    Args:
        max_open_orders:   Max concurrent active orders
        max_position_usd:  Max notional per symbol (USD)
        fee_rate:          Taker fee rate (default 0.001 = 0.1%)
        on_fill_callback:  Optional async callback(Fill)
    """

    def __init__(
        self,
        max_open_orders:  int   = 10,
        max_position_usd: float = 100_000.0,
        fee_rate:         float = 0.001,
        on_fill_callback: Optional[Callable] = None,
    ):
        self.max_open_orders  = max_open_orders
        self.max_position_usd = max_position_usd
        self.fee_rate         = fee_rate
        self._on_fill_cb      = on_fill_callback

        self._open_orders:    Dict[str, Order]    = {}
        self._positions:      Dict[str, Position] = {}
        self._order_history:  Deque[Order]        = deque(maxlen=500)
        self._lock            = asyncio.Lock()

        self._total_fills:   int   = 0
        self._total_volume:  float = 0.0
        self._total_fees:    float = 0.0

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def _check_limits(self, order: Order) -> Optional[str]:
        """Return rejection reason string, or None if OK."""
        if len(self._open_orders) >= self.max_open_orders:
            return f"Max open orders ({self.max_open_orders}) reached"
        pos = self._positions.get(order.symbol)
        if pos and not pos.is_flat:
            est_notional = pos.notional + (order.qty * (order.price or 0))
            if est_notional > self.max_position_usd:
                return f"Position notional {est_notional:.0f} > limit {self.max_position_usd:.0f}"
        return None

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Submit order after risk checks. Returns order with updated status."""
        async with self._lock:
            rejection = self._check_limits(order)
            if rejection:
                order.status = OrderStatus.REJECTED
                order.updated_at = time.time()
                self._order_history.append(order)
                return order

            order.status = OrderStatus.SUBMITTED
            order.updated_at = time.time()
            self._open_orders[order.order_id] = order
            return order

    async def cancel_order(self, order_id: str) -> Optional[Order]:
        """Cancel an open order. Returns order or None if not found."""
        async with self._lock:
            order = self._open_orders.get(order_id)
            if order and order.is_active:
                order.status     = OrderStatus.CANCELLED
                order.updated_at = time.time()
                self._open_orders.pop(order_id, None)
                self._order_history.append(order)
                return order
            return None

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    async def on_fill(self, fill: Fill) -> None:
        """Process an incoming fill event."""
        async with self._lock:
            order = self._open_orders.get(fill.order_id)
            if order is None:
                return

            order.apply_fill(fill)
            self._total_fills  += 1
            self._total_volume += fill.notional
            self._total_fees   += fill.fee

            # Update position
            self._update_position(order.symbol, fill, order.side)

            # Move to history if terminal
            if order.is_terminal:
                self._open_orders.pop(fill.order_id, None)
                self._order_history.append(order)

        if self._on_fill_cb:
            result = self._on_fill_cb(fill)
            if asyncio.iscoroutine(result):
                await result

    def _update_position(self, symbol: str, fill: Fill, side: OrderSide) -> None:
        """Update position from fill. Handles open, add, reduce, flip."""
        pos = self._positions.setdefault(
            symbol, Position(symbol=symbol)
        )
        qty   = fill.qty
        price = fill.price

        if side == OrderSide.BUY:
            if pos.side == PositionSide.SHORT:
                # Reduce / close short
                close_qty = min(qty, pos.qty)
                pnl = close_qty * (pos.avg_entry - price)
                pos.realised_pnl += pnl
                pos.qty -= close_qty
                qty     -= close_qty
                if pos.qty <= 1e-9:
                    pos.side = PositionSide.FLAT
                    pos.qty  = 0.0
            if qty > 1e-9:
                # Open / add long
                total_cost = pos.qty * pos.avg_entry + qty * price
                pos.qty   += qty
                pos.avg_entry = total_cost / pos.qty
                pos.side  = PositionSide.LONG
        else:  # SELL
            if pos.side == PositionSide.LONG:
                close_qty = min(qty, pos.qty)
                pnl = close_qty * (price - pos.avg_entry)
                pos.realised_pnl += pnl
                pos.qty -= close_qty
                qty     -= close_qty
                if pos.qty <= 1e-9:
                    pos.side = PositionSide.FLAT
                    pos.qty  = 0.0
            if qty > 1e-9:
                total_cost = pos.qty * pos.avg_entry + qty * price
                pos.qty   += qty
                pos.avg_entry = total_cost / pos.qty
                pos.side  = PositionSide.SHORT

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        orders = list(self._open_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._open_orders.get(order_id) or next(
            (o for o in self._order_history if o.order_id == order_id), None
        )

    @property
    def stats(self) -> dict:
        return {
            "open_orders":   len(self._open_orders),
            "total_fills":   self._total_fills,
            "total_volume":  round(self._total_volume, 2),
            "total_fees":    round(self._total_fees, 4),
            "positions":     {s: p.to_dict() for s, p in self._positions.items() if not p.is_flat},
        }
