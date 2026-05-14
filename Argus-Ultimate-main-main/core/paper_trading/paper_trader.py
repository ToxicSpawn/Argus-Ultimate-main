"""Push 70 — Simulated fill engine (PaperTrader).

Features:
  - Market + limit order simulation
  - Commission (bps) + slippage (bps) on every fill
  - Long + short positions with avg-entry tracking
  - Partial fills for limit orders (fill at limit if price crosses)
  - Cancel / modify open orders
  - Per-symbol position manager
  - Equity = cash + unrealised PnL across all positions
  - Trade history log
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    OPEN      = "OPEN"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    PARTIAL   = "PARTIAL"


@dataclass
class SimOrder:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float            # base asset quantity
    limit_price: Optional[float]
    status: OrderStatus = OrderStatus.OPEN
    filled_qty: float = 0.0
    fill_price: Optional[float] = None
    commission: float = 0.0
    created_at: float = field(default_factory=time.time)
    filled_at: Optional[float] = None


@dataclass
class SimPosition:
    symbol: str
    side: str               # "long" | "short"
    qty: float
    avg_entry: float
    realised_pnl: float = 0.0
    open_at: float = field(default_factory=time.time)

    def unrealised_pnl(self, mark_price: float) -> float:
        if self.side == "long":
            return (mark_price - self.avg_entry) * self.qty
        else:
            return (self.avg_entry - mark_price) * self.qty

    def notional(self, mark_price: float) -> float:
        return self.qty * mark_price


@dataclass
class FillEvent:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    fill_price: float
    commission: float
    pnl: float
    timestamp: float = field(default_factory=time.time)


class PaperTrader:
    """Simulated fill engine.

    Args:
        initial_cash:    Starting cash in USD
        commission_bps:  Per-fill commission in basis points
        slippage_bps:    Per-fill slippage in basis points
        on_fill:         Optional callback(FillEvent) on each fill
    """

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        on_fill: Optional[Callable[[FillEvent], None]] = None,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.on_fill = on_fill

        self._positions: Dict[str, SimPosition] = {}
        self._open_orders: Dict[str, SimOrder] = {}
        self._filled_orders: List[SimOrder] = []
        self._fill_events: List[FillEvent] = []
        self._last_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        current_price: float,
    ) -> SimOrder:
        """Place and immediately fill a market order."""
        order = SimOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=side,
            order_type=OrderType.MARKET,
            qty=qty, limit_price=None,
        )
        self._open_orders[order.order_id] = order
        self._fill_order(order, current_price)
        return order

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        limit_price: float,
    ) -> SimOrder:
        """Place a limit order (filled on next on_price_tick if crossed)."""
        order = SimOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=side,
            order_type=OrderType.LIMIT,
            qty=qty, limit_price=limit_price,
        )
        self._open_orders[order.order_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._open_orders.pop(order_id, None)
        if order and order.status == OrderStatus.OPEN:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def modify_order(
        self, order_id: str,
        new_qty: Optional[float] = None,
        new_limit: Optional[float] = None,
    ) -> bool:
        order = self._open_orders.get(order_id)
        if not order or order.status != OrderStatus.OPEN:
            return False
        if new_qty is not None:
            order.qty = new_qty
        if new_limit is not None:
            order.limit_price = new_limit
        return True

    # ------------------------------------------------------------------
    # Price tick processing
    # ------------------------------------------------------------------

    def on_price_tick(self, symbol: str, price: float) -> List[FillEvent]:
        """Process a new price tick. Fills pending limit orders if crossed."""
        self._last_prices[symbol] = price
        fills = []
        for oid, order in list(self._open_orders.items()):
            if order.symbol != symbol or order.status != OrderStatus.OPEN:
                continue
            if order.order_type == OrderType.LIMIT:
                triggered = (
                    (order.side == OrderSide.BUY  and price <= order.limit_price) or
                    (order.side == OrderSide.SELL and price >= order.limit_price)
                )
                if triggered:
                    fill = self._fill_order(order, order.limit_price)
                    if fill:
                        fills.append(fill)
        return fills

    # ------------------------------------------------------------------
    # Internal fill logic
    # ------------------------------------------------------------------

    def _fill_order(self, order: SimOrder, price: float) -> Optional[FillEvent]:
        slip_mult = 1 + (self.slippage_bps / 10_000) * (1 if order.side == OrderSide.BUY else -1)
        fill_price = price * slip_mult
        notional = fill_price * order.qty
        commission = notional * (self.commission_bps / 10_000)

        pnl = self._update_position(order.symbol, order.side, order.qty,
                                     fill_price, commission)

        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.fill_price = fill_price
        order.commission = commission
        order.filled_at = time.time()

        self._open_orders.pop(order.order_id, None)
        self._filled_orders.append(order)

        event = FillEvent(
            order_id=order.order_id, symbol=order.symbol,
            side=order.side, qty=order.qty,
            fill_price=fill_price, commission=commission, pnl=pnl,
        )
        self._fill_events.append(event)
        if self.on_fill:
            self.on_fill(event)
        return event

    def _update_position(
        self, symbol: str, side: OrderSide,
        qty: float, fill_price: float, commission: float,
    ) -> float:
        """Update position, deduct cash, return realised PnL if closing."""
        pnl = 0.0
        pos = self._positions.get(symbol)

        if side == OrderSide.BUY:
            notional = qty * fill_price + commission
            self.cash -= notional
            if pos is None or pos.side == "long":
                if pos is None:
                    self._positions[symbol] = SimPosition(
                        symbol=symbol, side="long",
                        qty=qty, avg_entry=fill_price,
                    )
                else:
                    total_qty = pos.qty + qty
                    pos.avg_entry = (pos.avg_entry * pos.qty + fill_price * qty) / total_qty
                    pos.qty = total_qty
            else:  # closing short
                pnl = (pos.avg_entry - fill_price) * min(qty, pos.qty) - commission
                pos.realised_pnl += pnl
                pos.qty -= qty
                if pos.qty <= 1e-9:
                    del self._positions[symbol]
                self.cash += pnl + commission  # return collateral
        else:  # SELL
            notional = qty * fill_price - commission
            self.cash += notional
            if pos is None or pos.side == "short":
                if pos is None:
                    self._positions[symbol] = SimPosition(
                        symbol=symbol, side="short",
                        qty=qty, avg_entry=fill_price,
                    )
                else:
                    total_qty = pos.qty + qty
                    pos.avg_entry = (pos.avg_entry * pos.qty + fill_price * qty) / total_qty
                    pos.qty = total_qty
            else:  # closing long
                pnl = (fill_price - pos.avg_entry) * min(qty, pos.qty) - commission
                pos.realised_pnl += pnl
                pos.qty -= qty
                if pos.qty <= 1e-9:
                    del self._positions[symbol]
        return pnl

    # ------------------------------------------------------------------
    # Equity + state
    # ------------------------------------------------------------------

    def equity(self, mark_prices: Optional[Dict[str, float]] = None) -> float:
        prices = mark_prices or self._last_prices
        unrealised = sum(
            pos.unrealised_pnl(prices.get(sym, pos.avg_entry))
            for sym, pos in self._positions.items()
        )
        return self.cash + unrealised

    @property
    def positions(self) -> Dict[str, SimPosition]:
        return self._positions

    @property
    def open_orders(self) -> Dict[str, SimOrder]:
        return self._open_orders

    @property
    def fill_history(self) -> List[FillEvent]:
        return self._fill_events

    @property
    def n_fills(self) -> int:
        return len(self._fill_events)

    def reset(self) -> None:
        self.cash = self.initial_cash
        self._positions.clear()
        self._open_orders.clear()
        self._filled_orders.clear()
        self._fill_events.clear()
        self._last_prices.clear()
