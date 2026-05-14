"""Level-3 order book with per-order queue priority."""
# pyright: reportMissingImports=false

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

Side = Literal["buy", "sell"]


@dataclass(slots=True)
class L3Order:
    order_id: str
    side: Side
    price: float
    quantity: float
    timestamp_ns: int
    owner: str = "market"
    filled_quantity: float = 0.0

    @property
    def remaining_quantity(self) -> float:
        return float(max(self.quantity - self.filled_quantity, 0.0))


@dataclass(slots=True)
class BookExecution:
    resting_order_id: str
    resting_owner: str
    price: float
    quantity: float
    timestamp_ns: int


@dataclass
class PriceLevel:
    price: float
    side: Side
    orders: Deque[L3Order] = field(default_factory=deque)

    @property
    def total_quantity(self) -> float:
        if not self.orders:
            return 0.0
        return float(np.sum([order.remaining_quantity for order in self.orders], dtype=float))

    def queue_ahead(self, order_id: Optional[str] = None) -> float:
        ahead = 0.0
        for order in self.orders:
            if order_id is not None and order.order_id == order_id:
                break
            ahead += order.remaining_quantity
        return float(ahead)


@dataclass
class L3OrderBook:
    symbol: str
    bids: Dict[float, PriceLevel] = field(default_factory=dict)
    asks: Dict[float, PriceLevel] = field(default_factory=dict)
    order_index: Dict[str, L3Order] = field(default_factory=dict)
    last_trade_price: float = 0.0

    def add_order(self, order: L3Order) -> None:
        if order.order_id in self.order_index:
            raise ValueError(f"duplicate order_id={order.order_id}")
        side_book = self.bids if order.side == "buy" else self.asks
        level = side_book.get(order.price)
        if level is None:
            level = PriceLevel(price=order.price, side=order.side)
            side_book[order.price] = level
        level.orders.append(order)
        self.order_index[order.order_id] = order

    def cancel_order(self, order_id: str, quantity: Optional[float] = None) -> float:
        order = self.order_index.get(order_id)
        if order is None:
            return 0.0
        reduction = order.remaining_quantity if quantity is None else min(quantity, order.remaining_quantity)
        order.quantity = max(order.quantity - reduction, order.filled_quantity)
        self._cleanup_order(order)
        return float(reduction)

    def modify_order(self, order_id: str, new_quantity: float) -> float:
        order = self.order_index.get(order_id)
        if order is None:
            raise ValueError(f"unknown order_id={order_id}")
        order.quantity = max(new_quantity, order.filled_quantity)
        self._cleanup_order(order)
        return order.remaining_quantity

    def execute_order(self, order_id: str, quantity: float, timestamp_ns: int = 0) -> BookExecution | None:
        order = self.order_index.get(order_id)
        if order is None or quantity <= 0:
            return None
        executed = min(quantity, order.remaining_quantity)
        order.filled_quantity += executed
        self.last_trade_price = order.price
        execution = BookExecution(
            resting_order_id=order.order_id,
            resting_owner=order.owner,
            price=order.price,
            quantity=float(executed),
            timestamp_ns=timestamp_ns,
        )
        self._cleanup_order(order)
        return execution

    def apply_trade(self, side: Side, price: float, quantity: float, timestamp_ns: int) -> list[BookExecution]:
        if quantity <= 0:
            return []
        aggressive_side = "buy" if side == "buy" else "sell"
        resting_side = self.asks if aggressive_side == "buy" else self.bids
        price_levels = self.ask_prices() if aggressive_side == "buy" else self.bid_prices()
        matches: list[BookExecution] = []
        remaining = float(quantity)
        for level_price in price_levels:
            if remaining <= 0:
                break
            if aggressive_side == "buy" and level_price > price + 1e-12:
                break
            if aggressive_side == "sell" and level_price < price - 1e-12:
                break
            level = resting_side.get(level_price)
            if level is None:
                continue
            for resting_order in list(level.orders):
                if remaining <= 0:
                    break
                executed = self.execute_order(resting_order.order_id, remaining, timestamp_ns)
                if executed is None:
                    continue
                matches.append(executed)
                remaining -= executed.quantity
        if matches:
            self.last_trade_price = float(matches[-1].price)
        return matches

    def sweep_market_order(self, side: Side, quantity: float, timestamp_ns: int = 0) -> list[BookExecution]:
        if quantity <= 0:
            return []
        remaining = float(quantity)
        matches: list[BookExecution] = []
        side_book = self.asks if side == "buy" else self.bids
        price_levels = self.ask_prices() if side == "buy" else self.bid_prices()
        for level_price in price_levels:
            if remaining <= 0:
                break
            level = side_book.get(level_price)
            if level is None:
                continue
            for resting_order in list(level.orders):
                if remaining <= 0:
                    break
                executed = self.execute_order(resting_order.order_id, remaining, timestamp_ns)
                if executed is None:
                    continue
                matches.append(executed)
                remaining -= executed.quantity
        return matches

    def queue_ahead(self, side: Side, price: float, order_id: Optional[str] = None) -> float:
        level = self.get_level(side, price)
        return 0.0 if level is None else level.queue_ahead(order_id)

    def total_depth(self, side: Side, max_levels: int | None = None) -> float:
        prices = self.bid_prices() if side == "buy" else self.ask_prices()
        if max_levels is not None:
            prices = prices[:max_levels]
        book = self.bids if side == "buy" else self.asks
        return float(np.sum([book[price].total_quantity for price in prices], dtype=float)) if prices else 0.0

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def mid_price(self) -> float | None:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return float((best_bid + best_ask) / 2.0)

    def spread(self) -> float | None:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return float(best_ask - best_bid)

    def bid_prices(self) -> list[float]:
        return sorted(self.bids, reverse=True)

    def ask_prices(self) -> list[float]:
        return sorted(self.asks)

    def depth_snapshot(self, levels: int = 5) -> dict[str, list[dict[str, float]]]:
        return {
            "bids": [
                {"price": price, "quantity": self.bids[price].total_quantity}
                for price in self.bid_prices()[:levels]
            ],
            "asks": [
                {"price": price, "quantity": self.asks[price].total_quantity}
                for price in self.ask_prices()[:levels]
            ],
        }

    def get_level(self, side: Side, price: float) -> PriceLevel | None:
        return (self.bids if side == "buy" else self.asks).get(price)

    def _cleanup_order(self, order: L3Order) -> None:
        if order.remaining_quantity > 1e-12:
            return
        side_book = self.bids if order.side == "buy" else self.asks
        level = side_book.get(order.price)
        if level is not None:
            level.orders = deque(
                item for item in level.orders if item.order_id != order.order_id and item.remaining_quantity > 1e-12
            )
            if not level.orders:
                _ = side_book.pop(order.price, None)
        _ = self.order_index.pop(order.order_id, None)
