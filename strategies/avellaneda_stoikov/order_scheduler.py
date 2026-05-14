"""Order lifecycle scheduling for market-making quotes."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .inventory风险管理 import InventoryRiskManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrderSchedulerConfig:
    refresh_interval_seconds: float = 1.0
    stale_after_seconds: float = 5.0
    replace_price_tolerance_bps: float = 2.0

    def __post_init__(self) -> None:
        if self.refresh_interval_seconds <= 0:
            raise ValueError("refresh_interval_seconds must be positive")
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        if self.replace_price_tolerance_bps < 0:
            raise ValueError("replace_price_tolerance_bps must be non-negative")


@dataclass(slots=True)
class ManagedOrder:
    side: str
    price: float
    quantity: float
    created_at: float = field(default_factory=time.time)
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "open"


@dataclass(slots=True)
class SchedulerAction:
    action: str
    order: ManagedOrder
    reason: str


class OrderScheduler:
    """Decides when quotes need refresh, replacement, or cancellation."""

    def __init__(self, config: OrderSchedulerConfig, inventory_manager: InventoryRiskManager):
        self.config = config
        self.inventory_manager = inventory_manager
        self._orders: Dict[str, ManagedOrder] = {}
        self._last_refresh: float = 0.0

    @property
    def open_orders(self) -> List[ManagedOrder]:
        return [order for order in self._orders.values() if order.status == "open"]

    def register_orders(self, orders: Iterable[ManagedOrder]) -> None:
        for order in orders:
            self._orders[order.order_id] = order
        self._last_refresh = time.time()

    def should_refresh(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return (now - self._last_refresh) >= self.config.refresh_interval_seconds

    def stale_orders(self, now: Optional[float] = None) -> List[ManagedOrder]:
        now = now if now is not None else time.time()
        return [
            order for order in self.open_orders
            if (now - order.created_at) >= self.config.stale_after_seconds
        ]

    def cancellation_actions(self, now: Optional[float] = None) -> List[SchedulerAction]:
        actions: List[SchedulerAction] = []
        for order in self.stale_orders(now):
            order.status = "cancelled"
            actions.append(SchedulerAction(action="cancel", order=order, reason="stale_order"))
        return actions

    def replacement_actions(
        self,
        target_bid: ManagedOrder,
        target_ask: ManagedOrder,
    ) -> List[SchedulerAction]:
        actions: List[SchedulerAction] = []
        targets = {"buy": target_bid, "sell": target_ask}
        tolerance = self.config.replace_price_tolerance_bps / 10_000.0

        for side, target in targets.items():
            current = next((order for order in self.open_orders if order.side == side), None)
            if current is None:
                actions.append(SchedulerAction(action="create", order=target, reason="missing_quote"))
                continue

            price_gap = abs(current.price - target.price) / max(target.price, 1e-12)
            size_gap = abs(current.quantity - target.quantity)
            if price_gap > tolerance or size_gap > 1e-12:
                current.status = "replaced"
                actions.append(SchedulerAction(action="replace", order=target, reason="quote_update"))

        return actions

    def process_fill(self, order_id: str, side: str, quantity: float, price: float) -> None:
        order = self._orders.get(order_id)
        if order is not None:
            order.status = "filled"
        self.inventory_manager.update_fill(side=side, quantity=quantity, price=price)

    def on_order_update(self, order_id: str, status: str) -> None:
        if order_id in self._orders:
            self._orders[order_id].status = status
