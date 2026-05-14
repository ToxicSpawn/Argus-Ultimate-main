"""Thread-safe in-memory OrderBook — Push 56."""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from core.execution.order_models import Order, OrderStatus

try:
    from prometheus_client import Counter
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

logger = logging.getLogger(__name__)

if _PROM_AVAILABLE:
    _CTR_ORDERS = Counter(
        "argus_orders_total",
        "Total orders submitted",
        ["side", "type", "status"],
    )
else:
    _CTR_ORDERS = None


class OrderBook:
    """Thread-safe in-memory order registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._orders: Dict[str, Order] = {}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> None:
        with self._lock:
            order.status = OrderStatus.OPEN
            self._orders[order.order_id] = order
        if _CTR_ORDERS:
            _CTR_ORDERS.labels(
                side=order.side.value,
                type=order.order_type.value,
                status=OrderStatus.OPEN.value,
            ).inc()
        logger.debug(
            "OrderBook: submitted %s %s %s qty=%.6f",
            order.order_type.value, order.side.value,
            order.symbol, order.qty,
        )

    def cancel(self, order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.is_complete:
                return False
            order.status = OrderStatus.CANCELLED
        logger.info("OrderBook: cancelled %s", order_id)
        return True

    def update(self, order: Order) -> None:
        with self._lock:
            self._orders[order.order_id] = order

    def reject(self, order_id: str, reason: str = "") -> None:
        with self._lock:
            order = self._orders.get(order_id)
            if order:
                order.status = OrderStatus.REJECTED
                order.client_ref = reason
        logger.warning("OrderBook: rejected %s reason=%s", order_id, reason)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, order_id: str) -> Optional[Order]:
        with self._lock:
            return self._orders.get(order_id)

    @property
    def open_orders(self) -> List[Order]:
        with self._lock:
            return [o for o in self._orders.values() if o.is_active]

    def orders_by_symbol(self, symbol: str) -> List[Order]:
        with self._lock:
            return [o for o in self._orders.values() if o.symbol == symbol]

    def orders_by_status(self, status: OrderStatus) -> List[Order]:
        with self._lock:
            return [o for o in self._orders.values() if o.status == status]

    @property
    def total_orders(self) -> int:
        with self._lock:
            return len(self._orders)

    def to_list(self) -> List[dict]:
        with self._lock:
            return [o.to_dict() for o in self._orders.values()]
