"""
Iceberg Order Execution
=======================

Splits large orders into smaller visible slices to minimize market impact
and avoid detection by other market participants.

Each slice is placed as a limit order; once filled, the next slice is placed
until the total quantity is exhausted or the order times out.

Visible slice sizes are optionally randomized +/-20% to further obscure intent.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IcebergStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ERROR = "error"


@dataclass
class IcebergSlice:
    """A single visible slice of an iceberg order."""
    slice_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    visible_qty: float = 0.0
    filled_qty: float = 0.0
    price: float = 0.0
    fill_price: float = 0.0
    placed_at: float = 0.0
    filled_at: float = 0.0
    exchange_order_id: Optional[str] = None
    status: str = "pending"  # pending, placed, filled, cancelled


@dataclass
class IcebergOrder:
    """Full iceberg order plan."""
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    symbol: str = ""
    side: str = "buy"  # buy or sell
    total_quantity: float = 0.0
    price: float = 0.0
    visible_pct: float = 0.1
    visible_qty: float = 0.0
    remaining_qty: float = 0.0
    min_visible_qty: float = 0.001
    randomize_visible: bool = True
    max_duration: float = 300.0  # seconds
    status: IcebergStatus = IcebergStatus.PENDING
    slices: List[IcebergSlice] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class IcebergResult:
    """Execution result for a completed iceberg order."""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    total_filled: float = 0.0
    total_quantity: float = 0.0
    avg_price: float = 0.0
    n_slices: int = 0
    duration: float = 0.0
    status: IcebergStatus = IcebergStatus.COMPLETED
    slippage_bps: float = 0.0
    detected_front_running: bool = False
    slices: List[IcebergSlice] = field(default_factory=list)


class IcebergOrderManager:
    """
    Manages iceberg order creation and execution.

    Splits large orders into smaller visible portions to reduce market impact.
    Each slice is placed as a limit order; once filled, the next slice is placed
    until the entire quantity is filled or the order times out.
    """

    def __init__(
        self,
        visible_pct: float = 0.1,
        min_visible_qty: float = 0.001,
        randomize_visible: bool = True,
        max_duration: float = 300.0,
        front_run_threshold_bps: float = 10.0,
    ) -> None:
        """
        Args:
            visible_pct: fraction of total order shown on book (default 10%)
            min_visible_qty: minimum visible quantity per slice
            randomize_visible: add +/-20% randomness to visible size to avoid detection
            max_duration: maximum execution time in seconds before timeout
            front_run_threshold_bps: slippage threshold (bps) to flag potential front-running
        """
        if not 0.0 < visible_pct <= 1.0:
            raise ValueError(f"visible_pct must be in (0, 1], got {visible_pct}")
        if min_visible_qty <= 0:
            raise ValueError(f"min_visible_qty must be positive, got {min_visible_qty}")
        if max_duration <= 0:
            raise ValueError(f"max_duration must be positive, got {max_duration}")

        self.visible_pct = float(visible_pct)
        self.min_visible_qty = float(min_visible_qty)
        self.randomize_visible = bool(randomize_visible)
        self.max_duration = float(max_duration)
        self.front_run_threshold_bps = float(front_run_threshold_bps)
        self._orders: Dict[str, IcebergOrder] = {}
        self._results: Dict[str, IcebergResult] = {}

    def create_iceberg(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        price: float,
        visible_pct: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> IcebergOrder:
        """Create iceberg order plan with visible/hidden split.

        Args:
            symbol: trading pair (e.g. "BTC/USD")
            side: "buy" or "sell"
            total_quantity: total order size
            price: limit price for slices
            visible_pct: override default visible percentage
            max_duration: override default max duration

        Returns:
            IcebergOrder with calculated visible_qty and remaining_qty
        """
        if not symbol or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if total_quantity <= 0:
            raise ValueError(f"total_quantity must be positive, got {total_quantity}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        pct = float(visible_pct if visible_pct is not None else self.visible_pct)
        pct = max(0.01, min(1.0, pct))

        visible_qty = max(self.min_visible_qty, total_quantity * pct)
        # Clamp visible_qty to total_quantity
        visible_qty = min(visible_qty, total_quantity)

        order = IcebergOrder(
            symbol=symbol.strip(),
            side=side,
            total_quantity=float(total_quantity),
            price=float(price),
            visible_pct=pct,
            visible_qty=float(visible_qty),
            remaining_qty=float(total_quantity),
            min_visible_qty=self.min_visible_qty,
            randomize_visible=self.randomize_visible,
            max_duration=float(max_duration if max_duration is not None else self.max_duration),
        )

        self._orders[order.order_id] = order
        logger.info(
            "Iceberg created: id=%s symbol=%s side=%s total=%.6f visible=%.6f price=%.2f",
            order.order_id, symbol, side, total_quantity, visible_qty, price,
        )
        return order

    def _next_visible_qty(self, order: IcebergOrder) -> float:
        """Calculate the next visible slice size, optionally randomized."""
        base = order.visible_qty

        if order.randomize_visible:
            # +/-20% randomization
            factor = 1.0 + random.uniform(-0.20, 0.20)
            base = base * factor

        # Enforce minimum
        base = max(order.min_visible_qty, base)
        # Don't exceed remaining
        base = min(base, order.remaining_qty)
        return float(base)

    async def execute_iceberg(
        self,
        order: IcebergOrder,
        exchange: Any,
        check_interval: float = 2.0,
    ) -> IcebergResult:
        """Execute iceberg: place visible portion, refill when filled, repeat until done.

        Args:
            order: IcebergOrder plan from create_iceberg()
            exchange: exchange object with place_limit_order(symbol, side, qty, price),
                      check_order(order_id), and cancel_order(order_id) methods
            check_interval: seconds between fill checks

        Returns:
            IcebergResult with execution statistics
        """
        if order.status not in (IcebergStatus.PENDING, IcebergStatus.ACTIVE):
            raise ValueError(f"Cannot execute order in status {order.status}")

        order.status = IcebergStatus.ACTIVE
        start_time = time.time()
        total_filled = 0.0
        weighted_price_sum = 0.0
        fill_prices: List[float] = []
        detected_front_running = False

        logger.info("Iceberg execution started: id=%s", order.order_id)

        try:
            while order.remaining_qty > 1e-12:
                elapsed = time.time() - start_time
                if elapsed >= order.max_duration:
                    order.status = IcebergStatus.TIMED_OUT
                    logger.warning(
                        "Iceberg timed out: id=%s filled=%.6f/%.6f after %.1fs",
                        order.order_id, total_filled, order.total_quantity, elapsed,
                    )
                    break

                # Calculate next slice
                slice_qty = self._next_visible_qty(order)
                if slice_qty < order.min_visible_qty:
                    # Remaining is too small for another slice
                    if order.remaining_qty >= order.min_visible_qty * 0.5:
                        slice_qty = order.remaining_qty
                    else:
                        break

                # Create slice record
                current_slice = IcebergSlice(
                    visible_qty=slice_qty,
                    price=order.price,
                    placed_at=time.time(),
                )
                order.slices.append(current_slice)

                # Place limit order
                try:
                    exchange_order_id = await exchange.place_limit_order(
                        order.symbol, order.side, slice_qty, order.price,
                    )
                    current_slice.exchange_order_id = str(exchange_order_id)
                    current_slice.status = "placed"
                except Exception as e:
                    current_slice.status = "cancelled"
                    logger.warning("Iceberg slice placement failed: %s", e)
                    order.status = IcebergStatus.ERROR
                    break

                # Poll for fill
                slice_filled = False
                while not slice_filled:
                    elapsed = time.time() - start_time
                    if elapsed >= order.max_duration:
                        # Cancel outstanding slice
                        try:
                            await exchange.cancel_order(current_slice.exchange_order_id)
                        except Exception as e:
                            logger.debug("Failed to cancel slice on timeout: %s", e)
                        current_slice.status = "cancelled"
                        break

                    await asyncio.sleep(check_interval)

                    try:
                        order_status = await exchange.check_order(
                            current_slice.exchange_order_id
                        )
                    except Exception as e:
                        logger.debug("Iceberg check_order error: %s", e)
                        continue

                    filled_qty = float(order_status.get("filled_qty", 0.0) or 0.0)
                    fill_price = float(
                        order_status.get("avg_price", order.price) or order.price
                    )

                    if filled_qty >= slice_qty * 0.999:
                        # Slice filled
                        current_slice.filled_qty = filled_qty
                        current_slice.fill_price = fill_price
                        current_slice.filled_at = time.time()
                        current_slice.status = "filled"

                        total_filled += filled_qty
                        order.remaining_qty -= filled_qty
                        weighted_price_sum += filled_qty * fill_price
                        fill_prices.append(fill_price)

                        # Check for front-running: price moving adversely between slices
                        if len(fill_prices) >= 2:
                            prev_price = fill_prices[-2]
                            price_move_bps = abs(fill_price - prev_price) / max(prev_price, 1e-9) * 10000
                            if price_move_bps > self.front_run_threshold_bps:
                                detected_front_running = True
                                logger.warning(
                                    "Iceberg potential front-running detected: id=%s move=%.1f bps",
                                    order.order_id, price_move_bps,
                                )

                        slice_filled = True
                        order.status = IcebergStatus.PARTIALLY_FILLED

                        logger.debug(
                            "Iceberg slice filled: id=%s slice=%s qty=%.6f price=%.2f remaining=%.6f",
                            order.order_id, current_slice.slice_id,
                            filled_qty, fill_price, order.remaining_qty,
                        )

            # Determine final status
            if order.remaining_qty <= 1e-12 and total_filled > 0:
                order.status = IcebergStatus.COMPLETED
            elif total_filled == 0 and order.status not in (
                IcebergStatus.ERROR, IcebergStatus.TIMED_OUT
            ):
                order.status = IcebergStatus.CANCELLED

        except Exception as e:
            logger.warning("Iceberg execution error: id=%s error=%s", order.order_id, e)
            order.status = IcebergStatus.ERROR

        duration = time.time() - start_time
        avg_price = (weighted_price_sum / total_filled) if total_filled > 0 else 0.0
        slippage_bps = 0.0
        if total_filled > 0 and order.price > 0:
            slippage_bps = abs(avg_price - order.price) / order.price * 10000

        result = IcebergResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            total_filled=total_filled,
            total_quantity=order.total_quantity,
            avg_price=avg_price,
            n_slices=len([s for s in order.slices if s.status == "filled"]),
            duration=duration,
            status=order.status,
            slippage_bps=slippage_bps,
            detected_front_running=detected_front_running,
            slices=list(order.slices),
        )
        self._results[order.order_id] = result

        logger.info(
            "Iceberg execution finished: id=%s status=%s filled=%.6f/%.6f slices=%d duration=%.1fs slippage=%.2f bps",
            order.order_id, order.status.value, total_filled,
            order.total_quantity, result.n_slices, duration, slippage_bps,
        )

        return result

    def get_execution_report(self, order_id: str) -> dict:
        """Return execution report for a completed/active iceberg order.

        Returns:
            dict with total_filled, avg_price, n_slices, duration, detected_front_running,
            status, slippage_bps, and slice details.
        """
        result = self._results.get(order_id)
        if result is not None:
            return {
                "order_id": result.order_id,
                "symbol": result.symbol,
                "side": result.side,
                "total_filled": result.total_filled,
                "total_quantity": result.total_quantity,
                "avg_price": result.avg_price,
                "n_slices": result.n_slices,
                "duration": result.duration,
                "status": result.status.value,
                "slippage_bps": result.slippage_bps,
                "detected_front_running": result.detected_front_running,
                "slices": [
                    {
                        "slice_id": s.slice_id,
                        "visible_qty": s.visible_qty,
                        "filled_qty": s.filled_qty,
                        "price": s.price,
                        "fill_price": s.fill_price,
                        "status": s.status,
                    }
                    for s in result.slices
                ],
            }

        order = self._orders.get(order_id)
        if order is not None:
            return {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "total_filled": order.total_quantity - order.remaining_qty,
                "total_quantity": order.total_quantity,
                "avg_price": 0.0,
                "n_slices": len(order.slices),
                "duration": time.time() - order.created_at,
                "status": order.status.value,
                "slippage_bps": 0.0,
                "detected_front_running": False,
                "slices": [],
            }

        return {}

    def cancel_order(self, order_id: str) -> bool:
        """Mark an iceberg order as cancelled. Returns True if found and cancelled."""
        order = self._orders.get(order_id)
        if order is None:
            return False
        if order.status in (IcebergStatus.COMPLETED, IcebergStatus.CANCELLED):
            return False
        order.status = IcebergStatus.CANCELLED
        logger.info("Iceberg cancelled: id=%s", order_id)
        return True

    @property
    def active_orders(self) -> List[IcebergOrder]:
        """Return list of currently active iceberg orders."""
        return [
            o for o in self._orders.values()
            if o.status in (IcebergStatus.ACTIVE, IcebergStatus.PARTIALLY_FILLED, IcebergStatus.PENDING)
        ]
