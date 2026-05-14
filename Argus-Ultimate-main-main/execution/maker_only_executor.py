"""Batch 1 — Maker-only execution engine.

Places limit orders at best-bid/ask to qualify for maker rebates.  
If the order isn't filled within `fill_timeout_s` it is cancelled and
re-priced with an optional aggression step.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: Side
    requested_qty: float
    filled_qty: float
    avg_price: float
    status: str
    latency_ms: float
    maker: bool = True
    metadata: Dict = field(default_factory=dict)


class MakerOnlyExecutor:
    """Submits post-only limit orders and manages fill/cancel lifecycle."""

    def __init__(
        self,
        exchange,  # ccxt.async_support exchange instance
        fill_timeout_s: float = 30.0,
        max_reprice_attempts: int = 3,
        aggression_tick: float = 0.0001,  # fraction of price per reprice
    ) -> None:
        self._exchange = exchange
        self._fill_timeout_s = fill_timeout_s
        self._max_reprice = max_reprice_attempts
        self._aggression_tick = aggression_tick

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        symbol: str,
        side: Side,
        qty: float,
        reference_price: Optional[float] = None,
    ) -> OrderResult:
        """Place a maker-only order; reprice up to max_reprice_attempts times."""
        price = reference_price or await self._get_best_price(symbol, side)
        t0 = time.monotonic()

        for attempt in range(self._max_reprice + 1):
            order_id, placed_price = await self._place_post_only(
                symbol, side, qty, price
            )
            filled_qty, avg_price, status = await self._wait_for_fill(
                symbol, order_id
            )
            latency_ms = (time.monotonic() - t0) * 1000

            if status == "closed":
                logger.info(
                    "Maker fill: %s %s %.6f @ %.6f in %.1fms",
                    side,
                    symbol,
                    filled_qty,
                    avg_price,
                    latency_ms,
                )
                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    requested_qty=qty,
                    filled_qty=filled_qty,
                    avg_price=avg_price,
                    status="filled",
                    latency_ms=latency_ms,
                    maker=True,
                )

            # Unfilled — cancel and reprice
            await self._cancel(symbol, order_id)
            remaining = qty - filled_qty
            if remaining <= 0 or attempt == self._max_reprice:
                break

            # Nudge price toward the book
            tick = placed_price * self._aggression_tick
            price = placed_price + tick if side == Side.BUY else placed_price - tick
            logger.warning(
                "Repricing %s %s attempt %d → %.6f", symbol, side, attempt + 1, price
            )
            qty = remaining

        return OrderResult(
            order_id="",
            symbol=symbol,
            side=side,
            requested_qty=qty,
            filled_qty=filled_qty if 'filled_qty' in dir() else 0.0,
            avg_price=avg_price if 'avg_price' in dir() else 0.0,
            status="partial" if (filled_qty if 'filled_qty' in dir() else 0) > 0 else "cancelled",
            latency_ms=(time.monotonic() - t0) * 1000,
            maker=True,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_best_price(self, symbol: str, side: Side) -> float:
        ob = await self._exchange.fetch_order_book(symbol, limit=1)
        if side == Side.BUY:
            return float(ob["bids"][0][0]) if ob["bids"] else 0.0
        return float(ob["asks"][0][0]) if ob["asks"] else 0.0

    async def _place_post_only(
        self, symbol: str, side: Side, qty: float, price: float
    ):
        params = {"postOnly": True, "timeInForce": "GTX"}
        order = await self._exchange.create_limit_order(
            symbol, side.value, qty, price, params=params
        )
        return order["id"], float(order.get("price", price))

    async def _wait_for_fill(
        self, symbol: str, order_id: str
    ):
        deadline = time.monotonic() + self._fill_timeout_s
        while time.monotonic() < deadline:
            order = await self._exchange.fetch_order(order_id, symbol)
            status = order.get("status", "open")
            filled = float(order.get("filled", 0))
            avg_price = float(order.get("average", 0) or order.get("price", 0))
            if status in ("closed", "canceled", "cancelled", "expired"):
                return filled, avg_price, status
            await asyncio.sleep(0.25)
        return 0.0, 0.0, "timeout"

    async def _cancel(self, symbol: str, order_id: str) -> None:
        try:
            await self._exchange.cancel_order(order_id, symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cancel failed for %s: %s", order_id, exc)
