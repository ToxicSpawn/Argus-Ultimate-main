"""FillSimulator — deterministic paper-trading fill engine — Push 56.

Fill rules:
  MARKET  — immediate fill at mid ± half-spread + slippage.
  LIMIT   — fills when market price crosses limit price.
  STOP    — triggers when price breaches stop; converts to market.
  STOP_LIMIT — triggers on stop, then limit fill logic.

Configuration via constructor or env::

    ARGUS_SIM_SPREAD_BPS     half-spread in bps (default 1.0)
    ARGUS_SIM_SLIPPAGE_BPS   slippage in bps (default 0.5)
    ARGUS_SIM_FILL_PROB      probability of fill per tick (default 1.0)
"""
from __future__ import annotations

import logging
import os
import random
from typing import List, Optional

from core.execution.order_models import Fill, Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)

_SPREAD_BPS = float(os.getenv("ARGUS_SIM_SPREAD_BPS", "1.0"))
_SLIPPAGE_BPS = float(os.getenv("ARGUS_SIM_SLIPPAGE_BPS", "0.5"))
_FILL_PROB = float(os.getenv("ARGUS_SIM_FILL_PROB", "1.0"))


class FillSimulator:
    """Simulates order fills for paper trading.

    Parameters
    ----------
    spread_bps : float
        Half-spread in basis points.
    slippage_bps : float
        Additional slippage in basis points.
    fill_probability : float
        Probability [0, 1] that a limit order fills on a given tick.
    fee_bps : float
        Taker fee in basis points.
    """

    def __init__(
        self,
        spread_bps: float = _SPREAD_BPS,
        slippage_bps: float = _SLIPPAGE_BPS,
        fill_probability: float = _FILL_PROB,
        fee_bps: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        self._spread_bps = spread_bps
        self._slippage_bps = slippage_bps
        self._fill_prob = fill_probability
        self._fee_bps = fee_bps
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Main tick processor
    # ------------------------------------------------------------------

    def process_tick(
        self, order: Order, mid_price: float, venue: str = "sim"
    ) -> List[Fill]:
        """Attempt to fill an order given the current market mid price.

        Returns a list of Fill objects (empty if no fill).
        """
        if order.is_complete or not order.is_active:
            return []

        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, mid_price, venue)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, mid_price, venue)
        elif order.order_type == OrderType.STOP:
            return self._fill_stop(order, mid_price, venue)
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._fill_stop_limit(order, mid_price, venue)
        return []

    # ------------------------------------------------------------------
    # Fill implementations
    # ------------------------------------------------------------------

    def _effective_price(self, mid: float, side: OrderSide) -> float:
        """Apply spread and slippage to get effective fill price."""
        bps = (self._spread_bps + self._slippage_bps) / 10_000
        if side == OrderSide.BUY:
            return mid * (1 + bps)
        return mid * (1 - bps)

    def _make_fill(self, order: Order, price: float, qty: float, venue: str) -> Fill:
        notional = price * qty
        fee = notional * (self._fee_bps / 10_000)
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=price,
            qty=qty,
            fee=fee,
            venue=venue,
        )

    def _fill_market(self, order: Order, mid: float, venue: str) -> List[Fill]:
        price = self._effective_price(mid, order.side)
        fill = self._make_fill(order, price, order.remaining_qty, venue)
        order.add_fill(fill)
        logger.debug("Sim: MARKET fill %s @ %.4f qty=%.6f", order.symbol, price, fill.qty)
        return [fill]

    def _fill_limit(self, order: Order, mid: float, venue: str) -> List[Fill]:
        if order.limit_price is None:
            return []
        if self._rng.random() > self._fill_prob:
            return []
        # Buy limit fills when mid <= limit; sell limit fills when mid >= limit
        if order.side == OrderSide.BUY and mid <= order.limit_price:
            price = min(mid, order.limit_price)
            fill = self._make_fill(order, price, order.remaining_qty, venue)
            order.add_fill(fill)
            return [fill]
        elif order.side == OrderSide.SELL and mid >= order.limit_price:
            price = max(mid, order.limit_price)
            fill = self._make_fill(order, price, order.remaining_qty, venue)
            order.add_fill(fill)
            return [fill]
        return []

    def _fill_stop(self, order: Order, mid: float, venue: str) -> List[Fill]:
        if order.stop_price is None:
            return []
        # Buy stop triggers when mid >= stop; sell stop when mid <= stop
        triggered = (
            (order.side == OrderSide.BUY and mid >= order.stop_price) or
            (order.side == OrderSide.SELL and mid <= order.stop_price)
        )
        if triggered:
            price = self._effective_price(mid, order.side)
            fill = self._make_fill(order, price, order.remaining_qty, venue)
            order.add_fill(fill)
            logger.debug("Sim: STOP triggered %s @ %.4f", order.symbol, price)
            return [fill]
        return []

    def _fill_stop_limit(
        self, order: Order, mid: float, venue: str
    ) -> List[Fill]:
        if order.stop_price is None or order.limit_price is None:
            return []
        stop_triggered = (
            (order.side == OrderSide.BUY and mid >= order.stop_price) or
            (order.side == OrderSide.SELL and mid <= order.stop_price)
        )
        if not stop_triggered:
            return []
        # Now apply limit logic
        order.order_type = OrderType.LIMIT
        return self._fill_limit(order, mid, venue)
