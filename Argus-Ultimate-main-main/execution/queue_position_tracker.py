"""
execution/queue_position_tracker.py
=====================================
L3 order-book queue depth estimation for HFT order management.

Provides:
  - QueuePositionTracker — tracks our orders' estimated position in the
    exchange queue at each price level, using L3 data where available
    and falling back to a heuristic when L3 is absent.

Design notes
------------
*  When L3 data (order-by-order messages) is available the processor
   supplies per-order arrival sequences; we directly count orders ahead
   of ours.

*  Without L3 data we estimate queue position as:
       position ≈ visible_size_ahead / avg_order_size_at_level
   where avg_order_size_at_level is computed from a rolling window of
   the last 500 trades at (symbol, price_level, side).

*  `update_queue` is called on every book delta (cancel / fill on the
   level).  It reduces visible_size_ahead conservatively: we never
   assume our order was hit unless we receive an explicit fill
   notification.

*  `should_cancel_and_rejoin` is used by higher-level executors to
   decide whether stale queue positions warrant a cancel-and-requeue.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass
class TrackedOrder:
    """Represents one of *our* orders registered with the tracker."""
    order_id      : str
    symbol        : str
    side          : str           # "buy" or "sell"
    price         : float
    size          : float
    timestamp_ns  : int           # time we submitted / registered
    queue_position: int = 0       # estimated orders ahead of us
    visible_size_ahead: float = 0.0  # estimated notional size ahead of us
    is_active     : bool = True

    def age_ms(self) -> float:
        return (time.perf_counter_ns() - self.timestamp_ns) / 1_000_000.0


@dataclass
class TradeSample:
    """One trade at a given price level — used for avg-order-size rolling window."""
    size: float
    timestamp_ns: int


# Key for per-(symbol, side, price_level) data
_LevelKey = Tuple[str, str, float]


# ---------------------------------------------------------------------------
# QueuePositionTracker
# ---------------------------------------------------------------------------

class QueuePositionTracker:
    """
    Estimates our orders' queue depth at each price level.

    Parameters
    ----------
    order_book_processor : Any
        Reference to the application's OrderBookProcessor.  The tracker
        uses it to query current visible size at a price level via:
          - order_book_processor.get_level_size(symbol, side, price) -> float
          - order_book_processor.get_l3_queue(symbol, side, price)   -> list[dict] | None
        If the processor does not expose these methods, heuristic-only
        mode is used.

    avg_order_window      : int   — rolling-window size for avg-order-size (default 500)
    cancel_queue_threshold: int   — queue position threshold triggering cancel advice
    cancel_age_threshold_ms: float — minimum order age before cancellation is advised
    """

    def __init__(
        self,
        order_book_processor: Any,
        avg_order_window: int = 500,
        cancel_queue_threshold: int = 50_000,
        cancel_age_threshold_ms: float = 500.0,
    ) -> None:
        self._obp                    = order_book_processor
        self._avg_order_window       = avg_order_window
        self._cancel_queue_threshold = cancel_queue_threshold
        self._cancel_age_threshold_ms= cancel_age_threshold_ms

        # Active tracked orders: order_id → TrackedOrder
        self._orders: Dict[str, TrackedOrder] = {}

        # Rolling trade samples per price level: (symbol, side, price) → deque
        self._trade_samples: DefaultDict[_LevelKey, Deque[TradeSample]] = defaultdict(
            lambda: deque(maxlen=self._avg_order_window)
        )

        # Level metadata: visible size ahead of earliest registered order
        # (symbol, side, price) → float
        self._level_size_ahead: DefaultDict[_LevelKey, float] = defaultdict(float)

        # Snapshot of queue depth at time of registration
        # (symbol, side, price) → (size_at_registration, l3_position)
        self._level_snapshot: Dict[_LevelKey, Tuple[float, int]] = {}

    # ------------------------------------------------------------------
    # Estimate queue depth (external query — does NOT require our order)
    # ------------------------------------------------------------------

    def estimate_queue_depth(
        self,
        symbol: str,
        side: str,
        price_level: float,
    ) -> int:
        """
        Return estimated number of orders ahead at *price_level* on *side*.

        Uses L3 data if available, otherwise falls back to heuristic.

        Parameters
        ----------
        symbol      : str   — instrument symbol
        side        : str   — "buy" or "sell"
        price_level : float — exact price level to query

        Returns
        -------
        int — estimated number of orders ahead (lower bound)
        """
        side = side.lower()
        l3_queue = self._get_l3_queue(symbol, side, price_level)

        if l3_queue is not None:
            # L3 available — direct count
            return len(l3_queue)

        # Heuristic: visible_size / avg_order_size
        visible_size = self._get_visible_size(symbol, side, price_level)
        avg_size     = self._avg_order_size(symbol, side, price_level)

        if avg_size <= 0:
            # No trade history yet; return a conservative estimate
            return int(visible_size)

        return max(0, int(visible_size / avg_size))

    # ------------------------------------------------------------------
    # Register our order
    # ------------------------------------------------------------------

    def register_our_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        size: float,
        timestamp_ns: int,
    ) -> None:
        """
        Register one of our limit orders with the tracker so that
        subsequent book deltas can be used to update our queue position.

        Parameters
        ----------
        order_id     : str   — unique order identifier
        symbol       : str
        side         : str   — "buy" or "sell"
        price        : float — limit price
        size         : float — order size
        timestamp_ns : int   — nanosecond timestamp of submission
        """
        side = side.lower()
        key  = (symbol, side, price)

        # Snapshot current visible size and L3 position at registration time
        visible_size = self._get_visible_size(symbol, side, price)
        l3_queue     = self._get_l3_queue(symbol, side, price)
        l3_pos       = len(l3_queue) if l3_queue is not None else -1
        self._level_snapshot[key] = (visible_size, l3_pos)

        # Estimate queue position at time of registration
        queue_position = self.estimate_queue_depth(symbol, side, price)

        to = TrackedOrder(
            order_id          = order_id,
            symbol            = symbol,
            side              = side,
            price             = price,
            size              = size,
            timestamp_ns      = timestamp_ns,
            queue_position    = queue_position,
            visible_size_ahead= visible_size,
        )
        self._orders[order_id] = to

        logger.debug(
            "QueueTracker: registered order %s | %s %s @ %.6f | "
            "est_queue=%d | visible_ahead=%.2f",
            order_id, side, symbol, price, queue_position, visible_size,
        )

    # ------------------------------------------------------------------
    # Update on book deltas
    # ------------------------------------------------------------------

    def update_queue(
        self,
        symbol: str,
        side: str,
        price_level: float,
        cancelled_size: float,
        filled_size: float,
    ) -> None:
        """
        Process a book delta at (*symbol*, *side*, *price_level*).

        Reduces the visible-size-ahead estimate for all our orders at
        this price level by (cancelled_size + filled_size).  Records
        the filled_size as a trade sample for the rolling avg-order-size.

        Parameters
        ----------
        symbol        : str   — instrument symbol
        side          : str   — "buy" or "sell"
        price_level   : float
        cancelled_size: float — quantity cancelled at this level
        filled_size   : float — quantity filled at this level (trades)
        """
        side = side.lower()
        key  = (symbol, side, price_level)

        # Record trade sample for avg-order-size estimation
        if filled_size > 0:
            self._trade_samples[key].append(
                TradeSample(size=filled_size, timestamp_ns=time.perf_counter_ns())
            )

        total_depleted = cancelled_size + filled_size
        if total_depleted <= 0:
            return

        # Update all our active orders at this price level
        for order_id, to in list(self._orders.items()):
            if not to.is_active:
                continue
            if to.symbol != symbol or to.side != side or to.price != price_level:
                continue

            # Reduce visible_size_ahead, floor at 0
            before = to.visible_size_ahead
            to.visible_size_ahead = max(0.0, to.visible_size_ahead - total_depleted)

            # Recompute estimated queue position
            avg_size = self._avg_order_size(symbol, side, price_level)
            if avg_size > 0:
                to.queue_position = max(0, int(to.visible_size_ahead / avg_size))
            else:
                to.queue_position = max(0, to.queue_position - 1)

            logger.debug(
                "QueueTracker: order %s | visible_ahead %.2f → %.2f | est_pos=%d",
                order_id, before, to.visible_size_ahead, to.queue_position,
            )

    # ------------------------------------------------------------------
    # Cancel-and-rejoin decision
    # ------------------------------------------------------------------

    def should_cancel_and_rejoin(
        self,
        order_id: str,
        max_wait_ms: float = 500.0,
    ) -> bool:
        """
        Return True if it makes tactical sense to cancel the order and
        re-queue at the current best price.

        Criteria (both must be satisfied):
          1. Estimated queue depth ahead > cancel_queue_threshold
          2. Order is older than max_wait_ms (and cancel_age_threshold_ms)

        Parameters
        ----------
        order_id    : str   — must have been registered via register_our_order
        max_wait_ms : float — caller-supplied maximum wait time override

        Returns
        -------
        bool
        """
        to = self._orders.get(order_id)
        if to is None:
            logger.warning("QueueTracker: unknown order_id=%s in should_cancel_and_rejoin", order_id)
            return False
        if not to.is_active:
            return False

        age_ms           = to.age_ms()
        min_age          = min(max_wait_ms, self._cancel_age_threshold_ms)
        age_old_enough   = age_ms >= min_age
        queue_too_deep   = to.queue_position > self._cancel_queue_threshold

        decision = age_old_enough and queue_too_deep

        logger.debug(
            "QueueTracker: should_cancel_and_rejoin(%s) "
            "age=%.1f ms (min=%.1f) queue=%d (threshold=%d) → %s",
            order_id, age_ms, min_age,
            to.queue_position, self._cancel_queue_threshold,
            decision,
        )
        return decision

    # ------------------------------------------------------------------
    # Mark order inactive (filled / cancelled)
    # ------------------------------------------------------------------

    def deactivate_order(self, order_id: str) -> None:
        """Mark an order as no longer active (filled, cancelled, expired)."""
        to = self._orders.get(order_id)
        if to:
            to.is_active = False
            logger.debug("QueueTracker: deactivated order %s", order_id)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_queue_stats(self, symbol: str) -> dict:
        """
        Return queue statistics for *symbol*.

        Includes:
          - avg_queue_depth_bid : average estimated depth at best bid
          - avg_queue_depth_ask : average estimated depth at best ask
          - our_orders          : list of our active orders for this symbol
        """
        our_active = [
            {
                "order_id":          to.order_id,
                "side":              to.side,
                "price":             to.price,
                "size":              to.size,
                "queue_position":    to.queue_position,
                "visible_size_ahead": to.visible_size_ahead,
                "age_ms":            to.age_ms(),
            }
            for to in self._orders.values()
            if to.symbol == symbol and to.is_active
        ]

        # Compute average queue depth for each side across active orders
        bid_positions = [o["queue_position"] for o in our_active if o["side"] == "buy"]
        ask_positions = [o["queue_position"] for o in our_active if o["side"] == "sell"]

        return {
            "symbol":               symbol,
            "our_active_orders":    our_active,
            "avg_queue_position_bid": (
                sum(bid_positions) / len(bid_positions) if bid_positions else 0
            ),
            "avg_queue_position_ask": (
                sum(ask_positions) / len(ask_positions) if ask_positions else 0
            ),
            "total_tracked_orders": len(self._orders),
            "cancel_queue_threshold": self._cancel_queue_threshold,
            "cancel_age_threshold_ms": self._cancel_age_threshold_ms,
        }

    def all_active_orders(self) -> List[TrackedOrder]:
        return [to for to in self._orders.values() if to.is_active]

    def get_order(self, order_id: str) -> Optional[TrackedOrder]:
        return self._orders.get(order_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_visible_size(self, symbol: str, side: str, price: float) -> float:
        """Query visible size at a price level from the order book processor."""
        try:
            return float(self._obp.get_level_size(symbol, side, price))
        except (AttributeError, TypeError, Exception):  # noqa: BLE001
            return 0.0

    def _get_l3_queue(
        self, symbol: str, side: str, price: float
    ) -> Optional[List[dict]]:
        """
        Retrieve the L3 order queue at a price level.

        Returns None if L3 data is not available or the processor does
        not expose this method.
        """
        try:
            queue = self._obp.get_l3_queue(symbol, side, price)
            if isinstance(queue, list):
                return queue
        except (AttributeError, TypeError, Exception):  # noqa: BLE001
            pass
        return None

    def _avg_order_size(self, symbol: str, side: str, price: float) -> float:
        """
        Rolling average order size at (*symbol*, *side*, *price*).

        Returns 0.0 if no trade samples are available.
        """
        key     = (symbol, side, price)
        samples = self._trade_samples.get(key)
        if not samples:
            return 0.0
        return sum(s.size for s in samples) / len(samples)

    def _rolling_avg_order_size_all_levels(self, symbol: str, side: str) -> float:
        """
        Fallback: average order size across all price levels for a symbol/side.
        Used when the specific price level has no trade history.
        """
        all_samples: List[float] = []
        for (sym, sd, _price), samples in self._trade_samples.items():
            if sym == symbol and sd == side:
                all_samples.extend(s.size for s in samples)
        if not all_samples:
            return 0.0
        return sum(all_samples) / len(all_samples)
