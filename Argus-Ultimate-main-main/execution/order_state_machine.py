"""
execution/order_state_machine.py
=================================
Explicit finite-state machine for HFT order lifecycle tracking.

Provides:
  - OrderState / OrderEvent enumerations
  - OrderStateMachine  — per-order FSM with fill accounting
  - OrderBook          — session-level order registry / position tracker
  - InvalidTransition  — exception raised on illegal state transitions
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderState(Enum):
    PENDING          = "PENDING"
    OPEN             = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED           = "FILLED"
    CANCELLED        = "CANCELLED"
    REJECTED         = "REJECTED"
    EXPIRED          = "EXPIRED"


class OrderEvent(Enum):
    SUBMIT      = "SUBMIT"
    ACK         = "ACK"
    PARTIAL_FILL = "PARTIAL_FILL"
    FULL_FILL   = "FULL_FILL"
    CANCEL_REQ  = "CANCEL_REQ"
    CANCEL_ACK  = "CANCEL_ACK"
    REJECT      = "REJECT"
    EXPIRE      = "EXPIRE"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class InvalidTransition(Exception):
    """Raised when an OrderEvent is illegal for the current OrderState."""

    def __init__(self, order_id: str, current_state: OrderState, event: OrderEvent) -> None:
        self.order_id      = order_id
        self.current_state = current_state
        self.event         = event
        super().__init__(
            f"Order {order_id}: cannot apply event {event.value} "
            f"in state {current_state.value}"
        )


# ---------------------------------------------------------------------------
# OrderStateMachine
# ---------------------------------------------------------------------------

# Transition table:  (current_state, event) -> next_state | None
# None means the state does not change (internal action only).
_TRANSITIONS: Dict[tuple, Optional[OrderState]] = {
    # PENDING ----------------------------------------------------------------
    (OrderState.PENDING, OrderEvent.SUBMIT): OrderState.OPEN,

    # OPEN -------------------------------------------------------------------
    (OrderState.OPEN, OrderEvent.PARTIAL_FILL): OrderState.PARTIALLY_FILLED,
    (OrderState.OPEN, OrderEvent.FULL_FILL):    OrderState.FILLED,
    (OrderState.OPEN, OrderEvent.CANCEL_REQ):   None,          # stays OPEN, flags cancel_pending
    (OrderState.OPEN, OrderEvent.CANCEL_ACK):   OrderState.CANCELLED,
    (OrderState.OPEN, OrderEvent.REJECT):       OrderState.REJECTED,
    (OrderState.OPEN, OrderEvent.EXPIRE):       OrderState.EXPIRED,

    # PARTIALLY_FILLED -------------------------------------------------------
    (OrderState.PARTIALLY_FILLED, OrderEvent.PARTIAL_FILL): OrderState.PARTIALLY_FILLED,
    (OrderState.PARTIALLY_FILLED, OrderEvent.FULL_FILL):    OrderState.FILLED,
    (OrderState.PARTIALLY_FILLED, OrderEvent.CANCEL_REQ):   None,  # stays, flags cancel_pending
    (OrderState.PARTIALLY_FILLED, OrderEvent.CANCEL_ACK):   OrderState.CANCELLED,
}

# Terminal states — no further transitions allowed
_TERMINAL_STATES = {
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
}


class OrderStateMachine:
    """
    Per-order FSM that tracks lifecycle state, fill accounting, and basic PnL.

    Parameters
    ----------
    order_id  : unique order identifier (string)
    symbol    : instrument symbol, e.g. "BTC-USD"
    side      : "buy" or "sell"
    price     : limit price (float; 0 for market orders)
    size      : total order size in base units
    exchange  : exchange / venue name
    """

    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        size: float,
        exchange: str,
    ) -> None:
        if side.lower() not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")

        self.order_id  = order_id
        self.symbol    = symbol
        self.side      = side.lower()
        self.price     = price
        self.size      = size
        self.exchange  = exchange

        self.state: OrderState    = OrderState.PENDING
        self.cancel_pending: bool = False

        # Fill accounting
        self.filled_size: float   = 0.0
        self.avg_fill_price: float = 0.0
        self._fill_notional: float = 0.0   # running notional for VWAP calc

        # PnL
        self.pnl_realised: float  = 0.0

        # Timestamps (nanoseconds via time.perf_counter_ns for sub-ms precision)
        self._created_ns: int     = time.perf_counter_ns()
        self._events: List[dict]  = []     # audit trail

    # ------------------------------------------------------------------
    # Core transition method
    # ------------------------------------------------------------------

    def transition(self, event: OrderEvent, **kwargs: Any) -> OrderState:
        """
        Apply *event* to the current state and return the new state.

        Keyword arguments are event-specific:
          PARTIAL_FILL / FULL_FILL:
            fill_size  (float) — size of this fill
            fill_price (float) — price of this fill
            cost_basis (float, optional) — average cost basis for PnL calc
          CANCEL_REQ / CANCEL_ACK:
            no mandatory kwargs

        Raises
        ------
        InvalidTransition
            If the (state, event) pair has no legal transition defined.
        """
        key = (self.state, event)

        # Guard: terminal states accept no further events
        if self.state in _TERMINAL_STATES:
            raise InvalidTransition(self.order_id, self.state, event)

        if key not in _TRANSITIONS:
            raise InvalidTransition(self.order_id, self.state, event)

        # Process fill accounting *before* changing state
        if event in (OrderEvent.PARTIAL_FILL, OrderEvent.FULL_FILL):
            self._apply_fill(event, **kwargs)

        # Record cancel-pending flag
        if event == OrderEvent.CANCEL_REQ:
            self.cancel_pending = True
            logger.debug(
                "Order %s: cancel requested (state stays %s)",
                self.order_id, self.state.value,
            )

        # Commit state change (None means no change)
        new_state = _TRANSITIONS[key]
        if new_state is not None:
            old_state  = self.state
            self.state = new_state
            logger.debug(
                "Order %s: %s + %s → %s",
                self.order_id, old_state.value, event.value, new_state.value,
            )

        # Audit trail
        self._events.append({
            "event":     event.value,
            "state":     self.state.value,
            "timestamp": time.perf_counter_ns(),
            "kwargs":    {k: v for k, v in kwargs.items()},
        })

        return self.state

    # ------------------------------------------------------------------
    # Fill accounting helpers
    # ------------------------------------------------------------------

    def _apply_fill(self, event: OrderEvent, **kwargs: Any) -> None:
        fill_size  = float(kwargs.get("fill_size",  0.0))
        fill_price = float(kwargs.get("fill_price", self.price))

        if fill_size <= 0:
            logger.warning(
                "Order %s: fill event with non-positive fill_size=%s; ignoring.",
                self.order_id, fill_size,
            )
            return

        # Clamp fill to remaining
        effective_fill = min(fill_size, self.remaining_size)

        self._fill_notional += effective_fill * fill_price
        self.filled_size    += effective_fill

        # VWAP
        if self.filled_size > 0:
            self.avg_fill_price = self._fill_notional / self.filled_size

        # Realised PnL (requires cost_basis)
        # For buys:  pnl = (fill_price - cost_basis) × size   (profitable if fill < cost)
        # For sells: pnl = (fill_price - cost_basis) × size   (profitable if fill > cost)
        # Both sides: positive pnl means we executed at a better price than cost_basis.
        cost_basis = kwargs.get("cost_basis")
        if cost_basis is not None:
            cost_basis = float(cost_basis)
            self.pnl_realised += (fill_price - cost_basis) * effective_fill

        logger.debug(
            "Order %s fill: size=%.6f @ %.6f | total_filled=%.6f | avg=%.6f",
            self.order_id, effective_fill, fill_price,
            self.filled_size, self.avg_fill_price,
        )

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def remaining_size(self) -> float:
        """Unfilled quantity."""
        return max(0.0, self.size - self.filled_size)

    def age_ms(self) -> float:
        """Milliseconds elapsed since order creation (uses perf_counter_ns)."""
        elapsed_ns = time.perf_counter_ns() - self._created_ns
        return elapsed_ns / 1_000_000.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Full state snapshot suitable for logging / persistence."""
        return {
            "order_id":      self.order_id,
            "symbol":        self.symbol,
            "side":          self.side,
            "price":         self.price,
            "size":          self.size,
            "exchange":      self.exchange,
            "state":         self.state.value,
            "cancel_pending": self.cancel_pending,
            "filled_size":   self.filled_size,
            "remaining_size": self.remaining_size,
            "avg_fill_price": self.avg_fill_price,
            "pnl_realised":  self.pnl_realised,
            "age_ms":        self.age_ms(),
            "event_count":   len(self._events),
            "events":        list(self._events),
        }

    def __repr__(self) -> str:
        return (
            f"<OrderStateMachine id={self.order_id} symbol={self.symbol} "
            f"side={self.side} state={self.state.value} "
            f"filled={self.filled_size}/{self.size}>"
        )


# ---------------------------------------------------------------------------
# OrderBook  (order-tracking registry — not an LOB)
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Session-level registry of all orders, with convenience queries for
    open orders, closed orders, net position per symbol, and session PnL.

    Not to be confused with a limit-order-book (LOB); this is purely
    an order-management / reconciliation data structure.
    """

    def __init__(self) -> None:
        self._orders: Dict[str, OrderStateMachine] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, osm: OrderStateMachine) -> None:
        """Add an OrderStateMachine to the book."""
        if osm.order_id in self._orders:
            raise ValueError(f"Order {osm.order_id} already registered.")
        self._orders[osm.order_id] = osm
        logger.debug("OrderBook: registered %s", osm.order_id)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, order_id: str) -> OrderStateMachine:
        """Retrieve an OSM by order_id; raises KeyError if not found."""
        return self._orders[order_id]

    def __contains__(self, order_id: str) -> bool:
        return order_id in self._orders

    def __len__(self) -> int:
        return len(self._orders)

    # ------------------------------------------------------------------
    # Filtered views
    # ------------------------------------------------------------------

    def open_orders(self) -> List[OrderStateMachine]:
        """Orders that are still live (PENDING, OPEN, PARTIALLY_FILLED)."""
        open_states = {
            OrderState.PENDING,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
        }
        return [o for o in self._orders.values() if o.state in open_states]

    def closed_orders(self) -> List[OrderStateMachine]:
        """Orders that have reached a terminal state."""
        return [
            o for o in self._orders.values()
            if o.state in _TERMINAL_STATES
        ]

    def orders_by_symbol(self, symbol: str) -> List[OrderStateMachine]:
        """All orders for a given symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    def orders_by_state(self, state: OrderState) -> List[OrderStateMachine]:
        return [o for o in self._orders.values() if o.state == state]

    # ------------------------------------------------------------------
    # Position / PnL
    # ------------------------------------------------------------------

    def net_position(self, symbol: str) -> float:
        """
        Sum of filled buys minus filled sells for *symbol*.

        Only fills that have actually settled (filled_size > 0) are counted,
        regardless of whether the order is still open or closed.
        """
        net = 0.0
        for osm in self._orders.values():
            if osm.symbol != symbol:
                continue
            if osm.filled_size <= 0:
                continue
            if osm.side == "buy":
                net += osm.filled_size
            else:
                net -= osm.filled_size
        return net

    def net_positions(self) -> Dict[str, float]:
        """Net position for every symbol with at least one fill."""
        symbols: set = {o.symbol for o in self._orders.values() if o.filled_size > 0}
        return {sym: self.net_position(sym) for sym in sorted(symbols)}

    def session_pnl(self) -> float:
        """Sum of pnl_realised across all orders."""
        return sum(o.pnl_realised for o in self._orders.values())

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "total_orders":  len(self._orders),
            "open_orders":   len(self.open_orders()),
            "closed_orders": len(self.closed_orders()),
            "net_positions": self.net_positions(),
            "session_pnl":   self.session_pnl(),
        }

    def __repr__(self) -> str:
        return (
            f"<OrderBook orders={len(self._orders)} "
            f"open={len(self.open_orders())} "
            f"session_pnl={self.session_pnl():.4f}>"
        )
