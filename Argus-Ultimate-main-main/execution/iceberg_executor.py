"""
execution/iceberg_executor.py
==============================
Iceberg (reserve / hidden) order execution for large orders.

Breaks a large order into random-sized visible slices to reduce market
impact and avoid detectable submission patterns.  Each slice is submitted
as a regular limit order; the executor monitors fills via polling and
replenishes the next slice when the active slice is almost fully consumed.

Provides:
  - IcebergState       — dataclass tracking one iceberg order's lifecycle
  - IcebergExecutor    — async executor managing concurrent iceberg orders
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


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class IcebergStatus(str, Enum):
    ACTIVE    = "ACTIVE"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# IcebergState dataclass
# ---------------------------------------------------------------------------

@dataclass
class IcebergState:
    """
    Complete lifecycle state for one iceberg order.

    Fields
    ------
    iceberg_id           : str   — unique identifier for this iceberg
    symbol               : str   — instrument symbol
    side                 : str   — "buy" or "sell"
    total_size           : float — full order quantity
    filled_size          : float — quantity filled so far
    remaining_size       : float — total_size - filled_size
    active_slice_order_id: str | None — exchange order_id of the current slice
    active_slice_size    : float — size submitted in the current slice
    slice_count          : int   — number of slices submitted so far
    avg_fill_price       : float — VWAP of all fills
    start_time_ns        : int   — perf_counter_ns at iceberg creation
    status               : IcebergStatus
    _fill_notional       : float — internal: running notional for VWAP
    """
    iceberg_id           : str
    symbol               : str
    side                 : str
    total_size           : float
    price                : float
    exchange_name        : str
    filled_size          : float          = 0.0
    active_slice_order_id: Optional[str]  = None
    active_slice_size    : float          = 0.0
    active_slice_filled  : float          = 0.0
    slice_count          : int            = 0
    avg_fill_price       : float          = 0.0
    start_time_ns        : int            = field(default_factory=time.perf_counter_ns)
    status               : IcebergStatus  = IcebergStatus.ACTIVE
    _fill_notional       : float          = field(default=0.0, repr=False)

    @property
    def remaining_size(self) -> float:
        return max(0.0, self.total_size - self.filled_size)

    @property
    def active_slice_remaining(self) -> float:
        return max(0.0, self.active_slice_size - self.active_slice_filled)

    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self.start_time_ns) / 1_000_000.0

    def apply_fill(self, fill_size: float, fill_price: float) -> None:
        """Update fill accounting with a new fill event."""
        effective = min(fill_size, self.remaining_size)
        self._fill_notional      += effective * fill_price
        self.filled_size         += effective
        self.active_slice_filled += effective
        if self.filled_size > 0:
            self.avg_fill_price = self._fill_notional / self.filled_size

    def to_dict(self) -> dict:
        return {
            "iceberg_id":            self.iceberg_id,
            "symbol":                self.symbol,
            "side":                  self.side,
            "exchange_name":         self.exchange_name,
            "total_size":            self.total_size,
            "filled_size":           self.filled_size,
            "remaining_size":        self.remaining_size,
            "active_slice_order_id": self.active_slice_order_id,
            "active_slice_size":     self.active_slice_size,
            "active_slice_remaining": self.active_slice_remaining,
            "slice_count":           self.slice_count,
            "avg_fill_price":        self.avg_fill_price,
            "status":                self.status.value,
            "elapsed_ms":            self.elapsed_ms(),
        }


# ---------------------------------------------------------------------------
# IcebergExecutor
# ---------------------------------------------------------------------------

class IcebergExecutor:
    """
    Async iceberg order executor.

    Parameters
    ----------
    exchange            : Any   — exchange client with async interface:
                                  submit_order(symbol, side, size, price, order_type) -> dict
                                  get_order(order_id) -> dict
                                  cancel_order(order_id) -> bool
    min_visible_pct     : float — minimum slice as fraction of total_size (default 0.10)
    max_visible_pct     : float — maximum slice as fraction of total_size (default 0.20)
    replenish_threshold : float — trigger next slice when active remaining < this
                                  fraction of total_size (default 0.05)
    poll_interval_ms    : float — polling interval in milliseconds (default 200)
    price_tolerance_ticks: int  — adverse price moves (ticks) before we cancel (default 2)
    tick_size           : float — tick size for adverse move calculation (default 0.01)
    """

    def __init__(
        self,
        exchange: Any,
        min_visible_pct: float     = 0.10,
        max_visible_pct: float     = 0.20,
        replenish_threshold: float = 0.05,
        poll_interval_ms: float    = 200.0,
        price_tolerance_ticks: int = 2,
        tick_size: float           = 0.01,
    ) -> None:
        if min_visible_pct >= max_visible_pct:
            raise ValueError(
                f"min_visible_pct ({min_visible_pct}) must be < max_visible_pct ({max_visible_pct})"
            )
        if not 0 < min_visible_pct < 1:
            raise ValueError("min_visible_pct must be in (0, 1)")
        if not 0 < max_visible_pct <= 1:
            raise ValueError("max_visible_pct must be in (0, 1]")

        self._exchange              = exchange
        self._min_visible_pct       = min_visible_pct
        self._max_visible_pct       = max_visible_pct
        self._replenish_threshold   = replenish_threshold
        self._poll_interval_s       = poll_interval_ms / 1000.0
        self._price_tolerance_ticks = price_tolerance_ticks
        self._tick_size             = tick_size

        # Active icebergs: iceberg_id → IcebergState
        self._icebergs: Dict[str, IcebergState] = {}

        # Background task handles
        self._tasks: Dict[str, asyncio.Task] = {}  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_iceberg(
        self,
        symbol: str,
        side: str,
        total_size: float,
        price: float,
        exchange_name: str,
    ) -> IcebergState:
        """
        Begin executing an iceberg order.

        Calculates a random visible slice, submits it, then starts a
        background polling loop that replenishes slices until the full
        total_size is filled or the order is cancelled.

        Parameters
        ----------
        symbol        : str   — instrument symbol
        side          : str   — "buy" or "sell"
        total_size    : float — total quantity to execute
        price         : float — limit price for slices
        exchange_name : str   — exchange name (informational / for state)

        Returns
        -------
        IcebergState
            The IcebergState object.  Monitor .status and .filled_size
            to track progress; await the background task for completion.
        """
        side = side.lower()
        iceberg_id = str(uuid.uuid4())

        state = IcebergState(
            iceberg_id    = iceberg_id,
            symbol        = symbol,
            side          = side,
            total_size    = total_size,
            price         = price,
            exchange_name = exchange_name,
        )
        self._icebergs[iceberg_id] = state

        # Submit initial slice
        slice_size = self._random_slice_size(total_size)
        await self._submit_slice(state, slice_size)

        # Launch background polling loop
        task = asyncio.create_task(
            self._polling_loop(state),
            name=f"iceberg_{iceberg_id[:8]}",
        )
        self._tasks[iceberg_id] = task

        logger.info(
            "Iceberg %s started: %s %s %.6f @ %.6f | initial_slice=%.6f",
            iceberg_id[:8], side, symbol, total_size, price, slice_size,
        )
        return state

    async def cancel_iceberg(self, iceberg_id: str) -> bool:
        """
        Cancel all active slices for an iceberg order.

        Parameters
        ----------
        iceberg_id : str — iceberg_id from IcebergState

        Returns
        -------
        bool — True if cancellation was initiated successfully
        """
        state = self._icebergs.get(iceberg_id)
        if state is None:
            logger.warning("cancel_iceberg: unknown iceberg_id=%s", iceberg_id)
            return False

        if state.status != IcebergStatus.ACTIVE:
            logger.info(
                "cancel_iceberg: iceberg %s already in terminal state %s",
                iceberg_id[:8], state.status.value,
            )
            return False

        # Cancel the active slice on the exchange
        success = True
        if state.active_slice_order_id:
            try:
                success = bool(
                    await self._exchange.cancel_order(state.active_slice_order_id)
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "cancel_iceberg: exchange cancel failed for order %s: %s",
                    state.active_slice_order_id, exc,
                )
                success = False

        state.status = IcebergStatus.CANCELLED
        state.active_slice_order_id = None

        # Cancel background task
        task = self._tasks.get(iceberg_id)
        if task and not task.done():
            task.cancel()

        logger.info(
            "Iceberg %s cancelled | filled=%.6f / %.6f | slices=%d",
            iceberg_id[:8], state.filled_size, state.total_size, state.slice_count,
        )
        return success

    def get_active_icebergs(self) -> List[IcebergState]:
        """Return all icebergs in ACTIVE status."""
        return [s for s in self._icebergs.values() if s.status == IcebergStatus.ACTIVE]

    def get_iceberg(self, iceberg_id: str) -> Optional[IcebergState]:
        return self._icebergs.get(iceberg_id)

    def get_all_icebergs(self) -> List[IcebergState]:
        return list(self._icebergs.values())

    # ------------------------------------------------------------------
    # Background polling loop
    # ------------------------------------------------------------------

    async def _polling_loop(self, state: IcebergState) -> None:
        """
        Poll the exchange for fill updates and manage slice replenishment.

        Loop exits when:
          - total_size is fully filled
          - iceberg is cancelled
          - an unrecoverable error occurs
        """
        while state.status == IcebergStatus.ACTIVE:
            await asyncio.sleep(self._poll_interval_s)

            if state.status != IcebergStatus.ACTIVE:
                break

            try:
                await self._check_and_update_slice(state)
            except asyncio.CancelledError:
                logger.debug("Iceberg %s polling loop cancelled.", state.iceberg_id[:8])
                break
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Iceberg %s polling error: %s — continuing.",
                    state.iceberg_id[:8], exc,
                )

        logger.debug(
            "Iceberg %s polling loop exited with status=%s",
            state.iceberg_id[:8], state.status.value,
        )

    async def _check_and_update_slice(self, state: IcebergState) -> None:
        """
        Query the current slice fill status and decide whether to
        replenish or re-price.
        """
        if not state.active_slice_order_id:
            return

        # Query exchange for fill status
        try:
            order_info = await self._exchange.get_order(state.active_slice_order_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Iceberg %s: get_order failed: %s", state.iceberg_id[:8], exc
            )
            return

        exchange_filled = float(order_info.get("filled_size", 0.0))
        exchange_price  = float(order_info.get("fill_price", state.price))
        order_status    = order_info.get("status", "open")

        # Update fill accounting (delta fill since last poll)
        delta_fill = exchange_filled - state.active_slice_filled
        if delta_fill > 0:
            state.apply_fill(delta_fill, exchange_price)
            logger.debug(
                "Iceberg %s slice fill: +%.6f @ %.6f | total_filled=%.6f / %.6f",
                state.iceberg_id[:8], delta_fill, exchange_price,
                state.filled_size, state.total_size,
            )

        # Check if totally filled
        if state.filled_size >= state.total_size - 1e-9:
            state.status = IcebergStatus.FILLED
            state.active_slice_order_id = None
            logger.info(
                "Iceberg %s FULLY FILLED: %.6f @ avg %.6f in %d slices (%.1f ms)",
                state.iceberg_id[:8], state.filled_size, state.avg_fill_price,
                state.slice_count, state.elapsed_ms(),
            )
            return

        # Check if slice is fully acknowledged as filled by exchange
        if order_status in ("filled", "closed"):
            # Slice done, submit next
            next_size = min(
                self._random_slice_size(state.total_size),
                state.remaining_size,
            )
            if next_size > 0:
                await self._submit_slice(state, next_size)
            return

        # Replenish threshold: remaining visible < replenish_threshold × total
        replenish_trigger = self._replenish_threshold * state.total_size
        if state.active_slice_remaining < replenish_trigger and state.remaining_size > 0:
            next_size = min(
                self._random_slice_size(state.total_size),
                state.remaining_size,
            )
            if next_size > 0:
                logger.debug(
                    "Iceberg %s: replenish triggered (remaining_visible=%.6f < %.6f)",
                    state.iceberg_id[:8], state.active_slice_remaining, replenish_trigger,
                )
                # Cancel current slice first if still open
                await self._try_cancel_current_slice(state)
                await self._submit_slice(state, next_size)
            return

        # Adverse price check: if price moved > tolerance, cancel and re-evaluate
        if await self._price_moved_adversely(state):
            logger.info(
                "Iceberg %s: adverse price move — cancelling slice and re-pricing.",
                state.iceberg_id[:8],
            )
            await self._try_cancel_current_slice(state)
            # Re-price: fetch current best price and re-submit
            new_price = await self._get_current_best_price(state)
            state.price = new_price
            next_size = min(
                self._random_slice_size(state.total_size),
                state.remaining_size,
            )
            if next_size > 0 and state.status == IcebergStatus.ACTIVE:
                await self._submit_slice(state, next_size)

    # ------------------------------------------------------------------
    # Slice management helpers
    # ------------------------------------------------------------------

    async def _submit_slice(self, state: IcebergState, slice_size: float) -> None:
        """Submit a new visible slice to the exchange."""
        state.slice_count += 1
        state.active_slice_size   = slice_size
        state.active_slice_filled = 0.0

        logger.info(
            "Iceberg %s: submitting slice #%d size=%.6f @ %.6f",
            state.iceberg_id[:8], state.slice_count, slice_size, state.price,
        )

        try:
            order_resp = await self._exchange.submit_order(
                symbol     = state.symbol,
                side       = state.side,
                size       = slice_size,
                price      = state.price,
                order_type = "limit",
            )
            state.active_slice_order_id = order_resp.get("order_id", str(uuid.uuid4()))
            logger.debug(
                "Iceberg %s: slice #%d exchange order_id=%s",
                state.iceberg_id[:8], state.slice_count, state.active_slice_order_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Iceberg %s: slice #%d submit failed: %s",
                state.iceberg_id[:8], state.slice_count, exc,
            )
            state.active_slice_order_id = None
            # Decrement slice counter since it didn't go through
            state.slice_count -= 1
            raise

    async def _try_cancel_current_slice(self, state: IcebergState) -> None:
        """Attempt to cancel the currently active slice (best effort)."""
        if not state.active_slice_order_id:
            return
        try:
            await self._exchange.cancel_order(state.active_slice_order_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Iceberg %s: cancel of slice order %s failed: %s",
                state.iceberg_id[:8], state.active_slice_order_id, exc,
            )
        state.active_slice_order_id = None

    async def _price_moved_adversely(self, state: IcebergState) -> bool:
        """
        Returns True if the current best price has moved adversely by
        more than price_tolerance_ticks relative to state.price.
        """
        try:
            current_price = await self._get_current_best_price(state)
            tolerance     = self._price_tolerance_ticks * self._tick_size
            if state.side == "buy":
                # We're buying — adverse if best ask rose above our limit + tolerance
                return current_price > state.price + tolerance
            else:
                # We're selling — adverse if best bid fell below our limit - tolerance
                return current_price < state.price - tolerance
        except Exception:  # noqa: BLE001
            return False

    async def _get_current_best_price(self, state: IcebergState) -> float:
        """
        Attempt to fetch the current best price for re-pricing.
        Falls back to state.price if the exchange doesn't expose this.
        """
        try:
            ticker = await self._exchange.get_ticker(state.symbol)
            if state.side == "buy":
                return float(ticker.get("ask", state.price))
            else:
                return float(ticker.get("bid", state.price))
        except Exception:  # noqa: BLE001
            return state.price

    # ------------------------------------------------------------------
    # Slice size calculation
    # ------------------------------------------------------------------

    def _random_slice_size(self, total_size: float) -> float:
        """
        Return a randomised slice size in [min_visible_pct, max_visible_pct]
        of total_size, rounded to 6 decimal places.

        Randomisation prevents detectable submission patterns.
        """
        pct = random.uniform(self._min_visible_pct, self._max_visible_pct)
        return round(pct * total_size, 6)
