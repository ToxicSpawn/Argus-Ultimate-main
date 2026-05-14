"""
execution/inventory_unwind.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-of-session TWAP inventory unwind.

Design
------
When approaching session close with a non-trivial inventory position, the
scheduler automatically slices the position into equal TWAP tranches and
works each tranche through a limit → market fallback cycle.

Slice execution flow
--------------------
1. Post a limit order at mid ± tolerance_bps.
2. Wait up to 60 s for a fill.
3. If unfilled, cancel and replace with a market order.
4. Track partial fills; adjust remaining slice size on each iteration.
5. Stop early if the position crosses zero (avoid over-shooting).
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class UnwindStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class UnwindState:
    """Live state for an active unwind."""
    symbol: str
    original_size: float          # Signed: positive = long, negative = short
    unwound_size: float = 0.0     # How much we have already unwound (unsigned)
    remaining_size: float = 0.0   # unsigned quantity still to go
    slices_total: int = 0
    slices_done: int = 0
    avg_fill_price: float = 0.0
    start_time_ns: int = 0
    status: UnwindStatus = UnwindStatus.PENDING
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.remaining_size == 0.0:
            self.remaining_size = abs(self.original_size)


@dataclass
class UnwindResult:
    """Final result from a completed (or cancelled) unwind."""
    symbol: str
    total_filled: float
    avg_price: float
    slippage_bps: float          # (avg_price - target_mid) / target_mid × 10 000 (unsigned)
    slices_used: int
    duration_s: float
    status: UnwindStatus


# ---------------------------------------------------------------------------
# Exchange interface shim
# ---------------------------------------------------------------------------

class _ExchangeInterface:
    """
    Thin wrapper around an exchange connector object.

    The real connector is expected to provide (at minimum):
      - async place_limit_order(symbol, side, size, price) → order_id
      - async place_market_order(symbol, side, size) → fill_price
      - async cancel_order(symbol, order_id) → bool
      - async get_order_fill(order_id) → Optional[float]  (None if not filled)
      - async get_mid_price(symbol) → float

    This wrapper adds defensive None-checks and converts between the
    InventoryUnwindScheduler's needs and whatever the real exchange object
    provides.  Tests can inject a mock object.
    """

    def __init__(self, exchange: Any) -> None:
        self._ex = exchange

    async def get_mid(self, symbol: str) -> float:
        if hasattr(self._ex, "get_mid_price"):
            return await self._ex.get_mid_price(symbol)
        if hasattr(self._ex, "get_mid"):
            return await self._ex.get_mid(symbol)
        raise NotImplementedError("Exchange must implement get_mid_price or get_mid")

    async def place_limit(
        self, symbol: str, side: str, size: float, price: float
    ) -> str:
        if hasattr(self._ex, "place_limit_order"):
            return await self._ex.place_limit_order(symbol, side, size, price)
        raise NotImplementedError("Exchange must implement place_limit_order")

    async def place_market(
        self, symbol: str, side: str, size: float
    ) -> float:
        if hasattr(self._ex, "place_market_order"):
            return await self._ex.place_market_order(symbol, side, size)
        raise NotImplementedError("Exchange must implement place_market_order")

    async def cancel(self, symbol: str, order_id: str) -> bool:
        if hasattr(self._ex, "cancel_order"):
            return await self._ex.cancel_order(symbol, order_id)
        return False

    async def get_fill(self, order_id: str) -> Optional[float]:
        """Return fill price if order is filled, else None."""
        if hasattr(self._ex, "get_order_fill"):
            return await self._ex.get_order_fill(order_id)
        return None


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------

class InventoryUnwindScheduler:
    """
    End-of-session TWAP inventory unwind manager.

    Parameters
    ----------
    exchange : Any
        Exchange connector object (see _ExchangeInterface for expected API).
    unwind_threshold : float
        Minimum absolute position size that triggers unwind scheduling.
        Default 0.01 (units of the base asset, e.g. BTC).
    session_end_utc_hour : int
        The UTC hour at which the trading session ends. Default 22 (10 pm UTC).
    limit_fill_timeout_s : float
        Seconds to wait for a limit order to fill before cancelling and
        falling back to a market order. Default 60.
    """

    UNWIND_TRIGGER_SECONDS_BEFORE_END: float = 1800.0  # 30 min
    SLICE_INTERVAL_SECONDS: float = 120.0  # target 1 slice per 2 min
    MIN_SLICES: int = 5

    def __init__(
        self,
        exchange: Any,
        unwind_threshold: float = 0.01,
        session_end_utc_hour: int = 22,
        limit_fill_timeout_s: float = 60.0,
    ) -> None:
        self._ex = _ExchangeInterface(exchange)
        self._unwind_threshold = abs(unwind_threshold)
        self._session_end_utc_hour = session_end_utc_hour % 24
        self._limit_fill_timeout_s = limit_fill_timeout_s

        self._lock = threading.RLock()
        self._active: Dict[str, UnwindState] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

        # Stats
        self._completed_unwinds: int = 0
        self._total_slippage_bps: float = 0.0
        self._positions_closed_clean: int = 0

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def _seconds_to_session_end(self, timestamp_ns: int) -> float:
        """Return seconds between *timestamp_ns* and the next session-end wall clock."""
        t = time.gmtime(timestamp_ns // 1_000_000_000)
        current_hour_frac = t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0
        end_hour = self._session_end_utc_hour
        if current_hour_frac < end_hour:
            remaining_h = end_hour - current_hour_frac
        else:
            # session end is tomorrow
            remaining_h = (24.0 - current_hour_frac) + end_hour
        return remaining_h * 3600.0

    def _num_slices(self, seconds_remaining: float) -> int:
        """Compute N slices: max(MIN_SLICES, floor(seconds / SLICE_INTERVAL))."""
        return max(
            self.MIN_SLICES,
            int(seconds_remaining / self.SLICE_INTERVAL_SECONDS),
        )

    # ------------------------------------------------------------------
    # Schedule check
    # ------------------------------------------------------------------

    def check_and_schedule(
        self,
        symbol: str,
        net_position: float,
        avg_cost: float,
        current_mid: float,
        timestamp_ns: int,
    ) -> bool:
        """Evaluate whether an unwind should be triggered.

        Returns True if an unwind was newly scheduled, False otherwise.

        Conditions
        ----------
        1. |net_position| > unwind_threshold
        2. time_to_session_end < UNWIND_TRIGGER_SECONDS_BEFORE_END
        3. No unwind already active for this symbol
        """
        if abs(net_position) <= self._unwind_threshold:
            return False

        seconds_remaining = self._seconds_to_session_end(timestamp_ns)
        if seconds_remaining >= self.UNWIND_TRIGGER_SECONDS_BEFORE_END:
            return False

        with self._lock:
            if symbol in self._active:
                existing = self._active[symbol]
                if existing.status in (UnwindStatus.RUNNING, UnwindStatus.PENDING):
                    return False  # already in progress

        # Schedule
        n_slices = self._num_slices(seconds_remaining)
        state = UnwindState(
            symbol=symbol,
            original_size=net_position,
            remaining_size=abs(net_position),
            slices_total=n_slices,
            start_time_ns=timestamp_ns,
            status=UnwindStatus.PENDING,
        )
        with self._lock:
            self._active[symbol] = state

        logger.info(
            "Scheduled unwind for %s: size=%.6f slices=%d seconds_remaining=%.0f",
            symbol,
            net_position,
            n_slices,
            seconds_remaining,
        )
        return True

    # ------------------------------------------------------------------
    # Unwind execution
    # ------------------------------------------------------------------

    async def execute_unwind(
        self,
        symbol: str,
        net_position: float,
        target_price_tolerance_bps: float = 10.0,
    ) -> UnwindResult:
        """Execute a TWAP unwind of *net_position* for *symbol*.

        Algorithm
        ---------
        1. Determine unwind side (sell long, buy short).
        2. Split total quantity into N slices.
        3. For each slice:
           a. Fetch current mid price.
           b. Place limit order at mid ± tolerance.
           c. Wait up to limit_fill_timeout_s.
           d. If filled → record fill.
           e. If unfilled → cancel limit; place market order.
        4. Accumulate fills; adjust remaining.
        5. Stop early if position reaches zero.

        Returns an UnwindResult with summary statistics.
        """
        with self._lock:
            state = self._active.get(symbol)
            if state is None:
                state = UnwindState(
                    symbol=symbol,
                    original_size=net_position,
                    remaining_size=abs(net_position),
                    start_time_ns=time.time_ns(),
                )
                self._active[symbol] = state

            state.status = UnwindStatus.RUNNING
            state.start_time_ns = time.time_ns()

        side = "sell" if net_position > 0 else "buy"
        total_qty = abs(net_position)
        seconds_remaining = self._seconds_to_session_end(state.start_time_ns)
        n_slices = self._num_slices(seconds_remaining)

        with self._lock:
            state.slices_total = n_slices

        slice_qty = total_qty / n_slices
        tol_factor = target_price_tolerance_bps / 10_000.0

        total_filled = 0.0
        fill_prices: List[float] = []
        slices_used = 0

        try:
            initial_mid = await self._ex.get_mid(symbol)
        except Exception as exc:
            logger.error("Could not fetch initial mid for %s: %s", symbol, exc)
            initial_mid = 0.0

        start_wall = time.monotonic()

        for i in range(n_slices):
            with self._lock:
                # Check for cancellation
                if state.status == UnwindStatus.CANCELLED:
                    break
                remaining = state.remaining_size

            if remaining <= 1e-12:
                break

            # Size this slice (last slice takes whatever is left)
            this_slice = min(slice_qty, remaining)

            slices_used += 1

            try:
                mid = await self._ex.get_mid(symbol)
            except Exception as exc:
                logger.warning("Mid fetch failed for %s slice %d: %s", symbol, i, exc)
                mid = initial_mid if initial_mid > 0 else 1.0

            # Limit price: add tolerance for buy, subtract for sell
            if side == "buy":
                limit_price = mid * (1 + tol_factor)
            else:
                limit_price = mid * (1 - tol_factor)

            fill_price: Optional[float] = None

            # --- Try limit order ---
            try:
                order_id = await self._ex.place_limit(symbol, side, this_slice, limit_price)
                fill_price = await self._wait_for_fill(order_id, symbol)
            except Exception as exc:
                logger.warning("Limit order failed for %s slice %d: %s", symbol, i, exc)
                fill_price = None

            # --- Fall back to market ---
            if fill_price is None:
                try:
                    fill_price = await self._ex.place_market(symbol, side, this_slice)
                except Exception as exc:
                    logger.error("Market order failed for %s slice %d: %s", symbol, i, exc)
                    with self._lock:
                        state.last_error = str(exc)
                    continue  # skip this slice; try next

            total_filled += this_slice
            fill_prices.append(fill_price)

            with self._lock:
                state.unwound_size += this_slice
                state.remaining_size = max(0.0, total_qty - state.unwound_size)
                state.slices_done += 1
                # Running average fill price
                if fill_prices:
                    state.avg_fill_price = sum(fill_prices) / len(fill_prices)

            # Early exit: don't overshoot
            with self._lock:
                if state.remaining_size <= 1e-12:
                    break

            # Pace slices: wait between slices if there is time
            if i < n_slices - 1:
                slice_delay = seconds_remaining / n_slices
                await asyncio.sleep(min(slice_delay, self.SLICE_INTERVAL_SECONDS))

        # Compute result stats
        duration_s = time.monotonic() - start_wall
        avg_price = (
            sum(fill_prices) / len(fill_prices) if fill_prices else 0.0
        )
        if initial_mid > 0 and avg_price > 0:
            slippage_bps = abs(avg_price - initial_mid) / initial_mid * 10_000.0
        else:
            slippage_bps = 0.0

        final_status = (
            UnwindStatus.COMPLETED
            if total_filled >= total_qty * 0.99
            else state.status
        )
        if final_status not in (UnwindStatus.CANCELLED, UnwindStatus.FAILED):
            final_status = UnwindStatus.COMPLETED

        with self._lock:
            state.status = final_status
            state.avg_fill_price = avg_price

            # Update engine-level stats
            if final_status == UnwindStatus.COMPLETED:
                self._completed_unwinds += 1
                self._total_slippage_bps += slippage_bps
                if state.remaining_size <= 1e-12:
                    self._positions_closed_clean += 1

        result = UnwindResult(
            symbol=symbol,
            total_filled=total_filled,
            avg_price=avg_price,
            slippage_bps=slippage_bps,
            slices_used=slices_used,
            duration_s=duration_s,
            status=final_status,
        )
        logger.info(
            "Unwind completed for %s: filled=%.6f avg_price=%.4f slippage=%.2f bps",
            symbol,
            total_filled,
            avg_price,
            slippage_bps,
        )
        return result

    async def _wait_for_fill(
        self, order_id: str, symbol: str
    ) -> Optional[float]:
        """Poll for fill up to limit_fill_timeout_s; cancel and return None if unfilled."""
        deadline = time.monotonic() + self._limit_fill_timeout_s
        poll_interval = 1.0  # seconds between checks

        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
            try:
                fill_price = await self._ex.get_fill(order_id)
                if fill_price is not None:
                    return fill_price
            except Exception as exc:
                logger.debug("Fill check failed for %s: %s", order_id, exc)

        # Timeout reached — cancel
        try:
            await self._ex.cancel(symbol, order_id)
        except Exception as exc:
            logger.debug("Cancel failed for %s: %s", order_id, exc)

        return None

    # ------------------------------------------------------------------
    # Task launcher
    # ------------------------------------------------------------------

    def launch_unwind_task(
        self,
        symbol: str,
        net_position: float,
        target_price_tolerance_bps: float = 10.0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Optional[asyncio.Task]:
        """Launch execute_unwind as a fire-and-forget asyncio task.

        Returns the Task object so callers can await it if desired.
        Uses the provided loop, or the running loop if None.
        """
        try:
            if loop is None:
                loop = asyncio.get_running_loop()
            task = loop.create_task(
                self.execute_unwind(symbol, net_position, target_price_tolerance_bps)
            )
            with self._lock:
                self._tasks[symbol] = task
            return task
        except RuntimeError:
            logger.warning("No running event loop; cannot launch unwind task for %s", symbol)
            return None

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_active_unwinds(self) -> List[UnwindState]:
        """Return a list of all UnwindState objects with PENDING or RUNNING status."""
        with self._lock:
            return [
                s for s in self._active.values()
                if s.status in (UnwindStatus.PENDING, UnwindStatus.RUNNING)
            ]

    def cancel_unwind(self, symbol: str) -> bool:
        """Request cancellation of an active unwind.

        The running coroutine checks this flag at each slice boundary.
        Returns True if a running/pending unwind was found and flagged.
        """
        with self._lock:
            state = self._active.get(symbol)
            if state is None:
                return False
            if state.status not in (UnwindStatus.PENDING, UnwindStatus.RUNNING):
                return False
            state.status = UnwindStatus.CANCELLED
            # Also cancel the asyncio task if tracked
            task = self._tasks.get(symbol)

        if task is not None and not task.done():
            task.cancel()

        logger.info("Unwind cancellation requested for %s", symbol)
        return True

    def get_stats(self) -> dict:
        """Return aggregate statistics for all unwinds this session."""
        with self._lock:
            completed = self._completed_unwinds
            total_slippage = self._total_slippage_bps
            positions_clean = self._positions_closed_clean
            active_count = sum(
                1
                for s in self._active.values()
                if s.status in (UnwindStatus.PENDING, UnwindStatus.RUNNING)
            )

        avg_slippage = (
            total_slippage / completed if completed > 0 else 0.0
        )
        return {
            "completed_unwinds": completed,
            "avg_slippage_bps": avg_slippage,
            "positions_closed_clean": positions_clean,
            "active_unwinds": active_count,
            "total_unwinds_tracked": len(self._active),
        }

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    def get_unwind_state(self, symbol: str) -> Optional[UnwindState]:
        """Return the UnwindState for *symbol*, or None."""
        with self._lock:
            return self._active.get(symbol)

    def clear_completed(self) -> int:
        """Remove completed/cancelled/failed unwind records.

        Returns count removed.
        """
        with self._lock:
            done = [
                sym
                for sym, s in self._active.items()
                if s.status
                in (UnwindStatus.COMPLETED, UnwindStatus.CANCELLED, UnwindStatus.FAILED)
            ]
            for sym in done:
                del self._active[sym]
                self._tasks.pop(sym, None)
        return len(done)
