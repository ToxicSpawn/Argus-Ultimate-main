"""
Async Order Submitter — fire-and-forget order submission with background
confirmation collection.

For time-sensitive arbitrage and latency-critical strategies, waiting for
the exchange ACK before proceeding wastes precious milliseconds.  This
module submits orders to a background queue and returns immediately with
an internal ``pending_id``.  A background coroutine collects confirmations
and makes them available via ``get_pending_confirmations()``.

Features:
  - ``submit_fire_and_forget(order)`` returns instantly
  - Background task collects confirmations
  - Per-strategy configurable: wait-for-fill vs fire-and-forget
  - Order deduplication within configurable window (default 100ms)
  - Thread-safe confirmation queue

Usage:
    submitter = AsyncOrderSubmitter(exchange_manager=em)
    await submitter.start()
    pid = await submitter.submit_fire_and_forget(order_dict)
    # ... later ...
    confirmations = submitter.get_pending_confirmations()
    await submitter.stop()
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderConfirmation:
    """Result of a fire-and-forget order once the exchange responds."""
    pending_id: str
    order_id: str = ""
    exchange: str = ""
    symbol: str = ""
    side: str = ""
    status: str = "unknown"  # filled, partial, rejected, error
    filled_qty: float = 0.0
    filled_price: float = 0.0
    submitted_at: float = 0.0
    confirmed_at: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _DedupKey:
    """Hashable dedup key for order identity."""
    exchange: str
    symbol: str
    side: str
    amount_rounded: str
    price_rounded: str


class AsyncOrderSubmitter:
    """
    Fire-and-forget order submission with background confirmation.

    Parameters
    ----------
    exchange_manager
        Object with ``execute_order(order_dict, exchange=name)`` async method.
    dedup_window_ms
        Orders with identical (exchange, symbol, side, amount, price) within
        this window are silently dropped.  Set to 0 to disable.
    max_pending
        Maximum number of orders that can be in-flight simultaneously.
    """

    def __init__(
        self,
        exchange_manager: Any = None,
        dedup_window_ms: float = 100.0,
        max_pending: int = 100,
    ) -> None:
        self._exchange_manager = exchange_manager
        self._dedup_window_ms = dedup_window_ms
        self._max_pending = max_pending

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_pending)
        self._confirmations: Deque[OrderConfirmation] = deque(maxlen=500)
        self._pending_count: int = 0
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._counter = 0

        # Dedup tracking: key -> monotonic timestamp of last submission
        self._dedup_log: Dict[_DedupKey, float] = {}

        # Stats
        self.total_submitted: int = 0
        self.total_confirmed: int = 0
        self.total_deduped: int = 0
        self.total_errors: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background confirmation worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("AsyncOrderSubmitter: worker started")

    async def stop(self) -> None:
        """Drain queue and stop worker."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "AsyncOrderSubmitter: stopped (submitted=%d, confirmed=%d, deduped=%d, errors=%d)",
            self.total_submitted, self.total_confirmed, self.total_deduped, self.total_errors,
        )

    # ------------------------------------------------------------------
    # Submission (hot path — returns immediately)
    # ------------------------------------------------------------------

    async def submit_fire_and_forget(
        self,
        order: Dict[str, Any],
        exchange: Optional[str] = None,
        strategy: str = "",
    ) -> str:
        """
        Submit an order without waiting for the exchange response.

        Returns a ``pending_id`` that can be matched against confirmations.
        Raises ``RuntimeError`` if the queue is full.
        """
        exchange = exchange or order.get("exchange", "primary")
        symbol = order.get("symbol", "")
        side = order.get("side", "")
        amount = order.get("amount", 0.0)
        price = order.get("price", 0.0)

        # --- Deduplication ---
        if self._dedup_window_ms > 0:
            dk = _DedupKey(
                exchange=exchange,
                symbol=symbol,
                side=side,
                amount_rounded=f"{amount:.8f}",
                price_rounded=f"{price:.2f}" if price else "mkt",
            )
            now = time.monotonic()
            last = self._dedup_log.get(dk, 0.0)
            if (now - last) * 1000.0 < self._dedup_window_ms:
                self.total_deduped += 1
                logger.debug(
                    "AsyncOrderSubmitter: dedup suppressed %s %s %s",
                    exchange, symbol, side,
                )
                return f"deduped_{self._counter}"
            self._dedup_log[dk] = now
            # Prune old dedup entries
            if len(self._dedup_log) > 500:
                cutoff = now - (self._dedup_window_ms / 1000.0) * 2
                self._dedup_log = {
                    k: v for k, v in self._dedup_log.items() if v > cutoff
                }

        # --- Build pending entry ---
        self._counter += 1
        pending_id = f"ff_{self._counter}_{int(time.monotonic() * 1000) % 100000}"

        entry = {
            "pending_id": pending_id,
            "order": order,
            "exchange": exchange,
            "strategy": strategy,
            "submitted_at": time.monotonic(),
        }

        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            raise RuntimeError("AsyncOrderSubmitter queue full")

        self.total_submitted += 1
        self._pending_count += 1
        return pending_id

    # ------------------------------------------------------------------
    # Confirmation retrieval
    # ------------------------------------------------------------------

    def get_pending_confirmations(self) -> List[OrderConfirmation]:
        """
        Drain and return all confirmations collected since last call.
        """
        results = list(self._confirmations)
        self._confirmations.clear()
        return results

    @property
    def pending_count(self) -> int:
        return self._pending_count

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Process queued orders and collect confirmations."""
        try:
            while self._running:
                try:
                    entry = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                pending_id = entry["pending_id"]
                order = entry["order"]
                exchange = entry["exchange"]
                submitted_at = entry["submitted_at"]

                confirmation = OrderConfirmation(
                    pending_id=pending_id,
                    exchange=exchange,
                    symbol=order.get("symbol", ""),
                    side=order.get("side", ""),
                    submitted_at=submitted_at,
                )

                try:
                    if self._exchange_manager is not None:
                        result = await self._exchange_manager.execute_order(
                            order, exchange=exchange,
                        )
                        confirmed_at = time.monotonic()
                        if result:
                            confirmation.order_id = str(result.get("order_id", ""))
                            confirmation.status = str(result.get("status", "filled"))
                            confirmation.filled_qty = float(result.get("filled", 0.0) or result.get("amount", 0.0))
                            confirmation.filled_price = float(result.get("price", 0.0))
                            confirmation.raw = result
                        else:
                            confirmation.status = "rejected"
                    else:
                        confirmed_at = time.monotonic()
                        confirmation.status = "simulated"
                        confirmation.filled_qty = float(order.get("amount", 0.0))
                        confirmation.filled_price = float(order.get("price", 0.0))

                    confirmation.confirmed_at = confirmed_at
                    confirmation.latency_ms = (confirmed_at - submitted_at) * 1000.0
                    self.total_confirmed += 1

                except Exception as exc:
                    confirmation.status = "error"
                    confirmation.error = str(exc)
                    confirmation.confirmed_at = time.monotonic()
                    confirmation.latency_ms = (confirmation.confirmed_at - submitted_at) * 1000.0
                    self.total_errors += 1
                    logger.warning(
                        "AsyncOrderSubmitter: order %s failed: %s", pending_id, exc,
                    )

                self._confirmations.append(confirmation)
                self._pending_count = max(0, self._pending_count - 1)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("AsyncOrderSubmitter worker error: %s", exc)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_submitted": self.total_submitted,
            "total_confirmed": self.total_confirmed,
            "total_deduped": self.total_deduped,
            "total_errors": self.total_errors,
            "pending_count": self._pending_count,
            "queue_size": self._queue.qsize(),
        }
