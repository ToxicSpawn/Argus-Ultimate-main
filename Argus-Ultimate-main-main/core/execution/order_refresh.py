"""Push 67 — Order refresh loop (Hummingbot Rule 6).

Cancels and reposts all unfilled limit orders every
ORDER_REFRESH_SECS (default 30s) to prevent stale quotes
from being adversely selected in thin / fast markets.

Design:
  - OrderRefreshLoop runs as a background asyncio task
  - Calls exec_engine.get_unfilled_older_than(age_secs)
  - Cancels stale orders, reposts at current mid price
  - Emits refresh metrics to Prometheus counter
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, List, Optional


@dataclass
class RefreshStats:
    total_refreshes: int = 0
    total_orders_cancelled: int = 0
    total_orders_reposted: int = 0
    last_refresh_ts: float = field(default=0.0, repr=False)


class OrderRefreshLoop:
    """Async cancel+repost loop for stale limit orders.

    Args:
        refresh_secs:     Interval between refresh cycles (default 30s)
        stale_age_secs:   Age threshold to consider order stale
        get_stale_fn:     Async callable returning stale order list
        cancel_fn:        Async callable to cancel an order by id
        repost_fn:        Async callable to repost an order
        on_refresh_cb:    Optional callback after each refresh cycle
    """

    def __init__(
        self,
        refresh_secs: float = 30.0,
        stale_age_secs: float = 30.0,
        get_stale_fn: Optional[Callable] = None,
        cancel_fn: Optional[Callable] = None,
        repost_fn: Optional[Callable] = None,
        on_refresh_cb: Optional[Callable] = None,
    ):
        self.refresh_secs = refresh_secs
        self.stale_age_secs = stale_age_secs
        self._get_stale = get_stale_fn
        self._cancel = cancel_fn
        self._repost = repost_fn
        self._on_refresh = on_refresh_cb
        self.stats = RefreshStats()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the refresh loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Gracefully stop the refresh loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.refresh_secs)
            if not self._running:
                break
            await self._refresh_cycle()

    async def _refresh_cycle(self) -> None:
        self.stats.total_refreshes += 1
        self.stats.last_refresh_ts = time.time()

        cancelled_count = 0
        reposted_count = 0

        if self._get_stale:
            stale_orders = await self._get_stale(self.stale_age_secs)
            for order in stale_orders:
                if self._cancel:
                    try:
                        await self._cancel(order)
                        cancelled_count += 1
                    except Exception:
                        pass
                if self._repost:
                    try:
                        await self._repost(order)
                        reposted_count += 1
                    except Exception:
                        pass

        self.stats.total_orders_cancelled += cancelled_count
        self.stats.total_orders_reposted += reposted_count

        if self._on_refresh:
            try:
                await self._on_refresh(self.stats)
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._running and (self._task is not None) and (not self._task.done())
