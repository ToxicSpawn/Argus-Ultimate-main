#!/usr/bin/env python3
"""
Async Write Queue — non-blocking background write queue for DB/file I/O.

Writers enqueue a callable (e.g. lambda: db.execute(...)); a single daemon
async worker drains the queue so the hot trading loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AsyncWriteQueue:
    """
    Async queue-backed write coalescer.

    Usage:
        q = AsyncWriteQueue(max_size=1000)
        asyncio.create_task(q.run())
        q.put(lambda: con.execute("INSERT …"))
    """

    def __init__(self, max_size: int = 2000, drain_interval: float = 0.05):
        self.max_size = max(1, int(max_size))
        self.drain_interval = max(0.001, float(drain_interval))
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_size)
        self._running = False
        self._written = 0
        self._dropped = 0
        self._last_error: Optional[Exception] = None

    def put(self, fn: Callable[[], Any]) -> bool:
        """Enqueue a write callable. Returns False if queue is full (dropped)."""
        try:
            self._queue.put_nowait(fn)
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            return False

    async def run(self) -> None:
        """Run the drain loop — create_task this in your asyncio loop."""
        self._running = True
        while self._running:
            try:
                fn = await asyncio.wait_for(self._queue.get(), timeout=self.drain_interval)
                try:
                    result = fn()
                    if asyncio.iscoroutine(result):
                        await result
                    self._written += 1
                except Exception as e:
                    self._last_error = e
                    logger.debug("AsyncWriteQueue fn error: %s", e)
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("AsyncWriteQueue loop error: %s", e)

    async def flush(self, timeout: float = 5.0) -> None:
        """Wait until queue is drained or timeout."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    def stop(self) -> None:
        self._running = False

    def stats(self) -> dict:
        return {
            "queued": self._queue.qsize(),
            "written": self._written,
            "dropped": self._dropped,
            "last_error": str(self._last_error) if self._last_error else None,
        }
