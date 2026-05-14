"""
feed_health_monitor.py
----------------------
Per-feed staleness watchdog.

Monitors each registered WSFeedBase.  If no message arrives within
`stale_threshold_s` seconds it:
  1. Logs a warning.
  2. Emits a 'feed.stale' event on the EventBus if one is provided.
  3. Calls feed.stop() then feed.start() to force reconnect.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .ws_feed_base import WSFeedBase, FeedState

logger = logging.getLogger(__name__)


class FeedHealthMonitor:
    """
    Parameters
    ----------
    stale_threshold_s : float
        Seconds without a message before a feed is declared stale (default 30).
    check_interval_s : float
        How often the monitor polls each feed (default 10).
    event_bus : optional
        Any object with an `emit(event_name, payload)` method.
    """

    def __init__(
        self,
        stale_threshold_s: float = 30.0,
        check_interval_s: float = 10.0,
        event_bus: Optional[Any] = None,
    ) -> None:
        self._threshold = stale_threshold_s
        self._interval  = check_interval_s
        self._event_bus = event_bus
        self._feeds: List[WSFeedBase] = []
        self._task: Optional[asyncio.Task] = None
        self._stale_counts: Dict[str, int] = {}

    def register(self, feed: WSFeedBase) -> None:
        self._feeds.append(feed)
        self._stale_counts[feed.venue] = 0

    async def start(self) -> None:
        self._task = asyncio.ensure_future(self._loop())
        logger.info("FeedHealthMonitor started (%d feeds, threshold=%.0fs)",
                    len(self._feeds), self._threshold)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            now = time.monotonic()
            for feed in self._feeds:
                if feed.state == FeedState.STOPPED:
                    continue
                last = feed.stats.last_message_ts
                if last == 0.0:
                    # Feed hasn't received any message yet; skip until first msg
                    continue
                age = now - last
                if age > self._threshold:
                    self._stale_counts[feed.venue] = self._stale_counts.get(feed.venue, 0) + 1
                    logger.warning(
                        "Feed '%s' stale for %.1f s (occurrence #%d) — forcing reconnect",
                        feed.venue, age, self._stale_counts[feed.venue]
                    )
                    self._emit_stale(feed.venue, age)
                    asyncio.ensure_future(self._reconnect(feed))
                else:
                    self._stale_counts[feed.venue] = 0   # reset on healthy

    async def _reconnect(self, feed: WSFeedBase) -> None:
        try:
            await feed.stop()
        except Exception:
            pass
        # Brief wait before restart to avoid tight loop on persistent failure
        await asyncio.sleep(2.0)
        try:
            feed.state = FeedState.DISCONNECTED
            await feed.start()
        except Exception as exc:
            logger.error("FeedHealthMonitor: failed to restart %s: %s", feed.venue, exc)

    def _emit_stale(self, venue: str, age: float) -> None:
        if self._event_bus is not None:
            try:
                self._event_bus.emit("feed.stale", {"venue": venue, "stale_age_s": age})
            except Exception:
                pass

    @property
    def stale_counts(self) -> Dict[str, int]:
        return dict(self._stale_counts)
