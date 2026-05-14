"""
feed_router.py
--------------
Central feed hub.

Responsibilities:
  1. Owns one feed instance per (venue, category).
  2. Receives canonical ticks/books/trades from all feeds.
  3. Deduplicates: drops stale messages (ts < last seen for same symbol+venue).
  4. Fans out to per-symbol subscriber queues (asyncio.Queue).
  5. Publishes summary stats: msg/s per venue, lag per symbol.

Usage::

    router = FeedRouter()
    router.add_feed(BybitFeed(symbols=["BTC/USDT"]))
    router.add_feed(BinanceFeed(symbols=["BTC/USDT"]))
    q = router.subscribe("tick", "BTC/USDT")
    await router.start()
    while True:
        tick = await q.get()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .ws_feed_base import WSFeedBase
from .feed_normaliser import CanonicalTick, CanonicalBook, CanonicalTrade

logger = logging.getLogger(__name__)

_CHANNEL_TICK  = "tick"
_CHANNEL_BOOK  = "book"
_CHANNEL_TRADE = "trade"


class FeedRouter:
    """
    Multi-venue feed hub with deduplication and fan-out.

    Parameters
    ----------
    queue_maxsize : int
        Max depth of each subscriber queue (default 1000; oldest dropped on full).
    dedup_window_ms : float
        Messages with ts within this window of last-seen are silently dropped.
    """

    def __init__(self, queue_maxsize: int = 1000, dedup_window_ms: float = 0.0) -> None:
        self._feeds: List[WSFeedBase] = []
        self._queue_maxsize = queue_maxsize
        self._dedup_window_ms = dedup_window_ms / 1000.0

        # (channel, symbol) → list of subscriber queues
        self._subs: Dict[Tuple[str, str], List[asyncio.Queue]] = defaultdict(list)

        # last seen ts per (venue, channel, symbol) for dedup
        self._last_ts: Dict[Tuple[str, str, str], float] = {}

        # stats
        self._msg_count: Dict[str, int] = defaultdict(int)   # venue → count
        self._started = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_feed(self, feed: WSFeedBase) -> None:
        """Register a feed. Wire up callbacks before starting."""
        self._wire_feed(feed)
        self._feeds.append(feed)

    def subscribe(self, channel: str, symbol: str) -> asyncio.Queue:
        """
        Return an asyncio.Queue that will receive canonical objects
        for (channel, symbol). channel = 'tick' | 'book' | 'trade'.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subs[(channel, symbol)].append(q)
        return q

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all registered feeds."""
        for feed in self._feeds:
            await feed.start()
        self._started = True
        logger.info("FeedRouter started with %d feeds", len(self._feeds))

    async def stop(self) -> None:
        for feed in self._feeds:
            await feed.stop()
        self._started = False
        logger.info("FeedRouter stopped")

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire_feed(self, feed: WSFeedBase) -> None:
        """Monkey-patch on_tick/on_book/on_trade onto the feed."""
        router = self

        async def _on_tick(tick: CanonicalTick) -> None:
            await router._dispatch(_CHANNEL_TICK, tick.symbol, tick.venue, tick.ts, tick)

        async def _on_book(book: CanonicalBook) -> None:
            await router._dispatch(_CHANNEL_BOOK, book.symbol, book.venue, book.ts, book)

        async def _on_trade(trade: CanonicalTrade) -> None:
            # Trades: don't dedup (each trade is unique)
            await router._fan_out(_CHANNEL_TRADE, trade.symbol, trade)

        # Only wire if the feed has these attributes
        if hasattr(feed, "_on_tick"):
            feed._on_tick = _on_tick
        if hasattr(feed, "_on_book"):
            feed._on_book = _on_book
        if hasattr(feed, "_on_trade"):
            feed._on_trade = _on_trade

    async def _dispatch(
        self,
        channel: str,
        symbol: str,
        venue: str,
        ts: float,
        obj: Any,
    ) -> None:
        key = (venue, channel, symbol)
        last = self._last_ts.get(key, 0.0)
        if ts < last + self._dedup_window_ms:
            return   # stale / duplicate
        self._last_ts[key] = ts
        self._msg_count[venue] += 1
        await self._fan_out(channel, symbol, obj)

    async def _fan_out(self, channel: str, symbol: str, obj: Any) -> None:
        queues = self._subs.get((channel, symbol), [])
        for q in queues:
            if q.full():
                try:
                    q.get_nowait()   # drop oldest
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(obj)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "feeds": len(self._feeds),
            "msg_count": dict(self._msg_count),
            "feed_states": {f.venue: f.state.name for f in self._feeds},
        }
