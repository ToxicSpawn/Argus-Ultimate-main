"""
binance_feed.py
---------------
Binance combined stream feed.

Uses the combined stream endpoint:
  wss://stream.binance.com:9443/stream?streams=<s1>/<s2>/...

Subscribes to:
  - {symbol}@bookTicker   (best bid/ask, real-time)
  - {symbol}@aggTrade     (aggregated trade stream)

For a local order book, the caller should use DepthCache from ccxt or
implement snapshot + incremental diff — this feed provides raw diff events
via on_book callback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .ws_feed_base import WSFeedBase
from .feed_normaliser import FeedNormaliser, CanonicalTick, CanonicalBook, CanonicalTrade

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"


class BinanceFeed(WSFeedBase):
    """
    Binance combined-stream WebSocket feed.

    Parameters
    ----------
    symbols : list[str]
        Canonical symbols e.g. ["BTC/USDT", "ETH/USDT"].
    depth_levels : int
        5 | 10 | 20 — for partial book stream (bookTicker used for BBO).
    on_tick  : async callback(tick: CanonicalTick)
    on_book  : async callback(book: CanonicalBook)   (diff depth events)
    on_trade : async callback(trade: CanonicalTrade)
    """

    def __init__(
        self,
        symbols: List[str],
        depth_levels: int = 5,
        on_tick:  Optional[Callable[[CanonicalTick],  Coroutine]] = None,
        on_book:  Optional[Callable[[CanonicalBook],  Coroutine]] = None,
        on_trade: Optional[Callable[[CanonicalTrade], Coroutine]] = None,
        emitter: Optional[Any] = None,
    ) -> None:
        self._symbols = symbols
        self._raw_symbols = [s.replace("/", "").lower() for s in symbols]
        self._depth_levels = depth_levels
        self._on_tick  = on_tick
        self._on_book  = on_book
        self._on_trade = on_trade
        streams = self._build_streams()
        url = f"{BINANCE_WS_BASE}?streams={'/'.join(streams)}"
        super().__init__(url=url, venue="binance", emitter=emitter)

    def _build_streams(self) -> List[str]:
        streams = []
        for raw in self._raw_symbols:
            streams.append(f"{raw}@bookTicker")
            streams.append(f"{raw}@aggTrade")
            streams.append(f"{raw}@depth{self._depth_levels}@100ms")
        return streams

    # ------------------------------------------------------------------
    # WSFeedBase interface
    # ------------------------------------------------------------------

    async def _subscribe(self) -> None:
        # Binance combined stream: subscriptions are encoded in the URL.
        # No additional subscription frame needed.
        logger.debug("Binance combined stream active: %s symbols", len(self._symbols))

    async def _handle_message(self, raw: str) -> None:
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Combined stream wraps in {"stream": "...", "data": {...}}
        stream_name: str = outer.get("stream", "")
        data: Dict = outer.get("data", outer)

        event_type = data.get("e", "")
        stream_lower = stream_name.lower()

        if "bookticker" in stream_lower or event_type == "bookTicker":
            tick = FeedNormaliser.binance_ticker(data)
            if tick and self._on_tick:
                await self._fire(self._on_tick, tick)

        elif "aggtrade" in stream_lower or event_type == "aggTrade":
            trade = FeedNormaliser.binance_trade(data)
            if trade and self._on_trade:
                await self._fire(self._on_trade, trade)

        elif "depth" in stream_lower or event_type == "depthUpdate":
            # Derive symbol from stream name e.g. "btcusdt@depth5@100ms"
            raw_sym = stream_lower.split("@")[0].upper()
            book = FeedNormaliser.binance_book(data, raw_sym)
            if book and self._on_book:
                await self._fire(self._on_book, book)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fire(self, cb: Callable, arg: Any) -> None:
        try:
            result = cb(arg)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("Binance callback error: %s", exc)
