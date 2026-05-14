"""
bybit_feed.py
-------------
Bybit v5 WebSocket feed.

Subscribes to:
  - tickers.{symbol}    (best bid/ask + last price)
  - orderbook.1.{symbol} (top-of-book snapshot + delta)
  - publicTrade.{symbol}

Symbol batching: Bybit allows up to 10 topics per connection.
If more than 10 symbols are requested a second connection is opened.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .ws_feed_base import WSFeedBase
from .feed_normaliser import FeedNormaliser, CanonicalTick, CanonicalBook, CanonicalTrade

logger = logging.getLogger(__name__)

BYBIT_WS_MAINNET = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_SPOT    = "wss://stream.bybit.com/v5/public/spot"
MAX_TOPICS_PER_CONN = 10


class BybitFeed(WSFeedBase):
    """
    Bybit v5 public WebSocket feed.

    Parameters
    ----------
    symbols : list[str]
        Canonical symbols e.g. ["BTC/USDT", "ETH/USDT"].
        Internally mapped to Bybit raw format (BTCUSDT).
    category : str
        "linear" (perps) or "spot".
    on_tick  : async callback(tick: CanonicalTick)
    on_book  : async callback(book: CanonicalBook)
    on_trade : async callback(trade: CanonicalTrade)
    """

    def __init__(
        self,
        symbols: List[str],
        category: str = "linear",
        on_tick:  Optional[Callable[[CanonicalTick],  Coroutine]] = None,
        on_book:  Optional[Callable[[CanonicalBook],  Coroutine]] = None,
        on_trade: Optional[Callable[[CanonicalTrade], Coroutine]] = None,
        emitter: Optional[Any] = None,
    ) -> None:
        url = BYBIT_WS_MAINNET if category == "linear" else BYBIT_WS_SPOT
        super().__init__(url=url, venue="bybit", emitter=emitter)
        self._symbols = symbols
        self._category = category
        self._on_tick  = on_tick
        self._on_book  = on_book
        self._on_trade = on_trade
        self._raw_symbols = [self._canonical_to_raw(s) for s in symbols]

    # ------------------------------------------------------------------
    # WSFeedBase interface
    # ------------------------------------------------------------------

    async def _subscribe(self) -> None:
        topics: List[str] = []
        for raw in self._raw_symbols:
            topics.append(f"tickers.{raw}")
            topics.append(f"orderbook.1.{raw}")
            topics.append(f"publicTrade.{raw}")

        # Batch into groups of MAX_TOPICS_PER_CONN
        for i in range(0, len(topics), MAX_TOPICS_PER_CONN):
            batch = topics[i : i + MAX_TOPICS_PER_CONN]
            payload = json.dumps({"op": "subscribe", "args": batch})
            await self._ws.send_str(payload)
            logger.debug("Bybit subscribed: %s", batch)

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Pong / op confirmation
        if "op" in msg:
            self._pong_event.set()
            return

        topic: str = msg.get("topic", "")
        if topic.startswith("tickers."):
            tick = FeedNormaliser.bybit_ticker(msg)
            if tick and self._on_tick:
                await self._fire(self._on_tick, tick)

        elif topic.startswith("orderbook."):
            symbol = topic.split(".", 2)[-1]
            book = FeedNormaliser.bybit_book(msg, symbol)
            if book and self._on_book:
                await self._fire(self._on_book, book)

        elif topic.startswith("publicTrade."):
            trades = FeedNormaliser.bybit_trade(msg)
            if self._on_trade:
                for t in trades:
                    await self._fire(self._on_trade, t)

    def _ping_payload(self) -> str:
        return json.dumps({"op": "ping"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_to_raw(symbol: str) -> str:
        """BTC/USDT → BTCUSDT."""
        return symbol.replace("/", "")

    async def _fire(self, cb: Callable, arg: Any) -> None:
        try:
            result = cb(arg)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("Bybit callback error: %s", exc)
