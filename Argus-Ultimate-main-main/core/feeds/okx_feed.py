"""
okx_feed.py
-----------
OKX v5 public WebSocket feed.

Endpoint: wss://ws.okx.com:8443/ws/v5/public

Subscribes to:
  - tickers channel  (spot/swap best bid/ask/last)
  - books5 channel   (5-level order book snapshot, push on change)
  - trades channel   (trade stream)

All channels are login-less public endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .ws_feed_base import WSFeedBase
from .feed_normaliser import FeedNormaliser, CanonicalTick, CanonicalBook, CanonicalTrade

logger = logging.getLogger(__name__)

OKX_WS_PUBLIC = "wss://ws.okx.com:8443/ws/v5/public"


class OKXFeed(WSFeedBase):
    """
    OKX v5 public WebSocket feed.

    Parameters
    ----------
    symbols : list[str]
        Canonical symbols e.g. ["BTC/USDT"]. Mapped to OKX instId (BTC-USDT).
    inst_type : str
        "SPOT" | "SWAP" | "FUTURES".
    on_tick  : async callback(tick: CanonicalTick)
    on_book  : async callback(book: CanonicalBook)
    on_trade : async callback(trade: CanonicalTrade)
    """

    def __init__(
        self,
        symbols: List[str],
        inst_type: str = "SWAP",
        on_tick:  Optional[Callable[[CanonicalTick],  Coroutine]] = None,
        on_book:  Optional[Callable[[CanonicalBook],  Coroutine]] = None,
        on_trade: Optional[Callable[[CanonicalTrade], Coroutine]] = None,
        emitter: Optional[Any] = None,
    ) -> None:
        super().__init__(url=OKX_WS_PUBLIC, venue="okx", emitter=emitter)
        self._symbols = symbols
        self._inst_type = inst_type
        self._on_tick  = on_tick
        self._on_book  = on_book
        self._on_trade = on_trade
        # Map canonical → OKX instId
        suffix = "-SWAP" if inst_type == "SWAP" else ""
        self._inst_ids = [s.replace("/", "-") + suffix for s in symbols]

    # ------------------------------------------------------------------
    # WSFeedBase interface
    # ------------------------------------------------------------------

    async def _subscribe(self) -> None:
        args = []
        for inst_id in self._inst_ids:
            args.append({"channel": "tickers",  "instId": inst_id})
            args.append({"channel": "books5",   "instId": inst_id})
            args.append({"channel": "trades",   "instId": inst_id})
        payload = json.dumps({"op": "subscribe", "args": args})
        await self._ws.send_str(payload)
        logger.debug("OKX subscribed: %d instIds", len(self._inst_ids))

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Ping/pong or op confirmation
        if msg.get("event") in ("subscribe", "pong"):
            self._pong_event.set()
            return
        if msg == "pong":  # raw string pong
            self._pong_event.set()
            return

        channel = msg.get("arg", {}).get("channel", "")
        data_list: List[Dict] = msg.get("data", [])

        for item in data_list:
            if channel == "tickers":
                tick = FeedNormaliser.okx_ticker(item)
                if tick and self._on_tick:
                    await self._fire(self._on_tick, tick)

            elif channel == "books5":
                book = FeedNormaliser.okx_book(item)
                if book and self._on_book:
                    await self._fire(self._on_book, book)

            elif channel == "trades":
                trade = FeedNormaliser.okx_trade(item)
                if trade and self._on_trade:
                    await self._fire(self._on_trade, trade)

    def _ping_payload(self) -> str:
        return "ping"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fire(self, cb: Callable, arg: Any) -> None:
        try:
            result = cb(arg)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("OKX callback error: %s", exc)
