"""
MEXCWSFeed — MEXC Spot WebSocket L2 Order Book + Trade Feed
===========================================================

Streams and maintains a local Level-2 order book and trade feed for one or
more MEXC spot symbols.  Compatible with the ``core/ws_l2_book_feed.py``
pattern used by other Argus exchange feeds.

MEXC depth subscription channel format:
    spot@public.limit.depth.v3.api@{SYMBOL}@{DEPTH}

MEXC trade subscription channel format:
    spot@public.deals.v3.api@{SYMBOL}

Snapshot / delta message format (depth):
    {
        "c": "spot@public.limit.depth.v3.api@BTCUSDT@20",
        "d": {
            "bids": [["price", "qty"], ...],
            "asks": [["price", "qty"], ...],
            "e": <event_time_ms>
        },
        "t": <timestamp_ms>
    }
    The first message after subscription is a full snapshot; subsequent
    messages are incremental deltas where qty=0 means remove the level.

Trade message format:
    {
        "c": "spot@public.deals.v3.api@BTCUSDT",
        "d": {
            "deals": [
                {"S": 1, "p": "29000.00", "v": "0.001", "t": <ts_ms>}
            ]
        },
        "t": <timestamp_ms>
    }
    S=1 → buy taker, S=2 → sell taker.

Usage::

    async def on_book(symbol, bids, asks, ts_ns):
        best_bid = max(bids, key=lambda x: x[0])[0] if bids else None
        print(symbol, best_bid)

    feed = MEXCWSFeed(
        symbols=["BTCUSDT", "ETHUSDT"],
        on_book_update=on_book,
    )
    await feed.start()
    ...
    await feed.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import aiohttp

from exchanges.mexc_client import MEXC_SPOT_WS_URL

log = logging.getLogger("argus.mexc_ws_feed")

# Reconnection back-off parameters
_BACKOFF_INIT: float = 1.0
_BACKOFF_MAX: float = 30.0
_BACKOFF_MUL: float = 2.0

# Default depth level
_DEFAULT_DEPTH: int = 20


# ---------------------------------------------------------------------------
# Per-symbol order book state (mirrors core/ws_l2_book_feed.OrderBook)
# ---------------------------------------------------------------------------

@dataclass
class _OrderBook:
    """Maintains a sorted L2 order book for one MEXC symbol."""

    symbol: str
    bids: Dict[float, float] = field(default_factory=dict)   # price → size
    asks: Dict[float, float] = field(default_factory=dict)   # price → size
    last_seq: int = -1
    last_update_ns: int = 0
    is_snapshot: bool = False

    # -- Accessors -----------------------------------------------------------

    def get_best_bid(self) -> Optional[float]:
        """Return highest bid price, or None if empty."""
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Return lowest ask price, or None if empty."""
        return min(self.asks.keys()) if self.asks else None

    def get_mid(self) -> Optional[float]:
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    def get_spread_bps(self) -> Optional[float]:
        """Return bid-ask spread in basis points, or None."""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is not None and ba is not None and bb > 0:
            return (ba - bb) / bb * 10_000.0
        return None

    def get_snapshot(self, levels: int = 20) -> Dict[str, List[List[float]]]:
        """Return top-of-book as {bids: [[p,q],...], asks: [[p,q],...]}."""
        sorted_bids = sorted(self.bids.items(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.items())[:levels]
        return {
            "bids": [[p, s] for p, s in sorted_bids],
            "asks": [[p, s] for p, s in sorted_asks],
        }

    # -- Mutation ------------------------------------------------------------

    def apply_bid_delta(self, price: float, size: float) -> None:
        if size == 0.0:
            self.bids.pop(price, None)
        else:
            self.bids[price] = size

    def apply_ask_delta(self, price: float, size: float) -> None:
        if size == 0.0:
            self.asks.pop(price, None)
        else:
            self.asks[price] = size

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.last_seq = -1
        self.is_snapshot = False


# ---------------------------------------------------------------------------
# MEXCWSFeed
# ---------------------------------------------------------------------------

BookUpdateCallback = Callable[
    [str, Dict[float, float], Dict[float, float], int],  # symbol, bids, asks, ts_ns
    None,
]

TradeCallback = Callable[
    [str, str, float, float, int],  # symbol, side, size, price, ts_ns
    None,
]


class MEXCWSFeed:
    """
    MEXC spot WebSocket L2 order book + trade feed.

    Maintains a local order book per symbol.  Emits ``on_book_update`` and
    ``on_trade`` callbacks on each message.

    Parameters
    ----------
    symbols:
        List of MEXC spot symbols, e.g. ``["BTCUSDT", "ETHUSDT"]``.
    on_book_update:
        Called with ``(symbol, bids_dict, asks_dict, timestamp_ns)``
        on each depth update.
    on_trade:
        Called with ``(symbol, side, size, price, timestamp_ns)``
        on each trade tick.
    depth:
        Subscription depth (5, 10, or 20).
    """

    def __init__(
        self,
        symbols: List[str],
        on_book_update: Optional[BookUpdateCallback] = None,
        on_trade: Optional[TradeCallback] = None,
        depth: int = _DEFAULT_DEPTH,
    ) -> None:
        self.symbols = list(symbols)
        self._on_book_update = on_book_update
        self._on_trade = on_trade
        self.depth = depth

        # Per-symbol state
        self._books: Dict[str, _OrderBook] = {
            s: _OrderBook(symbol=s) for s in self.symbols
        }

        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

        log.info(
            "MEXCWSFeed init: symbols=%s depth=%d",
            self.symbols, self.depth,
        )

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Connect to MEXC spot WebSocket and start streaming.

        Launches the connection loop as a background task.
        """
        if self._running:
            log.warning("MEXCWSFeed.start() called while already running")
            return
        self._running = True
        self._task = asyncio.create_task(
            self._connect_loop(), name="mexc_ws_feed"
        )
        log.info("MEXCWSFeed started for %s", self.symbols)

    async def stop(self) -> None:
        """Gracefully stop the feed and close connections."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        log.info("MEXCWSFeed stopped")

    async def __aenter__(self) -> "MEXCWSFeed":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_best_bid(self, symbol: str) -> Optional[float]:
        """Return best bid price for *symbol*, or None."""
        book = self._books.get(symbol)
        return book.get_best_bid() if book else None

    def get_best_ask(self, symbol: str) -> Optional[float]:
        """Return best ask price for *symbol*, or None."""
        book = self._books.get(symbol)
        return book.get_best_ask() if book else None

    def get_spread_bps(self, symbol: str) -> Optional[float]:
        """Return bid-ask spread in basis points for *symbol*, or None."""
        book = self._books.get(symbol)
        return book.get_spread_bps() if book else None

    def get_mid(self, symbol: str) -> Optional[float]:
        """Return mid-price for *symbol*, or None."""
        book = self._books.get(symbol)
        return book.get_mid() if book else None

    def get_book_snapshot(
        self, symbol: str, levels: int = 20
    ) -> Dict[str, List[List[float]]]:
        """Return top-of-book snapshot as {bids: [...], asks: [...]}."""
        book = self._books.get(symbol)
        if book is None:
            return {"bids": [], "asks": []}
        return book.get_snapshot(levels=levels)

    def is_ready(self, symbol: str) -> bool:
        """Return True if at least one snapshot has been received."""
        book = self._books.get(symbol)
        return (
            book is not None
            and book.is_snapshot
            and bool(book.bids)
            and bool(book.asks)
        )

    def register_book_callback(self, callback: BookUpdateCallback) -> None:
        """Register (or replace) the book-update callback."""
        self._on_book_update = callback

    def register_trade_callback(self, callback: TradeCallback) -> None:
        """Register (or replace) the trade callback."""
        self._on_trade = callback

    @property
    def books(self) -> Dict[str, _OrderBook]:
        """Direct access to the underlying _OrderBook objects."""
        return self._books

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _connect_loop(self) -> None:
        """Outer reconnect loop with exponential backoff."""
        backoff = _BACKOFF_INIT
        while self._running:
            try:
                await self._session_mexc()
                backoff = _BACKOFF_INIT  # clean disconnect resets backoff
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning(
                    "MEXCWSFeed: error — %s — reconnect in %.1fs", exc, backoff
                )
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * _BACKOFF_MUL, _BACKOFF_MAX)
            # Clear books on reconnect so next snapshot starts fresh
            for book in self._books.values():
                book.clear()

    # ------------------------------------------------------------------
    # MEXC session
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, sock_read=60.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _session_mexc(self) -> None:
        """Single WebSocket session: connect, subscribe, consume."""
        session = await self._get_session()

        # Build subscription list: depth + trades for each symbol
        depth_channels = [
            f"spot@public.limit.depth.v3.api@{sym}@{self.depth}"
            for sym in self.symbols
        ]
        trade_channels = [
            f"spot@public.deals.v3.api@{sym}"
            for sym in self.symbols
            if self._on_trade is not None
        ]
        all_channels = depth_channels + trade_channels

        sub_msg = json.dumps({
            "method": "SUBSCRIPTION",
            "params": all_channels,
        })

        log.info("MEXCWSFeed: connecting to %s", MEXC_SPOT_WS_URL)

        async with session.ws_connect(
            MEXC_SPOT_WS_URL,
            heartbeat=20.0,
            receive_timeout=60.0,
        ) as ws:
            await ws.send_str(sub_msg)
            log.info(
                "MEXCWSFeed: subscribed to %d channels for %s",
                len(all_channels), self.symbols,
            )

            async for msg in ws:
                if not self._running:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._parse_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # MEXC may send gzip-compressed frames
                    try:
                        import zlib
                        text = zlib.decompress(msg.data, -15).decode("utf-8")
                        self._parse_message(text)
                    except Exception:
                        pass
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    log.warning("MEXCWSFeed: WebSocket closed/error")
                    break

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def _parse_message(self, raw: str) -> None:
        """Dispatch incoming WebSocket message to the correct handler."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.debug("MEXCWSFeed: JSON error %s", exc)
            return

        if not isinstance(data, dict):
            return

        # Subscription confirmations / pong / errors
        if "id" in data or "msg" in data or "code" in data:
            return

        channel = data.get("c", "")

        if "limit.depth" in channel:
            self._parse_depth(data, channel)
        elif "deals" in channel:
            self._parse_trades(data, channel)

    def _parse_depth(self, data: dict, channel: str) -> None:
        """
        Process a depth update message.

        MEXC sends full snapshot on first message, then incremental deltas.
        A delta entry with qty=0.0 means that price level should be removed.
        """
        symbol = self._channel_symbol(channel)
        if symbol is None:
            return

        book = self._books.get(symbol)
        if book is None:
            return

        d = data.get("d", {})
        if not d:
            return

        ts_ms = data.get("t", int(time.time() * 1000))
        ts_ns = ts_ms * 1_000_000

        raw_bids = d.get("bids", [])
        raw_asks = d.get("asks", [])

        # Determine if this is a snapshot.
        # MEXC signals snapshot by including 'e' = 1000 in 'd'.
        event_type = d.get("e", 0)
        is_snapshot = (event_type == 1000) or (not book.is_snapshot)

        if is_snapshot:
            book.clear()
            for entry in raw_bids:
                price, qty = float(entry[0]), float(entry[1])
                if qty > 0:
                    book.bids[price] = qty
            for entry in raw_asks:
                price, qty = float(entry[0]), float(entry[1])
                if qty > 0:
                    book.asks[price] = qty
            book.is_snapshot = True
            book.last_update_ns = ts_ns
            log.debug(
                "MEXCWSFeed: snapshot %s bids=%d asks=%d",
                symbol, len(book.bids), len(book.asks),
            )
        else:
            if not book.is_snapshot:
                log.warning(
                    "MEXCWSFeed: delta before snapshot for %s, skipping", symbol
                )
                return
            for entry in raw_bids:
                book.apply_bid_delta(float(entry[0]), float(entry[1]))
            for entry in raw_asks:
                book.apply_ask_delta(float(entry[0]), float(entry[1]))
            book.last_update_ns = ts_ns

        self._fire_book_callback(symbol, book, ts_ns)

    def _parse_trades(self, data: dict, channel: str) -> None:
        """Process a trade tick message."""
        if self._on_trade is None:
            return

        symbol = self._channel_symbol(channel)
        if symbol is None:
            return

        d = data.get("d", {})
        deals = d.get("deals", [])

        for deal in deals:
            side = "BUY" if deal.get("S") == 1 else "SELL"
            price = float(deal.get("p", 0))
            size = float(deal.get("v", 0))
            ts_ms = deal.get("t", data.get("t", int(time.time() * 1000)))
            ts_ns = int(ts_ms) * 1_000_000
            self._fire_trade_callback(symbol, side, size, price, ts_ns)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _channel_symbol(self, channel: str) -> Optional[str]:
        """
        Extract the MEXC symbol from a channel string.

        e.g.  "spot@public.limit.depth.v3.api@BTCUSDT@20"  → "BTCUSDT"
              "spot@public.deals.v3.api@ETHUSDT"            → "ETHUSDT"
        """
        parts = channel.split("@")
        # Format: spot@<channel_name>@<SYMBOL>[@depth]
        if len(parts) >= 3:
            candidate = parts[2]
            if candidate in self._books:
                return candidate
        return None

    def _fire_book_callback(
        self, symbol: str, book: _OrderBook, ts_ns: int
    ) -> None:
        """Invoke the book update callback, catching all errors."""
        if self._on_book_update is None:
            return
        try:
            self._on_book_update(symbol, book.bids, book.asks, ts_ns)
        except Exception as exc:
            log.debug("MEXCWSFeed: book callback error: %s", exc)

    def _fire_trade_callback(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        ts_ns: int,
    ) -> None:
        """Invoke the trade callback, catching all errors."""
        if self._on_trade is None:
            return
        try:
            self._on_trade(symbol, side, size, price, ts_ns)
        except Exception as exc:
            log.debug("MEXCWSFeed: trade callback error: %s", exc)


# ---------------------------------------------------------------------------
# Factory convenience function
# ---------------------------------------------------------------------------

def create_mexc_feed(
    symbols: List[str],
    on_book_update: Optional[BookUpdateCallback] = None,
    on_trade: Optional[TradeCallback] = None,
    depth: int = _DEFAULT_DEPTH,
) -> MEXCWSFeed:
    """Factory helper to create a configured MEXCWSFeed."""
    return MEXCWSFeed(
        symbols=symbols,
        on_book_update=on_book_update,
        on_trade=on_trade,
        depth=depth,
    )
