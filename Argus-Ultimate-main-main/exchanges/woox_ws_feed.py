"""
WOOXWSFeed — WOO X WebSocket L2 Order Book + Trade Feed
=========================================================

Streams and maintains a local Level-2 order book and trade feed for one or
more WOO X symbols.  Compatible with the MEXCWSFeed pattern used by other
Argus exchange feeds.

WOO X depth subscription format:
    {"event": "subscribe", "topic": "SPOT_BTC_USDT@orderbook"}

WOO X book update format:
    {
        "topic": "SPOT_BTC_USDT@orderbook",
        "ts": 1234567890123,
        "data": {
            "bids": [[price, qty, count], ...],
            "asks": [[price, qty, count], ...],
            "ts": 1234567890123,
            "checksum": 12345678  (CRC32 for integrity verification)
        }
    }

WOO X trade subscription format:
    {"event": "subscribe", "topic": "SPOT_BTC_USDT@trade"}

WOO X trade update format:
    {
        "topic": "SPOT_BTC_USDT@trade",
        "ts": 1234567890123,
        "data": {
            "symbol": "SPOT_BTC_USDT",
            "price": 29000.00,
            "size": 0.001,
            "side": "BUY",
            "source": 0,
            "ts": 1234567890123
        }
    }

The first orderbook message after subscription is a full snapshot; subsequent
messages are incremental deltas where qty=0 means remove the level.

Usage::

    async def on_book(symbol, bids, asks, ts_ns):
        best_bid = max(bids, key=lambda x: x[0])[0] if bids else None
        print(symbol, best_bid)

    feed = WOOXWSFeed(
        symbols=["SPOT_BTC_USDT", "SPOT_ETH_USDT"],
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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import aiohttp

from exchanges.woox_client import WOOX_WS_PUBLIC, to_woox_symbol, from_woox_symbol

log = logging.getLogger("argus.woox_ws_feed")

# Reconnection back-off parameters
_BACKOFF_INIT: float = 1.0
_BACKOFF_MAX: float = 30.0
_BACKOFF_MUL: float = 2.0

# WOO X heartbeat interval (send ping every 30s)
_HEARTBEAT_INTERVAL: float = 30.0


# ---------------------------------------------------------------------------
# Per-symbol order book state (mirrors MEXCWSFeed._OrderBook pattern)
# ---------------------------------------------------------------------------

@dataclass
class _OrderBook:
    """Maintains a sorted L2 order book for one WOO X symbol."""

    symbol: str
    bids: Dict[float, float] = field(default_factory=dict)   # price → size
    asks: Dict[float, float] = field(default_factory=dict)   # price → size
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
        """Return mid-price, or None."""
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
        self.is_snapshot = False


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

BookUpdateCallback = Callable[
    [str, Dict[float, float], Dict[float, float], int],  # symbol, bids, asks, ts_ns
    None,
]

TradeCallback = Callable[
    [str, str, float, float, int],  # symbol, side, size, price, ts_ns
    None,
]


# ---------------------------------------------------------------------------
# WOOXWSFeed
# ---------------------------------------------------------------------------

class WOOXWSFeed:
    """
    WOO X WebSocket L2 order book + trade feed.

    Maintains a local order book per symbol.  Emits ``on_book_update`` and
    ``on_trade`` callbacks on each message.  Automatically reconnects on
    disconnect using exponential backoff.

    Parameters
    ----------
    symbols:
        List of WOO X symbols, e.g. ``["SPOT_BTC_USDT", "SPOT_ETH_USDT"]``.
        Symbols in slash-format are auto-converted: "BTC/USDT" → "SPOT_BTC_USDT".
    on_book_update:
        Called with ``(symbol, bids_dict, asks_dict, timestamp_ns)``
        on each depth update.
    on_trade:
        Called with ``(symbol, side, size, price, timestamp_ns)``
        on each trade tick.
    testnet:
        If True, connect to WOO X staging WebSocket.
    """

    def __init__(
        self,
        symbols: List[str],
        on_book_update: Optional[BookUpdateCallback] = None,
        on_trade: Optional[TradeCallback] = None,
        testnet: bool = False,
    ) -> None:
        # Normalise all symbols to WOO X SPOT format
        self.symbols: List[str] = [
            to_woox_symbol(s) if "/" in s else s.upper()
            for s in symbols
        ]
        self._on_book_update = on_book_update
        self._on_trade = on_trade

        self._ws_url = (
            "wss://wss.staging.woo.org/v2/ws/public" if testnet else WOOX_WS_PUBLIC
        )

        # Per-symbol state
        self._books: Dict[str, _OrderBook] = {
            s: _OrderBook(symbol=s) for s in self.symbols
        }

        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

        log.info(
            "WOOXWSFeed init: symbols=%s url=%s",
            self.symbols, self._ws_url,
        )

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Connect to WOO X WebSocket and start streaming.

        Launches the connection loop as a background task.
        """
        if self._running:
            log.warning("WOOXWSFeed.start() called while already running")
            return
        self._running = True
        self._task = asyncio.create_task(
            self._connect_loop(), name="woox_ws_feed"
        )
        log.info("WOOXWSFeed started for %s", self.symbols)

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
        log.info("WOOXWSFeed stopped")

    async def __aenter__(self) -> "WOOXWSFeed":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_best_bid(self, symbol: str) -> Optional[float]:
        """Return best bid price for *symbol*, or None."""
        # Accept both slash and WOO X format
        woox_sym = to_woox_symbol(symbol) if "/" in symbol else symbol.upper()
        book = self._books.get(woox_sym)
        return book.get_best_bid() if book else None

    def get_best_ask(self, symbol: str) -> Optional[float]:
        """Return best ask price for *symbol*, or None."""
        woox_sym = to_woox_symbol(symbol) if "/" in symbol else symbol.upper()
        book = self._books.get(woox_sym)
        return book.get_best_ask() if book else None

    def get_spread_bps(self, symbol: str) -> Optional[float]:
        """Return bid-ask spread in basis points for *symbol*, or None."""
        woox_sym = to_woox_symbol(symbol) if "/" in symbol else symbol.upper()
        book = self._books.get(woox_sym)
        return book.get_spread_bps() if book else None

    def get_mid(self, symbol: str) -> Optional[float]:
        """Return mid-price for *symbol*, or None."""
        woox_sym = to_woox_symbol(symbol) if "/" in symbol else symbol.upper()
        book = self._books.get(woox_sym)
        return book.get_mid() if book else None

    def get_book_snapshot(
        self, symbol: str, levels: int = 20
    ) -> Dict[str, List[List[float]]]:
        """Return top-of-book snapshot as {bids: [...], asks: [...]}."""
        woox_sym = to_woox_symbol(symbol) if "/" in symbol else symbol.upper()
        book = self._books.get(woox_sym)
        if book is None:
            return {"bids": [], "asks": []}
        return book.get_snapshot(levels=levels)

    def is_ready(self, symbol: str) -> bool:
        """Return True if at least one snapshot has been received."""
        woox_sym = to_woox_symbol(symbol) if "/" in symbol else symbol.upper()
        book = self._books.get(woox_sym)
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
                await self._session_woox()
                backoff = _BACKOFF_INIT  # clean disconnect resets backoff
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning(
                    "WOOXWSFeed: error — %s — reconnect in %.1fs", exc, backoff
                )
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * _BACKOFF_MUL, _BACKOFF_MAX)
            # Clear books on reconnect so next snapshot starts fresh
            for book in self._books.values():
                book.clear()

    # ------------------------------------------------------------------
    # WOO X session
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, sock_read=60.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _session_woox(self) -> None:
        """Single WebSocket session: connect, subscribe, consume."""
        session = await self._get_session()

        log.info("WOOXWSFeed: connecting to %s", self._ws_url)

        async with session.ws_connect(
            self._ws_url,
            heartbeat=_HEARTBEAT_INTERVAL,
            receive_timeout=90.0,
        ) as ws:
            # Subscribe to orderbook channels for all symbols
            for symbol in self.symbols:
                ob_msg = json.dumps({
                    "event": "subscribe",
                    "topic": f"{symbol}@orderbook",
                })
                await ws.send_str(ob_msg)
                log.debug("WOOXWSFeed: subscribed to %s@orderbook", symbol)

            # Subscribe to trade channels if trade callback registered
            if self._on_trade is not None:
                for symbol in self.symbols:
                    trade_msg = json.dumps({
                        "event": "subscribe",
                        "topic": f"{symbol}@trade",
                    })
                    await ws.send_str(trade_msg)
                    log.debug("WOOXWSFeed: subscribed to %s@trade", symbol)

            log.info(
                "WOOXWSFeed: subscribed to %d symbols", len(self.symbols)
            )

            async for msg in ws:
                if not self._running:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._parse_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # WOO X may send compressed frames
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
                    log.warning("WOOXWSFeed: WebSocket closed/error")
                    break

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def _parse_message(self, raw: str) -> None:
        """Dispatch incoming WebSocket message to the correct handler."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.debug("WOOXWSFeed: JSON error %s", exc)
            return

        if not isinstance(data, dict):
            return

        # Handle subscription confirmations, pings, errors
        event = data.get("event", "")
        if event in ("subscribe", "unsubscribe", "ping", "pong", "error"):
            if event == "error":
                log.warning("WOOXWSFeed: WS error event: %s", data)
            return

        topic = data.get("topic", "")

        if "@orderbook" in topic:
            self._parse_orderbook(data, topic)
        elif "@trade" in topic:
            self._parse_trade(data, topic)

    def _parse_orderbook(self, data: dict, topic: str) -> None:
        """
        Process a WOO X orderbook update message.

        WOO X sends a full snapshot on the first message, then incremental deltas.
        A delta entry with qty=0 means that price level should be removed.

        Message format:
        {
            "topic": "SPOT_BTC_USDT@orderbook",
            "ts": 1234567890123,
            "data": {
                "bids": [[price, qty, count], ...],
                "asks": [[price, qty, count], ...],
                "ts": 1234567890123,
                "checksum": 12345678
            }
        }
        """
        # Extract symbol: "SPOT_BTC_USDT@orderbook" → "SPOT_BTC_USDT"
        symbol = topic.replace("@orderbook", "")
        book = self._books.get(symbol)
        if book is None:
            # Dynamically add if not pre-registered
            log.debug("WOOXWSFeed: received data for unregistered symbol %s", symbol)
            book = _OrderBook(symbol=symbol)
            self._books[symbol] = book

        book_data = data.get("data", {})
        if not book_data:
            return

        ts_ms = data.get("ts", book_data.get("ts", int(time.time() * 1000)))
        ts_ns = int(ts_ms) * 1_000_000

        raw_bids = book_data.get("bids", [])
        raw_asks = book_data.get("asks", [])

        # WOO X: first message after subscribe is always a full snapshot.
        # Delta updates follow. Distinguish by is_snapshot flag:
        # WOO X does not explicitly flag snapshot vs delta — we treat the
        # first received message as snapshot, subsequent as deltas.
        is_snapshot = not book.is_snapshot

        if is_snapshot:
            book.clear()
            for entry in raw_bids:
                price = float(entry[0])
                qty = float(entry[1])
                if qty > 0:
                    book.bids[price] = qty
            for entry in raw_asks:
                price = float(entry[0])
                qty = float(entry[1])
                if qty > 0:
                    book.asks[price] = qty
            book.is_snapshot = True
            book.last_update_ns = ts_ns
            log.debug(
                "WOOXWSFeed: snapshot %s bids=%d asks=%d",
                symbol, len(book.bids), len(book.asks),
            )
        else:
            # Delta update
            if not book.is_snapshot:
                log.warning(
                    "WOOXWSFeed: delta before snapshot for %s, skipping", symbol
                )
                return
            for entry in raw_bids:
                book.apply_bid_delta(float(entry[0]), float(entry[1]))
            for entry in raw_asks:
                book.apply_ask_delta(float(entry[0]), float(entry[1]))
            book.last_update_ns = ts_ns

        self._fire_book_callback(symbol, book, ts_ns)

    def _parse_trade(self, data: dict, topic: str) -> None:
        """Process a WOO X trade tick message."""
        if self._on_trade is None:
            return

        # Extract symbol: "SPOT_BTC_USDT@trade" → "SPOT_BTC_USDT"
        symbol = topic.replace("@trade", "")

        trade_data = data.get("data", {})
        if not trade_data:
            return

        ts_ms = trade_data.get("ts", data.get("ts", int(time.time() * 1000)))
        ts_ns = int(ts_ms) * 1_000_000

        side = str(trade_data.get("side", "BUY")).upper()
        price = float(trade_data.get("price", 0))
        size = float(trade_data.get("size", 0))

        self._fire_trade_callback(symbol, side, size, price, ts_ns)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fire_book_callback(
        self, symbol: str, book: _OrderBook, ts_ns: int
    ) -> None:
        """Invoke the book update callback, catching all errors."""
        if self._on_book_update is None:
            return
        try:
            result = self._on_book_update(symbol, book.bids, book.asks, ts_ns)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
        except Exception as exc:
            log.debug("WOOXWSFeed: book callback error: %s", exc)

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
            result = self._on_trade(symbol, side, size, price, ts_ns)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
        except Exception as exc:
            log.debug("WOOXWSFeed: trade callback error: %s", exc)


# ---------------------------------------------------------------------------
# Factory convenience function
# ---------------------------------------------------------------------------

def create_woox_feed(
    symbols: List[str],
    on_book_update: Optional[BookUpdateCallback] = None,
    on_trade: Optional[TradeCallback] = None,
    testnet: bool = False,
) -> WOOXWSFeed:
    """Factory helper to create a configured WOOXWSFeed.

    Parameters
    ----------
    symbols:
        List of symbols in any format: ["BTC/USDT", "ETH/USDT"] or
        ["SPOT_BTC_USDT", "SPOT_ETH_USDT"].
    on_book_update:
        Callback receiving (symbol, bids_dict, asks_dict, timestamp_ns).
    on_trade:
        Callback receiving (symbol, side, size, price, timestamp_ns).
    testnet:
        If True, connect to WOO X staging environment.

    Returns
    -------
    WOOXWSFeed
        Configured feed instance. Call ``await feed.start()`` to connect.
    """
    return WOOXWSFeed(
        symbols=symbols,
        on_book_update=on_book_update,
        on_trade=on_trade,
        testnet=testnet,
    )
