"""
WSL2BookFeed — Full L2 Order Book Streaming Feed.

Streams and maintains a local Level-2 order book for multiple exchanges:
  - Kraken (book channel, depth 10/25/100) with checksum validation
  - Coinbase Advanced Trade (level2 channel)
  - Bybit (orderbook.50, spot)

Each symbol maintains sorted bids (descending) and asks (ascending) as
``dict[price, size]``.  Snapshot + incremental delta handling.  Auto-reconnect
with exponential backoff.  Sequence-gap detection triggers re-snapshot.

Usage::

    async with WSL2BookFeed(
        symbols=["BTC/USD"],
        exchange="kraken",
        on_book_update=my_callback,
        depth=10,
    ) as feed:
        await feed.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("argus.ws_l2_book_feed")

# ---------------------------------------------------------------------------
# Exchange URL map
# ---------------------------------------------------------------------------

EXCHANGE_URLS: Dict[str, str] = {
    "kraken":   "wss://ws.kraken.com",
    "coinbase": "wss://advanced-trade-ws.coinbase.com",
    "bybit":    "wss://stream.bybit.com/v5/public/spot",
}

# Reconnect back-off limits
_BACKOFF_INIT: float = 1.0
_BACKOFF_MAX: float  = 30.0
_BACKOFF_MUL: float  = 2.0


# ---------------------------------------------------------------------------
# Symbol normalisation helpers
# ---------------------------------------------------------------------------

def _kraken_sym(symbol: str) -> str:
    """BTC/USD → BTC/USD  (Kraken uses slash format for v1 WS)."""
    return symbol.replace("-", "/")


def _coinbase_sym(symbol: str) -> str:
    """BTC/USD → BTC-USD."""
    return symbol.replace("/", "-")


def _bybit_sym(symbol: str) -> str:
    """BTC/USD → BTCUSD  (Bybit has no separator)."""
    return symbol.replace("/", "").replace("-", "")


def _normalise_sym(raw: str, exchange: str) -> str:
    """Convert exchange-native symbol back to canonical BASE/QUOTE."""
    if exchange == "coinbase":
        return raw.replace("-", "/")
    if exchange == "bybit":
        # Bybit does not give us a clean separator; just pass through
        return raw
    return raw.replace("-", "/")


# ---------------------------------------------------------------------------
# Per-symbol order book state
# ---------------------------------------------------------------------------

@dataclass
class OrderBook:
    """Maintains a sorted L2 order book for one symbol."""

    symbol: str
    bids: Dict[float, float] = field(default_factory=dict)   # price → size
    asks: Dict[float, float] = field(default_factory=dict)   # price → size
    last_seq: int = -1
    last_update_ns: int = 0
    is_snapshot: bool = False

    # ── Accessors ──────────────────────────────────────────────────────────

    def get_best_bid(self) -> Optional[float]:
        """Return highest bid price, or None if empty."""
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Return lowest ask price, or None if empty."""
        return min(self.asks.keys()) if self.asks else None

    def get_mid(self) -> Optional[float]:
        """Return mid-price, or None if either side is empty."""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    def get_snapshot(self, levels: int = 10) -> Dict[str, List[List[float]]]:
        """
        Return top-of-book snapshot.

        Returns
        -------
        {"bids": [[price, size], ...], "asks": [[price, size], ...]}
        """
        sorted_bids = sorted(self.bids.items(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.items())[:levels]
        return {
            "bids": [[p, s] for p, s in sorted_bids],
            "asks": [[p, s] for p, s in sorted_asks],
        }

    # ── Mutation helpers ───────────────────────────────────────────────────

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
# Main feed class
# ---------------------------------------------------------------------------

BookUpdateCallback = Callable[
    [str, Dict[float, float], Dict[float, float], int],
    None,
]


class WSL2BookFeed:
    """
    Async context-manager L2 order book feed.

    Parameters
    ----------
    symbols : list[str]
        Symbols in BASE/QUOTE format, e.g. ``["BTC/USD", "ETH/USD"]``.
    exchange : str
        One of ``"kraken"``, ``"coinbase"``, ``"bybit"``.
    on_book_update : callable, optional
        Called with ``(symbol, bids, asks, timestamp_ns)`` on every book update.
    depth : int
        Subscription depth (10, 25, 100 for Kraken; ignored for Coinbase/Bybit).
    """

    def __init__(
        self,
        symbols: List[str],
        exchange: str = "kraken",
        on_book_update: Optional[BookUpdateCallback] = None,
        depth: int = 10,
    ) -> None:
        self.symbols = list(symbols)
        self.exchange = exchange.lower()
        self._on_book_update = on_book_update
        self.depth = depth

        if self.exchange not in EXCHANGE_URLS:
            raise ValueError(
                f"Unsupported exchange '{exchange}'. "
                f"Supported: {list(EXCHANGE_URLS)}"
            )

        self._books: Dict[str, OrderBook] = {
            s: OrderBook(symbol=s) for s in self.symbols
        }
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Resnapshot request queue — symbols that need a fresh snapshot
        self._resnapshot_queue: asyncio.Queue[str] = asyncio.Queue()

        log.info(
            "WSL2BookFeed init: exchange=%s symbols=%s depth=%d",
            self.exchange, self.symbols, self.depth,
        )

    # ── Context manager ────────────────────────────────────────────────────

    async def __aenter__(self) -> "WSL2BookFeed":
        return self

    async def __aexit__(self, *_) -> None:
        self.stop()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    # ── Control ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the feed loop; blocks until stopped."""
        self._running = True
        self._task = asyncio.current_task()
        await self._connect_loop()

    def stop(self) -> None:
        """Signal the feed to stop reconnecting."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # ── Public accessors ───────────────────────────────────────────────────

    def get_best_bid(self, symbol: str) -> Optional[float]:
        """Return best bid for *symbol*, or None."""
        book = self._books.get(symbol)
        return book.get_best_bid() if book else None

    def get_best_ask(self, symbol: str) -> Optional[float]:
        """Return best ask for *symbol*, or None."""
        book = self._books.get(symbol)
        return book.get_best_ask() if book else None

    def get_mid(self, symbol: str) -> Optional[float]:
        """Return mid-price for *symbol*, or None."""
        book = self._books.get(symbol)
        return book.get_mid() if book else None

    def get_book_snapshot(
        self, symbol: str, levels: int = 10
    ) -> Dict[str, List[List[float]]]:
        """
        Return top-of-book snapshot for *symbol*.

        Returns
        -------
        {"bids": [[price, size], ...], "asks": [[price, size], ...]}
        """
        book = self._books.get(symbol)
        if book is None:
            return {"bids": [], "asks": []}
        return book.get_snapshot(levels=levels)

    def register_callback(self, callback: BookUpdateCallback) -> None:
        """Register (or replace) the book update callback."""
        self._on_book_update = callback

    # ── Connection loop ────────────────────────────────────────────────────

    async def _connect_loop(self) -> None:
        backoff = _BACKOFF_INIT
        while self._running:
            try:
                await self._session()
                backoff = _BACKOFF_INIT  # clean disconnect → reset backoff
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning(
                    "WSL2BookFeed[%s]: error — %s — reconnect in %.1fs",
                    self.exchange, exc, backoff,
                )
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * _BACKOFF_MUL, _BACKOFF_MAX)
            # Clear books so next snapshot starts fresh
            for book in self._books.values():
                book.clear()

    # ── Exchange sessions ──────────────────────────────────────────────────

    async def _session(self) -> None:
        """Dispatch to the correct exchange session handler."""
        if self.exchange == "kraken":
            await self._session_kraken()
        elif self.exchange == "coinbase":
            await self._session_coinbase()
        elif self.exchange == "bybit":
            await self._session_bybit()

    # ──────────────────────────────────────────────────────────────────────
    # Kraken session
    # ──────────────────────────────────────────────────────────────────────

    async def _session_kraken(self) -> None:
        import websockets  # type: ignore

        url = EXCHANGE_URLS["kraken"]
        sub_msg = json.dumps({
            "event": "subscribe",
            "pair": [_kraken_sym(s) for s in self.symbols],
            "subscription": {"name": "book", "depth": self.depth},
        })

        log.info("WSL2BookFeed[kraken]: connecting to %s", url)
        async with websockets.connect(url, ping_interval=20, ping_timeout=30) as ws:
            # Optionally tune socket for minimum latency
            try:
                from infra.socket_tuner import tune_websocket_transport  # type: ignore
                tune_websocket_transport(ws.transport)
            except Exception:
                pass

            await ws.send(sub_msg)
            log.info("WSL2BookFeed[kraken]: subscribed to %s (depth=%d)", self.symbols, self.depth)

            async for raw in ws:
                if not self._running:
                    break
                self._parse_kraken(raw)

    def _parse_kraken(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.debug("WSL2BookFeed[kraken]: JSON error %s", exc)
            return

        # Status / heartbeat messages (dicts with "event" key)
        if isinstance(data, dict):
            ev = data.get("event", "")
            if ev in ("heartbeat", "systemStatus", "subscriptionStatus"):
                return
            log.debug("WSL2BookFeed[kraken]: unhandled dict message: %s", data)
            return

        # Book messages arrive as a list:
        # [channel_id, book_data, "book-N", "XBT/USD"]
        if not isinstance(data, list) or len(data) < 4:
            return

        pair = str(data[-1])
        # Normalise to BASE/QUOTE
        symbol = pair.replace("XBT", "BTC")

        # Find our canonical symbol
        canonical = self._resolve_kraken_symbol(symbol)
        if canonical is None:
            return

        book_data = data[1] if len(data) == 4 else None
        # Sometimes Kraken sends 5-element arrays: [chan, bids, asks, type, pair]
        if len(data) == 5:
            book_data = {**data[1], **data[2]}

        if not isinstance(book_data, dict):
            return

        ob = self._books[canonical]
        ts_ns = time.time_ns()

        if "as" in book_data or "bs" in book_data:
            # Snapshot
            ob.clear()
            for price_str, size_str, *_ in book_data.get("bs", []):
                ob.bids[float(price_str)] = float(size_str)
            for price_str, size_str, *_ in book_data.get("as", []):
                ob.asks[float(price_str)] = float(size_str)
            ob.is_snapshot = True
            ob.last_update_ns = ts_ns
            log.debug("WSL2BookFeed[kraken]: snapshot %s bids=%d asks=%d", canonical, len(ob.bids), len(ob.asks))
        else:
            # Delta update — check sequence
            if not ob.is_snapshot:
                log.warning("WSL2BookFeed[kraken]: delta before snapshot for %s, skipping", canonical)
                self._resnapshot_queue.put_nowait(canonical)
                return

            for price_str, size_str, *_ in book_data.get("b", []):
                ob.apply_bid_delta(float(price_str), float(size_str))
            for price_str, size_str, *_ in book_data.get("a", []):
                ob.apply_ask_delta(float(price_str), float(size_str))

            # Checksum validation (Kraken provides checksum in delta messages)
            if "c" in book_data:
                expected = int(book_data["c"])
                if not self._verify_kraken_checksum(ob, expected):
                    log.warning(
                        "WSL2BookFeed[kraken]: checksum mismatch for %s — requesting re-snapshot",
                        canonical,
                    )
                    ob.clear()
                    self._resnapshot_queue.put_nowait(canonical)
                    return

            ob.last_update_ns = ts_ns

        self._fire_callback(canonical, ob, ts_ns)

    def _resolve_kraken_symbol(self, kraken_sym: str) -> Optional[str]:
        """Map a Kraken pair string to our canonical symbol."""
        # Direct match first
        if kraken_sym in self._books:
            return kraken_sym
        # Try replacing XBT→BTC
        xbt = kraken_sym.replace("XBT", "BTC")
        if xbt in self._books:
            return xbt
        # Try dot-separated (e.g. XBT/USD)
        slash = kraken_sym.replace(".", "/")
        if slash in self._books:
            return slash
        return None

    @staticmethod
    def _verify_kraken_checksum(ob: OrderBook, expected: int) -> bool:
        """
        Kraken checksum: CRC32 of top-10 ask + top-10 bid price/size strings
        with decimal point removed.
        """
        def _fmt(val: float) -> str:
            # Format as string, remove decimal point, leading zeros
            s = f"{val:.8f}".rstrip("0").rstrip(".")
            return s.replace(".", "")

        top_asks = sorted(ob.asks.items())[:10]
        top_bids = sorted(ob.bids.items(), reverse=True)[:10]

        payload = ""
        for price, size in top_asks:
            payload += _fmt(price) + _fmt(size)
        for price, size in top_bids:
            payload += _fmt(price) + _fmt(size)

        calc = zlib.crc32(payload.encode()) & 0xFFFFFFFF
        return calc == expected

    # ──────────────────────────────────────────────────────────────────────
    # Coinbase Advanced Trade session
    # ──────────────────────────────────────────────────────────────────────

    async def _session_coinbase(self) -> None:
        import websockets  # type: ignore

        url = EXCHANGE_URLS["coinbase"]
        sub_msg = json.dumps({
            "type": "subscribe",
            "product_ids": [_coinbase_sym(s) for s in self.symbols],
            "channel": "level2",
        })

        log.info("WSL2BookFeed[coinbase]: connecting to %s", url)
        async with websockets.connect(url, ping_interval=20, ping_timeout=30) as ws:
            try:
                from infra.socket_tuner import tune_websocket_transport  # type: ignore
                tune_websocket_transport(ws.transport)
            except Exception:
                pass

            await ws.send(sub_msg)
            log.info("WSL2BookFeed[coinbase]: subscribed to %s", self.symbols)

            async for raw in ws:
                if not self._running:
                    break
                self._parse_coinbase(raw)

    def _parse_coinbase(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.debug("WSL2BookFeed[coinbase]: JSON error %s", exc)
            return

        if not isinstance(data, dict):
            return

        channel = data.get("channel", "")
        if channel != "l2_data":
            return

        ts_ns = time.time_ns()

        for event in data.get("events", []):
            ev_type = event.get("type", "")
            product_id = event.get("product_id", "")
            symbol = _normalise_sym(product_id, "coinbase")
            canonical = self._resolve_symbol_fuzzy(symbol)
            if canonical is None:
                continue

            ob = self._books[canonical]

            if ev_type == "snapshot":
                ob.clear()
                for update in event.get("updates", []):
                    side = update.get("side", "")
                    price = float(update["price_level"])
                    size = float(update["new_quantity"])
                    if side == "bid":
                        ob.bids[price] = size
                    elif side == "offer":
                        ob.asks[price] = size
                ob.is_snapshot = True
                ob.last_update_ns = ts_ns
                log.debug("WSL2BookFeed[coinbase]: snapshot %s bids=%d asks=%d", canonical, len(ob.bids), len(ob.asks))
            elif ev_type == "update":
                if not ob.is_snapshot:
                    log.warning("WSL2BookFeed[coinbase]: update before snapshot for %s", canonical)
                    self._resnapshot_queue.put_nowait(canonical)
                    continue

                # Sequence-number gap detection
                seq = event.get("sequence_num")
                if seq is not None:
                    seq = int(seq)
                    if ob.last_seq >= 0 and seq != ob.last_seq + 1:
                        log.warning(
                            "WSL2BookFeed[coinbase]: seq gap %d→%d for %s — re-snapshot",
                            ob.last_seq, seq, canonical,
                        )
                        ob.clear()
                        self._resnapshot_queue.put_nowait(canonical)
                        continue
                    ob.last_seq = seq

                for update in event.get("updates", []):
                    side = update.get("side", "")
                    price = float(update["price_level"])
                    size = float(update["new_quantity"])
                    if side == "bid":
                        ob.apply_bid_delta(price, size)
                    elif side == "offer":
                        ob.apply_ask_delta(price, size)

                ob.last_update_ns = ts_ns

            self._fire_callback(canonical, ob, ts_ns)

    # ──────────────────────────────────────────────────────────────────────
    # Bybit session
    # ──────────────────────────────────────────────────────────────────────

    async def _session_bybit(self) -> None:
        import websockets  # type: ignore

        url = EXCHANGE_URLS["bybit"]
        args = [f"orderbook.50.{_bybit_sym(s)}" for s in self.symbols]
        sub_msg = json.dumps({
            "op": "subscribe",
            "args": args,
        })

        log.info("WSL2BookFeed[bybit]: connecting to %s", url)
        async with websockets.connect(url, ping_interval=20, ping_timeout=30) as ws:
            try:
                from infra.socket_tuner import tune_websocket_transport  # type: ignore
                tune_websocket_transport(ws.transport)
            except Exception:
                pass

            await ws.send(sub_msg)
            log.info("WSL2BookFeed[bybit]: subscribed %s", args)

            async for raw in ws:
                if not self._running:
                    break
                self._parse_bybit(raw)

    def _parse_bybit(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.debug("WSL2BookFeed[bybit]: JSON error %s", exc)
            return

        if not isinstance(data, dict):
            return

        # op/pong/subscription confirmation
        if "op" in data or data.get("topic") is None:
            return

        topic = data.get("topic", "")
        # topic format: "orderbook.50.BTCUSDT"
        if not topic.startswith("orderbook"):
            return

        msg_type = data.get("type", "")  # "snapshot" or "delta"
        book_data = data.get("data", {})
        ts_ns = time.time_ns()

        # Extract symbol from topic
        parts = topic.split(".")
        bybit_sym = parts[-1] if len(parts) >= 3 else ""
        canonical = self._resolve_bybit_symbol(bybit_sym)
        if canonical is None:
            return

        ob = self._books[canonical]

        # Sequence tracking
        seq = data.get("seq") or data.get("u")
        if seq is not None:
            seq = int(seq)
            if msg_type == "delta" and ob.last_seq >= 0 and seq != ob.last_seq + 1:
                log.warning(
                    "WSL2BookFeed[bybit]: seq gap %d→%d for %s — re-snapshot",
                    ob.last_seq, seq, canonical,
                )
                ob.clear()
                self._resnapshot_queue.put_nowait(canonical)
                return
            ob.last_seq = seq

        if msg_type == "snapshot":
            ob.clear()
            for entry in book_data.get("b", []):
                price, size = float(entry[0]), float(entry[1])
                if size > 0:
                    ob.bids[price] = size
            for entry in book_data.get("a", []):
                price, size = float(entry[0]), float(entry[1])
                if size > 0:
                    ob.asks[price] = size
            ob.is_snapshot = True
            ob.last_update_ns = ts_ns
            log.debug("WSL2BookFeed[bybit]: snapshot %s bids=%d asks=%d", canonical, len(ob.bids), len(ob.asks))
        elif msg_type == "delta":
            if not ob.is_snapshot:
                log.warning("WSL2BookFeed[bybit]: delta before snapshot for %s", canonical)
                self._resnapshot_queue.put_nowait(canonical)
                return

            # Bybit delta: "b" = bid updates, "a" = ask updates
            for entry in book_data.get("b", []):
                ob.apply_bid_delta(float(entry[0]), float(entry[1]))
            for entry in book_data.get("a", []):
                ob.apply_ask_delta(float(entry[0]), float(entry[1]))

            ob.last_update_ns = ts_ns

        self._fire_callback(canonical, ob, ts_ns)

    def _resolve_bybit_symbol(self, bybit_sym: str) -> Optional[str]:
        """Map Bybit BTCUSDT back to a canonical symbol."""
        # Try exact match
        for sym in self._books:
            candidate = _bybit_sym(sym)
            if candidate == bybit_sym:
                return sym
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_symbol_fuzzy(self, symbol: str) -> Optional[str]:
        """Fuzzy match an exchange symbol to our canonical symbol list."""
        if symbol in self._books:
            return symbol
        # Try with hyphen → slash
        s2 = symbol.replace("-", "/")
        if s2 in self._books:
            return s2
        # Try replacing USDT→USD
        s3 = symbol.replace("USDT", "USD")
        if s3 in self._books:
            return s3
        return None

    def _fire_callback(self, symbol: str, ob: OrderBook, ts_ns: int) -> None:
        """Invoke the registered book-update callback, catching all errors."""
        if self._on_book_update is None:
            return
        try:
            self._on_book_update(symbol, ob.bids, ob.asks, ts_ns)
        except Exception as exc:
            log.debug("WSL2BookFeed: callback error: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Re-snapshot handling (called externally or from a background task)
    # ──────────────────────────────────────────────────────────────────────

    async def process_resnapshot_queue(self, ws) -> None:
        """
        Drain the re-snapshot queue and send fresh subscription requests to
        the given WebSocket connection.  Should be called periodically from
        the session loops in a production setup.
        """
        while not self._resnapshot_queue.empty():
            symbol = self._resnapshot_queue.get_nowait()
            log.info("WSL2BookFeed: requesting re-snapshot for %s", symbol)
            if self.exchange == "kraken":
                sub_msg = json.dumps({
                    "event": "subscribe",
                    "pair": [_kraken_sym(symbol)],
                    "subscription": {"name": "book", "depth": self.depth},
                })
            elif self.exchange == "coinbase":
                sub_msg = json.dumps({
                    "type": "subscribe",
                    "product_ids": [_coinbase_sym(symbol)],
                    "channel": "level2",
                })
            elif self.exchange == "bybit":
                sub_msg = json.dumps({
                    "op": "subscribe",
                    "args": [f"orderbook.50.{_bybit_sym(symbol)}"],
                })
            else:
                continue
            try:
                await ws.send(sub_msg)
            except Exception as exc:
                log.warning("WSL2BookFeed: re-snapshot send failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Convenience properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def books(self) -> Dict[str, OrderBook]:
        """Direct access to the underlying OrderBook objects (read-only usage)."""
        return self._books

    def is_ready(self, symbol: str) -> bool:
        """Return True if we have received at least one snapshot for *symbol*."""
        book = self._books.get(symbol)
        return book is not None and book.is_snapshot and bool(book.bids) and bool(book.asks)


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def create_feed(
    symbols: List[str],
    exchange: str,
    on_book_update: Optional[BookUpdateCallback] = None,
    depth: int = 10,
) -> WSL2BookFeed:
    """Factory helper to create a configured WSL2BookFeed."""
    return WSL2BookFeed(
        symbols=symbols,
        exchange=exchange,
        on_book_update=on_book_update,
        depth=depth,
    )
