"""
core/feeds/kraken_feed.py
=========================
Kraken REST + WebSocket feed — OHLCV, trades, and order book.

Kraken is available to Australian users and is the primary venue for
ARGUS at NANO / MICRO capital tiers.  This module provides:

  KrakenOHLCVFeed    — REST polling for OHLCV candles
  KrakenTradeFeed    — WebSocket live trade stream
  KrakenLOBFeed      — WebSocket order book (book-10 channel)

Symbol convention
-----------------
Kraken uses its own symbol names (XBT/USD, ETH/USD) but also accepts
standard pairs over WS v2 (BTC/USD, ETH/USD).  This implementation
normalises ARGUS-style symbols (BTC/USDT → BTC/USD) automatically.

WebSocket API
-------------
Kraken WS v2: wss://ws.kraken.com/v2
Docs: https://docs.kraken.com/api/docs/websocket-v2/book

Rate limits
-----------
REST: 1 req/s public endpoints (safe for 10-s OHLCV polling)
WS:   no explicit per-message limit; 20 subscriptions per connection
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.feeds.lob_feed import LOBBook, LOBDelta, LOBFeed, LOBSnapshot

logger = logging.getLogger("argus.core.feeds.kraken_feed")

# Kraken WS v2 endpoint (public)
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"
KRAKEN_REST_URL = "https://api.kraken.com/0/public"

# USDT → USD mapping (Kraken does not list USDT perps for AU retail)
_SYM_MAP: Dict[str, str] = {
    "BTC/USDT": "BTC/USD",
    "ETH/USDT": "ETH/USD",
    "SOL/USDT": "SOL/USD",
    "XRP/USDT": "XRP/USD",
    "ADA/USDT": "ADA/USD",
    "DOGE/USDT": "DOGE/USD",
    "DOT/USDT": "DOT/USD",
    "AVAX/USDT": "AVAX/USD",
    "LINK/USDT": "LINK/USD",
    "MATIC/USDT": "MATIC/USD",
    "BTC/USD": "BTC/USD",
    "ETH/USD": "ETH/USD",
}


def _kraken_sym(argus_sym: str) -> str:
    """Convert ARGUS symbol to Kraken WS v2 symbol."""
    return _SYM_MAP.get(argus_sym.upper(), argus_sym.upper().replace("USDT", "USD"))


# ---------------------------------------------------------------------------
# KrakenLOBFeed
# ---------------------------------------------------------------------------

class KrakenLOBFeed(LOBFeed):
    """
    Kraken WS v2 order book — ``book`` channel, depth 10.

    Handles snapshot + delta messages and feeds LOBBook exactly like
    the Bybit / OKX implementations.

    Usage::

        feed = KrakenLOBFeed("BTC/USDT", on_snapshot=my_callback)
        await feed.start()
    """

    def _ws_url(self) -> str:
        return KRAKEN_WS_URL

    async def _on_open(self, ws: Any) -> None:
        sym = _kraken_sym(self.symbol)
        sub = {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": [sym],
                "depth": 10,
            },
        }
        await ws.send(json.dumps(sub))
        logger.info("KrakenLOBFeed: subscribed to book channel for %s (%s)", self.symbol, sym)

    def _parse_message(self, msg: dict) -> Optional[LOBSnapshot]:
        # Heartbeat / status messages
        if msg.get("channel") not in ("book",):
            return None

        msg_type = msg.get("type")   # "snapshot" | "update"
        data_list = msg.get("data", [])
        if not data_list:
            return None

        data = data_list[0]
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])

        # WS v2 format: [{"price": 60000.0, "qty": 0.5}, ...]
        bids: List[Tuple[float, float]] = [
            (float(b["price"]), float(b["qty"])) for b in bids_raw
        ]
        asks: List[Tuple[float, float]] = [
            (float(a["price"]), float(a["qty"])) for a in asks_raw
        ]

        if msg_type == "snapshot":
            return self._book.apply_snapshot(bids, asks)

        # Delta — apply level by level
        ts = time.time_ns()
        for p, q in bids:
            self._book.apply_delta(
                LOBDelta(self.symbol, "Kraken", ts, "bid", p, q,
                         "delete" if q == 0.0 else "update")
            )
        for p, q in asks:
            self._book.apply_delta(
                LOBDelta(self.symbol, "Kraken", ts, "ask", p, q,
                         "delete" if q == 0.0 else "update")
            )
        return self._book.snapshot()


# ---------------------------------------------------------------------------
# KrakenTradeFeed
# ---------------------------------------------------------------------------

class KrakenTradeFeed:
    """
    Kraken WS v2 ``trade`` channel — real-time trade stream.

    Calls ``on_trade(trade_dict)`` for every fill where trade_dict has:
      symbol, price, qty, side ("buy"|"sell"), ts_ns, ord_type

    Usage::

        feed = KrakenTradeFeed(["BTC/USDT", "ETH/USDT"], on_trade=handler)
        await feed.start()
    """

    def __init__(
        self,
        symbols: List[str],
        on_trade: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._symbols = symbols
        self._on_trade = on_trade
        self._running = False
        self._trade_count = 0

    async def start(self) -> None:
        self._running = True
        logger.info("KrakenTradeFeed: starting for %s", self._symbols)
        await self._connect()

    async def stop(self) -> None:
        self._running = False

    async def _connect(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError:
            logger.error("websockets not installed — pip install websockets")
            return

        kraken_syms = [_kraken_sym(s) for s in self._symbols]
        sub = {
            "method": "subscribe",
            "params": {"channel": "trade", "symbol": kraken_syms},
        }

        try:
            async with websockets.connect(KRAKEN_WS_URL, ping_interval=20) as ws:
                await ws.send(json.dumps(sub))
                logger.info("KrakenTradeFeed: subscribed %s", kraken_syms)
                async for raw in ws:
                    if not self._running:
                        break
                    await self._handle(raw)
        except Exception as exc:
            logger.error("KrakenTradeFeed WS error: %s", exc)

    async def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            if msg.get("channel") != "trade":
                return
            for d in msg.get("data", []):
                trade = {
                    "symbol":   d.get("symbol", ""),
                    "price":    float(d.get("price", 0)),
                    "qty":      float(d.get("qty", 0)),
                    "side":     d.get("side", ""),
                    "ts_ns":    int(d.get("timestamp", 0) or 0),
                    "ord_type": d.get("ord_type", ""),
                }
                self._trade_count += 1
                if self._on_trade:
                    if asyncio.iscoroutinefunction(self._on_trade):
                        await self._on_trade(trade)
                    else:
                        self._on_trade(trade)
        except Exception as exc:
            logger.debug("KrakenTradeFeed parse error: %s", exc)

    @property
    def stats(self) -> Dict[str, Any]:
        return {"trades_received": self._trade_count, "symbols": self._symbols}


# ---------------------------------------------------------------------------
# KrakenOHLCVFeed  (REST polling — no WS candle channel needed)
# ---------------------------------------------------------------------------

class KrakenOHLCVFeed:
    """
    Polls Kraken REST ``/OHLC`` endpoint at a configurable interval.

    Calls ``on_candle(symbol, ohlcv_list)`` where each ohlcv is:
      [timestamp_s, open, high, low, close, vwap, volume, count]

    Usage::

        feed = KrakenOHLCVFeed(
            symbols=["BTC/USDT", "ETH/USDT"],
            interval_min=1,
            on_candle=handler,
        )
        await feed.start()   # runs until stop()
    """

    def __init__(
        self,
        symbols: List[str],
        interval_min: int = 1,
        on_candle: Optional[Callable] = None,
        poll_interval_s: float = 10.0,
    ) -> None:
        self._symbols = symbols
        self._interval = interval_min
        self._on_candle = on_candle
        self._poll_s = poll_interval_s
        self._running = False
        # Track last seen timestamp per symbol to avoid re-emitting old candles
        self._last_ts: Dict[str, int] = {}

    async def start(self) -> None:
        self._running = True
        logger.info(
            "KrakenOHLCVFeed: starting for %s interval=%dm poll=%.0fs",
            self._symbols, self._interval, self._poll_s,
        )
        while self._running:
            for sym in self._symbols:
                try:
                    await self._poll(sym)
                except Exception as exc:
                    logger.debug("KrakenOHLCVFeed poll error %s: %s", sym, exc)
            await asyncio.sleep(self._poll_s)

    async def stop(self) -> None:
        self._running = False

    async def _poll(self, argus_sym: str) -> None:
        try:
            import aiohttp  # type: ignore
        except ImportError:
            logger.error("aiohttp not installed — pip install aiohttp")
            return

        kraken_sym = _kraken_sym(argus_sym).replace("/", "")
        since = self._last_ts.get(argus_sym, 0)
        url = (
            f"{KRAKEN_REST_URL}/OHLC"
            f"?pair={kraken_sym}&interval={self._interval}&since={since}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                body = await resp.json()

        if body.get("error"):
            logger.warning("KrakenOHLCVFeed REST error for %s: %s", argus_sym, body["error"])
            return

        result = body.get("result", {})
        # Kraken returns the pair key + "last" timestamp
        pair_key = next((k for k in result if k != "last"), None)
        if not pair_key:
            return

        candles = result[pair_key]
        last_ts = int(result.get("last", 0))
        if last_ts:
            self._last_ts[argus_sym] = last_ts

        # Emit only candles newer than our last seen timestamp
        new_candles = [
            c for c in candles
            if int(c[0]) > self._last_ts.get(argus_sym, 0) - self._interval * 60 * 2
        ]

        if new_candles and self._on_candle:
            if asyncio.iscoroutinefunction(self._on_candle):
                await self._on_candle(argus_sym, new_candles)
            else:
                self._on_candle(argus_sym, new_candles)
