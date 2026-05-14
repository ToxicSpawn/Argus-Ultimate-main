"""
Real-Time WebSocket Price Feed — Kraken + Coinbase.

Subscribes to public ticker WebSocket streams from Kraken and Coinbase.
Maintains a live ``prices`` dict and pushes updates to a registered callback.

Handles automatic reconnection with exponential back-off.

Usage::

    feed = WSPriceFeed(
        symbols=["BTC/USD", "ETH/USD"],
        on_price=lambda sym, price: print(sym, price),
        exchanges=["kraken", "coinbase"],
    )
    asyncio.run(feed.run())

Or start as a background task::

    task = asyncio.create_task(feed.run())
    ...
    feed.stop()
    await task
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# WebSocket endpoints
_KRAKEN_WS_URL     = "wss://ws.kraken.com/v2"
_COINBASE_WS_URL   = "wss://advanced-trade-ws.coinbase.com"

# Reconnection back-off
_INITIAL_BACKOFF  = 1.0   # seconds
_MAX_BACKOFF      = 60.0  # seconds
_BACKOFF_FACTOR   = 2.0

# Bar builder
_BAR_TF_SECS  = 60    # 1-minute bars
_BAR_BUF_SIZE = 100   # completed bars kept per symbol
_TICK_BUF_SIZE = 500  # ticks kept per symbol


def _kraken_symbol(symbol: str) -> str:
    """Convert 'BTC/USD' → 'BTC/USD' (Kraken v2 uses slash format)."""
    return symbol.replace("-", "/")


def _coinbase_symbol(symbol: str) -> str:
    """Convert 'BTC/USD' → 'BTC-USD' (Coinbase uses dash format)."""
    return symbol.replace("/", "-")


class WSPriceFeed:
    """
    Concurrent WebSocket price feed for Kraken and/or Coinbase.

    Parameters
    ----------
    symbols : list of str
        Symbols to subscribe to in 'BASE/QUOTE' format (e.g. 'BTC/USD').
    on_price : callable, optional
        Called with (symbol: str, price: float) on each ticker update.
    exchanges : list of str
        Which exchanges to connect to. Supports 'kraken' and 'coinbase'.
    heartbeat_interval : float
        Seconds between internal heartbeat pings.
    """

    def __init__(
        self,
        symbols: List[str],
        on_price: Optional[Callable[[str, float], None]] = None,
        exchanges: Optional[List[str]] = None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self.symbols = list(symbols)
        self.on_price = on_price
        self.exchanges: List[str] = [e.lower() for e in (exchanges or ["kraken", "coinbase"])]
        self.heartbeat_interval = float(heartbeat_interval)

        # Live prices — readable by external callers
        self.prices: Dict[str, float] = {}

        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._last_update: Dict[str, float] = {}    # symbol → timestamp

        # Bar builder — 1-minute OHLCV bars built from ticks
        self.on_bar_close: Optional[Callable[[str, dict], None]] = None
        self._tick_buf:      Dict[str, Deque[Tuple[float, float]]] = {}  # symbol → [(ts, price)]
        self._current_bar:   Dict[str, dict] = {}                        # symbol → open bar
        self._completed_bars: Dict[str, Deque[dict]] = {}               # symbol → closed bars

    # ── Control ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start all feed tasks and block until stopped."""
        self._running = True
        self._tasks = []

        if "kraken" in self.exchanges:
            self._tasks.append(
                asyncio.create_task(self._connect_kraken(), name="ws_kraken")
            )
        if "coinbase" in self.exchanges:
            self._tasks.append(
                asyncio.create_task(self._connect_coinbase(), name="ws_coinbase")
            )

        if not self._tasks:
            logger.warning("WSPriceFeed: no exchange tasks started")
            return

        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Signal all feed tasks to stop."""
        self._running = False
        for task in self._tasks:
            task.cancel()

    @property
    def is_running(self) -> bool:
        return self._running

    def stale_symbols(self, max_age_seconds: float = 60.0) -> Set[str]:
        """Return symbols whose last update is older than max_age_seconds."""
        now = time.time()
        return {s for s, t in self._last_update.items() if now - t > max_age_seconds}

    # ── Bar / Tick accessors ──────────────────────────────────────────────────

    def current_bar(self, symbol: str) -> Optional[dict]:
        """Return the currently open (incomplete) 1-minute bar for *symbol*, or None."""
        return self._current_bar.get(symbol)

    def completed_bars(self, symbol: str, n: int = 20) -> List[dict]:
        """Return up to *n* most-recently completed 1-minute bars for *symbol*."""
        buf = self._completed_bars.get(symbol)
        if not buf:
            return []
        bars = list(buf)
        return bars[-n:]

    def tick_buffer(self, symbol: str, n: int = 100) -> List[Tuple[float, float]]:
        """Return up to *n* most-recent (timestamp, price) ticks for *symbol*."""
        buf = self._tick_buf.get(symbol)
        if not buf:
            return []
        ticks = list(buf)
        return ticks[-n:]

    # ── Kraken WebSocket v2 ───────────────────────────────────────────────────

    async def _connect_kraken(self) -> None:
        backoff = _INITIAL_BACKOFF
        while self._running:
            try:
                await self._kraken_session(backoff)
                backoff = _INITIAL_BACKOFF  # reset on clean disconnect
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("WSPriceFeed[Kraken]: error — %s — reconnecting in %.1fs", exc, backoff)
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * _BACKOFF_FACTOR, _MAX_BACKOFF)

    async def _kraken_session(self, _backoff: float) -> None:
        try:
            import aiohttp
        except ImportError:
            logger.error("WSPriceFeed[Kraken]: aiohttp not installed — install aiohttp")
            self._running = False
            return

        subscribe_msg = json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "ticker",
                "symbol": [_kraken_symbol(s) for s in self.symbols],
            },
        })

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                _KRAKEN_WS_URL,
                heartbeat=self.heartbeat_interval,
                receive_timeout=120.0,
            ) as ws:
                logger.info(
                    "WSPriceFeed[Kraken]: connected — subscribing %s", self.symbols
                )
                await ws.send_str(subscribe_msg)

                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._parse_kraken(msg.data)
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        logger.warning("WSPriceFeed[Kraken]: connection closed/error")
                        break

    def _parse_kraken(self, raw: str) -> None:
        """Parse Kraken v2 ticker message."""
        try:
            data = json.loads(raw)
            # v2 format: {"channel": "ticker", "type": "update", "data": [...]}
            if not isinstance(data, dict):
                return
            if data.get("channel") != "ticker":
                return
            for item in data.get("data", []):
                sym = str(item.get("symbol", "")).replace("-", "/")
                last = item.get("last") or item.get("ask") or item.get("bid")
                if sym and last:
                    price = float(last)
                    self._emit(sym, price)
        except Exception as exc:
            logger.debug("WSPriceFeed[Kraken] parse error: %s", exc)

    # ── Coinbase Advanced Trade WebSocket ─────────────────────────────────────

    async def _connect_coinbase(self) -> None:
        backoff = _INITIAL_BACKOFF
        while self._running:
            try:
                await self._coinbase_session(backoff)
                backoff = _INITIAL_BACKOFF
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("WSPriceFeed[Coinbase]: error — %s — reconnecting in %.1fs", exc, backoff)
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * _BACKOFF_FACTOR, _MAX_BACKOFF)

    async def _coinbase_session(self, _backoff: float) -> None:
        try:
            import aiohttp
        except ImportError:
            logger.error("WSPriceFeed[Coinbase]: aiohttp not installed")
            self._running = False
            return

        subscribe_msg = json.dumps({
            "type": "subscribe",
            "product_ids": [_coinbase_symbol(s) for s in self.symbols],
            "channel": "ticker",
        })

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                _COINBASE_WS_URL,
                heartbeat=self.heartbeat_interval,
                receive_timeout=120.0,
            ) as ws:
                logger.info(
                    "WSPriceFeed[Coinbase]: connected — subscribing %s", self.symbols
                )
                await ws.send_str(subscribe_msg)

                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._parse_coinbase(msg.data)
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        logger.warning("WSPriceFeed[Coinbase]: connection closed/error")
                        break

    def _parse_coinbase(self, raw: str) -> None:
        """Parse Coinbase Advanced Trade ticker message."""
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return
            # Advanced Trade format: {"channel": "ticker", "events": [{"tickers": [...]}]}
            channel = data.get("channel", "") or data.get("type", "")
            if channel not in ("ticker", "ticker_batch"):
                return
            for event in data.get("events", []):
                for ticker in event.get("tickers", []):
                    pid = str(ticker.get("product_id", ""))
                    sym = pid.replace("-", "/")
                    price_str = ticker.get("price") or ticker.get("best_ask") or ticker.get("best_bid")
                    if sym and price_str:
                        self._emit(sym, float(price_str))
        except Exception as exc:
            logger.debug("WSPriceFeed[Coinbase] parse error: %s", exc)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, symbol: str, price: float) -> None:
        """Update price dict, bar builder, tick buffer, and fire callback."""
        now = time.time()
        self.prices[symbol] = price
        self._last_update[symbol] = now

        # Tick buffer
        if symbol not in self._tick_buf:
            self._tick_buf[symbol] = deque(maxlen=_TICK_BUF_SIZE)
        self._tick_buf[symbol].append((now, price))

        # 1-minute bar builder
        self._update_bar(symbol, price, now)

        if self.on_price is not None:
            try:
                self.on_price(symbol, price)
            except Exception as exc:
                logger.debug("WSPriceFeed on_price callback error: %s", exc)

    def _update_bar(self, symbol: str, price: float, ts: float) -> None:
        """Maintain rolling 1-minute OHLCV bar; fire on_bar_close when minute rolls."""
        bar_ts = int(ts // _BAR_TF_SECS) * _BAR_TF_SECS
        cb = self._current_bar.get(symbol)

        if cb is None or cb["ts"] != bar_ts:
            # Close the previous bar
            if cb is not None:
                if symbol not in self._completed_bars:
                    self._completed_bars[symbol] = deque(maxlen=_BAR_BUF_SIZE)
                self._completed_bars[symbol].append(cb)
                if self.on_bar_close is not None:
                    try:
                        self.on_bar_close(symbol, cb)
                    except Exception as exc:
                        logger.debug("WSPriceFeed on_bar_close error: %s", exc)
            # Open a new bar
            self._current_bar[symbol] = {
                "ts": bar_ts, "symbol": symbol,
                "open": price, "high": price, "low": price, "close": price,
                "ticks": 1,
            }
        else:
            cb["high"]  = max(cb["high"],  price)
            cb["low"]   = min(cb["low"],   price)
            cb["close"] = price
            cb["ticks"] += 1
