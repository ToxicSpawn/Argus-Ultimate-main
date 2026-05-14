"""
kraken_feed.py — Live Kraken WebSocket OHLCV feed.

Streams real-time OHLCV candles from Kraken's public WebSocket API v2
and maintains a rolling in-memory candle buffer compatible with the
Argus tentacle system (shape: (N, 6) [ts, open, high, low, close, vol]).

Features
--------
- Async WebSocket connection with auto-reconnect (exponential backoff)
- Subscribes to `ohlc` channel for configurable interval (1, 5, 15, 60 min)
- Thread-safe candle buffer with configurable max size
- Callback hooks: on_candle, on_reconnect, on_error
- Heartbeat watchdog: reconnects if no message received within timeout
- Graceful shutdown via stop()
- REST fallback via fetch_ohlcv_rest() for backfill on startup

Usage
-----
    feed = KrakenOHLCVFeed(symbol="XBT/USD", interval=1, buffer_size=500)
    await feed.start()
    candles = feed.get_candles()  # np.ndarray (N, 6)
    await feed.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

logger = logging.getLogger(__name__)

KRAKEN_WS_URL           = "wss://ws.kraken.com/v2"
VALID_INTERVALS         = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}
DEFAULT_RECONNECT_DELAY = 2.0
MAX_RECONNECT_DELAY     = 60.0
HEARTBEAT_TIMEOUT_SEC   = 30


class KrakenOHLCVFeed:
    """
    Async Kraken WebSocket OHLCV feed.

    Parameters
    ----------
    symbol       : Kraken pair e.g. "XBT/USD", "ETH/USD"
    interval     : candle interval in minutes (default: 1)
    buffer_size  : max candles kept in memory (default: 1000)
    on_candle    : optional callback(candle: np.ndarray) on each new candle
    on_reconnect : optional callback() after reconnect
    on_error     : optional callback(exc: Exception) on error
    """

    def __init__(
        self,
        symbol: str = "XBT/USD",
        interval: int = 1,
        buffer_size: int = 1000,
        on_candle: Optional[Callable] = None,
        on_reconnect: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"interval must be one of {VALID_INTERVALS}, got {interval}")
        self._symbol      = symbol
        self._interval    = interval
        self._buffer_size = buffer_size
        self._on_candle   = on_candle
        self._on_reconnect= on_reconnect
        self._on_error    = on_error
        self._buffer: Deque[List[float]] = deque(maxlen=buffer_size)
        self._lock        = asyncio.Lock()
        self._running     = False
        self._task: Optional[asyncio.Task] = None
        self._last_msg_ts = 0.0
        self._reconnect_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("KrakenOHLCVFeed started: %s interval=%dm", self._symbol, self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("KrakenOHLCVFeed stopped")

    def get_candles(self) -> np.ndarray:
        """Return buffer as (N, 6) float64 array [ts, open, high, low, close, vol]."""
        if not self._buffer:
            return np.empty((0, 6), dtype=np.float64)
        return np.array(list(self._buffer), dtype=np.float64)

    def get_latest_price(self) -> Optional[float]:
        return float(self._buffer[-1][4]) if self._buffer else None

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def candle_count(self) -> int:
        return len(self._buffer)

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    async def seed_from_rest(self, bars: int = 500) -> int:
        """
        Backfill buffer from Kraken REST API on startup.
        Returns number of candles loaded.
        """
        rest_symbol = self._symbol.replace("/", "")
        try:
            candles = await fetch_ohlcv_rest(rest_symbol, self._interval)
            async with self._lock:
                for row in candles[-bars:]:
                    self._buffer.append(list(row))
            logger.info("KrakenOHLCVFeed seeded %d candles from REST", len(self._buffer))
            return len(self._buffer)
        except Exception as exc:  # noqa: BLE001
            logger.warning("REST seed failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # WebSocket loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        delay = DEFAULT_RECONNECT_DELAY
        while self._running:
            try:
                await self._connect_and_stream()
                delay = DEFAULT_RECONNECT_DELAY
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("KrakenOHLCVFeed error: %s — reconnecting in %.1fs", exc, delay)
                if self._on_error:
                    try:
                        self._on_error(exc)
                    except Exception:  # noqa: BLE001
                        pass
                if self._running:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, MAX_RECONNECT_DELAY)
                    self._reconnect_count += 1
                    if self._on_reconnect:
                        try:
                            self._on_reconnect()
                        except Exception:  # noqa: BLE001
                            pass

    async def _connect_and_stream(self) -> None:
        if not _WS_AVAILABLE:
            raise RuntimeError("websockets not installed: pip install websockets")
        async with websockets.connect(
            KRAKEN_WS_URL, ping_interval=20, ping_timeout=10, close_timeout=5,
        ) as ws:
            logger.info("KrakenOHLCVFeed connected")
            await self._subscribe(ws)
            self._last_msg_ts = time.time()
            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=HEARTBEAT_TIMEOUT_SEC)
                    self._last_msg_ts = time.time()
                    await self._handle_message(raw)
                except asyncio.TimeoutError:
                    logger.warning("KrakenOHLCVFeed heartbeat timeout — reconnecting")
                    break

    async def _subscribe(self, ws) -> None:
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "ohlc",
                "symbol": [self._symbol],
                "interval": self._interval,
            },
        }))

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        channel = msg.get("channel")
        if channel in ("heartbeat", "status"):
            return
        if channel != "ohlc":
            return
        data     = msg.get("data", [])
        msg_type = msg.get("type", "update")
        async with self._lock:
            for candle_data in data:
                candle = self._parse_candle(candle_data)
                if candle is None:
                    continue
                if msg_type == "snapshot":
                    self._buffer.append(candle)
                else:
                    if self._buffer and self._buffer[-1][0] == candle[0]:
                        self._buffer[-1] = candle
                    else:
                        self._buffer.append(candle)
                if self._on_candle:
                    try:
                        self._on_candle(np.array(candle, dtype=np.float64))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("on_candle callback error: %s", exc)

    @staticmethod
    def _parse_candle(data: Dict[str, Any]) -> Optional[List[float]]:
        try:
            ts     = float(data.get("timestamp", data.get("interval_begin", 0)))
            open_  = float(data["open"])
            high   = float(data["high"])
            low    = float(data["low"])
            close  = float(data["close"])
            volume = float(data["volume"])
            return [ts, open_, high, low, close, volume]
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("KrakenOHLCVFeed parse error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# REST backfill helper
# ---------------------------------------------------------------------------

async def fetch_ohlcv_rest(
    symbol: str = "XBTUSD",
    interval: int = 1,
    since: Optional[int] = None,
) -> np.ndarray:
    """
    Fetch OHLCV history from Kraken REST API.
    Returns np.ndarray shape (N, 6): [ts, open, high, low, close, volume].

    Parameters
    ----------
    symbol   : Kraken REST pair e.g. "XBTUSD", "ETHUSD"
    interval : minutes
    since    : Unix timestamp (optional)
    """
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp required: pip install aiohttp")

    url    = "https://api.kraken.com/0/public/OHLC"
    params: Dict[str, Any] = {"pair": symbol, "interval": interval}
    if since:
        params["since"] = since

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params,
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            body = await resp.json()

    if body.get("error"):
        raise RuntimeError(f"Kraken REST error: {body['error']}")

    result_key = next(k for k in body["result"] if k != "last")
    rows = body["result"][result_key]
    # Kraken REST OHLC row: [time, open, high, low, close, vwap, volume, count]
    candles = np.array(
        [[float(r[0]), float(r[1]), float(r[2]),
          float(r[3]), float(r[4]), float(r[6])] for r in rows],
        dtype=np.float64,
    )
    logger.info("Fetched %d candles from Kraken REST (%s %dm)", len(candles), symbol, interval)
    return candles
