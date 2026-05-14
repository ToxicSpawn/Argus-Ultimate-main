"""
MTF Bar-Close Router for Argus Live.

Subscribes to exchange WS kline/OHLCV streams and routes confirmed
closed candles into the MTFConfluenceFilter push-based cache.

Supports two feed modes (auto-selected):
  1. **WS streaming** via CCXT Pro ``watch_ohlcv`` — preferred, zero-lag.
  2. **REST polling fallback** — fetches via ``fetch_ohlcv`` every
     ``poll_interval_s`` seconds when WS is unavailable.

On startup, pre-seeds each symbol/timeframe buffer with
``seed_candles`` historical closed candles via REST so the filter is
warm immediately and doesn't require waiting for 50+ live bar closes.

Usage (from live entrypoint)::

    from argus_live.mtf_bar_router import MTFBarCloseRouter
    from strategies.mtf_confluence import MTFConfluenceFilter

    mtf_filter = MTFConfluenceFilter(timeframes=["15m", "1h", "4h"])
    router = MTFBarCloseRouter(
        mtf_filter=mtf_filter,
        exchanges=exchanges,          # dict of ccxt exchange instances
        symbols=config.trading_pairs,
        timeframes=["15m", "1h", "4h"],
        seed_candles=100,
    )
    # Start alongside trading loop:
    asyncio.create_task(router.run())
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Seconds between REST fallback polls per (symbol, timeframe) pair.
_POLL_INTERVALS_S: Dict[str, float] = {
    "1m": 62.0,
    "3m": 185.0,
    "5m": 305.0,
    "15m": 910.0,
    "30m": 1810.0,
    "1h": 3610.0,
    "2h": 7210.0,
    "4h": 14410.0,
    "6h": 21610.0,
    "8h": 28810.0,
    "12h": 43210.0,
    "1d": 86410.0,
}


def _tf_ms(tf: str) -> int:
    """Return timeframe period in milliseconds."""
    _S = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
        "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
    }
    return _S.get(tf, 3600) * 1000


class MTFBarCloseRouter:
    """
    Routes confirmed closed OHLCV bars to the MTFConfluenceFilter cache.

    Attributes:
        mtf_filter:       MTFConfluenceFilter instance to feed.
        exchanges:        Dict of exchange_name -> ccxt exchange instance.
        symbols:          List of symbols to subscribe (e.g. ["BTC/USDT"]).
        timeframes:       List of timeframe labels (must match filter.timeframes).
        seed_candles:     How many historical candles to pre-load per (symbol, tf).
        ws_reconnect_s:   WS reconnect backoff in seconds.
        primary_exchange: Exchange to use for seeding / WS (defaults to first in exchanges).
    """

    def __init__(
        self,
        mtf_filter: Any,  # MTFConfluenceFilter — typed Any to avoid circular import
        exchanges: Dict[str, Any],
        symbols: List[str],
        timeframes: List[str],
        seed_candles: int = 100,
        ws_reconnect_s: float = 5.0,
        primary_exchange: Optional[str] = None,
    ):
        self.filter = mtf_filter
        self.exchanges = exchanges
        self.symbols = list(symbols)
        self.timeframes = list(timeframes)
        self.seed_candles = max(1, int(seed_candles))
        self.ws_reconnect_s = max(1.0, float(ws_reconnect_s))
        self.primary_exchange = primary_exchange or (next(iter(exchanges)) if exchanges else "")

        # Track last seen candle open-ts per (symbol, tf) to dedup
        self._last_open_ts: Dict[Tuple[str, str], int] = defaultdict(int)
        # Track last REST poll time per (symbol, tf)
        self._last_poll_ts: Dict[Tuple[str, str], float] = defaultdict(float)
        self._seeded: Set[Tuple[str, str]] = set()
        self._running = False
        self._ws_tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main coroutine: seed buffers, then run WS + polling loops.
        Call as ``asyncio.create_task(router.run())``.
        """
        self._running = True
        logger.info("MTFBarCloseRouter: starting (symbols=%s tfs=%s)",
                    self.symbols, self.timeframes)
        # Seed REST on startup
        await self._seed_all()

        # Prefer WS; fall back to polling per pair
        exchange = self.exchanges.get(self.primary_exchange)
        ws_available = self._has_ws(exchange)

        if ws_available:
            logger.info("MTFBarCloseRouter: WS available — using watch_ohlcv")
            tasks = []
            for symbol in self.symbols:
                for tf in self.timeframes:
                    tasks.append(asyncio.create_task(
                        self._ws_loop(symbol, tf)
                    ))
            self._ws_tasks = tasks
            # Also run the REST fallback for timeframes where WS may stall
            tasks.append(asyncio.create_task(self._poll_loop()))
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.info("MTFBarCloseRouter: no WS — using REST poll fallback")
            await self._poll_loop()

    def stop(self) -> None:
        """Signal the router to stop after the current iteration."""
        self._running = False
        for t in self._ws_tasks:
            t.cancel()

    # ------------------------------------------------------------------
    # WS stream loop
    # ------------------------------------------------------------------

    async def _ws_loop(self, symbol: str, tf: str) -> None:
        """
        Watch a single (symbol, tf) pair via CCXT Pro watch_ohlcv.
        Reconnects on error with exponential backoff.
        """
        exchange = self.exchanges.get(self.primary_exchange)
        if exchange is None:
            return
        backoff = self.ws_reconnect_s
        while self._running:
            try:
                ohlcv_list = await exchange.watch_ohlcv(symbol, tf)
                for candle in (ohlcv_list or []):
                    await self._handle_candle(symbol, tf, candle, source="ws")
                backoff = self.ws_reconnect_s  # reset on success
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "MTFBarCloseRouter WS[%s/%s] error: %s — reconnect in %.1fs",
                    symbol, tf, exc, backoff,
                )
                await asyncio.sleep(min(backoff, 60.0))
                backoff = min(backoff * 2.0, 120.0)

    # ------------------------------------------------------------------
    # REST poll fallback
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """
        Periodically fetch closed candles via REST for all (symbol, tf) pairs.
        Only fires when the timeframe period has elapsed since last poll.
        """
        while self._running:
            now = time.monotonic()
            for symbol in self.symbols:
                for tf in self.timeframes:
                    interval = _POLL_INTERVALS_S.get(tf, 3700.0)
                    last = self._last_poll_ts[(symbol, tf)]
                    if now - last < interval:
                        continue
                    await self._rest_fetch_and_feed(symbol, tf, limit=3)
                    self._last_poll_ts[(symbol, tf)] = time.monotonic()
            await asyncio.sleep(5.0)  # check every 5 s

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    async def _seed_all(self) -> None:
        """Pre-load historical closed candles for all (symbol, tf) pairs."""
        for symbol in self.symbols:
            for tf in self.timeframes:
                if (symbol, tf) in self._seeded:
                    continue
                try:
                    await self._rest_fetch_and_feed(symbol, tf, limit=self.seed_candles)
                    self._seeded.add((symbol, tf))
                    logger.info(
                        "MTFBarCloseRouter: seeded %s[%s] with %d candles",
                        symbol, tf, self.seed_candles,
                    )
                except Exception as exc:
                    logger.warning(
                        "MTFBarCloseRouter: seed failed %s[%s]: %s",
                        symbol, tf, exc,
                    )
                await asyncio.sleep(0.1)  # gentle rate limit

    # ------------------------------------------------------------------
    # Core: fetch + feed
    # ------------------------------------------------------------------

    async def _rest_fetch_and_feed(self, symbol: str, tf: str, limit: int = 3) -> None:
        """
        Fetch ``limit`` most recent closed candles via REST and feed to filter.
        The current (live) candle is always excluded by checking open_ts.
        """
        exchange = self.exchanges.get(self.primary_exchange)
        if exchange is None:
            return
        fetch_ohlcv = getattr(exchange, "fetch_ohlcv", None)
        if not callable(fetch_ohlcv):
            return
        try:
            candles = await fetch_ohlcv(symbol, tf, limit=limit + 1)
        except Exception as exc:
            logger.debug("MTFBarCloseRouter REST fetch %s[%s]: %s", symbol, tf, exc)
            return

        if not candles:
            return

        # Exclude the last candle if it's still open (open_ts + period > now)
        now_ms = int(time.time() * 1000)
        period_ms = _tf_ms(tf)
        # candles is list of [open_ts_ms, open, high, low, close, volume]
        # A candle is closed if open_ts_ms + period_ms <= now_ms (with 2s grace)
        closed = [
            c for c in candles
            if isinstance(c, (list, tuple)) and len(c) >= 6
            and int(c[0]) + period_ms <= now_ms + 2000
        ]
        for candle in closed:
            await self._handle_candle(symbol, tf, candle, source="rest")

    async def _handle_candle(self, symbol: str, tf: str,
                             candle: Any, source: str = "ws") -> None:
        """
        Process a single OHLCV candle array.

        candle format: [open_ts_ms, open, high, low, close, volume]

        Only feeds the filter if:
          1. The candle open_ts has not been seen before (dedup).
          2. The candle is confirmed closed (open_ts + period <= now + 2s grace).
        """
        try:
            if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                return
            open_ts_ms = int(candle[0])
            close_price = float(candle[4])
            volume = float(candle[5])
            open_price = float(candle[1])
            high = float(candle[2])
            low = float(candle[3])
        except (TypeError, ValueError, IndexError):
            return

        # Dedup
        last = self._last_open_ts[(symbol, tf)]
        if open_ts_ms <= last:
            return

        # Closed-candle confirmation
        now_ms = int(time.time() * 1000)
        period_ms = _tf_ms(tf)
        if open_ts_ms + period_ms > now_ms + 2000:
            # Still live — do not feed
            return

        self._last_open_ts[(symbol, tf)] = open_ts_ms
        close_ts_utc = datetime.fromtimestamp(open_ts_ms / 1000.0 + period_ms / 1000.0,
                                              tz=timezone.utc)

        self.filter.feed_closed_candle(
            symbol=symbol,
            timeframe=tf,
            close=close_price,
            close_ts=close_ts_utc,
            open=open_price,
            high=high,
            low=low,
            volume=volume,
        )
        logger.debug(
            "MTFBarCloseRouter[%s]: fed %s[%s] close=%.6f ts=%s src=%s",
            source, symbol, tf, close_price, close_ts_utc.isoformat(), source,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_ws(exchange: Any) -> bool:
        """Return True if exchange instance supports watch_ohlcv (CCXT Pro)."""
        return callable(getattr(exchange, "watch_ohlcv", None))
