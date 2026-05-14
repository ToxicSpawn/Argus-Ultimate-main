"""tick_injector.py — Push 42.

Live WebSocket trade-tick bridge: subscribes to the exchange trade feed
and pipes every tick into LiveOFIStream and LiveVPINStream in real time.

Architecture
------------
  TickInjector runs as an asyncio background task.
  It uses ccxt.pro (async) for WebSocket; falls back to REST poll if
  ccxt.pro is unavailable.

  Exchange -> WS trades -> TickInjector.on_tick()
                                  |
                    +-------------+-------------+
                    v                           v
             LiveOFIStream               LiveVPINStream
           (on_trade + book_delta)        (on_trade)

Reconnection
------------
  Exponential back-off: 1s, 2s, 4s, 8s ... capped at 60s.
  Max retries configurable (default unlimited).

Usage
-----
  injector = TickInjector(
      ofi_stream=ofi, vpin_stream=vpin,
      symbol="BTC/USDT", exchange_id="binance",
  )
  await injector.start()   # launches background task
  ...                      # injector feeds ofi/vpin automatically
  await injector.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from alpha.microstructure.live_ofi_stream import LiveOFIStream
from alpha.microstructure.live_vpin_stream import LiveVPINStream

logger = logging.getLogger(__name__)

_DEFAULT_EXCHANGE    = "binance"
_DEFAULT_SYMBOL      = "BTC/USDT"
_REST_POLL_INTERVAL  = 2.0      # seconds between REST fallback polls
_MAX_BACKOFF         = 60.0     # max reconnect delay seconds
_INIT_BACKOFF        = 1.0
_REST_TRADES_LIMIT   = 50       # trades per REST poll


class TickInjector:
    """Streams live trades into OFI and VPIN streams.

    Parameters
    ----------
    ofi_stream   : LiveOFIStream instance
    vpin_stream  : LiveVPINStream instance
    symbol       : Market symbol e.g. 'BTC/USDT'
    exchange_id  : ccxt exchange id e.g. 'binance'
    max_retries  : Max reconnection attempts (None = unlimited)
    """

    def __init__(
        self,
        ofi_stream:  LiveOFIStream,
        vpin_stream: LiveVPINStream,
        symbol:      str = _DEFAULT_SYMBOL,
        exchange_id: str = _DEFAULT_EXCHANGE,
        max_retries: Optional[int] = None,
    ) -> None:
        self._ofi   = ofi_stream
        self._vpin  = vpin_stream
        self._sym   = symbol
        self._exid  = exchange_id
        self._max_retries = max_retries
        self._task:  Optional[asyncio.Task] = None
        self._running = False
        self._tick_count   = 0
        self._error_count  = 0
        self._last_tick_ts = 0.0
        self._last_price   = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the background feed task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._run_with_retry(), name=f"tick_injector_{self._exid}_{self._sym}"
        )
        logger.info("TickInjector started | exchange=%s symbol=%s", self._exid, self._sym)

    async def stop(self) -> None:
        """Cancel the background task and clean up."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._close_exchange()
        logger.info(
            "TickInjector stopped | ticks=%d errors=%d",
            self._tick_count, self._error_count,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    @property
    def last_tick_ts(self) -> float:
        return self._last_tick_ts

    # ------------------------------------------------------------------
    # Core feed loop
    # ------------------------------------------------------------------

    async def _run_with_retry(self) -> None:
        backoff  = _INIT_BACKOFF
        attempts = 0
        while self._running:
            try:
                await self._run_feed()
                backoff = _INIT_BACKOFF   # successful run resets backoff
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._error_count += 1
                attempts += 1
                if self._max_retries is not None and attempts >= self._max_retries:
                    logger.error("TickInjector max retries reached (%d). Stopping.", attempts)
                    self._running = False
                    break
                logger.warning(
                    "TickInjector error (attempt %d): %s. Reconnecting in %.1fs",
                    attempts, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def _run_feed(self) -> None:
        """Try ccxt.pro WS first; fall back to REST poll."""
        try:
            import ccxt.pro as ccxtpro
            exchange = getattr(ccxtpro, self._exid)()
            try:
                await self._ws_feed(exchange)
            finally:
                try:
                    await exchange.close()
                except Exception:
                    pass
        except (ImportError, AttributeError):
            logger.info("ccxt.pro unavailable — using REST poll fallback")
            await self._rest_poll_feed()

    async def _ws_feed(self, exchange) -> None:
        """Consume trades from ccxt.pro WebSocket."""
        logger.info("WS feed active | %s %s", self._exid, self._sym)
        while self._running:
            trades = await exchange.watch_trades(self._sym)
            for trade in trades:
                if not self._running:
                    return
                self._ingest(trade)

    async def _rest_poll_feed(self) -> None:
        """Poll REST fetch_trades as fallback."""
        try:
            import ccxt
            exchange = getattr(ccxt, self._exid)({"enableRateLimit": True})
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(f"ccxt not available: {exc}") from exc

        logger.info("REST poll feed active | %s %s (interval=%.1fs)",
                    self._exid, self._sym, _REST_POLL_INTERVAL)
        seen_ids: set = set()
        try:
            while self._running:
                try:
                    trades = exchange.fetch_trades(
                        self._sym, limit=_REST_TRADES_LIMIT)
                    for trade in trades:
                        tid = trade.get("id") or trade.get("timestamp", 0)
                        if tid in seen_ids:
                            continue
                        seen_ids.add(tid)
                        # Keep seen_ids bounded
                        if len(seen_ids) > 500:
                            seen_ids = set(list(seen_ids)[-250:])
                        self._ingest(trade)
                except Exception as exc:
                    logger.warning("REST poll error: %s", exc)
                await asyncio.sleep(_REST_POLL_INTERVAL)
        finally:
            try:
                exchange.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Tick ingestion
    # ------------------------------------------------------------------

    def _ingest(self, trade: dict) -> None:
        """Normalise a ccxt trade dict and inject into OFI + VPIN."""
        try:
            price  = float(trade.get("price")  or 0.0)
            amount = float(trade.get("amount") or trade.get("size") or 0.0)
            side   = str(trade.get("side") or "").lower()
            if price <= 0 or amount <= 0:
                return

            norm = {"price": price, "amount": amount, "side": side}

            self._ofi.on_trade(norm)
            self._vpin.on_trade(norm)

            # Synthetic book delta from price movement
            if self._last_price > 0:
                delta = price - self._last_price
                self._ofi.on_book_delta({
                    "bid_delta":  delta * 0.5,
                    "ask_delta": -delta * 0.5,
                })
            self._last_price   = price
            self._last_tick_ts = time.time()
            self._tick_count  += 1

        except Exception as exc:
            logger.debug("Tick ingest error: %s", exc)

    _exchange_ref = None

    async def _close_exchange(self) -> None:
        if self._exchange_ref is not None:
            try:
                await self._exchange_ref.close()
            except Exception:
                pass
            self._exchange_ref = None
