"""
ws_feed_base.py
---------------
Abstract async WebSocket feed base.

Features:
  - Auto-reconnect with exponential backoff + full jitter (capped at 60 s)
  - Heartbeat / ping watchdog: fires reconnect if no pong within timeout
  - Per-message inbound latency tracking (exchange_ts vs local monotonic)
  - Prometheus counters via optional PrometheusEmitter
  - Subclasses implement: _subscribe(), _handle_message(), _ping_payload()
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class FeedState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    STOPPED = auto()


@dataclass
class FeedStats:
    messages_received: int = 0
    reconnects: int = 0
    last_message_ts: float = 0.0          # monotonic
    latency_sum_ms: float = 0.0
    latency_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_sum_ms / self.latency_count if self.latency_count else 0.0


class WSFeedBase(ABC):
    """
    Abstract WebSocket feed.

    Parameters
    ----------
    url : str
        WebSocket endpoint URL.
    venue : str
        Human-readable venue name (bybit / binance / okx …).
    ping_interval : float
        Seconds between pings.
    pong_timeout : float
        Seconds to wait for a pong before declaring the connection stale.
    max_backoff : float
        Maximum reconnect backoff in seconds.
    on_message : optional async callback(venue, raw_msg)
    emitter : optional PrometheusEmitter-compatible object
    """

    def __init__(
        self,
        url: str,
        venue: str,
        ping_interval: float = 20.0,
        pong_timeout: float = 10.0,
        max_backoff: float = 60.0,
        on_message: Optional[Callable[[str, Any], Coroutine]] = None,
        emitter: Optional[Any] = None,
    ) -> None:
        self.url = url
        self.venue = venue
        self.ping_interval = ping_interval
        self.pong_timeout = pong_timeout
        self.max_backoff = max_backoff
        self._on_message = on_message
        self._emitter = emitter

        self.state: FeedState = FeedState.DISCONNECTED
        self.stats: FeedStats = FeedStats()
        self._ws: Any = None                    # aiohttp / websockets ws object
        self._stop_event = asyncio.Event()
        self._pong_event = asyncio.Event()
        self._attempt: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin connect-loop in background; returns immediately."""
        self._stop_event.clear()
        asyncio.ensure_future(self._connect_loop())

    async def stop(self) -> None:
        """Graceful shutdown."""
        self.state = FeedState.STOPPED
        self._stop_event.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def _subscribe(self) -> None:
        """Send subscription frames after connection is established."""

    @abstractmethod
    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch one raw message string."""

    def _ping_payload(self) -> Optional[str]:
        """Return a JSON ping string, or None to use native WS ping."""
        return None

    # ------------------------------------------------------------------
    # Connect loop
    # ------------------------------------------------------------------

    async def _connect_loop(self) -> None:
        while not self._stop_event.is_set():
            self.state = FeedState.CONNECTING
            try:
                await self._run_session()
            except Exception as exc:
                if self.state == FeedState.STOPPED:
                    return
                logger.warning("%s feed error: %s", self.venue, exc)
                self._inc("feed_error")

            if self._stop_event.is_set():
                return

            self.state = FeedState.RECONNECTING
            self.stats.reconnects += 1
            backoff = min(self.max_backoff, (2 ** self._attempt)) * (0.5 + random.random() * 0.5)
            self._attempt = min(self._attempt + 1, 10)
            logger.info("%s reconnecting in %.1f s (attempt %d)", self.venue, backoff, self._attempt)
            self._inc("feed_reconnect")
            await asyncio.sleep(backoff)

    async def _run_session(self) -> None:
        """Establish WS, subscribe, receive loop + ping watchdog."""
        # Dynamic import so the base class has no hard dependency.
        try:
            import aiohttp
            session = aiohttp.ClientSession()
            ws_ctx = session.ws_connect(self.url)
        except ImportError:
            raise RuntimeError("aiohttp not installed; install it to use WSFeedBase")

        async with ws_ctx as ws:
            self._ws = ws
            self.state = FeedState.CONNECTED
            self._attempt = 0
            logger.info("%s connected to %s", self.venue, self.url)
            await self._subscribe()

            ping_task = asyncio.ensure_future(self._ping_loop(ws))
            try:
                async for msg in ws:
                    import aiohttp as _aio
                    if msg.type == _aio.WSMsgType.TEXT:
                        self.stats.messages_received += 1
                        self.stats.last_message_ts = time.monotonic()
                        self._pong_event.set()          # any message resets watchdog
                        self._inc("feed_message")
                        try:
                            await self._handle_message(msg.data)
                            if self._on_message:
                                await self._on_message(self.venue, msg.data)
                        except Exception as exc:
                            logger.debug("%s message handler error: %s", self.venue, exc)
                    elif msg.type in (_aio.WSMsgType.CLOSED, _aio.WSMsgType.ERROR):
                        break
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
                await session.close()

    async def _ping_loop(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(self.ping_interval)
            self._pong_event.clear()
            payload = self._ping_payload()
            try:
                if payload:
                    await ws.send_str(payload)
                else:
                    await ws.ping()
            except Exception:
                return

            try:
                await asyncio.wait_for(self._pong_event.wait(), timeout=self.pong_timeout)
            except asyncio.TimeoutError:
                logger.warning("%s pong timeout — forcing reconnect", self.venue)
                self._inc("feed_pong_timeout")
                await ws.close()
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _inc(self, metric: str) -> None:
        if self._emitter and hasattr(self._emitter, "inc_counter"):
            try:
                self._emitter.inc_counter(metric, labels={"venue": self.venue})
            except Exception:
                pass

    def record_latency(self, exchange_ts_ms: float) -> None:
        """Call with exchange-reported timestamp (epoch ms) to track latency."""
        delta = (time.time() * 1000) - exchange_ts_ms
        self.stats.latency_sum_ms += delta
        self.stats.latency_count += 1

    @property
    def is_healthy(self) -> bool:
        return self.state == FeedState.CONNECTED
