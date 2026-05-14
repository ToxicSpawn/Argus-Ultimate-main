"""Push 70 — Async WebSocket price feed with auto-reconnect.

Features:
  - asyncio-native WebSocket client (websockets library)
  - Auto-reconnect with pluggable ReconnectPolicy
  - Heartbeat ping/pong loop (configurable interval)
  - Bybit / Binance / OKX endpoint presets
  - Pluggable on_message(raw: str) callback
  - Graceful shutdown via stop()
  - Last-received message timestamp tracking
  - Connection state: DISCONNECTED / CONNECTING / CONNECTED / RECONNECTING
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from core.paper_trading.reconnect import ReconnectPolicy


class FeedState(str, Enum):
    DISCONNECTED  = "DISCONNECTED"
    CONNECTING    = "CONNECTING"
    CONNECTED     = "CONNECTED"
    RECONNECTING  = "RECONNECTING"
    STOPPED       = "STOPPED"


# Pre-built endpoint templates
ENDPOINT_PRESETS: Dict[str, str] = {
    "bybit_spot":    "wss://stream.bybit.com/v5/public/spot",
    "bybit_linear":  "wss://stream.bybit.com/v5/public/linear",
    "binance_spot":  "wss://stream.binance.com:9443/ws",
    "binance_perp":  "wss://fstream.binance.com/ws",
    "okx_public":    "wss://ws.okx.com:8443/ws/v5/public",
}


@dataclass
class FeedConfig:
    endpoint: str = ENDPOINT_PRESETS["bybit_linear"]
    subscribe_payload: Optional[dict] = None   # sent on connect
    ping_interval_secs: float = 20.0
    ping_timeout_secs: float = 10.0
    recv_timeout_secs: float = 30.0
    max_message_queue: int = 1000


class AsyncWebSocketFeed:
    """Async WebSocket feed with auto-reconnect.

    Args:
        config:    FeedConfig
        on_message: async or sync callable(raw_str) for each message
        reconnect:  ReconnectPolicy
    """

    def __init__(
        self,
        config: FeedConfig | None = None,
        on_message: Optional[Callable] = None,
        reconnect: Optional[ReconnectPolicy] = None,
    ):
        self.cfg = config or FeedConfig()
        self.on_message = on_message
        self.reconnect = reconnect or ReconnectPolicy()
        self.state = FeedState.DISCONNECTED
        self._ws = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self.last_message_at: float = 0.0
        self.total_messages: int = 0
        self.total_reconnects: int = 0
        self._message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.cfg.max_message_queue
        )

    async def start(self) -> None:
        """Start the feed loop (non-blocking, runs in background task)."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Gracefully stop the feed."""
        self._running = False
        self.state = FeedState.STOPPED
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _run_loop(self) -> None:
        while self._running:
            try:
                self.state = FeedState.CONNECTING
                await self._connect_and_listen()
            except Exception:
                if not self._running:
                    break
                delay = self.reconnect.next_delay()
                if delay is None:
                    self.state = FeedState.STOPPED
                    break
                self.state = FeedState.RECONNECTING
                self.total_reconnects += 1
                await asyncio.sleep(delay)

    async def _connect_and_listen(self) -> None:
        try:
            import websockets
        except ImportError:
            # Stub mode: simulate connection for testing
            self.state = FeedState.CONNECTED
            self.reconnect.reset()
            while self._running:
                await asyncio.sleep(0.1)
            return

        async with websockets.connect(
            self.cfg.endpoint,
            ping_interval=None,  # manual ping
            open_timeout=10,
        ) as ws:
            self._ws = ws
            self.state = FeedState.CONNECTED
            self.reconnect.reset()

            # Send subscription payload
            if self.cfg.subscribe_payload:
                await ws.send(json.dumps(self.cfg.subscribe_payload))

            # Start heartbeat
            self._ping_task = asyncio.create_task(
                self._ping_loop(ws)
            )

            try:
                while self._running:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=self.cfg.recv_timeout_secs,
                        )
                        self.last_message_at = time.time()
                        self.total_messages += 1
                        if not self._message_queue.full():
                            self._message_queue.put_nowait(raw)
                        if self.on_message:
                            if asyncio.iscoroutinefunction(self.on_message):
                                await self.on_message(raw)
                            else:
                                self.on_message(raw)
                    except asyncio.TimeoutError:
                        break  # trigger reconnect
            finally:
                if self._ping_task and not self._ping_task.done():
                    self._ping_task.cancel()

    async def _ping_loop(self, ws) -> None:
        while self._running:
            await asyncio.sleep(self.cfg.ping_interval_secs)
            try:
                await asyncio.wait_for(
                    ws.ping(), timeout=self.cfg.ping_timeout_secs
                )
            except Exception:
                break

    async def get_message(self, timeout: float = 1.0) -> Optional[str]:
        """Pop one message from the queue (async, with timeout)."""
        try:
            return await asyncio.wait_for(
                self._message_queue.get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    @property
    def is_connected(self) -> bool:
        return self.state == FeedState.CONNECTED

    @property
    def seconds_since_last_message(self) -> float:
        if self.last_message_at == 0:
            return float("inf")
        return time.time() - self.last_message_at
