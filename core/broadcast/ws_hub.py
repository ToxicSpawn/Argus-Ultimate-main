"""WsHub — async WebSocket broadcast hub — Push 57.

Manages a set of connected WebSocket clients and fans out messages
to all of them. Dead clients (closed connections) are removed on
next broadcast attempt.

Usage::

    hub = WsHub()
    await hub.register(websocket)       # called from FastAPI endpoint
    await hub.broadcast(msg)            # fans out to all clients
    await hub.unregister(websocket)

Prometheus gauge::

    argus_ws_clients_connected   — number of live WS clients
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Set

from core.broadcast.ws_message import WsMessage, MessageType

try:
    from prometheus_client import Gauge
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

logger = logging.getLogger(__name__)

if _PROM_AVAILABLE:
    _G_CLIENTS = Gauge("argus_ws_clients_connected", "Live WebSocket clients")
else:
    _G_CLIENTS = None


class WsHub:
    """Async WebSocket broadcast hub."""

    def __init__(self) -> None:
        self._clients: Set[Any] = set()
        self._lock = asyncio.Lock()
        self._broadcast_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    async def register(self, ws: Any) -> None:
        """Register a new WebSocket client."""
        async with self._lock:
            self._clients.add(ws)
        self._update_gauge()
        logger.info("WsHub: client connected (total=%d)", len(self._clients))

    async def unregister(self, ws: Any) -> None:
        """Remove a WebSocket client."""
        async with self._lock:
            self._clients.discard(ws)
        self._update_gauge()
        logger.info("WsHub: client disconnected (total=%d)", len(self._clients))

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    async def broadcast(self, msg: WsMessage) -> int:
        """Fan out a message to all connected clients.

        Returns the number of clients that received the message.
        Dead clients are automatically removed.
        """
        if not self._clients:
            return 0

        payload = msg.to_json()
        dead: list = []
        sent = 0

        async with self._lock:
            clients_snapshot = list(self._clients)

        for ws in clients_snapshot:
            try:
                await ws.send_text(payload)
                sent += 1
            except Exception:  # noqa: BLE001
                dead.append(ws)
                self._error_count += 1

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)
            self._update_gauge()
            logger.debug("WsHub: removed %d dead clients", len(dead))

        self._broadcast_count += 1
        return sent

    async def broadcast_json(self, data: dict, msg_type: MessageType) -> int:
        """Convenience wrapper — builds WsMessage and broadcasts."""
        return await self.broadcast(WsMessage(type=msg_type, data=data))

    async def send_to(self, ws: Any, msg: WsMessage) -> bool:
        """Send a message to a single client."""
        try:
            await ws.send_text(msg.to_json())
            return True
        except Exception:  # noqa: BLE001
            await self.unregister(ws)
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def broadcast_count(self) -> int:
        return self._broadcast_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def status(self) -> dict:
        return {
            "clients": self.client_count,
            "broadcasts": self._broadcast_count,
            "errors": self._error_count,
        }

    def _update_gauge(self) -> None:
        if _G_CLIENTS:
            _G_CLIENTS.set(len(self._clients))
