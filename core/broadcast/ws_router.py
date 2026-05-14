"""FastAPI WebSocket router for live dashboard feed — Push 57.

Endpoints::

    WS  /ws/dashboard         — live WebSocket stream
    GET /ws/dashboard/status  — hub stats (client count, broadcasts)

On connect, the client receives an initial full snapshot
(MessageType.CONNECTED) before the live stream begins.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[assignment,misc]

from core.broadcast.ws_hub import WsHub
from core.broadcast.ws_message import MessageType, WsMessage
from core.broadcast.dashboard_feed import DashboardFeed

logger = logging.getLogger(__name__)

# Module-level singletons (replaced at app startup)
_hub: Optional[WsHub] = None
_feed: Optional[DashboardFeed] = None


def get_hub() -> WsHub:
    global _hub
    if _hub is None:
        _hub = WsHub()
    return _hub


def get_feed() -> Optional[DashboardFeed]:
    return _feed


def set_hub_and_feed(hub: WsHub, feed: DashboardFeed) -> None:
    """Inject pre-configured hub and feed at app startup."""
    global _hub, _feed
    _hub = hub
    _feed = feed


if _FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/ws", tags=["dashboard"])

    @router.websocket("/dashboard")
    async def dashboard_ws(websocket: WebSocket) -> None:
        """Live dashboard WebSocket stream."""
        hub = get_hub()
        await websocket.accept()
        await hub.register(websocket)
        try:
            # Send initial full snapshot on connect
            feed = get_feed()
            if feed is not None:
                snapshot = feed.full_snapshot()
            else:
                snapshot = hub.status()
            await hub.send_to(
                websocket,
                WsMessage(type=MessageType.CONNECTED, data=snapshot),
            )
            # Keep connection alive — client drives keep-alive via ping
            while True:
                await websocket.receive_text()  # absorb any client messages
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("WS dashboard: client error: %s", exc)
        finally:
            await hub.unregister(websocket)

    @router.get("/dashboard/status")
    async def dashboard_status() -> Dict[str, Any]:
        """Return WsHub stats."""
        return get_hub().status()

else:
    router = None  # type: ignore[assignment]
