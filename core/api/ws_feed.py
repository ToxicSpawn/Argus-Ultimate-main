"""Push 79 — WebSocket ConnectionManager + feed handlers.

Endpoints (mounted by app.py):
  WS /ws/prices   — real-time price tick broadcast
  WS /ws/signals  — Signal JSON broadcast (from SignalBus)
  WS /ws/risk     — RiskEvent JSON broadcast

ConnectionManager:
  - Tracks active WebSocket connections per channel
  - Handles disconnect gracefully (no crash on stale sockets)
  - broadcast(channel, data) fans out to all subscribers
  - get_connection_count(channel) for monitoring
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

try:
    from fastapi import WebSocket
except ImportError:  # pragma: no cover
    WebSocket = Any  # type: ignore


class ConnectionManager:
    """Manages WebSocket connections across named channels."""

    def __init__(self):
        self._connections: Dict[str, Set] = defaultdict(set)
        self._message_counts: Dict[str, int] = defaultdict(int)

    async def connect(self, channel: str, ws: Any) -> None:
        await ws.accept()
        self._connections[channel].add(ws)

    async def disconnect(self, channel: str, ws: Any) -> None:
        self._connections[channel].discard(ws)

    async def broadcast(self, channel: str, data: Any) -> int:
        """Broadcast data to all connections in channel.
        Returns number of successful deliveries."""
        if isinstance(data, dict):
            payload = json.dumps(data)
        elif isinstance(data, str):
            payload = data
        else:
            payload = json.dumps(str(data))

        delivered = 0
        dead: List[Any] = []
        for ws in list(self._connections[channel]):
            try:
                await ws.send_text(payload)
                delivered += 1
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._connections[channel].discard(ws)

        self._message_counts[channel] += delivered
        return delivered

    def get_connection_count(self, channel: str) -> int:
        return len(self._connections[channel])

    @property
    def stats(self) -> dict:
        return {
            "channels":   list(self._connections.keys()),
            "connections": {ch: len(conns) for ch, conns in self._connections.items()},
            "messages_sent": dict(self._message_counts),
        }


# ---------------------------------------------------------------------------
# Feed helpers — called from app.py route handlers
# ---------------------------------------------------------------------------

def signal_to_ws_payload(signal: Any) -> dict:
    """Convert a Signal to a JSON-serialisable dict for WS broadcast."""
    return {
        "type":        "signal",
        "symbol":      signal.symbol,
        "side":        signal.side.value if hasattr(signal.side, "value") else str(signal.side),
        "strength":    round(signal.strength, 4),
        "strategy_id": signal.strategy_id,
        "timestamp":   signal.timestamp,
        "ts_iso":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(signal.timestamp)),
    }


def risk_event_to_ws_payload(event: Any) -> dict:
    """Convert a RiskEvent to a JSON-serialisable dict."""
    return {
        "type":       "risk_event",
        "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
        "message":    event.message,
        "symbol":     event.symbol,
        "value":      event.value,
        "threshold":  event.threshold,
        "timestamp":  event.timestamp,
    }


def price_tick_to_ws_payload(symbol: str, price: float) -> dict:
    return {
        "type":      "price",
        "symbol":    symbol,
        "price":     price,
        "timestamp": time.time(),
    }
