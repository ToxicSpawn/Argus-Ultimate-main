"""WsMessage dataclass and MessageType enum — Push 57.

All messages exchanged over the Argus dashboard WebSocket are
serialised as JSON using this schema::

    {
        "type": "pnl_snapshot",
        "ts":   1713300000.123,
        "data": { ... }
    }
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class MessageType(str, Enum):
    TICK = "tick"
    ORDER_UPDATE = "order_update"
    FILL = "fill"
    PNL_SNAPSHOT = "pnl_snapshot"
    RISK_STATUS = "risk_status"
    SESSION_STATS = "session_stats"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    CONNECTED = "connected"


@dataclass
class WsMessage:
    """A single WebSocket broadcast message."""
    type: MessageType
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "ts": self.ts,
            "data": self.data,
        })

    @classmethod
    def from_json(cls, raw: str) -> "WsMessage":
        d = json.loads(raw)
        return cls(
            type=MessageType(d["type"]),
            data=d.get("data", {}),
            ts=d.get("ts", time.time()),
        )

    @classmethod
    def heartbeat(cls) -> "WsMessage":
        return cls(type=MessageType.HEARTBEAT, data={"ping": "pong"})

    @classmethod
    def alert(cls, level: str, message: str) -> "WsMessage":
        return cls(type=MessageType.ALERT, data={"level": level, "message": message})

    @classmethod
    def tick(cls, symbol: str, price: float, bid: float, ask: float) -> "WsMessage":
        return cls(
            type=MessageType.TICK,
            data={"symbol": symbol, "price": price, "bid": bid, "ask": ask},
        )
