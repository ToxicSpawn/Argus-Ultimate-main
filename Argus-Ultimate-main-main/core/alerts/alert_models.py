"""AlertEvent dataclass and AlertLevel enum — Push 60."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


class AlertLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @property
    def emoji(self) -> str:
        return {
            10: "🔍",
            20: "ℹ️",
            30: "⚠️",
            40: "🔴",
            50: "🚨",
        }.get(self.value, "📢")


@dataclass
class AlertEvent:
    """A single alert event dispatched through the AlertManager."""
    level: AlertLevel
    title: str
    body: str = ""
    symbol: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    source: str = "argus"
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level.name,
            "title": self.title,
            "body": self.body,
            "symbol": self.symbol,
            "tags": self.tags,
            "ts": self.ts,
            "source": self.source,
            "extra": self.extra,
        }

    def formatted_text(self, include_emoji: bool = True) -> str:
        """Human-readable single-string representation."""
        prefix = f"{self.level.emoji} " if include_emoji else ""
        parts = [f"{prefix}[{self.level.label}] {self.title}"]
        if self.symbol:
            parts.append(f"Symbol: {self.symbol}")
        if self.body:
            parts.append(self.body)
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(parts)

    @classmethod
    def info(cls, title: str, body: str = "", **kwargs) -> "AlertEvent":
        return cls(level=AlertLevel.INFO, title=title, body=body, **kwargs)

    @classmethod
    def warning(cls, title: str, body: str = "", **kwargs) -> "AlertEvent":
        return cls(level=AlertLevel.WARNING, title=title, body=body, **kwargs)

    @classmethod
    def error(cls, title: str, body: str = "", **kwargs) -> "AlertEvent":
        return cls(level=AlertLevel.ERROR, title=title, body=body, **kwargs)

    @classmethod
    def critical(cls, title: str, body: str = "", **kwargs) -> "AlertEvent":
        return cls(level=AlertLevel.CRITICAL, title=title, body=body, **kwargs)
