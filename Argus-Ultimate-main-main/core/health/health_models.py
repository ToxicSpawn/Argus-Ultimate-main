"""Health models — Push 62."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


class HealthStatus(IntEnum):
    HEALTHY = 0
    DEGRADED = 1
    UNHEALTHY = 2

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def ok(self) -> bool:
        return self != HealthStatus.UNHEALTHY


@dataclass
class ComponentHealth:
    """Health state of a single system component."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_checked: float = field(default_factory=time.time)
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.label,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "last_checked": self.last_checked,
            "extra": self.extra,
        }


@dataclass
class SystemHealth:
    """Aggregated health state of the entire Argus system."""
    overall: HealthStatus
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    uptime_s: float = 0.0
    version: str = "7.8.0"
    env: str = "production"
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "status": self.overall.label,
            "ok": self.overall.ok,
            "version": self.version,
            "env": self.env,
            "uptime_s": round(self.uptime_s, 1),
            "ts": self.ts,
            "components": {
                name: ch.to_dict()
                for name, ch in self.components.items()
            },
        }

    @property
    def is_ready(self) -> bool:
        """Ready if not fully UNHEALTHY."""
        return self.overall != HealthStatus.UNHEALTHY

    @property
    def is_live(self) -> bool:
        """Always True — process is running."""
        return True
