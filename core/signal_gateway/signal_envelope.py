"""SignalEnvelope — typed container for every inbound trading signal."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.signal_gateway.signal_source import SignalSource


@dataclass
class SignalEnvelope:
    """Immutable signal envelope passed through the gateway pipeline.

    Attributes
    ----------
    source:       Originating signal source enum.
    direction:    "long", "short", or "flat".
    confidence:   Float in [0.0, 1.0].
    timestamp_ns: Wall-clock nanoseconds at signal creation (time.time_ns()).
    metadata:     Arbitrary extra fields (e.g. symbol, regime, logits).
    ttl_ms:       Time-to-live in milliseconds. 0 = never expires.
    """

    source: SignalSource
    direction: str  # "long" | "short" | "flat"
    confidence: float
    timestamp_ns: int = field(default_factory=time.time_ns)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl_ms: int = 500

    # ------------------------------------------------------------------
    # Validity helpers
    # ------------------------------------------------------------------

    def is_expired(self, now_ns: Optional[int] = None) -> bool:
        """Return True if the envelope has exceeded its TTL."""
        if self.ttl_ms <= 0:
            return False
        now = now_ns if now_ns is not None else time.time_ns()
        age_ms = (now - self.timestamp_ns) / 1_000_000
        return age_ms > self.ttl_ms

    def age_ms(self, now_ns: Optional[int] = None) -> float:
        """Return envelope age in milliseconds."""
        now = now_ns if now_ns is not None else time.time_ns()
        return (now - self.timestamp_ns) / 1_000_000

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "direction": self.direction,
            "confidence": self.confidence,
            "timestamp_ns": self.timestamp_ns,
            "metadata": self.metadata,
            "ttl_ms": self.ttl_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalEnvelope":
        return cls(
            source=SignalSource(data["source"]),
            direction=data["direction"],
            confidence=float(data["confidence"]),
            timestamp_ns=int(data["timestamp_ns"]),
            metadata=data.get("metadata", {}),
            ttl_ms=int(data.get("ttl_ms", 500)),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SignalEnvelope(source={self.source.name}, "
            f"direction={self.direction!r}, "
            f"confidence={self.confidence:.3f}, "
            f"age_ms={self.age_ms():.1f})"
        )
