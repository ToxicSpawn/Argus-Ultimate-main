"""GatewayConfig — runtime configuration for the Signal Gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from core.signal_gateway.signal_source import SignalSource, DEFAULT_SOURCE_WEIGHTS

_DEFAULT_CONFIG_PATH = Path("config/signal_gateway.json")


@dataclass
class GatewayConfig:
    """Configuration for SignalGateway pipeline.

    Attributes
    ----------
    consensus_threshold: Minimum weighted-vote fraction to fire consensus [0,1].
    min_sources:         Minimum number of distinct sources that must agree.
    ttl_ms:              Default signal TTL injected when envelope.ttl_ms == 0.
    dedup_window_ms:     Rolling window for deduplication fingerprinting.
    source_weights:      Per-source vote weights (overrides DEFAULT_SOURCE_WEIGHTS).
    enabled_sources:     Set of sources accepted by the gateway; None = all.
    """

    consensus_threshold: float = 0.55
    min_sources: int = 2
    ttl_ms: int = 500
    dedup_window_ms: int = 200
    source_weights: Dict[SignalSource, float] = field(
        default_factory=lambda: dict(DEFAULT_SOURCE_WEIGHTS)
    )
    enabled_sources: Optional[list] = None  # None → accept all sources

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_json(self, path: Path = _DEFAULT_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "consensus_threshold": self.consensus_threshold,
            "min_sources": self.min_sources,
            "ttl_ms": self.ttl_ms,
            "dedup_window_ms": self.dedup_window_ms,
            "source_weights": {
                src.value: w for src, w in self.source_weights.items()
            },
            "enabled_sources": (
                [s.value for s in self.enabled_sources]
                if self.enabled_sources
                else None
            ),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load_from_json(
        cls, path: Path = _DEFAULT_CONFIG_PATH
    ) -> "GatewayConfig":
        if not path.exists():
            return cls()  # defaults
        payload = json.loads(path.read_text())
        weights = {
            SignalSource(k): float(v)
            for k, v in payload.get("source_weights", {}).items()
        }
        enabled = (
            [SignalSource(s) for s in payload["enabled_sources"]]
            if payload.get("enabled_sources")
            else None
        )
        return cls(
            consensus_threshold=float(
                payload.get("consensus_threshold", 0.55)
            ),
            min_sources=int(payload.get("min_sources", 2)),
            ttl_ms=int(payload.get("ttl_ms", 500)),
            dedup_window_ms=int(payload.get("dedup_window_ms", 200)),
            source_weights=weights or dict(DEFAULT_SOURCE_WEIGHTS),
            enabled_sources=enabled,
        )
