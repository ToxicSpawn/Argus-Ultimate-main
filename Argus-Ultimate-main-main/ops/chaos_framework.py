#!/usr/bin/env python3
"""
Chaos engineering framework for ARGUS.

Injects controlled faults (latency, disconnects, stale data, order rejections)
to validate system resilience.  Only active when ``chaos.enabled: true``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Any, Set

logger = logging.getLogger(__name__)


class ChaosInjector:
    """Probabilistic fault injector for resilience testing."""

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._enabled: bool = bool(_cfg(config, "chaos.enabled", False))

        # Probabilities
        self._latency_prob: float = float(
            _cfg(config, "chaos.latency_probability", 0.05)
        )
        self._latency_min_ms: int = int(
            _cfg(config, "chaos.latency_min_ms", 50)
        )
        self._latency_max_ms: int = int(
            _cfg(config, "chaos.latency_max_ms", 500)
        )
        self._conn_drop_prob: float = float(
            _cfg(config, "chaos.connection_drop_probability", 0.01)
        )
        self._conn_drop_dur: float = float(
            _cfg(config, "chaos.connection_drop_duration_s", 5)
        )
        self._stale_prob: float = float(
            _cfg(config, "chaos.stale_data_probability", 0.02)
        )
        self._stale_dur: float = float(
            _cfg(config, "chaos.stale_data_duration_s", 30)
        )
        self._reject_prob: float = float(
            _cfg(config, "chaos.order_rejection_probability", 0.03)
        )

        # Thread-safe stale symbol tracking
        self._lock = threading.Lock()
        self._stale_symbols: dict[str, float] = {}  # symbol -> expiry timestamp

        if self._enabled:
            logger.warning("ChaosInjector ENABLED — faults will be injected")
        else:
            logger.info("ChaosInjector disabled (chaos.enabled=false)")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Probabilistic checks
    # ------------------------------------------------------------------

    def should_inject(self, fault_type: str) -> bool:
        """Return True if the given fault should fire this time."""
        if not self._enabled:
            return False

        prob_map = {
            "latency": self._latency_prob,
            "connection_drop": self._conn_drop_prob,
            "stale_data": self._stale_prob,
            "order_rejection": self._reject_prob,
        }
        prob = prob_map.get(fault_type, 0.0)
        return random.random() < prob

    # ------------------------------------------------------------------
    # Fault implementations
    # ------------------------------------------------------------------

    async def inject_latency(self) -> None:
        """Await a random sleep if latency injection fires."""
        if not self.should_inject("latency"):
            return
        delay_ms = random.randint(self._latency_min_ms, self._latency_max_ms)
        logger.info("CHAOS: injecting %d ms latency", delay_ms)
        await asyncio.sleep(delay_ms / 1000.0)

    def inject_connection_drop(self) -> bool:
        """Return True if a connection drop should be simulated right now."""
        if not self.should_inject("connection_drop"):
            return False
        logger.info(
            "CHAOS: simulating connection drop for %.1fs", self._conn_drop_dur
        )
        return True

    @property
    def connection_drop_duration_s(self) -> float:
        return self._conn_drop_dur

    def inject_stale_data(self, symbol: str | None = None) -> bool:
        """Mark a symbol as stale (frozen price) if the fault fires."""
        if not self.should_inject("stale_data"):
            return False
        target = symbol or "BTC/AUD"
        expiry = time.time() + self._stale_dur
        with self._lock:
            self._stale_symbols[target] = expiry
        logger.info("CHAOS: freezing price for %s until %.0f", target, expiry)
        return True

    def inject_order_rejection(self) -> bool:
        """Return True if the next order should be rejected."""
        if not self.should_inject("order_rejection"):
            return False
        logger.info("CHAOS: simulating order rejection")
        return True

    # ------------------------------------------------------------------
    # Stale data queries
    # ------------------------------------------------------------------

    def get_stale_symbols(self) -> Set[str]:
        """Return the set of symbols currently marked as stale."""
        now = time.time()
        with self._lock:
            # Purge expired entries
            expired = [s for s, exp in self._stale_symbols.items() if exp <= now]
            for s in expired:
                del self._stale_symbols[s]
            return set(self._stale_symbols.keys())

    def is_stale(self, symbol: str) -> bool:
        """Check if a specific symbol is currently stale."""
        return symbol in self.get_stale_symbols()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(config: Any, dotted_key: str, default: Any) -> Any:
    if config is None:
        return default
    parts = dotted_key.split(".")
    obj = config
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj
