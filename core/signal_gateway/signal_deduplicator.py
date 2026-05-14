"""SignalDeduplicator — rolling deduplication window for signal envelopes."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Tuple

from core.signal_gateway.signal_envelope import SignalEnvelope
from core.signal_gateway.gateway_config import GatewayConfig

# Fingerprint: (source_value, direction, confidence_bucket)
_Fingerprint = Tuple[str, str, int]

_CONFIDENCE_BUCKETS = 20  # 5% bucket size → 20 buckets in [0,1]


def _fingerprint(envelope: SignalEnvelope) -> _Fingerprint:
    bucket = int(envelope.confidence * _CONFIDENCE_BUCKETS)
    return (envelope.source.value, envelope.direction, bucket)


class SignalDeduplicator:
    """Blocks duplicate signals within a rolling time window.

    A signal is considered a duplicate if an envelope with the same
    (source, direction, confidence_bucket) fingerprint was seen within
    the last *dedup_window_ms* milliseconds.

    Thread-safe via asyncio.Lock — intended for use inside an async loop.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._window_ms = config.dedup_window_ms
        # Deque of (timestamp_ns, fingerprint)
        self._seen: Deque[Tuple[int, _Fingerprint]] = deque()
        self._lock = asyncio.Lock()
        self._blocked_total: int = 0

    async def is_duplicate(self, envelope: SignalEnvelope) -> bool:
        """Return True if *envelope* is a duplicate within the window."""
        async with self._lock:
            now_ns = time.time_ns()
            cutoff_ns = now_ns - int(self._window_ms * 1_000_000)

            # Evict stale entries.
            while self._seen and self._seen[0][0] < cutoff_ns:
                self._seen.popleft()

            fp = _fingerprint(envelope)
            for _ts, seen_fp in self._seen:
                if seen_fp == fp:
                    self._blocked_total += 1
                    return True

            # Not a duplicate — record it.
            self._seen.append((now_ns, fp))
            return False

    @property
    def blocked_total(self) -> int:
        return self._blocked_total

    def reset(self) -> None:
        """Clear seen fingerprints (for testing / session reset)."""
        self._seen.clear()
        self._blocked_total = 0
