"""Push 71 — Async token-bucket rate limiter for Bybit V5.

Bybit V5 rate limits (default tier):
  Order endpoints:    10 req/s
  Position endpoints: 10 req/s
  Market endpoints:   20 req/s

Token-bucket algorithm:
  - Each category has its own bucket
  - Tokens refill continuously at refill_rate tokens/sec
  - Burst capacity = max_tokens
  - wait() blocks until a token is available
  - try_acquire() returns False immediately if no token
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BucketConfig:
    max_tokens: float = 10.0
    refill_rate: float = 10.0   # tokens per second


DEFAULT_BUCKETS: Dict[str, BucketConfig] = {
    "order":    BucketConfig(max_tokens=10, refill_rate=10),
    "position": BucketConfig(max_tokens=10, refill_rate=10),
    "market":   BucketConfig(max_tokens=20, refill_rate=20),
    "default":  BucketConfig(max_tokens=10, refill_rate=10),
}


class _TokenBucket:
    def __init__(self, max_tokens: float, refill_rate: float):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self._tokens = max_tokens
        self._last_refill = time.monotonic()
        self._total_waits: int = 0
        self._total_acquired: int = 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_tokens,
            self._tokens + elapsed * self.refill_rate,
        )
        self._last_refill = now

    def try_acquire(self) -> bool:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            self._total_acquired += 1
            return True
        return False

    def time_until_token(self) -> float:
        self._refill()
        if self._tokens >= 1.0:
            return 0.0
        return (1.0 - self._tokens) / self.refill_rate

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


class AsyncRateLimiter:
    """Async token-bucket rate limiter with per-category buckets.

    Args:
        buckets: dict of category -> BucketConfig
    """

    def __init__(self, buckets: Dict[str, BucketConfig] | None = None):
        configs = buckets or DEFAULT_BUCKETS
        self._buckets: Dict[str, _TokenBucket] = {
            cat: _TokenBucket(cfg.max_tokens, cfg.refill_rate)
            for cat, cfg in configs.items()
        }
        self._wait_count: int = 0

    async def wait(self, category: str = "default") -> None:
        """Wait until a token is available for category."""
        bucket = self._buckets.get(category) or self._buckets["default"]
        while not bucket.try_acquire():
            wait_secs = bucket.time_until_token()
            self._wait_count += 1
            await asyncio.sleep(max(wait_secs, 0.001))

    def try_acquire(self, category: str = "default") -> bool:
        """Attempt to acquire immediately. Returns False if throttled."""
        bucket = self._buckets.get(category) or self._buckets["default"]
        return bucket.try_acquire()

    def available_tokens(self, category: str = "default") -> float:
        bucket = self._buckets.get(category) or self._buckets["default"]
        return bucket.available

    def add_category(
        self, name: str, max_tokens: float, refill_rate: float
    ) -> None:
        self._buckets[name] = _TokenBucket(max_tokens, refill_rate)

    @property
    def total_waits(self) -> int:
        return self._wait_count
