"""Token-bucket rate limiter for exchange API calls.

Usage:
    limiter = RateLimiter(calls_per_second=5.0)

    async with limiter:  # async context manager
        await exchange.fetch_ticker(symbol)

    # Or as a decorator:
    @limiter.limit
    async def fetch_data():
        ...
"""
from __future__ import annotations

import asyncio
import logging
import time
import threading
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe and asyncio-compatible token-bucket rate limiter."""

    def __init__(self, calls_per_second: float = 5.0, burst: int = 10) -> None:
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be positive")
        self._rate = float(calls_per_second)
        self._burst = max(1, int(burst))
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock() if self._has_running_loop() else None

    @staticmethod
    def _has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time (call while holding lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def _consume_sync(self) -> float:
        """Consume one token synchronously. Returns seconds to wait (0 if immediate)."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            wait = (1.0 - self._tokens) / self._rate
            return wait

    def acquire_sync(self) -> None:
        """Block until a token is available (synchronous)."""
        while True:
            wait = self._consume_sync()
            if wait <= 0:
                return
            logger.debug("RateLimiter: throttling %.3fs", wait)
            time.sleep(wait)

    async def acquire(self) -> None:
        """Await until a token is available (async)."""
        while True:
            wait = self._consume_sync()
            if wait <= 0:
                return
            logger.debug("RateLimiter: async throttling %.3fs", wait)
            await asyncio.sleep(wait)

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    def __enter__(self) -> "RateLimiter":
        self.acquire_sync()
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def limit(self, func: Callable) -> Callable:
        """Decorator that applies rate limiting to a coroutine or regular function."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await self.acquire()
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                self.acquire_sync()
                return func(*args, **kwargs)
            return sync_wrapper


# Shared default limiters — import and use these instead of creating per-call
kraken_limiter = RateLimiter(calls_per_second=1.0, burst=5)     # Kraken: 1 call/sec
coinbase_limiter = RateLimiter(calls_per_second=3.0, burst=10)  # Coinbase: 3 calls/sec
generic_limiter = RateLimiter(calls_per_second=5.0, burst=10)   # Generic default
