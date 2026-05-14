"""
Exchange rate limiter using asyncio.Semaphore plus exponential backoff.

Usage:
    from core.exchange_rate_limiter import RateLimiter, rate_limited

    limiter = RateLimiter(max_calls_per_second=3)

    @rate_limited(limiter)
    async def fetch_ticker(exchange, symbol):
        return await exchange.fetch_ticker(symbol)
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Token-bucket style rate limiter for async exchange API calls."""

    def __init__(
        self,
        max_calls_per_second: float = 3.0,
        burst: int = 10,
        max_retries: int = 5,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0,
    ) -> None:
        if max_calls_per_second <= 0:
            raise ValueError("max_calls_per_second must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")

        self._interval = 1.0 / max_calls_per_second
        self._semaphore = asyncio.Semaphore(burst)
        self._last_call = 0.0
        self._lock = asyncio.Lock()
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff

    async def acquire(self) -> None:
        """Wait until the next call is allowed."""
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def call(self, coro_fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Call an async function with rate limiting and retry backoff."""
        last_error: BaseException | None = None

        for attempt in range(self.max_retries + 1):
            await self.acquire()
            try:
                async with self._semaphore:
                    return await coro_fn(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_rate_limit_error(exc):
                    raise

                backoff = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                logger.warning("Rate limit hit; retrying in %.2fs", backoff)
                await asyncio.sleep(backoff)

        if last_error is not None:
            raise last_error
        raise RuntimeError("RateLimiter.call exited without result")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for common exchange rate-limit exceptions without requiring ccxt."""
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return "ratelimit" in name or "rate limit" in message or "too many requests" in message


_limiters: dict[str, RateLimiter] = {}


def get_limiter(exchange_id: str, max_calls_per_second: float = 3.0) -> RateLimiter:
    """Get or create a shared rate limiter for an exchange identifier."""
    if exchange_id not in _limiters:
        _limiters[exchange_id] = RateLimiter(max_calls_per_second=max_calls_per_second)
    return _limiters[exchange_id]


def rate_limited(
    limiter: RateLimiter,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for async functions that should use a shared RateLimiter."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await limiter.call(func, *args, **kwargs)

        return wrapper

    return decorator
