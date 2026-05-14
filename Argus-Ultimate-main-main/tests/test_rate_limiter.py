"""Tests for the token-bucket rate limiter."""
from __future__ import annotations

import asyncio
import time
import pytest
from utils.rate_limiter import RateLimiter


def test_sync_acquire_immediate():
    limiter = RateLimiter(calls_per_second=100.0, burst=10)
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire_sync()
    assert time.monotonic() - start < 0.5


def test_context_manager_sync():
    limiter = RateLimiter(calls_per_second=100.0, burst=10)
    with limiter:
        pass  # should not raise


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        RateLimiter(calls_per_second=0.0)


def test_decorator_sync():
    limiter = RateLimiter(calls_per_second=100.0, burst=5)
    results = []

    @limiter.limit
    def work(n):
        results.append(n)
        return n

    for i in range(3):
        assert work(i) == i
    assert results == [0, 1, 2]


@pytest.mark.asyncio
async def test_async_acquire():
    limiter = RateLimiter(calls_per_second=100.0, burst=10)
    async with limiter:
        pass


@pytest.mark.asyncio
async def test_decorator_async():
    limiter = RateLimiter(calls_per_second=100.0, burst=5)

    @limiter.limit
    async def async_work(n):
        return n * 2

    result = await async_work(3)
    assert result == 6
