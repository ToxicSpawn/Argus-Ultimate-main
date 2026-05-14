"""
Tests for core/async_utils.py — Issue #20 security fix.
"""
from __future__ import annotations

import asyncio
import pytest

from core.async_utils import get_running_loop, gather_safe


def test_get_running_loop_outside_coroutine_raises() -> None:
    """Calling get_running_loop() outside a coroutine must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        get_running_loop()


@pytest.mark.asyncio
async def test_get_running_loop_inside_coroutine() -> None:
    """Inside a coroutine get_running_loop() returns the active loop."""
    loop = get_running_loop()
    assert loop is asyncio.get_event_loop()  # same object


@pytest.mark.asyncio
async def test_gather_safe_returns_all_results() -> None:
    async def _val(n: int) -> int:
        return n

    results = await gather_safe(_val(1), _val(2), _val(3))
    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_gather_safe_propagates_exception() -> None:
    async def _boom() -> None:
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        await gather_safe(_boom())
