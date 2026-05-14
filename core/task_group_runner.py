"""M09 — asyncio.TaskGroup wrapper so swallowed exceptions surface.

Python < 3.11 shim included via ``asyncio.TaskGroup`` backport detection.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# ── TaskGroup availability ────────────────────────────────────────────────────
try:
    _TaskGroup = asyncio.TaskGroup  # Python 3.11+
except AttributeError:  # pragma: no cover  # Python < 3.11
    _TaskGroup = None  # type: ignore[assignment]


async def run_tasks(
    *coros: Coroutine[Any, Any, Any],
    name: str = "argus",
) -> list[Any]:
    """Run *coros* concurrently; raise ``ExceptionGroup`` if any fail.

    Uses ``asyncio.TaskGroup`` on Python 3.11+ so that all exceptions are
    surfaced together rather than silently swallowed.

    Args:
        *coros: Awaitable coroutines to run concurrently.
        name:   Label prefix for created tasks (aids debugging).

    Returns:
        List of task results in submission order.

    Raises:
        ExceptionGroup: If one or more tasks raise.
    """
    if _TaskGroup is not None:
        results: list[Any] = [None] * len(coros)
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(coro, name=f"{name}-{i}")
                for i, coro in enumerate(coros)
            ]
        results = [t.result() for t in tasks]
        return results

    # ── Python < 3.11 fallback using gather ──────────────────────────────────
    tasks_gather = [asyncio.ensure_future(c) for c in coros]  # type: ignore[arg-type]
    results_gather = await asyncio.gather(*tasks_gather, return_exceptions=False)
    return list(results_gather)


class SupervisedTaskRunner:
    """Long-running supervisor: restarts crashed tasks with backoff."""

    def __init__(self, max_restarts: int = 5) -> None:
        self._max_restarts = max_restarts
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def spawn(
        self,
        coro_factory,  # () -> Coroutine
        *,
        name: str,
    ) -> None:
        """Spawn *coro_factory()* as a supervised background task."""
        restarts = 0

        async def _wrapper() -> None:
            nonlocal restarts
            while True:
                try:
                    await coro_factory()
                    return  # clean exit
                except asyncio.CancelledError:
                    raise
                except Exception:
                    restarts += 1
                    logger.exception(
                        "Supervised task %r crashed (restart %d/%d)",
                        name, restarts, self._max_restarts,
                    )
                    if restarts >= self._max_restarts:
                        logger.error("Task %r exceeded max restarts — giving up", name)
                        raise
                    backoff = min(2 ** restarts, 60)
                    await asyncio.sleep(backoff)

        task = asyncio.create_task(_wrapper(), name=name)
        self._tasks[name] = task

    def cancel_all(self) -> None:
        """Cancel all supervised tasks."""
        for t in self._tasks.values():
            t.cancel()
        self._tasks.clear()
