"""async_utils — safe asyncio event-loop helpers with uvloop optimisation.

Improvements over original:
1. uvloop drop-in — install_uvloop() is called automatically at import
   time when uvloop is available. uvloop is a C-based replacement for
   asyncio's default event loop that cuts Python async overhead by ~40%.
   Falls back silently to the standard asyncio loop if uvloop is not
   installed, so there is zero breakage on environments without it.
2. run_async() helper — single call-site for running the top-level
   coroutine with the correct policy already installed. Use this instead
   of asyncio.run() directly so uvloop is always active.
3. All original helpers preserved (get_running_loop, run_coroutine_threadsafe,
   gather_safe) for full backward compatibility.

To install uvloop:
    pip install uvloop
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Awaitable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# uvloop installation
# ---------------------------------------------------------------------------

_UVLOOP_ACTIVE = False


def install_uvloop() -> bool:
    """
    Install uvloop as the default asyncio event loop policy.

    Returns True if uvloop was successfully installed, False if it is not
    available (falls back to standard asyncio loop transparently).

    Safe to call multiple times — idempotent.
    """
    global _UVLOOP_ACTIVE
    if _UVLOOP_ACTIVE:
        return True
    try:
        import uvloop  # type: ignore
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        _UVLOOP_ACTIVE = True
        logger.info(
            "async_utils: uvloop installed as event loop policy "
            "(~40%% lower async overhead)"
        )
        return True
    except ImportError:
        logger.debug(
            "async_utils: uvloop not available — using standard asyncio loop. "
            "Install with: pip install uvloop"
        )
        return False
    except Exception as exc:
        logger.warning("async_utils: uvloop install failed: %s", exc)
        return False


def uvloop_active() -> bool:
    """Return True if uvloop is currently the active event loop policy."""
    return _UVLOOP_ACTIVE


# Install at import time — no-op if uvloop not present
install_uvloop()


# ---------------------------------------------------------------------------
# Top-level entrypoint helper
# ---------------------------------------------------------------------------

def run_async(coro: Coroutine[Any, Any, T], *, debug: bool = False) -> T:
    """
    Run a top-level coroutine with the correct event loop policy.

    Use this instead of asyncio.run() directly:
        from core.async_utils import run_async
        run_async(main())

    This ensures uvloop is active regardless of whether install_uvloop()
    was called before asyncio.run() elsewhere.
    """
    install_uvloop()   # idempotent; ensures policy is set before loop creation
    return asyncio.run(coro, debug=debug)


# ---------------------------------------------------------------------------
# Original helpers (unchanged, preserved for backward compatibility)
# ---------------------------------------------------------------------------

def get_running_loop() -> asyncio.AbstractEventLoop:
    """
    Return the running event loop.

    Raises RuntimeError if called from outside a running loop, which is
    the correct behaviour — it forces callers to be explicit about context
    instead of silently creating a new loop (the old get_event_loop() bug).
    """
    return asyncio.get_running_loop()


def run_coroutine_threadsafe(
    coro: Coroutine[Any, Any, T],
    loop: asyncio.AbstractEventLoop,
) -> concurrent.futures.Future[T]:
    """
    Schedule a coroutine on a running loop from a non-async thread.

    The loop must already be running (e.g. started by run_async() in
    the main thread). Returns a concurrent.futures.Future; call .result()
    to block until done.
    """
    return asyncio.run_coroutine_threadsafe(coro, loop)


async def gather_safe(*awaitables: Awaitable[Any]) -> list[Any]:
    """
    Thin wrapper around asyncio.gather that propagates all exceptions
    (return_exceptions=False) so nothing is silently swallowed.
    """
    try:
        return list(await asyncio.gather(*awaitables))
    except Exception as e:
        logger.error(f"gather_safe failed: {e}")
        raise


async def gather_safe_tolerant(*awaitables: Awaitable[Any]) -> list[Any]:
    """
    Like gather_safe but returns exceptions as values instead of raising.
    Use when partial failure is acceptable (e.g. multi-exchange fan-out).
    """
    try:
        results = await asyncio.gather(*awaitables, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("gather_safe_tolerant: suppressed exception: %s", r)
        return list(results)
    except Exception as e:
        logger.error(f"gather_safe_tolerant failed: {e}")
        return [e] * len(awaitables)


async def create_task_safe(coro: Coroutine[Any, Any, T], name: Optional[str] = None) -> asyncio.Task[T]:
    """
    Safe task creation with proper error handling and naming.
    
    Args:
        coro: Coroutine to run as task
        name: Optional task name for debugging
        
    Returns:
        Created asyncio.Task
        
    Raises:
        RuntimeError: If no event loop is running
    """
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro, name=name)
        logger.debug(f"Created task: {name or 'unnamed'}")
        return task
    except RuntimeError as e:
        logger.error(f"No running event loop for task {name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create task {name}: {e}")
        raise


async def wait_for_safe(coro: Coroutine[Any, Any, T], timeout: float, name: Optional[str] = None) -> T:
    """
    Safe wait_for with proper error handling and logging.
    
    Args:
        coro: Coroutine to wait for
        timeout: Timeout in seconds
        name: Optional operation name for logging
        
    Returns:
        Result of the coroutine
        
    Raises:
        asyncio.TimeoutError: If operation times out
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        logger.debug(f"Operation {name or 'unnamed'} completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Operation {name or 'unnamed'} timed out after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"Operation {name or 'unnamed'} failed: {e}")
        raise


async def gather_with_timeout(awaitables: list[Awaitable[Any]], timeout: float, 
                              tolerant: bool = False) -> list[Any]:
    """
    Gather multiple awaitables with timeout and error handling.
    
    Args:
        awaitables: List of awaitables to gather
        timeout: Timeout in seconds
        tolerant: Whether to return exceptions as values
        
    Returns:
        List of results or exceptions
    """
    try:
        if tolerant:
            results = await asyncio.wait_for(
                asyncio.gather(*awaitables, return_exceptions=True), 
                timeout=timeout
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"gather_with_timeout: suppressed exception: {r}")
        else:
            results = await asyncio.wait_for(
                asyncio.gather(*awaitables, return_exceptions=False), 
                timeout=timeout
            )
        return list(results)
    except asyncio.TimeoutError:
        logger.error(f"gather_with_timeout timed out after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"gather_with_timeout failed: {e}")
        if tolerant:
            return [e] * len(awaitables)
        else:
            raise
