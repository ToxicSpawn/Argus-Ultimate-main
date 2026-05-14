"""
Retry helpers (import-safe).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Optional, Tuple, Type

logger = logging.getLogger(__name__)


try:
    from core.errors import ArgusError as _ArgusError  # type: ignore
except Exception:
    _ArgusError = Exception  # type: ignore


class RetryError(_ArgusError):
    """Raised when retry attempts are exhausted."""


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_instance: Optional[logging.Logger] = None,
):
    """Retry decorator with exponential backoff for sync and async callables."""

    log = logger_instance or logger
    attempts = max(1, int(max_attempts))
    delay_s = float(delay)
    backoff_f = float(backoff)

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay_s
            last: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last = e
                    if attempt >= attempts:
                        log.error("All %s retry attempts exhausted for %s: %s", attempts, func.__name__, e)
                        raise RetryError(f"Failed after {attempts} attempts: {e}") from e
                    log.warning(
                        "Attempt %s/%s failed for %s: %s. Retrying in %.2fs",
                        attempt,
                        attempts,
                        func.__name__,
                        e,
                        current_delay,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_f
            raise RetryError(f"Unexpected retry failure for {func.__name__}: {last}")

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay_s
            last: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last = e
                    if attempt >= attempts:
                        log.error("All %s retry attempts exhausted for %s: %s", attempts, func.__name__, e)
                        raise RetryError(f"Failed after {attempts} attempts: {e}") from e
                    log.warning(
                        "Attempt %s/%s failed for %s: %s. Retrying in %.2fs",
                        attempt,
                        attempts,
                        func.__name__,
                        e,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_f
            raise RetryError(f"Unexpected retry failure for {func.__name__}: {last}")

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


async def retry_async(
    func: Callable,
    *args,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_instance: Optional[logging.Logger] = None,
    **kwargs,
):
    """Retry an async callable with exponential backoff."""
    log = logger_instance or logger
    attempts = max(1, int(max_attempts))
    current_delay = float(delay)
    for attempt in range(1, attempts + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            if attempt >= attempts:
                log.error("All %s retry attempts exhausted: %s", attempts, e)
                raise RetryError(f"Failed after {attempts} attempts: {e}") from e
            log.warning("Attempt %s/%s failed: %s. Retrying in %.2fs", attempt, attempts, e, current_delay)
            await asyncio.sleep(current_delay)
            current_delay *= float(backoff)
    raise RetryError("Unexpected retry failure")
