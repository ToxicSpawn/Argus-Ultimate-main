"""Thread-safety utilities for shared-state classes.

Provides:
  - ThreadSafeMixin: adds a reentrant lock to any class
  - @thread_safe: decorator for individual methods
  - @thread_safe_property: descriptor for thread-safe property access
"""
from __future__ import annotations

import logging
import threading
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class ThreadSafeMixin:
    """Mixin that adds a reentrant lock (_lock) to any class.

    Usage:
        class MySharedClass(ThreadSafeMixin):
            def update(self, value):
                with self._lock:
                    self._data = value
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Use RLock so methods can call other locked methods on the same instance
        self._lock: threading.RLock = threading.RLock()


def thread_safe(func: F) -> F:
    """Decorator that acquires self._lock before calling the method.

    The decorated class must have a ``_lock`` attribute (use ThreadSafeMixin or
    set ``self._lock = threading.RLock()`` in ``__init__``).

    Usage:
        class Foo:
            def __init__(self):
                self._lock = threading.RLock()

            @thread_safe
            def increment(self):
                self.counter += 1
    """
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock = getattr(self, "_lock", None)
        if lock is None:
            logger.warning(
                "@thread_safe: %s.%s has no _lock — running unprotected",
                type(self).__name__, func.__name__,
            )
            return func(self, *args, **kwargs)
        with lock:
            return func(self, *args, **kwargs)
    return wrapper  # type: ignore[return-value]


class thread_safe_property:  # noqa: N801
    """Descriptor for a thread-safe property backed by self._lock.

    Usage:
        class Foo:
            def __init__(self):
                self._lock = threading.RLock()
                self._value = 0

            @thread_safe_property
            def value(self):
                return self._value
    """

    def __init__(self, fget: Callable) -> None:
        self.fget = fget
        self.__doc__ = fget.__doc__

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        lock = getattr(obj, "_lock", None)
        if lock is None:
            return self.fget(obj)
        with lock:
            return self.fget(obj)
