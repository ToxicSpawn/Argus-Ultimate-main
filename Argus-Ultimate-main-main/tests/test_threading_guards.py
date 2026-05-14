"""Tests for threading guard utilities."""
from __future__ import annotations

import threading
from utils.threading_guards import ThreadSafeMixin, thread_safe, thread_safe_property


class Counter(ThreadSafeMixin):
    def __init__(self):
        super().__init__()
        self.value = 0

    @thread_safe
    def increment(self):
        self.value += 1


def test_thread_safe_mixin_has_lock():
    c = Counter()
    assert hasattr(c, "_lock")


def test_concurrent_increments():
    c = Counter()
    threads = [threading.Thread(target=c.increment) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.value == 100


def test_thread_safe_decorator_no_lock_warning(caplog):
    class NoLock:
        @thread_safe
        def do(self):
            return 42

    obj = NoLock()
    import logging
    with caplog.at_level(logging.WARNING):
        result = obj.do()
    assert result == 42
    assert "no _lock" in caplog.text


def test_thread_safe_property():
    class Foo:
        def __init__(self):
            self._lock = threading.RLock()
            self._x = 99

        @thread_safe_property
        def x(self):
            return self._x

    foo = Foo()
    assert foo.x == 99
