"""core/shared_state.py — thread-safe shared state container.

Replaces the ad-hoc module-level dicts that were being mutated from
both the update thread and the main trading loop concurrently.

Usage:
    from core.shared_state import SharedState
    state = SharedState()          # or SharedState.instance() for singleton
    state.set("btc_price", 65000.0)
    price = state.get("btc_price", default=0.0)
    state.update({"btc_price": 65100.0, "eth_price": 3200.0})
    snapshot = state.snapshot()    # full dict copy, safe to iterate
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class SharedState:
    """Thread-safe key-value state store.

    All public methods acquire an RLock — safe to call from any thread
    including async coroutines run in a thread pool.
    """

    _singleton: Optional["SharedState"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._version: int = 0

    # ------------------------------------------------------------------ #
    #  Singleton                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def instance(cls) -> "SharedState":
        """Return the process-wide singleton SharedState."""
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton (testing only)."""
        with cls._singleton_lock:
            cls._singleton = None

    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._version += 1

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._version += 1
                return True
            return False

    def update(self, mapping: Dict[str, Any]) -> None:
        """Bulk update — single lock acquisition."""
        with self._lock:
            self._data.update(mapping)
            self._version += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy safe to read outside the lock."""
        with self._lock:
            return dict(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._version += 1

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def version(self) -> int:
        """Monotonically increasing write counter."""
        with self._lock:
            return self._version

    # ------------------------------------------------------------------ #
    #  Namespaced access                                                   #
    # ------------------------------------------------------------------ #

    def ns(self, namespace: str) -> "NamespacedState":
        """Return a namespaced view: ns('prices').set('BTC', 65000)."""
        return NamespacedState(self, namespace)


class NamespacedState:
    """Thin namespace wrapper over SharedState.

    Keys are stored as 'namespace:key' in the parent SharedState.
    """

    def __init__(self, parent: SharedState, namespace: str) -> None:
        self._parent = parent
        self._ns = namespace

    def _k(self, key: str) -> str:
        return f"{self._ns}:{key}"

    def get(self, key: str, default: Any = None) -> Any:
        return self._parent.get(self._k(key), default)

    def set(self, key: str, value: Any) -> None:
        self._parent.set(self._k(key), value)

    def delete(self, key: str) -> bool:
        return self._parent.delete(self._k(key))

    def update(self, mapping: Dict[str, Any]) -> None:
        self._parent.update({self._k(k): v for k, v in mapping.items()})

    def snapshot(self) -> Dict[str, Any]:
        prefix = self._ns + ":"
        full = self._parent.snapshot()
        return {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
