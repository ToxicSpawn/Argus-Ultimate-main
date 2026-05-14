"""utils/thread_safe_cache.py — RLock-protected LRU cache with TTL.

Replaces bare dict caches in feature_pipeline and data_feed that were
being read/written from multiple threads without locks.

Usage:
    cache = ThreadSafeCache(maxsize=512, ttl=60.0)  # 60s TTL
    cache.set("btc_ohlcv_1m", data)
    val = cache.get("btc_ohlcv_1m")  # None if expired or missing
    cache.set_with_ttl("key", data, ttl=5.0)  # per-item TTL override
    cache.invalidate("btc_ohlcv_1m")
    cache.clear()
    stats = cache.stats  # hits, misses, size, evictions
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_SENTINEL = object()


class ThreadSafeCache:
    """
    Thread-safe LRU cache with optional TTL.

    Args:
        maxsize:  Max number of entries before LRU eviction. 0 = unlimited.
        ttl:      Default time-to-live in seconds. 0 = no expiry.
    """

    def __init__(self, maxsize: int = 256, ttl: float = 0.0) -> None:
        self.maxsize = maxsize
        self.default_ttl = ttl
        self._store: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return default

            value, expires_at = self._store[key]

            if expires_at > 0 and time.monotonic() > expires_at:
                del self._store[key]
                self._misses += 1
                return default

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value. Uses per-item ttl if provided, else default_ttl."""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = (time.monotonic() + effective_ttl) if effective_ttl > 0 else 0.0
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            self._evict_if_needed()

    # Alias for API consistency with batch 5 callers
    set_with_ttl = set

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key, _SENTINEL) is not _SENTINEL

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def _evict_if_needed(self) -> None:
        """Evict LRU entry if over maxsize. Call inside lock."""
        if self.maxsize > 0:
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)
                self._evictions += 1

    @property
    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "size": len(self._store),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": (
                    self._hits / max(self._hits + self._misses, 1)
                ),
            }
