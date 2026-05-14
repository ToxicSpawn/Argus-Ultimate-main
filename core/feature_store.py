#!/usr/bin/env python3
"""
In-Memory Feature Store — TTL-aware, thread-safe feature cache for ML pipelines.

Provides O(1) lookups via nested dicts with per-symbol locking. Expired features
are transparently evicted on read and via a periodic background cleaner.

Standalone usage:
    store = FeatureStore(background=True)
    store.set("BTC/AUD", "rsi_14", 62.5, ttl_s=120)
    val = store.get("BTC/AUD", "rsi_14")   # 62.5 or None if expired
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal storage types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Entry:
    """Single feature value with expiry metadata."""

    value: Any
    created_at: float
    expires_at: float  # epoch seconds; float("inf") for no expiry


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Thread-safe in-memory feature store with per-entry TTL.

    Parameters
    ----------
    default_ttl_s : float
        Default time-to-live in seconds (default 300 = 5 minutes).
    cleanup_interval_s : float
        How often the background cleaner runs (default 60 seconds).
    background : bool
        If True, start a daemon thread that periodically evicts expired entries.
    max_history : int
        Maximum entries per (symbol, feature) — only latest is kept (default 1).
    """

    def __init__(
        self,
        default_ttl_s: float = 300.0,
        cleanup_interval_s: float = 60.0,
        background: bool = False,
        max_history: int = 1,
    ):
        self.default_ttl_s = default_ttl_s
        self.cleanup_interval_s = cleanup_interval_s
        self.max_history = max_history

        # symbol -> feature_name -> _Entry
        self._data: Dict[str, Dict[str, _Entry]] = {}
        # Per-symbol locks
        self._symbol_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        # Stats
        self._total_sets: int = 0
        self._total_gets: int = 0
        self._total_expired: int = 0

        self._background = background
        self._stop_event = threading.Event()
        self._cleaner_thread: Optional[threading.Thread] = None

        if background:
            self._cleaner_thread = threading.Thread(
                target=self._cleanup_loop, daemon=True, name="FeatureStore-cleaner"
            )
            self._cleaner_thread.start()
            logger.info(
                "FeatureStore started with background cleaner (interval=%.0fs, default_ttl=%.0fs)",
                cleanup_interval_s, default_ttl_s,
            )
        else:
            logger.info(
                "FeatureStore initialised (default_ttl=%.0fs, no background cleaner)",
                default_ttl_s,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(self, symbol: str, feature_name: str, value: Any, ttl_s: Optional[float] = None) -> None:
        """
        Store a feature value with optional TTL override.

        Parameters
        ----------
        symbol : str
            Asset identifier.
        feature_name : str
            Name of the feature (e.g. "rsi_14", "spread_bps").
        value : Any
            The feature value.
        ttl_s : float, optional
            Seconds until expiry. Defaults to ``self.default_ttl_s``.
            Pass 0 or negative for no expiry.
        """
        now = time.time()
        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        expires = now + ttl if ttl > 0 else float("inf")
        entry = _Entry(value=value, created_at=now, expires_at=expires)

        lock = self._get_symbol_lock(symbol)
        with lock:
            if symbol not in self._data:
                self._data[symbol] = {}
            self._data[symbol][feature_name] = entry

        self._total_sets += 1

    def get(self, symbol: str, feature_name: str) -> Any:
        """
        Retrieve a feature value if it exists and has not expired.

        Parameters
        ----------
        symbol : str
            Asset identifier.
        feature_name : str
            Feature name.

        Returns
        -------
        Any
            The stored value, or None if missing/expired.
        """
        self._total_gets += 1
        lock = self._get_symbol_lock(symbol)
        with lock:
            sym_data = self._data.get(symbol)
            if sym_data is None:
                return None
            entry = sym_data.get(feature_name)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del sym_data[feature_name]
                self._total_expired += 1
                return None
            return entry.value

    def get_all(self, symbol: str) -> Dict[str, Any]:
        """
        Return all non-expired features for a symbol.

        Parameters
        ----------
        symbol : str
            Asset identifier.

        Returns
        -------
        dict
            feature_name -> value (only live entries).
        """
        lock = self._get_symbol_lock(symbol)
        now = time.time()
        result: Dict[str, Any] = {}
        expired_keys: List[str] = []

        with lock:
            sym_data = self._data.get(symbol)
            if sym_data is None:
                return {}
            for fname, entry in sym_data.items():
                if now > entry.expires_at:
                    expired_keys.append(fname)
                else:
                    result[fname] = entry.value
            for k in expired_keys:
                del sym_data[k]
                self._total_expired += 1

        return result

    def get_feature_vector(self, symbol: str, feature_names: List[str]) -> List[Any]:
        """
        Return an ordered list of feature values (None for missing/expired).

        Parameters
        ----------
        symbol : str
            Asset identifier.
        feature_names : list of str
            Ordered feature names.

        Returns
        -------
        list
            Values aligned with *feature_names*.
        """
        all_feats = self.get_all(symbol)
        return [all_feats.get(fn) for fn in feature_names]

    def set_batch(self, symbol: str, features: Dict[str, Any], ttl_s: Optional[float] = None) -> None:
        """
        Bulk-set multiple features for a symbol in one lock acquisition.

        Parameters
        ----------
        symbol : str
            Asset identifier.
        features : dict
            feature_name -> value.
        ttl_s : float, optional
            TTL for all entries (defaults to ``self.default_ttl_s``).
        """
        now = time.time()
        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        expires = now + ttl if ttl > 0 else float("inf")

        lock = self._get_symbol_lock(symbol)
        with lock:
            if symbol not in self._data:
                self._data[symbol] = {}
            for fname, val in features.items():
                self._data[symbol][fname] = _Entry(value=val, created_at=now, expires_at=expires)

        self._total_sets += len(features)

    def get_stats(self) -> Dict[str, Any]:
        """
        Return store statistics.

        Returns
        -------
        dict
            Keys: total_features, expired_count, symbols_count, avg_age_s,
            total_sets, total_gets.
        """
        now = time.time()
        total_features = 0
        total_age = 0.0
        expired_count = 0
        symbols_count = 0

        with self._global_lock:
            symbol_list = list(self._data.keys())

        for sym in symbol_list:
            lock = self._get_symbol_lock(sym)
            with lock:
                sym_data = self._data.get(sym)
                if not sym_data:
                    continue
                symbols_count += 1
                for entry in sym_data.values():
                    if now > entry.expires_at:
                        expired_count += 1
                    else:
                        total_features += 1
                        total_age += now - entry.created_at

        avg_age = total_age / total_features if total_features > 0 else 0.0

        return {
            "total_features": total_features,
            "expired_count": expired_count,
            "symbols_count": symbols_count,
            "avg_age_s": round(avg_age, 2),
            "total_sets": self._total_sets,
            "total_gets": self._total_gets,
            "total_expired_evictions": self._total_expired,
        }

    def cleanup(self) -> int:
        """
        Remove all expired entries from the store.

        Returns
        -------
        int
            Number of entries removed.
        """
        now = time.time()
        removed = 0

        with self._global_lock:
            symbol_list = list(self._data.keys())

        for sym in symbol_list:
            lock = self._get_symbol_lock(sym)
            with lock:
                sym_data = self._data.get(sym)
                if not sym_data:
                    continue
                expired = [k for k, e in sym_data.items() if now > e.expires_at]
                for k in expired:
                    del sym_data[k]
                    removed += 1
                # Remove empty symbol buckets
                if not sym_data:
                    del self._data[sym]

        if removed > 0:
            self._total_expired += removed
            logger.debug("FeatureStore cleanup: removed %d expired entries", removed)

        return removed

    def stop(self) -> None:
        """Stop the background cleaner thread (if running)."""
        self._stop_event.set()
        if self._cleaner_thread and self._cleaner_thread.is_alive():
            self._cleaner_thread.join(timeout=5.0)
            logger.info("FeatureStore background cleaner stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_symbol_lock(self, symbol: str) -> threading.Lock:
        """Return or create a per-symbol lock."""
        with self._global_lock:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]

    def _cleanup_loop(self) -> None:
        """Background thread target: periodically call cleanup()."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.cleanup_interval_s)
            if self._stop_event.is_set():
                break
            try:
                self.cleanup()
            except Exception:
                logger.warning("FeatureStore cleanup error", exc_info=True)
