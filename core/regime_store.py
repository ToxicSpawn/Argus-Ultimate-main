"""
Regime Persistence Store — caches detected market regimes across process restarts.

The strategy engine detects regime via TFT/heuristics, but resets to RANGE on
every restart. This store persists the last N regime readings per symbol with
timestamps so the system starts with real regime context.

Usage:
    store = RegimeStore()
    store.save("BTC/USD", "TREND_UP", confidence=0.82, source="tft")
    regime, meta = store.load("BTC/USD")  # ("TREND_UP", {...}) or ("RANGE", {})
    store.cleanup_stale(max_age_hours=6)  # purge old entries
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB = "data/regime_store.db"
MAX_AGE_SECONDS = 6 * 3600  # 6 hours — stale regime is untrustworthy
VALID_REGIMES = {"TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL", "CRISIS"}

# DDL statements
_CREATE_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS regime_cache (
    symbol      TEXT    PRIMARY KEY,
    regime      TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.5,
    source      TEXT    NOT NULL DEFAULT 'unknown',
    ts          REAL    NOT NULL,
    meta_json   TEXT    NOT NULL DEFAULT '{}'
)
"""

_CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS regime_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    regime      TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.5,
    source      TEXT    NOT NULL DEFAULT 'unknown',
    ts          REAL    NOT NULL,
    meta_json   TEXT    NOT NULL DEFAULT '{}'
)
"""

_CREATE_HISTORY_IDX = """
CREATE INDEX IF NOT EXISTS idx_regime_history_symbol_ts
    ON regime_history (symbol, ts DESC)
"""


class RegimeStore:
    """
    Thread-safe SQLite-backed store for market regime state.

    Two tables:
      * regime_cache   — latest regime per symbol (UPSERT on save)
      * regime_history — append-only log of all regime readings
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        max_age_seconds: float = MAX_AGE_SECONDS,
    ) -> None:
        self._db_path = db_path
        self._max_age_seconds = float(max_age_seconds)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(_CREATE_CACHE_TABLE)
            conn.execute(_CREATE_HISTORY_TABLE)
            conn.execute(_CREATE_HISTORY_IDX)
            conn.commit()
        logger.debug("RegimeStore initialised at %s", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection with WAL mode enabled."""
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def save(
        self,
        symbol: str,
        regime: str,
        confidence: float = 0.5,
        source: str = "unknown",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Upsert the latest regime for *symbol* in regime_cache and append
        a row to regime_history.

        Raises ValueError if *regime* is not in VALID_REGIMES.
        """
        if regime not in VALID_REGIMES:
            raise ValueError(
                f"Invalid regime '{regime}'. Must be one of {sorted(VALID_REGIMES)}"
            )

        confidence = float(max(0.0, min(1.0, confidence)))
        meta_json = json.dumps(meta or {})
        ts = time.time()

        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO regime_cache
                            (symbol, regime, confidence, source, ts, meta_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (symbol, regime, confidence, source, ts, meta_json),
                    )
                    conn.execute(
                        """
                        INSERT INTO regime_history
                            (symbol, regime, confidence, source, ts, meta_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (symbol, regime, confidence, source, ts, meta_json),
                    )
                    conn.commit()
                logger.debug(
                    "RegimeStore.save: %s -> %s (conf=%.2f, src=%s)",
                    symbol, regime, confidence, source,
                )
            except sqlite3.Error as exc:
                logger.error("RegimeStore.save failed for %s: %s", symbol, exc)
                raise

    def load(
        self,
        symbol: str,
        max_age_seconds: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Return the most recent regime for *symbol* if it is not stale.

        Parameters
        ----------
        max_age_seconds:
            Override the instance-level TTL.  If ``None``, uses the value
            passed at construction time (default 3600s).

        Returns:
            (regime, meta_dict) — e.g. ("TREND_UP", {"confidence": 0.82, "source": "tft"})
            or ("UNKNOWN", {}) if the entry is missing or older than max_age_seconds.
        """
        effective_ttl = max_age_seconds if max_age_seconds is not None else self._max_age_seconds
        cutoff = time.time() - effective_ttl

        with self._lock:
            try:
                with self._connect() as conn:
                    row = conn.execute(
                        """
                        SELECT regime, confidence, source, ts, meta_json
                        FROM regime_cache
                        WHERE symbol = ?
                        """,
                        (symbol,),
                    ).fetchone()
            except sqlite3.Error as exc:
                logger.error("RegimeStore.load failed for %s: %s", symbol, exc)
                return "UNKNOWN", {}

        if row is None:
            logger.debug("RegimeStore.load: no entry for %s; returning UNKNOWN", symbol)
            return "UNKNOWN", {}

        if float(row["ts"]) < cutoff:
            age_seconds = time.time() - float(row["ts"])
            logger.warning(
                "RegimeStore.load: stale regime data for %s — age %.0fs exceeds TTL %.0fs; "
                "returning UNKNOWN",
                symbol, age_seconds, effective_ttl,
            )
            return "UNKNOWN", {}

        try:
            extra_meta = json.loads(row["meta_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            extra_meta = {}

        meta: Dict[str, Any] = {
            "confidence": float(row["confidence"]),
            "source": str(row["source"]),
            "ts": float(row["ts"]),
        }
        meta.update(extra_meta)

        return str(row["regime"]), meta

    def is_stale(
        self,
        symbol: str,
        max_age_seconds: Optional[float] = None,
    ) -> bool:
        """
        Return True if the stored regime for *symbol* is older than TTL
        or does not exist.

        Parameters
        ----------
        max_age_seconds:
            Override the instance-level TTL. If ``None``, uses the value
            passed at construction time.
        """
        effective_ttl = max_age_seconds if max_age_seconds is not None else self._max_age_seconds
        cutoff = time.time() - effective_ttl

        with self._lock:
            try:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT ts FROM regime_cache WHERE symbol = ?",
                        (symbol,),
                    ).fetchone()
            except sqlite3.Error as exc:
                logger.error("RegimeStore.is_stale failed for %s: %s", symbol, exc)
                return True  # Treat errors as stale

        if row is None:
            return True

        return float(row["ts"]) < cutoff

    def load_all(
        self,
        max_age_seconds: float = MAX_AGE_SECONDS,
    ) -> Dict[str, Tuple[str, Dict[str, Any]]]:
        """
        Return all non-stale regime entries keyed by symbol.

        Returns:
            {symbol: (regime, meta_dict), ...}
        """
        cutoff = time.time() - max_age_seconds
        result: Dict[str, Tuple[str, Dict[str, Any]]] = {}

        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT symbol, regime, confidence, source, ts, meta_json
                        FROM regime_cache
                        WHERE ts >= ?
                        ORDER BY symbol
                        """,
                        (cutoff,),
                    ).fetchall()
            except sqlite3.Error as exc:
                logger.error("RegimeStore.load_all failed: %s", exc)
                return {}

        for row in rows:
            try:
                extra_meta = json.loads(row["meta_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                extra_meta = {}

            meta: Dict[str, Any] = {
                "confidence": float(row["confidence"]),
                "source": str(row["source"]),
                "ts": float(row["ts"]),
            }
            meta.update(extra_meta)
            result[str(row["symbol"])] = (str(row["regime"]), meta)

        logger.debug("RegimeStore.load_all: returned %d non-stale entries", len(result))
        return result

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_stale(self, max_age_hours: float = 6.0) -> int:
        """
        Delete entries from both tables that are older than *max_age_hours*.

        Returns the number of rows deleted from regime_cache.
        """
        cutoff = time.time() - max_age_hours * 3600.0

        with self._lock:
            try:
                with self._connect() as conn:
                    cur = conn.execute(
                        "DELETE FROM regime_cache WHERE ts < ?", (cutoff,)
                    )
                    deleted_cache = cur.rowcount

                    # Also prune history older than 7 × max_age_hours to keep the table bounded
                    history_cutoff = time.time() - max_age_hours * 3600.0 * 7
                    conn.execute(
                        "DELETE FROM regime_history WHERE ts < ?", (history_cutoff,)
                    )
                    conn.commit()
            except sqlite3.Error as exc:
                logger.error("RegimeStore.cleanup_stale failed: %s", exc)
                return 0

        logger.info(
            "RegimeStore.cleanup_stale: removed %d stale cache entries (cutoff %.0fh)",
            deleted_cache, max_age_hours,
        )
        return deleted_cache

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_regime_history(
        self,
        symbol: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return the last *limit* regime readings for *symbol* from regime_history,
        ordered newest first.

        Each entry is a dict with keys: id, symbol, regime, confidence, source, ts, meta.
        """
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT id, symbol, regime, confidence, source, ts, meta_json
                        FROM regime_history
                        WHERE symbol = ?
                        ORDER BY ts DESC
                        LIMIT ?
                        """,
                        (symbol, limit),
                    ).fetchall()
            except sqlite3.Error as exc:
                logger.error(
                    "RegimeStore.get_regime_history failed for %s: %s", symbol, exc
                )
                return []

        history: List[Dict[str, Any]] = []
        for row in rows:
            try:
                extra_meta = json.loads(row["meta_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                extra_meta = {}

            entry: Dict[str, Any] = {
                "id": int(row["id"]),
                "symbol": str(row["symbol"]),
                "regime": str(row["regime"]),
                "confidence": float(row["confidence"]),
                "source": str(row["source"]),
                "ts": float(row["ts"]),
                "meta": extra_meta,
            }
            history.append(entry)

        return history
