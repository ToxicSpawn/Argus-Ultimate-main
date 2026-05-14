"""
Strategy Similarity Detector — identifies redundant strategies by comparing
the overlap (Jaccard similarity) of their signal streams.

When two strategies consistently fire on the same symbols at the same time,
capital allocated to both is effectively single-strategy risk.  This module
quantifies that overlap so the allocator can diversify.

Persistence: SQLite at ``data/strategy_similarity.db``.

Usage::

    det = StrategySimilarityDetector()
    det.record_signal("momentum", "BTC/AUD", "long", ts)
    det.record_signal("trend_follow", "BTC/AUD", "long", ts)
    sim = det.compute_similarity("momentum", "trend_follow")
    print(sim)  # 0.0 – 1.0
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "strategy_similarity.db"


class StrategySimilarityDetector:
    """Detects redundant strategies via signal overlap analysis.

    Parameters
    ----------
    db_path : str | None
        Path to SQLite database.  Defaults to ``data/strategy_similarity.db``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_dir = _DEFAULT_DB_DIR
            db_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_dir / _DEFAULT_DB_NAME)
        else:
            self._db_path = str(db_path)

        self._lock = threading.Lock()
        self._init_db()
        logger.info("StrategySimilarityDetector initialised, db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy    TEXT    NOT NULL,
                        symbol      TEXT    NOT NULL,
                        direction   TEXT    NOT NULL,
                        signal_time TEXT    NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_strategy
                    ON signals(strategy)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_time
                    ON signals(signal_time)
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_signal(
        self,
        strategy: str,
        symbol: str,
        direction: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a strategy signal for similarity analysis.

        Parameters
        ----------
        strategy : str
            Strategy that emitted the signal.
        symbol : str
            Trading pair, e.g. ``"BTC/AUD"``.
        direction : str
            ``"long"`` or ``"short"``.
        timestamp : datetime | None
            UTC timestamp.  Defaults to now.
        """
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO signals (strategy, symbol, direction, signal_time) VALUES (?, ?, ?, ?)",
                    (strategy, symbol, direction, ts),
                )
                conn.commit()
            finally:
                conn.close()
        logger.debug("Recorded signal: %s %s %s", strategy, symbol, direction)

    def compute_similarity(
        self,
        strategy_a: str,
        strategy_b: str,
        lookback_days: int = 7,
    ) -> float:
        """Compute Jaccard similarity between two strategies' signal sets.

        A signal is represented as the tuple ``(symbol, direction, date_bucket)``
        where date_bucket is the ISO date (YYYY-MM-DD) of the signal.  This
        groups signals into daily buckets to tolerate minor timing differences.

        Parameters
        ----------
        strategy_a, strategy_b : str
            The two strategies to compare.
        lookback_days : int
            Window to consider.

        Returns
        -------
        float
            Jaccard index in ``[0.0, 1.0]``.  1.0 means identical signal sets.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                rows_a = conn.execute(
                    "SELECT symbol, direction, signal_time FROM signals WHERE strategy = ? AND signal_time >= ?",
                    (strategy_a, cutoff),
                ).fetchall()
                rows_b = conn.execute(
                    "SELECT symbol, direction, signal_time FROM signals WHERE strategy = ? AND signal_time >= ?",
                    (strategy_b, cutoff),
                ).fetchall()
            finally:
                conn.close()

        def _to_set(rows: list) -> set:
            result = set()
            for r in rows:
                # Bucket by date to allow intra-day timing tolerance
                date_bucket = r["signal_time"][:10]
                result.add((r["symbol"], r["direction"], date_bucket))
            return result

        set_a = _to_set(rows_a)
        set_b = _to_set(rows_b)

        if not set_a and not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        similarity = intersection / union if union > 0 else 0.0
        logger.debug(
            "Similarity(%s, %s) = %.4f (intersection=%d, union=%d)",
            strategy_a, strategy_b, similarity, intersection, union,
        )
        return similarity

    def get_similarity_matrix(
        self,
        strategies: List[str],
        lookback_days: int = 7,
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise similarity matrix for a list of strategies.

        Parameters
        ----------
        strategies : list[str]
            Strategy names.
        lookback_days : int
            Lookback window in days.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict: ``matrix[a][b]`` = Jaccard similarity.
        """
        matrix: Dict[str, Dict[str, float]] = {}
        for a in strategies:
            matrix[a] = {}
            for b in strategies:
                if a == b:
                    matrix[a][b] = 1.0
                elif b in matrix and a in matrix[b]:
                    # Symmetric — reuse already computed value
                    matrix[a][b] = matrix[b][a]
                else:
                    matrix[a][b] = self.compute_similarity(a, b, lookback_days)
        return matrix

    def find_redundant_pairs(
        self,
        threshold: float = 0.8,
        lookback_days: int = 7,
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of strategies whose similarity exceeds the threshold.

        Parameters
        ----------
        threshold : float
            Similarity threshold in ``[0, 1]``.  Pairs above this are redundant.
        lookback_days : int
            Lookback window.

        Returns
        -------
        list[tuple[str, str, float]]
            List of ``(strategy_a, strategy_b, similarity)`` tuples.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT DISTINCT strategy FROM signals").fetchall()
            finally:
                conn.close()

        strategies = [r["strategy"] for r in rows]
        redundant: List[Tuple[str, str, float]] = []
        seen: set = set()

        for i, a in enumerate(strategies):
            for b in strategies[i + 1:]:
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                sim = self.compute_similarity(a, b, lookback_days)
                if sim >= threshold:
                    redundant.append((a, b, sim))

        redundant.sort(key=lambda x: x[2], reverse=True)
        logger.info("find_redundant_pairs: %d pairs above threshold %.2f", len(redundant), threshold)
        return redundant

    def get_diversification_score(
        self,
        active_strategies: List[str],
        lookback_days: int = 7,
    ) -> float:
        """Compute a diversification score for the active strategy set.

        The score is ``1 - mean(pairwise similarity)``.  A score of 1.0 means
        all strategies are perfectly uncorrelated; 0.0 means they are identical.

        Parameters
        ----------
        active_strategies : list[str]
            Currently active strategies.
        lookback_days : int
            Lookback window.

        Returns
        -------
        float
            Diversification score in ``[0, 1]``.
        """
        if len(active_strategies) < 2:
            return 1.0

        matrix = self.get_similarity_matrix(active_strategies, lookback_days)
        pair_count = 0
        total_similarity = 0.0

        for i, a in enumerate(active_strategies):
            for b in active_strategies[i + 1:]:
                total_similarity += matrix[a][b]
                pair_count += 1

        if pair_count == 0:
            return 1.0

        mean_similarity = total_similarity / pair_count
        score = 1.0 - mean_similarity
        logger.info(
            "Diversification score: %.4f (mean_sim=%.4f, %d pairs)",
            score, mean_similarity, pair_count,
        )
        return score
