"""
Meta-Learning Model Selector — learns which ML model performs best per regime.

Records historical performance of each model under different market regimes
and feature conditions, then uses that history to select the optimal model
for current conditions.  This is *meta-learning* in the practical sense:
learning *about* models rather than learning parameters within a model.

Persistence is via SQLite at ``data/meta_learner.db``.

Example workflow::

    ml = MetaLearner()
    # After each prediction cycle, record how well each model did:
    ml.record_model_performance("xgboost", "trending", {"vol": 0.3}, 0.72)
    ml.record_model_performance("lstm", "trending", {"vol": 0.3}, 0.65)

    # When a new cycle starts, ask which model to use:
    best = ml.select_model("trending", {"vol": 0.3})  # → "xgboost"
    rankings = ml.get_model_rankings("trending")
    # → [("xgboost", 0.72), ("lstm", 0.65)]

Pure Python + sqlite3.  No exchange or config dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join("data", "meta_learner.db")


@dataclass
class ModelRecord:
    """A single performance record for a model."""

    model_name: str
    regime: str
    features: Dict[str, float]
    accuracy: float
    timestamp: float


class MetaLearner:
    """Meta-learning model selector with SQLite persistence.

    Learns which ML model works best in which market regime by tracking
    historical performance.  Selection uses a combination of:

    1. **Regime match** — only considers records from the same regime.
    2. **Feature similarity** — weights records by how similar their
       feature context is to the current query (Euclidean distance in
       normalised feature space).
    3. **Recency weighting** — more recent records count more via
       exponential decay.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    decay_half_life_hours : float
        Half-life for exponential recency decay (default 168 = 1 week).
    min_records : int
        Minimum records for a model in a regime before it is eligible
        for selection (default 3).
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        decay_half_life_hours: float = 168.0,
        min_records: int = 3,
    ) -> None:
        self.db_path = db_path
        self.decay_half_life = decay_half_life_hours * 3600.0  # convert to seconds
        self.min_records = min_records

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        logger.info(
            "MetaLearner initialised: db=%s decay_half_life=%.0fh min_records=%d",
            self.db_path, decay_half_life_hours, self.min_records,
        )

    def _create_tables(self) -> None:
        """Create the performance tracking table if it does not exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                regime TEXT NOT NULL,
                features_json TEXT NOT NULL,
                accuracy REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_perf_regime
            ON model_performance(regime, model_name)
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Record performance
    # ------------------------------------------------------------------

    def record_model_performance(
        self,
        model_name: str,
        regime: str,
        features: Dict[str, float],
        accuracy: float,
    ) -> None:
        """Record a model's performance for later selection.

        Parameters
        ----------
        model_name : str
            Identifier for the model (e.g. "xgboost", "lstm", "random_forest").
        regime : str
            Market regime label (e.g. "trending", "mean_reverting", "volatile").
        features : dict[str, float]
            Feature context at the time of the prediction (e.g. volatility,
            volume, spread).
        accuracy : float
            Accuracy or score metric for this prediction cycle.  Higher
            is better; typically in [0, 1] but not enforced.
        """
        ts = time.time()
        features_json = json.dumps(features, sort_keys=True)

        self._conn.execute(
            """INSERT INTO model_performance
               (model_name, regime, features_json, accuracy, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (model_name, regime, features_json, accuracy, ts),
        )
        self._conn.commit()

        logger.debug(
            "Recorded: model=%s regime=%s accuracy=%.4f features=%s",
            model_name, regime, accuracy, features,
        )

    # ------------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------------

    def select_model(
        self,
        regime: str,
        features: Optional[Dict[str, float]] = None,
    ) -> str:
        """Select the best model for the given regime and features.

        Parameters
        ----------
        regime : str
            Current market regime.
        features : dict[str, float], optional
            Current feature context.  If provided, records are weighted
            by feature similarity.

        Returns
        -------
        str
            Name of the best-performing model.  Returns empty string if
            no qualifying records exist.
        """
        rankings = self.get_model_rankings(regime, features)
        if not rankings:
            logger.warning("No qualifying models for regime '%s'", regime)
            return ""

        best_model, best_score = rankings[0]
        logger.info(
            "Selected model '%s' for regime '%s' (score=%.4f)",
            best_model, regime, best_score,
        )
        return best_model

    def get_model_rankings(
        self,
        regime: str,
        features: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Rank all models for a given regime by expected accuracy.

        Parameters
        ----------
        regime : str
            Market regime to query.
        features : dict[str, float], optional
            Current features for similarity weighting.

        Returns
        -------
        list[tuple[str, float]]
            Sorted list of (model_name, expected_accuracy), best first.
        """
        cursor = self._conn.execute(
            """SELECT model_name, features_json, accuracy, timestamp
               FROM model_performance
               WHERE regime = ?
               ORDER BY timestamp DESC""",
            (regime,),
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        now = time.time()
        # Aggregate by model
        model_scores: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for model_name, features_json, accuracy, ts in rows:
            # Recency weight
            age = now - ts
            recency_weight = math.exp(-math.log(2) * age / self.decay_half_life)

            # Feature similarity weight
            feature_weight = 1.0
            if features:
                try:
                    record_features = json.loads(features_json)
                    feature_weight = self._feature_similarity(features, record_features)
                except (json.JSONDecodeError, TypeError):
                    pass

            combined_weight = recency_weight * feature_weight
            model_scores[model_name].append((accuracy, combined_weight))

        # Compute weighted average accuracy per model
        rankings: List[Tuple[str, float]] = []
        for model_name, score_weights in model_scores.items():
            if len(score_weights) < self.min_records:
                continue

            total_weight = sum(w for _, w in score_weights)
            if total_weight < 1e-12:
                continue

            weighted_acc = sum(a * w for a, w in score_weights) / total_weight
            rankings.append((model_name, weighted_acc))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_all_regimes(self) -> List[str]:
        """Return all distinct regimes in the database."""
        cursor = self._conn.execute(
            "SELECT DISTINCT regime FROM model_performance ORDER BY regime"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_model_history(
        self,
        model_name: str,
        regime: Optional[str] = None,
        limit: int = 100,
    ) -> List[ModelRecord]:
        """Retrieve recent performance records for a specific model.

        Parameters
        ----------
        model_name : str
            Model to query.
        regime : str, optional
            Filter to a specific regime.
        limit : int
            Maximum records to return.

        Returns
        -------
        list[ModelRecord]
        """
        if regime:
            cursor = self._conn.execute(
                """SELECT model_name, regime, features_json, accuracy, timestamp
                   FROM model_performance
                   WHERE model_name = ? AND regime = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (model_name, regime, limit),
            )
        else:
            cursor = self._conn.execute(
                """SELECT model_name, regime, features_json, accuracy, timestamp
                   FROM model_performance
                   WHERE model_name = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (model_name, limit),
            )

        records: List[ModelRecord] = []
        for name, reg, fj, acc, ts in cursor.fetchall():
            try:
                feats = json.loads(fj)
            except (json.JSONDecodeError, TypeError):
                feats = {}
            records.append(ModelRecord(
                model_name=name, regime=reg, features=feats,
                accuracy=acc, timestamp=ts,
            ))
        return records

    def get_record_count(self, regime: Optional[str] = None) -> int:
        """Return total number of performance records."""
        if regime:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM model_performance WHERE regime = ?",
                (regime,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM model_performance"
            )
        return cursor.fetchone()[0]

    def purge_old_records(self, max_age_days: int = 90) -> int:
        """Delete records older than *max_age_days*.

        Returns
        -------
        int
            Number of records deleted.
        """
        cutoff = time.time() - max_age_days * 86400
        cursor = self._conn.execute(
            "DELETE FROM model_performance WHERE timestamp < ?",
            (cutoff,),
        )
        self._conn.commit()
        deleted = cursor.rowcount
        logger.info("Purged %d records older than %d days", deleted, max_age_days)
        return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_similarity(
        query: Dict[str, float],
        record: Dict[str, float],
    ) -> float:
        """Compute a similarity score between two feature dicts.

        Uses a Gaussian kernel on the Euclidean distance of shared
        feature dimensions.  Missing features are ignored.

        Returns
        -------
        float
            Similarity in (0, 1].  1.0 = identical features.
        """
        shared_keys = set(query.keys()) & set(record.keys())
        if not shared_keys:
            return 1.0  # No shared features → no distance info → neutral

        diffs = []
        for k in shared_keys:
            q_val = query[k]
            r_val = record[k]
            # Normalise by max to get relative difference
            scale = max(abs(q_val), abs(r_val), 1e-8)
            diffs.append(((q_val - r_val) / scale) ** 2)

        dist_sq = sum(diffs) / len(diffs)
        # Gaussian kernel with bandwidth = 1
        return math.exp(-0.5 * dist_sq)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("MetaLearner database closed")

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
