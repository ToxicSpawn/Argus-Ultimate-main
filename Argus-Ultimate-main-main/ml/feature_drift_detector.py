"""
ml/feature_drift_detector.py — Feature Importance Drift Detection

Monitors how feature importances shift over time for any ML model.
When features drift significantly, it signals that the model's learned
relationships may be stale and retraining is warranted.

Classes
-------
DriftReport     Immutable dataclass for a drift analysis result.
FeatureDriftDetector
    Record importance snapshots, detect drift, and advise on retraining.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftReport:
    """Result of a drift detection analysis."""
    model_name: str
    drifted_features: List[str]
    drift_scores: Dict[str, float]      # feature → absolute drift magnitude
    alert: bool                         # True if significant drift detected
    timestamp: str                      # ISO-8601 UTC


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS feature_importance_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name    TEXT    NOT NULL,
    importances   TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL
);
"""

_IDX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_fis_model ON feature_importance_snapshots(model_name);",
    "CREATE INDEX IF NOT EXISTS idx_fis_ts ON feature_importance_snapshots(timestamp);",
]


class FeatureDriftDetector:
    """Track feature importance drift across model snapshots.

    Parameters
    ----------
    db_path : str, optional
        Override the default ``data/feature_drift.db`` path.
    default_threshold : float
        Default drift threshold for ``detect_drift()``.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        default_threshold: float = 0.3,
    ) -> None:
        self._db_path = db_path or os.path.join(_DB_DIR, "feature_drift.db")
        self._default_threshold = default_threshold
        self._lock = threading.Lock()
        self._ensure_schema()
        log.info("FeatureDriftDetector initialised  db=%s  threshold=%.2f", self._db_path, self._default_threshold)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with self._lock:
            con = sqlite3.connect(self._db_path)
            try:
                con.execute(_CREATE_SQL)
                for idx in _IDX_SQL:
                    con.execute(idx)
                con.commit()
            finally:
                con.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_feature_importance(
        self,
        model_name: str,
        feature_importances: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Store a feature-importance snapshot for *model_name*.

        Parameters
        ----------
        model_name : str
            Identifier for the ML model (e.g. ``"regime_classifier"``).
        feature_importances : dict
            Mapping of feature name → importance value (any scale;
            normalised internally before comparison).
        timestamp : datetime, optional
            Snapshot time; defaults to now UTC.
        """
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        payload = json.dumps(feature_importances, sort_keys=True)

        with self._lock:
            con = self._connect()
            try:
                con.execute(
                    "INSERT INTO feature_importance_snapshots (model_name, importances, timestamp) "
                    "VALUES (?, ?, ?)",
                    (model_name, payload, ts),
                )
                con.commit()
            finally:
                con.close()

        log.debug("Recorded %d feature importances for %s", len(feature_importances), model_name)

    def detect_drift(
        self,
        model_name: str,
        window_days: int = 7,
        threshold: Optional[float] = None,
    ) -> DriftReport:
        """Compare oldest and newest snapshots inside *window_days*.

        Drift for a feature is the absolute change in its *normalised*
        importance (each snapshot's importances are scaled to sum to 1).

        Parameters
        ----------
        model_name : str
        window_days : int
            Look at snapshots from the last N days.
        threshold : float, optional
            Override the instance-level default threshold.

        Returns
        -------
        DriftReport
        """
        threshold = threshold if threshold is not None else self._default_threshold
        cutoff = datetime.now(timezone.utc).timestamp() - window_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    "SELECT importances, timestamp FROM feature_importance_snapshots "
                    "WHERE model_name = ? AND timestamp >= ? ORDER BY timestamp ASC",
                    (model_name, cutoff_iso),
                ).fetchall()
            finally:
                con.close()

        ts_now = datetime.now(timezone.utc).isoformat()

        if len(rows) < 2:
            log.debug("Not enough snapshots (%d) for drift detection on %s", len(rows), model_name)
            return DriftReport(
                model_name=model_name,
                drifted_features=[],
                drift_scores={},
                alert=False,
                timestamp=ts_now,
            )

        old_imp = json.loads(rows[0][0])
        new_imp = json.loads(rows[-1][0])

        old_norm = self._normalise(old_imp)
        new_norm = self._normalise(new_imp)

        all_features = sorted(set(old_norm) | set(new_norm))
        drift_scores: Dict[str, float] = {}
        drifted: List[str] = []

        for feat in all_features:
            old_val = old_norm.get(feat, 0.0)
            new_val = new_norm.get(feat, 0.0)
            drift = abs(new_val - old_val)
            drift_scores[feat] = round(drift, 6)
            if drift >= threshold:
                drifted.append(feat)

        alert = len(drifted) > 0

        report = DriftReport(
            model_name=model_name,
            drifted_features=sorted(drifted),
            drift_scores=drift_scores,
            alert=alert,
            timestamp=ts_now,
        )

        if alert:
            log.warning(
                "Feature drift detected for %s: %d features drifted (threshold=%.2f): %s",
                model_name, len(drifted), threshold, drifted,
            )
        else:
            log.debug("No significant drift for %s (%d features checked)", model_name, len(all_features))

        return report

    def get_declining_features(
        self,
        model_name: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return features whose importance has dropped the most.

        Returns a list of dicts: ``{"feature": ..., "old": ..., "new": ..., "change": ...}``.
        """
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    "SELECT importances FROM feature_importance_snapshots "
                    "WHERE model_name = ? ORDER BY timestamp ASC",
                    (model_name,),
                ).fetchall()
            finally:
                con.close()

        if len(rows) < 2:
            return []

        old_norm = self._normalise(json.loads(rows[0][0]))
        new_norm = self._normalise(json.loads(rows[-1][0]))

        declines: List[Dict[str, Any]] = []
        for feat in old_norm:
            old_v = old_norm[feat]
            new_v = new_norm.get(feat, 0.0)
            change = new_v - old_v
            if change < 0:
                declines.append({
                    "feature": feat,
                    "old": round(old_v, 6),
                    "new": round(new_v, 6),
                    "change": round(change, 6),
                })

        declines.sort(key=lambda d: d["change"])
        return declines[:top_n]

    def should_retrain(self, model_name: str) -> bool:
        """Return True if more than 3 features have drifted significantly.

        Uses the default window (7 days) and threshold.
        """
        report = self.detect_drift(model_name)
        retrain = len(report.drifted_features) > 3
        if retrain:
            log.info(
                "Retraining recommended for %s: %d drifted features",
                model_name, len(report.drifted_features),
            )
        return retrain

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(importances: Dict[str, float]) -> Dict[str, float]:
        """Normalise importance values to sum to 1."""
        total = sum(abs(v) for v in importances.values())
        if total == 0:
            return {k: 0.0 for k in importances}
        return {k: abs(v) / total for k, v in importances.items()}
