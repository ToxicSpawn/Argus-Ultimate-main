"""
Regime Ensemble — confidence-weighted combination of multiple regime detectors.

Combines:
  - HMM regime (ml.hmm_regime)
  - Autoencoder reconstruction signal (ml.autoencoder_regime)
  - Regime Predictor (adaptive.rolling_performance_feeder / regime_predictor)
  - XGBoost / pre-trained classifier (optional)

Each source votes on a regime label.  Votes are weighted by:
  1. Source-specific historical accuracy (updated on every regime transition)
  2. Current confidence score reported by the source

The ensemble breaks ties with a recency bias (most-recent confident reading wins).

Usage::

    ens = RegimeEnsemble()
    ens.update("hmm",          "TRENDING_UP",   confidence=0.72)
    ens.update("autoencoder",  "TRENDING_UP",   confidence=0.55)
    ens.update("regime_pred",  "MEAN_REVERT",   confidence=0.48)

    label, conf = ens.predict()
    # ("TRENDING_UP", 0.68)

    ens.record_outcome("TRENDING_UP")   # call when ground-truth regime is known
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Canonical regime labels
REGIME_LABELS = {
    "TRENDING_UP", "TRENDING_DOWN", "BREAKOUT", "BREAKDOWN",
    "MEAN_REVERT", "RANGE", "HIGH_VOL", "CRISIS", "NEUTRAL", "UNKNOWN",
}

# Minimum confidence to count a source vote
_MIN_CONFIDENCE = 0.20

# Minimum source accuracy to use (avoids silencing new sources immediately)
_MIN_SOURCE_ACCURACY = 0.30

# EMA decay for source accuracy updates
_ACCURACY_EMA = 0.15

# Maximum age of a source reading before it is considered stale
_MAX_AGE_SECONDS = 300.0  # 5 minutes


@dataclass
class SourceReading:
    """Latest reading from a single regime source."""
    source: str
    label: str
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def age(self) -> float:
        return time.time() - self.timestamp

    def is_stale(self) -> bool:
        return self.age() > _MAX_AGE_SECONDS


class RegimeEnsemble:
    """
    Confidence-weighted ensemble of multiple regime detection sources.

    Parameters
    ----------
    source_weights : dict | None
        Initial static weights per source name.  Updated via accuracy tracking.
        If None, all sources start with equal weight 1.0.
    min_sources : int
        Minimum number of non-stale sources required to return a non-UNKNOWN result.
    """

    def __init__(
        self,
        source_weights: Optional[Dict[str, float]] = None,
        min_sources: int = 1,
    ) -> None:
        self._readings: Dict[str, SourceReading] = {}
        self._base_weights: Dict[str, float] = dict(source_weights or {})
        self._accuracy: Dict[str, float] = {}   # EMA accuracy per source
        self._outcome_history: deque = deque(maxlen=200)  # (timestamp, predicted, actual)
        self.min_sources = min_sources
        self._last_prediction: Optional[Tuple[str, float]] = None

    # ── Feed ──────────────────────────────────────────────────────────────

    def update(self, source: str, label: str, confidence: float) -> None:
        """
        Record a regime reading from a named source.

        Parameters
        ----------
        source : str
            Source identifier (e.g. "hmm", "autoencoder", "regime_pred").
        label : str
            Regime label (e.g. "TRENDING_UP", "MEAN_REVERT").
        confidence : float
            Source's self-reported confidence in [0, 1].
        """
        label = label.upper() if label else "UNKNOWN"
        confidence = float(np.clip(confidence, 0.0, 1.0))
        self._readings[source] = SourceReading(
            source=source,
            label=label,
            confidence=confidence,
        )

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self) -> Tuple[str, float]:
        """
        Return (regime_label, ensemble_confidence) from weighted majority vote.

        Confidence = weighted_votes_for_winner / total_weighted_votes.
        Returns ("UNKNOWN", 0.0) if fewer than min_sources non-stale readings exist.
        """
        active = [r for r in self._readings.values() if not r.is_stale()]
        if len(active) < self.min_sources:
            return ("UNKNOWN", 0.0)

        # Aggregate weighted votes
        votes: Dict[str, float] = defaultdict(float)
        total_weight = 0.0

        for r in active:
            if r.confidence < _MIN_CONFIDENCE:
                continue
            source_acc = self._accuracy.get(r.source, 0.5)
            base_w = self._base_weights.get(r.source, 1.0)
            # Weight = base_weight × source_accuracy × confidence
            effective_w = base_w * max(_MIN_SOURCE_ACCURACY, source_acc) * r.confidence
            votes[r.label] += effective_w
            total_weight += effective_w

        if total_weight < 1e-9:
            return ("UNKNOWN", 0.0)

        winner = max(votes, key=votes.__getitem__)
        ensemble_conf = float(votes[winner] / total_weight)

        self._last_prediction = (winner, ensemble_conf)
        logger.debug(
            "RegimeEnsemble: %s (conf=%.2f) from %d sources — votes=%s",
            winner, ensemble_conf, len(active),
            {k: round(v / total_weight, 3) for k, v in votes.items()},
        )
        return winner, ensemble_conf

    def record_outcome(self, actual_label: str) -> None:
        """
        Record the ground-truth regime label to update source accuracy.

        Call this after a regime has been confirmed (e.g. after N bars in the
        new regime, or when a strategy's regime-conditional result is known).

        Parameters
        ----------
        actual_label : str
            The true regime that occurred.
        """
        actual_label = actual_label.upper()
        now = time.time()

        # Update per-source accuracy based on their most recent reading
        for source, r in self._readings.items():
            if r.is_stale():
                continue
            correct = 1.0 if r.label == actual_label else 0.0
            prev = self._accuracy.get(source, 0.5)
            self._accuracy[source] = (1.0 - _ACCURACY_EMA) * prev + _ACCURACY_EMA * correct

        if self._last_prediction is not None:
            self._outcome_history.append((now, self._last_prediction[0], actual_label))

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_source_accuracies(self) -> Dict[str, float]:
        """Return EMA accuracy per source."""
        return dict(self._accuracy)

    def snapshot(self) -> dict:
        """Return current ensemble state for advisory."""
        active = [r for r in self._readings.values() if not r.is_stale()]
        label, conf = self.predict()
        return {
            "regime": label,
            "confidence": round(conf, 4),
            "n_sources": len(active),
            "sources": {
                r.source: {
                    "label": r.label,
                    "confidence": round(r.confidence, 4),
                    "age_s": round(r.age(), 1),
                    "accuracy": round(self._accuracy.get(r.source, 0.5), 4),
                }
                for r in active
            },
        }
