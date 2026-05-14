"""
Regime Transition Forecaster — Markov-chain-based regime prediction.

Maintains a transition probability matrix from observed regime sequences,
augmented by feature-based adjustments (e.g., volatility, momentum, volume).
Predicts the most likely next regime and the probability of transition
within a configurable time horizon.

Usage:
    from adaptive.regime_forecaster import RegimeForecaster

    rf = RegimeForecaster()
    rf.update("bull", {"volatility": 0.02, "momentum": 0.5})
    rf.update("bull", {"volatility": 0.03, "momentum": 0.4})
    rf.update("crisis", {"volatility": 0.08, "momentum": -0.3})
    forecast = rf.predict_transition("crisis", horizon_hours=4)
    logger.info(forecast.predicted_regime, forecast.probability)
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TransitionForecast:
    """Forecast of a regime transition."""

    current_regime: str
    predicted_regime: str
    probability: float
    confidence: float
    key_features: List[str] = field(default_factory=list)
    horizon_hours: int = 4
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class _RegimeObservation:
    """Internal record of a regime observation."""

    regime: str
    features: Dict[str, float]
    timestamp: float  # epoch seconds


# ---------------------------------------------------------------------------
# Feature importance weights for regime transitions
# ---------------------------------------------------------------------------

_FEATURE_WEIGHTS: Dict[str, float] = {
    "volatility": 0.30,
    "momentum": 0.20,
    "volume_ratio": 0.15,
    "spread": 0.10,
    "fear_greed": 0.10,
    "funding_rate": 0.08,
    "correlation": 0.07,
}


# ---------------------------------------------------------------------------
# RegimeForecaster
# ---------------------------------------------------------------------------

class RegimeForecaster:
    """Predict regime transitions using Markov chains + feature adjustments.

    Maintains an empirical transition probability matrix from observed regime
    sequences.  Feature vectors at transition points allow the model to adjust
    base Markov probabilities based on current market conditions.

    Parameters
    ----------
    db_path : str or Path
        SQLite persistence path.
    min_observations : int
        Minimum transition observations before predictions are made.
    feature_adjustment_strength : float
        How much features shift the base Markov probability (0 to 1).
    """

    def __init__(
        self,
        db_path: str = "data/regime_forecasts.db",
        *,
        min_observations: int = 5,
        feature_adjustment_strength: float = 0.3,
    ) -> None:
        self.db_path = Path(db_path)
        self.min_observations = min_observations
        self.feature_adjustment_strength = max(0.0, min(1.0, feature_adjustment_strength))

        # In-memory state
        self._observations: List[_RegimeObservation] = []
        self._transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._regime_durations: Dict[str, List[float]] = defaultdict(list)  # hours per stay
        self._feature_profiles: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self._lock = threading.Lock()
        self._ensure_db()
        self._load_history()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create SQLite tables if absent."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_observations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime      TEXT    NOT NULL,
                    features    TEXT    DEFAULT '{}',
                    ts          REAL    NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transition_counts (
                    from_regime TEXT NOT NULL,
                    to_regime   TEXT NOT NULL,
                    count       INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (from_regime, to_regime)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    current_regime  TEXT NOT NULL,
                    predicted       TEXT NOT NULL,
                    probability     REAL NOT NULL,
                    confidence      REAL NOT NULL,
                    horizon_hours   INTEGER NOT NULL,
                    ts              TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_ts ON regime_observations(ts)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _load_history(self) -> None:
        """Load transition counts and recent observations from SQLite."""
        try:
            with self._connect() as conn:
                # Load transition counts
                cursor = conn.execute("SELECT from_regime, to_regime, count FROM transition_counts")
                for row in cursor.fetchall():
                    self._transition_counts[row[0]][row[1]] = row[2]

                # Load recent observations (last 1000)
                cursor = conn.execute(
                    "SELECT regime, features, ts FROM regime_observations ORDER BY ts DESC LIMIT 1000"
                )
                rows = cursor.fetchall()
                rows.reverse()
                for row in rows:
                    obs = _RegimeObservation(
                        regime=row[0],
                        features=json.loads(row[1]) if row[1] else {},
                        timestamp=float(row[2]),
                    )
                    self._observations.append(obs)

            logger.info(
                "RegimeForecaster: loaded %d observations, %d regime types from DB",
                len(self._observations),
                len(self._transition_counts),
            )
        except Exception:
            logger.exception("RegimeForecaster: error loading history from DB")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        regime_label: str,
        features: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a regime observation with associated features.

        Parameters
        ----------
        regime_label : str
            Current regime label (e.g. "bull", "bear", "crisis", "normal").
        features : dict or None
            Feature vector (e.g. {"volatility": 0.03, "momentum": 0.5}).
        timestamp : float or None
            Epoch seconds.  Defaults to now.
        """
        if features is None:
            features = {}
        if timestamp is None:
            timestamp = time.time()

        obs = _RegimeObservation(regime=regime_label, features=features, timestamp=timestamp)

        with self._lock:
            # Track transition from previous observation
            if self._observations:
                prev = self._observations[-1]
                self._transition_counts[prev.regime][regime_label] += 1

                # Duration tracking
                duration_hours = (timestamp - prev.timestamp) / 3600.0
                if prev.regime == regime_label:
                    # Same regime — accumulate
                    if self._regime_durations[regime_label]:
                        self._regime_durations[regime_label][-1] += duration_hours
                    else:
                        self._regime_durations[regime_label].append(duration_hours)
                else:
                    # New regime
                    self._regime_durations[regime_label].append(0.0)

            self._observations.append(obs)

            # Feature profiling (features observed when entering a regime)
            for feat_name, feat_val in features.items():
                self._feature_profiles[regime_label][feat_name].append(feat_val)

        # Persist
        self._persist_observation(obs)
        self._persist_transitions()

        logger.debug(
            "RegimeForecaster: updated regime='%s' features=%d ts=%.0f",
            regime_label, len(features), timestamp,
        )

    def predict_transition(
        self,
        current_regime: str,
        horizon_hours: int = 4,
    ) -> TransitionForecast:
        """Predict the most likely next regime transition.

        Combines the empirical Markov transition matrix with feature-based
        adjustments to produce a probability distribution over next regimes.

        Parameters
        ----------
        current_regime : str
            The current market regime.
        horizon_hours : int
            Prediction horizon in hours.

        Returns
        -------
        TransitionForecast
        """
        with self._lock:
            transition_row = dict(self._transition_counts.get(current_regime, {}))
            total_transitions = sum(transition_row.values())
            current_features = (
                self._observations[-1].features if self._observations else {}
            )

        # Base Markov probabilities
        if total_transitions < self.min_observations:
            # Not enough data — return uniform/stay prediction
            return TransitionForecast(
                current_regime=current_regime,
                predicted_regime=current_regime,
                probability=0.5,
                confidence=0.1,
                key_features=[],
                horizon_hours=horizon_hours,
            )

        probabilities: Dict[str, float] = {}
        for regime, count in transition_row.items():
            probabilities[regime] = count / total_transitions

        # Feature-based adjustment
        adjusted = self._adjust_with_features(probabilities, current_features, current_regime)

        # Find top prediction
        if not adjusted:
            predicted = current_regime
            probability = 0.5
        else:
            predicted = max(adjusted, key=adjusted.get)  # type: ignore[arg-type]
            probability = adjusted[predicted]

        # Confidence based on sample size and probability separation
        sorted_probs = sorted(adjusted.values(), reverse=True) if adjusted else [0.5]
        if len(sorted_probs) >= 2:
            separation = sorted_probs[0] - sorted_probs[1]
        else:
            separation = sorted_probs[0]
        sample_factor = min(1.0, total_transitions / 50.0)
        confidence = min(1.0, separation * sample_factor)

        # Key features driving the prediction
        key_features = self._get_key_features(current_features, predicted)

        # Horizon adjustment: longer horizon → less confident (mean-revert)
        horizon_decay = math.exp(-0.05 * max(0, horizon_hours - 1))
        confidence *= horizon_decay

        forecast = TransitionForecast(
            current_regime=current_regime,
            predicted_regime=predicted,
            probability=round(probability, 4),
            confidence=round(confidence, 4),
            key_features=key_features,
            horizon_hours=horizon_hours,
        )

        self._persist_forecast(forecast)
        logger.info(
            "RegimeForecaster: %s → %s (p=%.4f conf=%.4f horizon=%dh)",
            current_regime, predicted, probability, confidence, horizon_hours,
        )
        return forecast

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Return the empirical transition probability matrix.

        Returns
        -------
        dict of dict
            Mapping from_regime -> to_regime -> probability.
        """
        with self._lock:
            matrix: Dict[str, Dict[str, float]] = {}
            for from_r, targets in self._transition_counts.items():
                total = sum(targets.values())
                if total == 0:
                    continue
                matrix[from_r] = {
                    to_r: round(count / total, 4)
                    for to_r, count in targets.items()
                }
        return matrix

    def get_regime_duration_stats(self) -> Dict[str, Dict[str, float]]:
        """Return average duration statistics per regime.

        Returns
        -------
        dict
            Mapping regime -> {"avg_duration_hours": float, "std_hours": float}.
        """
        with self._lock:
            stats: Dict[str, Dict[str, float]] = {}
            for regime, durations in self._regime_durations.items():
                if not durations:
                    stats[regime] = {"avg_duration_hours": 0.0, "std_hours": 0.0}
                    continue
                avg = sum(durations) / len(durations)
                var = sum((d - avg) ** 2 for d in durations) / len(durations)
                std = var ** 0.5
                stats[regime] = {
                    "avg_duration_hours": round(avg, 2),
                    "std_hours": round(std, 2),
                }
        return stats

    # ------------------------------------------------------------------
    # Feature adjustment
    # ------------------------------------------------------------------

    def _adjust_with_features(
        self,
        base_probs: Dict[str, float],
        features: Dict[str, float],
        current_regime: str,
    ) -> Dict[str, float]:
        """Adjust Markov probabilities using current feature values.

        Compares each feature to the historical mean for each target regime.
        If a feature value is closer to the mean of a target regime, that
        regime's probability is boosted proportionally.

        Parameters
        ----------
        base_probs : dict
            Base Markov probabilities.
        features : dict
            Current feature values.
        current_regime : str
            Current regime (for context).

        Returns
        -------
        dict
            Adjusted probability distribution (sums to ~1).
        """
        if not features or not self._feature_profiles:
            return base_probs

        adjustments: Dict[str, float] = defaultdict(float)
        for regime in base_probs:
            for feat_name, feat_val in features.items():
                weight = _FEATURE_WEIGHTS.get(feat_name, 0.05)
                profile = self._feature_profiles.get(regime, {}).get(feat_name, [])
                if not profile:
                    continue
                hist_mean = sum(profile) / len(profile)
                hist_std = (sum((v - hist_mean) ** 2 for v in profile) / len(profile)) ** 0.5
                if hist_std < 1e-12:
                    continue
                # Z-score proximity: higher when current value is close to regime's mean
                z = abs(feat_val - hist_mean) / hist_std
                proximity = math.exp(-0.5 * z ** 2)  # Gaussian proximity
                adjustments[regime] += weight * proximity

        # Blend base probabilities with feature adjustments
        alpha = self.feature_adjustment_strength
        adjusted: Dict[str, float] = {}
        total_adj = sum(adjustments.values()) or 1.0

        for regime, base_p in base_probs.items():
            feat_p = adjustments.get(regime, 0.0) / total_adj
            adjusted[regime] = (1 - alpha) * base_p + alpha * feat_p

        # Normalise
        total = sum(adjusted.values())
        if total > 1e-12:
            adjusted = {r: p / total for r, p in adjusted.items()}

        return adjusted

    def _get_key_features(
        self,
        features: Dict[str, float],
        predicted_regime: str,
    ) -> List[str]:
        """Identify the features most influential in the prediction.

        Returns the top 3 features by weight that are present in the
        current feature vector.

        Parameters
        ----------
        features : dict
            Current feature values.
        predicted_regime : str
            The predicted next regime.

        Returns
        -------
        list of str
            Up to 3 feature names.
        """
        scored: List[Tuple[float, str]] = []
        for feat_name, feat_val in features.items():
            weight = _FEATURE_WEIGHTS.get(feat_name, 0.05)
            scored.append((weight * abs(feat_val), feat_name))
        scored.sort(reverse=True)
        return [name for _, name in scored[:3]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_observation(self, obs: _RegimeObservation) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO regime_observations (regime, features, ts) VALUES (?, ?, ?)",
                    (obs.regime, json.dumps(obs.features), obs.timestamp),
                )

    def _persist_transitions(self) -> None:
        with self._lock:
            with self._connect() as conn:
                for from_r, targets in self._transition_counts.items():
                    for to_r, count in targets.items():
                        conn.execute(
                            """
                            INSERT INTO transition_counts (from_regime, to_regime, count)
                            VALUES (?, ?, ?)
                            ON CONFLICT(from_regime, to_regime) DO UPDATE SET count = ?
                            """,
                            (from_r, to_r, count, count),
                        )

    def _persist_forecast(self, f: TransitionForecast) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO forecasts_log
                        (current_regime, predicted, probability, confidence, horizon_hours, ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (f.current_regime, f.predicted_regime, f.probability,
                     f.confidence, f.horizon_hours, f.timestamp),
                )
