"""Prediction confidence calibration — ensures model probabilities are reliable.

A model that says "80 % confident" should be correct 80 % of the time.
This module records (predicted_confidence, actual_outcome) pairs, computes
Expected Calibration Error (ECE), and applies Platt scaling or isotonic
regression to adjust raw confidences.

Persistence via SQLite so calibration data survives restarts.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn import
# ---------------------------------------------------------------------------
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.info("sklearn not available — ConfidenceCalibrator will use linear interpolation fallback")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    """Calibration quality report for a single model."""

    model_name: str
    bins: List[Dict[str, float]]  # [{"predicted_avg", "actual_rate", "count"}, ...]
    ece: float  # Expected Calibration Error (0 = perfect)
    overconfident: bool
    underconfident: bool
    total_predictions: int = 0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ConfidenceCalibrator
# ---------------------------------------------------------------------------

class ConfidenceCalibrator:
    """Record predictions, compute calibration metrics, and adjust confidences.

    Parameters
    ----------
    db_path : str | Path
        SQLite database for persistence.
        Default ``"data/confidence_calibration.db"``.
    min_samples_for_calibration : int
        Minimum number of predictions required before ``calibrate()`` adjusts
        confidences.  Below this count it returns the raw value.
    """

    def __init__(
        self,
        db_path: str | Path = "data/confidence_calibration.db",
        min_samples_for_calibration: int = 30,
    ) -> None:
        self._db_path = Path(db_path)
        self._min_samples = min_samples_for_calibration
        self._lock = threading.Lock()

        # In-memory cache: model_name -> list of (predicted, actual_bool)
        self._predictions: Dict[str, List[Tuple[float, bool]]] = {}
        # Fitted isotonic models per model_name
        self._fitted_models: Dict[str, object] = {}

        self._ensure_db()
        self._load_from_db()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        model_name: str,
        predicted_confidence: float,
        actual_outcome: bool,
    ) -> None:
        """Record a single (confidence, outcome) pair.

        Parameters
        ----------
        model_name : str
            Identifier for the model (e.g. ``"regime_classifier"``).
        predicted_confidence : float
            The model's stated confidence, in [0, 1].
        actual_outcome : bool
            Whether the predicted event actually occurred.
        """
        predicted_confidence = max(0.0, min(1.0, predicted_confidence))

        with self._lock:
            self._predictions.setdefault(model_name, []).append(
                (predicted_confidence, actual_outcome)
            )

        # Persist
        try:
            conn = sqlite3.connect(str(self._db_path))
            with conn:
                conn.execute(
                    "INSERT INTO predictions (model_name, predicted, actual, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (model_name, predicted_confidence, int(actual_outcome), time.time()),
                )
            conn.close()
        except Exception as exc:
            logger.warning("Failed to persist prediction: %s", exc)

        # Invalidate fitted model so next calibrate() re-fits
        self._fitted_models.pop(model_name, None)

        logger.debug(
            "Recorded prediction: model=%s conf=%.3f actual=%s",
            model_name, predicted_confidence, actual_outcome,
        )

    # ------------------------------------------------------------------
    # Calibration metrics
    # ------------------------------------------------------------------

    def get_calibration(
        self,
        model_name: str,
        n_bins: int = 10,
    ) -> CalibrationReport:
        """Compute calibration report for *model_name*.

        Parameters
        ----------
        model_name : str
            Model identifier.
        n_bins : int
            Number of bins to divide the [0, 1] confidence range into.

        Returns
        -------
        CalibrationReport
        """
        with self._lock:
            preds = list(self._predictions.get(model_name, []))

        if not preds:
            return CalibrationReport(
                model_name=model_name,
                bins=[],
                ece=0.0,
                overconfident=False,
                underconfident=False,
                total_predictions=0,
            )

        # Bin predictions
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        bins_data: List[Dict[str, float]] = []
        weighted_error = 0.0
        total_over = 0.0
        total_under = 0.0

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = [(p, a) for p, a in preds if lo <= p < hi or (i == n_bins - 1 and p == hi)]
            if not in_bin:
                continue

            pred_avg = sum(p for p, _ in in_bin) / len(in_bin)
            actual_rate = sum(1 for _, a in in_bin if a) / len(in_bin)

            bins_data.append({
                "predicted_avg": round(pred_avg, 4),
                "actual_rate": round(actual_rate, 4),
                "count": len(in_bin),
            })

            gap = abs(actual_rate - pred_avg)
            weighted_error += gap * len(in_bin)

            if pred_avg > actual_rate:
                total_over += gap * len(in_bin)
            else:
                total_under += gap * len(in_bin)

        ece = weighted_error / len(preds) if preds else 0.0

        return CalibrationReport(
            model_name=model_name,
            bins=bins_data,
            ece=round(ece, 6),
            overconfident=total_over > total_under,
            underconfident=total_under > total_over,
            total_predictions=len(preds),
        )

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model_name: str,
        raw_confidence: float,
    ) -> float:
        """Adjust *raw_confidence* using calibration data.

        Uses isotonic regression (sklearn) when available, or linear
        interpolation as a fallback.  Returns the raw value unchanged if
        insufficient data exists.

        Parameters
        ----------
        model_name : str
            Model identifier.
        raw_confidence : float
            The model's stated confidence in [0, 1].

        Returns
        -------
        float
            Calibrated confidence in [0, 1].
        """
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        with self._lock:
            preds = list(self._predictions.get(model_name, []))

        if len(preds) < self._min_samples:
            return raw_confidence

        # Try fitted model cache
        if model_name in self._fitted_models:
            return self._predict_with_model(model_name, raw_confidence)

        # Fit
        predicted = [p for p, _ in preds]
        actual = [float(a) for _, a in preds]

        if _HAS_SKLEARN:
            try:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(predicted, actual)
                self._fitted_models[model_name] = iso
                result = float(iso.predict([raw_confidence])[0])
                return max(0.0, min(1.0, result))
            except Exception as exc:
                logger.warning("Isotonic fit failed for %s: %s — using interpolation", model_name, exc)

        # Fallback: bin-based linear interpolation
        mapping = self._build_interpolation_mapping(predicted, actual)
        self._fitted_models[model_name] = mapping
        return self._interpolate(mapping, raw_confidence)

    def _predict_with_model(self, model_name: str, value: float) -> float:
        """Use cached fitted model to predict."""
        model = self._fitted_models[model_name]
        if _HAS_SKLEARN and hasattr(model, "predict"):
            try:
                result = float(model.predict([value])[0])  # type: ignore[union-attr]
                return max(0.0, min(1.0, result))
            except Exception as _e:
                logger.debug("confidence_calibrator error: %s", _e)
        # Interpolation mapping (list of (predicted_avg, actual_rate) tuples)
        if isinstance(model, list):
            return self._interpolate(model, value)
        return value

    # ------------------------------------------------------------------
    # Reliability diagram data
    # ------------------------------------------------------------------

    def get_reliability_diagram_data(
        self,
        model_name: str,
        n_bins: int = 10,
    ) -> Dict:
        """Return data suitable for plotting a reliability diagram.

        Returns
        -------
        dict
            Keys: ``model_name``, ``bins``, ``predicted_avgs``, ``actual_rates``,
            ``counts``, ``perfect_line`` ([0,1]).
        """
        report = self.get_calibration(model_name, n_bins)
        return {
            "model_name": model_name,
            "bins": report.bins,
            "predicted_avgs": [b["predicted_avg"] for b in report.bins],
            "actual_rates": [b["actual_rate"] for b in report.bins],
            "counts": [int(b["count"]) for b in report.bins],
            "perfect_line": [0.0, 1.0],
            "ece": report.ece,
        }

    @property
    def tracked_models(self) -> List[str]:
        """Return list of model names with recorded predictions."""
        with self._lock:
            return sorted(self._predictions.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_interpolation_mapping(
        self,
        predicted: List[float],
        actual: List[float],
        n_bins: int = 10,
    ) -> List[Tuple[float, float]]:
        """Build a (predicted_avg, actual_rate) mapping for linear interpolation."""
        pairs = sorted(zip(predicted, actual))
        mapping: List[Tuple[float, float]] = []
        bin_size = max(1, len(pairs) // n_bins)

        for i in range(0, len(pairs), bin_size):
            chunk = pairs[i: i + bin_size]
            pred_avg = sum(p for p, _ in chunk) / len(chunk)
            act_rate = sum(a for _, a in chunk) / len(chunk)
            mapping.append((pred_avg, act_rate))

        return mapping

    @staticmethod
    def _interpolate(mapping: List[Tuple[float, float]], value: float) -> float:
        """Linear interpolation on the mapping."""
        if not mapping:
            return value
        if len(mapping) == 1:
            return mapping[0][1]

        # Clamp to mapping range
        if value <= mapping[0][0]:
            return max(0.0, min(1.0, mapping[0][1]))
        if value >= mapping[-1][0]:
            return max(0.0, min(1.0, mapping[-1][1]))

        for i in range(len(mapping) - 1):
            x0, y0 = mapping[i]
            x1, y1 = mapping[i + 1]
            if x0 <= value <= x1:
                if x1 == x0:
                    return max(0.0, min(1.0, y0))
                t = (value - x0) / (x1 - x0)
                result = y0 + t * (y1 - y0)
                return max(0.0, min(1.0, result))

        return value

    def _ensure_db(self) -> None:
        """Create the SQLite tables if needed."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  model_name TEXT NOT NULL,"
                "  predicted REAL NOT NULL,"
                "  actual INTEGER NOT NULL,"
                "  timestamp REAL NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_model "
                "ON predictions (model_name, timestamp)"
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("Failed to initialise calibration DB at %s: %s", self._db_path, exc)

    def _load_from_db(self) -> None:
        """Load all prediction records from SQLite into memory."""
        if not self._db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(self._db_path))
            cur = conn.execute(
                "SELECT model_name, predicted, actual FROM predictions ORDER BY timestamp ASC"
            )
            count = 0
            with self._lock:
                for model_name, predicted, actual in cur:
                    self._predictions.setdefault(model_name, []).append(
                        (predicted, bool(actual))
                    )
                    count += 1
            conn.close()
            if count:
                logger.info("Loaded %d calibration records from %s", count, self._db_path)
        except Exception as exc:
            logger.warning("Failed to load calibration history: %s", exc)
