from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ADWINState:
    """State for ADWIN (Adaptive Windowing) drift detector."""
    window: Deque[float]
    total: float
    n: int
    delta: float
    min_samples: int
    
    @classmethod
    def create(cls, delta: float = 0.002, min_samples: int = 30) -> "ADWINState":
        return cls(
            window=deque(),
            total=0.0,
            n=0,
            delta=delta,
            min_samples=min_samples,
        )
    
    def add_sample(self, sample: float) -> None:
        self.window.append(sample)
        self.total += sample
        self.n += 1
    
    def cut_window(self, cut_point: int) -> None:
        for _ in range(cut_point):
            if self.window:
                removed = self.window.popleft()
                self.total -= removed
                self.n -= 1


@dataclass
class DriftMetrics:
    feature_drift_score: float
    prediction_drift_score: float
    concept_drift_detected: bool
    drift_magnitude: str
    affected_features: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConceptDriftDetector:
    """Detect feature, prediction, and concept drift for ML models.

    The detector maintains sliding windows for incoming features, predictions,
    and labels. When enough history is available, the oldest window acts as the
    reference distribution and the newest window acts as the current
    distribution. Optional integrations can bootstrap reference data from a
    FeatureStore, emit notifications through an alert system, and trigger model
    retraining through a ModelManager.
    """

    _EPSILON = 1e-12

    def __init__(
        self,
        model_name: str,
        *,
        feature_store: Any = None,
        model_manager: Any = None,
        alert_system: Any = None,
        train_fn: Any = None,
        feature_names: Optional[Sequence[str]] = None,
        window_size: int = 250,
        min_samples: int = 50,
        feature_drift_threshold: float = 0.1,
        prediction_drift_threshold: float = 0.2,
        retrain_threshold: float = 0.35,
        feature_store_key: Optional[str] = None,
        feature_set: str = "default",
    ) -> None:
        self.model_name = model_name
        self.feature_store = feature_store
        self.model_manager = model_manager
        self.alert_system = alert_system
        self.train_fn = train_fn
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.window_size = max(int(window_size), 10)
        self.min_samples = max(int(min_samples), 10)
        self.feature_drift_threshold = float(feature_drift_threshold)
        self.prediction_drift_threshold = float(prediction_drift_threshold)
        self.retrain_threshold = float(retrain_threshold)
        self.feature_store_key = feature_store_key or model_name
        self.feature_set = feature_set

        self._feature_windows: Deque[Dict[str, np.ndarray]] = deque(maxlen=2)
        self._prediction_windows: Deque[np.ndarray] = deque(maxlen=2)
        self._label_windows: Deque[np.ndarray] = deque(maxlen=2)
        self._reference_features: Optional[Dict[str, np.ndarray]] = self._load_reference_features()
        self._reference_predictions: Optional[np.ndarray] = None
        self._reference_errors: Optional[np.ndarray] = None
        self._last_metrics: Optional[DriftMetrics] = None
        self._last_visualization: Dict[str, Any] = {}
        
        # ADWIN state for fast drift detection
        self._adwin_error: ADWINState = ADWINState.create(delta=0.002, min_samples=30)
        self._adwin_prediction: ADWINState = ADWINState.create(delta=0.002, min_samples=30)
        self._adwin_drift_count: int = 0
        self._adwin_last_drift_at: Optional[datetime] = None

    def compute_ks_statistic(self, reference: Sequence[float], current: Sequence[float]) -> float:
        """Return the two-sample Kolmogorov-Smirnov statistic."""
        ref = self._clean_array(reference)
        cur = self._clean_array(current)
        if ref.size == 0 or cur.size == 0:
            return 0.0

        ref_sorted = np.sort(ref)
        cur_sorted = np.sort(cur)
        points = np.sort(np.concatenate([ref_sorted, cur_sorted]))
        ref_cdf = np.searchsorted(ref_sorted, points, side="right") / ref_sorted.size
        cur_cdf = np.searchsorted(cur_sorted, points, side="right") / cur_sorted.size
        return float(np.max(np.abs(ref_cdf - cur_cdf)))

    def compute_psi(self, reference: Sequence[float], current: Sequence[float]) -> float:
        """Return Population Stability Index between two samples."""
        ref = self._clean_array(reference)
        cur = self._clean_array(current)
        if ref.size == 0 or cur.size == 0:
            return 0.0

        if np.allclose(ref, ref[0]) and np.allclose(cur, cur[0]):
            return 0.0 if np.isclose(ref[0], cur[0]) else float(abs(cur[0] - ref[0]))

        bins = self._compute_psi_bins(ref)
        ref_hist = self._distribution(ref, bins)
        cur_hist = self._distribution(cur, bins)
        psi = np.sum((cur_hist - ref_hist) * np.log((cur_hist + self._EPSILON) / (ref_hist + self._EPSILON)))
        return float(max(psi, 0.0))

    def detect_feature_drift(
        self,
        reference_data: Any,
        current_data: Any,
        threshold: float = 0.1,
    ) -> List[str]:
        """Return feature names whose drift score exceeds *threshold*."""
        reference_features = self._coerce_feature_mapping(reference_data)
        current_features = self._coerce_feature_mapping(current_data)

        affected: List[str] = []
        feature_scores: Dict[str, Dict[str, float]] = {}
        for name in sorted(set(reference_features) & set(current_features)):
            ref = reference_features[name]
            cur = current_features[name]
            if ref.size < self.min_samples or cur.size < self.min_samples:
                continue
            ks_score = self.compute_ks_statistic(ref, cur)
            psi_score = self.compute_psi(ref, cur)
            js_score = self._compute_js_divergence(ref, cur)
            feature_scores[name] = {
                "ks": ks_score,
                "psi": psi_score,
                "js": js_score,
                "combined": ks_score,
            }
            if feature_scores[name]["combined"] >= threshold:
                affected.append(name)

        self._last_visualization["feature_metrics"] = feature_scores
        return affected

    def detect_prediction_drift(self, reference_preds: Sequence[float], current_preds: Sequence[float]) -> float:
        """Return aggregate drift score across prediction distributions."""
        ref = self._clean_array(reference_preds)
        cur = self._clean_array(current_preds)
        if ref.size == 0 or cur.size == 0:
            return 0.0

        ks_score = self.compute_ks_statistic(ref, cur)
        psi_score = self.compute_psi(ref, cur)
        js_score = self._compute_js_divergence(ref, cur)
        score = float(max(ks_score, js_score, min(psi_score, 1.0)))
        self._last_visualization["prediction_metrics"] = {
            "ks": ks_score,
            "psi": psi_score,
            "js": js_score,
            "combined": score,
        }
        return score
    
    def detect_drift_adwin(self, error: float, prediction: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect drift using ADWIN (Adaptive Windowing) algorithm.
        
        ADWIN automatically adjusts window size and detects drift by comparing
        sub-window means using Hoeffding's inequality. Faster than KS/PSI methods.
        
        Args:
            error: Prediction error (|y_true - y_pred|)
            prediction: Optional prediction value for separate ADWIN tracking
            
        Returns:
            Dict with drift detection results:
            - error_drift: bool - drift detected in error stream
            - prediction_drift: bool - drift detected in prediction stream
            - confidence: float - drift confidence [0, 1]
            - window_size: int - current ADWIN window size
        """
        result = {
            "error_drift": False,
            "prediction_drift": False,
            "confidence": 0.0,
            "window_size": self._adwin_error.n,
        }
        
        # Check error stream for drift
        error_drift = self._check_adwin_single(self._adwin_error, error)
        if error_drift:
            result["error_drift"] = True
            result["confidence"] += 0.6
            self._adwin_drift_count += 1
            self._adwin_last_drift_at = datetime.now(timezone.utc)
            logger.warning(
                "ConceptDriftDetector[%s]: ADWIN detected error drift at sample %d",
                self.model_name, self._adwin_error.n,
            )
        
        # Check prediction stream if provided
        if prediction is not None:
            pred_drift = self._check_adwin_single(self._adwin_prediction, prediction)
            if pred_drift:
                result["prediction_drift"] = True
                result["confidence"] += 0.4
        
        result["confidence"] = min(1.0, result["confidence"])
        result["window_size"] = self._adwin_error.n
        
        return result
    
    def _check_adwin_single(self, state: ADWINState, new_sample: float) -> bool:
        """
        ADWIN (Adaptive Windowing) algorithm for single stream.
        
        Maintains a sliding window and detects drift by comparing
        sub-window means using Hoeffding's inequality.
        """
        state.add_sample(new_sample)
        
        if state.n < 2 * state.min_samples:
            return False
        
        n = len(state.window)
        samples = list(state.window)
        
        best_cut = -1
        max_epsilon = 0.0
        
        for cut in range(state.min_samples, n - state.min_samples, max(1, n // 50)):
            n0 = cut
            n1 = n - cut
            sum0 = sum(samples[:cut])
            sum1 = sum(samples[cut:])
            
            mean0 = sum0 / n0
            mean1 = sum1 / n1
            
            n_harmonic = 2.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt((1.0 / n_harmonic) * np.log(2.0 / state.delta))
            
            if abs(mean0 - mean1) > epsilon:
                if abs(mean0 - mean1) > max_epsilon:
                    max_epsilon = abs(mean0 - mean1)
                    best_cut = cut
        
        if best_cut > 0:
            state.cut_window(best_cut)
            return True
        
        return False
    
    def get_adwin_stats(self) -> Dict[str, Any]:
        """Return ADWIN detector statistics."""
        return {
            "error_window_size": self._adwin_error.n,
            "prediction_window_size": self._adwin_prediction.n,
            "total_drifts": self._adwin_drift_count,
            "last_drift_at": self._adwin_last_drift_at.isoformat() if self._adwin_last_drift_at else None,
            "error_window_mean": float(np.mean(list(self._adwin_error.window))) if self._adwin_error.window else 0.0,
        }

    def detect_concept_drift(
        self,
        features: Any,
        predictions: Sequence[float],
        labels: Sequence[float],
    ) -> DriftMetrics:
        """Detect drift using sliding windows of features, predictions, and labels.
        
        Now includes ADWIN for faster, more adaptive drift detection alongside
        the existing KS/PSI/JS methods.
        """
        current_features = self._coerce_feature_mapping(features)
        current_predictions = self._clean_array(predictions)
        current_labels = self._clean_array(labels)

        if current_predictions.size == 0 or current_labels.size == 0:
            metrics = self._build_empty_metrics()
            self._last_metrics = metrics
            return metrics

        self._push_windows(current_features, current_predictions, current_labels)
        reference_features, reference_predictions, reference_errors = self._resolve_reference_state()

        if reference_features is None or reference_predictions is None or reference_errors is None:
            metrics = self._build_empty_metrics()
            self._last_metrics = metrics
            logger.debug("ConceptDriftDetector[%s]: insufficient reference history", self.model_name)
            return metrics

        affected_features = self.detect_feature_drift(
            reference_features,
            current_features,
            threshold=self.feature_drift_threshold,
        )
        feature_drift_score = self._aggregate_feature_score(reference_features, current_features)

        prediction_drift_score = self.detect_prediction_drift(reference_predictions, current_predictions)
        current_errors = self._compute_prediction_errors(current_predictions, current_labels)
        error_drift_score = self.detect_prediction_drift(reference_errors, current_errors)
        combined_prediction_score = float(max(prediction_drift_score, error_drift_score))
        
        # ADWIN fast drift detection on errors and predictions
        mean_error = float(np.mean(current_errors)) if current_errors.size > 0 else 0.0
        mean_prediction = float(np.mean(current_predictions)) if current_predictions.size > 0 else 0.0
        adwin_result = self.detect_drift_adwin(mean_error, mean_prediction)
        
        # Combine ADWIN with traditional methods for final decision
        adwin_drift = adwin_result["error_drift"] or adwin_result["prediction_drift"]

        concept_drift_detected = (
            feature_drift_score >= self.feature_drift_threshold
            or combined_prediction_score >= self.prediction_drift_threshold
            or bool(affected_features)
            or adwin_drift  # ADWIN detected drift
        )
        magnitude = self._classify_drift(max(feature_drift_score, combined_prediction_score))
        metrics = DriftMetrics(
            feature_drift_score=feature_drift_score,
            prediction_drift_score=combined_prediction_score,
            concept_drift_detected=concept_drift_detected,
            drift_magnitude=magnitude,
            affected_features=affected_features,
        )
        self._last_metrics = metrics
        self._last_visualization["summary"] = {
            "model_name": self.model_name,
            "feature_drift_score": feature_drift_score,
            "prediction_drift_score": combined_prediction_score,
            "error_drift_score": error_drift_score,
            "affected_features": affected_features,
            "timestamp": metrics.timestamp.isoformat(),
            "adwin_drift": adwin_result,
        }
        self._last_visualization["feature_distributions"] = self._build_feature_distribution_export(
            reference_features,
            current_features,
            affected_features,
        )

        if concept_drift_detected:
            logger.warning(
                "ConceptDriftDetector[%s]: concept drift detected magnitude=%s feature_score=%.4f prediction_score=%.4f adwin=%s affected=%s",
                self.model_name,
                metrics.drift_magnitude,
                metrics.feature_drift_score,
                metrics.prediction_drift_score,
                adwin_drift,
                metrics.affected_features,
            )
            self._dispatch_alert(metrics)
            self._trigger_retraining(metrics)
        else:
            logger.debug(
                "ConceptDriftDetector[%s]: no concept drift feature_score=%.4f prediction_score=%.4f adwin=%s",
                self.model_name,
                metrics.feature_drift_score,
                metrics.prediction_drift_score,
                adwin_drift,
            )

        return metrics

    def generate_drift_alert(self, metrics: DriftMetrics) -> dict:
        """Create an alert payload consumable by Argus alerting systems."""
        severity = self._severity_from_magnitude(metrics.drift_magnitude)
        title = f"Concept drift detected for {self.model_name}"
        message = (
            f"Model drift magnitude is {metrics.drift_magnitude}. "
            f"Feature drift={metrics.feature_drift_score:.4f}, "
            f"prediction drift={metrics.prediction_drift_score:.4f}."
        )
        return {
            "title": title,
            "message": message,
            "severity": severity,
            "category": "ml",
            "model_name": self.model_name,
            "metrics": {
                "feature_drift_score": metrics.feature_drift_score,
                "prediction_drift_score": metrics.prediction_drift_score,
                "concept_drift_detected": metrics.concept_drift_detected,
                "drift_magnitude": metrics.drift_magnitude,
                "affected_features": metrics.affected_features,
                "timestamp": metrics.timestamp.isoformat(),
            },
            "retrain_recommended": self.should_retrain(metrics),
        }

    def should_retrain(self, metrics: DriftMetrics) -> bool:
        """Return True when drift is strong enough to justify retraining."""
        return (
            metrics.concept_drift_detected
            and (
                metrics.prediction_drift_score >= self.retrain_threshold
                or metrics.feature_drift_score >= self.retrain_threshold
                or metrics.drift_magnitude == "severe"
                or len(metrics.affected_features) >= 3
            )
        )

    def export_visualization_data(self) -> Dict[str, Any]:
        """Return JSON-serialisable drift data for dashboards and plotting."""
        payload = {
            "model_name": self.model_name,
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "last_metrics": asdict(self._last_metrics) if self._last_metrics is not None else None,
            "visualization": self._last_visualization,
        }
        if payload["last_metrics"] is not None:
            payload["last_metrics"]["timestamp"] = self._last_metrics.timestamp.isoformat()
        return payload

    def _load_reference_features(self) -> Optional[Dict[str, np.ndarray]]:
        if self.feature_store is None or not hasattr(self.feature_store, "get_features"):
            return None
        try:
            reference_frame = self.feature_store.get_features(self.feature_store_key, self.feature_set)
            if reference_frame is None:
                return None
            reference = self._coerce_feature_mapping(reference_frame)
            if reference:
                logger.info(
                    "ConceptDriftDetector[%s]: bootstrapped %d reference features from FeatureStore",
                    self.model_name,
                    len(reference),
                )
            return reference or None
        except Exception as exc:
            logger.warning("ConceptDriftDetector[%s]: failed to load FeatureStore reference: %s", self.model_name, exc)
            return None

    def _push_windows(
        self,
        features: Dict[str, np.ndarray],
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        trimmed_features = {
            name: values[-self.window_size:]
            for name, values in features.items()
            if values.size > 0
        }
        trimmed_predictions = predictions[-self.window_size:]
        trimmed_labels = labels[-self.window_size:]
        self._feature_windows.append(trimmed_features)
        self._prediction_windows.append(trimmed_predictions)
        self._label_windows.append(trimmed_labels)

    def _resolve_reference_state(
        self,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[np.ndarray], Optional[np.ndarray]]:
        current_predictions = self._prediction_windows[-1] if self._prediction_windows else None
        current_labels = self._label_windows[-1] if self._label_windows else None
        if current_predictions is None or current_labels is None:
            return None, None, None

        reference_features = self._reference_features
        if reference_features is None and len(self._feature_windows) >= 2:
            reference_features = self._feature_windows[0]
            self._reference_features = reference_features

        if self._reference_predictions is None and len(self._prediction_windows) >= 2:
            self._reference_predictions = self._prediction_windows[0]

        if self._reference_errors is None and len(self._label_windows) >= 2:
            self._reference_errors = self._compute_prediction_errors(
                self._prediction_windows[0],
                self._label_windows[0],
            )

        if reference_features is None:
            return None, None, None
        if self._reference_predictions is None or self._reference_errors is None:
            return None, None, None
        return reference_features, self._reference_predictions, self._reference_errors

    def _aggregate_feature_score(
        self,
        reference_features: Mapping[str, np.ndarray],
        current_features: Mapping[str, np.ndarray],
    ) -> float:
        scores: List[float] = []
        for name in sorted(set(reference_features) & set(current_features)):
            ref = reference_features[name]
            cur = current_features[name]
            if ref.size < self.min_samples or cur.size < self.min_samples:
                continue
            scores.append(
                self.compute_ks_statistic(ref, cur)
            )
        if not scores:
            return 0.0
        return float(np.max(scores))

    def _compute_js_divergence(self, reference: Sequence[float], current: Sequence[float]) -> float:
        ref = self._clean_array(reference)
        cur = self._clean_array(current)
        if ref.size == 0 or cur.size == 0:
            return 0.0
        bins = self._compute_bins(ref, cur)
        ref_hist = self._distribution(ref, bins)
        cur_hist = self._distribution(cur, bins)
        midpoint = 0.5 * (ref_hist + cur_hist)
        js = 0.5 * (self._kl_divergence(ref_hist, midpoint) + self._kl_divergence(cur_hist, midpoint))
        return float(max(js, 0.0))

    def _compute_prediction_errors(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        size = min(predictions.size, labels.size)
        if size == 0:
            return np.asarray([], dtype=float)
        preds = predictions[:size]
        labs = labels[:size]
        return np.abs(labs - preds)

    def _build_feature_distribution_export(
        self,
        reference_features: Mapping[str, np.ndarray],
        current_features: Mapping[str, np.ndarray],
        affected_features: Sequence[str],
    ) -> Dict[str, Any]:
        export: Dict[str, Any] = {}
        for name in affected_features:
            ref = reference_features.get(name)
            cur = current_features.get(name)
            if ref is None or cur is None or ref.size == 0 or cur.size == 0:
                continue
            bins = self._compute_bins(ref, cur)
            ref_hist, edges = np.histogram(ref, bins=bins, density=True)
            cur_hist, _ = np.histogram(cur, bins=bins, density=True)
            export[name] = {
                "bin_edges": edges.tolist(),
                "reference_histogram": np.nan_to_num(ref_hist, nan=0.0, posinf=0.0, neginf=0.0).tolist(),
                "current_histogram": np.nan_to_num(cur_hist, nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            }
        return export

    def _dispatch_alert(self, metrics: DriftMetrics) -> None:
        alert_payload = self.generate_drift_alert(metrics)
        if self.alert_system is None:
            return

        try:
            from core.alerts.alert_models import AlertEvent
        except Exception:
            AlertEvent = None  # type: ignore[assignment]

        try:
            if AlertEvent is not None and hasattr(self.alert_system, "enqueue"):
                event = AlertEvent.warning(
                    alert_payload["title"],
                    alert_payload["message"],
                    source="ml.drift_detector",
                    tags=["ml", "drift", self.model_name],
                    extra=alert_payload,
                )
                self.alert_system.enqueue(event)
                return

            if hasattr(self.alert_system, "send_alert"):
                self.alert_system.send_alert(alert_payload)
                return

            if hasattr(self.alert_system, "send"):
                result = self.alert_system.send(alert_payload)
                if asyncio.iscoroutine(result):
                    self._run_coroutine_best_effort(result)
                return

            if hasattr(self.alert_system, "warning"):
                result = self.alert_system.warning(alert_payload["title"], alert_payload["message"])
                if asyncio.iscoroutine(result):
                    self._run_coroutine_best_effort(result)
        except Exception as exc:
            logger.warning("ConceptDriftDetector[%s]: alert dispatch failed: %s", self.model_name, exc)

    def _trigger_retraining(self, metrics: DriftMetrics) -> None:
        if not self.should_retrain(metrics):
            return
        if self.model_manager is None or self.train_fn is None:
            logger.info(
                "ConceptDriftDetector[%s]: retraining recommended but trigger unavailable",
                self.model_name,
            )
            return

        try:
            if hasattr(self.model_manager, "trigger_retrain"):
                self.model_manager.trigger_retrain(self.model_name, self.train_fn)
                logger.info("ConceptDriftDetector[%s]: triggered automated retraining", self.model_name)
        except Exception as exc:
            logger.warning("ConceptDriftDetector[%s]: retrain trigger failed: %s", self.model_name, exc)

    def _build_empty_metrics(self) -> DriftMetrics:
        return DriftMetrics(
            feature_drift_score=0.0,
            prediction_drift_score=0.0,
            concept_drift_detected=False,
            drift_magnitude="none",
            affected_features=[],
        )

    def _classify_drift(self, score: float) -> str:
        if score < self.feature_drift_threshold:
            return "none"
        if score < 0.2:
            return "minor"
        if score < self.retrain_threshold:
            return "moderate"
        return "severe"

    def _severity_from_magnitude(self, magnitude: str) -> str:
        return {
            "none": "info",
            "minor": "warning",
            "moderate": "error",
            "severe": "critical",
        }.get(magnitude, "warning")

    def _coerce_feature_mapping(self, data: Any) -> Dict[str, np.ndarray]:
        if data is None:
            return {}

        if isinstance(data, Mapping):
            return {
                str(name): self._clean_array(values)
                for name, values in data.items()
                if self._clean_array(values).size > 0
            }

        if hasattr(data, "to_dict") and hasattr(data, "columns"):
            columns = list(getattr(data, "columns"))
            return {
                str(name): self._clean_array(getattr(data[name], "to_numpy", lambda: data[name])())
                for name in columns
                if self._clean_array(getattr(data[name], "to_numpy", lambda: data[name])()).size > 0
            }

        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            name = self.feature_names[0] if self.feature_names else "feature_0"
            return {name: self._clean_array(array)}

        if array.ndim != 2:
            raise ValueError("features must be a mapping, DataFrame-like object, or 1D/2D array")

        names = self.feature_names or [f"feature_{idx}" for idx in range(array.shape[1])]
        return {
            str(name): self._clean_array(array[:, idx])
            for idx, name in enumerate(names[: array.shape[1]])
            if self._clean_array(array[:, idx]).size > 0
        }

    def _clean_array(self, values: Iterable[float]) -> np.ndarray:
        array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(-1)
        if array.size == 0:
            return array.astype(float)
        mask = np.isfinite(array)
        return array[mask].astype(float)

    def _compute_bins(self, reference: np.ndarray, current: np.ndarray) -> np.ndarray:
        data = np.concatenate([reference, current])
        if data.size == 0:
            return np.linspace(0.0, 1.0, 11)
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if np.isclose(data_min, data_max):
            data_min -= 0.5
            data_max += 0.5
        unique_count = len(np.unique(data))
        bin_count = max(10, min(20, unique_count))
        return np.linspace(data_min, data_max, bin_count + 1)

    def _compute_psi_bins(self, reference: np.ndarray) -> np.ndarray:
        if reference.size == 0:
            return np.linspace(0.0, 1.0, 11)
        quantiles = np.linspace(0.0, 1.0, 11)
        bins = np.quantile(reference, quantiles)
        bins = np.unique(bins)
        if bins.size < 2:
            value = float(reference[0]) if reference.size else 0.0
            return np.asarray([value - 0.5, value + 0.5], dtype=float)
        bins[0] = bins[0] - self._EPSILON
        bins[-1] = bins[-1] + self._EPSILON
        return bins

    def _distribution(self, values: np.ndarray, bins: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(values, bins=bins)
        hist = hist.astype(float) + self._EPSILON
        return hist / np.sum(hist)

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log((p + self._EPSILON) / (q + self._EPSILON))))

    def _run_coroutine_best_effort(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return
        loop.create_task(coro)
