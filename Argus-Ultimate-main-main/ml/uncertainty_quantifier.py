"""
Uncertainty quantification utilities for ARGUS ML predictions.

This module provides a lightweight uncertainty layer that can wrap existing
ARGUS models with minimal assumptions about their interface.  It supports:

1. Bayesian / stochastic predictive sampling when a model exposes a sampling API
2. Monte Carlo dropout for PyTorch-compatible models
3. Deep ensemble aggregation across multiple models
4. Calibration metrics (ECE, MCE) and simple post-hoc uncertainty scaling
5. Conformal prediction intervals from historical residuals
6. Integration helpers for uncertainty-aware position sizing and signal quality

The implementation is deliberately defensive because ARGUS contains a mix of
NumPy, callable, and PyTorch-backed models.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.stats import norm  # type: ignore[import-untyped]

    _HAS_SCIPY = True
except ImportError:
    norm = None  # type: ignore[assignment]
    _HAS_SCIPY = False


_Z_95 = float(norm.ppf(0.975)) if _HAS_SCIPY else 1.959963984540054
_Z_99 = float(norm.ppf(0.995)) if _HAS_SCIPY else 2.5758293035489004


@dataclass
class PredictionWithUncertainty:
    prediction: float
    mean: float
    std: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_99: Tuple[float, float]
    uncertainty_type: str
    calibration_score: float


class BayesianUncertainty:
    """Estimate predictive uncertainty for ARGUS ML models."""

    def __init__(
        self,
        calibration_bins: int = 10,
        min_std: float = 1e-8,
        residual_window: int = 1000,
    ) -> None:
        self.calibration_bins = max(2, int(calibration_bins))
        self.min_std = max(float(min_std), 1e-12)
        self.residual_window = max(10, int(residual_window))
        self._residual_history: List[float] = []

    def predict_with_uncertainty(
        self,
        model: Any,
        X: Any,
        n_samples: int = 100,
    ) -> PredictionWithUncertainty:
        """
        Estimate uncertainty from a model's stochastic predictive interface.

        Preferred interfaces tried in order:
        - model.sample_predictive(X, n_samples=...)
        - model.predict_samples(X, n_samples=...)
        - repeated model.predict(X, training=True / stochastic=True)
        - Monte Carlo dropout for torch-like models
        - deterministic fallback with zero uncertainty
        """
        samples: Optional[np.ndarray] = None

        for method_name in ("sample_predictive", "predict_samples"):
            sampler = getattr(model, method_name, None)
            if callable(sampler):
                try:
                    samples = self._coerce_samples(sampler(X, n_samples=n_samples))
                    logger.debug("Uncertainty estimated via %s", method_name)
                    break
                except TypeError:
                    try:
                        samples = self._coerce_samples(sampler(X, n_samples))
                        logger.debug("Uncertainty estimated via %s positional call", method_name)
                        break
                    except Exception as exc:
                        logger.debug("%s failed: %s", method_name, exc)
                except Exception as exc:
                    logger.debug("%s failed: %s", method_name, exc)

        if samples is None:
            samples = self._attempt_stochastic_predict_calls(model, X, n_samples)

        if samples is None and self._looks_like_torch_model(model):
            return self.monte_carlo_dropout(model, X, n_forward_passes=n_samples)

        if samples is None:
            raw_prediction = self._extract_numeric_prediction(self._call_model(model, X))
            logger.info(
                "Falling back to deterministic uncertainty estimate for %s",
                type(model).__name__,
            )
            return self._build_prediction_result(
                samples=np.asarray([raw_prediction], dtype=float),
                uncertainty_type="total",
            )

        return self._build_prediction_result(samples=samples, uncertainty_type="total")

    def monte_carlo_dropout(
        self,
        model: Any,
        X: Any,
        n_forward_passes: int = 50,
    ) -> PredictionWithUncertainty:
        """Estimate epistemic uncertainty with Monte Carlo dropout."""
        n_forward_passes = max(2, int(n_forward_passes))
        samples: List[float] = []

        if self._looks_like_torch_model(model):
            try:
                torch = __import__("torch")
            except ImportError:
                logger.warning(
                    "Torch is unavailable for Monte Carlo dropout on %s; using fallback path",
                    type(model).__name__,
                )
                torch = None

            if torch is None:
                base = self._extract_numeric_prediction(self._call_model(model, X))
                samples = [base] * n_forward_passes
                return self._build_prediction_result(
                    samples=np.asarray(samples, dtype=float),
                    uncertainty_type="epistemic",
                )

            model_was_training = bool(model.training)
            dropout_modules = self._collect_dropout_modules(model)
            dropout_training_states = [bool(module.training) for module in dropout_modules]
            model.eval()
            for module in dropout_modules:
                module.train()
            try:
                with torch.no_grad():
                    tensor_x = self._to_torch_input(X, torch)
                    for _ in range(n_forward_passes):
                        raw = model(tensor_x)
                        samples.append(self._extract_numeric_prediction(raw))
            finally:
                for module, was_training in zip(dropout_modules, dropout_training_states):
                    if was_training:
                        module.train()
                    else:
                        module.eval()
                if model_was_training:
                    model.train()
                else:
                    model.eval()
        else:
            stochastic_samples = self._attempt_stochastic_predict_calls(model, X, n_forward_passes)
            if stochastic_samples is not None:
                samples = stochastic_samples.tolist()
            else:
                base = self._extract_numeric_prediction(self._call_model(model, X))
                samples = [base] * n_forward_passes
                logger.warning(
                    "Model %s does not appear to support dropout; using deterministic fallback",
                    type(model).__name__,
                )

        return self._build_prediction_result(
            samples=np.asarray(samples, dtype=float),
            uncertainty_type="epistemic",
        )

    def deep_ensemble_predict(
        self,
        models: List[Any],
        X: Any,
    ) -> PredictionWithUncertainty:
        """Aggregate multiple models into an ensemble uncertainty estimate."""
        if not models:
            raise ValueError("deep_ensemble_predict requires at least one model")

        predictions = [self._extract_numeric_prediction(self._call_model(model, X)) for model in models]
        return self._build_prediction_result(
            samples=np.asarray(predictions, dtype=float),
            uncertainty_type="epistemic",
        )

    def compute_calibration_error(
        self,
        predictions: Sequence[float],
        actuals: Sequence[float],
    ) -> float:
        """Return Expected Calibration Error (ECE) for probability-like inputs."""
        metrics = self._compute_calibration_metrics(predictions, actuals)
        return metrics["ece"]

    def compute_sharpness(self, uncertainties: Sequence[float]) -> float:
        """Return sharpness as the mean uncertainty width / magnitude."""
        arr = np.asarray(list(uncertainties), dtype=float)
        if arr.size == 0:
            return 0.0
        arr = np.abs(arr[np.isfinite(arr)])
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr))

    def calibrate_uncertainty(
        self,
        predictions: Sequence[float],
        actuals: Sequence[float],
    ) -> Dict[str, Any]:
        """
        Fit a simple post-hoc uncertainty calibration from residuals.

        Returns ECE, MCE, conformal quantiles, and a scaling factor that can be
        applied to predicted standard deviations.
        """
        preds = np.asarray(list(predictions), dtype=float)
        acts = np.asarray(list(actuals), dtype=float)
        if preds.size == 0 or acts.size == 0 or preds.size != acts.size:
            raise ValueError("predictions and actuals must be non-empty and equal length")

        residuals = acts - preds
        abs_residuals = np.abs(residuals)
        self._extend_residual_history(abs_residuals)

        metrics = self._compute_calibration_metrics(preds, acts) if self._is_probability_calibration_input(preds, acts) else None
        uncertainty_scale = float(np.mean(abs_residuals) / max(np.std(preds), self.min_std))
        q95 = self._conformal_quantile(abs_residuals, 0.95)
        q99 = self._conformal_quantile(abs_residuals, 0.99)
        coverage_95 = float(np.mean(abs_residuals <= q95))
        coverage_99 = float(np.mean(abs_residuals <= q99))

        result = {
            "ece": metrics["ece"] if metrics is not None else float(abs(0.95 - coverage_95)),
            "mce": metrics["mce"] if metrics is not None else float(abs(0.99 - coverage_99)),
            "bin_stats": metrics["bins"] if metrics is not None else [],
            "residual_mean": float(np.mean(abs_residuals)),
            "residual_std": float(np.std(residuals)),
            "uncertainty_scale": uncertainty_scale,
            "conformal_q95": q95,
            "conformal_q99": q99,
            "coverage_95": coverage_95,
            "coverage_99": coverage_99,
            "calibration_mode": "probability" if metrics is not None else "regression",
        }
        logger.info(
            "Uncertainty calibration updated: ece=%.6f mce=%.6f scale=%.6f",
            result["ece"],
            result["mce"],
            result["uncertainty_scale"],
        )
        return result

    def uncertainty_adjusted_position_size(
        self,
        base_position_size: float,
        uncertainty: PredictionWithUncertainty,
        max_reduction: float = 0.75,
    ) -> Dict[str, float]:
        """Risk helper: reduce size as predictive uncertainty rises."""
        base = max(0.0, float(base_position_size))
        reduction_cap = min(max(float(max_reduction), 0.0), 0.99)
        relative_uncertainty = uncertainty.std / max(abs(uncertainty.mean), 1.0)
        reduction = min(relative_uncertainty, reduction_cap)
        adjusted = base * (1.0 - reduction)
        return {
            "base_position_size": base,
            "adjusted_position_size": adjusted,
            "uncertainty_penalty": reduction,
            "confidence_multiplier": 1.0 - reduction,
        }

    def score_signal_quality(
        self,
        uncertainty: PredictionWithUncertainty,
        baseline_confidence: Optional[float] = None,
    ) -> Dict[str, float]:
        """Signal-quality helper compatible with confidence-based gates."""
        baseline = 1.0 if baseline_confidence is None else max(0.0, min(1.0, float(baseline_confidence)))
        uncertainty_penalty = uncertainty.std / max(abs(uncertainty.mean), 1.0)
        interval_width = uncertainty.confidence_interval_95[1] - uncertainty.confidence_interval_95[0]
        quality = baseline * max(0.0, 1.0 - min(uncertainty_penalty, 1.0))
        quality *= max(0.0, 1.0 - min(interval_width, 1.0))
        calibration_bonus = max(0.0, 1.0 - min(uncertainty.calibration_score, 1.0))
        quality *= calibration_bonus
        return {
            "quality": float(max(0.0, min(1.0, quality))),
            "uncertainty_penalty": float(uncertainty_penalty),
            "interval_width_95": float(interval_width),
            "calibration_penalty": float(uncertainty.calibration_score),
        }

    def _attempt_stochastic_predict_calls(
        self,
        model: Any,
        X: Any,
        n_samples: int,
    ) -> Optional[np.ndarray]:
        samples: List[float] = []
        predict = getattr(model, "predict", None)
        if not callable(predict):
            return None

        signature = None
        try:
            signature = inspect.signature(predict)
        except (TypeError, ValueError):
            signature = None

        for _ in range(max(2, int(n_samples))):
            try:
                if signature and "training" in signature.parameters:
                    raw = predict(X, training=True)
                elif signature and "stochastic" in signature.parameters:
                    raw = predict(X, stochastic=True)
                elif signature and "mc_dropout" in signature.parameters:
                    raw = predict(X, mc_dropout=True)
                else:
                    return None
                samples.append(self._extract_numeric_prediction(raw))
            except Exception as exc:
                logger.debug("stochastic predict attempt failed: %s", exc)
                return None

        return np.asarray(samples, dtype=float) if samples else None

    def _build_prediction_result(
        self,
        samples: np.ndarray,
        uncertainty_type: str,
    ) -> PredictionWithUncertainty:
        samples = self._coerce_samples(samples)
        mean = float(np.mean(samples))
        std = float(max(np.std(samples), self.min_std))
        prediction = float(samples[-1]) if samples.size else mean

        ci_95 = self._normal_interval(mean, std, _Z_95)
        ci_99 = self._normal_interval(mean, std, _Z_99)

        conformal_95 = self._get_conformal_interval(mean, 0.95)
        conformal_99 = self._get_conformal_interval(mean, 0.99)
        ci_95 = self._merge_intervals(ci_95, conformal_95)
        ci_99 = self._merge_intervals(ci_99, conformal_99)

        calibration_score = self._interval_miscalibration_score(mean, std)
        return PredictionWithUncertainty(
            prediction=prediction,
            mean=mean,
            std=std,
            confidence_interval_95=ci_95,
            confidence_interval_99=ci_99,
            uncertainty_type=uncertainty_type,
            calibration_score=calibration_score,
        )

    def _compute_calibration_metrics(
        self,
        predictions: Sequence[float],
        actuals: Sequence[float],
    ) -> Dict[str, Any]:
        preds = np.asarray(list(predictions), dtype=float)
        acts = np.asarray(list(actuals), dtype=float)
        if preds.size == 0 or acts.size == 0 or preds.size != acts.size:
            raise ValueError("predictions and actuals must be non-empty and equal length")

        if not self._is_probability_calibration_input(preds, acts):
            raise ValueError(
                "ECE/MCE calibration requires probability-like predictions and actuals in [0, 1]"
            )

        probs = preds
        labels = acts
        edges = np.linspace(0.0, 1.0, self.calibration_bins + 1)

        ece = 0.0
        mce = 0.0
        bins: List[Dict[str, float]] = []

        for idx in range(self.calibration_bins):
            lo = edges[idx]
            hi = edges[idx + 1]
            if idx == self.calibration_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)
            if not np.any(mask):
                continue

            bin_probs = probs[mask]
            bin_labels = labels[mask]
            confidence = float(np.mean(bin_probs))
            accuracy = float(np.mean(bin_labels))
            gap = abs(confidence - accuracy)
            weight = float(mask.sum() / len(probs))
            ece += gap * weight
            mce = max(mce, gap)
            bins.append(
                {
                    "low": float(lo),
                    "high": float(hi),
                    "confidence": confidence,
                    "accuracy": accuracy,
                    "gap": float(gap),
                    "count": float(mask.sum()),
                }
            )

        return {"ece": float(ece), "mce": float(mce), "bins": bins}

    def _interval_miscalibration_score(self, mean: float, std: float) -> float:
        if not self._residual_history:
            return 0.0
        residuals = np.asarray(self._residual_history, dtype=float)
        nominal_95 = _Z_95 * std
        if nominal_95 <= 0:
            return 1.0
        empirical_covered = float(np.mean(residuals <= nominal_95))
        return float(abs(0.95 - empirical_covered))

    def _get_conformal_interval(self, center: float, coverage: float) -> Tuple[float, float]:
        if not self._residual_history:
            return (center, center)
        q = self._conformal_quantile(np.asarray(self._residual_history, dtype=float), coverage)
        return (float(center - q), float(center + q))

    @staticmethod
    def _is_probability_calibration_input(predictions: np.ndarray, actuals: np.ndarray) -> bool:
        return bool(
            np.all(np.isfinite(predictions))
            and np.all(np.isfinite(actuals))
            and np.all((predictions >= 0.0) & (predictions <= 1.0))
            and np.all((actuals >= 0.0) & (actuals <= 1.0))
        )

    @staticmethod
    def _merge_intervals(
        interval_a: Tuple[float, float],
        interval_b: Tuple[float, float],
    ) -> Tuple[float, float]:
        return (float(min(interval_a[0], interval_b[0])), float(max(interval_a[1], interval_b[1])))

    def _conformal_quantile(self, residuals: np.ndarray, coverage: float) -> float:
        arr = np.abs(np.asarray(residuals, dtype=float))
        if arr.size == 0:
            return 0.0
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        quantile = min(max(float(coverage), 0.0), 0.9999)
        return float(np.quantile(arr, quantile, method="higher"))

    def _extend_residual_history(self, residuals: Iterable[float]) -> None:
        self._residual_history.extend(float(abs(x)) for x in residuals if np.isfinite(x))
        if len(self._residual_history) > self.residual_window:
            self._residual_history = self._residual_history[-self.residual_window :]

    @staticmethod
    def _normal_interval(mean: float, std: float, z_score: float) -> Tuple[float, float]:
        width = z_score * std
        return (float(mean - width), float(mean + width))

    @staticmethod
    def _coerce_samples(values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("No finite predictive samples available")
        return finite.astype(float)

    def _call_model(self, model: Any, X: Any) -> Any:
        if hasattr(model, "predict") and callable(model.predict):
            return model.predict(X)
        if callable(model):
            return model(X)
        raise TypeError(f"Unsupported model type for prediction: {type(model).__name__}")

    def _extract_numeric_prediction(self, raw: Any) -> float:
        if hasattr(raw, "prediction"):
            return float(getattr(raw, "prediction"))
        if hasattr(raw, "confidence") and hasattr(raw, "magnitude_pct") and hasattr(raw, "direction"):
            direction = str(getattr(raw, "direction", "up")).lower()
            magnitude = float(getattr(raw, "magnitude_pct", 0.0))
            sign = 1.0 if direction in {"up", "buy", "long"} else -1.0
            return sign * magnitude
        if isinstance(raw, tuple) and raw:
            return self._extract_numeric_prediction(raw[0])
        if isinstance(raw, list) and raw:
            return self._extract_numeric_prediction(raw[0])
        if isinstance(raw, np.ndarray):
            return float(raw.item()) if raw.ndim == 0 else float(raw.reshape(-1)[0])

        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Could not extract numeric prediction from {type(raw).__name__}") from exc

    @staticmethod
    def _looks_like_torch_model(model: Any) -> bool:
        return all(hasattr(model, attr) for attr in ("train", "eval", "__call__")) and hasattr(model, "training")

    @staticmethod
    def _collect_dropout_modules(model: Any) -> List[Any]:
        modules_fn = getattr(model, "modules", None)
        if not callable(modules_fn):
            return []

        dropout_modules: List[Any] = []
        for module in modules_fn():
            class_name = type(module).__name__.lower()
            if "dropout" in class_name:
                dropout_modules.append(module)
                continue

            p_value = getattr(module, "p", None)
            train_fn = getattr(module, "train", None)
            eval_fn = getattr(module, "eval", None)
            if isinstance(p_value, (int, float)) and callable(train_fn) and callable(eval_fn):
                dropout_modules.append(module)

        return dropout_modules

    def _to_torch_input(self, X: Any, torch_module: Any) -> Any:
        if hasattr(X, "detach"):
            return X
        arr = np.asarray(X, dtype=np.float32)
        tensor = torch_module.tensor(arr, dtype=torch_module.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor


__all__ = ["PredictionWithUncertainty", "BayesianUncertainty"]
