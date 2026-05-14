"""
Tests for ML pipeline features:
  1. InferenceService (ml/inference_service.py)
  2. Experiment tracking in ModelManager (ml/model_manager.py)
  3. OnlineLearner drift callback (ml/online_learner.py)
  4. Dead stub replacements (ml/ensemble.py, ml/feature_engineer.py)
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false, reportUninitializedInstanceVariable=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalSubscript=false

import math
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from ml.inference_service import InferenceResult, InferenceService
from ml.model_manager import ModelManager
from ml.online_learner import DriftAlert, OnlineLearner


# ===========================================================================
# Helpers
# ===========================================================================

class _FakeModel:
    """Minimal model with a .predict() method."""

    def __init__(self, return_value=0.42):
        self._rv = return_value

    def predict(self, features):
        return self._rv


class _FakeClassifier:
    """Returns a (prediction, confidence) tuple."""

    def predict(self, features):
        return ("BUY", 0.95)


class _BrokenModel:
    """Always raises on predict."""

    def predict(self, features):
        raise RuntimeError("model exploded")


def _make_model_manager_with_fake(model_name="alpha_model", model_obj=None):
    """Create a ModelManager and inject a fake loaded object."""
    mm = ModelManager(models_dir="/tmp/fake_models")
    if model_obj is None:
        model_obj = _FakeModel()
    mm._loaded_objects[model_name] = model_obj
    mm._registry[model_name].is_loaded = True
    return mm


# ===========================================================================
# 1. InferenceService tests
# ===========================================================================

class TestInferenceService:
    """Tests for ml.inference_service.InferenceService."""

    def test_basic_predict(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        result = svc.predict("alpha_model", [1.0, 2.0, 3.0])
        assert isinstance(result, InferenceResult)
        assert result.prediction == 0.42
        assert result.latency_ms >= 0.0

    def test_predict_returns_confidence(self):
        mm = _make_model_manager_with_fake(model_obj=_FakeClassifier())
        svc = InferenceService(mm)
        result = svc.predict("alpha_model", [1.0])
        assert result.prediction == "BUY"
        assert result.confidence == 0.95

    def test_predict_cache_hit(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        r1 = svc.predict("alpha_model", [1.0])
        assert r1.cache_hit is False
        r2 = svc.predict("alpha_model", [2.0])
        assert r2.cache_hit is True

    def test_predict_nan_raises(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        with pytest.raises(ValueError, match="NaN"):
            svc.predict("alpha_model", [float("nan"), 1.0])

    def test_predict_inf_raises(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        with pytest.raises(ValueError, match="Inf"):
            svc.predict("alpha_model", [float("inf")])

    def test_predict_model_failure_returns_fallback(self):
        mm = _make_model_manager_with_fake(model_obj=_BrokenModel())
        svc = InferenceService(mm)
        result = svc.predict("alpha_model", [1.0])
        assert result.prediction == 0.0
        assert result.confidence == 0.0

    def test_predict_classifier_fallback(self):
        mm = _make_model_manager_with_fake(
            model_name="regime_classifier", model_obj=_BrokenModel()
        )
        svc = InferenceService(mm)
        result = svc.predict("regime_classifier", [1.0])
        assert result.prediction == "HOLD"

    def test_get_fallback_regression(self):
        svc = InferenceService(MagicMock())
        assert svc.get_fallback("alpha_model") == 0.0

    def test_get_fallback_classifier(self):
        svc = InferenceService(MagicMock())
        assert svc.get_fallback("regime_classifier") == "HOLD"

    def test_predict_batch(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        results = svc.predict_batch("alpha_model", [[1.0], [2.0], [3.0]])
        assert len(results) == 3
        assert all(isinstance(r, InferenceResult) for r in results)

    def test_clear_cache_specific(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        svc.predict("alpha_model", [1.0])
        assert "alpha_model" in svc._cache
        svc.clear_cache("alpha_model")
        assert "alpha_model" not in svc._cache

    def test_clear_cache_all(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        svc.predict("alpha_model", [1.0])
        svc.clear_cache()
        assert len(svc._cache) == 0

    def test_get_stats(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        svc.predict("alpha_model", [1.0])
        svc.predict("alpha_model", [2.0])
        stats = svc.get_stats()
        assert stats["total_predictions"] == 2
        assert stats["hit_rate"] == 0.5  # first miss, second hit
        assert stats["avg_latency_ms"] >= 0.0
        assert stats["errors"] == 0

    def test_get_stats_with_errors(self):
        mm = _make_model_manager_with_fake(model_obj=_BrokenModel())
        svc = InferenceService(mm)
        svc.predict("alpha_model", [1.0])
        stats = svc.get_stats()
        assert stats["errors"] == 1

    def test_model_version_tracked(self):
        mm = _make_model_manager_with_fake()
        mm._registry["alpha_model"].version = 7
        svc = InferenceService(mm)
        result = svc.predict("alpha_model", [1.0])
        assert result.model_version == 7

    def test_predict_unknown_model_raises(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        svc = InferenceService(mm)
        with pytest.raises(RuntimeError, match="could not be loaded"):
            svc.predict("nonexistent_model", [1.0])


# ===========================================================================
# 2. Experiment tracking tests
# ===========================================================================

class TestExperimentTracking:
    """Tests for ModelManager experiment tracking."""

    def test_start_experiment_returns_run_id(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        run_id = mm.start_experiment("regime_classifier", {"lr": 0.01})
        assert isinstance(run_id, str)
        assert len(run_id) == 12

    def test_end_experiment_records_metrics(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        run_id = mm.start_experiment("regime_classifier", {"lr": 0.01})
        mm.end_experiment(run_id, {"accuracy": 0.9, "sharpe": 2.1})
        exp = mm._experiments[0]
        assert exp.metrics["accuracy"] == 0.9
        assert exp.metrics["sharpe"] == 2.1
        assert exp.duration_seconds >= 0.0

    def test_end_experiment_unknown_raises(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        with pytest.raises(ValueError, match="Unknown experiment"):
            mm.end_experiment("bogus_id", {"accuracy": 0.5})

    def test_get_best_experiment(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        r1 = mm.start_experiment("alpha_model", {"lr": 0.01})
        mm.end_experiment(r1, {"accuracy": 0.7})
        r2 = mm.start_experiment("alpha_model", {"lr": 0.001})
        mm.end_experiment(r2, {"accuracy": 0.85})
        best = mm.get_best_experiment("alpha_model", metric="accuracy")
        assert best is not None
        assert best.run_id == r2

    def test_get_best_experiment_no_match(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        result = mm.get_best_experiment("alpha_model")
        assert result is None

    def test_compare_experiments(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        r1 = mm.start_experiment("alpha_model", {"lr": 0.01})
        mm.end_experiment(r1, {"accuracy": 0.7})
        r2 = mm.start_experiment("alpha_model", {"lr": 0.001})
        mm.end_experiment(r2, {"accuracy": 0.85})
        cmp = mm.compare_experiments([r1, r2])
        assert r1 in cmp and r2 in cmp
        assert cmp[r1]["metrics"]["accuracy"] == 0.7
        assert cmp[r2]["metrics"]["accuracy"] == 0.85

    def test_compare_experiments_missing_id(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        cmp = mm.compare_experiments(["nonexistent"])
        assert cmp["nonexistent"] is None

    def test_experiment_run_dataclass(self):
        exp = ExperimentRun(
            run_id="abc123",
            model_name="test",
            hyperparams={"lr": 0.1},
            metrics={"accuracy": 0.9},
            duration_seconds=42.0,
            dataset_info={"rows": 1000},
        )
        assert exp.run_id == "abc123"
        assert exp.dataset_info["rows"] == 1000

    def test_queue_retrain_alias(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        job = mm.queue_retrain("regime_classifier")
        assert job.model_name == "regime_classifier"
        assert job.trigger == "concept_drift"
        assert job.status == "pending"


# ===========================================================================
# 3. OnlineLearner drift callback tests
# ===========================================================================

class TestOnlineLearnerDriftCallback:
    """Tests for OnlineLearner on_drift_callback."""

    def test_callback_invoked_on_drift(self):
        callback_calls = []

        def on_drift(info):
            callback_calls.append(info)

        learner = OnlineLearner(
            feature_dim=3,
            drift_threshold=0.1,
            name="test_model",
            on_drift_callback=on_drift,
        )
        # Feed extreme alternating errors to trigger drift quickly
        for i in range(200):
            label = 1.0 if i % 2 == 0 else -1.0
            features = [1.0, 0.0, 0.0]
            learner.partial_fit(features, label)
            learner.check_drift()

        assert len(callback_calls) > 0
        info = callback_calls[0]
        assert "model_name" in info
        assert info["model_name"] == "test_model"
        assert "drift_score" in info
        assert "samples_since_last_drift" in info

    def test_no_callback_when_none(self):
        learner = OnlineLearner(feature_dim=3, drift_threshold=0.1)
        # Should not raise even when drift occurs
        for i in range(200):
            label = 1.0 if i % 2 == 0 else -1.0
            learner.partial_fit([1.0, 0.0, 0.0], label)
            learner.check_drift()

    def test_callback_exception_does_not_crash(self):
        def bad_callback(info):
            raise RuntimeError("callback error")

        learner = OnlineLearner(
            feature_dim=3,
            drift_threshold=0.1,
            on_drift_callback=bad_callback,
        )
        for i in range(200):
            label = 1.0 if i % 2 == 0 else -1.0
            learner.partial_fit([1.0, 0.0, 0.0], label)
            learner.check_drift()
        # No exception propagated

    def test_drift_callback_receives_correct_score(self):
        scores = []

        def on_drift(info):
            scores.append(info["drift_score"])

        learner = OnlineLearner(
            feature_dim=2,
            drift_threshold=0.1,
            on_drift_callback=on_drift,
        )
        for i in range(300):
            label = 1.0 if i % 2 == 0 else -1.0
            learner.partial_fit([1.0, -1.0], label)
            learner.check_drift()

        if scores:
            assert all(s > 0.1 for s in scores)


# ===========================================================================
# 4. Dead stub replacement tests
# ===========================================================================

class TestDeadStubReplacements:
    """Tests that ml/ensemble.py and ml/feature_engineer.py re-export correctly."""

    def test_ensemble_imports_signal_hub(self):
        from ml.ensemble import EnsembleSignalHub
        from ml.ensemble_signal_hub import EnsembleSignalHub as DirectImport
        assert EnsembleSignalHub is DirectImport

    def test_ensemble_model_alias(self):
        from ml.ensemble import EnsembleModel
        from ml.ensemble_signal_hub import EnsembleSignalHub
        assert EnsembleModel is EnsembleSignalHub

    def test_feature_engineer_imports_pipeline(self):
        from ml.feature_engineer import compress_features
        from ml.feature_pipeline import compress_features as DirectImport
        assert compress_features is DirectImport

    def test_feature_engineer_transform(self):
        from ml.feature_engineer import transform_for_model
        from ml.feature_pipeline import transform_for_model as DirectImport
        assert transform_for_model is DirectImport

    def test_ensemble_no_runtime_error(self):
        """Old stub would raise RuntimeError on any access — verify it doesn't."""
        from ml import ensemble
        # Accessing EnsembleSignalHub should NOT raise
        cls = ensemble.EnsembleSignalHub
        assert cls is not None

    def test_feature_engineer_no_runtime_error(self):
        """Old stub would raise RuntimeError on any access — verify it doesn't."""
        from ml import feature_engineer
        fn = feature_engineer.compress_features
        assert callable(fn)


# ===========================================================================
# 5. Integration: drift -> retrain wiring
# ===========================================================================

class TestDriftRetrainIntegration:
    """Verify that OnlineLearner drift triggers ModelManager.queue_retrain."""

    def test_drift_queues_retrain_via_callback(self):
        mm = ModelManager(models_dir="/tmp/fake_models")
        initial_queue_len = len(mm._retrain_queue)

        def on_drift(info):
            model = info.get("model_name", "regime_classifier")
            mm.queue_retrain(model)

        learner = OnlineLearner(
            feature_dim=3,
            drift_threshold=0.1,
            name="regime_classifier",
            on_drift_callback=on_drift,
        )

        for i in range(300):
            label = 1.0 if i % 2 == 0 else -1.0
            learner.partial_fit([1.0, 0.0, 0.0], label)
            learner.check_drift()

        # At least one retrain should have been queued
        assert len(mm._retrain_queue) > initial_queue_len
        job = mm._retrain_queue[0]
        assert job.model_name == "regime_classifier"
        assert job.trigger == "concept_drift"


# ===========================================================================
# 6. InferenceResult dataclass tests
# ===========================================================================

class TestInferenceResult:

    def test_dataclass_fields(self):
        r = InferenceResult(
            prediction=1.5,
            confidence=0.9,
            latency_ms=2.3,
            model_version=3,
            cache_hit=True,
        )
        assert r.prediction == 1.5
        assert r.confidence == 0.9
        assert r.latency_ms == 2.3
        assert r.model_version == 3
        assert r.cache_hit is True


# ===========================================================================
# 7. InferenceService edge cases
# ===========================================================================

class TestInferenceServiceEdgeCases:

    def test_predict_numpy_array_output(self):
        """Model returning np.array should be handled."""

        class NumpyModel:
            def predict(self, features):
                return np.array([0.77])

        mm = _make_model_manager_with_fake(model_obj=NumpyModel())
        svc = InferenceService(mm)
        result = svc.predict("alpha_model", [1.0])
        assert abs(result.prediction - 0.77) < 1e-6

    def test_predict_scalar_output(self):
        mm = _make_model_manager_with_fake(model_obj=_FakeModel(return_value=3.14))
        svc = InferenceService(mm)
        result = svc.predict("alpha_model", [1.0])
        assert abs(result.prediction - 3.14) < 1e-6
        assert result.confidence == 1.0

    def test_clear_cache_nonexistent_key(self):
        mm = _make_model_manager_with_fake()
        svc = InferenceService(mm)
        svc.clear_cache("no_such_model")  # should not raise
