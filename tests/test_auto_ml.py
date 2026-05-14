"""Tests for the lightweight AutoML pipeline."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from ml.auto_ml import AutoMLPipeline, AutoMLResult, ModelType, NumpyBaselineModel, TrialResult


def _classification_data(n_samples: int = 80, n_features: int = 4, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    y = (X @ weights > 0).astype(int)
    return X, y


def _regression_data(n_samples: int = 80, n_features: int = 4, seed: int = 11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    y = X @ weights + rng.normal(scale=0.05, size=n_samples)
    return X, y


class TestTrialResult:
    def test_to_dict_includes_scores_and_error(self):
        trial = TrialResult(
            trial_id=1,
            model_type="numpy_baseline",
            params={"task_type": "classification"},
            train_score=0.75,
            val_score=0.7,
            cv_scores=[0.6, 0.8],
            training_time=0.123,
            n_features=3,
            error=None,
        )

        data = trial.to_dict()

        assert data["trial_id"] == 1
        assert data["cv_mean"] == 0.7
        assert data["cv_std"] == 0.1
        assert data["error"] is None


class TestAutoMLResult:
    def test_to_dict_sorts_top_trials(self):
        low = TrialResult(0, "a", {}, 0.1, 0.1, [], 0.0, 2)
        high = TrialResult(1, "b", {}, 0.9, 0.9, [], 0.0, 2)
        result = AutoMLResult(
            best_model=object(),
            best_params={},
            best_model_type="b",
            best_score=0.9,
            all_trials=[low, high],
            feature_importance=None,
            training_time=1.234,
            timestamp=datetime.now(),
        )

        data = result.to_dict()

        assert data["best_model_type"] == "b"
        assert data["top_trials"][0]["model_type"] == "b"
        assert data["n_trials"] == 2


class TestNumpyBaselineModel:
    def test_classification_predicts_majority_class(self):
        X, _ = _classification_data(n_samples=10)
        y = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1])
        model = NumpyBaselineModel(task_type="classification").fit(X, y)

        preds = model.predict(X[:3])

        assert preds.tolist() == [1, 1, 1]

    def test_regression_predicts_numeric_values(self):
        X, y = _regression_data(n_samples=30)
        model = NumpyBaselineModel(task_type="regression").fit(X, y)

        preds = model.predict(X[:5])

        assert preds.shape == (5,)
        assert np.issubdtype(preds.dtype, np.floating)


class TestAutoMLPipeline:
    def test_initialization_bounds_values(self):
        automl = AutoMLPipeline(
            time_limit_minutes=-1,
            n_trials=-5,
            cv_folds=1,
            early_stopping_rounds=0,
        )

        assert automl.n_trials == 0
        assert automl.cv_folds == 2
        assert automl.early_stopping_rounds == 1

    def test_fit_with_numpy_baseline_classification(self):
        X, y = _classification_data()
        automl = AutoMLPipeline(n_trials=3, metric="accuracy", random_seed=1)

        result = automl.fit(X[:60], y[:60], X[60:], y[60:], model_types=[ModelType.NUMPY_BASELINE.value])

        assert result.best_model_type == ModelType.NUMPY_BASELINE.value
        assert result.best_model is not None
        assert len(result.all_trials) == 3
        assert 0.0 <= result.best_score <= 1.0

    def test_fit_with_numpy_baseline_regression(self):
        X, y = _regression_data()
        automl = AutoMLPipeline(n_trials=2, metric="mse", task_type="regression", random_seed=2)

        result = automl.fit(X[:60], y[:60], X[60:], y[60:], model_types=["numpy_baseline"])

        assert result.best_model_type == "numpy_baseline"
        assert result.best_score <= 0.0

    def test_fit_falls_back_when_requested_model_unavailable(self):
        X, y = _classification_data()
        automl = AutoMLPipeline(n_trials=1, metric="accuracy")

        result = automl.fit(X, y, model_types=["not_a_model"])

        assert result.best_model_type == "numpy_baseline"
        assert len(result.all_trials) == 1

    def test_fit_adds_fallback_trial_when_no_trials_requested(self):
        X, y = _classification_data()
        automl = AutoMLPipeline(n_trials=0, metric="accuracy")

        result = automl.fit(X, y)

        assert result.best_model_type == "numpy_baseline"
        assert len(result.all_trials) == 1

    def test_fit_applies_cost_model_to_ranking_score(self):
        X, y = _classification_data()
        automl = AutoMLPipeline(n_trials=1, metric="accuracy")

        result = automl.fit(
            X,
            y,
            model_types=["numpy_baseline"],
            cost_model=lambda model_type, params, train_score, val_score: 0.25,
        )

        trial = result.all_trials[0]
        assert trial.cost_penalty == pytest.approx(0.25)
        assert trial.ranking_score == pytest.approx(trial.val_score - 0.25)
        assert result.best_score == pytest.approx(trial.ranking_score)

    def test_compute_accuracy_score(self):
        automl = AutoMLPipeline(metric="accuracy")

        score = automl._compute_score(np.array([1, 0, 1]), np.array([1, 1, 1]))

        assert score == pytest.approx(2 / 3)

    def test_compute_f1_score(self):
        automl = AutoMLPipeline(metric="f1")

        score = automl._compute_score(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))

        assert score == pytest.approx(0.5)

    def test_compute_sharpe_proxy_score(self):
        automl = AutoMLPipeline(metric="sharpe")

        score = automl._compute_score(np.array([1, -1, 1]), np.array([1, -1, -1]))

        assert score == pytest.approx((2 / 3) * 10 - 5)

    def test_invalid_train_shapes_raise(self):
        automl = AutoMLPipeline(n_trials=1)
        X, y = _classification_data()

        with pytest.raises(ValueError, match="same number of samples"):
            automl.fit(X, y[:-1])

    def test_validation_data_must_be_paired(self):
        automl = AutoMLPipeline(n_trials=1)
        X, y = _classification_data()

        with pytest.raises(ValueError, match="provided together"):
            automl.fit(X, y, X_val=X[:5])

    def test_sample_random_forest_params_handles_none_depth(self):
        automl = AutoMLPipeline(random_seed=42)

        for _ in range(20):
            params = automl._sample_params("random_forest")
            assert params["max_depth"] is None or isinstance(params["max_depth"], int)

    def test_extract_feature_importance_normalizes_coefficients(self):
        model = NumpyBaselineModel(task_type="regression")
        model.coef_ = np.array([1.0, -3.0])
        automl = AutoMLPipeline()

        importance = automl._extract_feature_importance(model, n_features=2)

        assert importance == {"feature_0": 0.25, "feature_1": 0.75}
