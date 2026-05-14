"""ML Pipeline Orchestrator tests — validates new ML components.

Tests:
  - MLPipelineOrchestrator (train, predict, health, export)
  - AutoRetrainPipeline (schedule, drift, execution)
  - ModelExplainer (global importance, local explanation, summary)
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))


def _random_data(n_samples=500, n_features=10, seed=42):
    """Generate random features and labels."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    weights = rng.normal(0, 1, n_features)
    y = X @ weights + rng.normal(0, 0.1, n_samples)
    y_binary = (y > np.median(y)).astype(int)
    return X, y_binary


# ---------------------------------------------------------------------------
# MLPipelineOrchestrator (6 tests)
# ---------------------------------------------------------------------------

class TestMLPipelineOrchestrator:
    def test_initialization(self):
        from ml.pipeline_orchestrator import MLPipelineOrchestrator
        pipeline = MLPipelineOrchestrator(config={
            "model_name": "test_model",
            "model_type": "random_forest",
        })
        assert pipeline.config.model_name == "test_model"
        assert pipeline._status.value == "initializing"
    
    def test_train_and_predict(self):
        from ml.pipeline_orchestrator import MLPipelineOrchestrator
        pipeline = MLPipelineOrchestrator(config={
            "model_name": "test_model",
            "model_type": "random_forest",
            "min_training_samples": 10,
        })
        
        X, y = _random_data(n_samples=100, n_features=5)
        result = pipeline.train(X, y)
        
        assert result["status"] == "success"
        assert "model_id" in result
        
        pred = pipeline.predict(X[:5])
        assert len(pred.predictions) == 5
    
    def test_health_check(self):
        from ml.pipeline_orchestrator import MLPipelineOrchestrator
        pipeline = MLPipelineOrchestrator(config={
            "model_name": "test_model",
            "model_type": "random_forest",
            "min_training_samples": 10,
            "drift_threshold": 0.5,  # High threshold to avoid false drift
        })
        
        X, y = _random_data(n_samples=100)
        pipeline.train(X, y)
        
        health = pipeline.check_health()
        # Status can be ready or degraded depending on drift detection
        assert health.model_status.value in ("ready", "degraded")
    
    def test_export_state(self):
        from ml.pipeline_orchestrator import MLPipelineOrchestrator
        pipeline = MLPipelineOrchestrator(config={
            "model_name": "test_model",
            "model_type": "random_forest",
            "min_training_samples": 10,
        })
        
        X, y = _random_data(n_samples=100)
        pipeline.train(X, y)
        
        state = pipeline.export_state()
        assert "config" in state
        assert "metadata" in state
    
    def test_training_history(self):
        from ml.pipeline_orchestrator import MLPipelineOrchestrator
        pipeline = MLPipelineOrchestrator(config={
            "model_name": "test_model",
            "model_type": "random_forest",
            "min_training_samples": 10,
        })
        
        X, y = _random_data(n_samples=100)
        pipeline.train(X, y)
        pipeline.train(X, y)
        
        history = pipeline.get_training_history()
        assert len(history) == 2
    
    def test_insufficient_data_raises(self):
        from ml.pipeline_orchestrator import MLPipelineOrchestrator
        pipeline = MLPipelineOrchestrator(config={
            "model_name": "test_model",
            "model_type": "random_forest",
            "min_training_samples": 100,
        })
        
        X, y = _random_data(n_samples=10)
        result = pipeline.train(X, y)
        assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# AutoRetrainPipeline (5 tests)
# ---------------------------------------------------------------------------

class TestAutoRetrainPipeline:
    def test_initialization(self):
        from ml.auto_retrain import AutoRetrainPipeline
        retrain = AutoRetrainPipeline(model_name="test_model")
        assert retrain.model_name == "test_model"
    
    def test_should_retrain_schedule(self):
        from ml.auto_retrain import AutoRetrainPipeline, RetrainConfig
        config = RetrainConfig(schedule="none")
        retrain = AutoRetrainPipeline(model_name="test_model", config=config)
        
        should, reason = retrain.should_retrain()
        assert should is False
    
    def test_should_retrain_drift(self):
        from ml.auto_retrain import AutoRetrainPipeline, RetrainConfig
        config = RetrainConfig(consecutive_drift_count=2)
        retrain = AutoRetrainPipeline(model_name="test_model", config=config)
        
        retrain.report_drift(0.2, True)
        retrain.report_drift(0.2, True)
        
        should, reason = retrain.should_retrain()
        assert should is True
    
    def test_execute_retrain(self):
        from ml.auto_retrain import AutoRetrainPipeline, RetrainReason
        retrain = AutoRetrainPipeline(model_name="test_model")
        
        def mock_train(features, labels):
            return "new_model_v1", {"accuracy": 0.85}
        
        def mock_evaluate(features, labels, model_id=None):
            return {"accuracy": 0.85}
        
        retrain.set_callbacks(train_fn=mock_train, evaluate_fn=mock_evaluate)
        
        X, y = _random_data(n_samples=50)
        job = retrain.execute(reason=RetrainReason.MANUAL, features=X, labels=y)
        
        assert job.new_model_id == "new_model_v1"
        assert job.status.value == "completed"
    
    def test_get_stats(self):
        from ml.auto_retrain import AutoRetrainPipeline, RetrainReason
        retrain = AutoRetrainPipeline(model_name="test_model")
        
        def mock_train(features, labels):
            return "model_v1", {"accuracy": 0.8}
        
        retrain.set_callbacks(train_fn=mock_train)
        
        X, y = _random_data(n_samples=50)
        retrain.execute(reason=RetrainReason.MANUAL, features=X, labels=y)
        
        stats = retrain.get_stats()
        assert stats["total_retrains"] == 1


# ---------------------------------------------------------------------------
# ModelExplainer (5 tests)
# ---------------------------------------------------------------------------

class TestModelExplainer:
    def _train_simple_model(self):
        from sklearn.ensemble import RandomForestClassifier
        X, y = _random_data(n_samples=100, n_features=5)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X, y
    
    def test_initialization(self):
        from ml.model_explainer import ModelExplainer
        model, X, y = self._train_simple_model()
        
        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )
        
        assert explainer.n_features == 5
        assert explainer._explainer_type == "tree"
    
    def test_global_importance(self):
        from ml.model_explainer import ModelExplainer
        model, X, y = self._train_simple_model()
        
        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )
        
        importance = explainer.global_importance(X)
        assert len(importance.feature_names) == 5
        assert len(importance.importance_values) == 5
    
    def test_local_explanation(self):
        from ml.model_explainer import ModelExplainer
        model, X, y = self._train_simple_model()
        
        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )
        
        explanation = explainer.explain(X[0])
        assert len(explanation.shap_values) == 5
        assert explanation.method in ("shap", "gradient_fallback")
    
    def test_summary(self):
        from ml.model_explainer import ModelExplainer
        model, X, y = self._train_simple_model()
        
        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )
        
        summary = explainer.summary(X[:50])
        assert summary.n_samples == 50
        assert len(summary.feature_correlation) == 5
    
    def test_to_dict_output(self):
        from ml.model_explainer import ModelExplainer
        model, X, y = self._train_simple_model()
        
        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )
        
        importance = explainer.global_importance(X)
        d = importance.to_dict()
        assert "feature_names" in d
        assert "top_features" in d


# ---------------------------------------------------------------------------
# FeaturePipeline (3 tests)
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    def test_initialization(self):
        from ml.pipeline_orchestrator import FeaturePipeline
        pipeline = FeaturePipeline(feature_sources=["prices", "volume"])
        assert len(pipeline.feature_sources) == 2
    
    def test_transform(self):
        from ml.pipeline_orchestrator import FeaturePipeline
        pipeline = FeaturePipeline(feature_sources=["prices", "volume"])
        
        raw_data = {
            "prices": np.array([100.0, 101.0, 102.0]),
            "volume": np.array([1000.0, 1100.0, 1200.0]),
        }
        
        features = pipeline.transform(raw_data)
        assert features.shape == (1, 6)
    
    def test_feature_hash(self):
        from ml.pipeline_orchestrator import FeaturePipeline
        pipeline = FeaturePipeline(feature_sources=["prices"])
        
        X = np.random.randn(100, 5)
        hash1 = pipeline.compute_feature_hash(X)
        hash2 = pipeline.compute_feature_hash(X)
        assert hash1 == hash2
        assert len(hash1) == 16


# ---------------------------------------------------------------------------
# ModelRegistry (3 tests)
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_get(self):
        from ml.pipeline_orchestrator import ModelRegistry, ModelMetadata, ModelStatus
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(model_dir=tmpdir)
            
            metadata = ModelMetadata(
                model_id="test_v1",
                model_name="test_model",
                model_type="xgboost",
                version=1,
                created_at=datetime.now(timezone.utc),
                trained_samples=1000,
                feature_hash="abc123",
                status=ModelStatus.READY,
            )
            
            registry.register(metadata)
            retrieved = registry.get("test_v1")
            assert retrieved is not None
    
    def test_get_latest(self):
        from ml.pipeline_orchestrator import ModelRegistry, ModelMetadata, ModelStatus
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(model_dir=tmpdir)
            
            for v in [1, 2, 3]:
                metadata = ModelMetadata(
                    model_id=f"test_v{v}",
                    model_name="test_model",
                    model_type="xgboost",
                    version=v,
                    created_at=datetime.now(timezone.utc),
                    trained_samples=1000,
                    feature_hash="abc123",
                    status=ModelStatus.READY,
                )
                registry.register(metadata)
            
            latest = registry.get_latest("test_model")
            assert latest.version == 3
    
    def test_archive_old_versions(self):
        from ml.pipeline_orchestrator import ModelRegistry, ModelMetadata, ModelStatus
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(model_dir=tmpdir)
            
            for v in range(1, 6):
                metadata = ModelMetadata(
                    model_id=f"test_v{v}",
                    model_name="test_model",
                    model_type="xgboost",
                    version=v,
                    created_at=datetime.now(timezone.utc),
                    trained_samples=1000,
                    feature_hash="abc123",
                    status=ModelStatus.READY,
                )
                registry.register(metadata)
            
            archived = registry.archive_old_versions("test_model", keep=2)
            assert len(archived) == 3
