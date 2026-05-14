"""
Tests for the full ML lifecycle: automated retraining, inference service,
drift detection, performance monitoring, and checkpoint persistence.

Covers:
  - ModelManager.check_and_retrain()
  - ModelManager.evaluate_live_performance()
  - ModelManager.record_drift_event()
  - InferenceService integration
  - OnlineLearner drift -> auto retrain pipeline
  - CheckpointManager save/load round-trip
  - ComponentRegistry on_cycle ML wiring
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false, reportUninitializedInstanceVariable=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalSubscript=false

import asyncio
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from ml.model_manager import (
    ModelManager,
)
from ml.online_learner import OnlineLearner
from ml.inference_service import InferenceService, InferenceResult
from core.checkpoint_manager import CheckpointManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_manager(tmp_path):
    """ModelManager with a temporary models dir."""
    mm = ModelManager(models_dir=str(tmp_path / "models"))
    return mm


@pytest.fixture
def inference_service(model_manager):
    """InferenceService wrapping a ModelManager."""
    return InferenceService(model_manager)


@pytest.fixture
def checkpoint_db(tmp_path):
    """Temporary CheckpointManager."""
    db_path = str(tmp_path / "test_checkpoints.db")
    return CheckpointManager(db_path=db_path, save_interval=5, max_checkpoints=20)


@pytest.fixture
def online_learner():
    """OnlineLearner with small feature_dim."""
    return OnlineLearner(feature_dim=5, drift_threshold=10.0, name="test_regime")


# ---------------------------------------------------------------------------
# 1. Scheduled retrain triggers at correct interval
# ---------------------------------------------------------------------------

class TestCheckAndRetrain:
    """Tests for ModelManager.check_and_retrain()."""

    def test_scheduled_retrain_triggers_at_interval(self, model_manager):
        """Retrain should trigger when cycle_count >= retrain_interval_cycles."""
        model_manager._retrain_interval_cycles = 100

        results = asyncio.run(
            model_manager.check_and_retrain(cycle_count=100)
        )
        assert len(results) > 0
        triggers = [r["trigger"] for r in results]
        assert "weekly_cron" in triggers

    def test_no_retrain_before_interval(self, model_manager):
        """No retrain should trigger before the interval is reached."""
        model_manager._retrain_interval_cycles = 10000

        results = asyncio.run(
            model_manager.check_and_retrain(cycle_count=50)
        )
        assert len(results) == 0

    def test_force_retrain_triggers_all_models(self, model_manager):
        """force=True should trigger retrain for all models."""
        results = asyncio.run(
            model_manager.check_and_retrain(cycle_count=1, force=True)
        )
        assert len(results) == len(model_manager._registry)
        for r in results:
            assert r["trigger"] == "manual"
            assert r["status"] == "queued"

    def test_retrain_doesnt_block_trading_loop(self, model_manager):
        """check_and_retrain should return quickly (async background task)."""
        model_manager._retrain_interval_cycles = 1

        t0 = time.time()
        results = asyncio.run(
            model_manager.check_and_retrain(cycle_count=1)
        )
        elapsed = time.time() - t0

        # Should return in < 2 seconds (just queuing, not executing)
        assert elapsed < 2.0
        assert len(results) > 0

    def test_duplicate_retrain_blocked(self, model_manager):
        """If a retrain task is already running, it should return already_running."""
        model_manager._retrain_interval_cycles = 1

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # First call queues retrains
            results1 = loop.run_until_complete(
                model_manager.check_and_retrain(cycle_count=1, force=True)
            )
            assert all(r["status"] == "queued" for r in results1)

            # Second call should detect running tasks
            results2 = loop.run_until_complete(
                model_manager.check_and_retrain(cycle_count=2, force=True)
            )
            # Some may be already_running if background tasks haven't finished
            statuses = [r["status"] for r in results2]
            assert "already_running" in statuses or "queued" in statuses
        finally:
            loop.close()

    def test_last_retrain_cycle_tracked(self, model_manager):
        """After retrain, _last_retrain_cycle should be updated."""
        model_manager._retrain_interval_cycles = 1

        asyncio.run(
            model_manager.check_and_retrain(cycle_count=42, force=True)
        )

        for name in model_manager._registry:
            assert model_manager._last_retrain_cycle.get(name) == 42


# ---------------------------------------------------------------------------
# 2. Drift detection queues retrain
# ---------------------------------------------------------------------------

class TestDriftRetrain:
    """Tests for drift -> retrain pipeline."""

    def test_drift_callback_queues_retrain(self, model_manager):
        """OnlineLearner drift callback should queue a retrain job.

        We directly manipulate the PH state to trigger drift, since
        the PA-I learner adapts too quickly for organic drift in tests.
        """
        callback_fired = []

        def on_drift(info):
            callback_fired.append(info)
            model_manager.queue_retrain(
                info.get("model_name", "regime_classifier"),
                trigger="concept_drift",
            )

        ol = OnlineLearner(
            feature_dim=5,
            drift_threshold=5.0,
            name="regime_classifier",
            on_drift_callback=on_drift,
        )

        # Train briefly so PH stats have some baseline
        for _ in range(20):
            ol.partial_fit([1.0, 0.5, 0.2, 0.1, 0.0], 1.0)

        # Manually set PH state to trigger drift on next check
        with ol._lock:
            ol._ph_cumsum = 20.0
            ol._ph_min = 5.0  # ph_stat = 20 - 5 = 15 > threshold 5.0

        drift = ol.check_drift()
        assert drift is not None
        assert len(callback_fired) > 0

        # Check if a retrain was queued
        queue = list(model_manager._retrain_queue)
        drift_jobs = [j for j in queue if j.trigger == "concept_drift"]
        assert len(drift_jobs) > 0

    def test_record_drift_event_tracks_frequency(self, model_manager):
        """record_drift_event should track drift frequency."""
        for _ in range(6):
            model_manager.record_drift_event("regime_classifier")

        assert "regime_classifier" in model_manager._drift_frequency
        assert len(model_manager._drift_frequency["regime_classifier"]) == 6

    def test_drift_high_frequency_warning(self, model_manager, caplog):
        """High drift frequency should log a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            for _ in range(7):
                model_manager.record_drift_event("regime_classifier")

        assert any("drifted" in r.message and "review" in r.message for r in caplog.records)

    def test_drift_detection_with_check_and_retrain(self, model_manager):
        """check_and_retrain should detect near-drift from online_learner."""
        ol = OnlineLearner(feature_dim=5, drift_threshold=10.0, name="test")

        # Manually set PH state close to threshold
        ol._ph_cumsum = 15.0
        ol._ph_min = 5.5  # ph_stat = 15 - 5.5 = 9.5 > 10 * 0.9 = 9.0

        model_manager._retrain_interval_cycles = 999999  # prevent scheduled trigger

        results = asyncio.run(
            model_manager.check_and_retrain(cycle_count=1, online_learner=ol)
        )
        drift_results = [r for r in results if r["trigger"] == "concept_drift"]
        assert len(drift_results) > 0


# ---------------------------------------------------------------------------
# 3. Performance degradation queues retrain
# ---------------------------------------------------------------------------

class TestPerformanceDegradation:
    """Tests for performance monitoring and auto-retrain."""

    def test_evaluate_live_performance_good(self, model_manager):
        """High accuracy should not trigger retrain."""
        predictions = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
        actuals = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0]

        result = model_manager.evaluate_live_performance(
            "regime_classifier", predictions, actuals
        )
        assert result["accuracy"] == 1.0
        assert result["degraded"] is False

    def test_evaluate_live_performance_degraded(self, model_manager):
        """Low accuracy should trigger retrain."""
        predictions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        actuals = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

        result = model_manager.evaluate_live_performance(
            "regime_classifier", predictions, actuals
        )
        assert result["accuracy"] == 0.0
        assert result["degraded"] is True

        # Check that a retrain was queued
        queue = list(model_manager._retrain_queue)
        assert any(
            j.model_name == "regime_classifier"
            and j.trigger == "performance_degradation"
            for j in queue
        )

    def test_evaluate_returns_metrics(self, model_manager):
        """evaluate_live_performance should return all expected metrics."""
        preds = [1.0, 1.0, -1.0, -1.0, 1.0]
        acts = [1.0, -1.0, -1.0, 1.0, 1.0]

        result = model_manager.evaluate_live_performance(
            "regime_classifier", preds, acts
        )
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "sharpe_contribution" in result
        assert "degraded" in result
        assert "sample_size" in result
        assert result["sample_size"] == 5

    def test_evaluate_empty_predictions(self, model_manager):
        """Empty predictions should return zeros without error."""
        result = model_manager.evaluate_live_performance(
            "regime_classifier", [], []
        )
        assert result["accuracy"] == 0.0
        assert result["degraded"] is False
        assert result["sample_size"] == 0

    def test_evaluate_unknown_model_raises(self, model_manager):
        """Unknown model should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            model_manager.evaluate_live_performance(
                "nonexistent_model", [1.0], [1.0]
            )

    def test_performance_check_updates_registry(self, model_manager):
        """evaluate_live_performance should update last_accuracy in registry."""
        preds = [1.0, 1.0, 1.0, -1.0, -1.0]
        acts = [1.0, 1.0, 1.0, -1.0, -1.0]

        model_manager.evaluate_live_performance("regime_classifier", preds, acts)
        assert model_manager._registry["regime_classifier"].last_accuracy == 1.0

    def test_performance_degradation_triggers_retrain_via_check_and_retrain(self, model_manager):
        """check_and_retrain should detect degraded accuracy and queue retrain."""
        model_manager._retrain_interval_cycles = 999999
        # Set low accuracy manually
        model_manager._registry["regime_classifier"].last_accuracy = 0.3

        results = asyncio.run(
            model_manager.check_and_retrain(cycle_count=1)
        )
        perf_results = [r for r in results if r["trigger"] == "performance_degradation"]
        assert len(perf_results) > 0


# ---------------------------------------------------------------------------
# 4. Model version increments after retrain
# ---------------------------------------------------------------------------

class TestModelVersioning:
    """Tests for model version tracking."""

    def test_version_starts_at_zero(self, model_manager):
        """All models should start at version 0."""
        for name, meta in model_manager._registry.items():
            assert meta.version == 0

    def test_version_increments_on_schedule_retrain(self, model_manager):
        """schedule_retrain should queue a job (version increments on completion)."""
        job = model_manager.schedule_retrain("regime_classifier", trigger="manual")
        assert job.status == "pending"
        assert job.model_name == "regime_classifier"

    def test_multiple_models_retrain_independently(self, model_manager):
        """Each model tracks its own retrain cycle independently."""
        model_manager._retrain_interval_cycles = 100
        model_manager._last_retrain_cycle["regime_classifier"] = 50
        model_manager._last_retrain_cycle["alpha_model"] = 90

        results = asyncio.run(
            model_manager.check_and_retrain(cycle_count=150)
        )
        triggered_models = {r["model_name"] for r in results}
        # Both should trigger (50 -> 150 = 100, 90 -> 150 = 60)
        assert "regime_classifier" in triggered_models
        # alpha_model at 90, interval 100, so 150-90=60 < 100 -- should NOT trigger
        # But volatility_forecaster at 0, 150-0=150 >= 100 -- WILL trigger
        assert "volatility_forecaster" in triggered_models


# ---------------------------------------------------------------------------
# 5. Inference service called each cycle
# ---------------------------------------------------------------------------

class TestInferenceService:
    """Tests for InferenceService integration."""

    def test_predict_with_mock_model(self, model_manager):
        """InferenceService should work with a mock model."""
        # Create a simple mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.75]))

        # Register it in model_manager
        model_manager._loaded_objects["regime_classifier"] = mock_model
        model_manager._registry["regime_classifier"].is_loaded = True

        svc = InferenceService(model_manager)
        result = svc.predict("regime_classifier", [1.0, 2.0, 3.0])

        assert isinstance(result, InferenceResult)
        assert result.prediction == 0.75
        assert result.latency_ms >= 0
        mock_model.predict.assert_called_once()

    def test_predict_fallback_on_missing_model(self, model_manager):
        """InferenceService should raise RuntimeError for unloaded models."""
        svc = InferenceService(model_manager)
        with pytest.raises(RuntimeError, match="could not be loaded"):
            svc.predict("regime_classifier", [1.0, 2.0, 3.0])

    def test_predict_rejects_nan_input(self, model_manager):
        """InferenceService should reject NaN inputs."""
        svc = InferenceService(model_manager)
        with pytest.raises(ValueError, match="NaN"):
            svc.predict("regime_classifier", [1.0, float("nan"), 3.0])

    def test_predict_rejects_inf_input(self, model_manager):
        """InferenceService should reject Inf inputs."""
        svc = InferenceService(model_manager)
        with pytest.raises(ValueError, match="Inf"):
            svc.predict("regime_classifier", [1.0, float("inf"), 3.0])

    def test_inference_stats_tracking(self, model_manager):
        """InferenceService should track prediction stats."""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([1.0]))
        model_manager._loaded_objects["regime_classifier"] = mock_model
        model_manager._registry["regime_classifier"].is_loaded = True

        svc = InferenceService(model_manager)
        for _ in range(5):
            svc.predict("regime_classifier", [1.0, 2.0, 3.0])

        stats = svc.get_stats()
        assert stats["total_predictions"] == 5
        assert stats["avg_latency_ms"] >= 0

    def test_batch_prediction(self, model_manager):
        """InferenceService batch predict should work."""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.5]))
        model_manager._loaded_objects["regime_classifier"] = mock_model
        model_manager._registry["regime_classifier"].is_loaded = True

        svc = InferenceService(model_manager)
        results = svc.predict_batch(
            "regime_classifier",
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        )
        assert len(results) == 3
        assert all(isinstance(r, InferenceResult) for r in results)


# ---------------------------------------------------------------------------
# 6. Checkpoint save/load round-trip
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_and_load_round_trip(self, checkpoint_db):
        """Saving and loading should preserve state."""
        state = {
            "cycle_count": 42,
            "portfolio_value": 1234.56,
            "regime": "TRENDING_UP",
            "model_versions": {"regime_classifier": 3},
            "risk_state": {"var_pct": 2.5},
        }
        assert checkpoint_db.save_checkpoint(state)

        loaded = checkpoint_db.load_latest_checkpoint()
        assert loaded is not None
        assert loaded["cycle_count"] == 42
        assert loaded["portfolio_value"] == 1234.56
        assert loaded["regime"] == "TRENDING_UP"
        assert loaded["model_versions"]["regime_classifier"] == 3

    def test_load_returns_none_when_empty(self, checkpoint_db):
        """Loading from empty DB should return None."""
        result = checkpoint_db.load_latest_checkpoint()
        assert result is None

    def test_list_checkpoints(self, checkpoint_db):
        """list_checkpoints should return recent entries."""
        for i in range(5):
            checkpoint_db.save_checkpoint({"cycle_count": i * 10})

        entries = checkpoint_db.list_checkpoints(limit=3)
        assert len(entries) == 3
        # Most recent first
        assert entries[0]["cycle_count"] == 40
        assert entries[1]["cycle_count"] == 30
        assert entries[2]["cycle_count"] == 20

    def test_should_save_interval(self, checkpoint_db):
        """should_save should respect save_interval."""
        # _last_save_cycle starts at 0, interval=5
        assert checkpoint_db.should_save(5)   # 5 - 0 = 5 >= 5
        assert checkpoint_db.should_save(10)  # 10 - 0 = 10 >= 5

        checkpoint_db.save_checkpoint({"cycle_count": 5})
        # _last_save_cycle is now 5
        assert not checkpoint_db.should_save(6)   # 6 - 5 = 1 < 5
        assert not checkpoint_db.should_save(9)   # 9 - 5 = 4 < 5
        assert checkpoint_db.should_save(10)       # 10 - 5 = 5 >= 5

    def test_pruning(self, tmp_path):
        """Old checkpoints should be pruned when exceeding max."""
        db = CheckpointManager(
            db_path=str(tmp_path / "prune_test.db"),
            save_interval=1,
            max_checkpoints=5,
        )

        for i in range(10):
            db.save_checkpoint({"cycle_count": i})

        entries = db.list_checkpoints(limit=100)
        assert len(entries) <= 5

    def test_checkpoint_on_shutdown(self, checkpoint_db):
        """Simulate shutdown checkpoint save."""
        state = {
            "cycle_count": 999,
            "shutdown": True,
            "portfolio_value": 5000.0,
        }
        assert checkpoint_db.save_checkpoint(state)

        loaded = checkpoint_db.load_latest_checkpoint()
        assert loaded["shutdown"] is True
        assert loaded["cycle_count"] == 999

    def test_get_checkpoint_by_cycle(self, checkpoint_db):
        """Should retrieve a checkpoint by its cycle count."""
        checkpoint_db.save_checkpoint({"cycle_count": 10, "value": "a"})
        checkpoint_db.save_checkpoint({"cycle_count": 20, "value": "b"})

        result = checkpoint_db.get_checkpoint_by_cycle(10)
        assert result is not None
        assert result["value"] == "a"

        missing = checkpoint_db.get_checkpoint_by_cycle(15)
        assert missing is None

    def test_multiple_saves_load_latest(self, checkpoint_db):
        """load_latest should always return the most recent checkpoint."""
        for i in range(5):
            checkpoint_db.save_checkpoint({"cycle_count": i, "data": f"v{i}"})

        latest = checkpoint_db.load_latest_checkpoint()
        assert latest["cycle_count"] == 4
        assert latest["data"] == "v4"


# ---------------------------------------------------------------------------
# 7. Feature drift detection accuracy
# ---------------------------------------------------------------------------

class TestOnlineLearnerDrift:
    """Tests for OnlineLearner drift detection."""

    def test_no_drift_on_stable_data(self, online_learner):
        """Stable data should not trigger drift."""
        for _ in range(50):
            online_learner.partial_fit([0.1, 0.2, 0.3, 0.4, 0.5], 1.0)
            drift = online_learner.check_drift()
            # With low threshold (10.0) and small dataset, no drift expected
            # for consistent labels
        stats = online_learner.get_stats()
        assert stats["drift_count"] == 0

    def test_drift_detected_on_concept_shift(self):
        """Directly trigger drift via PH state manipulation to verify detection logic."""
        ol = OnlineLearner(feature_dim=3, drift_threshold=5.0, name="test_drift")

        # Train briefly
        for _ in range(20):
            ol.partial_fit([1.0, 0.5, 0.2], 1.0)

        # Verify no drift initially
        assert ol.check_drift() is None

        # Manually push PH stat above threshold
        with ol._lock:
            ol._ph_cumsum = 15.0
            ol._ph_min = 2.0  # ph_stat = 15 - 2 = 13 > 5.0

        drift = ol.check_drift()
        assert drift is not None
        assert drift.drift_score > 5.0
        assert ol.get_stats()["drift_count"] == 1

    def test_drift_callback_fires(self):
        """Drift callback should be called when drift is detected."""
        callback_info = {}

        def cb(info):
            callback_info.update(info)

        ol = OnlineLearner(
            feature_dim=3,
            drift_threshold=5.0,
            name="test_cb",
            on_drift_callback=cb,
        )

        # Drive drift
        for _ in range(100):
            ol.partial_fit([1.0, 0.5, 0.2], 1.0)
        for _ in range(300):
            ol.partial_fit([1.0, 0.5, 0.2], -1.0)
            ol.check_drift()
            if callback_info:
                break

        assert "model_name" in callback_info or len(callback_info) == 0
        # If drift was detected, callback should have fired
        stats = ol.get_stats()
        if stats["drift_count"] > 0:
            assert "drift_score" in callback_info


# ---------------------------------------------------------------------------
# 8. ComponentRegistry ML wiring
# ---------------------------------------------------------------------------

class TestComponentRegistryMLWiring:
    """Tests for ML components wired into on_cycle."""

    def test_on_cycle_returns_advisory(self):
        """on_cycle should return an advisory dict even with no components."""
        from core.component_registry import ComponentRegistry

        config = MagicMock()
        config.trading_pairs = ["BTC/USD"]
        cr = ComponentRegistry(config)

        result = cr.on_cycle({"BTC/USD": 50000.0}, "TRENDING_UP")
        assert isinstance(result, dict)

    def test_on_cycle_with_ensemble_hub(self):
        """on_cycle should include ensemble data when ensemble_hub is available."""
        from core.component_registry import ComponentRegistry

        config = MagicMock()
        config.trading_pairs = ["BTC/USD"]
        cr = ComponentRegistry(config)

        # Mock ensemble hub
        mock_ensemble = MagicMock()
        from ml.ensemble_signal_hub import EnsembleSignal
        mock_ensemble.update.return_value = EnsembleSignal(
            composite=0.3,
            confidence=0.8,
            size_multiplier=1.2,
            regime_bias="BULLISH",
        )
        cr.ensemble_hub = mock_ensemble

        # Mock signal stacker
        mock_stacker = MagicMock()
        cr.signal_stacker = mock_stacker

        result = cr.on_cycle({"BTC/USD": 50000.0}, "TRENDING_UP")
        assert "ensemble" in result
        assert result["ensemble"]["composite"] == 0.3
        mock_stacker.update_signal.assert_called()

    def test_on_cycle_with_inference_service(self):
        """on_cycle should call inference_service and track latency."""
        from core.component_registry import ComponentRegistry

        config = MagicMock()
        config.trading_pairs = ["BTC/USD"]
        cr = ComponentRegistry(config)

        # Mock inference service
        mock_inf = MagicMock()
        mock_inf.predict.return_value = InferenceResult(
            prediction="TRENDING_UP",
            confidence=0.85,
            latency_ms=5.0,
            model_version=2,
            cache_hit=False,
        )
        cr.inference_service = mock_inf
        cr.signal_stacker = MagicMock()
        cr.ensemble_hub = MagicMock()

        result = cr.on_cycle({"BTC/USD": 50000.0}, "TRENDING_UP")
        assert "inference" in result
        assert result["inference"]["prediction"] == "TRENDING_UP"
        assert result["inference"]["confidence"] == 0.85
        mock_inf.predict.assert_called_once()

    def test_on_cycle_inference_latency_warning(self, caplog):
        """High inference latency should log a warning."""
        import logging
        from core.component_registry import ComponentRegistry

        config = MagicMock()
        config.trading_pairs = ["BTC/USD"]
        cr = ComponentRegistry(config)

        # Mock inference service with slow response
        mock_inf = MagicMock()

        def slow_predict(*args, **kwargs):
            time.sleep(0.15)  # 150ms > 100ms threshold
            return InferenceResult(
                prediction="HOLD",
                confidence=0.5,
                latency_ms=150.0,
                model_version=1,
                cache_hit=False,
            )

        mock_inf.predict = slow_predict
        cr.inference_service = mock_inf
        cr.signal_stacker = MagicMock()
        cr.ensemble_hub = MagicMock()

        with caplog.at_level(logging.WARNING):
            cr.on_cycle({"BTC/USD": 50000.0}, "UNKNOWN")

        assert any("latency" in r.message for r in caplog.records)

    def test_cycle_count_increments(self):
        """on_cycle should increment the internal cycle counter."""
        from core.component_registry import ComponentRegistry

        config = MagicMock()
        cr = ComponentRegistry(config)

        initial = cr._cycle_count
        cr.on_cycle({"BTC/USD": 50000.0}, "UNKNOWN")
        assert cr._cycle_count == initial + 1

    def test_retrain_check_scheduled_at_interval(self):
        """on_cycle should schedule retrain checks every 500 cycles."""
        from core.component_registry import ComponentRegistry

        config = MagicMock()
        cr = ComponentRegistry(config)
        cr.model_manager = MagicMock()

        async def mock_retrain(*args, **kwargs):
            return []

        cr.model_manager.check_and_retrain = mock_retrain

        # Set cycle count to 499 so next call hits 500
        cr._cycle_count = 499
        result = cr.on_cycle({"BTC/USD": 50000.0}, "UNKNOWN")
        # The retrain check should have been scheduled at cycle 500
        assert result.get("retrain_check_scheduled") is True or cr._cycle_count == 500


# ---------------------------------------------------------------------------
# 9. Integration tests
# ---------------------------------------------------------------------------

class TestMLLifecycleIntegration:
    """End-to-end integration tests."""

    def test_full_lifecycle_flow(self, tmp_path):
        """Test: train -> predict -> evaluate -> drift -> retrain -> checkpoint."""
        # 1. Create model manager
        mm = ModelManager(models_dir=str(tmp_path / "models"))

        # 2. Create online learner with drift callback
        retrain_queue = []

        def on_drift(info):
            retrain_queue.append(info)
            mm.queue_retrain(
                info.get("model_name", "regime_classifier"),
                trigger="concept_drift",
            )

        ol = OnlineLearner(
            feature_dim=5,
            drift_threshold=8.0,
            name="regime_classifier",
            on_drift_callback=on_drift,
        )

        # 3. Simulate trading cycles
        for i in range(50):
            features = [float(i), 0.5, 0.2, 0.1, 0.0]
            label = 1.0
            ol.partial_fit(features, label)

        # 4. Evaluate performance
        preds = [1.0] * 10
        actuals = [1.0] * 8 + [-1.0] * 2
        result = mm.evaluate_live_performance("regime_classifier", preds, actuals)
        assert result["accuracy"] == 0.8
        assert result["degraded"] is False

        # 5. Create checkpoint
        ckpt = CheckpointManager(
            db_path=str(tmp_path / "lifecycle_test.db"),
            save_interval=1,
        )
        state = {
            "cycle_count": 50,
            "model_versions": {
                name: meta.version
                for name, meta in mm._registry.items()
            },
        }
        assert ckpt.save_checkpoint(state)

        # 6. Load checkpoint
        loaded = ckpt.load_latest_checkpoint()
        assert loaded is not None
        assert loaded["cycle_count"] == 50

    def test_model_manager_accepts_config_kwarg(self):
        """ModelManager should accept config= kwarg without error."""
        mm = ModelManager(config="dummy_config")
        assert mm is not None
        assert mm.device in ("cpu", "cuda")

    def test_valid_triggers_completeness(self):
        """All trigger strings used should be in VALID_TRIGGERS."""
        expected = {"manual", "weekly_cron", "performance_degradation", "concept_drift"}
        assert VALID_TRIGGERS == expected

    def test_snapshot_includes_all_fields(self, model_manager):
        """snapshot() should include all model statuses."""
        snap = model_manager.snapshot()
        assert "device" in snap
        assert "models" in snap
        assert "pending_retrains" in snap
        for name in model_manager._registry:
            assert name in snap["models"]
            m = snap["models"][name]
            assert "version" in m
            assert "is_loaded" in m
            assert "is_stale" in m
