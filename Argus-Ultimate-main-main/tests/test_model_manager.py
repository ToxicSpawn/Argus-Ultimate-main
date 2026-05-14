"""
Tests for ml.model_manager.ModelManager.

Covers:
  - ModelMetadata and RetrainJob dataclasses
  - GPU detection / CPU fallback
  - load() returns False for missing files
  - load() returns False for unknown model names
  - load_all() returns a dict with all five model keys
  - is_stale() based on trained_at timestamp
  - schedule_retrain() queues correctly and logs the canonical message
  - schedule_retrain() raises on invalid trigger
  - run_pending_retrains() changes job status (mock subprocess)
  - hot_swap() updates registry path/version on success
  - hot_swap() reverts path on failure (file not found)
  - performance_check() queues retrain when accuracy < threshold
  - performance_check() does NOT queue when accuracy is fine
  - snapshot() contains all required keys and all five models
  - get_object() returns None before loading
"""

from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_NAMES = [
    "regime_classifier",
    "volatility_forecaster",
    "alpha_model",
    "rl_agent",
    "tft_forecaster",
]


def _make_manager(tmp_path: Path):
    """Return a ModelManager whose models_dir points to a temp directory."""
    from ml.model_manager import ModelManager
    return ModelManager(models_dir=str(tmp_path / "models"))


# ---------------------------------------------------------------------------
# Dataclass sanity tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_model_metadata_defaults(self):
        from ml.model_manager import ModelMetadata
        m = ModelMetadata(name="foo", path="models/foo.pkl")
        assert m.version == 0
        assert m.trained_at is None
        assert m.is_loaded is False
        assert m.device == "cpu"
        assert m.last_accuracy == 0.0

    def test_retrain_job_defaults(self):
        from ml.model_manager import RetrainJob
        j = RetrainJob(model_name="rl_agent", trigger="manual")
        assert j.status == "pending"
        assert j.queued_at is not None
        # queued_at should be timezone-aware
        assert j.queued_at.tzinfo is not None


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

class TestGPUDetection:
    def test_device_is_cpu_when_torch_missing(self, tmp_path):
        """When torch is not importable the device must fall back to 'cpu'."""
        with patch.dict(sys.modules, {"torch": None}):
            from ml import model_manager as mm
            # Patch the module-level flag
            with patch.object(mm, "_TORCH_AVAILABLE", False):
                device = mm._detect_device()
        assert device == "cpu"

    def test_device_is_cpu_when_cuda_unavailable(self, tmp_path):
        """When torch is present but CUDA is unavailable, device is 'cpu'."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            from ml import model_manager as mm
            with patch.object(mm, "_TORCH_AVAILABLE", True), \
                 patch.object(mm, "_torch", mock_torch):
                device = mm._detect_device()
        assert device == "cpu"

    def test_device_is_cuda_when_available(self, tmp_path):
        """When torch.cuda.is_available() returns True, device is 'cuda'."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        from ml import model_manager as mm
        with patch.object(mm, "_TORCH_AVAILABLE", True), \
             patch.object(mm, "_torch", mock_torch):
            device = mm._detect_device()
        assert device == "cuda"


# ---------------------------------------------------------------------------
# load() — missing file / unknown name
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_unknown_model_returns_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.load("does_not_exist")
        assert result is False

    def test_load_missing_file_returns_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        # regime_classifier file does not exist
        result = mgr.load("regime_classifier")
        assert result is False
        assert mgr.registry["regime_classifier"].is_loaded is False

    def test_load_missing_rl_agent_returns_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.load("rl_agent")
        assert result is False

    def test_load_missing_tft_returns_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.load("tft_forecaster")
        assert result is False

    def test_load_volatility_forecaster_always_succeeds(self, tmp_path):
        """volatility_forecaster is in-memory only — should always return True."""
        mgr = _make_manager(tmp_path)
        result = mgr.load("volatility_forecaster")
        assert result is True
        assert mgr.registry["volatility_forecaster"].is_loaded is True

    def test_load_joblib_file(self, tmp_path):
        """Creates a pickle file and verifies it loads via the joblib/pickle path."""
        obj = {"sentinel": 42}
        pkl_path = tmp_path / "models" / "regime_classifier.pkl"
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(obj, fh)

        from ml.model_manager import ModelManager, _JOBLIB_AVAILABLE
        mgr = ModelManager(models_dir=str(tmp_path / "models"))
        # Override path to point at the file we just created
        mgr.registry["regime_classifier"].path = str(pkl_path)
        result = mgr.load("regime_classifier")
        assert result is True
        assert mgr.registry["regime_classifier"].is_loaded is True
        assert mgr.get_object("regime_classifier") == obj


# ---------------------------------------------------------------------------
# load_all()
# ---------------------------------------------------------------------------

class TestLoadAll:
    def test_load_all_returns_dict_with_all_models(self, tmp_path):
        mgr = _make_manager(tmp_path)
        results = mgr.load_all()
        assert isinstance(results, dict)
        for name in MODEL_NAMES:
            assert name in results

    def test_load_all_values_are_bool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        results = mgr.load_all()
        for name, loaded in results.items():
            assert isinstance(loaded, bool), f"{name}: expected bool, got {type(loaded)}"

    def test_load_all_in_memory_model_is_true(self, tmp_path):
        mgr = _make_manager(tmp_path)
        results = mgr.load_all()
        # volatility_forecaster has no disk artefact — should always be True
        assert results["volatility_forecaster"] is True


# ---------------------------------------------------------------------------
# is_stale()
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_never_trained_is_stale(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.is_stale("regime_classifier") is True

    def test_recently_trained_is_not_stale(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.registry["regime_classifier"].trained_at = datetime.now(timezone.utc)
        assert mgr.is_stale("regime_classifier", max_age_hours=168) is False

    def test_old_model_is_stale(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.registry["rl_agent"].trained_at = (
            datetime.now(timezone.utc) - timedelta(days=10)
        )
        assert mgr.is_stale("rl_agent", max_age_hours=168) is True

    def test_custom_max_age(self, tmp_path):
        mgr = _make_manager(tmp_path)
        # Trained 2 hours ago
        mgr.registry["alpha_model"].trained_at = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        )
        assert mgr.is_stale("alpha_model", max_age_hours=1) is True
        assert mgr.is_stale("alpha_model", max_age_hours=4) is False

    def test_unknown_model_is_stale(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.is_stale("ghost_model") is True


# ---------------------------------------------------------------------------
# schedule_retrain()
# ---------------------------------------------------------------------------

class TestScheduleRetrain:
    def test_schedule_adds_to_queue(self, tmp_path):
        mgr = _make_manager(tmp_path)
        job = mgr.schedule_retrain("regime_classifier")
        assert len(mgr._retrain_queue) == 1
        assert job.model_name == "regime_classifier"
        assert job.trigger == "manual"
        assert job.status == "pending"

    def test_schedule_logs_canonical_message(self, tmp_path, caplog):
        import logging
        mgr = _make_manager(tmp_path)
        with caplog.at_level(logging.INFO, logger="ml.model_manager"):
            mgr.schedule_retrain("rl_agent", trigger="weekly_cron")
        assert "Retraining queued: rl_agent (trigger=weekly_cron)" in caplog.text

    def test_schedule_multiple_triggers(self, tmp_path):
        mgr = _make_manager(tmp_path)
        triggers = ["manual", "weekly_cron", "performance_degradation", "concept_drift"]
        for t in triggers:
            mgr.schedule_retrain("alpha_model", trigger=t)
        assert len(mgr._retrain_queue) == 4

    def test_schedule_invalid_trigger_raises(self, tmp_path):
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="Invalid trigger"):
            mgr.schedule_retrain("regime_classifier", trigger="bad_trigger")

    def test_schedule_unknown_model_raises(self, tmp_path):
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="Unknown model"):
            mgr.schedule_retrain("ghost_model")


# ---------------------------------------------------------------------------
# run_pending_retrains()
# ---------------------------------------------------------------------------

class TestRunPendingRetrains:
    def test_run_empty_queue(self, tmp_path):
        mgr = _make_manager(tmp_path)
        processed = mgr.run_pending_retrains()
        assert processed == []

    def test_run_subprocess_success(self, tmp_path):
        """Mocks subprocess.run to return rc=0 and verifies job becomes 'done'."""
        mgr = _make_manager(tmp_path)
        mgr.schedule_retrain("rl_agent", trigger="manual")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("ml.model_manager.subprocess.run", return_value=mock_result):
            processed = mgr.run_pending_retrains()

        assert len(processed) == 1
        assert processed[0].status == "done"
        # Version should have been bumped
        assert mgr.registry["rl_agent"].version == 1

    def test_run_subprocess_failure(self, tmp_path):
        """Mocks subprocess.run to return non-zero and verifies job becomes 'failed'."""
        mgr = _make_manager(tmp_path)
        mgr.schedule_retrain("tft_forecaster", trigger="manual")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some error"

        with patch("ml.model_manager.subprocess.run", return_value=mock_result):
            processed = mgr.run_pending_retrains()

        assert processed[0].status == "failed"

    def test_run_clears_queue(self, tmp_path):
        """After run_pending_retrains the queue should be empty."""
        mgr = _make_manager(tmp_path)
        mgr.schedule_retrain("rl_agent")
        mgr.schedule_retrain("tft_forecaster")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("ml.model_manager.subprocess.run", return_value=mock_result):
            mgr.run_pending_retrains()

        assert len(mgr._retrain_queue) == 0


# ---------------------------------------------------------------------------
# hot_swap()
# ---------------------------------------------------------------------------

class TestHotSwap:
    def test_hot_swap_unknown_model_returns_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.hot_swap("ghost_model", "models/ghost.pkl")
        assert result is False

    def test_hot_swap_missing_file_returns_false_and_reverts(self, tmp_path):
        mgr = _make_manager(tmp_path)
        original_path = mgr.registry["regime_classifier"].path
        result = mgr.hot_swap("regime_classifier", "models/nonexistent.pkl")
        assert result is False
        # Path should be reverted to original
        assert mgr.registry["regime_classifier"].path == original_path

    def test_hot_swap_success_updates_registry(self, tmp_path):
        """Creates a valid pickle file and verifies hot_swap updates version/path."""
        new_pkl = tmp_path / "new_regime.pkl"
        with open(new_pkl, "wb") as fh:
            pickle.dump({"model": "new_version"}, fh)

        from ml.model_manager import ModelManager
        mgr = ModelManager(models_dir=str(tmp_path / "models"))
        # Point path at absolute new file
        result = mgr.hot_swap("regime_classifier", str(new_pkl))
        assert result is True
        assert mgr.registry["regime_classifier"].is_loaded is True
        assert mgr.registry["regime_classifier"].version == 1
        assert mgr.registry["regime_classifier"].path == str(new_pkl)


# ---------------------------------------------------------------------------
# performance_check()
# ---------------------------------------------------------------------------

class TestPerformanceCheck:
    def test_below_threshold_queues_retrain(self, tmp_path):
        mgr = _make_manager(tmp_path)
        triggered = mgr.performance_check("regime_classifier", recent_accuracy=0.40)
        assert triggered is True
        assert len(mgr._retrain_queue) == 1
        assert mgr._retrain_queue[0].trigger == "performance_degradation"

    def test_above_threshold_does_not_queue(self, tmp_path):
        mgr = _make_manager(tmp_path)
        triggered = mgr.performance_check("regime_classifier", recent_accuracy=0.80)
        assert triggered is False
        assert len(mgr._retrain_queue) == 0

    def test_accuracy_recorded_in_registry(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.performance_check("alpha_model", recent_accuracy=0.72)
        assert mgr.registry["alpha_model"].last_accuracy == pytest.approx(0.72)

    def test_at_threshold_boundary(self, tmp_path):
        """Exactly at threshold (0.55) should NOT trigger retrain."""
        mgr = _make_manager(tmp_path)
        triggered = mgr.performance_check("regime_classifier", recent_accuracy=0.55)
        assert triggered is False

    def test_unknown_model_returns_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.performance_check("ghost_model", recent_accuracy=0.3)
        assert result is False


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_has_required_keys(self, tmp_path):
        mgr = _make_manager(tmp_path)
        snap = mgr.snapshot()
        assert "device" in snap
        assert "models" in snap
        assert "pending_retrains" in snap
        assert "pending_retrain_count" in snap

    def test_snapshot_contains_all_models(self, tmp_path):
        mgr = _make_manager(tmp_path)
        snap = mgr.snapshot()
        for name in MODEL_NAMES:
            assert name in snap["models"], f"'{name}' missing from snapshot"

    def test_snapshot_model_entry_keys(self, tmp_path):
        mgr = _make_manager(tmp_path)
        snap = mgr.snapshot()
        required = {
            "name", "path", "version", "trained_at",
            "training_samples", "val_score", "is_loaded",
            "device", "last_accuracy", "is_stale",
        }
        for name in MODEL_NAMES:
            entry_keys = set(snap["models"][name].keys())
            missing = required - entry_keys
            assert not missing, f"'{name}' snapshot missing keys: {missing}"

    def test_snapshot_pending_retrain_count(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.schedule_retrain("rl_agent")
        mgr.schedule_retrain("alpha_model")
        snap = mgr.snapshot()
        assert snap["pending_retrain_count"] == 2
        assert len(snap["pending_retrains"]) == 2

    def test_snapshot_stale_flag_true_when_never_trained(self, tmp_path):
        mgr = _make_manager(tmp_path)
        snap = mgr.snapshot()
        # All models start with trained_at=None → all stale
        for name in MODEL_NAMES:
            assert snap["models"][name]["is_stale"] is True, (
                f"'{name}' should be stale when never trained"
            )

    def test_snapshot_stale_flag_false_after_recent_train(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.registry["alpha_model"].trained_at = datetime.now(timezone.utc)
        snap = mgr.snapshot()
        assert snap["models"]["alpha_model"]["is_stale"] is False


# ---------------------------------------------------------------------------
# get_object()
# ---------------------------------------------------------------------------

class TestGetObject:
    def test_get_object_before_load_returns_none(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.get_object("regime_classifier") is None

    def test_get_object_after_load_returns_object(self, tmp_path):
        """volatility_forecaster is in-memory; object is None but is_loaded True."""
        mgr = _make_manager(tmp_path)
        mgr.load("volatility_forecaster")
        # Object stored is None for in-memory model — but entry exists in dict
        assert "volatility_forecaster" in mgr._loaded_objects
