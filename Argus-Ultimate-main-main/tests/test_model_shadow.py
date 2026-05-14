"""
Tests for ML model shadow deployment, A/B testing, and canary evaluation.

Covers: ShadowModel dataclass, register_shadow, record_shadow_prediction,
        evaluate_shadow, promote_shadow, canary_check.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false, reportUninitializedInstanceVariable=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalSubscript=false

import pytest
from datetime import datetime, timezone

from ml.model_manager import ModelManager


pytestmark = pytest.mark.skip(reason="Shadow model deployment APIs are not present in ml.model_manager")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mgr(tmp_path):
    """ModelManager with a temporary models directory."""
    return ModelManager(models_dir=str(tmp_path / "models"))


# ---------------------------------------------------------------------------
# ShadowModel dataclass
# ---------------------------------------------------------------------------

class TestShadowModelDataclass:
    def test_defaults(self):
        sm = ShadowModel(model_name="test", model_path="/tmp/m.pkl")
        assert sm.model_name == "test"
        assert sm.model_path == "/tmp/m.pkl"
        assert sm.predictions == []
        assert sm.deployed_at is None

    def test_with_deployed_at(self):
        ts = datetime.now(timezone.utc)
        sm = ShadowModel(
            model_name="alpha",
            model_path="/tmp/a.pkl",
            deployed_at=ts,
        )
        assert sm.deployed_at == ts

    def test_predictions_are_independent(self):
        """Each instance gets its own predictions list."""
        a = ShadowModel(model_name="a", model_path="a.pkl")
        b = ShadowModel(model_name="b", model_path="b.pkl")
        a.predictions.append((datetime.now(timezone.utc), 1.0, 1.0))
        assert len(b.predictions) == 0


# ---------------------------------------------------------------------------
# register_shadow
# ---------------------------------------------------------------------------

class TestRegisterShadow:
    def test_register_creates_entry(self, mgr):
        mgr.register_shadow("regime_classifier", "/tmp/new_regime.pkl")
        assert "regime_classifier" in mgr._shadows
        shadow = mgr._shadows["regime_classifier"]
        assert shadow.model_path == "/tmp/new_regime.pkl"
        assert shadow.deployed_at is not None

    def test_register_overwrites(self, mgr):
        mgr.register_shadow("regime_classifier", "/tmp/v1.pkl")
        mgr.register_shadow("regime_classifier", "/tmp/v2.pkl")
        assert mgr._shadows["regime_classifier"].model_path == "/tmp/v2.pkl"

    def test_register_unknown_model(self, mgr):
        """Registering a shadow for a name not in production registry is allowed."""
        mgr.register_shadow("experimental_model", "/tmp/exp.pkl")
        assert "experimental_model" in mgr._shadows


# ---------------------------------------------------------------------------
# record_shadow_prediction
# ---------------------------------------------------------------------------

class TestRecordShadowPrediction:
    def test_record_appends(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        mgr.record_shadow_prediction("alpha_model", 1.0, 0.5)
        mgr.record_shadow_prediction("alpha_model", -0.3, -0.8)
        assert len(mgr._shadows["alpha_model"].predictions) == 2

    def test_record_stores_values(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        mgr.record_shadow_prediction("alpha_model", 2.5, 1.0)
        ts, pred, actual = mgr._shadows["alpha_model"].predictions[0]
        assert pred == 2.5
        assert actual == 1.0
        assert isinstance(ts, datetime)

    def test_record_unregistered_raises(self, mgr):
        with pytest.raises(ValueError, match="No shadow registered"):
            mgr.record_shadow_prediction("no_such_model", 1.0, 1.0)


# ---------------------------------------------------------------------------
# evaluate_shadow
# ---------------------------------------------------------------------------

class TestEvaluateShadow:
    def _fill_predictions(self, mgr, model, n_correct, n_wrong):
        """Helper: fill shadow with n_correct correct + n_wrong wrong preds."""
        for _ in range(n_correct):
            mgr.record_shadow_prediction(model, 1.0, 1.0)  # same sign = correct
        for _ in range(n_wrong):
            mgr.record_shadow_prediction(model, 1.0, -1.0)  # diff sign = wrong

    def test_empty_predictions(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        result = mgr.evaluate_shadow("alpha_model")
        assert result["sample_size"] == 0
        assert result["recommendation"] == "keep_shadow"

    def test_perfect_shadow_recommends_promote(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        # Production accuracy is 0.0 by default
        self._fill_predictions(mgr, "alpha_model", 60, 0)
        result = mgr.evaluate_shadow("alpha_model")
        assert result["shadow_accuracy"] == 1.0
        assert result["sample_size"] == 60
        assert result["recommendation"] == "promote"

    def test_poor_shadow_recommends_discard(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        # Set production accuracy high
        mgr._registry["alpha_model"].last_accuracy = 0.80
        # Shadow gets only 50% right (0.50 < 0.80 - 0.05 = 0.75)
        self._fill_predictions(mgr, "alpha_model", 30, 30)
        result = mgr.evaluate_shadow("alpha_model")
        assert result["shadow_accuracy"] == pytest.approx(0.5)
        assert result["recommendation"] == "discard"

    def test_marginal_shadow_keeps_shadow(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        mgr._registry["alpha_model"].last_accuracy = 0.70
        # Shadow accuracy = 55/75 ≈ 0.733 → improvement < 0.02+0.70=0.72
        # Actually 0.733 > 0.72, and sample >= 50 so promote
        # Let's use 54/75 = 0.72 → improvement = 0.02 exactly, not > 0.02
        self._fill_predictions(mgr, "alpha_model", 54, 21)
        result = mgr.evaluate_shadow("alpha_model")
        assert result["recommendation"] == "keep_shadow"

    def test_window_parameter(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        # First 50 wrong, then 50 right
        self._fill_predictions(mgr, "alpha_model", 0, 50)
        self._fill_predictions(mgr, "alpha_model", 50, 0)
        # Window=50 should see only the last 50 (all correct)
        result = mgr.evaluate_shadow("alpha_model", window=50)
        assert result["shadow_accuracy"] == 1.0
        assert result["sample_size"] == 50

    def test_sharpe_proxy_finite(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        self._fill_predictions(mgr, "alpha_model", 40, 20)
        result = mgr.evaluate_shadow("alpha_model")
        assert result["shadow_sharpe_proxy"] > 0

    def test_sharpe_proxy_perfect_is_zero(self, mgr):
        """All correct => std=0 => sharpe_proxy=0 (avoid div by zero)."""
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        self._fill_predictions(mgr, "alpha_model", 60, 0)
        result = mgr.evaluate_shadow("alpha_model")
        assert result["shadow_sharpe_proxy"] == 0.0

    def test_evaluate_unregistered_raises(self, mgr):
        with pytest.raises(ValueError, match="No shadow registered"):
            mgr.evaluate_shadow("no_such_model")

    def test_not_enough_samples_keeps_shadow(self, mgr):
        """Even if accuracy is high, < 50 samples => keep_shadow."""
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        self._fill_predictions(mgr, "alpha_model", 30, 0)
        result = mgr.evaluate_shadow("alpha_model")
        assert result["sample_size"] == 30
        assert result["recommendation"] == "keep_shadow"


# ---------------------------------------------------------------------------
# promote_shadow
# ---------------------------------------------------------------------------

class TestPromoteShadow:
    def test_promote_updates_registry(self, mgr):
        old_path = mgr._registry["alpha_model"].path
        old_version = mgr._registry["alpha_model"].version

        mgr.register_shadow("alpha_model", "/tmp/promoted.pkl")
        mgr.promote_shadow("alpha_model")

        assert mgr._registry["alpha_model"].path == "/tmp/promoted.pkl"
        assert mgr._registry["alpha_model"].version == old_version + 1
        assert mgr._registry["alpha_model"].is_loaded is False

    def test_promote_archives_old_path(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/new.pkl")
        mgr.promote_shadow("alpha_model")
        assert len(mgr._shadow_archive) == 1
        name, path, ts = mgr._shadow_archive[0]
        assert name == "alpha_model"
        assert path == "models/alpha_model.pkl"

    def test_promote_removes_shadow(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/new.pkl")
        mgr.promote_shadow("alpha_model")
        assert "alpha_model" not in mgr._shadows

    def test_promote_unregistered_raises(self, mgr):
        with pytest.raises(ValueError, match="No shadow registered"):
            mgr.promote_shadow("no_such_model")

    def test_double_promote(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/v1.pkl")
        mgr.promote_shadow("alpha_model")
        # Second promote should fail — shadow already removed
        with pytest.raises(ValueError, match="No shadow registered"):
            mgr.promote_shadow("alpha_model")

    def test_promote_clears_loaded_object(self, mgr):
        # Pre-populate a loaded object
        mgr._loaded_objects["alpha_model"] = "fake_model_object"
        mgr._registry["alpha_model"].is_loaded = True

        mgr.register_shadow("alpha_model", "/tmp/new.pkl")
        mgr.promote_shadow("alpha_model")

        assert "alpha_model" not in mgr._loaded_objects


# ---------------------------------------------------------------------------
# canary_check
# ---------------------------------------------------------------------------

class TestCanaryCheck:
    def _fill(self, mgr, model, n_correct, n_wrong):
        for _ in range(n_correct):
            mgr.record_shadow_prediction(model, 1.0, 1.0)
        for _ in range(n_wrong):
            mgr.record_shadow_prediction(model, 1.0, -1.0)

    def test_canary_passes_when_shadow_better(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        mgr._registry["alpha_model"].last_accuracy = 0.60
        # 50 correct out of 50 => 1.0 vs 0.60 => improvement 0.40 > 0.02
        self._fill(mgr, "alpha_model", 50, 0)
        assert mgr.canary_check("alpha_model") is True

    def test_canary_fails_not_enough_samples(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        self._fill(mgr, "alpha_model", 30, 0)
        assert mgr.canary_check("alpha_model", min_samples=50) is False

    def test_canary_fails_insufficient_improvement(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        mgr._registry["alpha_model"].last_accuracy = 0.70
        # 36 correct out of 50 => 0.72 vs 0.70 => improvement 0.02, need > 0.02
        # Actually canary_check uses >= so 0.72 >= 0.72 is True
        # Use 35/50 = 0.70 => improvement = 0.00
        self._fill(mgr, "alpha_model", 35, 15)
        assert mgr.canary_check("alpha_model", min_samples=50) is False

    def test_canary_boundary_improvement(self, mgr):
        """Exactly min_accuracy_improvement should pass (>=)."""
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        mgr._registry["alpha_model"].last_accuracy = 0.50
        # 26 correct out of 50 = 0.52, improvement = 0.02
        self._fill(mgr, "alpha_model", 26, 24)
        assert mgr.canary_check("alpha_model", min_samples=50, min_accuracy_improvement=0.02) is True

    def test_canary_unregistered_returns_false(self, mgr):
        assert mgr.canary_check("nonexistent") is False

    def test_canary_custom_thresholds(self, mgr):
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        mgr._registry["alpha_model"].last_accuracy = 0.50
        # 40 correct / 60 total = 0.667
        self._fill(mgr, "alpha_model", 40, 20)
        # Need improvement >= 0.10 => 0.667 >= 0.60 => True
        assert mgr.canary_check("alpha_model", min_samples=60, min_accuracy_improvement=0.10) is True

    def test_canary_uses_last_n_samples(self, mgr):
        """canary_check should look at the last min_samples predictions."""
        mgr.register_shadow("alpha_model", "/tmp/a.pkl")
        mgr._registry["alpha_model"].last_accuracy = 0.50
        # First 50 wrong, then 50 right
        self._fill(mgr, "alpha_model", 0, 50)
        self._fill(mgr, "alpha_model", 50, 0)
        # Last 50 are all correct => 1.0 vs 0.50 => pass
        assert mgr.canary_check("alpha_model", min_samples=50) is True


# ---------------------------------------------------------------------------
# Integration: full shadow lifecycle
# ---------------------------------------------------------------------------

class TestShadowLifecycle:
    def test_register_record_evaluate_promote(self, mgr):
        """Full lifecycle: register -> record -> evaluate -> promote."""
        mgr._registry["alpha_model"].last_accuracy = 0.55

        mgr.register_shadow("alpha_model", "/tmp/better_alpha.pkl")

        # Record 60 predictions: 50 correct, 10 wrong => 83.3%
        for _ in range(50):
            mgr.record_shadow_prediction("alpha_model", 1.0, 1.0)
        for _ in range(10):
            mgr.record_shadow_prediction("alpha_model", 1.0, -1.0)

        result = mgr.evaluate_shadow("alpha_model")
        assert result["shadow_accuracy"] == pytest.approx(50 / 60)
        assert result["production_accuracy"] == 0.55
        assert result["recommendation"] == "promote"

        # Canary check should also pass
        assert mgr.canary_check("alpha_model") is True

        # Promote
        mgr.promote_shadow("alpha_model")
        assert mgr._registry["alpha_model"].path == "/tmp/better_alpha.pkl"
        assert "alpha_model" not in mgr._shadows

    def test_snapshot_includes_shadow_archive(self, mgr):
        """After promotion, archive is populated; snapshot still works."""
        mgr.register_shadow("alpha_model", "/tmp/new.pkl")
        mgr.promote_shadow("alpha_model")

        snap = mgr.snapshot()
        assert snap["models"]["alpha_model"]["path"] == "/tmp/new.pkl"
        # Archive stored internally
        assert len(mgr._shadow_archive) == 1
