"""Tests for drift-triggered model quarantine routing."""

from __future__ import annotations

from ml.model_quarantine import ModelQuarantineManager, ModelRouteDecision


class TestModelQuarantineManager:
    def test_config_hash_is_stable(self, tmp_path):
        manager = ModelQuarantineManager(path=tmp_path / "quarantine.jsonl")

        first = manager.config_hash_for("model_v2", model_name="regime", version="2", feature_hash="abc")
        second = manager.config_hash_for("model_v2", model_name="regime", version="2", feature_hash="abc")

        assert first == second
        assert len(first) == 16

    def test_should_quarantine_critical_health(self, tmp_path):
        manager = ModelQuarantineManager(path=tmp_path / "quarantine.jsonl")

        should_quarantine, reasons = manager.should_quarantine({"status": "critical", "drift_score": 0.1})

        assert should_quarantine is True
        assert "critical_health_status" in reasons

    def test_should_quarantine_drift_and_error_rate(self, tmp_path):
        manager = ModelQuarantineManager(
            path=tmp_path / "quarantine.jsonl",
            drift_threshold=0.4,
            error_rate_threshold=0.2,
        )

        should_quarantine, reasons = manager.should_quarantine({"drift_score": 0.5, "error_rate": 0.25})

        assert should_quarantine is True
        assert any(reason.startswith("drift_score") for reason in reasons)
        assert any(reason.startswith("error_rate") for reason in reasons)

    def test_route_allowed_model_without_quarantine(self, tmp_path):
        manager = ModelQuarantineManager(path=tmp_path / "quarantine.jsonl")

        decision = manager.route_model("model_v1")

        assert isinstance(decision, ModelRouteDecision)
        assert decision.active_model_id == "model_v1"
        assert decision.quarantined is False
        assert decision.fallback_used is False

    def test_quarantined_model_routes_to_fallback(self, tmp_path):
        manager = ModelQuarantineManager(
            path=tmp_path / "quarantine.jsonl",
            fallback_models={"model_v2": "model_v1"},
        )

        manager.quarantine_model(run_id="run-1", model_id="model_v2", reasons=["drift"])
        decision = manager.route_model("model_v2")

        assert decision.quarantined is True
        assert decision.fallback_used is True
        assert decision.active_model_id == "model_v1"
        assert decision.reason == "quarantined_using_fallback"

    def test_quarantined_model_without_fallback_records_no_fallback(self, tmp_path):
        manager = ModelQuarantineManager(path=tmp_path / "quarantine.jsonl")

        manager.quarantine_model(run_id="run-1", model_id="model_v2", reasons=["drift"])
        decision = manager.route_model("model_v2")

        assert decision.quarantined is True
        assert decision.fallback_used is False
        assert decision.reason == "quarantined_no_fallback"

    def test_evaluate_and_route_quarantines_degraded_model(self, tmp_path):
        manager = ModelQuarantineManager(
            path=tmp_path / "quarantine.jsonl",
            fallback_models={"model_v2": "model_v1"},
            drift_threshold=0.5,
        )

        decision = manager.evaluate_and_route(
            run_id="run-2",
            model_id="model_v2",
            health={"status": "degraded", "drift_score": 0.7, "error_rate": 0.1},
        )

        assert decision.active_model_id == "model_v1"
        assert decision.fallback_used is True
        assert manager.store.latest() is not None
