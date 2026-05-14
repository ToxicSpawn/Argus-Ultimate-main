"""Tests for champion/challenger ML model promotion gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.ab_testing import (
    ABTestEngine,
    ChampionChallengerConfig,
    ChampionChallengerDecision,
    ChampionChallengerManager,
    ChampionChallengerSummary,
)


def _manager(tmp_path: Path) -> ChampionChallengerManager:
    engine = ABTestEngine(
        db_path=str(tmp_path / "ab_testing.db"),
        bayesian_enabled=False,
        early_stopping_enabled=True,
        bandit_enabled=False,
        bayesian_samples=1000,
    )
    return ChampionChallengerManager(engine=engine)


def _config() -> ChampionChallengerConfig:
    return ChampionChallengerConfig(
        challenge_name="regime_model_upgrade",
        champion_model="regime_v1",
        challenger_model="regime_v2",
        primary_metric="accuracy",
        traffic_to_champion=0.8,
        minimum_samples=20,
        significance_level=0.05,
        test_duration_hours=1,
    )


class TestChampionChallengerManager:
    def test_start_challenge_creates_backing_ab_test(self, tmp_path):
        manager = _manager(tmp_path)

        test_id = manager.start_challenge(_config())

        summary = manager.evaluate(test_id)
        assert isinstance(summary, ChampionChallengerSummary)
        assert summary.status == "active"
        assert summary.champion_samples == 0
        assert summary.challenger_samples == 0

    def test_start_challenge_rejects_same_model_names(self, tmp_path):
        manager = _manager(tmp_path)
        config = _config()
        config.challenger_model = config.champion_model

        with pytest.raises(ValueError, match="must be different"):
            manager.start_challenge(config)

    def test_assign_model_returns_model_name_and_arm(self, tmp_path):
        manager = _manager(tmp_path)
        test_id = manager.start_challenge(_config())

        decision = manager.assign_model(test_id, "symbol=BTCUSDT|regime=trend|1")

        assert isinstance(decision, ChampionChallengerDecision)
        assert decision.test_id == test_id
        assert decision.assigned_arm in {"A", "B"}
        assert decision.assigned_model in {"regime_v1", "regime_v2"}
        assert decision.is_challenger == (decision.assigned_arm == "B")

    def test_assign_model_is_stable_for_same_request(self, tmp_path):
        manager = _manager(tmp_path)
        test_id = manager.start_challenge(_config())

        first = manager.assign_model(test_id, "request-123")
        second = manager.assign_model(test_id, "request-123")

        assert first.assigned_arm == second.assigned_arm
        assert first.assigned_model == second.assigned_model

    def test_record_outcome_by_model_name(self, tmp_path):
        manager = _manager(tmp_path)
        test_id = manager.start_challenge(_config())


        manager.record_outcome(test_id, "req-a", "regime_v1", 0.7)
        manager.record_outcome(test_id, "req-b", "regime_v2", 0.9)

        summary = manager.evaluate(test_id)
        assert summary.champion_samples == 1
        assert summary.challenger_samples == 1
        assert summary.champion_mean == pytest.approx(0.7)
        assert summary.challenger_mean == pytest.approx(0.9)

    def test_record_outcome_rejects_unknown_model(self, tmp_path):
        manager = _manager(tmp_path)
        test_id = manager.start_challenge(_config())

        with pytest.raises(ValueError, match="not part of challenge"):
            manager.record_outcome(test_id, "req-a", "other_model", 0.5)

    def test_evaluate_identifies_clear_challenger_winner(self, tmp_path):
        manager = _manager(tmp_path)
        test_id = manager.start_challenge(_config())

        for i in range(30):
            manager.record_outcome(test_id, f"champion-{i}", "regime_v1", 0.50 + (i % 5) * 0.01)
            manager.record_outcome(test_id, f"challenger-{i}", "regime_v2", 0.85 + (i % 5) * 0.01)

        summary = manager.evaluate(test_id)

        assert summary.winner == "B"
        assert summary.promoted_model == "regime_v2"
        assert summary.challenger_mean > summary.champion_mean
        assert summary.p_value <= 0.05

    def test_promote_if_significant_promotes_challenger(self, tmp_path):
        manager = _manager(tmp_path)
        test_id = manager.start_challenge(_config())

        for i in range(30):
            manager.record_outcome(test_id, f"champion-{i}", "regime_v1", 0.45 + (i % 5) * 0.01)
            manager.record_outcome(test_id, f"challenger-{i}", "regime_v2", 0.90 + (i % 5) * 0.01)

        result = manager.promote_if_significant(test_id)

        assert result["promoted"] is True
        assert result["winner"] == "B"
        assert result["promoted_model"] == "regime_v2"

    def test_promote_if_significant_blocks_champion_when_required(self, tmp_path):
        manager = _manager(tmp_path)
        test_id = manager.start_challenge(_config())

        for i in range(30):
            manager.record_outcome(test_id, f"champion-{i}", "regime_v1", 0.90 + (i % 5) * 0.01)
            manager.record_outcome(test_id, f"challenger-{i}", "regime_v2", 0.45 + (i % 5) * 0.01)

        result = manager.promote_if_significant(test_id)

        assert result["promoted"] is False
        assert result["winner"] == "A"
        assert result["reason"] == "challenger_not_decisive"

    def test_promote_if_significant_can_reaffirm_champion(self, tmp_path):
        manager = _manager(tmp_path)
        test_id = manager.start_challenge(_config())

        for i in range(30):
            manager.record_outcome(test_id, f"champion-{i}", "regime_v1", 0.90 + (i % 5) * 0.01)
            manager.record_outcome(test_id, f"challenger-{i}", "regime_v2", 0.45 + (i % 5) * 0.01)

        result = manager.promote_if_significant(test_id, require_challenger_win=False)

        assert result["promoted"] is True
        assert result["winner"] == "A"
        assert result["promoted_model"] == "regime_v1"
