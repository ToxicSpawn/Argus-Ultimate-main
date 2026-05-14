"""Tests for trading ML decision controls."""

from __future__ import annotations

import pytest

from ml.trading_decision_controls import (
    ConfidenceTradeGate,
    CostAwareScorer,
    CostModel,
    PredictionCalibrator,
)


class TestCostAwareScorer:
    def test_cost_model_converts_bps_to_penalty(self):
        cost_model = CostModel(fee_bps=2.0, slippage_bps=8.0, spread_bps=5.0, turnover=2.0)

        assert cost_model.total_cost_bps == pytest.approx(30.0)
        assert cost_model.penalty() == pytest.approx(0.003)

    def test_cost_aware_score_penalizes_high_cost_model(self):
        scorer = CostAwareScorer()
        low_cost = scorer.score(0.80, CostModel(slippage_bps=5.0))
        high_cost = scorer.score(0.81, CostModel(slippage_bps=500.0))

        ranked = scorer.rank({"low_cost": low_cost, "high_cost": high_cost})

        assert low_cost.net_score > high_cost.net_score
        assert ranked[0][0] == "low_cost"

    def test_cost_model_from_tca_report(self):
        report = {
            "fee_bps": 1.0,
            "average_costs_bps": {
                "spread": 2.0,
                "slippage": 3.0,
                "market_impact": 4.0,
            },
        }

        cost_model = CostModel.from_tca_report(report, turnover=2.0)

        assert cost_model.total_cost_bps == pytest.approx(20.0)


class TestPredictionCalibrator:
    def test_identity_before_fit(self):
        calibrator = PredictionCalibrator()

        result = calibrator.calibrate(0.72)

        assert result.calibrated_confidence == pytest.approx(0.72)
        assert result.method == "identity"

    def test_fit_and_calibrate_uses_empirical_accuracy(self):
        calibrator = PredictionCalibrator(n_bins=2, min_bucket_samples=2, shrinkage=0.0)
        calibrator.fit(
            confidences=[0.1, 0.2, 0.7, 0.8, 0.9],
            outcomes=[False, False, True, True, False],
        )

        low = calibrator.calibrate(0.2)
        high = calibrator.calibrate(0.85)

        assert low.calibrated_confidence == pytest.approx(0.0)
        assert high.calibrated_confidence == pytest.approx(2 / 3)
        assert high.uncertainty == pytest.approx(1 / 3)

    def test_fit_rejects_mismatched_lengths(self):
        calibrator = PredictionCalibrator()

        with pytest.raises(ValueError, match="same length"):
            calibrator.fit([0.1, 0.2], [True])


class TestConfidenceTradeGate:
    def test_low_confidence_blocks_trade(self):
        gate = ConfidenceTradeGate(min_confidence=0.6)

        decision = gate.evaluate("buy", confidence=0.4)

        assert decision.should_trade is False
        assert decision.action == "hold"
        assert decision.size_multiplier == 0.0
        assert decision.reason == "confidence_below_threshold"

    def test_high_confidence_allows_scaled_trade(self):
        gate = ConfidenceTradeGate(min_confidence=0.5, max_uncertainty=0.8)

        decision = gate.evaluate("sell", confidence=0.8, uncertainty=0.2)

        assert decision.should_trade is True
        assert decision.action == "sell"
        assert 0.0 < decision.size_multiplier <= 1.0
        assert decision.reason == "trade_allowed"

    def test_non_entry_actions_are_not_traded(self):
        gate = ConfidenceTradeGate(min_confidence=0.1)

        decision = gate.evaluate("reduce", confidence=0.9)

        assert decision.should_trade is False
        assert decision.reason == "non_entry_action"
