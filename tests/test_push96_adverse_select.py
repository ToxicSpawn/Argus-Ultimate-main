"""Push 96 — Tests for AdverseSelect tier (v8.32.0)."""
from __future__ import annotations

import pytest

from core.signals.adverse_selection_model import (
    AdverseSelectionFeatures,
    AdverseSelectionGate,
    AdverseSelectionModel,
    TradeOutcome,
)
from core.signals.fill_probability_model import (
    FillOutcome,
    FillProbabilityAdvisor,
    FillProbabilityFeatures,
    FillProbabilityModel,
)
from core.signals.funding_rate_predictor import (
    FundingOutcome,
    FundingRateAdvisor,
    FundingRateFeatures,
    FundingRatePredictor,
)
from core.signals.slippage_calibrator import (
    SlippageEstimator,
    SlippageSample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_adv_features(obi: float = 0.2, spread: float = 5.0) -> AdverseSelectionFeatures:
    return AdverseSelectionFeatures(
        obi=obi, spread_bps=spread, vol_ratio=1.5,
        trade_flow=0.1, depth_ratio=0.6,
        microprice=0.01, momentum=0.002,
    )


def make_fill_features(dist: float = 2.0, budget: float = 30.0) -> FillProbabilityFeatures:
    return FillProbabilityFeatures(
        price_distance_bps=dist, queue_position=0.3,
        depth_at_level=0.05, vol_ratio=1.2,
        spread_bps=4.0, obi=0.1,
        time_budget_secs=budget, side_sign=1.0,
    )


def make_funding_features(current: float = 0.0001) -> FundingRateFeatures:
    return FundingRateFeatures(
        current_funding=current, funding_ema_3=current * 0.9,
        funding_ema_8=current * 0.8, oi_delta_pct=0.02,
        basis_bps=3.0, vol_ratio=1.1,
        obi_ema=0.05, liq_imbalance=0.55,
    )


# ---------------------------------------------------------------------------
# AdverseSelectionModel
# ---------------------------------------------------------------------------

class TestAdverseSelectionModel:
    def test_predict_returns_probability(self):
        m = AdverseSelectionModel()
        p = m.predict(make_adv_features())
        assert 0.0 <= p <= 1.0

    def test_heuristic_high_spread_raises_prob(self):
        m = AdverseSelectionModel()
        p_low  = m.predict(make_adv_features(spread=1.0))
        p_high = m.predict(make_adv_features(spread=50.0))
        assert p_high > p_low

    def test_record_outcome_increments_samples(self):
        m = AdverseSelectionModel(min_train_size=9999)  # prevent retrain
        for _ in range(10):
            m.record_outcome(TradeOutcome(features=make_adv_features(), adverse=True))
        assert m.stats["total_samples"] == 10

    def test_stats_structure(self):
        m = AdverseSelectionModel()
        s = m.stats
        assert "total_samples" in s
        assert "model_ready" in s

    def test_is_adverse_threshold(self):
        m = AdverseSelectionModel()
        # With very high spread + negative OBI, heuristic should be high
        f = AdverseSelectionFeatures(
            obi=-0.9, spread_bps=100.0, vol_ratio=3.0,
            trade_flow=-0.8, depth_ratio=0.1,
            microprice=0.0, momentum=-0.01,
        )
        assert m.is_adverse(f, threshold=0.3)


# ---------------------------------------------------------------------------
# AdverseSelectionGate
# ---------------------------------------------------------------------------

class TestAdverseSelectionGate:
    def test_should_block_returns_bool(self):
        g = AdverseSelectionGate()
        result = g.should_block(make_adv_features())
        assert isinstance(result, bool)

    def test_record_increments_nothing_but_doesnt_crash(self):
        g = AdverseSelectionGate()
        g.record(make_adv_features(), adverse=False)
        g.record(make_adv_features(), adverse=True)

    def test_stats_contains_blocked_passed(self):
        g = AdverseSelectionGate()
        g.should_block(make_adv_features(obi=-0.95, spread=200.0))  # likely blocked
        s = g.stats
        assert "blocked" in s and "passed" in s


# ---------------------------------------------------------------------------
# FillProbabilityModel
# ---------------------------------------------------------------------------

class TestFillProbabilityModel:
    def test_predict_in_range(self):
        m = FillProbabilityModel()
        p = m.predict(make_fill_features())
        assert 0.0 <= p <= 1.0

    def test_passive_order_lower_fill_prob(self):
        m = FillProbabilityModel()
        p_close   = m.predict(make_fill_features(dist=0.5))
        p_passive = m.predict(make_fill_features(dist=50.0))
        assert p_close > p_passive

    def test_longer_budget_higher_fill_prob(self):
        m = FillProbabilityModel()
        p_short = m.predict(make_fill_features(budget=1.0))
        p_long  = m.predict(make_fill_features(budget=300.0))
        assert p_long > p_short

    def test_record_outcome_no_crash(self):
        m = FillProbabilityModel(min_train_size=9999)
        m.record_outcome(FillOutcome(features=make_fill_features(), filled=True, fill_time=0.8))


# ---------------------------------------------------------------------------
# FillProbabilityAdvisor
# ---------------------------------------------------------------------------

class TestFillProbabilityAdvisor:
    def test_convert_threshold_respected(self):
        adv = FillProbabilityAdvisor(convert_threshold=0.99)  # almost always convert
        result = adv.should_convert_to_market(make_fill_features(dist=100.0, budget=0.1))
        assert isinstance(result, bool)

    def test_stats_has_converted(self):
        adv = FillProbabilityAdvisor()
        adv.should_convert_to_market(make_fill_features())
        assert "converted" in adv.stats


# ---------------------------------------------------------------------------
# FundingRatePredictor
# ---------------------------------------------------------------------------

class TestFundingRatePredictor:
    def test_predict_returns_float(self):
        p = FundingRatePredictor()
        val = p.predict(make_funding_features())
        assert isinstance(val, float)

    def test_record_outcome_no_crash(self):
        p = FundingRatePredictor(min_train_size=9999)
        p.record_outcome(FundingOutcome(
            features=make_funding_features(), actual_funding=0.0001
        ))

    def test_heuristic_follows_ema(self):
        p = FundingRatePredictor()
        f = make_funding_features(current=0.001)
        val = p.predict(f)
        # Should be in the ballpark of current_funding
        assert abs(val) < 0.01


# ---------------------------------------------------------------------------
# FundingRateAdvisor
# ---------------------------------------------------------------------------

class TestFundingRateAdvisor:
    def test_best_entry_high_positive_funding(self):
        adv = FundingRateAdvisor(pos_threshold=0.00001)
        f = make_funding_features(current=0.001)  # very high positive
        assert adv.best_entry_side(f) == "SHORT"

    def test_best_entry_neutral_funding(self):
        adv = FundingRateAdvisor()
        f = make_funding_features(current=0.0001)  # below threshold
        result = adv.best_entry_side(f)
        assert result in ("LONG", "SHORT", "NEUTRAL")

    def test_record_no_crash(self):
        adv = FundingRateAdvisor()
        adv.record(make_funding_features(), actual_funding=0.0002)


# ---------------------------------------------------------------------------
# SlippageEstimator
# ---------------------------------------------------------------------------

class TestSlippageEstimator:
    def test_default_half_spread_heuristic(self):
        est = SlippageEstimator()
        val = est.estimate_bps("BTCUSDT", "BUY", spread_bps=10.0, vol_ratio=1.0)
        assert val == pytest.approx(5.0)

    def test_ingest_fill_no_crash(self):
        est = SlippageEstimator()
        est.ingest_fill(
            symbol="BTCUSDT", side="BUY",
            qty=0.1, notional=5000.0,
            spread_bps=6.0, vol_ratio=1.2,
            intended_mid=50000.0, actual_fill=50003.0,
        )

    def test_calibrated_estimate_after_many_fills(self):
        est = SlippageEstimator()
        for i in range(60):
            est.ingest_fill(
                symbol="ETHUSDT", side="SELL",
                qty=1.0, notional=3000.0,
                spread_bps=5.0 + i * 0.1,
                vol_ratio=1.0 + i * 0.01,
                intended_mid=3000.0, actual_fill=3001.5,
            )
        val = est.estimate_bps("ETHUSDT", "SELL", spread_bps=5.5, vol_ratio=1.1)
        assert val >= 0.0

    def test_stats_known_pairs(self):
        est = SlippageEstimator()
        s = est.stats
        assert "known_pairs" in s
