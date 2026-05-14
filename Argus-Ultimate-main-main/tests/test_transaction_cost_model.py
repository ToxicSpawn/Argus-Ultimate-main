#!/usr/bin/env python3
"""Tests for risk.transaction_cost_model — cost estimation components."""

from __future__ import annotations

import pytest

from risk.transaction_cost_model import TransactionCostModel, CostEstimate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return TransactionCostModel(config=None)


@pytest.fixture
def custom_model():
    cfg = {
        "transaction_cost_model": {
            "default_spread_bps": 3.0,
            "commission_bps": 10.0,
            "slippage_pct": 0.002,
            "market_impact_coefficient": 0.2,
        }
    }
    return TransactionCostModel(config=cfg)


# ---------------------------------------------------------------------------
# Basic cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:

    def test_returns_cost_estimate(self, model):
        est = model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        assert isinstance(est, CostEstimate)

    def test_default_commission(self, model):
        est = model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        assert est.commission_bps == 26.0

    def test_default_spread(self, model):
        est = model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        assert est.spread_bps == 5.0

    def test_default_slippage(self, model):
        est = model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        assert est.slippage_bps == 10.0  # 0.001 * 10000

    def test_total_bps_sum(self, model):
        est = model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        expected = est.spread_bps + est.commission_bps + est.market_impact_bps + est.slippage_bps
        assert est.total_bps == pytest.approx(expected, abs=0.1)

    def test_total_usd_calculated(self, model):
        est = model.estimate_cost("BTC/AUD", 1.0, "buy", 100_000.0)
        # total_usd = (total_bps / 10000) * notional
        assert est.total_usd > 0

    def test_custom_config(self, custom_model):
        est = custom_model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        assert est.commission_bps == 10.0
        assert est.spread_bps == 3.0
        assert est.slippage_bps == 20.0  # 0.002 * 10000


# ---------------------------------------------------------------------------
# L2 data injection
# ---------------------------------------------------------------------------

class TestL2Data:

    def test_l2_spread_overrides_default(self, model):
        model.update_l2_spread("BTC/AUD", 2.0)
        est = model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        assert est.spread_bps == 2.0

    def test_clear_l2_falls_back(self, model):
        model.update_l2_spread("BTC/AUD", 2.0)
        model.clear_l2_spread("BTC/AUD")
        est = model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        assert est.spread_bps == 5.0  # default


# ---------------------------------------------------------------------------
# Market impact
# ---------------------------------------------------------------------------

class TestMarketImpact:

    def test_no_adv_means_zero_impact(self, model):
        est = model.estimate_cost("BTC/AUD", 0.1, "buy", 100_000.0)
        assert est.market_impact_bps == 0.0

    def test_with_adv_positive_impact(self, model):
        model.update_adv("BTC/AUD", 50_000_000.0)  # 50M daily volume
        est = model.estimate_cost("BTC/AUD", 1.0, "buy", 100_000.0)
        assert est.market_impact_bps > 0.0

    def test_impact_scales_with_size(self, model):
        model.update_adv("BTC/AUD", 50_000_000.0)
        small = model.estimate_cost("BTC/AUD", 0.01, "buy", 100_000.0)
        large = model.estimate_cost("BTC/AUD", 10.0, "buy", 100_000.0)
        assert large.market_impact_bps > small.market_impact_bps


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_price(self, model):
        est = model.estimate_cost("BTC/AUD", 1.0, "buy", 0.0)
        assert est.total_usd == 0.0

    def test_zero_quantity(self, model):
        est = model.estimate_cost("BTC/AUD", 0.0, "buy", 100_000.0)
        assert est.total_usd == 0.0
