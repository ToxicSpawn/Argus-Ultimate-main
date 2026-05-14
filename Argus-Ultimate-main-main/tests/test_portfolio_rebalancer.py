#!/usr/bin/env python3
"""Tests for risk.portfolio_rebalancer — drift detection, regime targets, cost-aware skip."""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock

from risk.portfolio_rebalancer import PortfolioRebalancer, RebalanceOrder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rebalancer():
    """Default rebalancer with 5% drift threshold."""
    return PortfolioRebalancer(config=None)


@pytest.fixture
def tight_rebalancer():
    """Rebalancer with 2% drift threshold for easier triggering."""
    cfg = {"portfolio_rebalancer": {"drift_threshold_pct": 2.0, "min_rebalance_interval_hours": 0}}
    return PortfolioRebalancer(config=cfg)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class TestDriftDetection:

    def test_no_drift_returns_empty(self, rebalancer):
        weights = {"BTC/AUD": 0.50, "ETH/AUD": 0.50}
        orders = rebalancer.check_drift(weights, weights)
        assert orders == []

    def test_small_drift_below_threshold_ignored(self, rebalancer):
        current = {"BTC/AUD": 0.50, "ETH/AUD": 0.50}
        target = {"BTC/AUD": 0.52, "ETH/AUD": 0.48}  # 2% drift < 5% threshold
        orders = rebalancer.check_drift(current, target)
        assert orders == []

    def test_large_drift_triggers_rebalance(self, rebalancer):
        current = {"BTC/AUD": 0.30, "ETH/AUD": 0.70}
        target = {"BTC/AUD": 0.50, "ETH/AUD": 0.50}  # 20% drift > 5%
        orders = rebalancer.check_drift(current, target)
        assert len(orders) == 2
        btc = [o for o in orders if o.symbol == "BTC/AUD"][0]
        assert btc.side == "buy"
        assert btc.delta_weight == pytest.approx(0.20)
        eth = [o for o in orders if o.symbol == "ETH/AUD"][0]
        assert eth.side == "sell"
        assert eth.delta_weight == pytest.approx(-0.20)

    def test_mixed_drift_partial_rebalance(self, tight_rebalancer):
        current = {"BTC/AUD": 0.50, "ETH/AUD": 0.30, "SOL/AUD": 0.20}
        target = {"BTC/AUD": 0.50, "ETH/AUD": 0.35, "SOL/AUD": 0.15}
        orders = tight_rebalancer.check_drift(current, target)
        symbols = {o.symbol for o in orders}
        assert "BTC/AUD" not in symbols  # no drift
        assert "ETH/AUD" in symbols
        assert "SOL/AUD" in symbols

    def test_new_symbol_in_target(self, tight_rebalancer):
        current = {"BTC/AUD": 1.0}
        target = {"BTC/AUD": 0.50, "ETH/AUD": 0.50}
        orders = tight_rebalancer.check_drift(current, target)
        assert len(orders) == 2


# ---------------------------------------------------------------------------
# Regime targets
# ---------------------------------------------------------------------------

class TestRegimeTargets:

    def test_crisis_regime_defensive(self, rebalancer):
        targets = rebalancer.get_regime_targets("crisis")
        assert targets["BTC/AUD"] == 0.35
        assert targets["ETH/AUD"] == 0.25

    def test_bull_regime_equal_weight(self, rebalancer):
        targets = rebalancer.get_regime_targets("bull")
        weights = list(targets.values())
        # All equal
        assert all(w == weights[0] for w in weights)

    def test_trending_regime_equals_bull(self, rebalancer):
        bull = rebalancer.get_regime_targets("bull")
        trending = rebalancer.get_regime_targets("trending")
        assert bull == trending

    def test_bear_regime(self, rebalancer):
        targets = rebalancer.get_regime_targets("bear")
        assert targets["BTC/AUD"] == 0.40
        assert targets["ETH/AUD"] == 0.20

    def test_unknown_regime_fallback(self, rebalancer):
        targets = rebalancer.get_regime_targets("UNKNOWN_REGIME")
        # Falls back to equal weight
        assert len(targets) > 0
        weights = list(targets.values())
        assert all(w == weights[0] for w in weights)


# ---------------------------------------------------------------------------
# Cost-aware skip
# ---------------------------------------------------------------------------

class TestCostAwareSkip:

    def test_high_cost_skips_rebalance(self):
        cost_model = MagicMock()
        cost_est = MagicMock()
        cost_est.total_bps = 99999.0  # absurdly high cost
        cost_model.estimate_cost.return_value = cost_est

        cfg = {
            "portfolio_rebalancer": {
                "drift_threshold_pct": 2.0,
                "cost_aware_skip": True,
                "min_rebalance_interval_hours": 0,
            }
        }
        rb = PortfolioRebalancer(config=cfg, transaction_cost_model=cost_model)
        current = {"BTC/AUD": 0.30}
        target = {"BTC/AUD": 0.50}
        orders = rb.check_drift(current, target)
        assert orders == []


# ---------------------------------------------------------------------------
# Min interval enforcement
# ---------------------------------------------------------------------------

class TestMinInterval:

    def test_interval_blocks_rapid_rebalance(self):
        cfg = {
            "portfolio_rebalancer": {
                "drift_threshold_pct": 2.0,
                "min_rebalance_interval_hours": 4,
            }
        }
        rb = PortfolioRebalancer(config=cfg)
        current = {"BTC/AUD": 0.30}
        target = {"BTC/AUD": 0.60}

        # First call should generate orders
        orders1 = rb.check_drift(current, target)
        assert len(orders1) > 0

        # Immediate second call should be blocked by interval
        orders2 = rb.check_drift(current, target)
        assert orders2 == []

    def test_interval_allows_after_expiry(self):
        cfg = {
            "portfolio_rebalancer": {
                "drift_threshold_pct": 2.0,
                "min_rebalance_interval_hours": 1,
            }
        }
        rb = PortfolioRebalancer(config=cfg)
        current = {"BTC/AUD": 0.30}
        target = {"BTC/AUD": 0.60}

        orders1 = rb.check_drift(current, target)
        assert len(orders1) > 0

        # Manually set last rebalance to >1 hour ago
        rb.last_rebalance_ts = time.time() - 7200
        orders2 = rb.check_drift(current, target)
        assert len(orders2) > 0


# ---------------------------------------------------------------------------
# RebalanceOrder dataclass
# ---------------------------------------------------------------------------

class TestRebalanceOrder:

    def test_dataclass_fields(self):
        order = RebalanceOrder(
            symbol="BTC/AUD",
            side="buy",
            target_weight=0.50,
            current_weight=0.30,
            delta_weight=0.20,
            estimated_cost_bps=5.0,
        )
        assert order.symbol == "BTC/AUD"
        assert order.side == "buy"
        assert order.estimated_cost_bps == 5.0
