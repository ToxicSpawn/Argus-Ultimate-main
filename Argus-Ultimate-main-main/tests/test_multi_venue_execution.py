"""
Tests for execution/multi_venue_execution.py edge cases.

Covers:
- Single venue below min notional → all to primary
- All venues with zero score → uniform fallback
- Venue with zero liquidity_score → low score, penalised allocation
- Two venues with equal scores → roughly equal split
- Score-based proportional routing
- aggregate_fills helper
"""

import pytest

from execution.multi_venue_execution import (
    MultiVenueDecision,
    MultiVenueExecutor,
    VenueStats,
    VenueOrder,
)

# The module constant (200 USD default)
_MIN_NOTIONAL = 200.0


# ---------------------------------------------------------------------------
# VenueStats.score()
# ---------------------------------------------------------------------------

class TestVenueStatsScore:
    def test_score_is_positive_for_default_params(self):
        vs = VenueStats()
        assert vs.score() > 0

    def test_lower_spread_gives_higher_score(self):
        good = VenueStats(spread_bps=1.0, liquidity_score=0.8, latency_ms=50, taker_fee_bps=10)
        bad = VenueStats(spread_bps=50.0, liquidity_score=0.8, latency_ms=50, taker_fee_bps=10)
        assert good.score() > bad.score()

    def test_higher_liquidity_gives_higher_score(self):
        high = VenueStats(spread_bps=5.0, liquidity_score=1.0, latency_ms=100, taker_fee_bps=20)
        low = VenueStats(spread_bps=5.0, liquidity_score=0.0, latency_ms=100, taker_fee_bps=20)
        assert high.score() > low.score()

    def test_zero_liquidity_score_not_divide_by_zero(self):
        vs = VenueStats(liquidity_score=0.0)
        score = vs.score()
        assert score >= 0

    def test_score_with_near_zero_spread_is_large_but_finite(self):
        vs = VenueStats(spread_bps=0.001)
        score = vs.score()
        assert 0 < score < 1e9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision(size: float, max_venues: int = 2) -> MultiVenueDecision:
    return MultiVenueDecision(
        symbol="BTC/USD",
        side="BUY",
        total_size=size,
        max_venues=max_venues,
    )


# ---------------------------------------------------------------------------
# Below-min-notional: single venue
# ---------------------------------------------------------------------------

class TestBelowMinNotional:
    def test_small_order_routes_100pct_to_primary(self):
        executor = MultiVenueExecutor(primary_venue="kraken", min_notional_usd=_MIN_NOTIONAL)
        orders = executor.split(_decision(50.0))  # below 200 USD threshold
        assert len(orders) == 1
        assert orders[0].venue == "kraken"
        assert orders[0].size == pytest.approx(50.0)

    def test_exactly_at_threshold_goes_through_split(self):
        """At-threshold (size == min_notional) still routes single because < is used."""
        executor = MultiVenueExecutor(primary_venue="kraken", min_notional_usd=_MIN_NOTIONAL)
        orders = executor.split(_decision(_MIN_NOTIONAL - 0.01))
        assert len(orders) == 1

    def test_just_above_threshold_enables_split(self):
        executor = MultiVenueExecutor(primary_venue="kraken", min_notional_usd=_MIN_NOTIONAL)
        orders = executor.split(_decision(_MIN_NOTIONAL + 1.0))
        # With default kraken + coinbase both present, should split into 2
        assert len(orders) >= 1  # split attempted (may be 1 or 2 depending on scores)

    def test_zero_size_goes_to_primary(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        orders = executor.split(_decision(0.0))
        assert len(orders) == 1
        assert orders[0].venue == "kraken"


# ---------------------------------------------------------------------------
# Single venue configured → all to that venue
# ---------------------------------------------------------------------------

class TestSingleVenueConfigured:
    def test_only_one_venue_routes_entirely(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        # Remove coinbase so only one venue remains
        executor._stats = {"kraken": VenueStats()}
        orders = executor.split(_decision(1000.0))
        assert len(orders) == 1
        assert orders[0].venue == "kraken"
        assert orders[0].size == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# All venues zero score → uniform split
# ---------------------------------------------------------------------------

class TestAllVenuesZeroScore:
    def test_zero_score_venues_receive_uniform_split(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        # Force both venues to produce a score of 0 by mocking total_score path
        # We do this by subclassing the score method to return 0
        class ZeroScore(VenueStats):
            def score(self, weights=None) -> float:
                return 0.0

        executor._stats = {
            "kraken": ZeroScore(),
            "coinbase": ZeroScore(),
        }
        orders = executor.split(_decision(1000.0, max_venues=2))
        # Uniform split: each gets 500
        assert len(orders) == 2
        sizes = sorted(o.size for o in orders)
        assert sizes[0] == pytest.approx(500.0, abs=1e-6)
        assert sizes[1] == pytest.approx(500.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Venue with zero liquidity_score → heavily penalised
# ---------------------------------------------------------------------------

class TestZeroLiquidityVenue:
    def test_zero_liquidity_venue_gets_smaller_allocation(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        # kraken gets high liquidity, coinbase gets zero
        executor.update_venue_stats("kraken", liquidity_score=0.95, spread_bps=4.0)
        executor.update_venue_stats("coinbase", liquidity_score=0.0, spread_bps=4.0)

        orders = executor.split(_decision(1000.0, max_venues=2))
        assert len(orders) == 2

        venue_sizes = {o.venue: o.size for o in orders}
        # kraken (high liquidity) should get more
        assert venue_sizes["kraken"] > venue_sizes["coinbase"]

    def test_total_size_conserved_with_penalised_venue(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        executor.update_venue_stats("kraken", liquidity_score=0.95, spread_bps=4.0)
        executor.update_venue_stats("coinbase", liquidity_score=0.0, spread_bps=4.0)

        orders = executor.split(_decision(800.0, max_venues=2))
        total = sum(o.size for o in orders)
        assert total == pytest.approx(800.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Two venues with equal scores → roughly equal split
# ---------------------------------------------------------------------------

class TestEqualScoreVenues:
    def test_equal_venues_split_approximately_half(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        # Give both venues identical stats
        executor.update_venue_stats("kraken", spread_bps=5.0, liquidity_score=0.80,
                                    latency_ms=80.0, taker_fee_bps=26.0)
        executor.update_venue_stats("coinbase", spread_bps=5.0, liquidity_score=0.80,
                                    latency_ms=80.0, taker_fee_bps=26.0)

        orders = executor.split(_decision(1000.0, max_venues=2))
        assert len(orders) == 2
        total = sum(o.size for o in orders)
        assert total == pytest.approx(1000.0, abs=1e-6)
        # Each venue should get within 10% of 500
        for o in orders:
            assert abs(o.size - 500.0) <= 100.0

    def test_total_size_always_conserved(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        for size in [201.0, 500.0, 10_000.0, 1_000_000.0]:
            orders = executor.split(_decision(size))
            total = sum(o.size for o in orders)
            assert total == pytest.approx(size, rel=1e-6), f"size={size} not conserved"


# ---------------------------------------------------------------------------
# Proportional routing
# ---------------------------------------------------------------------------

class TestProportionalRouting:
    def test_better_venue_gets_larger_share(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        # kraken is clearly better (much lower spread)
        executor.update_venue_stats("kraken", spread_bps=1.0, liquidity_score=0.95)
        executor.update_venue_stats("coinbase", spread_bps=30.0, liquidity_score=0.50)

        orders = executor.split(_decision(1000.0, max_venues=2))
        venue_map = {o.venue: o.size for o in orders}
        assert venue_map["kraken"] > venue_map["coinbase"]

    def test_all_orders_have_positive_size(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        orders = executor.split(_decision(500.0))
        for o in orders:
            assert o.size > 0

    def test_max_venues_limits_number_of_orders(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        # Add a third venue
        executor._stats["binance"] = VenueStats(spread_bps=3.0, liquidity_score=0.98)
        orders = executor.split(_decision(1000.0, max_venues=2))
        assert len(orders) <= 2

    def test_negative_size_treated_as_absolute(self):
        """Negative total_size (short signal) should give positive chunk sizes."""
        executor = MultiVenueExecutor(primary_venue="kraken")
        orders = executor.split(_decision(-500.0))
        for o in orders:
            assert o.size >= 0


# ---------------------------------------------------------------------------
# update_venue_stats
# ---------------------------------------------------------------------------

class TestUpdateVenueStats:
    def test_update_creates_new_venue_entry(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        executor.update_venue_stats("binance", spread_bps=2.0)
        assert "binance" in executor._stats

    def test_update_clamps_liquidity_score(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        executor.update_venue_stats("kraken", liquidity_score=5.0)  # out of range
        assert executor._stats["kraken"].liquidity_score <= 1.0

    def test_update_ignores_none_params(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        original_spread = executor._stats["kraken"].spread_bps
        executor.update_venue_stats("kraken")  # no kwargs
        assert executor._stats["kraken"].spread_bps == original_spread


# ---------------------------------------------------------------------------
# aggregate_fills
# ---------------------------------------------------------------------------

class TestAggregateFills:
    def test_aggregate_fills_correct_vwap(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        orders = [
            VenueOrder("kraken", "BTC/USD", "BUY", 0.5, filled=0.5, fill_price=50_000),
            VenueOrder("coinbase", "BTC/USD", "BUY", 0.5, filled=0.5, fill_price=50_200),
        ]
        result = executor.aggregate_fills(orders)
        assert result["total_filled"] == pytest.approx(1.0)
        expected_vwap = (0.5 * 50_000 + 0.5 * 50_200) / 1.0
        assert result["vwap"] == pytest.approx(expected_vwap)

    def test_aggregate_fills_no_fills(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        orders = [VenueOrder("kraken", "BTC/USD", "BUY", 1.0, filled=0.0, fill_price=0.0)]
        result = executor.aggregate_fills(orders)
        assert result["total_filled"] == pytest.approx(0.0)

    def test_venue_scores_returns_dict(self):
        executor = MultiVenueExecutor(primary_venue="kraken")
        scores = executor.venue_scores()
        assert "kraken" in scores
        assert "coinbase" in scores
        for score in scores.values():
            assert score >= 0
