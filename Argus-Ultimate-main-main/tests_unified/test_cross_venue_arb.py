"""
Tests for CrossVenueArbPipeline and associated functions.

Run with:
    pytest tests_unified/test_cross_venue_arb.py -v
"""
from __future__ import annotations

import time

import pytest

from alpha.arbitrage.cross_venue_arb_pipeline import (
    ArbOpportunity,
    CrossVenueArbPipeline,
    FundingRateSignal,
    TriangularArbOpportunity,
    VenueBook,
    scan_funding_arb,
    spot_perp_arb,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_book(
    venue: str,
    symbol: str = "BTC/USDT",
    bid: float = 50_000.0,
    ask: float = 50_010.0,
    bid_size: float = 1.0,
    ask_size: float = 1.0,
    fee_maker_bps: float = -2.0,
    fee_taker_bps: float = 10.0,
    age_ns: int = 0,
) -> VenueBook:
    """Factory for VenueBook test fixtures."""
    return VenueBook(
        venue=venue,
        symbol=symbol,
        best_bid=bid,
        best_ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        timestamp_ns=time.time_ns() - age_ns,
        fee_maker_bps=fee_maker_bps,
        fee_taker_bps=fee_taker_bps,
    )


def _make_pipeline(**kwargs) -> CrossVenueArbPipeline:
    return CrossVenueArbPipeline(position_limit_usd=10_000.0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: VenueBook properties
# ─────────────────────────────────────────────────────────────────────────────

class TestVenueBook:

    def test_mid_price(self):
        book = _make_book("binance", bid=50_000.0, ask=50_020.0)
        assert book.mid == pytest.approx(50_010.0)

    def test_market_spread_bps(self):
        book = _make_book("binance", bid=50_000.0, ask=50_020.0)
        # (50020 - 50000) / 50010 * 10000 ≈ 4 bps
        assert 0 < book.market_spread_bps < 10


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: scan_arbitrage — detects a clear opportunity
# ─────────────────────────────────────────────────────────────────────────────

class TestScanArbitrage:

    def test_detects_clear_opportunity(self):
        """Large spread between venues should produce an arb opportunity."""
        pipeline = _make_pipeline()
        # Binance ask=50_000, OKX bid=50_100 → ~20bps gross, 5bps each taker → ~10bps net
        pipeline.update_venue(_make_book("binance", bid=49_980.0, ask=50_000.0,
                                         fee_taker_bps=5.0))
        pipeline.update_venue(_make_book("okx",     bid=50_100.0, ask=50_120.0,
                                         fee_taker_bps=5.0))
        opps = pipeline.scan_arbitrage(min_net_spread_bps=0.5)
        assert len(opps) > 0

    def test_opportunity_fields_populated(self):
        """Returned ArbOpportunity must have all required fields."""
        pipeline = _make_pipeline()
        pipeline.update_venue(_make_book("bybit", bid=49_980.0, ask=50_000.0,
                                          fee_taker_bps=10.0))
        pipeline.update_venue(_make_book("kraken", bid=50_200.0, ask=50_220.0,
                                          fee_taker_bps=10.0))
        opps = pipeline.scan_arbitrage(min_net_spread_bps=0.0)
        assert len(opps) > 0
        opp = opps[0]
        assert isinstance(opp, ArbOpportunity)
        assert opp.buy_venue in ("bybit", "kraken")
        assert opp.sell_venue in ("bybit", "kraken")
        assert opp.buy_venue != opp.sell_venue
        assert opp.gross_spread_bps > 0
        assert opp.max_size_usd > 0
        assert 0.0 < opp.fill_probability <= 1.0

    def test_no_opportunity_when_spread_negative(self):
        """No arb when buy price ≥ sell price."""
        pipeline = _make_pipeline()
        # Both venues have same mid, ask > bid everywhere
        pipeline.update_venue(_make_book("binance", bid=49_990.0, ask=50_010.0,
                                          fee_taker_bps=20.0))
        pipeline.update_venue(_make_book("okx",     bid=49_990.0, ask=50_010.0,
                                          fee_taker_bps=20.0))
        opps = pipeline.scan_arbitrage(min_net_spread_bps=5.0)
        assert len(opps) == 0

    def test_stale_books_excluded(self):
        """Books older than max_age_ns should not generate signals."""
        pipeline = _make_pipeline(max_age_ns=1_000)  # 1 microsecond age limit
        # Book created 1 second ago → stale
        pipeline.update_venue(_make_book("binance", bid=49_980.0, ask=50_000.0,
                                          age_ns=1_000_000_000))
        pipeline.update_venue(_make_book("okx",     bid=50_200.0, ask=50_220.0,
                                          age_ns=1_000_000_000))
        opps = pipeline.scan_arbitrage()
        assert len(opps) == 0

    def test_sorted_by_expected_value(self):
        """Opportunities should be sorted descending by expected_value_usd."""
        pipeline = _make_pipeline()
        # Three venues: varied spread sizes
        pipeline.update_venue(_make_book("v1", bid=49_000.0, ask=50_000.0,
                                          bid_size=0.1, ask_size=0.1,
                                          fee_taker_bps=10.0))
        pipeline.update_venue(_make_book("v2", bid=51_000.0, ask=51_100.0,
                                          bid_size=2.0, ask_size=2.0,
                                          fee_taker_bps=10.0))
        pipeline.update_venue(_make_book("v3", bid=52_000.0, ask=52_100.0,
                                          bid_size=5.0, ask_size=5.0,
                                          fee_taker_bps=10.0))
        opps = pipeline.scan_arbitrage(min_net_spread_bps=0.0)
        evs = [o.expected_value_usd for o in opps]
        assert evs == sorted(evs, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: spot_perp_arb
# ─────────────────────────────────────────────────────────────────────────────

class TestSpotPerpArb:

    def test_positive_funding_yields_positive_ev(self):
        """Positive funding rate → carry trade is profitable (before fees)."""
        ev = spot_perp_arb(
            spot_price=50_000.0,
            perp_price=50_000.0,
            funding_rate_8h=0.001,   # 10bps per 8h → +ve longs pay shorts
            fee_bps=5.0,
        )
        assert ev > 0

    def test_zero_price_returns_zero(self):
        """Zero or negative prices should return 0."""
        assert spot_perp_arb(0.0, 50_000.0, 0.001, 5.0) == 0.0
        assert spot_perp_arb(50_000.0, 0.0, 0.001, 5.0) == 0.0

    def test_high_fees_produce_negative_ev(self):
        """If fees exceed funding, EV should be negative."""
        ev = spot_perp_arb(
            spot_price=50_000.0,
            perp_price=50_000.0,
            funding_rate_8h=0.0001,  # 1bp funding
            fee_bps=50.0,            # 50bp round-trip
        )
        assert ev < 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: scan_funding_arb
# ─────────────────────────────────────────────────────────────────────────────

class TestScanFundingArb:

    def _venues(self):
        return [
            {
                "venue": "binance",
                "symbol": "BTC/USDT",
                "spot_price": 50_000.0,
                "perp_price": 50_050.0,
                "funding_rate_8h": 0.001,   # 10bps/8h → ann ≈ 136%
                "fee_bps": 10.0,
            },
            {
                "venue": "okx",
                "symbol": "ETH/USDT",
                "spot_price": 3_500.0,
                "perp_price": 3_490.0,
                "funding_rate_8h": -0.0005,  # negative → shorts pay longs
                "fee_bps": 10.0,
            },
            {
                "venue": "bybit",
                "symbol": "SOL/USDT",
                "spot_price": 150.0,
                "perp_price": 150.0,
                "funding_rate_8h": 0.00001,  # tiny → below threshold
                "fee_bps": 10.0,
            },
        ]

    def test_returns_list_of_signals(self):
        signals = scan_funding_arb(self._venues(), min_annualised_yield=0.05)
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, FundingRateSignal)

    def test_sorted_by_absolute_yield(self):
        """Results should be sorted descending by |annualised_rate|."""
        signals = scan_funding_arb(self._venues(), min_annualised_yield=0.0)
        ann_rates = [abs(s.annualised_rate) for s in signals]
        assert ann_rates == sorted(ann_rates, reverse=True)

    def test_filters_below_threshold(self):
        """Signals below min_annualised_yield should be excluded."""
        signals = scan_funding_arb(self._venues(), min_annualised_yield=1.0)
        # Only the very high-yield BTC entry should survive
        for s in signals:
            assert abs(s.annualised_rate) >= 1.0

    def test_direction_field(self):
        """Direction should reflect sign of funding rate."""
        signals = scan_funding_arb(self._venues(), min_annualised_yield=0.0)
        for s in signals:
            if s.rate_8h >= 0:
                assert s.direction == "LONG_SPOT_SHORT_PERP"
            else:
                assert s.direction == "SHORT_SPOT_LONG_PERP"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: historical_decay_rate
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoricalDecayRate:

    def test_returns_default_with_no_history(self):
        """No spread history → return 1/decay_halflife_s."""
        pipeline = _make_pipeline(decay_halflife_s=0.5)
        rate = pipeline.historical_decay_rate("BTC/USDT", "binance_okx")
        assert rate == pytest.approx(2.0, rel=0.01)   # 1/0.5

    def test_decreasing_spread_yields_positive_decay(self):
        """If spreads are decreasing over time, decay rate should be > 0."""
        pipeline = _make_pipeline()
        symbol = "BTC/USDT"
        pair = "binance_okx"
        key = (symbol, pair)
        now_ns = time.time_ns()
        # Insert 10 spread readings that decrease over ~1s
        for i in range(10):
            spread = 20.0 - i * 1.5  # decreasing
            ts = now_ns + i * 100_000_000  # 100ms apart
            pipeline._spread_history[key].append((ts, max(spread, 0.1)))
        rate = pipeline.historical_decay_rate(symbol, pair)
        assert rate > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: signal_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalSummary:

    def test_empty_summary(self):
        """Empty pipeline returns zeroed summary."""
        pipeline = _make_pipeline()
        summary = pipeline.signal_summary()
        assert summary["total_opportunity_count"] == 0
        assert summary["avg_spread_bps"] == 0.0

    def test_summary_after_scan(self):
        """After a scan, summary reflects accumulated opportunities."""
        pipeline = _make_pipeline()
        pipeline.update_venue(_make_book("binance", bid=49_980.0, ask=50_000.0,
                                          fee_taker_bps=10.0))
        pipeline.update_venue(_make_book("okx",     bid=50_200.0, ask=50_220.0,
                                          fee_taker_bps=10.0))
        pipeline.scan_arbitrage(min_net_spread_bps=0.0)
        summary = pipeline.signal_summary()
        assert summary["total_opportunity_count"] > 0
        assert "top_opportunities" in summary
        assert "daily_pnl_estimate_usd" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: triangular arb scan
# ─────────────────────────────────────────────────────────────────────────────

class TestScanTriangular:

    def test_no_tri_arb_with_fair_prices(self):
        """With consistent prices across all pairs, no triangular arb exists."""
        pipeline = _make_pipeline()
        now_ns = time.time_ns()
        # BTC=50000, ETH=3000 → ETH/BTC = 0.06
        # Consistent pricing: no rounding errors
        pipeline.update_venue(VenueBook(
            "binance", "BTC/USDT", 49_990.0, 50_010.0, 1.0, 1.0,
            now_ns, -2.0, 10.0
        ))
        pipeline.update_venue(VenueBook(
            "binance", "ETH/USDT", 2_990.0, 3_010.0, 10.0, 10.0,
            now_ns, -2.0, 10.0
        ))
        pipeline.update_venue(VenueBook(
            "binance", "ETH/BTC", 0.0599, 0.0601, 100.0, 100.0,
            now_ns, -2.0, 10.0
        ))
        symbols = ["BTC/USDT", "ETH/USDT", "ETH/BTC"]
        results = pipeline.scan_triangular(symbols, min_net_profit_bps=100.0)
        # At very high threshold, no opportunity
        assert len(results) == 0

    def test_triangular_returns_list(self):
        """scan_triangular always returns a list."""
        pipeline = _make_pipeline()
        result = pipeline.scan_triangular([], min_net_profit_bps=0.0)
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: maker_net_spread_bps is better than taker-taker
# ─────────────────────────────────────────────────────────────────────────────

class TestMakerTakerComparison:

    def test_maker_net_spread_higher_than_taker_taker(self):
        """
        When maker fee is a rebate (negative bps), maker_net_spread_bps
        should exceed net_spread_bps (both taker).
        """
        pipeline = _make_pipeline()
        pipeline.update_venue(_make_book(
            "binance", bid=49_980.0, ask=50_000.0,
            fee_maker_bps=-2.0,   # rebate
            fee_taker_bps=10.0,
        ))
        pipeline.update_venue(_make_book(
            "okx",     bid=50_100.0, ask=50_120.0,
            fee_maker_bps=0.0,
            fee_taker_bps=8.0,
        ))
        opps = pipeline.scan_arbitrage(min_net_spread_bps=0.0)
        assert len(opps) > 0
        # Find the opportunity where binance is buy venue
        binance_buy = [o for o in opps if o.buy_venue == "binance"]
        if binance_buy:
            opp = binance_buy[0]
            # maker_net >= net_spread since maker fee is cheaper (rebate)
            assert opp.maker_net_spread_bps >= opp.net_spread_bps
