"""
Grover-driven arbitrage search tests.

Phase D4 of the quantum overhaul.
"""

from __future__ import annotations

import pytest

from strategies.quantum_arb_search import (
    ArbCandidate,
    ArbSignal,
    QuantumArbSearcher,
    VenuePrice,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _build_clean_market() -> dict:
    """All venues quote BTC at the same price (no arb)."""
    return {
        "kraken": {
            "BTC/USD": VenuePrice("kraken", "BTC/USD", bid=50000.0, ask=50001.0, fee_bps=5),
            "ETH/USD": VenuePrice("kraken", "ETH/USD", bid=3000.0, ask=3001.0, fee_bps=5),
        },
        "coinbase": {
            "BTC/USD": VenuePrice("coinbase", "BTC/USD", bid=50000.0, ask=50001.0, fee_bps=5),
            "ETH/USD": VenuePrice("coinbase", "ETH/USD", bid=3000.0, ask=3001.0, fee_bps=5),
        },
    }


def _build_arb_market(edge_bps: float = 100.0) -> dict:
    """One known arb opportunity: BTC cheap on kraken, expensive on coinbase."""
    base = 50000.0
    edge_usd = base * edge_bps / 10000.0
    return {
        "kraken": {
            "BTC/USD": VenuePrice("kraken", "BTC/USD", bid=base - 1, ask=base, fee_bps=5),
        },
        "coinbase": {
            "BTC/USD": VenuePrice("coinbase", "BTC/USD", bid=base + edge_usd, ask=base + edge_usd + 1, fee_bps=5),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Core arb detection
# ═════════════════════════════════════════════════════════════════════════════


class TestGroverArbDetection:

    def test_finds_known_arb_opportunity(self):
        """Big spread between venues should produce a signal."""
        prices = _build_arb_market(edge_bps=100.0)  # 100bps gap
        searcher = QuantumArbSearcher(threshold_multiplier=1.2, min_edge_bps=3.0)
        signals = searcher.find_opportunities(prices)
        assert len(signals) >= 1
        sig = signals[0]
        assert sig.symbol == "BTC/USD"
        assert sig.venue_buy == "kraken"
        assert sig.venue_sell == "coinbase"
        assert sig.expected_edge_bps > 50.0

    def test_no_signals_in_clean_market(self):
        """Tight market with no arb should return zero signals."""
        prices = _build_clean_market()
        searcher = QuantumArbSearcher(threshold_multiplier=1.5, min_edge_bps=5.0)
        signals = searcher.find_opportunities(prices)
        assert len(signals) == 0

    def test_signal_metadata_complete(self):
        """ArbSignal should include all expected metadata."""
        prices = _build_arb_market(edge_bps=80.0)
        searcher = QuantumArbSearcher(threshold_multiplier=1.2, min_edge_bps=3.0)
        signals = searcher.find_opportunities(prices)
        assert len(signals) >= 1
        sig = signals[0]
        assert sig.metadata["method"] == "grover_arb_search"
        assert "buy_price" in sig.metadata
        assert "sell_price" in sig.metadata
        assert "fee_total_bps" in sig.metadata
        assert "gross_edge_bps" in sig.metadata
        assert sig.metadata["gross_edge_bps"] > sig.metadata["fee_total_bps"]


class TestThresholdBehavior:

    def test_threshold_multiplier_filters_marginal_arbs(self):
        """High threshold_multiplier should filter out marginal opportunities."""
        prices = _build_arb_market(edge_bps=15.0)  # tight 15bps
        # 5+5 = 10 fee bps, gross edge ~15bps. With multiplier 1.5, requires 15 bps net.
        searcher = QuantumArbSearcher(threshold_multiplier=1.5, min_edge_bps=3.0)
        signals = searcher.find_opportunities(prices)
        # Should reject — gross 15 < 10 * 1.5 = 15 (just at boundary)
        # Use 1.6 multiplier to clearly reject
        searcher2 = QuantumArbSearcher(threshold_multiplier=1.6, min_edge_bps=3.0)
        signals2 = searcher2.find_opportunities(prices)
        assert len(signals2) == 0

    def test_min_edge_bps_filters_tiny_arbs(self):
        """Below min_edge_bps should be rejected."""
        prices = _build_arb_market(edge_bps=12.0)  # 2bps net edge after fees
        searcher = QuantumArbSearcher(threshold_multiplier=1.0, min_edge_bps=10.0)
        signals = searcher.find_opportunities(prices)
        assert len(signals) == 0


class TestSnapshot:

    def test_snapshot_tracks_runs(self):
        searcher = QuantumArbSearcher()
        prices = _build_clean_market()
        searcher.find_opportunities(prices)
        searcher.find_opportunities(prices)
        snap = searcher.snapshot()
        assert snap["n_runs"] == 2
        assert snap["n_signals_emitted"] == 0

    def test_snapshot_tracks_emitted_signals(self):
        searcher = QuantumArbSearcher(threshold_multiplier=1.2, min_edge_bps=3.0)
        prices = _build_arb_market(edge_bps=100.0)
        searcher.find_opportunities(prices)
        snap = searcher.snapshot()
        assert snap["n_signals_emitted"] >= 1


class TestEdgeCases:

    def test_empty_market(self):
        searcher = QuantumArbSearcher()
        signals = searcher.find_opportunities({})
        assert signals == []

    def test_single_venue(self):
        searcher = QuantumArbSearcher()
        prices = {"kraken": {"BTC/USD": VenuePrice("kraken", "BTC/USD", bid=50000, ask=50001)}}
        signals = searcher.find_opportunities(prices)
        # Need at least 2 venues for arb
        assert signals == []

    def test_no_overlapping_symbols(self):
        searcher = QuantumArbSearcher()
        prices = {
            "kraken": {"BTC/USD": VenuePrice("kraken", "BTC/USD", bid=50000, ask=50001)},
            "coinbase": {"ETH/USD": VenuePrice("coinbase", "ETH/USD", bid=3000, ask=3001)},
        }
        signals = searcher.find_opportunities(prices)
        # No symbol on both venues
        assert signals == []
