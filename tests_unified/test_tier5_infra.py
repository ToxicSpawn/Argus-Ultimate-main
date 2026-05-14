"""
Tier 5 Infrastructure Tests
============================
Tests for:
    - infra/fpga_interface.py
    - execution/position_netting.py
    - execution/strategy_circuit_breaker.py
    - infra/colocation_config_generator.py
    - infra/grafana_dashboard.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infra.fpga_interface import FPGAInterface, FPGASignal, SoftwareFPGASim
from execution.position_netting import CrossVenuePositionNetter
from execution.strategy_circuit_breaker import (
    BreakerConfig,
    BreakerState,
    StrategyBreakerPanel,
    StrategyCircuitBreaker,
)
from infra.colocation_config_generator import ColocationConfigGenerator
from infra.grafana_dashboard import GrafanaDashboardBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_ns() -> int:
    return time.time_ns()


# ===========================================================================
# FPGA Interface Tests
# ===========================================================================

class TestFPGAInterface:

    def test_fpga_interface_sim_mode(self):
        """enabled=False → software sim, no connection attempt, no crash."""
        iface = FPGAInterface(host="localhost", port=9999, enabled=False)
        assert not iface.is_hardware_mode()
        # connect should succeed in sim mode without network
        result = asyncio.get_event_loop().run_until_complete(iface.connect())
        assert result is True
        assert not iface.is_hardware_mode()

    def test_fpga_signal_format(self):
        """send_book_delta → recv_signal returns FPGASignal with correct fields."""
        iface = FPGAInterface(enabled=False)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(iface.connect())

        # Seed the order book with enough data to generate a signal
        for price in range(100, 105):
            loop.run_until_complete(
                iface.send_book_delta("BTC/USD", "bid", float(price), 1.0, _ts_ns())
            )
        for price in range(105, 110):
            loop.run_until_complete(
                iface.send_book_delta("BTC/USD", "ask", float(price), 1.0, _ts_ns())
            )

        signal = loop.run_until_complete(iface.recv_signal())
        assert signal is not None, "Expected a signal after seeding order book"
        assert isinstance(signal, FPGASignal)
        assert signal.signal_type in ("direction", "quote", "halt")
        assert isinstance(signal.symbol, str) and signal.symbol != ""
        assert -1.0 <= signal.value <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert isinstance(signal.latency_ns, int)
        assert signal.latency_ns >= 0

    def test_fpga_latency_stats_after_signals(self):
        """get_latency_stats returns populated dict after processing deltas."""
        iface = FPGAInterface(enabled=False)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(iface.connect())

        for p in range(200, 205):
            loop.run_until_complete(
                iface.send_book_delta("ETH/USD", "bid", float(p), 0.5, _ts_ns())
            )
        for p in range(205, 210):
            loop.run_until_complete(
                iface.send_book_delta("ETH/USD", "ask", float(p), 0.5, _ts_ns())
            )

        stats = iface.get_latency_stats()
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert "count" in stats


# ===========================================================================
# Position Netting Tests
# ===========================================================================

class TestPositionNetting:

    def test_position_netting_single_venue(self):
        """One venue: net position equals the raw position."""
        netter = CrossVenuePositionNetter()
        netter.update_position("kraken", "BTC/USD", 1.5, _ts_ns())
        assert netter.get_net_position("BTC/USD") == pytest.approx(1.5)

    def test_position_netting_cross_venue(self):
        """Long on Kraken + short on Coinbase same symbol → net = 0."""
        netter = CrossVenuePositionNetter()
        netter.update_position("kraken",   "BTC/USD",  1.0, _ts_ns())
        netter.update_position("coinbase", "BTC/USD", -1.0, _ts_ns())
        assert netter.get_net_position("BTC/USD") == pytest.approx(0.0)

    def test_position_netting_correlation(self):
        """BTC/USD + BTC/USDT detected as correlated (pre-seeded correlation)."""
        netter = CrossVenuePositionNetter(correlation_threshold=0.85)
        netter.update_position("kraken",  "BTC/USD",  1.0, _ts_ns())
        netter.update_position("binance", "BTC/USDT", 1.0, _ts_ns())

        # Both positions should be flagged as potential double-count
        assert netter.is_double_counted("kraken", "BTC/USD", "binance", "BTC/USDT")

        # Netted exposure should merge them into one group (net = 2.0, not 1.0+1.0 separately)
        netted = netter.get_netted_exposure()
        # The two correlated symbols collapse to one root key; total = 2.0
        total = sum(netted.values())
        assert total == pytest.approx(2.0)

    def test_position_netting_session_summary(self):
        """get_session_summary returns expected structure."""
        netter = CrossVenuePositionNetter()
        netter.update_position("kraken",  "BTC/USD", 0.5, _ts_ns())
        netter.update_position("binance", "ETH/USD", 2.0, _ts_ns())
        summary = netter.get_session_summary()
        assert "positions" in summary
        assert "netted_exposures" in summary
        assert "double_counted_pairs" in summary
        assert summary["total_positions"] == 2

    def test_hedge_recommendation_flat(self):
        """Flat position → no hedge recommendation."""
        netter = CrossVenuePositionNetter()
        netter.update_position("kraken",   "BTC/USD",  1.0, _ts_ns())
        netter.update_position("coinbase", "BTC/USD", -1.0, _ts_ns())
        rec = netter.get_hedge_recommendation("BTC/USD")
        assert rec is None


# ===========================================================================
# Circuit Breaker Tests
# ===========================================================================

class TestCircuitBreaker:

    def test_circuit_breaker_drawdown_trip(self):
        """Exceed max_drawdown → state = OPEN."""
        cfg = BreakerConfig(max_drawdown_pct=5.0, peak_pnl_seed=1000.0)
        breaker = StrategyCircuitBreaker("test_strat", cfg)
        # Record a large loss: session PnL drops from 0 to -60,
        # drawdown = 60/1000 * 100 = 6% > 5%
        breaker.record_trade(-60.0, _ts_ns())
        state = breaker.check()
        assert state == BreakerState.OPEN
        assert "drawdown" in breaker._trip_reason.lower()

    def test_circuit_breaker_consecutive_losses(self):
        """11 consecutive losses → OPEN (drawdown guard disabled by huge peak)."""
        # Set max_drawdown_pct very high so it never fires; peak_pnl_seed huge so
        # even many tiny losses can't trigger drawdown.
        cfg = BreakerConfig(
            max_consecutive_losses=10,
            max_drawdown_pct=99.0,   # effectively disabled
            peak_pnl_seed=1_000_000.0,
        )
        breaker = StrategyCircuitBreaker("test_strat2", cfg)
        for i in range(11):
            breaker.record_trade(-0.01, _ts_ns())  # tiny loss, drawdown negligible
        state = breaker.check()
        assert state == BreakerState.OPEN
        assert "consecutive" in breaker._trip_reason.lower()

    def test_circuit_breaker_reset(self):
        """OPEN → reset() → CLOSED."""
        cfg = BreakerConfig(max_drawdown_pct=5.0, peak_pnl_seed=1000.0)
        breaker = StrategyCircuitBreaker("test_reset", cfg)
        breaker.record_trade(-60.0, _ts_ns())
        breaker.check()
        assert breaker._state == BreakerState.OPEN
        breaker.reset()
        assert breaker._state == BreakerState.CLOSED

    def test_circuit_breaker_status_keys(self):
        """get_status returns all expected keys."""
        cfg = BreakerConfig()
        breaker = StrategyCircuitBreaker("test_status", cfg)
        status = breaker.get_status()
        required_keys = [
            "strategy", "state", "trip_reason", "session_pnl",
            "drawdown_pct", "consecutive_losses", "order_rate_per_min",
        ]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_breaker_panel_trip_all(self):
        """trip_all → all registered strategies are OPEN."""
        panel = StrategyBreakerPanel()
        panel.trip_all("unit test emergency")
        states = panel.check_all()
        for name, state in states.items():
            assert state == BreakerState.OPEN, f"{name} should be OPEN, got {state}"
        assert panel.any_open()

    def test_breaker_panel_default_strategies(self):
        """Panel pre-registers the 4 default Argus strategies."""
        panel = StrategyBreakerPanel()
        for name in ("market_maker", "cross_venue_arb", "hft_scalping", "void_breaker"):
            assert name in panel, f"Strategy '{name}' not registered"

    def test_breaker_panel_register_and_get(self):
        """register + get returns a working breaker."""
        panel = StrategyBreakerPanel()
        cfg = BreakerConfig(max_drawdown_pct=2.0)
        panel.register("new_strat", cfg)
        breaker = panel.get("new_strat")
        assert isinstance(breaker, StrategyCircuitBreaker)
        assert breaker.config.max_drawdown_pct == 2.0


# ===========================================================================
# Colocation Config Generator Tests
# ===========================================================================

class TestColocationConfigGenerator:

    def test_colocation_config_yaml(self):
        """generate() produces valid YAML with exchange entries."""
        import tempfile
        gen = ColocationConfigGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "colocation.yaml")
            config = asyncio.get_event_loop().run_until_complete(
                gen.generate(exchanges=["binance", "bybit"], output_path=out_path)
            )
            assert len(config.exchanges) == 2
            assert os.path.exists(out_path)
            with open(out_path) as f:
                yaml_text = f.read()
            assert "binance" in yaml_text
            assert "bybit" in yaml_text
            assert "recommended_dc" in yaml_text
            assert "tier" in yaml_text

    def test_colocation_tier_assignment(self):
        """Tier is correctly assigned based on RTT."""
        from infra.colocation_config_generator import _classify_tier
        assert _classify_tier(0.5) == "COLOCATED"
        assert _classify_tier(3.0) == "PROXIMATE"
        assert _classify_tier(12.0) == "REMOTE"
        assert _classify_tier(50.0) == "RETAIL"

    def test_colocation_recommendations_not_empty(self):
        """get_recommendations returns a non-empty list."""
        gen = ColocationConfigGenerator()
        recs = gen.get_recommendations("binance")
        assert isinstance(recs, list)
        assert len(recs) > 0


# ===========================================================================
# Grafana Dashboard Tests
# ===========================================================================

class TestGrafanaDashboard:

    def test_grafana_dashboard_panels(self):
        """Built JSON has >= 8 panels."""
        builder = GrafanaDashboardBuilder()
        dashboard = builder.build()
        assert "panels" in dashboard
        assert len(dashboard["panels"]) >= 8

    def test_grafana_dashboard_structure(self):
        """Dashboard has required top-level keys."""
        builder = GrafanaDashboardBuilder()
        dashboard = builder.build()
        required_keys = ["uid", "title", "panels", "templating", "time", "refresh"]
        for key in required_keys:
            assert key in dashboard, f"Missing key: {key}"

    def test_grafana_dashboard_all_panel_types(self):
        """Dashboard contains timeseries, gauge, stat, barchart, state-timeline panels."""
        builder = GrafanaDashboardBuilder()
        dashboard = builder.build()
        types_found = {p["type"] for p in dashboard["panels"]}
        assert "timeseries" in types_found
        assert "gauge" in types_found
        assert "stat" in types_found
        assert "barchart" in types_found
        assert "state-timeline" in types_found

    def test_grafana_dashboard_save(self, tmp_path):
        """save() writes valid JSON to the given path."""
        builder = GrafanaDashboardBuilder()
        out_path = str(tmp_path / "test_dashboard.json")
        builder.save(out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["uid"] == "argus-hft-v1"
        assert len(loaded["panels"]) >= 8
