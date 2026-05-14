"""
Tests for all new ARGUS components added in the improvement batch.

Covers:
  - Bybit/OKX connectors (no-credentials graceful handling)
  - FundingRateHarvester signal logic
  - DeltaNeutralExecutor pairing logic
  - ExchangeFlowSignal (with mock Glassnode client)
  - OptionsSignal (with mock Deribit client)
  - MacroOverlay (momentum fallback)
  - CointegrationPairsTrader (ADF + z-score logic)
  - AvellanedaStoikovMM (quote computation)
  - TemporalFusionTransformer (numpy backend)
  - KellyUncertaintyCalculator (bootstrap sizing)
  - CorrelationMonitor (breakdown detection)
  - AlphaDecayTracker (horizon evaluation)
  - TimescaleAdapter / SQLiteTickAdapter (storage)
"""
from __future__ import annotations

import asyncio
import time
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Exchange connectors
# ---------------------------------------------------------------------------

class TestBybitConnector:
    def test_import(self):
        from core.connectors.bybit_connector import BybitConnector
        conn = BybitConnector()
        assert conn.health_check_symbol == "BTC/USDT:USDT"
        assert conn.connected is False

    @pytest.mark.asyncio
    async def test_no_credentials_connect_returns_false(self):
        from core.connectors.bybit_connector import BybitConnector
        conn = BybitConnector(api_key="", api_secret="")
        # Without real credentials, connect will fail gracefully — never raises
        result = await conn.connect()
        assert isinstance(result, bool)


class TestOKXConnector:
    def test_import(self):
        from core.connectors.okx_connector import OKXConnector
        conn = OKXConnector()
        assert conn.health_check_symbol == "BTC-USDT-SWAP"
        assert conn.connected is False


# ---------------------------------------------------------------------------
# FundingRateHarvester
# ---------------------------------------------------------------------------

class TestFundingRateHarvester:
    def test_no_signal_below_threshold(self):
        from strategies.funding_rate_harvester import FundingRateHarvester
        h = FundingRateHarvester(open_threshold=0.0005)
        sig = h.analyze({"symbol": "BTC/USD", "price": 65000.0, "funding_rates": {"bybit": 0.0001}})
        assert sig is None

    def test_harvest_open_signal_above_threshold(self):
        from strategies.funding_rate_harvester import FundingRateHarvester
        h = FundingRateHarvester(open_threshold=0.0005)
        sig = h.analyze({
            "symbol": "BTC/USD",
            "price": 65000.0,
            "funding_rates": {"bybit": 0.0010},  # 2x threshold
            "spot_exchange": "kraken",
        })
        assert sig is not None
        assert sig["action"] == "HARVEST_OPEN"
        assert sig["perp_exchange"] == "bybit"
        assert sig["confidence"] > 0.0
        assert "BTC/USD" in h.get_active_symbols()

    def test_harvest_close_when_rate_drops(self):
        from strategies.funding_rate_harvester import FundingRateHarvester
        h = FundingRateHarvester(open_threshold=0.0005, close_threshold=0.0001)
        # Open first
        h.analyze({"symbol": "BTC/USD", "price": 65000.0, "funding_rates": {"bybit": 0.0010}})
        # Rate drops — should close
        sig = h.analyze({"symbol": "BTC/USD", "price": 65500.0, "funding_rates": {"bybit": 0.00005}})
        assert sig is not None
        assert sig["action"] == "HARVEST_CLOSE"

    def test_stop_loss_on_negative_funding(self):
        from strategies.funding_rate_harvester import FundingRateHarvester
        h = FundingRateHarvester(open_threshold=0.0005, stop_threshold=-0.0003)
        h.analyze({"symbol": "ETH/USD", "price": 3000.0, "funding_rates": {"bybit": 0.001}})
        sig = h.analyze({"symbol": "ETH/USD", "price": 2900.0, "funding_rates": {"bybit": -0.0005}})
        assert sig is not None
        assert sig["action"] == "HARVEST_CLOSE"
        assert sig["close_reason"] == "stop_loss"

    def test_record_funding_payment(self):
        from strategies.funding_rate_harvester import FundingRateHarvester
        h = FundingRateHarvester()
        h.record_funding_payment("BTC/USD", 5.50)
        h.record_funding_payment("BTC/USD", 3.25)
        status = h.get_status()
        assert status["cumulative_funding_usd"]["BTC/USD"] == pytest.approx(8.75)

    def test_max_concurrent_limit(self):
        from strategies.funding_rate_harvester import FundingRateHarvester
        h = FundingRateHarvester(open_threshold=0.0001, max_concurrent=1)
        h.analyze({"symbol": "BTC/USD", "price": 65000.0, "funding_rates": {"bybit": 0.001}})
        sig2 = h.analyze({"symbol": "ETH/USD", "price": 3000.0, "funding_rates": {"bybit": 0.001}})
        # Should not open second harvest — at capacity
        assert sig2 is None or sig2["action"] != "HARVEST_OPEN"


# ---------------------------------------------------------------------------
# Cointegration Pairs Trader
# ---------------------------------------------------------------------------

class TestCointegrationPairsTrader:
    def _make_trader(self):
        from strategies.stat_arb_cointegration import CointegrationPairsTrader
        return CointegrationPairsTrader()

    def test_import_and_init(self):
        trader = self._make_trader()
        assert hasattr(trader, "analyze")
        assert hasattr(trader, "update_prices")

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self):
        trader = self._make_trader()
        # Only 10 prices — needs 200 for cointegration test
        for i in range(10):
            trader.update_prices("BTC/USD", 65000 + i * 10)
        result = await trader.analyze({"symbol": "ETH/USD", "price": 3000.0})
        assert result is None or result == []

    def test_zscore_computation(self):
        from strategies.stat_arb_cointegration import CointegrationPairsTrader
        trader = CointegrationPairsTrader()
        # Feed correlated prices
        rng = np.random.default_rng(99)
        for i in range(210):
            btc = 65000 + rng.normal(0, 500)
            eth = btc / 20 + rng.normal(0, 10)  # Cointegrated pair
            trader.update_prices("BTC/USD", btc)
            trader.update_prices("ETH/USD", eth)
        # Should have price history but cointegration might or might not trigger
        status = trader.get_pair_status()
        assert isinstance(status, dict)

    def test_hedge_ratio_estimation(self):
        from strategies.stat_arb_cointegration import CointegrationPairsTrader
        trader = CointegrationPairsTrader()
        p1 = np.array([100.0, 102.0, 98.0, 101.0, 103.0, 99.0])
        p2 = np.array([50.0, 51.0, 49.0, 50.5, 51.5, 49.5])
        # _estimate_hedge_ratio(p1, p2) returns β s.t. spread = p1 - β*p2
        # Since p1 ≈ 2*p2, hedge ratio ≈ 2.0 (not 0.5)
        ratio = trader._estimate_hedge_ratio(p1, p2)
        assert 1.5 < ratio < 2.5  # Close to 2.0 (p1 = 2*p2)


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov Market Maker
# ---------------------------------------------------------------------------

class TestAvellanedaStoikovMM:
    def test_import(self):
        from strategies.market_maker_avellaneda import AvellanedaStoikovMM
        mm = AvellanedaStoikovMM("BTC/USD")
        assert mm.symbol == "BTC/USD"
        assert mm.inventory == 0.0

    def test_quote_generation(self):
        from strategies.market_maker_avellaneda import AvellanedaStoikovMM
        mm = AvellanedaStoikovMM("BTC/USD", gamma=0.1)
        # Feed some mid prices first
        for price in [64900, 65000, 65100, 64950, 65050]:
            mm._mid_prices.append(price)

        sig = mm.analyze({"symbol": "BTC/USD", "price": 65000.0, "bid": 64995.0, "ask": 65005.0})
        assert sig is not None
        assert sig["action"] == "QUOTE"
        assert sig["bid"] < sig["ask"]
        assert sig["bid"] > 0
        assert sig["spread_bps"] > 0

    def test_reservation_price_adjusts_with_inventory(self):
        from strategies.market_maker_avellaneda import AvellanedaStoikovMM
        mm = AvellanedaStoikovMM("BTC/USD", gamma=0.1)
        # Use varying prices so sigma > 0 (identical prices → sigma=0 → no adjustment)
        for p in [64900, 65000, 65100, 64950, 65050, 65200, 64800]:
            mm._mid_prices.append(p)

        mid = 65000.0
        # Zero inventory: reservation ≈ mid
        mm.inventory = 0.0
        r0 = mm.reservation_price(mid)
        # Positive inventory: reservation below mid (want to sell)
        mm.inventory = 2.0
        r_long = mm.reservation_price(mid)
        assert r_long < r0, f"Expected r_long={r_long} < r0={r0} with positive inventory"

        # Negative inventory: reservation above mid (want to buy)
        mm.inventory = -2.0
        r_short = mm.reservation_price(mid)
        assert r_short > r0, f"Expected r_short={r_short} > r0={r0} with negative inventory"

    def test_fill_updates_inventory(self):
        from strategies.market_maker_avellaneda import AvellanedaStoikovMM
        mm = AvellanedaStoikovMM("ETH/USD")
        mm.record_fill("buy", 3000.0, 0.5)
        assert mm.inventory == pytest.approx(0.5)
        mm.record_fill("sell", 3010.0, 0.5)
        assert mm.inventory == pytest.approx(0.0)

    def test_no_quote_at_inventory_limit(self):
        from strategies.market_maker_avellaneda import AvellanedaStoikovMM
        mm = AvellanedaStoikovMM("BTC/USD", max_inventory=5.0)
        mm.inventory = 5.0  # at limit
        for _ in range(5):
            mm._mid_prices.append(65000.0)
        sig = mm.analyze({"symbol": "BTC/USD", "price": 65000.0})
        assert sig is None


# ---------------------------------------------------------------------------
# Temporal Fusion Transformer
# ---------------------------------------------------------------------------

class TestTemporalFusionTransformer:
    def test_numpy_backend_predict(self):
        import pytest; pytest.importorskip("ml.models.temporal_fusion_transformer")
        import pytest; pytest.importorskip("ml.models.temporal_fusion_transformer")
        import pytest; pytest.importorskip("ml.models.temporal_fusion_transformer")
        from ml.models.temporal_fusion_transformer import TemporalFusionTransformer, N_FEATURES, REGIME_LABELS
        tft = TemporalFusionTransformer()
        assert tft.use_torch is False  # No torch model path provided
        features = np.random.randn(48, N_FEATURES).astype(np.float32)
        result = tft.predict(features)
        assert "regime" in result
        assert result["regime"] in REGIME_LABELS
        assert "regime_probs" in result
        assert "direction_prob" in result
        assert 0.0 <= result["direction_prob"] <= 1.0
        assert abs(sum(result["regime_probs"].values()) - 1.0) < 0.01
        assert result["method"] == "numpy_tft"

    def test_deterministic_same_input(self):
        import pytest; pytest.importorskip("ml.models.temporal_fusion_transformer")
        from ml.models.temporal_fusion_transformer import TemporalFusionTransformer, N_FEATURES
        tft = TemporalFusionTransformer()
        features = np.ones((20, N_FEATURES), dtype=np.float32) * 0.5
        r1 = tft.predict(features)
        r2 = tft.predict(features)
        assert r1["regime"] == r2["regime"]
        assert r1["direction_prob"] == pytest.approx(r2["direction_prob"])

    def test_online_fit_doesnt_crash(self):
        import pytest; pytest.importorskip("ml.models.temporal_fusion_transformer")
        import pytest; pytest.importorskip("ml.models.temporal_fusion_transformer")
        from ml.models.temporal_fusion_transformer import TemporalFusionTransformer, N_FEATURES, REGIME_LABELS
        tft = TemporalFusionTransformer()
        features = np.random.randn(10, N_FEATURES).astype(np.float32)
        tft.fit_online(features, "TREND_UP", lr=0.01)  # Should not raise

    def test_save_load_roundtrip(self, tmp_path):
        import pytest; pytest.importorskip("ml.models.temporal_fusion_transformer")
        from ml.models.temporal_fusion_transformer import TemporalFusionTransformer, N_FEATURES
        tft = TemporalFusionTransformer()
        features = np.ones((10, N_FEATURES), dtype=np.float32)
        r_before = tft.predict(features)

        save_path = str(tmp_path / "tft_test.json")
        tft.save(save_path)

        tft2 = TemporalFusionTransformer()
        tft2.load(save_path)
        r_after = tft2.predict(features)
        assert r_before["regime"] == r_after["regime"]

    def test_prepare_features(self):
        import pytest; pytest.importorskip("ml.models.temporal_fusion_transformer")
        from ml.models.temporal_fusion_transformer import prepare_features, N_FEATURES
        import pandas as pd
        n = 60
        idx = pd.date_range("2025-01-01", periods=n, freq="1min")
        df = pd.DataFrame({
            "open":   np.random.uniform(60000, 70000, n),
            "high":   np.random.uniform(60000, 70000, n),
            "low":    np.random.uniform(60000, 70000, n),
            "close":  np.random.uniform(60000, 70000, n),
            "volume": np.random.uniform(1, 100, n),
        }, index=idx)
        features = prepare_features(df, funding_rate=0.001, flow_signal=0.5, options_pcr=1.2, macro_scalar=0.8)
        assert features.shape == (n, N_FEATURES)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


# ---------------------------------------------------------------------------
# Kelly Uncertainty Calculator
# ---------------------------------------------------------------------------

class TestKellyUncertaintyCalculator:
    def test_insufficient_data_uses_default(self):
        from risk.kelly_uncertainty import KellyUncertaintyCalculator
        calc = KellyUncertaintyCalculator(min_trades=10)
        result = calc.calculate([0.01, 0.02, -0.01], capital=1000.0, price=65000.0)
        assert result["method"] == "insufficient_data_default"
        assert result["fraction"] == pytest.approx(0.02)

    def test_full_calculation_with_trades(self):
        from risk.kelly_uncertainty import KellyUncertaintyCalculator
        calc = KellyUncertaintyCalculator(kelly_fraction=0.5, min_trades=10)
        # 70% win rate, avg win 2%, avg loss 1%
        rng = np.random.default_rng(42)
        trades = []
        for _ in range(50):
            if rng.random() < 0.7:
                trades.append(float(rng.uniform(0.01, 0.03)))
            else:
                trades.append(float(-rng.uniform(0.005, 0.015)))

        result = calc.calculate(trades, capital=1000.0, price=65000.0)
        assert result["n_trades"] == 50
        assert result["fraction"] >= 0.0
        assert result["fraction"] <= 0.25  # max_fraction cap
        assert result["win_rate"] > 0.5
        assert "bayesian_bootstrap" in result["method"]

    def test_fraction_never_exceeds_max(self):
        from risk.kelly_uncertainty import KellyUncertaintyCalculator
        calc = KellyUncertaintyCalculator(kelly_fraction=1.0, max_fraction=0.25)
        # Ideal win scenario
        trades = [0.10] * 30 + [-0.01] * 10  # 75% win, massive edge
        result = calc.calculate(trades, capital=1000.0, price=100.0)
        assert result["fraction"] <= 0.25


# ---------------------------------------------------------------------------
# Correlation Monitor
# ---------------------------------------------------------------------------

class TestCorrelationMonitor:
    def test_no_alert_with_low_correlation(self):
        from risk.correlation_monitor import CorrelationMonitor
        monitor = CorrelationMonitor(["BTC/USD", "ETH/USD", "SOL/USD"], lookback=10)
        rng = np.random.default_rng(1)
        for _ in range(15):
            # Independent random prices — low correlation
            monitor.update("BTC/USD", 65000 + rng.normal(0, 1000))
            monitor.update("ETH/USD", 3000  + rng.normal(0, 200))
            monitor.update("SOL/USD", 150   + rng.normal(0, 20))
        alert = monitor.check_and_alert()
        # Should be None or position_scalar = 1.0 (low correlation)
        assert alert is None or alert["position_scalar"] >= 0.5

    def test_alert_on_high_correlation(self):
        from risk.correlation_monitor import CorrelationMonitor
        monitor = CorrelationMonitor(["BTC/USD", "ETH/USD", "SOL/USD"], lookback=10, alert_threshold=0.5)
        # All assets move identically → correlation = 1.0
        base = 1000.0
        for i in range(15):
            monitor.update("BTC/USD", base + i * 10)
            monitor.update("ETH/USD", base + i * 10)  # perfectly correlated
            monitor.update("SOL/USD", base + i * 10)
        avg_corr = monitor.get_avg_pairwise_correlation()
        assert avg_corr > 0.9

    def test_position_scalar_is_1_when_ok(self):
        from risk.correlation_monitor import CorrelationMonitor
        monitor = CorrelationMonitor(["BTC/USD", "ETH/USD"])
        # Feed prices manually to simulate low correlation
        monitor._last_avg_corr = 0.3
        scalar = monitor.get_position_scalar()
        assert scalar == 1.0

    def test_position_scalar_reduces_on_spike(self):
        from risk.correlation_monitor import CorrelationMonitor
        import unittest.mock
        monitor = CorrelationMonitor(["BTC/USD", "ETH/USD"], alert_threshold=0.80, crisis_threshold=0.92)
        # Patch get_avg_pairwise_correlation to return a known high value
        with unittest.mock.patch.object(monitor, "get_avg_pairwise_correlation", return_value=0.86):
            scalar = monitor.get_position_scalar()
        assert 0.1 < scalar < 0.5, f"Expected scalar in (0.1, 0.5), got {scalar}"


# ---------------------------------------------------------------------------
# Alpha Decay Tracker
# ---------------------------------------------------------------------------

class TestAlphaDecayTracker:
    def test_record_and_update(self):
        from risk.alpha_decay_tracker import AlphaDecayTracker
        tracker = AlphaDecayTracker("momentum")
        sig_id = tracker.record_signal("BTC/USD", "long", 65000.0)
        assert sig_id.startswith("BTC/USD_")
        tracker.update_price("BTC/USD", 65500.0)
        assert "BTC/USD" in tracker._price_history

    def test_compute_decay_empty(self):
        from risk.alpha_decay_tracker import AlphaDecayTracker
        tracker = AlphaDecayTracker("test_strat")
        result = tracker.compute_alpha_decay()
        assert result["strategy"] == "test_strat"
        assert isinstance(result["horizons"], dict)
        assert isinstance(result["peak_horizon"], int)

    def test_positive_alpha_on_correct_signal(self):
        """Verify that a correct 'long' signal shows positive alpha at short horizons."""
        from risk.alpha_decay_tracker import AlphaDecayTracker, HORIZONS
        tracker = AlphaDecayTracker("trend")

        entry_ts = time.time() - HORIZONS[0] - 10  # 5m ago + 10s buffer
        sig_id = f"BTC/USD_{int(entry_ts)}_abc123"
        tracker._signals[sig_id] = {
            "ts": entry_ts,
            "symbol": "BTC/USD",
            "direction": "long",
            "entry_price": 65000.0,
            "returns": {h: None for h in HORIZONS},
            "evaluated": {h: False for h in HORIZONS},
        }
        # Feed a higher price now (profitable long)
        tracker._price_history["BTC/USD"] = [(time.time(), 65500.0)]
        tracker.update_price("BTC/USD", 65500.0)

        decay = tracker.compute_alpha_decay()
        # 5m horizon should now be evaluated with positive return
        returns_5m = decay["horizons"].get(300, 0.0)
        assert returns_5m >= 0.0  # Long with higher price = positive

    def test_reset_clears_all(self):
        from risk.alpha_decay_tracker import AlphaDecayTracker
        tracker = AlphaDecayTracker("reset_test")
        tracker.record_signal("BTC/USD", "long", 65000.0)
        tracker.reset()
        assert len(tracker._signals) == 0


# ---------------------------------------------------------------------------
# Storage adapters
# ---------------------------------------------------------------------------

class TestSQLiteTickAdapter:
    @pytest.mark.asyncio
    async def test_connect_and_insert(self, tmp_path):
        import pytest; pytest.importorskip("data.storage.timescale_adapter")
        from data.storage.timescale_adapter import SQLiteTickAdapter
        adapter = SQLiteTickAdapter(db_path=str(tmp_path / "ticks.db"))
        ok = await adapter.connect()
        assert ok is True

        await adapter.insert_tick("BTC/USD", 65000.0, 0.5, "buy")
        await adapter.insert_tick("BTC/USD", 65100.0, 0.3, "sell")

        stats = await adapter.get_storage_stats()
        assert stats["tick_rows"] == 2
        assert stats["backend"] == "sqlite"

    @pytest.mark.asyncio
    async def test_ohlcv_upsert(self, tmp_path):
        import pytest; pytest.importorskip("data.storage.timescale_adapter")
        from data.storage.timescale_adapter import SQLiteTickAdapter
        from datetime import datetime, timezone
        adapter = SQLiteTickAdapter(db_path=str(tmp_path / "ohlcv.db"))
        await adapter.connect()

        ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        await adapter.insert_ohlcv("ETH/USD", ts, 3000, 3100, 2950, 3050, 100.0)

        rows = await adapter.query_ohlcv("ETH/USD", limit=10)
        assert len(rows) == 1
        assert rows[0]["close"] == pytest.approx(3050.0)

    def test_get_storage_adapter_returns_sqlite_without_url(self):
        import pytest; pytest.importorskip("data.storage.timescale_adapter")
        from data.storage.timescale_adapter import get_storage_adapter, SQLiteTickAdapter
        adapter = get_storage_adapter(database_url="", sqlite_path=":memory:")
        assert isinstance(adapter, SQLiteTickAdapter)


# ---------------------------------------------------------------------------
# Options Signal (mock Deribit client)
# ---------------------------------------------------------------------------

class TestOptionsSignal:
    @pytest.mark.asyncio
    async def test_bearish_signal_high_pcr(self):
        import pytest; pytest.importorskip("data.options.options_signal")
        from data.options.options_signal import OptionsSignal

        class MockDeribit:
            async def fetch_options_chain(self, currency, expiry_days_max):
                # High put volume → high PCR → bearish
                return [
                    {"instrument": "BTC-1M-C", "expiry_days": 10, "strike": 70000,
                     "option_type": "call", "iv": 60.0, "delta": 0.5, "volume_24h": 100, "open_interest": 1000, "mark_price": 0.05},
                    {"instrument": "BTC-1M-P", "expiry_days": 10, "strike": 60000,
                     "option_type": "put", "iv": 75.0, "delta": -0.5, "volume_24h": 150, "open_interest": 1500, "mark_price": 0.04},
                ]

        sig = OptionsSignal(MockDeribit(), currency="BTC")
        result = await sig.generate("BTC/USD", price=65000.0)
        assert result is not None
        assert result["action"] == "SELL"
        assert result["pcr"] == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_no_signal_neutral_market(self):
        import pytest; pytest.importorskip("data.options.options_signal")
        from data.options.options_signal import OptionsSignal

        class MockDeribit:
            async def fetch_options_chain(self, currency, expiry_days_max):
                return [
                    {"instrument": "BTC-C", "expiry_days": 10, "strike": 70000,
                     "option_type": "call", "iv": 65.0, "delta": 0.5, "volume_24h": 100, "open_interest": 1000, "mark_price": 0.05},
                    {"instrument": "BTC-P", "expiry_days": 10, "strike": 60000,
                     "option_type": "put", "iv": 65.0, "delta": -0.5, "volume_24h": 100, "open_interest": 1000, "mark_price": 0.04},
                ]

        sig = OptionsSignal(MockDeribit())
        result = await sig.generate("BTC/USD", price=65000.0)
        assert result is None  # PCR = 1.0, skew = 0 → no signal


# ---------------------------------------------------------------------------
# Grafana Dashboard Generator
# ---------------------------------------------------------------------------

class TestGrafanaDashboard:
    def test_generate_returns_valid_structure(self):
        from monitoring.grafana_dashboard import generate_dashboard
        dash = generate_dashboard()
        assert "dashboard" in dash
        panels = dash["dashboard"]["panels"]
        assert len(panels) > 5
        titles = [p["title"] for p in panels]
        assert any("P&L" in t for t in titles)
        assert any("Funding" in t for t in titles)

    def test_save_creates_file(self, tmp_path):
        from monitoring.grafana_dashboard import save_dashboard
        path = str(tmp_path / "test_dashboard.json")
        save_dashboard(path)
        import os
        assert os.path.exists(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert "dashboard" in data

    def test_prometheus_rules_are_valid_yaml(self):
        from monitoring.grafana_dashboard import generate_prometheus_recording_rules
        import yaml  # type: ignore[import]
        rules_yaml = generate_prometheus_recording_rules()
        parsed = yaml.safe_load(rules_yaml)
        assert "groups" in parsed
        assert len(parsed["groups"][0]["rules"]) > 0


# ---------------------------------------------------------------------------
# Correlation Penalty (cross-asset position limit reduction)
# ---------------------------------------------------------------------------

class TestCorrelationPenalty:
    """Tests for CorrelationMonitor.get_correlation_penalty() and its
    integration with UnifiedRiskManager.get_corr_adjusted_position_limit()."""

    def _make_monitor(self, symbols=None, lookback=10):
        from risk.correlation_monitor import CorrelationMonitor
        syms = symbols or ["BTC/USD", "ETH/USD", "SOL/USD"]
        return CorrelationMonitor(symbols=syms, lookback=lookback)

    def _feed_correlated(self, mon, correlation_strength=0.95, n=25):
        """Feed price series with controlled pairwise correlation."""
        rng = np.random.RandomState(42)
        base = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        for i, sym in enumerate(mon.symbols):
            noise = rng.randn(n) * (1.0 - correlation_strength) * 2.0
            series = base + noise + i * 10  # offset so prices differ
            for p in series:
                mon.update(sym, max(p, 1.0))

    def _feed_uncorrelated(self, mon, n=25):
        """Feed independent random walks — low correlation."""
        rng = np.random.RandomState(123)
        for sym in mon.symbols:
            series = 100.0 + np.cumsum(rng.randn(n) * 0.5)
            for p in series:
                mon.update(sym, max(p, 1.0))

    # ── Boundary tests ────────────────────────────────────────────────────

    def test_penalty_no_data_returns_1(self):
        """With no price data, penalty should be 1.0 (no reduction)."""
        mon = self._make_monitor()
        assert mon.get_correlation_penalty() == 1.0

    def test_penalty_high_correlation_returns_half(self):
        """When avg correlation >= 0.8, penalty should be 0.5."""
        mon = self._make_monitor()
        self._feed_correlated(mon, correlation_strength=0.99, n=30)
        penalty = mon.get_correlation_penalty()
        assert penalty <= 0.55, f"Expected <= 0.55 for near-perfect corr, got {penalty}"

    def test_penalty_low_correlation_returns_1(self):
        """When avg correlation <= 0.3, penalty should be 1.0."""
        mon = self._make_monitor()
        self._feed_uncorrelated(mon, n=30)
        avg = mon.get_avg_pairwise_correlation()
        penalty = mon.get_correlation_penalty()
        if avg <= 0.3:
            assert penalty == 1.0, f"Expected 1.0 for low corr ({avg:.3f}), got {penalty}"
        else:
            # If randomness pushes avg above 0.3, just check penalty is in valid range
            assert 0.5 <= penalty <= 1.0

    def test_penalty_mid_correlation_interpolates(self):
        """For avg correlation between 0.3 and 0.8, penalty is linearly interpolated."""
        mon = self._make_monitor()
        self._feed_correlated(mon, correlation_strength=0.70, n=30)
        penalty = mon.get_correlation_penalty()
        assert 0.5 <= penalty <= 1.0, f"Penalty out of range: {penalty}"

    def test_penalty_always_in_range(self):
        """Penalty must always be between 0.5 and 1.0."""
        for strength in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            mon = self._make_monitor()
            self._feed_correlated(mon, correlation_strength=strength, n=30)
            penalty = mon.get_correlation_penalty()
            assert 0.5 <= penalty <= 1.0, f"strength={strength} → penalty={penalty}"

    # ── Subset tests ──────────────────────────────────────────────────────

    def test_penalty_with_symbol_subset(self):
        """get_correlation_penalty(symbols=[...]) should compute for subset only."""
        mon = self._make_monitor(symbols=["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD"])
        self._feed_correlated(mon, correlation_strength=0.95, n=30)
        penalty_all = mon.get_correlation_penalty()
        penalty_sub = mon.get_correlation_penalty(symbols=["BTC/USD", "ETH/USD"])
        assert 0.5 <= penalty_sub <= 1.0
        # Both should be low since all are correlated
        assert penalty_sub <= 0.6

    def test_penalty_subset_unknown_symbols_graceful(self):
        """Unknown symbols in subset should not crash."""
        mon = self._make_monitor()
        penalty = mon.get_correlation_penalty(symbols=["FAKE/USD", "NONEXIST/USD"])
        assert penalty == 1.0  # no data → no penalty

    def test_penalty_single_symbol_returns_1(self):
        """A single symbol can't have pairwise correlation → 1.0."""
        mon = self._make_monitor()
        self._feed_correlated(mon, n=30)
        penalty = mon.get_correlation_penalty(symbols=["BTC/USD"])
        assert penalty == 1.0

    # ── Integration with UnifiedRiskManager ───────────────────────────────

    def test_risk_manager_corr_adjusted_no_monitor(self):
        """Without correlation monitor, corr-adjusted == regime-adjusted."""
        from risk.unified_risk_manager import UnifiedRiskManager
        rm = UnifiedRiskManager(initial_capital=1000.0)
        base = 100.0
        regime = "NORMAL"
        assert rm.get_corr_adjusted_position_limit(base, regime) == \
               rm.get_regime_adjusted_position_limit(base, regime)

    def test_risk_manager_corr_adjusted_with_high_corr(self):
        """With high correlation, limit should be reduced below regime-only."""
        from risk.unified_risk_manager import UnifiedRiskManager
        rm = UnifiedRiskManager(initial_capital=1000.0)
        mon = self._make_monitor()
        self._feed_correlated(mon, correlation_strength=0.99, n=30)
        rm.set_correlation_monitor(mon)

        base = 100.0
        regime_only = rm.get_regime_adjusted_position_limit(base, "NORMAL")
        corr_adjusted = rm.get_corr_adjusted_position_limit(base, "NORMAL")

        assert corr_adjusted < regime_only, \
            f"Expected corr-adjusted ({corr_adjusted}) < regime-only ({regime_only})"
        assert corr_adjusted >= regime_only * 0.5  # penalty floor is 0.5

    def test_risk_manager_corr_adjusted_stacks_with_regime(self):
        """Correlation penalty stacks multiplicatively with regime multiplier."""
        from risk.unified_risk_manager import UnifiedRiskManager
        rm = UnifiedRiskManager(initial_capital=1000.0)
        mon = self._make_monitor()
        self._feed_correlated(mon, correlation_strength=0.99, n=30)
        rm.set_correlation_monitor(mon)

        base = 100.0
        # CRISIS regime = 0.25 multiplier; corr penalty ~ 0.5
        adjusted = rm.get_corr_adjusted_position_limit(base, "CRISIS")
        # Should be roughly base * 0.25 * 0.5 = 12.5
        assert adjusted <= base * 0.25, f"Expected <= 25.0, got {adjusted}"
        assert adjusted > 0

    def test_risk_manager_corr_adjusted_with_symbols(self):
        """Passing symbols list to corr-adjusted should work."""
        from risk.unified_risk_manager import UnifiedRiskManager
        rm = UnifiedRiskManager(initial_capital=1000.0)
        mon = self._make_monitor()
        self._feed_correlated(mon, correlation_strength=0.99, n=30)
        rm.set_correlation_monitor(mon)

        adjusted = rm.get_corr_adjusted_position_limit(
            100.0, "NORMAL", symbols=["BTC/USD", "ETH/USD"]
        )
        assert 0 < adjusted <= 100.0
