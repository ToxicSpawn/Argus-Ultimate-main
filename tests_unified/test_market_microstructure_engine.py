from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive.market_microstructure_engine import MarketMicrostructureEngine
from execution.target_portfolio_engine import TargetPortfolioEngine
from unified_trading_system import OmegaSQLiteStore, SystemState, UnifiedConfig, UnifiedSystemArchitecture


class _Signal(SimpleNamespace):
    pass


class _FakeMonitoring:
    async def update_metrics(self, _metrics):
        return None


class _FakeRiskManager:
    def update_capital(self, *_args, **_kwargs):
        return None

    def set_total_exposure(self, *_args, **_kwargs):
        return None

    def check_circuit_breaker(self):
        return False


class _FakeCapitalOptimizer:
    def update_capital(self, *_args, **_kwargs):
        return None

    async def optimize_signals(self, signals):
        return list(signals)


class _FakeExecutionEngine:
    def __init__(self):
        self.captured_signals = None
        self.trade_ledger = SimpleNamespace(record_event=lambda **_kwargs: None)

    async def execute_signals(self, signals, correlation_id=None):
        _ = correlation_id
        self.captured_signals = list(signals)
        return []


class _FakeAIBrain:
    def __init__(self, signals):
        self._signals = list(signals)

    async def generate_trading_signals(self):
        return list(self._signals)


class TestMarketMicrostructureEngine(unittest.TestCase):
    def test_spread_obi_and_microprice(self) -> None:
        engine = MarketMicrostructureEngine(rolling_window=10)
        st = engine.update_symbol(
            symbol="BTC/USD",
            best_bid=99.0,
            best_ask=101.0,
            bid_size=3.0,
            ask_size=1.0,
        )
        self.assertAlmostEqual(st.spread_bps, 200.0, places=6)
        self.assertAlmostEqual(st.order_book_imbalance, 0.5, places=6)
        self.assertGreater(st.microprice, st.mid_price)
        self.assertEqual(st.microstructure_bias, "up")

    def test_liquidity_vacuum_and_adverse_selection(self) -> None:
        engine = MarketMicrostructureEngine(
            rolling_window=10,
            vacuum_spread_jump_bps=2.0,
            vacuum_depth_drop_ratio=0.7,
            high_adverse_selection_threshold=0.4,
        )
        engine.update_symbol(
            symbol="ETH/USD",
            best_bid=100.0,
            best_ask=100.02,
            bid_size=10.0,
            ask_size=10.0,
            spread_bps=2.0,
            trade_velocity=1.0,
        )
        st = engine.update_symbol(
            symbol="ETH/USD",
            best_bid=100.0,
            best_ask=100.20,
            bid_size=1.0,
            ask_size=1.0,
            spread_bps=20.0,
            trade_velocity=8.0,
        )
        self.assertTrue(st.liquidity_vacuum_flag)
        self.assertGreaterEqual(st.adverse_selection_risk, 0.4)

    def test_annotation_writes_microstructure_fields(self) -> None:
        engine = MarketMicrostructureEngine(rolling_window=5)
        sig = _Signal(
            symbol="BTC/USD",
            entry_price=100.0,
            best_bid=99.9,
            best_ask=100.1,
            top_of_book_bid_size=5.0,
            top_of_book_ask_size=4.0,
            spread_bps=20.0,
            trade_velocity=4.0,
            confidence=0.8,
        )
        engine.update_from_signals([sig])
        out = engine.annotate_signals([sig])[0]
        self.assertTrue(hasattr(out, "order_book_imbalance"))
        self.assertTrue(hasattr(out, "microprice"))
        self.assertTrue(hasattr(out, "adverse_selection_risk"))
        self.assertTrue(hasattr(out, "microstructure_bias"))


class TestMicrostructureIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_persists_microstructure_fields(self) -> None:
        cfg = UnifiedConfig()
        cfg.run_mode = "paper"
        cfg.continuous_scan_enabled = False
        cfg.self_improvement_enabled = False
        cfg.ai_enabled = True
        cfg.quant_fund_upgrades_enabled = False
        cfg.targets_enabled = True
        cfg.target_convergence_alpha = 1.0
        cfg.target_cluster_cap_pct = 1.0
        cfg.feature_store_enabled = True
        cfg.feature_store_window = 10
        cfg.regime_classifier_enabled = True
        cfg.regime_gating_enabled = False
        cfg.market_microstructure_enabled = True
        cfg.market_microstructure_rolling_window = 10
        cfg.market_microstructure_vacuum_spread_jump_bps = 2.0
        cfg.market_microstructure_vacuum_depth_drop_ratio = 0.7
        cfg.market_microstructure_high_adverse_selection_threshold = 0.5
        cfg.market_microstructure_use_in_execution_alpha = True
        cfg.market_microstructure_use_in_liquidity_risk = True
        cfg.reconciliation_interval_cycles = 0
        cfg.paper_trading_peak_mode = False
        cfg.max_trades_per_day = 0
        cfg.edge_cost_gate_enabled = False
        cfg.max_consecutive_losses = 1000
        cfg.multi_language_enabled = False

        sys = UnifiedSystemArchitecture(cfg)
        sys.state = SystemState.RUNNING
        sys.monitoring = _FakeMonitoring()
        sys.unified_risk_manager = _FakeRiskManager()
        sys.capital_optimizer = _FakeCapitalOptimizer()
        sys.execution_engine = _FakeExecutionEngine()
        sys.ai_brain = _FakeAIBrain(
            [
                _Signal(
                    symbol="BTC/USD",
                    action="BUY",
                    confidence=0.9,
                    quantity=0.05,
                    entry_price=100.0,
                    strategy="momentum",
                    source_strategy="momentum",
                    best_bid=99.95,
                    best_ask=100.05,
                    top_of_book_bid_size=5.0,
                    top_of_book_ask_size=4.0,
                    spread_bps=10.0,
                    trade_velocity=6.0,
                )
            ]
        )
        sys.target_engine = TargetPortfolioEngine(cfg)

        td = Path(tempfile.mkdtemp())
        omega_db = td / "omega_micro.db"
        sys.omega_store = OmegaSQLiteStore(str(omega_db))
        sys.omega_store.init_schema()

        await sys.run_trading_loop(cycle_seconds=0.0, max_cycles=1)

        conn = sqlite3.connect(str(omega_db))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT details_json
            FROM decision_snapshots
            WHERE reason_code IN ('PRE_EXEC', 'PRE_PUBLISH')
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        details = json.loads(str(row[0] or "{}"))
        self.assertIn("spread_bps", details)
        self.assertIn("order_book_imbalance", details)
        self.assertIn("microprice", details)
        self.assertIn("trade_velocity", details)
        self.assertIn("liquidity_vacuum_flag", details)
        self.assertIn("adverse_selection_risk", details)
        self.assertIn("microstructure_bias", details)


if __name__ == "__main__":
    unittest.main()
