from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from execution.target_portfolio_engine import TargetPortfolioEngine
from risk.liquidity_risk_engine import LiquidityRiskEngine
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


class TestLiquidityRiskEngine(unittest.TestCase):
    def _cfg(self, **overrides):
        cfg = SimpleNamespace(
            liquidity_risk_depth_fraction_limit=0.04,
            liquidity_risk_thin_spread_threshold_bps=6.0,
            liquidity_risk_danger_spread_threshold_bps=12.0,
            liquidity_risk_min_depth_threshold=0.5,
            liquidity_risk_slippage_threshold_bps=10.0,
            liquidity_risk_min_liquidity_score=0.2,
            liquidity_risk_score_weights={"depth": 1.0, "spread": 1.0, "fill_ratio": 0.75},
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_liquidity_score_calculation(self) -> None:
        engine = LiquidityRiskEngine(self._cfg())
        state = engine.evaluate_state(
            symbol="BTC/USD",
            spread_bps=2.0,
            bid_size=8.0,
            ask_size=7.0,
            depth_estimate=15.0,
            slippage_estimate_bps=3.0,
            maker_fill_ratio=0.8,
        )
        self.assertEqual(state.liquidity_state, "normal")
        self.assertGreater(state.liquidity_score, 0.5)
        self.assertGreater(state.max_safe_trade_size, 0.0)

    def test_trade_size_clamping(self) -> None:
        cfg = self._cfg(liquidity_risk_depth_fraction_limit=0.10)
        engine = LiquidityRiskEngine(cfg)
        target = _Signal(
            symbol="BTC/USD",
            target_exposure_pct=0.25,
            current_exposure_pct=0.00,
            delta_exposure_pct=0.25,
            priority_score=1.0,
            expected_net_edge_bps=15.0,
            regime_label="trend_up:mid_vol",
            reasons=[],
            price=100.0,
            reference_price=100.0,
            current_qty=0.0,
            target_qty=2.5,
            delta_qty=2.5,
        )
        adjusted, states, clamp_count = engine.adjust_targets(
            targets=[target],
            symbol_market_state={
                "BTC/USD": {
                    "spread_bps": 4.0,
                    "top_of_book_bid_size": 1.0,
                    "top_of_book_ask_size": 1.0,
                    "orderbook_depth_estimate": 2.0,
                    "price": 100.0,
                }
            },
            execution_telemetry={"BTC/USD": {"maker_fill_ratio": 0.6, "slippage_p90": 5.0}},
            equity_aud=1000.0,
            aud_to_usd=1.0,
        )
        self.assertEqual(clamp_count, 1)
        t = adjusted[0]
        self.assertTrue(bool(getattr(t, "liquidity_clamp_flag", False)))
        self.assertLessEqual(abs(float(getattr(t, "delta_qty", 0.0))), float(states["BTC/USD"].max_safe_trade_size) + 1e-9)
        self.assertAlmostEqual(float(getattr(t, "delta_exposure_pct", 0.0)), 0.01, places=6)

    def test_danger_liquidity_suppression(self) -> None:
        engine = LiquidityRiskEngine(self._cfg())
        target = _Signal(
            symbol="ETH/USD",
            target_exposure_pct=0.15,
            current_exposure_pct=0.00,
            delta_exposure_pct=0.15,
            priority_score=1.0,
            expected_net_edge_bps=10.0,
            regime_label="range:high_vol",
            reasons=[],
            price=100.0,
            reference_price=100.0,
            current_qty=0.0,
            target_qty=1.5,
            delta_qty=1.5,
        )
        adjusted, states, _ = engine.adjust_targets(
            targets=[target],
            symbol_market_state={
                "ETH/USD": {
                    "spread_bps": 30.0,
                    "top_of_book_bid_size": 0.0,
                    "top_of_book_ask_size": 0.0,
                    "orderbook_depth_estimate": 0.0,
                    "price": 100.0,
                }
            },
            execution_telemetry={"ETH/USD": {"maker_fill_ratio": 0.1, "slippage_p90": 25.0}},
            equity_aud=1000.0,
            aud_to_usd=1.0,
        )
        t = adjusted[0]
        self.assertEqual(states["ETH/USD"].liquidity_state, "danger")
        self.assertAlmostEqual(float(getattr(t, "delta_qty", 0.0)), 0.0, places=9)
        self.assertTrue(any(str(r).startswith("suppressed:liquidity_danger") for r in list(getattr(t, "reasons", []) or [])))

    def test_low_liquidity_score_suppression(self) -> None:
        engine = LiquidityRiskEngine(self._cfg(liquidity_risk_min_liquidity_score=0.8))
        target = _Signal(
            symbol="BTC/USD",
            target_exposure_pct=0.15,
            current_exposure_pct=0.00,
            delta_exposure_pct=0.15,
            priority_score=1.0,
            expected_net_edge_bps=10.0,
            regime_label="range:high_vol",
            reasons=[],
            price=100.0,
            reference_price=100.0,
            current_qty=0.0,
            target_qty=1.5,
            delta_qty=1.5,
        )
        adjusted, _states, _ = engine.adjust_targets(
            targets=[target],
            symbol_market_state={
                "BTC/USD": {
                    "spread_bps": 9.0,
                    "top_of_book_bid_size": 0.4,
                    "top_of_book_ask_size": 0.3,
                    "orderbook_depth_estimate": 0.7,
                    "price": 100.0,
                }
            },
            execution_telemetry={"BTC/USD": {"maker_fill_ratio": 0.2, "slippage_p90": 15.0}},
            equity_aud=1000.0,
            aud_to_usd=1.0,
        )
        t = adjusted[0]
        self.assertAlmostEqual(float(getattr(t, "delta_qty", 0.0)), 0.0, places=9)
        self.assertTrue(any(str(r).startswith("suppressed:liquidity_score_low") for r in list(getattr(t, "reasons", []) or [])))


class TestLiquidityIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_persists_liquidity_fields(self) -> None:
        cfg = UnifiedConfig()
        cfg.run_mode = "paper"
        cfg.continuous_scan_enabled = False
        cfg.self_improvement_enabled = False
        cfg.ai_enabled = True
        cfg.quant_fund_upgrades_enabled = False
        cfg.targets_enabled = True
        cfg.target_convergence_alpha = 1.0
        cfg.target_cluster_cap_pct = 1.0
        cfg.feature_store_enabled = False
        cfg.regime_classifier_enabled = False
        cfg.regime_gating_enabled = False
        cfg.reconciliation_interval_cycles = 0
        cfg.paper_trading_peak_mode = False
        cfg.max_trades_per_day = 0
        cfg.edge_cost_gate_enabled = False
        cfg.max_consecutive_losses = 1000
        cfg.multi_language_enabled = False
        cfg.liquidity_risk_enabled = True
        cfg.liquidity_risk_depth_fraction_limit = 0.05
        cfg.liquidity_risk_thin_spread_threshold_bps = 6.0
        cfg.liquidity_risk_danger_spread_threshold_bps = 12.0
        cfg.liquidity_risk_min_depth_threshold = 0.5
        cfg.liquidity_risk_slippage_threshold_bps = 10.0
        cfg.liquidity_risk_score_weights = {"depth": 1.0, "spread": 1.0, "fill_ratio": 0.75}

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
                    confidence=0.95,
                    quantity=0.2,
                    entry_price=100.0,
                    strategy="trend_following",
                    source_strategy="trend_following",
                    spread_bps=8.0,
                    top_of_book_bid_size=5.0,
                    top_of_book_ask_size=5.0,
                    orderbook_depth_estimate=10.0,
                )
            ]
        )
        sys.target_engine = TargetPortfolioEngine(cfg)
        sys.liquidity_risk_engine = LiquidityRiskEngine(cfg)

        td = Path(tempfile.mkdtemp())
        omega_db = td / "omega_liquidity.db"
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
        self.assertIn("liquidity_score", details)
        self.assertIn("liquidity_state", details)
        self.assertIn("max_safe_trade_size", details)
        self.assertIn("adjusted_target_exposure_pct", details)
        self.assertIn("liquidity_clamp_flag", details)

        sent = sys.execution_engine.captured_signals or []
        self.assertEqual(len(sent), 1)
        sig = sent[0]
        self.assertTrue(hasattr(sig, "liquidity_score"))
        self.assertTrue(hasattr(sig, "liquidity_state"))


if __name__ == "__main__":
    unittest.main()
