from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive.feature_store import RollingFeatureStore
from adaptive.regime_engine import DeterministicRegimeClassifier
from monitoring.jsonl_logger import JSONLLogger
from monitoring.ops_metrics import OpsMetrics
from execution.target_portfolio_engine import TargetPortfolioEngine
from research.walk_forward_harness import run_walk_forward
from scripts.daily_report import build_report
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
        self.captured_signals = list(signals)
        return []


class _FakeAIBrain:
    def __init__(self, signals):
        self._signals = list(signals)

    async def generate_trading_signals(self):
        return list(self._signals)


class TestPR07TargetsEngine(unittest.TestCase):
    def test_targets_build_deterministic_execution_signals(self) -> None:
        cfg = SimpleNamespace(
            aud_to_usd=0.65,
            max_position_pct=0.25,
            max_position_size_aud=500.0,
            target_convergence_alpha=0.5,
            target_vol_pct=2.0,
            realized_vol_pct=2.0,
            target_cluster_map={"BTC/USD": "majors", "ETH/USD": "majors"},
            target_cluster_cap_pct=0.20,
        )
        engine = TargetPortfolioEngine(cfg)
        signals = [
            _Signal(symbol="ETH/USD", action="BUY", confidence=1.0, quantity=0.5, entry_price=2000.0, strategy="s1"),
            _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, quantity=0.05, entry_price=60000.0, strategy="s2"),
        ]
        targets = engine.build_targets(signals=signals, current_positions={}, equity_aud=10000.0)
        self.assertEqual([t.symbol for t in targets], ["BTC/USD", "ETH/USD"])

        aud_notional = 0.0
        for t in targets:
            px_aud = t.price / cfg.aud_to_usd
            aud_notional += abs(t.target_qty * px_aud)
        self.assertLessEqual(aud_notional, 2000.0 + 1e-6)

        exec_signals = engine.build_execution_signals(targets=targets, regime_label="trend_up:mid_vol")
        self.assertEqual(len(exec_signals), 2)
        self.assertTrue(all(getattr(s, "strategy", "") == "target_rebalance" for s in exec_signals))
        self.assertTrue(all(getattr(s, "regime_label", "") == "trend_up:mid_vol" for s in exec_signals))

    def test_target_engine_v1_aggregation_caps_and_scaling(self) -> None:
        cfg = SimpleNamespace(
            aud_to_usd=1.0,
            max_position_pct=0.10,
            max_total_exposure_pct=0.15,
            target_cluster_cap_pct=1.0,
            target_rebalance_min_delta_pct=0.02,
            target_score_confidence_weight=1.0,
            target_score_net_edge_weight=1.0,
            target_regime_boost_enabled=False,
        )
        engine = TargetPortfolioEngine(cfg)
        signals = [
            _Signal(symbol="BTC/USD", action="BUY", confidence=0.9, expected_net_edge_bps=20.0, quantity=0.1, entry_price=100.0, strategy="s1"),
            _Signal(symbol="BTC/USD", action="BUY", confidence=0.7, expected_net_edge_bps=15.0, quantity=0.1, entry_price=100.0, strategy="s2"),
            _Signal(symbol="ETH/USD", action="BUY", confidence=0.8, expected_net_edge_bps=10.0, quantity=0.1, entry_price=100.0, strategy="s3"),
        ]
        targets = engine.build_targets(signals=signals, current_positions={}, equity_aud=1000.0, regime_label="range:mid_vol")

        by_symbol = {t.symbol: t for t in targets}
        self.assertIn("BTC/USD", by_symbol)
        self.assertIn("ETH/USD", by_symbol)
        self.assertGreater(by_symbol["BTC/USD"].priority_score, by_symbol["ETH/USD"].priority_score)
        self.assertLessEqual(by_symbol["BTC/USD"].target_exposure_pct, 0.10 + 1e-9)
        self.assertLessEqual(by_symbol["ETH/USD"].target_exposure_pct, 0.10 + 1e-9)
        self.assertLessEqual(sum(max(0.0, t.target_exposure_pct) for t in targets), 0.15 + 1e-9)

        scaled = engine.build_targets(
            signals=signals,
            current_positions={},
            equity_aud=1000.0,
            regime_label="range:mid_vol",
            risk_scale=0.5,
        )
        self.assertLessEqual(
            sum(max(0.0, t.target_exposure_pct) for t in scaled),
            sum(max(0.0, t.target_exposure_pct) for t in targets) * 0.5 + 1e-9,
        )

    def test_target_engine_v1_small_delta_suppression(self) -> None:
        cfg = SimpleNamespace(
            aud_to_usd=1.0,
            max_position_pct=0.10,
            max_total_exposure_pct=0.05,
            target_cluster_cap_pct=1.0,
            target_rebalance_min_delta_pct=0.02,
            target_score_confidence_weight=1.0,
            target_score_net_edge_weight=1.0,
            target_regime_boost_enabled=False,
        )
        engine = TargetPortfolioEngine(cfg)
        current_positions = {"BTC/USD": {"quantity": 0.49, "current_price": 100.0}}
        signals = [
            _Signal(symbol="BTC/USD", action="BUY", confidence=1.0, expected_net_edge_bps=20.0, quantity=0.1, entry_price=100.0, strategy="s1"),
        ]
        targets = engine.build_targets(signals=signals, current_positions=current_positions, equity_aud=1000.0, regime_label="range:mid_vol")
        btc = [t for t in targets if t.symbol == "BTC/USD"][0]
        self.assertLess(abs(btc.delta_exposure_pct), 0.02)
        self.assertTrue(any(str(r).startswith("suppressed:small_delta") for r in btc.reasons))
        self.assertEqual(engine.build_execution_signals(targets=targets, regime_label="range:mid_vol"), [])


class TestPR08RegimeEngine(unittest.TestCase):
    def test_regime_classifier_and_strategy_gating(self) -> None:
        store = RollingFeatureStore(window=10)
        for px in [100.0, 101.0, 102.5, 103.5, 105.0]:
            store.update(symbol="BTC/USD", price=px, spread_bps=3.0, depth=10.0, volume=50.0)
        feats = store.snapshot()

        clf = DeterministicRegimeClassifier(trend_threshold=0.004, high_vol_pct=2.0, low_vol_pct=0.5)
        regime = clf.classify(feats)
        self.assertIn(regime.trend_state, {"trend_up", "trend_down", "range"})
        self.assertIn(regime.vol_state, {"low_vol", "mid_vol", "high_vol"})

        cfg = UnifiedConfig()
        cfg.regime_gating_enabled = True
        cfg.regime_strategy_map = {"trend_up:low_vol": ["trend_following"]}
        sys = UnifiedSystemArchitecture(cfg)
        sigs = [
            _Signal(symbol="BTC/USD", strategy="trend_following", source_strategy="trend_following"),
            _Signal(symbol="BTC/USD", strategy="mean_reversion", source_strategy="mean_reversion"),
        ]
        kept = sys._apply_regime_strategy_gating(sigs, "trend_up:low_vol")
        self.assertEqual(len(kept), 1)
        self.assertEqual(getattr(kept[0], "strategy"), "trend_following")


class TestPR07PR08LoopIntegration(unittest.IsolatedAsyncioTestCase):
    async def _build_system(self, *, strategy_map):
        cfg = UnifiedConfig()
        cfg.run_mode = "paper"
        cfg.continuous_scan_enabled = False
        cfg.self_improvement_enabled = False
        cfg.ai_enabled = True
        cfg.quant_fund_upgrades_enabled = False
        cfg.targets_enabled = True
        cfg.target_convergence_alpha = 1.0
        cfg.target_cluster_cap_pct = 0.9
        cfg.feature_store_enabled = True
        cfg.feature_store_window = 10
        cfg.regime_classifier_enabled = True
        cfg.regime_gating_enabled = True
        cfg.regime_strategy_map = strategy_map
        cfg.reconciliation_interval_cycles = 0
        cfg.paper_trading_peak_mode = False
        cfg.max_trades_per_day = 0
        cfg.edge_cost_gate_enabled = False
        cfg.max_consecutive_losses = 1000

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
                    quantity=0.01,
                    entry_price=100.0,
                    strategy="trend_following",
                    source_strategy="trend_following",
                    spread_bps=2.0,
                    volume=100.0,
                    depth=50.0,
                ),
                _Signal(
                    symbol="BTC/USD",
                    action="BUY",
                    confidence=0.8,
                    quantity=0.01,
                    entry_price=101.0,
                    strategy="trend_following",
                    source_strategy="trend_following",
                    spread_bps=2.0,
                    volume=100.0,
                    depth=50.0,
                ),
                _Signal(
                    symbol="BTC/USD",
                    action="BUY",
                    confidence=0.7,
                    quantity=0.01,
                    entry_price=102.0,
                    strategy="trend_following",
                    source_strategy="trend_following",
                    spread_bps=2.0,
                    volume=100.0,
                    depth=50.0,
                ),
            ]
        )
        sys.target_engine = TargetPortfolioEngine(cfg)
        sys.feature_store = RollingFeatureStore(window=10)
        sys.regime_classifier = DeterministicRegimeClassifier(trend_threshold=0.001, high_vol_pct=50.0, low_vol_pct=0.0)
        sys.market_data_service = None
        return sys

    async def test_loop_converts_to_target_rebalance_signal(self) -> None:
        sys = await self._build_system(strategy_map={"trend_up:mid_vol": ["trend_following"], "trend_up:low_vol": ["trend_following"], "range:mid_vol": ["trend_following"]})
        await sys.run_trading_loop(cycle_seconds=0.0, max_cycles=1)

        sent = sys.execution_engine.captured_signals
        self.assertIsNotNone(sent)
        self.assertEqual(len(sent), 1)
        self.assertEqual(getattr(sent[0], "strategy", ""), "target_rebalance")
        self.assertTrue(str(getattr(sent[0], "regime_label", "")))

    async def test_loop_regime_gate_can_block_all_signals(self) -> None:
        sys = await self._build_system(strategy_map={"trend_up:mid_vol": ["mean_reversion"], "trend_up:low_vol": ["mean_reversion"], "range:mid_vol": ["mean_reversion"]})
        await sys.run_trading_loop(cycle_seconds=0.0, max_cycles=1)

        sent = sys.execution_engine.captured_signals
        self.assertIsNotNone(sent)
        self.assertEqual(sent, [])

    async def test_rejected_candidates_are_snapshotted_with_run_and_trace(self) -> None:
        sys = await self._build_system(strategy_map={"trend_up:mid_vol": ["mean_reversion"], "trend_up:low_vol": ["mean_reversion"], "range:mid_vol": ["mean_reversion"]})
        td = Path(tempfile.mkdtemp())
        omega_db = td / "omega.db"
        sys.omega_store = OmegaSQLiteStore(str(omega_db))
        sys.omega_store.init_schema()

        await sys.run_trading_loop(cycle_seconds=0.0, max_cycles=1)

        conn = sqlite3.connect(str(omega_db))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT run_id, trace_id, reason_code, allowed
            FROM decision_snapshots
            WHERE reason_code = 'PRE_TRADE_RISK_BLOCK'
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        run_id, trace_id, reason_code, allowed = row
        self.assertTrue(str(run_id))
        self.assertTrue(str(trace_id))
        self.assertEqual(str(reason_code), "PRE_TRADE_RISK_BLOCK")
        self.assertEqual(int(allowed), 0)

    async def test_target_fields_are_snapshotted(self) -> None:
        sys = await self._build_system(strategy_map={"trend_up:mid_vol": ["trend_following"], "trend_up:low_vol": ["trend_following"], "range:mid_vol": ["trend_following"]})
        td = Path(tempfile.mkdtemp())
        omega_db = td / "omega_targets.db"
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
        self.assertIn("target_exposure_pct", details)
        self.assertIn("current_exposure_pct", details)
        self.assertIn("delta_exposure_pct", details)
        self.assertIn("priority_score", details)
        self.assertIn("expected_net_edge_bps", details)


class TestPR09ResearchHarness(unittest.TestCase):
    def test_walk_forward_creates_report_and_bundle(self) -> None:
        td = Path(tempfile.mkdtemp())
        cfg_path = td / "unified_config.yaml"
        cfg_path.write_text("config_version: 1\nruntime:\n  mode: paper\n", encoding="utf-8")

        result = run_walk_forward(config_path=str(cfg_path), report_dir="reports", root_dir=str(td))

        report_path = Path(result.report_path)
        bundle_path = Path(result.bundle_path)
        self.assertTrue(report_path.exists())
        self.assertTrue(bundle_path.exists())
        self.assertTrue((bundle_path / "config.yaml").exists())
        self.assertTrue((bundle_path / "params.json").exists())
        self.assertTrue((bundle_path / "build_info.json").exists())
        self.assertTrue((bundle_path / "hashes.txt").exists())
        self.assertTrue((bundle_path / "changelog.md").exists())


class TestPR10OpsObservability(unittest.TestCase):
    def test_ops_metrics_jsonl_and_daily_report(self) -> None:
        td = Path(tempfile.mkdtemp())
        metrics = OpsMetrics()
        metrics.observe_decision(allowed=False, reason_code="EDGE_COST_REJECT", net_edge_bps=-3.0, latency_ms=12.0)
        metrics.observe_trade(slippage_bps=4.0, fee=1.5, notional=300.0)
        summary = metrics.summary()
        self.assertEqual(summary["reject_histogram"].get("EDGE_COST_REJECT"), 1)
        self.assertGreater(summary["slippage_p50_bps"], 0.0)
        self.assertGreater(summary["fee_churn_ratio"], 0.0)

        log_path = td / "events.jsonl"
        jl = JSONLLogger(str(log_path))
        jl.write({"kind": "decision_snapshot", "reason_code": "EDGE_COST_REJECT"})
        row = json.loads(log_path.read_text(encoding="utf-8").strip())
        self.assertEqual(row.get("kind"), "decision_snapshot")
        self.assertIn("ts", row)

        db_path = td / "ledger.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE trades (price REAL, size REAL, slippage REAL, commission REAL, value REAL)")
        cur.execute("CREATE TABLE decision_snapshots (allowed INTEGER, reason_code TEXT, cost_json TEXT)")
        cur.execute("INSERT INTO trades VALUES (100.0, 1.0, 0.0005, 0.1, 100.0)")
        cur.execute(
            "INSERT INTO decision_snapshots VALUES (0, 'EDGE_COST_REJECT', ?)",
            (json.dumps({"net_edge_bps": -1.2}),),
        )
        conn.commit()
        conn.close()

        report = build_report(str(db_path))
        self.assertEqual(report["trade_count"], 1)
        self.assertEqual(report["decision_count"], 1)
        self.assertEqual(report["reject_histogram"].get("EDGE_COST_REJECT"), 1)


if __name__ == "__main__":
    unittest.main()
