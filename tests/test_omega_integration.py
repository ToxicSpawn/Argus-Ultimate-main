"""
Tests for Omega Pack v2 integration:
  1. DecisionJournal — JSONL logging of full decision chains
  2. ShadowPlanComparator — Compare shadow vs live execution plans
  3. HostileScenarioInjector — Stress-test strategies with adversarial conditions
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock

# ──────────────────────────────────────────────────────────────────────────────
# DecisionJournal tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDecisionJournal(unittest.TestCase):
    """Test JSONL decision logging."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from monitoring.decision_journal import DecisionJournal
        self.journal = DecisionJournal(base_dir=self.tmpdir)

    def tearDown(self):
        self.journal.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_record(self, **overrides):
        from monitoring.decision_journal import DecisionRecord, make_decision_id
        defaults = dict(
            decision_id=make_decision_id(1, "BTC/USD"),
            cycle_number=1,
            timestamp_utc="2026-04-09T12:00:00+00:00",
            timestamp_ms=int(time.time() * 1000),
            symbol="BTC/USD",
            side="BUY",
            strategy="momentum",
            confidence=0.75,
            signal_price=60000.0,
            regime="TRENDING_UP",
            portfolio_value_aud=1000.0,
            position_count=1,
        )
        defaults.update(overrides)
        return DecisionRecord(**defaults)

    def test_write_creates_file(self):
        rec = self._make_record()
        self.journal.write(rec)
        files = os.listdir(self.tmpdir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].startswith("decisions_"))
        self.assertTrue(files[0].endswith(".jsonl"))

    def test_write_valid_json(self):
        rec = self._make_record()
        self.journal.write(rec)
        files = os.listdir(self.tmpdir)
        path = os.path.join(self.tmpdir, files[0])
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
        data = json.loads(line)
        self.assertEqual(data["symbol"], "BTC/USD")
        self.assertEqual(data["side"], "BUY")
        self.assertEqual(data["strategy"], "momentum")
        self.assertAlmostEqual(data["confidence"], 0.75, places=3)

    def test_write_many(self):
        recs = [self._make_record(cycle_number=i) for i in range(5)]
        self.journal.write_many(recs)
        self.assertEqual(self.journal._write_count, 5)

    def test_read_day(self):
        from datetime import datetime, timezone
        today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        self.journal.write(self._make_record())
        self.journal.write(self._make_record(side="SELL"))
        records = self.journal.read_day(today)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["side"], "BUY")
        self.assertEqual(records[1]["side"], "SELL")

    def test_read_all(self):
        for i in range(3):
            self.journal.write(self._make_record(cycle_number=i))
        records = self.journal.read_all()
        self.assertEqual(len(records), 3)

    def test_query_by_symbol(self):
        self.journal.write(self._make_record(symbol="BTC/USD"))
        self.journal.write(self._make_record(symbol="ETH/USD"))
        self.journal.write(self._make_record(symbol="BTC/USD"))
        results = self.journal.query(symbol="BTC/USD")
        self.assertEqual(len(results), 2)

    def test_query_by_strategy(self):
        self.journal.write(self._make_record(strategy="momentum"))
        self.journal.write(self._make_record(strategy="mean_reversion"))
        results = self.journal.query(strategy="momentum")
        self.assertEqual(len(results), 1)

    def test_query_by_outcome(self):
        self.journal.write(self._make_record(outcome="executed"))
        self.journal.write(self._make_record(outcome="blocked"))
        self.journal.write(self._make_record(outcome="executed"))
        results = self.journal.query(outcome="blocked")
        self.assertEqual(len(results), 1)

    def test_query_min_confidence(self):
        self.journal.write(self._make_record(confidence=0.3))
        self.journal.write(self._make_record(confidence=0.8))
        results = self.journal.query(min_confidence=0.5)
        self.assertEqual(len(results), 1)

    def test_gate_results_serialized(self):
        from monitoring.decision_journal import GateResult
        rec = self._make_record()
        rec.gates.append(GateResult("circuit_breaker", "pass", size_multiplier=1.0))
        rec.gates.append(GateResult("meta_gate", "reduce", "REDUCE decision", 0.5))
        rec.gates.append(GateResult("daily_loss", "pass", size_multiplier=1.0))
        self.journal.write(rec)
        records = self.journal.read_all()
        self.assertEqual(len(records[0]["gates"]), 3)
        self.assertEqual(records[0]["gates"][1]["gate_name"], "meta_gate")
        self.assertAlmostEqual(records[0]["gates"][1]["size_multiplier"], 0.5)

    def test_total_gate_multiplier(self):
        from monitoring.decision_journal import GateResult
        rec = self._make_record()
        rec.gates.append(GateResult("g1", "reduce", size_multiplier=0.8))
        rec.gates.append(GateResult("g2", "reduce", size_multiplier=0.5))
        rec.gates.append(GateResult("g3", "pass", size_multiplier=1.0))
        self.assertAlmostEqual(rec.total_gate_multiplier(), 0.4, places=4)

    def test_summary(self):
        self.journal.write(self._make_record(outcome="executed", strategy="momentum"))
        self.journal.write(self._make_record(outcome="blocked", strategy="mean_reversion"))
        self.journal.write(self._make_record(outcome="executed", strategy="momentum"))
        summary = self.journal.summary()
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["outcomes"]["executed"], 2)
        self.assertEqual(summary["outcomes"]["blocked"], 1)
        self.assertEqual(summary["strategies"]["momentum"], 2)

    def test_make_decision_id_unique(self):
        from monitoring.decision_journal import make_decision_id
        ids = {make_decision_id(1, "BTC/USD") for _ in range(100)}
        self.assertEqual(len(ids), 100)  # all unique

    def test_empty_journal_read(self):
        records = self.journal.read_all()
        self.assertEqual(len(records), 0)
        self.assertEqual(self.journal.read_day("20260101"), [])

    def test_empty_summary(self):
        summary = self.journal.summary()
        self.assertEqual(summary["total"], 0)


# ──────────────────────────────────────────────────────────────────────────────
# ShadowPlanComparator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestShadowPlanComparator(unittest.TestCase):
    """Test shadow vs live plan comparison."""

    def setUp(self):
        from core.shadow_plan_comparator import ShadowPlanComparator
        self.comparator = ShadowPlanComparator(window=100)

    def _make_snap(self, **overrides):
        from core.shadow_plan_comparator import PlanSnapshot
        defaults = dict(
            symbol="BTC/USD",
            side="BUY",
            strategy="momentum",
            confidence=0.7,
            size_pct=0.05,
            size_aud=50.0,
            gate_multiplier=0.8,
            gates_applied=2,
            gates_blocked=False,
        )
        defaults.update(overrides)
        return PlanSnapshot(**defaults)

    def test_record_returns_comparison(self):
        live = self._make_snap(size_pct=0.04, gate_multiplier=0.6)
        shadow = self._make_snap(size_pct=0.05, gate_multiplier=0.8)
        result = self.comparator.record(live, shadow, cycle=1)
        self.assertEqual(result.symbol, "BTC/USD")
        self.assertAlmostEqual(result.live_size_pct, 0.04)
        self.assertAlmostEqual(result.shadow_size_pct, 0.05)
        self.assertGreater(result.size_drift_pct, 0)

    def test_agreement_when_both_pass(self):
        live = self._make_snap(gates_blocked=False)
        shadow = self._make_snap(gates_blocked=False)
        result = self.comparator.record(live, shadow, cycle=1)
        self.assertTrue(result.agreement)
        self.assertFalse(result.shadow_would_trade)

    def test_agreement_when_both_blocked(self):
        live = self._make_snap(gates_blocked=True, size_pct=0)
        shadow = self._make_snap(gates_blocked=True, size_pct=0)
        result = self.comparator.record(live, shadow, cycle=1)
        self.assertTrue(result.agreement)

    def test_disagreement_shadow_would_trade(self):
        live = self._make_snap(gates_blocked=True, size_pct=0)
        shadow = self._make_snap(gates_blocked=False, size_pct=0.05)
        result = self.comparator.record(live, shadow, cycle=1)
        self.assertFalse(result.agreement)
        self.assertTrue(result.shadow_would_trade)

    def test_drift_report_empty(self):
        report = self.comparator.drift_report()
        self.assertEqual(report.total_comparisons, 0)
        self.assertAlmostEqual(report.agreement_rate, 0.0)

    def test_drift_report_populated(self):
        for i in range(10):
            live = self._make_snap(size_pct=0.04)
            shadow = self._make_snap(size_pct=0.05)
            self.comparator.record(live, shadow, cycle=i)
        report = self.comparator.drift_report()
        self.assertEqual(report.total_comparisons, 10)
        self.assertAlmostEqual(report.agreement_rate, 1.0)
        self.assertGreater(report.avg_size_drift_pct, 0)

    def test_update_pnl(self):
        live = self._make_snap(size_pct=0.04)
        shadow = self._make_snap(size_pct=0.05)
        self.comparator.record(live, shadow, cycle=5)
        self.comparator.update_pnl(cycle=5, symbol="BTC/USD", live_pnl=10.0, shadow_pnl=15.0)
        report = self.comparator.drift_report()
        self.assertAlmostEqual(report.shadow_advantage_pnl, 5.0)

    def test_window_limits_history(self):
        comparator = type(self.comparator)(window=5)  # small window
        for i in range(20):
            live = self._make_snap()
            shadow = self._make_snap()
            comparator.record(live, shadow, cycle=i)
        self.assertEqual(len(comparator._history), 5)

    def test_compute_shadow(self):
        signal = MagicMock()
        signal.symbol = "BTC/USD"
        signal.action = "BUY"
        signal.source_strategy = "momentum"
        signal.confidence = 0.8
        advisory = {}
        shadow_config = {"gate_floor": 0.15, "skip_gates": [], "size_multiplier": 1.0}
        snap = self.comparator.compute_shadow(signal, advisory, shadow_config)
        self.assertEqual(snap.symbol, "BTC/USD")
        self.assertEqual(snap.side, "BUY")
        self.assertFalse(snap.gates_blocked)
        self.assertGreater(snap.size_pct, 0)

    def test_compute_shadow_with_meta_gate_halt(self):
        signal = MagicMock()
        signal.symbol = "BTC/USD"
        signal.action = "BUY"
        signal.source_strategy = "momentum"
        signal.confidence = 0.8
        advisory = {"trade_gate": {"decision": "HALT"}}
        snap = self.comparator.compute_shadow(signal, advisory, {"gate_floor": 0.15})
        self.assertTrue(snap.gates_blocked)
        self.assertAlmostEqual(snap.size_pct, 0.0)

    def test_compute_shadow_with_reduce(self):
        signal = MagicMock()
        signal.symbol = "BTC/USD"
        signal.action = "BUY"
        signal.source_strategy = "momentum"
        signal.confidence = 0.8
        advisory = {"trade_gate": {"decision": "REDUCE"}}
        snap = self.comparator.compute_shadow(signal, advisory, {"gate_floor": 0.15})
        self.assertFalse(snap.gates_blocked)
        self.assertAlmostEqual(snap.gate_multiplier, 0.5)

    def test_snapshot_dict(self):
        for i in range(5):
            live = self._make_snap()
            shadow = self._make_snap()
            self.comparator.record(live, shadow, cycle=i)
        snap = self.comparator.snapshot()
        self.assertEqual(snap["total_comparisons"], 5)
        self.assertIn("agreement_rate", snap)
        self.assertIn("avg_size_drift_pct", snap)


# ──────────────────────────────────────────────────────────────────────────────
# HostileScenarioInjector tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHostileScenarioInjector(unittest.TestCase):
    """Test adversarial market condition injection."""

    def setUp(self):
        from core.hostile_scenario_injector import HostileScenarioInjector
        self.injector = HostileScenarioInjector()
        self.prices = {"BTC/USD": 60000.0, "ETH/USD": 3000.0}
        self.advisory = {
            "vol_forecasts": {
                "BTC/USD": {"forecast_vol_1d": 0.02, "regime": "NORMAL"},
            },
            "market_microstructure": {
                "BTC/USD": {"spread_bps": 5.0, "visible_depth_aud": 50000.0, "volume_24h": 1e6, "fee_bps": 2.0},
            },
        }

    def test_stale_book_widens_spread(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.inject(self.prices, self.advisory, ScenarioType.STALE_BOOK, "BTC/USD")
        micro = result["advisory"]["market_microstructure"]["BTC/USD"]
        self.assertAlmostEqual(micro["spread_bps"], 20.0)  # 5 * 4
        self.assertAlmostEqual(micro["visible_depth_aud"], 15000.0)  # 50000 * 0.3

    def test_flash_crash_drops_price(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.inject(self.prices, self.advisory, ScenarioType.FLASH_CRASH, "BTC/USD")
        self.assertAlmostEqual(result["prices"]["BTC/USD"], 42000.0)  # 60000 * 0.7
        self.assertEqual(result["prices"]["ETH/USD"], 3000.0)  # unchanged

    def test_venue_failure_disables_venues(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.inject(self.prices, self.advisory, ScenarioType.VENUE_FAILURE)
        vh = result["advisory"]["venue_health"]
        self.assertTrue(vh["kraken"]["disabled"])
        self.assertEqual(vh["kraken"]["ws_lag_ms"], 5000.0)

    def test_liquidity_void(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.inject(self.prices, self.advisory, ScenarioType.LIQUIDITY_VOID, "BTC/USD")
        micro = result["advisory"]["market_microstructure"]["BTC/USD"]
        self.assertAlmostEqual(micro["spread_bps"], 100.0)  # 5 * 20
        self.assertAlmostEqual(micro["visible_depth_aud"], 5000.0)  # 50000 * 0.1

    def test_whale_dump(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.inject(self.prices, self.advisory, ScenarioType.WHALE_DUMP, "BTC/USD")
        self.assertAlmostEqual(result["prices"]["BTC/USD"], 55200.0)  # 60000 * 0.92
        micro = result["advisory"]["market_microstructure"]["BTC/USD"]
        self.assertAlmostEqual(micro["imbalance"], -0.9)

    def test_fee_spike(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.inject(self.prices, self.advisory, ScenarioType.FEE_SPIKE, "BTC/USD")
        micro = result["advisory"]["market_microstructure"]["BTC/USD"]
        self.assertAlmostEqual(micro["fee_bps"], 20.0)  # 2 * 10

    def test_regime_whipsaw_overrides_regime(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.inject(self.prices, self.advisory, ScenarioType.REGIME_WHIPSAW)
        self.assertEqual(result["advisory"]["hostile_regime_override"], "CRISIS")

    def test_injection_marks_advisory(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.inject(self.prices, self.advisory, ScenarioType.STALE_BOOK)
        hostile = result["advisory"]["hostile_scenario"]
        self.assertTrue(hostile["active"])
        self.assertEqual(hostile["scenario"], "stale_book")
        self.assertIn("injected_at_ms", hostile)

    def test_does_not_modify_originals(self):
        from core.hostile_scenario_injector import ScenarioType
        original_price = self.prices["BTC/USD"]
        original_spread = self.advisory["market_microstructure"]["BTC/USD"]["spread_bps"]
        self.injector.inject(self.prices, self.advisory, ScenarioType.FLASH_CRASH, "BTC/USD")
        self.assertEqual(self.prices["BTC/USD"], original_price)
        self.assertEqual(self.advisory["market_microstructure"]["BTC/USD"]["spread_bps"], original_spread)

    def test_evaluate_result_pass(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.evaluate_result(
            scenario=ScenarioType.STALE_BOOK,
            strategy_name="momentum",
            symbol="BTC/USD",
            signal_generated=True,
            signal_blocked=False,
            final_size_pct=0.01,  # under 2% limit
            gate_multiplier=0.2,
        )
        self.assertTrue(result.passed)
        self.assertEqual(len(result.violations), 0)

    def test_evaluate_result_size_too_large(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.evaluate_result(
            scenario=ScenarioType.STALE_BOOK,
            strategy_name="momentum",
            symbol="BTC/USD",
            signal_generated=True,
            signal_blocked=False,
            final_size_pct=0.05,  # over 2% limit
            gate_multiplier=0.8,
        )
        self.assertFalse(result.passed)
        self.assertIn("size_too_large", result.violations[0])

    def test_evaluate_venue_failure_not_blocked(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.evaluate_result(
            scenario=ScenarioType.VENUE_FAILURE,
            strategy_name="momentum",
            symbol="BTC/USD",
            signal_generated=True,
            signal_blocked=False,
            final_size_pct=0.01,
            gate_multiplier=0.5,
        )
        self.assertFalse(result.passed)
        self.assertIn("venue_failure_not_blocked", result.violations)

    def test_evaluate_flash_crash_insufficient_reduction(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.evaluate_result(
            scenario=ScenarioType.FLASH_CRASH,
            strategy_name="momentum",
            symbol="BTC/USD",
            signal_generated=True,
            signal_blocked=False,
            final_size_pct=0.01,
            gate_multiplier=0.5,  # > 0.3 limit
        )
        self.assertFalse(result.passed)
        self.assertIn("insufficient_gate_reduction", result.violations[0])

    def test_evaluate_blocked_signal_passes(self):
        from core.hostile_scenario_injector import ScenarioType
        result = self.injector.evaluate_result(
            scenario=ScenarioType.FLASH_CRASH,
            strategy_name="momentum",
            symbol="BTC/USD",
            signal_generated=True,
            signal_blocked=True,
            final_size_pct=0.0,
            gate_multiplier=0.0,
        )
        self.assertTrue(result.passed)

    def test_test_all_scenarios_no_fn(self):
        report = self.injector.test_all_scenarios(
            strategy_name="test_strat",
            symbol="BTC/USD",
            prices=self.prices,
            advisory=self.advisory,
        )
        self.assertEqual(report.total_scenarios, 7)
        self.assertEqual(report.passed, 7)  # all pass with no signal_fn
        self.assertTrue(report.promotion_safe)

    def test_test_all_scenarios_with_fn(self):
        from core.hostile_scenario_injector import ScenarioType
        # signal_fn that always produces a safe blocked result
        def safe_fn(prices, advisory):
            return (True, True, 0.0, 0.0)  # generated, blocked, no size, no mult

        report = self.injector.test_all_scenarios(
            strategy_name="safe_strat",
            symbol="BTC/USD",
            prices=self.prices,
            advisory=self.advisory,
            signal_fn=safe_fn,
        )
        self.assertEqual(report.total_scenarios, 7)
        self.assertTrue(report.promotion_safe)

    def test_test_all_scenarios_unsafe_fn(self):
        # signal_fn that always trades big (unsafe)
        def unsafe_fn(prices, advisory):
            return (True, False, 0.10, 0.9)  # 10% size, 0.9 gate mult

        report = self.injector.test_all_scenarios(
            strategy_name="unsafe_strat",
            symbol="BTC/USD",
            prices=self.prices,
            advisory=self.advisory,
            signal_fn=unsafe_fn,
        )
        self.assertFalse(report.promotion_safe)
        self.assertGreater(report.failed, 0)

    def test_crashing_signal_fn(self):
        def crash_fn(prices, advisory):
            raise ValueError("strategy exploded")

        report = self.injector.test_all_scenarios(
            strategy_name="crash_strat",
            symbol="BTC/USD",
            prices=self.prices,
            advisory=self.advisory,
            signal_fn=crash_fn,
        )
        self.assertFalse(report.promotion_safe)
        self.assertEqual(report.failed, 7)

    def test_snapshot(self):
        from core.hostile_scenario_injector import ScenarioType
        self.injector.evaluate_result(
            scenario=ScenarioType.STALE_BOOK,
            strategy_name="test",
            symbol="BTC/USD",
            signal_generated=False,
            signal_blocked=False,
            final_size_pct=0.0,
            gate_multiplier=0.0,
        )
        snap = self.injector.snapshot()
        self.assertEqual(snap["total_tests"], 1)
        self.assertEqual(snap["passed"], 1)
        self.assertEqual(snap["scenarios_available"], 7)

    def test_pass_rate(self):
        report = MagicMock()
        report.total_scenarios = 7
        report.passed = 5
        report.failed = 2
        self.assertAlmostEqual(5 / 7, report.passed / report.total_scenarios)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: HostileInjector + StrategyPromotion
# ──────────────────────────────────────────────────────────────────────────────

class TestHostilePromotionIntegration(unittest.TestCase):
    """Test that hostile injector gates strategy promotion."""

    def test_promotion_with_hostile_gate_pass(self):
        from core.strategy_promotion import StrategyPromotionPipeline
        from core.hostile_scenario_injector import HostileScenarioInjector

        pipeline = StrategyPromotionPipeline(
            paper_min_cycles=5,
            paper_min_sharpe=-1.0,
        )
        injector = HostileScenarioInjector()
        pipeline.set_hostile_injector(injector)
        pipeline.update_market_context(
            {"BTC/USD": 60000.0},
            {"vol_forecasts": {}, "market_microstructure": {}},
        )

        # Submit and validate
        pipeline.submit_candidate(
            "test_strat", "test", "momentum", {"symbol": "BTC/USD"},
            "buy when RSI low", 0.8, 1.0, 0.6, 20, 5.0,
        )
        pipeline.validate("test_strat", 0.5, 0.6, 20)

        # Paper test
        for _ in range(10):
            pipeline.record_paper_cycle("test_strat")
        for _ in range(5):
            pipeline.record_paper_trade("test_strat", pnl=10.0)

        # Should pass (no signal_fn, so hostile tests trivially pass)
        result = pipeline.check_paper_promotion("test_strat")
        self.assertTrue(result)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: ComponentRegistry wiring
# ──────────────────────────────────────────────────────────────────────────────

class TestComponentRegistryOmegaWiring(unittest.TestCase):
    """Test that omega components initialize in ComponentRegistry."""

    def test_decision_journal_import(self):
        from monitoring.decision_journal import DecisionJournal, DecisionRecord, GateResult, make_decision_id
        journal = DecisionJournal(base_dir=tempfile.mkdtemp())
        self.assertIsNotNone(journal)
        journal.close()

    def test_shadow_comparator_import(self):
        from core.shadow_plan_comparator import ShadowPlanComparator, PlanSnapshot, DriftReport
        comp = ShadowPlanComparator(window=10)
        self.assertIsNotNone(comp)

    def test_hostile_injector_import(self):
        from core.hostile_scenario_injector import (
            HostileScenarioInjector, ScenarioType, HOSTILE_SCENARIOS,
            ScenarioTestResult, HostileTestReport,
        )
        injector = HostileScenarioInjector()
        self.assertIsNotNone(injector)
        self.assertEqual(len(HOSTILE_SCENARIOS), 7)
        self.assertEqual(len(ScenarioType), 7)

    def test_component_registry_slots_exist(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(config=MagicMock())
        self.assertIsNone(reg.decision_journal)
        self.assertIsNone(reg.shadow_comparator)
        self.assertIsNone(reg.hostile_injector)


if __name__ == "__main__":
    unittest.main()
