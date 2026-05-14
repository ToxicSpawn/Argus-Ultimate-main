"""
Tests for ARGUS institutional layer:
- Event-sourced truth layer
- Replay engine
- Execution attribution
- Drift governance
- Executive meta-controller
- Constitutional enforcement
- Chaos harness
"""

import os
import tempfile
import unittest


class TestTruthEventStore(unittest.TestCase):
    """Event store: append, load, ordering."""

    def test_append_and_load(self):
        from argus.truth.contracts import TruthEvent
        from argus.truth.event_store import TruthEventStore

        db = os.path.join(tempfile.gettempdir(), f"argus_test_{os.getpid()}_1.db")
        try:
            store = TruthEventStore(db)
            events = [
                TruthEvent(event_id="e1", run_id="run1", cycle_ts_ms=1000,
                           event_ts_ms=1001, event_type="fill_received",
                           aggregate_type="position", aggregate_id="BTC/USD",
                           payload={"qty": 1.0}, metadata={}),
                TruthEvent(event_id="e2", run_id="run1", cycle_ts_ms=1000,
                           event_ts_ms=1002, event_type="cycle_completed",
                           aggregate_type="cycle", aggregate_id="run1",
                           payload={}, metadata={}),
            ]
            store.append(events)
            loaded = store.load_cycle("run1", 1000)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].event_id, "e1")
            self.assertEqual(loaded[1].event_id, "e2")
        finally:
            try: os.unlink(db)
            except OSError: pass

    def test_duplicate_replace(self):
        from argus.truth.contracts import TruthEvent
        from argus.truth.event_store import TruthEventStore

        db = os.path.join(tempfile.gettempdir(), f"argus_test_{os.getpid()}_2.db")
        try:
            store = TruthEventStore(db)
            e1 = TruthEvent(event_id="e1", run_id="run1", cycle_ts_ms=1000,
                            event_ts_ms=1001, event_type="fill_received",
                            aggregate_type="position", aggregate_id="BTC/USD",
                            payload={"qty": 1.0}, metadata={})
            store.append([e1])
            e1_updated = TruthEvent(event_id="e1", run_id="run1", cycle_ts_ms=1000,
                                    event_ts_ms=1001, event_type="fill_received",
                                    aggregate_type="position", aggregate_id="BTC/USD",
                                    payload={"qty": 2.0}, metadata={})
            store.append([e1_updated])
            loaded = store.load_cycle("run1", 1000)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].payload["qty"], 2.0)
        finally:
            try: os.unlink(db)
            except OSError: pass


class TestProjector(unittest.TestCase):
    """Position projection from events."""

    def test_buy_then_sell_pnl(self):
        from argus.truth.contracts import TruthEvent
        from argus.truth.projector import project_positions

        events = [
            TruthEvent(event_id="e1", run_id="r", cycle_ts_ms=0, event_ts_ms=1,
                       event_type="fill_received", aggregate_type="pos", aggregate_id="BTC",
                       payload={"symbol": "BTC/USD", "side": "buy", "qty": 1.0, "price": 50000, "fee": 5}),
            TruthEvent(event_id="e2", run_id="r", cycle_ts_ms=0, event_ts_ms=2,
                       event_type="fill_received", aggregate_type="pos", aggregate_id="BTC",
                       payload={"symbol": "BTC/USD", "side": "sell", "qty": 0.5, "price": 55000, "fee": 5}),
        ]
        positions = project_positions(events)
        btc = positions["BTC/USD"]
        self.assertAlmostEqual(btc.qty, 0.5)
        self.assertAlmostEqual(btc.realized_pnl, 2500.0)  # 0.5 * (55000-50000)
        self.assertAlmostEqual(btc.fees_paid, 10.0)

    def test_full_close_zero_qty(self):
        from argus.truth.contracts import TruthEvent
        from argus.truth.projector import project_positions

        events = [
            TruthEvent(event_id="e1", run_id="r", cycle_ts_ms=0, event_ts_ms=1,
                       event_type="fill_received", aggregate_type="pos", aggregate_id="BTC",
                       payload={"symbol": "BTC/USD", "side": "buy", "qty": 1.0, "price": 50000}),
            TruthEvent(event_id="e2", run_id="r", cycle_ts_ms=0, event_ts_ms=2,
                       event_type="fill_received", aggregate_type="pos", aggregate_id="BTC",
                       payload={"symbol": "BTC/USD", "side": "sell", "qty": 1.0, "price": 48000}),
        ]
        positions = project_positions(events)
        btc = positions["BTC/USD"]
        self.assertAlmostEqual(btc.qty, 0.0)
        self.assertAlmostEqual(btc.realized_pnl, -2000.0)
        self.assertAlmostEqual(btc.avg_cost, 0.0)


class TestHashlock(unittest.TestCase):
    """Stable hashing."""

    def test_deterministic(self):
        from argus.truth.hashlock import stable_hash
        h1 = stable_hash({"b": 2, "a": 1})
        h2 = stable_hash({"a": 1, "b": 2})
        self.assertEqual(h1, h2)

    def test_different_data(self):
        from argus.truth.hashlock import stable_hash
        h1 = stable_hash({"a": 1})
        h2 = stable_hash({"a": 2})
        self.assertNotEqual(h1, h2)


class TestReplayVerifier(unittest.TestCase):
    """Replay bundle verification."""

    def test_identical_passes(self):
        from argus.replay.contracts import CycleArtifactBundle
        from argus.replay.verifier import verify_bundle

        b = CycleArtifactBundle(run_id="r", cycle_ts_ms=1000,
                                config_hash="aaa", feature_hash="bbb",
                                regime_hash="ccc", strategy_hash="ddd",
                                allocation_hash="eee", risk_hash="fff")
        result = verify_bundle(b, b)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.mismatches), 0)

    def test_mismatch_detected(self):
        from argus.replay.contracts import CycleArtifactBundle
        from argus.replay.verifier import verify_bundle

        expected = CycleArtifactBundle(run_id="r", cycle_ts_ms=1000,
                                       config_hash="aaa", feature_hash="bbb",
                                       regime_hash="ccc", strategy_hash="ddd",
                                       allocation_hash="eee", risk_hash="fff")
        actual = CycleArtifactBundle(run_id="r", cycle_ts_ms=1000,
                                     config_hash="aaa", feature_hash="CHANGED",
                                     regime_hash="ccc", strategy_hash="ddd",
                                     allocation_hash="eee", risk_hash="fff")
        result = verify_bundle(expected, actual)
        self.assertFalse(result.passed)
        self.assertIn("feature_hash", result.mismatches)


class TestExecutionAttribution(unittest.TestCase):
    """Execution shortfall decomposition."""

    def test_buy_side_slippage(self):
        from argus.execution_analytics.shortfall import build_execution_attribution

        record = build_execution_attribution(
            strategy_id="momentum", symbol="BTC/USD", order_id="o1",
            intended_qty=1.0, filled_qty=1.0, side="buy",
            decision_price=50000, arrival_price=50010,
            avg_fill_price=50020, exit_reference_price=50100,
            fees=5.0,
        )
        self.assertGreater(record.slippage_cost, 0)  # paid more than arrival
        self.assertGreater(record.retained_alpha, 0)  # exit > decision

    def test_sell_side_slippage(self):
        from argus.execution_analytics.shortfall import build_execution_attribution

        record = build_execution_attribution(
            strategy_id="mr", symbol="ETH/USD", order_id="o2",
            intended_qty=10.0, filled_qty=10.0, side="sell",
            decision_price=2000, arrival_price=1995,
            avg_fill_price=1990, exit_reference_price=1950,
            fees=3.0,
        )
        # Selling: slippage is negative if we got worse price
        self.assertIsNotNone(record.retained_alpha)


class TestDriftGovernance(unittest.TestCase):
    """Drift detection and governance actions."""

    def test_divergence_triggers_shadow(self):
        from argus.drift.detectors import live_shadow_divergence_alert
        from argus.drift.governance import governance_from_alerts

        alert = live_shadow_divergence_alert(1000, "breakout", 0.15, 0.10)
        self.assertIsNotNone(alert)
        actions = governance_from_alerts(1000, [alert])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "move_to_shadow")

    def test_feature_shift_reduces_budget(self):
        from argus.drift.detectors import feature_mean_shift_alert
        from argus.drift.governance import governance_from_alerts

        # Shift of 1.0 with threshold 0.2 → 1.0 > 0.4 (2*threshold) → severity "high"
        alert = feature_mean_shift_alert(1000, "trend", 0.5, 1.5, 0.2)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, "high")
        actions = governance_from_alerts(1000, [alert])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "reduce_family_budget")

    def test_no_alert_below_threshold(self):
        from argus.drift.detectors import feature_mean_shift_alert

        alert = feature_mean_shift_alert(1000, "trend", 0.5, 0.55, 0.2)
        self.assertIsNone(alert)


class TestExecutivePolicy(unittest.TestCase):
    """Executive meta-controller decisions."""

    def test_replay_failure_shadow_only(self):
        from argus.executive.policy import decide_mode
        d = decide_mode(1000, [], exchange_health_ok=True, replay_ok=False)
        self.assertEqual(d.trading_mode, "shadow_only")
        self.assertAlmostEqual(d.gross_multiplier, 0.0)

    def test_exchange_degraded(self):
        from argus.executive.policy import decide_mode
        d = decide_mode(1000, [], exchange_health_ok=False, replay_ok=True)
        self.assertEqual(d.trading_mode, "no_new_risk")
        self.assertAlmostEqual(d.gross_multiplier, 0.25)

    def test_critical_alert_flatten(self):
        from argus.drift.contracts import DriftAlert
        from argus.executive.policy import decide_mode

        alert = DriftAlert(alert_id="a1", ts_ms=1000, alert_type="test",
                           severity="critical", entity_type="strategy",
                           entity_id="x", score=1.0, threshold=0.5)
        d = decide_mode(1000, [alert], exchange_health_ok=True, replay_ok=True)
        self.assertEqual(d.trading_mode, "flatten_only")

    def test_multiple_high_alerts_conservative(self):
        from argus.drift.contracts import DriftAlert
        from argus.executive.policy import decide_mode

        alerts = [
            DriftAlert(alert_id="a1", ts_ms=1000, alert_type="t", severity="high",
                       entity_type="s", entity_id="x", score=0.5, threshold=0.3),
            DriftAlert(alert_id="a2", ts_ms=1000, alert_type="t", severity="high",
                       entity_type="s", entity_id="y", score=0.6, threshold=0.3),
        ]
        d = decide_mode(1000, alerts, exchange_health_ok=True, replay_ok=True)
        self.assertEqual(d.trading_mode, "live_conservative")
        self.assertAlmostEqual(d.gross_multiplier, 0.5)

    def test_normal_full_live(self):
        from argus.executive.policy import decide_mode
        d = decide_mode(1000, [], exchange_health_ok=True, replay_ok=True)
        self.assertEqual(d.trading_mode, "full_live")
        self.assertAlmostEqual(d.gross_multiplier, 1.0)


class TestChaosInjectors(unittest.TestCase):
    """Chaos injection functions."""

    def test_duplicate_fill(self):
        from argus.chaos.injectors import duplicate_fill
        fill = {"order_id": "o1", "qty": 1.0}
        result = duplicate_fill(fill)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], result[1])

    def test_clock_skew(self):
        from argus.chaos.injectors import clock_skew
        events = [{"timestamp": 1000}, {"timestamp": 2000}]
        skewed = clock_skew(events, 500)
        self.assertEqual(skewed[0]["timestamp"], 1500)
        self.assertEqual(skewed[1]["timestamp"], 2500)

    def test_fee_anomaly(self):
        from argus.chaos.injectors import fee_anomaly
        fill = {"fee": 1.0}
        result = fee_anomaly(fill, 10.0)
        self.assertAlmostEqual(result["fee"], 10.0)


class TestConstitution(unittest.TestCase):
    """Constitutional capital enforcement."""

    def test_enforce_within_limits(self):
        from argus.executive.constitution import Constitution, enforce_capital_limits
        const = Constitution(
            version=1,
            allowed_hashes=[],
            max_safe_aum_by_regime={"liquid": 1000000, "thin": 50000},
            drawdown_deescalation={"warning": 0.03, "conservative": 0.05,
                                    "no_new_risk": 0.07, "flatten_only": 0.10},
            family_caps={"trend": 0.12, "mean_reversion": 0.08},
        )
        result = enforce_capital_limits(const, current_aum=500, liquidity_regime="liquid", drawdown_pct=0.01)
        self.assertTrue(result["allowed"])

    def test_enforce_exceeds_aum(self):
        from argus.executive.constitution import Constitution, enforce_capital_limits
        const = Constitution(
            version=1, allowed_hashes=[],
            max_safe_aum_by_regime={"liquid": 1000, "thin": 50},
            drawdown_deescalation={"warning": 0.03, "flatten_only": 0.10},
            family_caps={},
        )
        result = enforce_capital_limits(const, current_aum=2000, liquidity_regime="liquid", drawdown_pct=0.01)
        self.assertFalse(result["allowed"])

    def test_drawdown_deescalation(self):
        from argus.executive.constitution import Constitution, enforce_capital_limits
        const = Constitution(
            version=1, allowed_hashes=[],
            max_safe_aum_by_regime={"liquid": 1000000},
            drawdown_deescalation={"warning": 0.03, "conservative": 0.05,
                                    "no_new_risk": 0.07, "flatten_only": 0.10},
            family_caps={},
        )
        result = enforce_capital_limits(const, current_aum=500, liquidity_regime="liquid", drawdown_pct=0.08)
        self.assertEqual(result["mode"], "no_new_risk")


if __name__ == "__main__":
    unittest.main()
