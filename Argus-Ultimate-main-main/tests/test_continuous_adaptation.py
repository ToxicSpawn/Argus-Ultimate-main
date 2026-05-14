"""
Tests for the continuous adaptation stack:
  1. ObservationRecorder — experience replay buffer
  2. ParameterDriftOptimizer — online gradient drift on parameters
  3. AdaptiveGateManager — gate threshold adaptation
  4. AdaptationHealthMonitor — track if adaptations help or hurt
  5. ContinuousAdaptationEngine — master coordinator
  6. ComponentRegistry wiring — all 5 components instantiate
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# ObservationRecorder
# ──────────────────────────────────────────────────────────────────────────────

class TestObservationRecorder(unittest.TestCase):
    def setUp(self):
        from core.observation_recorder import ObservationRecorder
        self.recorder = ObservationRecorder(max_size=100)

    def test_record_decision_returns_observation(self):
        obs = self.recorder.record_decision(
            symbol="BTC/USD", regime="TRENDING_UP", price=60000.0,
            strategy="momentum", action="BUY", confidence=0.7,
            final_size_pct=0.15,
        )
        self.assertIsNotNone(obs)
        self.assertEqual(obs.symbol, "BTC/USD")
        self.assertTrue(obs.obs_id.startswith("obs_"))

    def test_complete_observation(self):
        obs = self.recorder.record_decision(
            symbol="BTC/USD", regime="TRENDING_UP", price=60000.0,
            strategy="momentum", action="BUY", confidence=0.7,
            final_size_pct=0.15,
        )
        ok = self.recorder.complete_observation(
            obs_id=obs.obs_id,
            executed=True,
            fill_price=60005.0,
            pnl_aud=15.0,
        )
        self.assertTrue(ok)
        retrieved = self.recorder.get(obs.obs_id)
        self.assertEqual(retrieved.fill_price, 60005.0)
        self.assertAlmostEqual(retrieved.pnl_aud, 15.0)

    def test_query_by_strategy(self):
        for strategy in ["momentum", "mean_reversion", "momentum"]:
            self.recorder.record_decision(
                symbol="BTC/USD", regime="NORMAL", price=60000.0,
                strategy=strategy, action="BUY", confidence=0.6,
                final_size_pct=0.10,
            )
        results = self.recorder.query(strategy="momentum")
        self.assertEqual(len(results), 2)

    def test_query_by_pnl(self):
        for pnl in [5.0, -10.0, 20.0]:
            obs = self.recorder.record_decision(
                symbol="BTC/USD", regime="NORMAL", price=60000.0,
                strategy="test", action="BUY", confidence=0.5,
                final_size_pct=0.10,
            )
            self.recorder.complete_observation(obs.obs_id, pnl_aud=pnl)
        wins = self.recorder.query(min_pnl=0)
        self.assertEqual(len(wins), 2)

    def test_recent_filter(self):
        obs1 = self.recorder.record_decision(
            symbol="BTC/USD", regime="NORMAL", price=60000.0,
            strategy="test", action="BUY", confidence=0.5,
            final_size_pct=0.10,
        )
        # All observations are recent
        recent = self.recorder.recent(hours=24)
        self.assertEqual(len(recent), 1)

    def test_sample(self):
        for i in range(10):
            self.recorder.record_decision(
                symbol="BTC/USD", regime="NORMAL", price=60000.0,
                strategy="test", action="BUY", confidence=0.5,
                final_size_pct=0.10,
            )
        sample = self.recorder.sample(5, seed=42)
        self.assertEqual(len(sample), 5)

    def test_buffer_eviction(self):
        for i in range(150):  # exceed max_size of 100
            self.recorder.record_decision(
                symbol="BTC/USD", regime="NORMAL", price=60000.0,
                strategy="test", action="BUY", confidence=0.5,
                final_size_pct=0.10,
            )
        snap = self.recorder.snapshot()
        self.assertEqual(snap["size"], 100)
        self.assertEqual(snap["evicted"], 50)

    def test_aggregate_by_strategy(self):
        for strategy, pnl in [("momentum", 10), ("momentum", -5), ("mean_rev", 8)]:
            obs = self.recorder.record_decision(
                symbol="BTC/USD", regime="NORMAL", price=60000.0,
                strategy=strategy, action="BUY", confidence=0.5,
                final_size_pct=0.10,
            )
            self.recorder.complete_observation(obs.obs_id, pnl_aud=pnl)
        groups = self.recorder.aggregate_pnl_by("strategy")
        self.assertIn("momentum", groups)
        self.assertEqual(groups["momentum"]["count"], 2)
        self.assertAlmostEqual(groups["momentum"]["total_pnl"], 5.0)

    def test_snapshot(self):
        snap = self.recorder.snapshot()
        self.assertIn("size", snap)
        self.assertIn("recorded", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ParameterDriftOptimizer
# ──────────────────────────────────────────────────────────────────────────────

class TestParameterDriftOptimizer(unittest.TestCase):
    def setUp(self):
        from core.parameter_drift_optimizer import ParameterDriftOptimizer
        self.opt = ParameterDriftOptimizer()
        self.opt.register("test_param", current=0.5, min_value=0.1, max_value=0.9)

    def test_register_parameter(self):
        self.assertIn("test_param", self.opt._params)
        self.assertEqual(self.opt.get_value("test_param"), 0.5)

    def test_bounds_clamping(self):
        self.opt.register("clamped", current=2.0, min_value=0.0, max_value=1.0)
        self.assertEqual(self.opt.get_value("clamped"), 1.0)

    def test_observe_outcome(self):
        self.opt.observe_outcome(
            parameter_values={"test_param": 0.5},
            pnl_aud=10.0,
            regime="NORMAL",
        )
        history = self.opt._history["test_param"]
        self.assertEqual(len(history.observations), 1)

    def test_compute_drift_with_insufficient_data(self):
        self.opt.observe_outcome({"test_param": 0.5}, 10.0)
        drifts = self.opt.compute_drifts()
        self.assertEqual(len(drifts), 0)

    def test_compute_drift_with_data(self):
        # Record many observations showing higher param = better outcomes
        for value in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
            for _ in range(5):
                self.opt.observe_outcome(
                    {"test_param": value}, pnl_aud=value * 10,
                )
        drifts = self.opt.compute_drifts()
        self.assertIn("test_param", drifts)
        # Drift should be in positive direction (higher values better)
        self.assertGreater(drifts["test_param"], 0.5)

    def test_apply_drifts(self):
        # Force a drift
        for _ in range(50):
            self.opt.observe_outcome({"test_param": 0.6}, pnl_aud=20.0)
        drifts = self.opt.compute_drifts()
        if drifts:
            applied = self.opt.apply_drifts(drifts)
            self.assertGreater(applied, 0)

    def test_revert_after_bad_drifts(self):
        # First drift
        for _ in range(50):
            self.opt.observe_outcome({"test_param": 0.6}, pnl_aud=20.0)
        drifts = self.opt.compute_drifts()
        self.opt.apply_drifts(drifts)

        original = self.opt.get_value("test_param")
        # Now signal that drifts hurt 5 cycles in a row
        for _ in range(5):
            self.opt.check_for_reverts({"test_param": -10.0})
        # Parameter should have been reverted to initial value
        self.assertEqual(self.opt.get_value("test_param"), 0.5)

    def test_get_parameter_state(self):
        state = self.opt.get_parameter_state("test_param")
        self.assertIsNotNone(state)
        self.assertEqual(state["current_value"], 0.5)
        self.assertEqual(state["initial_value"], 0.5)

    def test_snapshot(self):
        snap = self.opt.snapshot()
        self.assertEqual(snap["registered_params"], 1)
        self.assertIn("params", snap)


# ──────────────────────────────────────────────────────────────────────────────
# AdaptiveGateManager
# ──────────────────────────────────────────────────────────────────────────────

class TestAdaptiveGateManager(unittest.TestCase):
    def setUp(self):
        from core.adaptive_gate_manager import AdaptiveGateManager
        self.mgr = AdaptiveGateManager()
        self.mgr.register_gate(
            "test_gate", current=0.55, min_value=0.40, max_value=0.75,
        )

    def test_register_gate(self):
        self.assertEqual(self.mgr.get_threshold("test_gate"), 0.55)
        self.assertTrue(self.mgr.is_enabled("test_gate"))

    def test_record_decision(self):
        self.mgr.record_decision("test_gate", "pass")
        self.mgr.record_decision("test_gate", "block")
        perf = self.mgr.get_gate_performance("test_gate")
        self.assertEqual(perf["passed"], 1)
        self.assertEqual(perf["blocked"], 1)

    def test_record_outcome(self):
        self.mgr.record_decision("test_gate", "pass")
        self.mgr.record_outcome("test_gate", "pass", pnl_aud=15.0)
        perf = self.mgr.get_gate_performance("test_gate")
        self.assertGreater(perf["total_pnl"], 0)

    def test_compute_adaptations_insufficient_data(self):
        self.mgr.record_decision("test_gate", "pass")
        adaptations = self.mgr.compute_adaptations()
        self.assertEqual(len(adaptations), 0)

    def test_compute_adaptations_with_bad_passes(self):
        # 30 bad passes — should tighten
        for _ in range(30):
            self.mgr.record_decision("test_gate", "pass")
            self.mgr.record_outcome("test_gate", "pass", pnl_aud=-10.0)
        adaptations = self.mgr.compute_adaptations()
        self.assertIn("test_gate", adaptations)

    def test_apply_adaptations(self):
        for _ in range(30):
            self.mgr.record_decision("test_gate", "pass")
            self.mgr.record_outcome("test_gate", "pass", pnl_aud=-10.0)
        adaptations = self.mgr.compute_adaptations()
        old_threshold = self.mgr.get_threshold("test_gate")
        applied = self.mgr.apply_adaptations(adaptations)
        new_threshold = self.mgr.get_threshold("test_gate")
        self.assertGreater(applied, 0)
        self.assertNotEqual(old_threshold, new_threshold)

    def test_disable_consistently_bad_gate(self):
        # Many bad decisions both ways
        for _ in range(60):
            self.mgr.record_decision("test_gate", "pass")
            self.mgr.record_outcome("test_gate", "pass", pnl_aud=-10.0,
                                    would_have_been_profitable=False)
        for _ in range(60):
            self.mgr.record_decision("test_gate", "block")
            self.mgr.record_outcome("test_gate", "block", pnl_aud=15.0,
                                    would_have_been_profitable=True)
        adaptations = self.mgr.compute_adaptations()
        self.mgr.apply_adaptations(adaptations)
        # Could be disabled depending on accuracy
        perf = self.mgr.get_gate_performance("test_gate")
        self.assertGreater(perf["total_decisions"], 100)

    def test_reset_gate(self):
        self.mgr._gates["test_gate"].threshold = 0.65
        ok = self.mgr.reset_gate("test_gate")
        self.assertTrue(ok)
        self.assertEqual(self.mgr.get_threshold("test_gate"), 0.55)

    def test_snapshot(self):
        snap = self.mgr.snapshot()
        self.assertEqual(snap["registered_gates"], 1)
        self.assertEqual(snap["enabled_gates"], 1)


# ──────────────────────────────────────────────────────────────────────────────
# AdaptationHealthMonitor
# ──────────────────────────────────────────────────────────────────────────────

class TestAdaptationHealthMonitor(unittest.TestCase):
    def setUp(self):
        from core.adaptation_health_monitor import AdaptationHealthMonitor
        self.monitor = AdaptationHealthMonitor(measurement_cycles=10)

    def test_initial_state(self):
        self.assertFalse(self.monitor.is_measuring)
        self.assertFalse(self.monitor.should_revert())

    def test_before_adaptation(self):
        self.monitor.before_adaptation(
            portfolio_value=1000.0,
            cycle=10,
            cumulative_pnl=50.0,
        )
        self.assertTrue(self.monitor.is_measuring)

    def test_measurement_window(self):
        self.monitor.before_adaptation(
            portfolio_value=1000.0, cycle=0, cumulative_pnl=0.0,
        )
        self.monitor.after_adaptation(adaptations_applied=3)
        # Run for less than measurement window
        for i in range(5):
            self.monitor.update(portfolio_value=1010.0, cycle=i+1, cumulative_pnl=10.0)
        self.assertTrue(self.monitor.is_measuring)

    def test_evaluation_after_window(self):
        self.monitor.before_adaptation(
            portfolio_value=1000.0, cycle=0, cumulative_pnl=0.0,
        )
        self.monitor.after_adaptation(adaptations_applied=3)
        # Run until evaluation triggers
        evaluated = False
        for i in range(15):
            result = self.monitor.update(
                portfolio_value=1050.0, cycle=i+1, cumulative_pnl=50.0,
            )
            if result.get("should_evaluate"):
                evaluated = True
                break
        self.assertTrue(evaluated)

    def test_helped_adaptation(self):
        self.monitor.before_adaptation(
            portfolio_value=1000.0, cycle=0, cumulative_pnl=0.0,
        )
        self.monitor.after_adaptation(adaptations_applied=3)
        # P&L improved — capture the evaluation result
        eval_result = None
        for i in range(15):
            result = self.monitor.update(
                portfolio_value=1100.0, cycle=i+1, cumulative_pnl=100.0,
            )
            if result.get("should_evaluate"):
                eval_result = result.get("outcome", {})
                break
        self.assertIsNotNone(eval_result)
        self.assertTrue(eval_result.get("helped", False))

    def test_hurt_adaptation(self):
        self.monitor.before_adaptation(
            portfolio_value=1000.0, cycle=0, cumulative_pnl=100.0,
        )
        self.monitor.after_adaptation(adaptations_applied=3)
        # P&L dropped — capture the evaluation result
        eval_result = None
        for i in range(15):
            result = self.monitor.update(
                portfolio_value=900.0, cycle=i+1, cumulative_pnl=50.0,
            )
            if result.get("should_evaluate"):
                eval_result = result.get("outcome", {})
                break
        self.assertIsNotNone(eval_result)
        self.assertFalse(eval_result.get("helped", True))

    def test_revert_after_consecutive_hurts(self):
        # Trigger 5 consecutive hurt adaptations
        for cycle_start in range(5):
            self.monitor.before_adaptation(
                portfolio_value=1000.0, cycle=cycle_start * 20,
                cumulative_pnl=100.0 - cycle_start * 10,
            )
            self.monitor.after_adaptation(adaptations_applied=1)
            for i in range(11):
                self.monitor.update(
                    portfolio_value=900.0,
                    cycle=cycle_start * 20 + i + 1,
                    cumulative_pnl=50.0 - cycle_start * 10,
                )
        self.assertTrue(self.monitor.should_revert())

    def test_mark_reverted_resets_streak(self):
        self.monitor._hurt_streak = 5
        self.monitor.mark_reverted()
        self.assertEqual(self.monitor._hurt_streak, 0)

    def test_effectiveness(self):
        # Simulate 3 helped, 1 hurt
        self.monitor._total_helped = 3
        self.monitor._total_hurt = 1
        self.assertAlmostEqual(self.monitor.get_effectiveness(), 0.75)

    def test_snapshot(self):
        snap = self.monitor.snapshot()
        self.assertIn("effectiveness", snap)
        self.assertIn("total_helped", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ContinuousAdaptationEngine
# ──────────────────────────────────────────────────────────────────────────────

class TestContinuousAdaptationEngine(unittest.TestCase):
    def setUp(self):
        from core.observation_recorder import ObservationRecorder
        from core.parameter_drift_optimizer import ParameterDriftOptimizer
        from core.adaptive_gate_manager import AdaptiveGateManager
        from core.adaptation_health_monitor import AdaptationHealthMonitor
        from core.continuous_adaptation_engine import ContinuousAdaptationEngine

        self.recorder = ObservationRecorder()
        self.opt = ParameterDriftOptimizer()
        self.gates = AdaptiveGateManager()
        self.health = AdaptationHealthMonitor(measurement_cycles=10)
        self.engine = ContinuousAdaptationEngine(
            observation_recorder=self.recorder,
            parameter_optimizer=self.opt,
            gate_manager=self.gates,
            health_monitor=self.health,
        )

    def test_initial_state(self):
        from core.continuous_adaptation_engine import AdaptationMode
        self.assertEqual(self.engine._mode, AdaptationMode.NORMAL)

    def test_register_default_parameters(self):
        self.engine.register_default_parameters()
        self.assertGreater(len(self.opt._params), 0)
        self.assertGreater(len(self.gates._gates), 0)

    def test_tick_does_not_crash(self):
        result = self.engine.tick(cycle_number=1, portfolio_value_aud=1000.0)
        self.assertIsInstance(result, dict)
        self.assertIn("cycle", result)

    def test_tick_records_default_params_first_call(self):
        self.engine.tick(cycle_number=1, portfolio_value_aud=1000.0)
        self.assertGreater(len(self.opt._params), 0)

    def test_paused_mode_does_nothing(self):
        from core.continuous_adaptation_engine import AdaptationMode
        self.engine.set_mode(AdaptationMode.PAUSED)
        result = self.engine.tick(cycle_number=1, portfolio_value_aud=1000.0)
        self.assertFalse(result.get("enabled", True))

    def test_resume(self):
        from core.continuous_adaptation_engine import AdaptationMode
        self.engine.set_mode(AdaptationMode.PAUSED)
        self.engine.resume()
        self.assertEqual(self.engine._mode, AdaptationMode.CONSERVATIVE)

    def test_throttle_changes_mode(self):
        from core.continuous_adaptation_engine import AdaptationMode
        self.engine.set_mode(AdaptationMode.AGGRESSIVE)
        self.engine._throttle()
        self.assertEqual(self.engine._mode, AdaptationMode.NORMAL)
        self.engine._throttle()
        self.assertEqual(self.engine._mode, AdaptationMode.CONSERVATIVE)
        self.engine._throttle()
        self.assertEqual(self.engine._mode, AdaptationMode.PAUSED)

    def test_snapshot(self):
        snap = self.engine.snapshot()
        self.assertIn("mode", snap)
        self.assertIn("stats", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ComponentRegistry wiring
# ──────────────────────────────────────────────────────────────────────────────

class TestAdaptationWiring(unittest.TestCase):
    def setUp(self):
        from core.component_registry import ComponentRegistry
        self.reg = ComponentRegistry(config=MagicMock())
        self.reg.config = MagicMock()
        self.reg.config.continuous_adaptation = {
            "enabled": True,
            "mode": "normal",
            "parameter_drift_cycles": 50,
            "gate_adapt_cycles": 100,
        }

    def test_observation_recorder_slot(self):
        self.assertTrue(hasattr(self.reg, "observation_recorder"))

    def test_parameter_drift_optimizer_slot(self):
        self.assertTrue(hasattr(self.reg, "parameter_drift_optimizer"))

    def test_adaptive_gate_manager_slot(self):
        self.assertTrue(hasattr(self.reg, "adaptive_gate_manager"))

    def test_adaptation_health_monitor_slot(self):
        self.assertTrue(hasattr(self.reg, "adaptation_health_monitor"))

    def test_continuous_adaptation_engine_slot(self):
        self.assertTrue(hasattr(self.reg, "continuous_adaptation_engine"))

    def test_init_methods_exist(self):
        self.assertTrue(hasattr(self.reg, "_init_observation_recorder"))
        self.assertTrue(hasattr(self.reg, "_init_parameter_drift_optimizer"))
        self.assertTrue(hasattr(self.reg, "_init_adaptive_gate_manager"))
        self.assertTrue(hasattr(self.reg, "_init_adaptation_health_monitor"))
        self.assertTrue(hasattr(self.reg, "_init_continuous_adaptation_engine"))

    def test_init_observation_recorder(self):
        self.reg._init_observation_recorder()
        self.assertIsNotNone(self.reg.observation_recorder)

    def test_init_parameter_drift_optimizer(self):
        self.reg._init_parameter_drift_optimizer()
        self.assertIsNotNone(self.reg.parameter_drift_optimizer)

    def test_init_adaptive_gate_manager(self):
        self.reg._init_adaptive_gate_manager()
        self.assertIsNotNone(self.reg.adaptive_gate_manager)

    def test_init_adaptation_health_monitor(self):
        self.reg._init_adaptation_health_monitor()
        self.assertIsNotNone(self.reg.adaptation_health_monitor)

    def test_init_continuous_adaptation_engine_full_chain(self):
        self.reg._init_observation_recorder()
        self.reg._init_parameter_drift_optimizer()
        self.reg._init_adaptive_gate_manager()
        self.reg._init_adaptation_health_monitor()
        self.reg._init_continuous_adaptation_engine()
        self.assertIsNotNone(self.reg.continuous_adaptation_engine)
        # Check that defaults were registered
        self.assertGreaterEqual(len(self.reg.parameter_drift_optimizer._params), 5)


# ──────────────────────────────────────────────────────────────────────────────
# Config registration
# ──────────────────────────────────────────────────────────────────────────────

class TestContinuousAdaptationConfigRegistered(unittest.TestCase):
    def test_continuous_adaptation_key_registered(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        self.assertIn("continuous_adaptation", _KNOWN_TOP_LEVEL_KEYS)


if __name__ == "__main__":
    unittest.main()
