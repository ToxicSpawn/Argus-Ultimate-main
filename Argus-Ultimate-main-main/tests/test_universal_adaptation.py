"""
Tests for the universal adaptation stack — every parameter in ARGUS adapts:
  1. UniversalParameterRegistry — central registry with auto-discovery
  2. ClusterManager — joint adaptation of correlated params
  3. ParameterDependencyGraph — constraint enforcement
  4. ParameterAttributionTracker — per-param P&L attribution
  5. HierarchicalHealthMonitor — multi-level health tracking
  6. UniversalAdaptationEngine — master coordinator
  7. ComponentRegistry wiring
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock


# ──────────────────────────────────────────────────────────────────────────────
# UniversalParameterRegistry
# ──────────────────────────────────────────────────────────────────────────────

class TestUniversalParameterRegistry(unittest.TestCase):
    def setUp(self):
        from core.universal_parameter_registry import UniversalParameterRegistry
        self.reg = UniversalParameterRegistry()

    def test_register_parameter(self):
        from core.universal_parameter_registry import ParameterCategory
        ok = self.reg.register(
            "test_param", initial=0.5, min_value=0.1, max_value=0.9,
            category=ParameterCategory.SIZING,
        )
        self.assertTrue(ok)
        self.assertEqual(self.reg.get_value("test_param"), 0.5)

    def test_register_duplicate_returns_false(self):
        self.reg.register("dup", 0.5, 0.0, 1.0)
        result = self.reg.register("dup", 0.7, 0.0, 1.0)
        self.assertFalse(result)

    def test_set_value_clamps_to_bounds(self):
        self.reg.register("clamp", 0.5, 0.1, 0.9)
        self.reg.set_value("clamp", 2.0)  # above max
        self.assertEqual(self.reg.get_value("clamp"), 0.9)
        self.reg.set_value("clamp", -1.0)  # below min
        self.assertEqual(self.reg.get_value("clamp"), 0.1)

    def test_revert_resets_to_initial(self):
        self.reg.register("revert_me", 0.5, 0.1, 0.9)
        self.reg.set_value("revert_me", 0.7)
        self.reg.revert("revert_me")
        self.assertEqual(self.reg.get_value("revert_me"), 0.5)

    def test_yaml_discovery(self):
        # Use the actual unified_config.yaml
        count = self.reg.discover_from_yaml("unified_config.yaml")
        # Should find a substantial number of params
        self.assertGreater(count, 100)
        self.assertGreater(self.reg.parameter_count(), 100)

    def test_categories_populated(self):
        self.reg.discover_from_yaml("unified_config.yaml")
        snap = self.reg.snapshot()
        self.assertIn("sizing", snap["by_category"])
        self.assertIn("risk", snap["by_category"])

    def test_clusters_populated(self):
        self.reg.discover_from_yaml("unified_config.yaml")
        snap = self.reg.snapshot()
        self.assertIn("position_sizing", snap["by_cluster"])

    def test_register_many(self):
        params = [
            {"name": "p1", "initial": 0.5, "min": 0.0, "max": 1.0},
            {"name": "p2", "initial": 0.6, "min": 0.0, "max": 1.0},
            {"name": "p3", "initial": 0.7, "min": 0.0, "max": 1.0},
        ]
        count = self.reg.register_many(params)
        self.assertEqual(count, 3)

    def test_revert_all(self):
        self.reg.register("a", 0.5, 0.0, 1.0)
        self.reg.register("b", 0.5, 0.0, 1.0)
        self.reg.set_value("a", 0.8)
        self.reg.set_value("b", 0.3)
        count = self.reg.revert_all()
        self.assertEqual(count, 2)

    def test_get_state(self):
        self.reg.register("state_test", 0.5, 0.1, 0.9)
        state = self.reg.get_state("state_test")
        self.assertIsNotNone(state)
        self.assertEqual(state["current_value"], 0.5)

    def test_snapshot_format(self):
        snap = self.reg.snapshot()
        self.assertIn("total_params", snap)
        self.assertIn("by_category", snap)
        self.assertIn("by_cluster", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ClusterManager
# ──────────────────────────────────────────────────────────────────────────────

class TestClusterManager(unittest.TestCase):
    def setUp(self):
        from core.universal_parameter_registry import UniversalParameterRegistry, ParameterCategory
        from core.parameter_clusters import ClusterManager
        self.reg = UniversalParameterRegistry()
        self.reg.register("max_position_pct", 0.25, 0.10, 0.50,
                          ParameterCategory.SIZING, cluster="position_sizing")
        self.reg.register("kelly_fraction", 0.5, 0.25, 1.0,
                          ParameterCategory.SIZING, cluster="position_sizing")
        self.mgr = ClusterManager(registry=self.reg)

    def test_default_clusters_exist(self):
        clusters = self.mgr.list_clusters()
        self.assertIn("position_sizing", clusters)
        self.assertIn("stops", clusters)
        self.assertIn("risk_limits", clusters)

    def test_auto_populate_from_registry(self):
        added = self.mgr.auto_populate_from_registry()
        self.assertEqual(added, 2)

    def test_apply_multiplier(self):
        self.mgr.auto_populate_from_registry()
        count = self.mgr.apply_multiplier("position_sizing", 0.9)
        self.assertGreater(count, 0)
        # Check params actually changed
        new_pos = self.reg.get_value("max_position_pct")
        self.assertAlmostEqual(new_pos, 0.225, places=4)

    def test_revert_cluster(self):
        self.mgr.auto_populate_from_registry()
        self.mgr.apply_multiplier("position_sizing", 0.8)
        self.mgr.revert_cluster("position_sizing")
        new_pos = self.reg.get_value("max_position_pct")
        self.assertAlmostEqual(new_pos, 0.25, places=4)

    def test_multiplier_clamped_to_bounds(self):
        self.mgr.auto_populate_from_registry()
        self.mgr.apply_multiplier("position_sizing", 10.0)  # absurd
        cluster = self.mgr.get_cluster("position_sizing")
        self.assertLessEqual(cluster.multiplier, cluster.max_multiplier)

    def test_record_outcome(self):
        self.mgr.auto_populate_from_registry()
        self.mgr.record_outcome("position_sizing", 15.0, helped=True)
        cluster = self.mgr.get_cluster("position_sizing")
        self.assertEqual(cluster.helped_count, 1)

    def test_snapshot(self):
        snap = self.mgr.snapshot()
        self.assertIn("total_clusters", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ParameterDependencyGraph
# ──────────────────────────────────────────────────────────────────────────────

class TestParameterDependencyGraph(unittest.TestCase):
    def setUp(self):
        from core.parameter_dependencies import ParameterDependencyGraph
        self.graph = ParameterDependencyGraph()

    def test_default_constraints_exist(self):
        constraints = self.graph.list_constraints()
        self.assertIn("tp_gt_sl", constraints)
        self.assertIn("cvar_gt_var", constraints)

    def test_validate_passing(self):
        proposed = {"take_profit_pct": 0.030, "stop_loss_pct": 0.010}
        valid, violations = self.graph.validate(proposed)
        self.assertTrue(valid)

    def test_validate_failing(self):
        proposed = {"take_profit_pct": 0.005, "stop_loss_pct": 0.020}
        valid, violations = self.graph.validate(proposed)
        self.assertFalse(valid)
        self.assertGreater(len(violations), 0)

    def test_filter_safe_changes(self):
        current = {"take_profit_pct": 0.030, "stop_loss_pct": 0.010}
        proposed = {
            "take_profit_pct": 0.004,  # would violate (TP < SL × 1.2)
            "stop_loss_pct": 0.012,     # safe
        }
        safe = self.graph.filter_safe_changes(current, proposed)
        self.assertNotIn("take_profit_pct", safe)
        self.assertIn("stop_loss_pct", safe)

    def test_add_custom_constraint(self):
        self.graph.add_constraint(
            name="custom",
            description="custom rule",
            params_involved=["param_a"],
            check_fn=lambda v: v.get("param_a", 0) > 0,
            error_message="must be positive",
        )
        valid, _ = self.graph.validate({"param_a": -1})
        self.assertFalse(valid)

    def test_get_constraints_for_param(self):
        constraints = self.graph.get_constraints_for_param("take_profit_pct")
        self.assertIn("tp_gt_sl", constraints)

    def test_snapshot(self):
        snap = self.graph.snapshot()
        self.assertIn("total_constraints", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ParameterAttributionTracker
# ──────────────────────────────────────────────────────────────────────────────

class TestParameterAttributionTracker(unittest.TestCase):
    def setUp(self):
        from core.parameter_attribution import ParameterAttributionTracker
        self.tracker = ParameterAttributionTracker()

    def test_record_trade(self):
        self.tracker.record_trade(
            trade_id="t1",
            pnl_aud=15.0,
            parameters={"max_position_pct": 0.25, "stop_loss_pct": 0.012},
        )
        snap = self.tracker.snapshot()
        self.assertEqual(snap["trades_recorded"], 1)
        self.assertEqual(snap["params_tracked"], 2)

    def test_compute_impacts(self):
        # Need at least 20 samples
        for i in range(30):
            self.tracker.record_trade(
                trade_id=f"t{i}",
                pnl_aud=15.0 if i % 2 == 0 else -10.0,
                parameters={"max_position_pct": 0.25, "stop_loss_pct": 0.012},
            )
        impacts = self.tracker.compute_impacts()
        self.assertIn("max_position_pct", impacts)

    def test_compute_correlation(self):
        # Higher param value → higher P&L
        for i in range(30):
            self.tracker.record_trade(
                trade_id=f"t{i}",
                pnl_aud=float(i),
                parameters={"test_param": float(i)},
            )
        impacts = self.tracker.compute_impacts()
        self.assertIn("test_param", impacts)
        impact = impacts["test_param"]
        self.assertGreater(impact.correlation_with_outcome, 0.9)

    def test_top_contributors(self):
        for i in range(30):
            self.tracker.record_trade(
                trade_id=f"t{i}",
                pnl_aud=float(i),
                parameters={"strong": float(i), "weak": 0.5},
            )
        top = self.tracker.top_contributors(n=5)
        self.assertGreater(len(top), 0)

    def test_compute_cluster_impacts(self):
        for i in range(30):
            self.tracker.record_trade(
                trade_id=f"t{i}",
                pnl_aud=float(i - 15),
                parameters={"p1": 0.5},
                cluster_multipliers={"position_sizing": 1.0 + i * 0.01},
            )
        impacts = self.tracker.compute_cluster_impacts()
        self.assertIn("position_sizing", impacts)

    def test_snapshot(self):
        snap = self.tracker.snapshot()
        self.assertIn("trades_recorded", snap)


# ──────────────────────────────────────────────────────────────────────────────
# HierarchicalHealthMonitor
# ──────────────────────────────────────────────────────────────────────────────

class TestHierarchicalHealthMonitor(unittest.TestCase):
    def setUp(self):
        from core.hierarchical_health_monitor import HierarchicalHealthMonitor
        self.monitor = HierarchicalHealthMonitor()
        self.monitor.register_parameter("test_param")
        self.monitor.register_cluster("test_cluster")
        self.monitor.register_module("test_module")

    def test_register_parameter(self):
        self.assertIn("test_param", self.monitor._parameters)

    def test_record_outcome(self):
        self.monitor.record_outcome(
            pnl_aud=15.0,
            parameter_name="test_param",
            cluster_name="test_cluster",
            module_path="test_module",
        )
        # Should be tracked in all 3 entities
        self.assertEqual(self.monitor._parameters["test_param"].total_helped, 1)
        self.assertEqual(self.monitor._clusters["test_cluster"].total_helped, 1)
        self.assertEqual(self.monitor._modules["test_module"].total_helped, 1)

    def test_compute_health_with_winning_trades(self):
        for _ in range(20):
            self.monitor.record_outcome(pnl_aud=10.0, cluster_name="test_cluster")
        snap = self.monitor.compute_health("test_cluster", "cluster")
        self.assertIsNotNone(snap)
        self.assertGreater(snap.score, 50)

    def test_compute_health_with_losing_trades(self):
        for _ in range(20):
            self.monitor.record_outcome(pnl_aud=-10.0, cluster_name="test_cluster")
        snap = self.monitor.compute_health("test_cluster", "cluster")
        self.assertIsNotNone(snap)
        self.assertLess(snap.score, 50)

    def test_find_unhealthy(self):
        for _ in range(20):
            self.monitor.record_outcome(pnl_aud=-10.0, cluster_name="test_cluster")
        unhealthy = self.monitor.find_unhealthy()
        self.assertGreater(len(unhealthy), 0)

    def test_mark_reverted(self):
        ok = self.monitor.mark_reverted("test_cluster", "cluster")
        self.assertTrue(ok)

    def test_snapshot(self):
        snap = self.monitor.snapshot()
        self.assertIn("system_score", snap)
        self.assertIn("clusters_tracked", snap)


# ──────────────────────────────────────────────────────────────────────────────
# UniversalAdaptationEngine
# ──────────────────────────────────────────────────────────────────────────────

class TestUniversalAdaptationEngine(unittest.TestCase):
    def setUp(self):
        from core.universal_adaptation_engine import (
            UniversalAdaptationEngine, UniversalAdaptationConfig,
        )
        self.engine = UniversalAdaptationEngine(config=UniversalAdaptationConfig(
            enabled=True,
            cluster_adapt_cycles=1,
            health_check_cycles=1,
            yaml_path="unified_config.yaml",
        ))

    def test_bootstrap(self):
        result = self.engine.bootstrap()
        self.assertGreater(result["params_registered"], 100)
        self.assertGreater(result["clusters_defined"], 5)
        self.assertGreater(result["constraints_defined"], 5)

    def test_tick_with_bootstrap(self):
        self.engine.bootstrap()
        result = self.engine.tick(cycle_number=1, portfolio_value_aud=1000.0)
        self.assertIn("cycle", result)

    def test_observe_trade(self):
        self.engine.bootstrap()
        for i in range(50):
            self.engine.observe_trade(
                trade_id=f"t{i}",
                pnl_aud=15.0 if i % 2 == 0 else -8.0,
            )
        snap = self.engine.snapshot()
        self.assertEqual(snap["attribution"]["trades_recorded"], 50)

    def test_paused_mode(self):
        from core.universal_adaptation_engine import UniversalAdaptationMode
        self.engine.set_mode(UniversalAdaptationMode.PAUSED)
        self.engine.bootstrap()
        result = self.engine.tick(cycle_number=1, portfolio_value_aud=1000.0)
        self.assertFalse(result.get("enabled", True))

    def test_revert_all(self):
        self.engine.bootstrap()
        count = self.engine.revert_all()
        self.assertGreaterEqual(count, 0)

    def test_snapshot_format(self):
        self.engine.bootstrap()
        snap = self.engine.snapshot()
        self.assertIn("registry", snap)
        self.assertIn("clusters", snap)
        self.assertIn("dependencies", snap)
        self.assertIn("attribution", snap)
        self.assertIn("health", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ComponentRegistry wiring
# ──────────────────────────────────────────────────────────────────────────────

class TestUniversalAdaptationWiring(unittest.TestCase):
    def setUp(self):
        from core.component_registry import ComponentRegistry
        self.reg = ComponentRegistry(config=MagicMock())
        self.reg.config = MagicMock()
        self.reg.config.universal_adaptation = {
            "enabled": True,
            "mode": "normal",
            "auto_discover_yaml": True,
            "yaml_path": "unified_config.yaml",
        }

    def test_universal_adaptation_engine_slot(self):
        self.assertTrue(hasattr(self.reg, "universal_adaptation_engine"))

    def test_init_universal_adaptation_engine(self):
        self.reg._init_universal_adaptation_engine()
        self.assertIsNotNone(self.reg.universal_adaptation_engine)

    def test_bootstrap_discovers_params(self):
        self.reg._init_universal_adaptation_engine()
        snap = self.reg.universal_adaptation_engine.snapshot()
        self.assertGreater(snap["registry"]["total_params"], 100)


# ──────────────────────────────────────────────────────────────────────────────
# Config registration
# ──────────────────────────────────────────────────────────────────────────────

class TestUniversalAdaptationConfig(unittest.TestCase):
    def test_universal_adaptation_key_registered(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        self.assertIn("universal_adaptation", _KNOWN_TOP_LEVEL_KEYS)


if __name__ == "__main__":
    unittest.main()
