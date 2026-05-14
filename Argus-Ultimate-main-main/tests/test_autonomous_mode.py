"""
Tests for the autonomous mode stack:
  1. CapitalTierManager — auto config switching
  2. AntifragileResponder — failure pattern detection + escalation
  3. ResourceAutoscaler — dynamic background workload scaling
  4. StrategyLifecycleScheduler — closes the autonomy loop
  5. MarketMakerStrategy / MarketMakerManager — structural income
  6. ComponentRegistry wiring — all 5 components instantiate
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# CapitalTierManager
# ──────────────────────────────────────────────────────────────────────────────

class TestCapitalTierManager(unittest.TestCase):
    def setUp(self):
        from core.capital_tier_manager import CapitalTierManager
        self.mgr = CapitalTierManager(starting_capital_aud=1000.0)

    def test_initial_tier_is_micro(self):
        from core.capital_tier_manager import CapitalTier
        self.assertEqual(self.mgr.current_tier, CapitalTier.MICRO)

    def test_micro_tier_definition(self):
        defn = self.mgr.current_definition
        self.assertEqual(defn.min_aud, 0)
        self.assertEqual(defn.max_aud, 5000)
        self.assertGreater(len(defn.enabled_strategies), 0)

    def test_check_no_transition_at_starting_capital(self):
        result = self.mgr.check_and_transition(1000.0)
        self.assertFalse(result["transitioned"])
        self.assertEqual(result["current_tier"], "micro")

    def test_promotion_requires_hysteresis_buffer(self):
        # Just at $5000 (boundary) — shouldn't promote yet (needs 10% buffer)
        for _ in range(150):  # more than confirmation cycles
            result = self.mgr.check_and_transition(5000.0)
        self.assertFalse(result["transitioned"])

    def test_promotion_requires_confirmation_cycles(self):
        from core.capital_tier_manager import CapitalTier
        # Above hysteresis threshold ($5500) but only 1 cycle
        result = self.mgr.check_and_transition(6000.0)
        self.assertFalse(result["transitioned"])
        self.assertEqual(self.mgr.current_tier, CapitalTier.MICRO)

    def test_promotion_after_confirmation(self):
        from core.capital_tier_manager import CapitalTier
        # Run enough cycles above threshold
        for _ in range(150):
            result = self.mgr.check_and_transition(6000.0)
        self.assertEqual(self.mgr.current_tier, CapitalTier.SMALL)

    def test_demotion_does_not_auto_apply(self):
        from core.capital_tier_manager import CapitalTier
        # Promote first
        for _ in range(150):
            self.mgr.check_and_transition(6000.0)
        # Wait for ramp to complete (must exceed RAMP_CYCLES + 1)
        for _ in range(400):
            self.mgr.check_and_transition(6000.0)
        # Now drop capital — shouldn't demote (tier stays SMALL)
        result = self.mgr.check_and_transition(2000.0)
        self.assertFalse(result["transitioned"])
        self.assertEqual(self.mgr.current_tier, CapitalTier.SMALL)
        # Demotion warning should fire
        self.assertTrue(result.get("demotion_warning", False))

    def test_manual_override(self):
        from core.capital_tier_manager import CapitalTier
        self.mgr.manual_override(CapitalTier.MEDIUM)
        self.assertEqual(self.mgr.current_tier, CapitalTier.MEDIUM)

    def test_ramp_progress_tracked(self):
        # Trigger promotion
        for _ in range(150):
            self.mgr.check_and_transition(6000.0)
        # Should be ramping
        self.assertTrue(self.mgr.is_ramping)
        self.assertLess(self.mgr.ramp_progress, 1.0)

    def test_snapshot_format(self):
        snap = self.mgr.snapshot()
        self.assertIn("tier", snap)
        self.assertIn("capital_aud", snap)
        self.assertIn("is_ramping", snap)
        self.assertIn("next_tier_threshold", snap)

    def test_transitions_logged(self):
        for _ in range(150):
            self.mgr.check_and_transition(6000.0)
        history = self.mgr.get_transition_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["direction"], "promote")


# ──────────────────────────────────────────────────────────────────────────────
# AntifragileResponder
# ──────────────────────────────────────────────────────────────────────────────

class TestAntifragileResponder(unittest.TestCase):
    def setUp(self):
        from core.antifragile_responder import AntifragileResponder
        self.responder = AntifragileResponder()

    def _make_loss(self, pnl=-50.0, strategy="momentum", symbol="BTC/USD", regime="HIGH_VOL"):
        return {
            "pnl": pnl,
            "source_strategy": strategy,
            "symbol": symbol,
            "side": "buy",
            "order_id": f"order_{int(time.time()*1000)}",
            "regime_label": regime,
            "confidence": 0.6,
            "size_pct": 0.20,
        }

    def test_winning_trade_no_action(self):
        result = self.responder.on_fill({"pnl": 50, "source_strategy": "test"}, 1000.0)
        self.assertEqual(result["action"], "none")

    def test_minor_loss_no_action(self):
        # 0.5% loss is below MINOR threshold
        result = self.responder.on_fill(self._make_loss(pnl=-5.0), 1000.0)
        self.assertEqual(result["action"], "none")

    def test_significant_loss_logged(self):
        result = self.responder.on_fill(self._make_loss(pnl=-50.0), 1000.0)
        self.assertNotEqual(result["action"], "none")

    def test_pattern_escalation(self):
        # Same pattern 3 times → should escalate
        for _ in range(3):
            result = self.responder.on_fill(self._make_loss(pnl=-50.0), 1000.0)
        self.assertTrue(result.get("disabled_pattern"))

    def test_different_patterns_no_escalation(self):
        # 3 different regimes → should NOT escalate
        for regime in ["HIGH_VOL", "TRENDING_UP", "RANGING"]:
            result = self.responder.on_fill(
                self._make_loss(pnl=-50.0, regime=regime), 1000.0,
            )
        self.assertFalse(result.get("disabled_pattern", False))

    def test_severity_classification(self):
        from core.antifragile_responder import FailureSeverity
        # 5% loss = SIGNIFICANT
        sev = self.responder._classify_severity(0.05)
        self.assertEqual(sev, FailureSeverity.SEVERE)

    def test_is_pattern_disabled(self):
        # Trigger escalation
        for _ in range(3):
            self.responder.on_fill(self._make_loss(pnl=-50.0), 1000.0)
        # Now check if the same pattern is blocked
        is_blocked = self.responder.is_pattern_disabled(
            "HIGH_VOL", "momentum", "BTC/USD", 0.20,
        )
        self.assertTrue(is_blocked)

    def test_pattern_reset(self):
        for _ in range(3):
            self.responder.on_fill(self._make_loss(), 1000.0)
        patterns = self.responder.get_lessons_learned()
        pid = patterns[0]["pattern_id"]
        success = self.responder.reset_pattern(pid)
        self.assertTrue(success)
        self.assertFalse(
            self.responder.is_pattern_disabled("HIGH_VOL", "momentum", "BTC/USD", 0.20),
        )

    def test_snapshot_format(self):
        self.responder.on_fill(self._make_loss(), 1000.0)
        snap = self.responder.snapshot()
        self.assertIn("total_failures", snap)
        self.assertIn("patterns_tracked", snap)
        self.assertIn("severity_breakdown", snap)


# ──────────────────────────────────────────────────────────────────────────────
# ResourceAutoscaler
# ──────────────────────────────────────────────────────────────────────────────

class TestResourceAutoscaler(unittest.TestCase):
    def setUp(self):
        from core.resource_autoscaler import ResourceAutoscaler, SystemMetrics
        # Use a mock metrics provider for deterministic tests
        self.metrics = SystemMetrics(cpu_pct=20.0, ram_pct=40.0)
        self.scaler = ResourceAutoscaler(metrics_provider=lambda: self.metrics)

    def test_initial_level(self):
        # Default starts at MODERATE
        from core.resource_autoscaler import LoadLevel
        self.assertEqual(self.scaler._current_level, LoadLevel.MODERATE)

    def test_idle_load_detected(self):
        from core.resource_autoscaler import LoadLevel
        self.metrics = type(self.metrics)(cpu_pct=20.0, ram_pct=40.0)
        self.scaler._metrics_provider = lambda: self.metrics
        result = self.scaler.update()
        self.assertEqual(result["target_level"], LoadLevel.IDLE.value)

    def test_critical_load_detected(self):
        from core.resource_autoscaler import LoadLevel, SystemMetrics
        self.scaler._metrics_provider = lambda: SystemMetrics(cpu_pct=90.0, ram_pct=95.0)
        result = self.scaler.update()
        self.assertEqual(result["target_level"], LoadLevel.CRITICAL.value)

    def test_hysteresis_prevents_immediate_apply(self):
        from core.resource_autoscaler import SystemMetrics
        self.scaler._metrics_provider = lambda: SystemMetrics(cpu_pct=20.0, ram_pct=40.0)
        # Single update — no level change yet (hysteresis)
        result = self.scaler.update()
        self.assertFalse(result["level_changed"])

    def test_hysteresis_eventually_applies(self):
        from core.resource_autoscaler import SystemMetrics, LoadLevel
        self.scaler._metrics_provider = lambda: SystemMetrics(cpu_pct=20.0, ram_pct=40.0)
        # Run enough cycles for hysteresis
        for _ in range(10):
            result = self.scaler.update()
        self.assertEqual(self.scaler._current_level, LoadLevel.IDLE)

    def test_callback_invoked_on_change(self):
        from core.resource_autoscaler import SystemMetrics
        callback = MagicMock()
        self.scaler.register_callback("gp_evolver", callback)
        self.scaler._metrics_provider = lambda: SystemMetrics(cpu_pct=20.0, ram_pct=40.0)
        # Trigger transition
        for _ in range(10):
            self.scaler.update()
        callback.assert_called()

    def test_force_level(self):
        from core.resource_autoscaler import LoadLevel
        self.scaler.force_level(LoadLevel.HEAVY)
        self.assertEqual(self.scaler._current_level, LoadLevel.HEAVY)

    def test_snapshot_format(self):
        snap = self.scaler.snapshot()
        self.assertIn("level", snap)
        self.assertIn("gp_population", snap)


# ──────────────────────────────────────────────────────────────────────────────
# StrategyLifecycleScheduler
# ──────────────────────────────────────────────────────────────────────────────

class TestStrategyLifecycleScheduler(unittest.TestCase):
    def setUp(self):
        from core.strategy_lifecycle_scheduler import StrategyLifecycleScheduler
        self.scheduler = StrategyLifecycleScheduler()

    def test_initial_state(self):
        self.assertEqual(self.scheduler._cycle_count, 0)
        snap = self.scheduler.snapshot()
        self.assertEqual(snap["stats"]["discovered"], 0)

    def test_tick_increments_cycle(self):
        result = self.scheduler.tick()
        self.assertEqual(result["cycle"], 1)

    def test_tick_with_no_components(self):
        # Should not crash even without attached components
        result = self.scheduler.tick()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["events_fired"], 0)

    def test_attach_components(self):
        evolver = MagicMock()
        promotion = MagicMock()
        self.scheduler.attach(evolver=evolver, promotion=promotion)
        self.assertIsNotNone(self.scheduler._evolver)
        self.assertIsNotNone(self.scheduler._promotion)

    def test_recent_events_format(self):
        events = self.scheduler.get_recent_events(n=10)
        self.assertIsInstance(events, list)


# ──────────────────────────────────────────────────────────────────────────────
# MarketMakerStrategy
# ──────────────────────────────────────────────────────────────────────────────

class TestMarketMakerStrategy(unittest.TestCase):
    def setUp(self):
        from strategies.market_maker import MarketMakerStrategy, MarketMakerConfig
        self.cfg = MarketMakerConfig(
            symbol="BTC/USD",
            base_spread_bps=10.0,
            quote_size_pct=0.05,
            max_inventory_pct=0.10,
        )
        self.mm = MarketMakerStrategy(self.cfg)

    def test_initial_state(self):
        self.assertTrue(self.mm.active)
        self.assertEqual(self.mm.inventory, 0.0)

    def test_compute_quote_returns_pair(self):
        quote = self.mm.compute_quote(
            mid_price=60000.0,
            bid_price=59995.0,
            ask_price=60005.0,
            volatility=0.005,
            available_capital_aud=1000.0,
        )
        self.assertIsNotNone(quote)
        self.assertLess(quote.bid_price, quote.mid_price)
        self.assertGreater(quote.ask_price, quote.mid_price)
        self.assertEqual(quote.symbol, "BTC/USD")

    def test_no_quote_when_paused(self):
        self.mm.pause()
        quote = self.mm.compute_quote(
            mid_price=60000.0, bid_price=59995.0, ask_price=60005.0,
            volatility=0.005, available_capital_aud=1000.0,
        )
        self.assertIsNone(quote)

    def test_high_vol_widens_spread(self):
        low_vol_quote = self.mm.compute_quote(
            mid_price=60000.0, bid_price=59995.0, ask_price=60005.0,
            volatility=0.001, available_capital_aud=1000.0,
        )
        # Reset state to get a fresh quote
        self.mm.last_quote = None
        high_vol_quote = self.mm.compute_quote(
            mid_price=60000.0, bid_price=59995.0, ask_price=60005.0,
            volatility=0.05, available_capital_aud=1000.0,
        )
        self.assertGreater(high_vol_quote.spread_bps, low_vol_quote.spread_bps)

    def test_inventory_skew_applied(self):
        # Build up long inventory
        self.mm.inventory = 0.001  # 0.001 BTC long

        quote = self.mm.compute_quote(
            mid_price=60000.0, bid_price=59995.0, ask_price=60005.0,
            volatility=0.005, available_capital_aud=1000.0,
        )
        # With long inventory, skew should be positive (encourages selling)
        self.assertNotEqual(quote.inventory_skew, 0.0)

    def test_on_fill_updates_inventory(self):
        result = self.mm.on_fill(side="buy", price=60000.0, qty=0.001)
        self.assertEqual(self.mm.inventory, 0.001)
        self.assertEqual(self.mm.stats.fills_buy, 1)

    def test_on_fill_sell_reduces_inventory(self):
        self.mm.inventory = 0.002
        result = self.mm.on_fill(side="sell", price=60000.0, qty=0.001)
        self.assertAlmostEqual(self.mm.inventory, 0.001)

    def test_pause_resume(self):
        self.mm.pause()
        self.assertFalse(self.mm.active)
        self.mm.resume()
        self.assertTrue(self.mm.active)

    def test_snapshot_format(self):
        snap = self.mm.snapshot()
        self.assertIn("symbol", snap)
        self.assertIn("inventory", snap)
        self.assertIn("net_pnl_aud", snap)


class TestMarketMakerManager(unittest.TestCase):
    def setUp(self):
        from strategies.market_maker import MarketMakerManager
        self.mgr = MarketMakerManager(
            symbols=["BTC/USD", "ETH/USD"],
            total_capital_pct=0.40,
        )

    def test_two_strategies_active(self):
        self.assertEqual(len(self.mgr._mms), 2)

    def test_get_strategy(self):
        btc_mm = self.mgr.get_strategy("BTC/USD")
        self.assertIsNotNone(btc_mm)
        self.assertIsNone(self.mgr.get_strategy("UNKNOWN/USD"))

    def test_compute_all_quotes(self):
        market_data = {
            "BTC/USD": {"mid": 60000, "bid": 59995, "ask": 60005, "volatility": 0.005},
            "ETH/USD": {"mid": 3000, "bid": 2999, "ask": 3001, "volatility": 0.005},
        }
        quotes = self.mgr.compute_all_quotes(market_data, 1000.0)
        self.assertEqual(len(quotes), 2)

    def test_snapshot_format(self):
        snap = self.mgr.snapshot()
        self.assertIn("total_count", snap)
        self.assertIn("by_symbol", snap)
        self.assertEqual(snap["total_count"], 2)


# ──────────────────────────────────────────────────────────────────────────────
# ComponentRegistry wiring
# ──────────────────────────────────────────────────────────────────────────────

class TestAutonomyWiring(unittest.TestCase):
    """Verify all 5 autonomy components are wired into ComponentRegistry."""

    def setUp(self):
        from core.component_registry import ComponentRegistry
        self.reg = ComponentRegistry(config=MagicMock())

    def test_capital_tier_manager_slot_exists(self):
        self.assertTrue(hasattr(self.reg, "capital_tier_manager"))

    def test_antifragile_responder_slot_exists(self):
        self.assertTrue(hasattr(self.reg, "antifragile_responder"))

    def test_resource_autoscaler_slot_exists(self):
        self.assertTrue(hasattr(self.reg, "resource_autoscaler"))

    def test_strategy_lifecycle_scheduler_slot_exists(self):
        self.assertTrue(hasattr(self.reg, "strategy_lifecycle_scheduler"))

    def test_market_maker_manager_slot_exists(self):
        self.assertTrue(hasattr(self.reg, "market_maker_manager"))

    def test_init_methods_exist(self):
        self.assertTrue(hasattr(self.reg, "_init_capital_tier_manager"))
        self.assertTrue(hasattr(self.reg, "_init_antifragile_responder"))
        self.assertTrue(hasattr(self.reg, "_init_resource_autoscaler"))
        self.assertTrue(hasattr(self.reg, "_init_strategy_lifecycle_scheduler"))
        self.assertTrue(hasattr(self.reg, "_init_market_maker_manager"))

    def test_init_capital_tier_manager_works(self):
        from unittest.mock import MagicMock
        self.reg.config = MagicMock()
        self.reg.config.starting_capital_aud = 1000.0
        self.reg._init_capital_tier_manager()
        self.assertIsNotNone(self.reg.capital_tier_manager)

    def test_init_antifragile_responder_works(self):
        self.reg._init_antifragile_responder()
        self.assertIsNotNone(self.reg.antifragile_responder)

    def test_init_resource_autoscaler_works(self):
        self.reg._init_resource_autoscaler()
        self.assertIsNotNone(self.reg.resource_autoscaler)

    def test_init_strategy_lifecycle_scheduler_works(self):
        self.reg.config = MagicMock()
        self.reg._init_strategy_lifecycle_scheduler()
        self.assertIsNotNone(self.reg.strategy_lifecycle_scheduler)

    def test_init_market_maker_manager_works(self):
        self.reg.config = MagicMock()
        self.reg.config.market_maker = {
            "symbols": ["BTC/USD"],
            "total_capital_pct": 0.40,
            "base_spread_bps": 8.0,
        }
        self.reg._init_market_maker_manager()
        self.assertIsNotNone(self.reg.market_maker_manager)


# ──────────────────────────────────────────────────────────────────────────────
# Tier profile YAMLs
# ──────────────────────────────────────────────────────────────────────────────

class TestTierProfiles(unittest.TestCase):
    """Verify all 4 tier profile YAML files exist and are valid."""

    def test_micro_profile_exists(self):
        from pathlib import Path
        path = Path("config/profiles/tier_micro.yaml")
        self.assertTrue(path.exists())

    def test_small_profile_exists(self):
        from pathlib import Path
        path = Path("config/profiles/tier_small.yaml")
        self.assertTrue(path.exists())

    def test_medium_profile_exists(self):
        from pathlib import Path
        path = Path("config/profiles/tier_medium.yaml")
        self.assertTrue(path.exists())

    def test_large_profile_exists(self):
        from pathlib import Path
        path = Path("config/profiles/tier_large.yaml")
        self.assertTrue(path.exists())

    def test_micro_profile_valid_yaml(self):
        import yaml
        with open("config/profiles/tier_micro.yaml") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("tier", cfg)
        self.assertEqual(cfg["tier"]["name"], "micro")
        self.assertEqual(cfg["max_concurrent_positions"], 5)

    def test_small_profile_has_market_maker(self):
        import yaml
        with open("config/profiles/tier_small.yaml") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("market_maker", cfg)
        self.assertTrue(cfg["market_maker"]["enabled"])

    def test_medium_profile_enables_hft(self):
        import yaml
        with open("config/profiles/tier_medium.yaml") as f:
            cfg = yaml.safe_load(f)
        self.assertTrue(cfg.get("hft_enabled"))

    def test_large_profile_enables_fpga(self):
        import yaml
        with open("config/profiles/tier_large.yaml") as f:
            cfg = yaml.safe_load(f)
        self.assertTrue(cfg.get("fpga_enabled"))


# ──────────────────────────────────────────────────────────────────────────────
# Config registration
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigKeysRegistered(unittest.TestCase):
    """Verify autonomous_mode and market_maker keys are registered in config_manager."""

    def test_autonomous_mode_key_registered(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        self.assertIn("autonomous_mode", _KNOWN_TOP_LEVEL_KEYS)

    def test_market_maker_key_registered(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        self.assertIn("market_maker", _KNOWN_TOP_LEVEL_KEYS)


if __name__ == "__main__":
    unittest.main()
