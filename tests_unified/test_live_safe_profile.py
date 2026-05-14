from __future__ import annotations

import unittest

from core.config_manager import load_unified_trading_config, load_unified_yaml, validate_unified_config_dict
from unified_trading_system import UnifiedSystemArchitecture


class TestLiveSafeProfile(unittest.TestCase):
    def test_live_safe_profile_validates_strictly(self) -> None:
        y = load_unified_yaml("unified_config.yaml", profile="live_safe")
        validate_unified_config_dict(y, source="unified_config.yaml + profile live_safe")

    def test_live_safe_scope_controls_disable_research_modules(self) -> None:
        cfg = load_unified_trading_config("unified_config.yaml", strict=True, profile="live_safe")
        system = UnifiedSystemArchitecture(cfg)
        self.assertTrue(system._is_live_safe_runtime())
        self.assertFalse(bool(getattr(system.config, "strategy_library_enabled", True)))
        self.assertFalse(bool(getattr(system.config, "quantum_features_enabled", True)))
        self.assertFalse(bool(getattr(system.config, "quant_fund_upgrades_enabled", True)))
        self.assertFalse(bool(getattr(system.config, "self_improvement_enabled", True)))
        self.assertFalse(bool(getattr(system.config, "self_optimizing_meta_enabled", True)))
        self.assertFalse(bool(getattr(system.config, "champion_challenger_enabled", True)))
        self.assertFalse(bool(getattr(system.config, "dynamic_universe_enabled", True)))
        self.assertFalse(bool(getattr(system.config, "adaptive_universe_enabled", True)))
        registry = system._build_runtime_module_registry()
        self.assertEqual(str((registry.get("strategy_library") or {}).get("status", "")), "disabled")
        self.assertEqual(str((registry.get("quantum_features") or {}).get("status", "")), "disabled")

    def test_live_safe_symbol_lock_restores_profile_pairs(self) -> None:
        cfg = load_unified_trading_config("unified_config.yaml", strict=True, profile="live_safe")
        system = UnifiedSystemArchitecture(cfg)
        self.assertTrue(system._is_live_safe_runtime())
        locked = list(getattr(system, "_live_safe_locked_symbols", []) or [])
        self.assertEqual(locked, ["BTC/USD", "ETH/USD"])
        setattr(system.config, "trading_pairs", ["BTC/USD", "XRP/USD", "ETH/USD"])
        system._enforce_live_safe_symbol_lock()
        self.assertEqual(list(getattr(system.config, "trading_pairs", []) or []), ["BTC/USD", "ETH/USD"])


if __name__ == "__main__":
    unittest.main()
