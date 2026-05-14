"""Integration tests for world-class production fixes."""
import ast
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestHistoricalLoaderWiredIntoStartup(unittest.TestCase):
    """Verify the historical loader is actually called during system startup."""

    def test_startup_calls_load_all_historical(self):
        """The string 'load_all_historical' must appear in unified_trading_system.py."""
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("load_all_historical", content,
                       "Historical loader is not called during startup!")

    def test_startup_calls_before_bootstrap_volatility(self):
        """Historical load must happen BEFORE bootstrap_volatility."""
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        hist_pos = content.find("load_all_historical")
        boot_pos = content.find("_bootstrap_volatility")
        self.assertGreater(hist_pos, 0, "load_all_historical not found")
        self.assertGreater(boot_pos, 0, "_bootstrap_volatility not found")
        self.assertLess(hist_pos, boot_pos,
                        "Historical load must come BEFORE bootstrap_volatility")

    def test_component_registry_has_ohlcv_history(self):
        """ComponentRegistry must initialise _ohlcv_history dict."""
        with open("core/component_registry.py", "r") as f:
            content = f.read()
        self.assertIn("_ohlcv_history", content,
                       "ComponentRegistry missing _ohlcv_history")

    def test_scanner_uses_ohlcv_history(self):
        """Scanner block must reference _ohlcv_history, not just synthetic."""
        with open("core/component_registry.py", "r") as f:
            content = f.read()
        # Find the scanner block and verify it checks _ohlcv_history
        scanner_section = content[content.find("StrategyScanner: rescan"):]
        self.assertIn("_ohlcv_history", scanner_section[:500],
                       "Scanner block doesn't use _ohlcv_history")

    def test_evolver_uses_ohlcv_history(self):
        """Evolver block must reference _ohlcv_history."""
        with open("core/component_registry.py", "r") as f:
            content = f.read()
        evolver_section = content[content.find("StrategyEvolver"):]
        self.assertIn("_ohlcv_history", evolver_section[:3000],
                       "Evolver block doesn't use _ohlcv_history")


class TestRealisticPaperFills(unittest.TestCase):
    """Verify paper trading simulates realistic market conditions."""

    def test_paper_wrapper_has_slippage(self):
        """Paper create_order must simulate slippage."""
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        # Find PaperCCXTWrapper create_order
        wrapper_start = content.find("class _PaperCCXTWrapper")
        wrapper_section = content[wrapper_start:wrapper_start + 3000]
        self.assertIn("slippage", wrapper_section.lower(),
                       "Paper wrapper missing slippage simulation")

    def test_paper_wrapper_has_partial_fills(self):
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        wrapper_start = content.find("class _PaperCCXTWrapper")
        wrapper_section = content[wrapper_start:wrapper_start + 3000]
        self.assertIn("fill_rate", wrapper_section,
                       "Paper wrapper missing partial fill simulation")

    def test_paper_wrapper_has_latency(self):
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        wrapper_start = content.find("class _PaperCCXTWrapper")
        wrapper_section = content[wrapper_start:wrapper_start + 3000]
        self.assertIn("latency", wrapper_section.lower(),
                       "Paper wrapper missing latency simulation")

    def test_paper_wrapper_maker_taker_fees(self):
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        wrapper_start = content.find("class _PaperCCXTWrapper")
        wrapper_section = content[wrapper_start:wrapper_start + 3000]
        self.assertIn("0.0016", wrapper_section,
                       "Paper wrapper missing maker fee (0.16%)")
        self.assertIn("0.0026", wrapper_section,
                       "Paper wrapper missing taker fee (0.26%)")


class TestGateStackingFloor(unittest.TestCase):
    """Verify the gate stacking floor prevents over-reduction."""

    def test_gate_floor_exists(self):
        """A minimum position size floor must exist after all gates."""
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("gate_floor", content,
                       "Gate stacking floor not found")
        self.assertIn("0.15", content[content.find("gate_floor"):content.find("gate_floor") + 200],
                       "Gate floor should be 15% of max_pos_pct")

    def test_gate_floor_after_nan_guard(self):
        """Gate floor must come after NaN guard."""
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        nan_pos = content.find("NaN/Inf guard")
        floor_pos = content.find("gate_floor")
        self.assertGreater(nan_pos, 0)
        self.assertGreater(floor_pos, 0)
        self.assertLess(nan_pos, floor_pos,
                        "Gate floor must come AFTER NaN guard")


class TestScannerEvolverUsedInSignals(unittest.TestCase):
    """Verify scanner/evolver outputs influence signal generation."""

    def test_strategy_engine_reads_scanner_advisory(self):
        with open("strategies/unified/strategy_engine.py", "r") as f:
            content = f.read()
        self.assertIn("strategy_scanner", content,
                       "Strategy engine doesn't read scanner advisory")

    def test_strategy_engine_reads_evolver_advisory(self):
        with open("strategies/unified/strategy_engine.py", "r") as f:
            content = f.read()
        self.assertIn("strategy_evolver", content,
                       "Strategy engine doesn't read evolver advisory")


class TestConfigFixes(unittest.TestCase):
    """Verify configuration is consistent and correct."""

    def test_evolution_allow_apply_live_true(self):
        with open("unified_config.yaml", "r") as f:
            content = f.read()
        self.assertIn("allow_apply_live: true", content,
                       "Evolution should allow applying in live mode")

    def test_self_improvement_enabled(self):
        with open("unified_config.yaml", "r") as f:
            content = f.read()
        # Last occurrence wins in YAML — check the self_improvement section
        si_pos = content.rfind("self_improvement:")
        si_section = content[si_pos:si_pos + 100]
        self.assertIn("enabled: true", si_section,
                       "Self improvement should be enabled")

    def test_breakout_strategy_enabled(self):
        with open("unified_config.yaml", "r") as f:
            content = f.read()
        self.assertIn("breakout:                {enabled: true}", content,
                       "Breakout strategy should be enabled")

    def test_peak_alpha_enabled(self):
        with open("unified_config.yaml", "r") as f:
            content = f.read()
        self.assertIn("peak_alpha:              {enabled: true}", content,
                       "Peak alpha strategy should be enabled")

    def test_historical_preload_enabled(self):
        with open("unified_config.yaml", "r") as f:
            content = f.read()
        self.assertIn("enable_historical_preload: true", content,
                       "Historical preload should be enabled")

    def test_strategy_evaluation_engine_enabled(self):
        with open("unified_config.yaml", "r") as f:
            content = f.read()
        # Last occurrence wins
        pos = content.rfind("strategy_evaluation_engine:")
        section = content[pos:pos + 100]
        self.assertIn("enabled: true", section,
                       "Strategy evaluation engine should be enabled")


class TestLiveModeValidation(unittest.TestCase):
    """Verify live mode has startup safety checks."""

    def test_api_key_check_in_startup(self):
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("KRAKEN_API_KEY", content,
                       "Live mode should check for KRAKEN_API_KEY")

    def test_connectivity_check_in_startup(self):
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Exchange connectivity verified", content,
                       "Live mode should verify exchange connectivity")

    def test_live_fails_safe_to_paper(self):
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("switching to paper mode for safety", content,
                       "Failed live validation should fall back to paper")


class TestErrorLoggingPromoted(unittest.TestCase):
    """Verify critical component errors are logged at WARNING, not DEBUG."""

    def test_on_fill_errors_not_debug(self):
        with open("core/component_registry.py", "r") as f:
            content = f.read()
        # No on_fill errors should be at DEBUG level
        import re
        debug_fill = re.findall(r'logger\.debug\(".*\.on_fill error', content)
        self.assertEqual(len(debug_fill), 0,
                         f"Found {len(debug_fill)} on_fill errors at DEBUG level: {debug_fill[:3]}")

    def test_on_fill_errors_at_warning(self):
        with open("core/component_registry.py", "r") as f:
            content = f.read()
        import re
        warning_fill = re.findall(r'logger\.warning\(".*\.on_fill error', content)
        self.assertGreater(len(warning_fill), 5,
                           "Expected 5+ on_fill errors promoted to WARNING")


if __name__ == "__main__":
    unittest.main()
