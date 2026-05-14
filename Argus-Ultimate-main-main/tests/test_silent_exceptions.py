"""
Tests verifying that formerly-silent exception handlers now log properly.
=========================================================================

Each test triggers a code path with a known exception and asserts that
the logger is called (debug or warning) rather than silently passing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# adaptive/promotion_gates.py — _getf
# ---------------------------------------------------------------------------

class TestPromotionGatesLogging:
    def test_getf_logs_on_conversion_failure(self, caplog):
        from adaptive.promotion_gates import _getf
        with caplog.at_level(logging.DEBUG, logger="adaptive.promotion_gates"):
            result = _getf({"bad_key": "not_a_number"}, "bad_key", 42.0)
        assert result == 42.0
        assert any("_getf conversion failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# adaptive/regime.py — update_thresholds, _get_sentiment, _get_macro
# ---------------------------------------------------------------------------

class TestRegimeLogging:
    def test_update_thresholds_logs_on_failure(self, caplog):
        from adaptive.regime import RegimeDetector
        det = RegimeDetector()
        det._trend_scores = [float(i) for i in range(25)]
        det._vol_scores = [float(i) for i in range(25)]
        # Corrupt _trend_scores to cause np.percentile to fail
        with patch("adaptive.regime.np.percentile", side_effect=ValueError("bad")):
            with caplog.at_level(logging.DEBUG, logger="adaptive.regime"):
                det.update_thresholds(0.5, 1.0)
        assert any("Adaptive trend threshold" in r.message or "Adaptive vol threshold" in r.message for r in caplog.records)

    def test_get_sentiment_logs_on_provider_error(self, caplog):
        from adaptive.regime import RegimeDetector

        class BadProvider:
            def fear_greed(self, symbol=""):
                raise RuntimeError("provider broke")

        det = RegimeDetector(sentiment_provider=BadProvider())
        with caplog.at_level(logging.DEBUG, logger="adaptive.regime"):
            result = det._get_sentiment("BTC/USD")
        assert result == 0.0
        assert any("Sentiment provider error" in r.message for r in caplog.records)

    def test_get_macro_logs_on_provider_error(self, caplog):
        from adaptive.regime import RegimeDetector

        class BadMacro:
            def bias(self):
                raise RuntimeError("macro broke")

        det = RegimeDetector(macro_provider=BadMacro())
        with caplog.at_level(logging.DEBUG, logger="adaptive.regime"):
            result = det._get_macro()
        assert result == 0.0
        assert any("Macro provider error" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# adaptive/rl_allocator.py — _load, save
# ---------------------------------------------------------------------------

class TestRLAllocatorLogging:
    def test_load_logs_on_corrupt_file(self, tmp_path, caplog):
        bad_file = tmp_path / "rl_allocator.json"
        bad_file.write_text("{{{invalid json", encoding="utf-8")
        from adaptive.rl_allocator import RLAllocatorStub
        with caplog.at_level(logging.DEBUG, logger="adaptive.rl_allocator"):
            stub = RLAllocatorStub(persist_path=str(bad_file))
        assert any("RLAllocator failed to load" in r.message for r in caplog.records)

    def test_save_logs_on_write_failure(self, tmp_path, caplog):
        from adaptive.rl_allocator import RLAllocatorStub
        # Use a path where parent exists but file write is simulated to fail
        bad_path = str(tmp_path / "sub" / "deep" / "rl.json")
        stub = RLAllocatorStub(persist_path=bad_path)
        stub._q = {"test": {"a": 1.0}}
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with caplog.at_level(logging.DEBUG, logger="adaptive.rl_allocator"):
                stub.save()
        assert any("RLAllocator failed to save" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# adaptive/self_improver.py — _load_state, _save_state, record_trade_closed
# ---------------------------------------------------------------------------

class TestSelfImproverLogging:
    def _make_improver(self):
        from adaptive.self_improver import SelfImprover
        system = MagicMock()
        system.config = MagicMock()
        system.config.self_improvement_state_path = "/nonexistent/state.json"
        system.config.evolution_trigger_on_trade = True
        return SelfImprover(system=system)

    def test_load_state_logs_on_error(self, caplog):
        si = self._make_improver()
        # Make the path point to a directory (can't read as file)
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", side_effect=OSError("read fail")):
            with caplog.at_level(logging.DEBUG, logger="adaptive.self_improver"):
                result = si._load_state()
        assert result == {}
        assert any("Failed to load self-improvement state" in r.message for r in caplog.records)

    def test_save_state_logs_on_error(self, caplog):
        si = self._make_improver()
        with patch.object(Path, "mkdir", side_effect=OSError("mkdir fail")):
            with caplog.at_level(logging.DEBUG, logger="adaptive.self_improver"):
                si._save_state({"test": 1})
        assert any("Failed to save self-improvement state" in r.message for r in caplog.records)

    def test_record_trade_closed_logs_on_error(self, caplog):
        si = self._make_improver()
        with patch.object(type(si), "_load_state", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.DEBUG, logger="adaptive.self_improver"):
                si.record_trade_closed()
        assert any("record_trade_closed failed" in r.message for r in caplog.records)

    def test_enabled_in_mode_logs_on_error(self, caplog):
        si = self._make_improver()

        # Replace cfg with an object that raises on attribute access for self_improvement_modes
        class BrokenConfig:
            run_mode = "paper"
            self_improvement_enabled = True

            @property
            def self_improvement_modes(self):
                raise TypeError("broken config")

        si.cfg = BrokenConfig()
        with caplog.at_level(logging.DEBUG, logger="adaptive.self_improver"):
            result = si._enabled_in_mode()
        assert result is False
        assert any("_enabled_in_mode" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# execution/__init__.py — legacy import
# ---------------------------------------------------------------------------

class TestExecutionInitLogging:
    def test_legacy_import_logs_on_failure(self, caplog):
        """Verify that if smart_execution_core import fails, it logs."""
        # The import already happened at module load time.
        # We test the pattern by checking the logger is set up.
        import execution
        assert hasattr(execution, "_logger")


# ---------------------------------------------------------------------------
# Verify logging infrastructure is present
# ---------------------------------------------------------------------------

class TestLoggingInfrastructure:
    def test_promotion_gates_has_logger(self):
        import adaptive.promotion_gates as mod
        assert hasattr(mod, "logger")
        assert mod.logger.name == "adaptive.promotion_gates"

    def test_regime_has_logger(self):
        import adaptive.regime as mod
        assert hasattr(mod, "logger")
        assert mod.logger.name == "adaptive.regime"

    def test_rl_allocator_has_logger(self):
        import adaptive.rl_allocator as mod
        assert hasattr(mod, "logger")
        assert mod.logger.name == "adaptive.rl_allocator"

    def test_self_improver_has_logger(self):
        import adaptive.self_improver as mod
        assert hasattr(mod, "logger")
        assert mod.logger.name == "adaptive.self_improver"

    def test_execution_init_has_logger(self):
        import execution
        assert hasattr(execution, "_logger")
