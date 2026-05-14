"""
Tests for the unified_trading_system.py refactoring:
  1. File header fix (no argus_live wrapper)
  2. Extracted sub-methods from _execute_signals
  3. _safe_call helper in component_registry
  4. Structured cycle metrics
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Header fix tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHeaderFix(unittest.TestCase):
    """Verify the argus_live wrapper was removed."""

    def test_file_starts_with_shebang(self):
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        self.assertEqual(first_line, "#!/usr/bin/env python3")

    def test_no_argus_live_imports_at_top(self):
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            first_20 = [f.readline() for _ in range(20)]
        text = "".join(first_20)
        self.assertNotIn("from argus_live", text)
        self.assertNotIn("UnifiedTradingInput", text)

    def test_no_duplicate_class_definition(self):
        """Only UnifiedSystemArchitecture should exist, not a wrapper UnifiedTradingSystem."""
        with open("unified_trading_system.py", "r", encoding="utf-8") as f:
            content = f.read()
        # The wrapper had: class UnifiedTradingSystem: """Legacy compatibility wrapper
        self.assertNotIn("Legacy compatibility wrapper", content[:5000])


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Extracted helper functions
# ──────────────────────────────────────────────────────────────────────────────

class TestPreExecuteContext(unittest.TestCase):
    """Test _pre_execute_context builds correct context dict."""

    def _make_self(self):
        mock = MagicMock()
        mock.config.run_mode = "paper"
        mock.config.aud_to_usd = 0.65
        mock.portfolio_value_aud = 1000.0
        mock.component_registry = None
        mock.unified_risk_manager = None
        mock._last_cycle_advisory = {}
        mock._latest_regime_label = "NORMAL"
        mock.REGIME_POSITION_SCALE = {"NORMAL": 1.0, "CRISIS": 0.3}
        mock.REGIME_STOP_SCALE = {"NORMAL": 1.0}
        mock.REGIME_TP_SCALE = {"NORMAL": 1.0}
        return mock

    def test_returns_context_dict(self):
        from core.execute_signals_helpers import _pre_execute_context
        mock = self._make_self()
        ctx = _pre_execute_context(mock)
        self.assertIn("regime", ctx)
        self.assertIn("session_mult", ctx)
        self.assertIn("portfolio_value", ctx)
        self.assertIn("_cycle_advisory", ctx)
        self.assertEqual(ctx["mode"], "paper")
        self.assertFalse(ctx["is_live"])

    def test_regime_defaults_to_normal(self):
        from core.execute_signals_helpers import _pre_execute_context
        mock = self._make_self()
        mock._latest_regime_label = ""
        mock._compute_fallback_regime = MagicMock(return_value="NORMAL")
        ctx = _pre_execute_context(mock)
        self.assertEqual(ctx["regime"], "NORMAL")


class TestExtractSignalFields(unittest.TestCase):
    """Test _extract_signal_fields extracts correctly."""

    def _make_signal(self, **overrides):
        sig = MagicMock()
        sig.symbol = "BTC/USD"
        sig.action = "BUY"
        sig.confidence = 0.8
        sig.strength = 0.7
        sig.entry_price = 60000.0
        sig.stop_loss = 58000.0
        sig.take_profit = 65000.0
        sig.reasoning = "momentum"
        sig.strategy = "momentum"
        sig.source_strategy = "momentum"
        sig.timestamp = None
        sig.num_confirmations = 3
        for k, v in overrides.items():
            setattr(sig, k, v)
        return sig

    def test_extracts_basic_fields(self):
        from core.execute_signals_helpers import _extract_signal_fields
        mock_self = MagicMock()
        sig = self._make_signal()
        fields = _extract_signal_fields(mock_self, sig)
        self.assertIsNotNone(fields)
        self.assertEqual(fields["symbol"], "BTC/USD")
        self.assertEqual(fields["action"], "BUY")
        self.assertAlmostEqual(fields["confidence"], 0.8, places=1)

    def test_returns_none_for_hold(self):
        from core.execute_signals_helpers import _extract_signal_fields
        sig = self._make_signal(action="HOLD")
        fields = _extract_signal_fields(MagicMock(), sig)
        self.assertIsNone(fields)

    def test_stale_signal_blocked(self):
        import time
        from core.execute_signals_helpers import _extract_signal_fields
        sig = self._make_signal(timestamp=time.time() - 200)  # 200s old
        fields = _extract_signal_fields(MagicMock(), sig)
        self.assertIsNotNone(fields)
        self.assertTrue(fields.get("_blocked"))


class TestApplyRiskGates(unittest.TestCase):
    """Test _apply_risk_gates correctly blocks signals."""

    def _make_ctx(self, **overrides):
        ctx = {
            "daily_loss_exceeded": False,
            "macro_event_imminent": False,
            "macro_event_name": "",
            "macro_event_hours": None,
            "var_breach": False,
            "aud_to_usd": 0.65,
            "portfolio_value": 1000.0,
            "regime": "NORMAL",
        }
        ctx.update(overrides)
        return ctx

    def _make_fields(self, **overrides):
        fields = {
            "symbol": "BTC/USD",
            "action": "BUY",
            "confidence": 0.8,
            "strength": 0.7,
            "entry_price": 60000.0,
            "source_strategy": "momentum",
            "_sig_obj": MagicMock(),
            "_num_confirmations": 3,
        }
        fields.update(overrides)
        return fields

    def test_daily_loss_blocks_buy(self):
        from core.execute_signals_helpers import _apply_risk_gates
        mock_self = MagicMock()
        mock_self._strategy_state_store = None
        mock_self.config.max_concurrent_positions = 5
        mock_self.positions = {}
        mock_self.unified_risk_manager = None
        mock_self.component_registry = None
        mock_self.peak_equity_aud = 1000.0
        mock_self.portfolio_value_aud = 950.0
        ctx = self._make_ctx(daily_loss_exceeded=True)
        approved, reason = _apply_risk_gates(mock_self, self._make_fields(), ctx)
        self.assertFalse(approved)
        self.assertIn("daily_loss", reason)

    def test_sell_passes_daily_loss(self):
        from core.execute_signals_helpers import _apply_risk_gates
        mock_self = MagicMock()
        mock_self._strategy_state_store = None
        mock_self.config.max_concurrent_positions = 5
        mock_self.positions = {}
        mock_self.unified_risk_manager = None
        mock_self.component_registry = None
        mock_self.peak_equity_aud = 1000.0
        mock_self.portfolio_value_aud = 950.0
        mock_self._get_current_vol.return_value = 0.005
        ctx = self._make_ctx(daily_loss_exceeded=True)
        fields = self._make_fields(action="SELL")
        approved, reason = _apply_risk_gates(mock_self, fields, ctx)
        self.assertTrue(approved)

    def test_macro_event_blocks_buy(self):
        from core.execute_signals_helpers import _apply_risk_gates
        mock_self = MagicMock()
        mock_self._strategy_state_store = None
        mock_self.config.max_concurrent_positions = 5
        mock_self.positions = {}
        mock_self.unified_risk_manager = None
        mock_self.component_registry = None
        mock_self.peak_equity_aud = 1000.0
        mock_self.portfolio_value_aud = 950.0
        ctx = self._make_ctx(macro_event_imminent=True, macro_event_name="FOMC")
        approved, reason = _apply_risk_gates(mock_self, self._make_fields(), ctx)
        self.assertFalse(approved)
        self.assertIn("macro", reason)


class TestComputePositionSize(unittest.TestCase):
    """Test _compute_position_size returns reasonable sizes."""

    def test_basic_sizing(self):
        from core.execute_signals_helpers import _compute_position_size
        mock_self = MagicMock()
        mock_self.component_registry = None
        mock_self.config.max_position_pct = 0.25
        mock_self.config.min_position_size_aud = 10.0
        mock_self._get_strategy_trade_stats.return_value = {"n_trades": 0, "win_rate": 0.5, "avg_win": 0, "avg_loss": 0}
        mock_self._get_current_vol.return_value = 0.005
        mock_self._vol_adjusted_size.side_effect = lambda s, v: s  # identity
        mock_self._get_signal_quality.return_value = None
        mock_self._get_strategy_multiplier.return_value = 1.0
        mock_self.positions = {}
        mock_self.peak_equity_aud = 1000.0
        mock_self.portfolio_value_aud = 1000.0
        mock_self._strategy_state_store = None
        mock_self._price_history = {}

        sig_fields = {
            "symbol": "BTC/USD", "action": "BUY", "confidence": 0.8,
            "strength": 0.7, "entry_price": 60000.0, "source_strategy": "momentum",
            "_num_confirmations": 3,
        }
        ctx = {
            "regime": "NORMAL",
            "regime_pos_mult": 1.0, "session_mult": 1.0,
            "macro_event_imminent": False,
            "aud_to_usd": 0.65, "portfolio_value": 1000.0,
        }
        size_pct, method = _compute_position_size(mock_self, sig_fields, ctx)
        self.assertGreater(size_pct, 0)
        self.assertLessEqual(size_pct, 0.25)
        self.assertIsInstance(method, str)


class TestApplyIntelligenceGates(unittest.TestCase):
    """Test _apply_intelligence_gates with empty advisory."""

    def test_empty_advisory_no_change(self):
        from core.execute_signals_helpers import _apply_intelligence_gates
        mock_self = MagicMock()
        mock_self.component_registry = None
        mock_self.positions = {}
        mock_self._mtf_bias = None
        mock_self._latest_regime_label = "NORMAL"
        mock_self.config.max_position_pct = 0.25
        mock_self.portfolio_value_aud = 1000.0
        mock_self.config.aud_to_usd = 0.65

        sig_fields = {"symbol": "BTC/USD", "action": "BUY", "source_strategy": "momentum", "entry_price": 60000.0}
        size_pct, method = _apply_intelligence_gates(mock_self, sig_fields, 0.10, {}, "default")
        self.assertGreater(size_pct, 0)
        self.assertFalse(method.startswith("BLOCKED:"))

    def test_system_status_critical_blocks_buy(self):
        from core.execute_signals_helpers import _apply_intelligence_gates
        mock_self = MagicMock()
        mock_self.component_registry = None
        mock_self.positions = {}
        mock_self._mtf_bias = None
        mock_self._latest_regime_label = "NORMAL"
        mock_self.config.max_position_pct = 0.25
        mock_self.portfolio_value_aud = 1000.0
        mock_self.config.aud_to_usd = 0.65

        advisory = {"system_status": {"status": "CRITICAL"}}
        sig_fields = {"symbol": "BTC/USD", "action": "BUY", "source_strategy": "momentum", "entry_price": 60000.0}
        size_pct, method = _apply_intelligence_gates(mock_self, sig_fields, 0.10, advisory, "default")
        self.assertTrue(method.startswith("BLOCKED:") or size_pct == 0.0)


class TestLogCycleSummary(unittest.TestCase):
    """Test _log_cycle_summary produces structured output."""

    def test_summary_with_results(self):
        from core.execute_signals_helpers import _log_cycle_summary
        mock_self = MagicMock()
        mock_self._cycle_number = 42
        results = [
            {"status": "filled", "pnl": 5.0, "symbol": "BTC/USD"},
            {"status": "blocked", "reason": "daily_loss"},
            {"status": "blocked", "reason": "macro_event"},
        ]
        ctx = {"regime": "NORMAL"}
        # Should not raise
        _log_cycle_summary(mock_self, results, ctx)

    def test_summary_empty_results(self):
        from core.execute_signals_helpers import _log_cycle_summary
        mock_self = MagicMock()
        mock_self._cycle_number = 0
        _log_cycle_summary(mock_self, [], {"regime": "UNKNOWN"})


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: _safe_call helper
# ──────────────────────────────────────────────────────────────────────────────

class TestSafeCall(unittest.TestCase):
    """Test _safe_call helper in ComponentRegistry."""

    def setUp(self):
        from core.component_registry import ComponentRegistry
        self.reg = ComponentRegistry(config=MagicMock())

    def test_returns_none_when_component_missing(self):
        result = self.reg._safe_call("nonexistent_component", lambda c: c.do_thing())
        self.assertIsNone(result)

    def test_calls_function_when_component_exists(self):
        self.reg.fill_tracker = MagicMock()
        self.reg.fill_tracker.get_count.return_value = 42
        result = self.reg._safe_call("fill_tracker", lambda ft: ft.get_count())
        self.assertEqual(result, 42)

    def test_catches_exceptions(self):
        self.reg.fill_tracker = MagicMock()
        self.reg.fill_tracker.do_thing.side_effect = ValueError("boom")
        result = self.reg._safe_call("fill_tracker", lambda ft: ft.do_thing())
        self.assertIsNone(result)

    def test_none_component_skipped(self):
        self.reg.vol_forecaster = None
        result = self.reg._safe_call("vol_forecaster", lambda vf: vf.update())
        self.assertIsNone(result)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: helpers import cleanly
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpersImport(unittest.TestCase):
    """Verify all 7 helper functions import."""

    def test_all_helpers_importable(self):
        from core.execute_signals_helpers import (
            _pre_execute_context,
            _extract_signal_fields,
            _apply_risk_gates,
            _compute_position_size,
            _apply_intelligence_gates,
            _compute_stops_and_quantity,
            _log_cycle_summary,
        )
        self.assertTrue(callable(_pre_execute_context))
        self.assertTrue(callable(_extract_signal_fields))
        self.assertTrue(callable(_apply_risk_gates))
        self.assertTrue(callable(_compute_position_size))
        self.assertTrue(callable(_apply_intelligence_gates))
        self.assertTrue(callable(_compute_stops_and_quantity))
        self.assertTrue(callable(_log_cycle_summary))


if __name__ == "__main__":
    unittest.main()
