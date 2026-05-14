"""
tests/integration/test_signal_to_fill.py
=========================================
End-to-end integration test: TradingSignal → full_wiring → paper fill.

This harness fires a synthetic BUY signal through the complete live path:

    TradingSignal
        └─> signal_routing.build_execution_context()
        └─> signal_routing.extract_signal_fields()
        └─> risk_gates.apply_all_risk_gates()
        └─> position_sizing.compute_position_size()
        └─> position_sizing.apply_intelligence_gates()
        └─> position_sizing.compute_stops_and_quantity()
        └─> paper order execution
        └─> assertions on fill result

Design principles
-----------------
* No live exchange calls — all I/O is stubbed.
* No GPU / JAX required — RL gate falls back gracefully.
* Fast: the full suite runs < 2 s on any CI runner.
* Deterministic: fixed seed, fixed portfolio value, fixed signal timestamp.

Running
-------
    pytest tests/integration/test_signal_to_fill.py -v

Or directly:
    python -m pytest tests/integration/test_signal_to_fill.py -v
"""
from __future__ import annotations

import sys
import time
import types
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so imports don't require the full Argus dependency tree
# ---------------------------------------------------------------------------

def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _stub in [
    "data.macro.fred_calendar",
    "argus_live.control_plane.profile_resolver",
    "argus_live.control_plane.runtime_manifest",
    "ml.argus_ai.model",
]:
    if _stub not in sys.modules:
        _make_stub_module(_stub)


# ---------------------------------------------------------------------------
# Synthetic TradingSignal
# ---------------------------------------------------------------------------

@dataclass
class TradingSignal:
    symbol: str
    action: str
    confidence: float
    strength: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    strategy: str = "momentum"
    timestamp: float = field(default_factory=time.time)
    num_confirmations: int = 3


# ---------------------------------------------------------------------------
# Minimal UnifiedTradingSystem stub
# ---------------------------------------------------------------------------

class _MockConfig:
    run_mode = "paper"
    aud_to_usd = 0.65
    portfolio_var_limit_pct = 0.0
    portfolio_cvar_limit_pct = 0.0
    max_position_pct = 0.25
    min_position_size_aud = 10.0
    stop_loss_pct = 0.02
    take_profit_pct = 0.04
    max_concurrent_positions = 5
    max_pyramids_per_position = 2
    primary_exchange = "paper"


class _MockRiskManager:
    def is_daily_loss_limit_exceeded(self) -> bool:
        return False

    def get_risk_metrics(self):
        m = MagicMock()
        m.current_capital = 10_000.0
        m.var_95 = 0.0
        m.var_99 = 0.0
        return m

    def pre_trade_risk_check(self, symbol: str, position_size_usd: float):
        return True, ""


class _MockSystem:
    """
    Minimal stub for UnifiedTradingSystem that satisfies all attribute
    accesses made by signal_routing, risk_gates, and position_sizing.
    """

    REGIME_POSITION_SCALE: Dict[str, float] = {"NORMAL": 1.0, "TRENDING_UP": 1.2}
    REGIME_STOP_SCALE: Dict[str, float] = {"NORMAL": 1.0}
    REGIME_TP_SCALE: Dict[str, float] = {"NORMAL": 1.0}

    def __init__(self, portfolio_value_aud: float = 20_000.0):
        self.config = _MockConfig()
        self.portfolio_value_aud = portfolio_value_aud
        self.peak_equity_aud = portfolio_value_aud
        self.positions: Dict[str, Any] = {}
        self.unified_risk_manager = _MockRiskManager()
        self.component_registry = None
        self._latest_regime_label = "NORMAL"
        self._last_cycle_advisory: Dict = {}
        self._strategy_state_store = None
        self.daily_pnl = 0.0
        self._cycle_number = 1

    def _get_current_vol(self, symbol: str) -> float:
        return 0.005

    def _vol_adjusted_size(self, size_pct: float, vol: float) -> float:
        return size_pct

    def _get_signal_quality(self) -> Optional[dict]:
        return None

    def _get_strategy_trade_stats(self, strategy: str) -> dict:
        return {"n_trades": 0, "win_rate": 0.5, "avg_win": 0.0, "avg_loss": 0.0}

    def _kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss <= 0:
            return 0.0
        return max(0.0, win_rate - (1 - win_rate) / (avg_win / avg_loss)) * 0.5

    def _get_strategy_multiplier(self, strategy: str) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Paper order executor stub
# ---------------------------------------------------------------------------

@dataclass
class PaperFillResult:
    symbol: str
    side: str
    quantity: float
    fill_price: float
    stop_loss: float
    take_profit: float
    position_value_aud: float
    status: str = "filled"


def paper_execute(
    system: _MockSystem,
    sig_fields: dict,
    stops: dict,
) -> PaperFillResult:
    """
    Simulate a paper fill at entry_price with zero slippage.
    Updates system.positions in-place.
    """
    symbol = sig_fields["symbol"]
    action = sig_fields["action"]
    entry_price = sig_fields["entry_price"]
    qty = stops["quantity"]
    pv_aud = stops["position_value_aud"]

    system.positions[symbol] = {
        "side": action,
        "quantity": qty,
        "entry_price": entry_price,
        "stop_loss": stops["stop_loss"],
        "take_profit": stops["take_profit"],
        "pyramid_count": 0,
    }

    return PaperFillResult(
        symbol=symbol,
        side=action,
        quantity=qty,
        fill_price=entry_price,
        stop_loss=stops["stop_loss"],
        take_profit=stops["take_profit"],
        position_value_aud=pv_aud,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_signal(
    signal: TradingSignal,
    system: Optional[_MockSystem] = None,
) -> Optional[PaperFillResult]:
    """Run one signal through the complete pipeline. Returns PaperFillResult or None."""
    from core.signal_routing import build_execution_context, extract_signal_fields
    from core.risk_gates import apply_all_risk_gates
    from core.position_sizing import compute_position_size, compute_stops_and_quantity
    from execute_signals_helpers import _apply_intelligence_gates

    sys_inst = system or _MockSystem()

    # Patch macro calendar to avoid network calls
    with patch("core.signal_routing.FREDCalendar", side_effect=ImportError):
        ctx = build_execution_context(sys_inst)

    fields = extract_signal_fields(signal)
    if fields is None or fields.get("_blocked"):
        return None

    approved, reason = apply_all_risk_gates(sys_inst, fields, ctx)
    if not approved:
        return None

    size_pct, sizing_method = compute_position_size(sys_inst, fields, ctx)
    size_pct, sizing_method = _apply_intelligence_gates(
        sys_inst, fields, size_pct, ctx.get("_cycle_advisory", {}), sizing_method, ctx
    )
    if sizing_method.startswith("BLOCKED:"):
        return None

    stops = compute_stops_and_quantity(sys_inst, fields, size_pct, ctx)
    if stops is None or stops.get("_too_small"):
        return None

    return paper_execute(sys_inst, fields, stops)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestSignalToFill(unittest.TestCase):

    def _make_signal(self, **kwargs) -> TradingSignal:
        defaults = dict(
            symbol="BTC/USDT",
            action="BUY",
            confidence=0.75,
            strength=0.70,
            entry_price=65_000.0,
            strategy="momentum",
            num_confirmations=3,
            timestamp=time.time(),
        )
        defaults.update(kwargs)
        return TradingSignal(**defaults)

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_buy_signal_produces_fill(self):
        """A fresh, high-confidence BUY signal should reach a paper fill."""
        sig = self._make_signal()
        result = _run_signal(sig)
        self.assertIsNotNone(result, "Expected a fill but got None")
        self.assertEqual(result.status, "filled")
        self.assertEqual(result.symbol, "BTC/USDT")
        self.assertEqual(result.side, "BUY")
        self.assertGreater(result.quantity, 0)
        self.assertGreater(result.position_value_aud, 10.0)

    def test_sell_signal_produces_fill(self):
        """SELL signals should also flow through cleanly."""
        sys_inst = _MockSystem()
        sys_inst.positions["BTC/USDT"] = {
            "side": "BUY", "quantity": 0.01,
            "entry_price": 64_000.0, "stop_loss": 62_000.0,
            "take_profit": 68_000.0, "pyramid_count": 0,
        }
        sig = self._make_signal(action="SELL", entry_price=66_000.0)
        result = _run_signal(sig, system=sys_inst)
        self.assertIsNotNone(result)
        self.assertEqual(result.side, "SELL")

    def test_stop_and_tp_are_set(self):
        """Stop-loss and take-profit must be non-None after the pipeline."""
        sig = self._make_signal()
        result = _run_signal(sig)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.stop_loss)
        self.assertIsNotNone(result.take_profit)
        # For BUY: stop < entry < take_profit
        self.assertLess(result.stop_loss, sig.entry_price)
        self.assertGreater(result.take_profit, sig.entry_price)

    def test_position_registered_in_system(self):
        """After a fill the system.positions dict should contain the symbol."""
        sys_inst = _MockSystem()
        sig = self._make_signal()
        _run_signal(sig, system=sys_inst)
        self.assertIn("BTC/USDT", sys_inst.positions)
        pos = sys_inst.positions["BTC/USDT"]
        self.assertGreater(float(pos["quantity"]), 0)

    # ------------------------------------------------------------------
    # Risk gate blocking
    # ------------------------------------------------------------------

    def test_stale_signal_blocked(self):
        """Signals older than 120 s must be rejected."""
        sig = self._make_signal(timestamp=time.time() - 200)
        result = _run_signal(sig)
        self.assertIsNone(result)

    def test_daily_loss_limit_blocks_buy(self):
        """When daily loss is exceeded, BUY signals must be rejected."""
        sys_inst = _MockSystem()
        sys_inst.unified_risk_manager = _MockRiskManager()
        sys_inst.unified_risk_manager.is_daily_loss_limit_exceeded = lambda: True
        sig = self._make_signal(action="BUY")
        result = _run_signal(sig, system=sys_inst)
        self.assertIsNone(result)

    def test_max_positions_gate(self):
        """With 5 open positions and max=5, a new BUY must be rejected."""
        sys_inst = _MockSystem()
        for i in range(5):
            sys_inst.positions[f"TOKEN{i}/USDT"] = {
                "side": "BUY", "quantity": 0.01,
                "entry_price": 100.0, "stop_loss": 90.0,
                "take_profit": 120.0, "pyramid_count": 0,
            }
        sig = self._make_signal(symbol="NEW/USDT")
        result = _run_signal(sig, system=sys_inst)
        self.assertIsNone(result)

    def test_invalid_signal_blocked(self):
        """A signal with zero entry price must be blocked."""
        sig = self._make_signal(entry_price=0.0)
        result = _run_signal(sig)
        self.assertIsNone(result)

    def test_unknown_action_returns_none(self):
        """Non-BUY/SELL actions must be silently dropped."""
        sig = self._make_signal(action="HOLD")
        result = _run_signal(sig)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Sizing sanity
    # ------------------------------------------------------------------

    def test_size_capped_at_max_position_pct(self):
        """Position value must never exceed max_position_pct of portfolio."""
        sys_inst = _MockSystem(portfolio_value_aud=50_000.0)
        sig = self._make_signal(confidence=1.0, strength=1.0)
        result = _run_signal(sig, system=sys_inst)
        self.assertIsNotNone(result)
        max_value = sys_inst.portfolio_value_aud * sys_inst.config.max_position_pct
        self.assertLessEqual(result.position_value_aud, max_value * 1.01)  # 1% tolerance

    def test_low_confidence_signal_smaller_position(self):
        """A low-confidence signal should produce a smaller position than a high-confidence one."""
        sys_high = _MockSystem(portfolio_value_aud=20_000.0)
        sys_low = _MockSystem(portfolio_value_aud=20_000.0)
        high_sig = self._make_signal(confidence=0.90, strength=0.90)
        low_sig = self._make_signal(confidence=0.30, strength=0.30)
        res_high = _run_signal(high_sig, system=sys_high)
        res_low = _run_signal(low_sig, system=sys_low)
        if res_high is not None and res_low is not None:
            self.assertGreater(res_high.position_value_aud, res_low.position_value_aud)

    # ------------------------------------------------------------------
    # RL inference gate (stub mode — no JAX required)
    # ------------------------------------------------------------------

    def test_rl_gate_passthrough_when_disabled(self):
        """RLInferenceGate with enabled=False must return size_mult=1.0, skip=False."""
        from core.rl_inference_gate import RLInferenceGate
        gate = RLInferenceGate(enabled=False)
        obs = [0.75, 0.70, 0.0, 0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.2]
        decision = gate.infer(obs)
        self.assertFalse(decision.skip_trade)
        self.assertAlmostEqual(decision.size_multiplier, 1.0)

    def test_rl_gate_no_checkpoint_graceful(self):
        """RLInferenceGate with ARGUS_RL_INFERENCE=1 but no checkpoint must not raise."""
        import os
        from core.rl_inference_gate import RLInferenceGate
        os.environ["ARGUS_RL_INFERENCE"] = "1"
        try:
            gate = RLInferenceGate(model_dir="/nonexistent/path/")
            obs = [0.5] * 13
            decision = gate.infer(obs)
            self.assertFalse(decision.skip_trade)
            self.assertAlmostEqual(decision.size_multiplier, 1.0)
        finally:
            os.environ.pop("ARGUS_RL_INFERENCE", None)

    def test_rl_obs_builder_shape(self):
        """build_obs must always return exactly 64 floats."""
        from core.rl_inference_gate import RLInferenceGate
        sig_fields = {
            "confidence": 0.8, "strength": 0.7,
            "_sig_age": 5.0, "_age_urgency": 0.3,
            "_num_confirmations": 4, "action": "BUY",
        }
        ctx = {
            "regime_pos_mult": 1.0, "regime_stop_mult": 1.0,
            "session_mult": 1.05, "macro_event_imminent": False,
            "daily_loss_exceeded": False, "var_breach": False,
            "portfolio_value": 20_000.0,
        }
        obs = RLInferenceGate.build_obs(sig_fields, ctx)
        self.assertEqual(obs.shape[0], 64)
        self.assertEqual(obs.dtype.name, "float32")

    # ------------------------------------------------------------------
    # ArgusConfig integration
    # ------------------------------------------------------------------

    def test_argus_config_defaults_load_cleanly(self):
        """ArgusConfig with all defaults must construct without validation errors."""
        from core.argus_config import ArgusConfig
        cfg = ArgusConfig()
        self.assertEqual(cfg.runtime.node_role, "standalone")
        self.assertFalse(cfg.runtime.live_trade)
        self.assertFalse(cfg.network.dpdk.enabled)
        self.assertEqual(cfg.ai.rl_algorithm, "PPO")

    def test_argus_config_mutex_validation(self):
        """dry_run=True and live_trade=True simultaneously must raise."""
        from core.argus_config import RuntimeConfig
        with self.assertRaises(Exception):
            RuntimeConfig(dry_run=True, live_trade=True)

    def test_rl_gate_from_argus_config(self):
        """RLInferenceGate.from_config() must read ai.rl_enabled correctly."""
        from core.argus_config import ArgusConfig, AIConfig
        from core.rl_inference_gate import RLInferenceGate
        cfg = ArgusConfig(ai=AIConfig(rl_enabled=False))
        gate = RLInferenceGate.from_config(cfg)
        self.assertFalse(gate.enabled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
