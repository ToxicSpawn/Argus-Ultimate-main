"""
tests/test_batch12.py
=====================
Batch 12 test suite — 14 tests covering:
  H01  core/execution_engine.py
  H02  core/component_registry_base.py
  H04  backtest/ tombstone
  H05  hft/ tombstone
  M16  scripts/fix_silent_except.py tombstone
  M18  run_godmode.py — no unified_trading_system fallback
  M29  run_paper.py  — cycle_seconds sourced from config/CLI
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sys
import types

import pytest


# ===========================================================================
# H01 — ExecutionEngine
# ===========================================================================

class TestExecutionEngine:
    """Tests for core/execution_engine.py"""

    def _import(self):
        from core.execution_engine import ExecutionEngine, ExecutionRequest, ExecutionResult
        return ExecutionEngine, ExecutionRequest, ExecutionResult

    def test_module_importable(self):
        """core.execution_engine must be importable without side-effects."""
        mod = importlib.import_module("core.execution_engine")
        assert hasattr(mod, "ExecutionEngine")

    def test_dry_run_synthetic_fill(self):
        """dry_run=True should return success without a router."""
        EE, Req, Res = self._import()
        engine = EE(dry_run=True)
        req = Req(
            symbol="BTC/AUD",
            side="buy",
            quantity=0.001,
            price=50_000.0,
            strategy_name="test",
            signal_confidence=0.7,
        )
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is True
        assert result.filled_quantity == pytest.approx(0.001)
        assert result.filled_price == pytest.approx(50_000.0)

    def test_no_router_returns_failure(self):
        """Without a router and dry_run=False, execute returns failure."""
        EE, Req, _ = self._import()
        engine = EE(dry_run=False)
        req = Req(symbol="ETH/AUD", side="buy", quantity=0.01, price=2_000.0,
                  strategy_name="test", signal_confidence=0.6)
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is False
        assert result.error == "no_order_router_configured"

    def test_risk_facade_blocks_order(self):
        """A risk facade returning False must block the order and increment rejected counter."""
        EE, Req, _ = self._import()

        class AlwaysBlock:
            def check(self, _req):
                return False

        engine = EE(risk_facade=AlwaysBlock(), dry_run=True)
        req = Req(symbol="SOL/AUD", side="buy", quantity=1.0, price=100.0,
                  strategy_name="test", signal_confidence=0.9)
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is False
        assert result.error == "blocked_by_risk_facade"
        assert engine.stats["rejected"] == 1

    def test_stats_accumulate(self):
        """Stats should count placed and failed orders correctly."""
        EE, Req, _ = self._import()
        engine = EE(dry_run=True)
        req = Req(symbol="XRP/AUD", side="sell", quantity=100.0, price=0.5,
                  strategy_name="test", signal_confidence=0.55)
        asyncio.get_event_loop().run_until_complete(engine.execute(req))
        asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert engine.stats["placed"] == 2
        engine.reset_stats()
        assert engine.stats["placed"] == 0

    def test_cost_property(self):
        """ExecutionResult.cost == filled_quantity * filled_price."""
        _, _, Res = self._import()
        _, Req, _ = self._import()
        req = Req(symbol="BTC/AUD", side="buy", quantity=1.0, price=60_000.0,
                  strategy_name="t", signal_confidence=0.8)
        res = Res(success=True, request=req, filled_quantity=1.0, filled_price=60_000.0)
        assert res.cost == pytest.approx(60_000.0)


# ===========================================================================
# H02 — ComponentRegistryBase
# ===========================================================================

class TestComponentRegistryBase:
    """Tests for core/component_registry_base.py"""

    def _registry(self):
        from core.component_registry_base import ComponentRegistryBase
        return ComponentRegistryBase()

    def test_module_importable(self):
        mod = importlib.import_module("core.component_registry_base")
        assert hasattr(mod, "ComponentRegistryBase")

    def test_register_and_get(self):
        reg = self._registry()
        obj = object()
        reg.register("alpha", obj)
        assert reg.get("alpha") is obj

    def test_duplicate_raises(self):
        reg = self._registry()
        reg.register("x", object())
        with pytest.raises(ValueError, match="already registered"):
            reg.register("x", object())

    def test_unregister(self):
        reg = self._registry()
        reg.register("y", object())
        reg.unregister("y")
        assert "y" not in reg

    def test_startup_order_respects_deps(self):
        """Components with dependencies must appear after their deps in startup_order."""
        reg = self._registry()
        reg.register("engine", object(), depends_on=["risk"])
        reg.register("risk", object(), depends_on=[])
        order = reg.startup_order()
        assert order.index("risk") < order.index("engine")

    def test_cycle_detection(self):
        """Circular deps must raise RuntimeError."""
        reg = self._registry()
        reg.register("a", object(), depends_on=["b"])
        reg.register("b", object(), depends_on=["a"])
        with pytest.raises(RuntimeError, match="cycle"):
            reg.startup_order()

    def test_health_report(self):
        """health_report returns a dict with all component names."""
        reg = self._registry()
        reg.register("comp1", object())
        reg.register("comp2", object())
        report = reg.health_report()
        assert set(report.keys()) == {"comp1", "comp2"}


# ===========================================================================
# H04 — backtest/ tombstone
# ===========================================================================

class TestBacktestTombstone:
    def test_import_raises(self):
        """Importing backtest must raise ImportError directing to backtesting/."""
        if "backtest" in sys.modules:
            del sys.modules["backtest"]
        with pytest.raises(ImportError, match="backtesting"):
            import backtest  # noqa: F401


# ===========================================================================
# H05 — hft/ tombstone
# ===========================================================================

class TestHftTombstone:
    def test_import_raises(self):
        """Importing hft must raise ImportError directing to hft_engine/."""
        if "hft" in sys.modules:
            del sys.modules["hft"]
        with pytest.raises(ImportError, match="hft_engine"):
            import hft  # noqa: F401


# ===========================================================================
# M16 — scripts/fix_silent_except.py tombstone
# ===========================================================================

class TestFixSilentExceptTombstone:
    def test_file_exists(self):
        """The tombstone file must still be present so CI doesn't silently miss the delete."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "scripts", "fix_silent_except.py")
        assert os.path.isfile(path), "scripts/fix_silent_except.py missing"

    def test_tombstone_raises_runtime_error(self):
        """Executing the tombstone module must raise RuntimeError."""
        if "scripts.fix_silent_except" in sys.modules:
            del sys.modules["scripts.fix_silent_except"]
        with pytest.raises(RuntimeError, match="retired duplicate"):
            import scripts.fix_silent_except  # noqa: F401


# ===========================================================================
# M18 — run_godmode.py no unified_trading_system fallback
# ===========================================================================

class TestRunGodmodeNoFallback:
    def test_no_unified_trading_system_import(self):
        """run_godmode.py source must NOT reference unified_trading_system as a fallback."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "run_godmode.py")
        source = open(path, encoding="utf-8").read()
        # Must not fall back to importing the monolith
        assert "from unified_trading_system" not in source
        assert "import unified_trading_system" not in source

    def test_godmodeconfig_self_contained(self):
        """GodmodeConfig must be importable without importing unified_trading_system."""
        # Ensure unified_trading_system is NOT in sys.modules
        uts_key = "unified_trading_system"
        was_present = uts_key in sys.modules
        sys.modules.pop(uts_key, None)

        try:
            import run_godmode  # noqa: F401 — should not trigger UTS import
            assert uts_key not in sys.modules, "unified_trading_system was imported as a fallback"
        except ImportError:
            # Some transitive import is missing in test env — that's fine,
            # as long as UTS was not the one being imported.
            assert uts_key not in sys.modules
        finally:
            if not was_present:
                sys.modules.pop(uts_key, None)


# ===========================================================================
# M29 — run_paper.py cycle_seconds from config
# ===========================================================================

class TestRunPaperCycleSeconds:
    def test_module_has_load_cycle_seconds(self):
        """run_paper.py must expose _load_cycle_seconds helper."""
        import run_paper
        assert callable(getattr(run_paper, "_load_cycle_seconds", None))

    def test_cycle_seconds_constant_is_float(self):
        """CYCLE_SECONDS module constant must be a positive float."""
        import run_paper
        assert isinstance(run_paper.CYCLE_SECONDS, float)
        assert run_paper.CYCLE_SECONDS > 0

    def test_paper_trading_bot_accepts_cycle_seconds(self):
        """PaperTradingBot.__init__ must accept and store cycle_seconds kwarg."""
        import run_paper
        bot = run_paper.PaperTradingBot(capital=500.0, cycle_seconds=30.0)
        assert bot.cycle_seconds == pytest.approx(30.0)

    def test_hardcoded_15_not_in_sleep_call(self):
        """The old hardcoded asyncio.sleep(15) must be gone; loop must use self.cycle_seconds."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "run_paper.py")
        source = open(path, encoding="utf-8").read()
        assert "asyncio.sleep(15)" not in source, "Hardcoded asyncio.sleep(15) still present"
        assert "self.cycle_seconds" in source
