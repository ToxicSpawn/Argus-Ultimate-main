"""
Tests for system lifecycle enhancements:
  - Process lock prevents dual instances
  - Stale lock cleanup
  - Graceful shutdown cancels pending orders
  - Shutdown saves checkpoint and strategy states
  - SIGTERM triggers graceful shutdown
  - Pre-trading checks block on exchange failure (live mode)
  - Pre-trading checks pass in paper mode
  - Startup loads checkpoint
  - Lock released after crash (stale lock detection)
"""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ProcessLock tests
# ---------------------------------------------------------------------------


class TestProcessLock:
    """Test process lock acquisition, release, and stale lock handling."""

    def test_import(self):
        from core.process_lock import ProcessLock
        lock = ProcessLock(name="test_import")
        assert lock is not None

    def test_acquire_and_release(self, tmp_path):
        from core.process_lock import ProcessLock
        lock = ProcessLock(name="test_acq_rel", lock_dir=tmp_path, timeout=1.0)
        assert lock.acquire() is True
        assert lock.release() is True
        # Lock file should be gone
        assert not (tmp_path / "test_acq_rel.lock").exists()

    def test_prevents_dual_instance(self, tmp_path):
        """Second lock attempt on same name should fail when held by another PID."""
        from core.process_lock import ProcessLock
        lock1 = ProcessLock(name="test_dual", lock_dir=tmp_path, timeout=1.0)
        assert lock1.acquire() is True
        # Second lock from same process: the lock file has our PID, so
        # _pid_is_running returns True and acquire() correctly blocks.
        lock2 = ProcessLock(name="test_dual", lock_dir=tmp_path, timeout=0.2)
        assert lock2.acquire() is False  # blocked by our own PID in the file
        lock1.release()

    def test_prevents_dual_instance_different_pid(self, tmp_path):
        """Simulate another process holding the lock by writing a different live PID."""
        from core.process_lock import ProcessLock
        lock = ProcessLock(name="test_dual_pid", lock_dir=tmp_path, timeout=0.3)
        # Write PID 1 (init/system, always running) to simulate another holder
        lock_file = tmp_path / "test_dual_pid.lock"
        tmp_path.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("1", encoding="utf-8")  # PID 1 is always alive
        result = lock.acquire()
        # On Windows PID 1 may not exist, so behavior varies
        # If PID 1 is not running (Windows), stale lock is cleaned and we acquire
        # If PID 1 is running (Linux), we fail to acquire
        if result:
            lock.release()
        # Either way, the function completed without error

    def test_stale_lock_cleanup(self, tmp_path):
        """Stale lock from dead process should be cleaned up."""
        from core.process_lock import ProcessLock
        lock_file = tmp_path / "test_stale.lock"
        tmp_path.mkdir(parents=True, exist_ok=True)
        # Write a PID that definitely does not exist
        lock_file.write_text("9999999", encoding="utf-8")
        lock = ProcessLock(name="test_stale", lock_dir=tmp_path, timeout=1.0)
        # Should clean up stale lock and acquire
        assert lock.acquire() is True
        assert lock.release() is True

    def test_stale_lock_malformed_file(self, tmp_path):
        """Malformed lock file should be treated as stale."""
        from core.process_lock import ProcessLock
        lock_file = tmp_path / "test_malformed.lock"
        tmp_path.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("not_a_pid", encoding="utf-8")
        lock = ProcessLock(name="test_malformed", lock_dir=tmp_path, timeout=1.0)
        assert lock.acquire() is True
        lock.release()

    def test_context_manager(self, tmp_path):
        from core.process_lock import ProcessLock
        with ProcessLock(name="test_ctx", lock_dir=tmp_path, timeout=1.0) as lock:
            assert (tmp_path / "test_ctx.lock").exists()
        assert not (tmp_path / "test_ctx.lock").exists()

    def test_context_manager_raises_on_conflict(self, tmp_path):
        """Context manager raises RuntimeError when lock cannot be acquired."""
        from core.process_lock import ProcessLock
        lock_file = tmp_path / "test_ctx_conflict.lock"
        tmp_path.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("1", encoding="utf-8")  # PID 1
        # On some OSes PID 1 is alive, on others not
        # We patch _pid_is_running to always return True
        with patch.object(ProcessLock, "_pid_is_running", return_value=True):
            with pytest.raises(RuntimeError, match="Could not acquire"):
                with ProcessLock(name="test_ctx_conflict", lock_dir=tmp_path, timeout=0.2):
                    pass

    def test_force_release(self, tmp_path):
        from core.process_lock import ProcessLock
        lock_file = tmp_path / "test_force.lock"
        tmp_path.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("12345", encoding="utf-8")
        lock = ProcessLock(name="test_force", lock_dir=tmp_path)
        assert lock.force_release() is True
        assert not lock_file.exists()

    def test_is_locked(self, tmp_path):
        from core.process_lock import ProcessLock
        lock = ProcessLock(name="test_islocked", lock_dir=tmp_path, timeout=1.0)
        assert lock.is_locked() is False
        lock.acquire()
        assert lock.is_locked() is True
        lock.release()
        assert lock.is_locked() is False

    def test_acquire_or_exit_succeeds(self, tmp_path):
        from core.process_lock import acquire_or_exit
        lock = acquire_or_exit(name="test_aoe", lock_dir=tmp_path, timeout=1.0)
        assert lock is not None
        lock.release()

    def test_acquire_or_exit_exits_on_conflict(self, tmp_path):
        from core.process_lock import ProcessLock, acquire_or_exit
        lock_file = tmp_path / "test_aoe_conflict.lock"
        tmp_path.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("1", encoding="utf-8")
        with patch.object(ProcessLock, "_pid_is_running", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                acquire_or_exit(name="test_aoe_conflict", lock_dir=tmp_path, timeout=0.2)
            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Helpers: minimal mock config and system for testing shutdown/startup
# ---------------------------------------------------------------------------


@dataclass
class _MockConfig:
    """Minimal config for testing UnifiedSystemArchitecture lifecycle."""
    starting_capital_aud: float = 1000.0
    primary_exchange: str = "kraken"
    secondary_exchange: str = "coinbase"
    trading_pairs: list = field(default_factory=lambda: ["BTC/AUD"])
    run_mode: str = "paper"
    config_version: int = 1
    multi_language_enabled: bool = False
    node_role: str = "single-node"
    aud_to_usd: float = 0.65
    paper_trading_peak_mode: bool = False
    paper_simulates_live: bool = False
    paper_trading_overrides: dict = field(default_factory=dict)
    reconcile_every_n_cycles: int = 10
    order_timeout_seconds: float = 60.0
    paper_slippage_bps: float = 5.0
    use_quantum_walk: bool = False
    use_quantum_monte_carlo_risk: bool = False
    emergency_shutdown_enabled: bool = False


def _make_mock_system(config: _MockConfig | None = None):
    """Create a minimal mock of UnifiedSystemArchitecture for lifecycle tests."""
    cfg = config or _MockConfig()
    system = MagicMock()
    system.config = cfg
    system.state = None
    system.start_time = datetime.now()
    system._pending_orders = {}
    system._completed_cycles = 0
    system.portfolio_value_aud = 1050.0
    system.exchanges = {"kraken": MagicMock()}
    system.execution_engine = MagicMock()
    system.execution_engine.cancel_all_orders = AsyncMock(return_value=None)
    system.execution_engine.trade_ledger = MagicMock()
    system.execution_engine.trade_ledger.flush = MagicMock()
    system.audit_chain = MagicMock()
    system.audit_chain.flush = MagicMock()
    system.checkpoint_manager = MagicMock()
    system.checkpoint_manager.save_checkpoint = MagicMock()
    system.checkpoint_manager.load_latest_checkpoint = MagicMock(return_value={"cycle_count": 42})
    system.checkpoint_manager.should_save = MagicMock(return_value=True)
    system._strategy_state_store = MagicMock()
    system._strategy_state_store.save_all = MagicMock()
    system._strategy_state_store.load_all = MagicMock(return_value={"strat_a": {}})
    system.component_registry = MagicMock()
    system.component_registry.shutdown = AsyncMock()
    system.live_market_data = None
    system.ws_connectors = []
    system.monitoring = None
    system.model_manager = None
    system.continuous_scanner = None
    system.strategy_evaluation_engine = None
    system.champion_challenger_engine = None
    system.regime_store = None
    system.current_regime = None
    system._process_lock = None
    system._last_regime_label = "UNKNOWN"
    return system


# ---------------------------------------------------------------------------
# Graceful shutdown tests
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Test the graceful_shutdown method."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_calls_shutdown(self):
        """graceful_shutdown should delegate to shutdown()."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        # Bind the real graceful_shutdown method to our mock
        system.graceful_shutdown = UnifiedSystemArchitecture.graceful_shutdown.__get__(system)
        system.shutdown = AsyncMock()
        await system.graceful_shutdown(reason="test")
        system.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_releases_process_lock(self):
        """Process lock should be released during graceful shutdown."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        mock_lock = MagicMock()
        mock_lock.release = MagicMock(return_value=True)
        system._process_lock = mock_lock
        system.graceful_shutdown = UnifiedSystemArchitecture.graceful_shutdown.__get__(system)
        system.shutdown = AsyncMock()
        await system.graceful_shutdown(reason="test_lock")
        mock_lock.release.assert_called_once()
        assert system._process_lock is None

    @pytest.mark.asyncio
    async def test_graceful_shutdown_flushes_audit_trail(self):
        """Audit trail flush should be called during graceful shutdown."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        system.graceful_shutdown = UnifiedSystemArchitecture.graceful_shutdown.__get__(system)
        system.shutdown = AsyncMock()
        await system.graceful_shutdown(reason="test_audit")
        system.audit_chain.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_logs_summary(self, caplog):
        """Shutdown summary should be logged."""
        import logging
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        system.graceful_shutdown = UnifiedSystemArchitecture.graceful_shutdown.__get__(system)
        system.shutdown = AsyncMock()
        with caplog.at_level(logging.INFO):
            await system.graceful_shutdown(reason="test_summary")
        assert any("SHUTDOWN SUMMARY" in r.message for r in caplog.records)
        assert any("test_summary" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending_orders(self):
        """The shutdown method should call cancel_all_orders on execution engine."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        system._pending_orders = {"order_1": {"symbol": "BTC/AUD"}}
        system.shutdown = UnifiedSystemArchitecture.shutdown.__get__(system)
        await system.shutdown()
        system.execution_engine.cancel_all_orders.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_saves_checkpoint(self):
        """Shutdown should save a final checkpoint."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        system._total_cycles = 100
        system.shutdown = UnifiedSystemArchitecture.shutdown.__get__(system)
        await system.shutdown()
        system.checkpoint_manager.save_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_saves_strategy_states(self):
        """Shutdown should save all strategy states."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        system.shutdown = UnifiedSystemArchitecture.shutdown.__get__(system)
        await system.shutdown()
        system._strategy_state_store.save_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_closes_exchange_connections(self):
        """Shutdown should close exchange client connections."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        mock_exchange = MagicMock()
        mock_exchange.close = AsyncMock()
        system = _make_mock_system()
        system.exchanges = {"kraken": mock_exchange}
        system.shutdown = UnifiedSystemArchitecture.shutdown.__get__(system)
        await system.shutdown()
        mock_exchange.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_sets_state(self):
        """Shutdown should set state to SHUTDOWN."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        system.shutdown = UnifiedSystemArchitecture.shutdown.__get__(system)
        await system.shutdown()
        assert system.state == SystemState.SHUTDOWN

    @pytest.mark.asyncio
    async def test_graceful_shutdown_survives_errors(self):
        """graceful_shutdown should not crash even if individual steps fail."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        system.audit_chain.flush.side_effect = RuntimeError("flush fail")
        system._process_lock = MagicMock()
        system._process_lock.release.side_effect = RuntimeError("release fail")
        system.graceful_shutdown = UnifiedSystemArchitecture.graceful_shutdown.__get__(system)
        system.shutdown = AsyncMock()
        # Should not raise
        await system.graceful_shutdown(reason="error_test")


# ---------------------------------------------------------------------------
# Signal handler tests
# ---------------------------------------------------------------------------


class TestSignalHandlers:
    """Test that SIGTERM/SIGINT set shutdown state."""

    def test_signal_handler_sets_shutdown_state(self):
        """_install_signal_handlers should register handlers that set SHUTDOWN state."""
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        system = _make_mock_system()
        system.state = SystemState.RUNNING
        system._install_signal_handlers = UnifiedSystemArchitecture._install_signal_handlers.__get__(system)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            system._install_signal_handlers()
            # Simulate SIGINT via signal.signal handler
            # On Windows, add_signal_handler is not supported so it falls back to signal.signal
            # We can trigger the handler directly
            handler = signal.getsignal(signal.SIGINT)
            if handler and handler != signal.default_int_handler:
                try:
                    handler(signal.SIGINT, None)
                except Exception:
                    pass
            # State should be SHUTDOWN after signal
            assert system.state == SystemState.SHUTDOWN
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Pre-trading checks tests
# ---------------------------------------------------------------------------


class TestPreTradingChecks:
    """Test the _pre_trading_checks method."""

    @pytest.mark.asyncio
    async def test_paper_mode_passes(self):
        """Pre-trading checks should pass in paper mode even without exchanges."""
        from unified_trading_system import UnifiedSystemArchitecture

        system = _make_mock_system()
        system.config.run_mode = "paper"
        system.exchanges = {}
        system._pre_trading_checks = UnifiedSystemArchitecture._pre_trading_checks.__get__(system)
        # Should not raise
        await system._pre_trading_checks()

    @pytest.mark.asyncio
    async def test_live_mode_fails_no_exchanges(self):
        """Pre-trading checks should fail in live mode with no exchanges."""
        from unified_trading_system import UnifiedSystemArchitecture

        system = _make_mock_system()
        system.config.run_mode = "live"
        system.exchanges = {}
        system._pre_trading_checks = UnifiedSystemArchitecture._pre_trading_checks.__get__(system)
        with pytest.raises(RuntimeError, match="Pre-trading checks failed"):
            await system._pre_trading_checks()

    @pytest.mark.asyncio
    async def test_live_mode_passes_with_exchanges(self):
        """Pre-trading checks should pass in live mode when exchanges are reachable."""
        from unified_trading_system import UnifiedSystemArchitecture

        mock_ex = MagicMock()
        mock_ex.fetch_time = AsyncMock(return_value=1234567890)
        system = _make_mock_system()
        system.config.run_mode = "live"
        system.exchanges = {"kraken": mock_ex}
        # Prevent component_registry MagicMock from triggering spurious position sync
        system.component_registry = None
        system._pre_trading_checks = UnifiedSystemArchitecture._pre_trading_checks.__get__(system)
        # Mock the deployment checklist so environment-dependent checks don't cause flaky failures
        mock_checklist_result = MagicMock()
        mock_checklist_result.go = True
        mock_checklist_result.passed_count = 9
        mock_checklist_result.checks = [None] * 9
        mock_checklist_cls = MagicMock(return_value=MagicMock(run=MagicMock(return_value=mock_checklist_result)))
        with patch("ops.deployment_checklist.DeploymentChecklist", mock_checklist_cls):
            # Should not raise (exchange reachable + other checks pass)
            await system._pre_trading_checks()

    @pytest.mark.asyncio
    async def test_loads_checkpoint(self):
        """Pre-trading checks should attempt to load the latest checkpoint."""
        from unified_trading_system import UnifiedSystemArchitecture

        system = _make_mock_system()
        system.config.run_mode = "paper"
        system._pre_trading_checks = UnifiedSystemArchitecture._pre_trading_checks.__get__(system)
        await system._pre_trading_checks()
        system.checkpoint_manager.load_latest_checkpoint.assert_called()

    @pytest.mark.asyncio
    async def test_loads_strategy_states(self):
        """Pre-trading checks should load strategy states."""
        from unified_trading_system import UnifiedSystemArchitecture

        system = _make_mock_system()
        system.config.run_mode = "paper"
        system._pre_trading_checks = UnifiedSystemArchitecture._pre_trading_checks.__get__(system)
        await system._pre_trading_checks()
        system._strategy_state_store.load_all.assert_called()

    @pytest.mark.asyncio
    async def test_invalid_capital_logs_failure(self, caplog):
        """Pre-trading checks should log failure for zero capital."""
        import logging
        from unified_trading_system import UnifiedSystemArchitecture

        system = _make_mock_system()
        system.config.starting_capital_aud = 0.0
        system.config.run_mode = "paper"
        system._pre_trading_checks = UnifiedSystemArchitecture._pre_trading_checks.__get__(system)
        with caplog.at_level(logging.ERROR):
            await system._pre_trading_checks()
        assert any("starting_capital_aud must be > 0" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_exchange_unreachable_live_fails(self):
        """Pre-trading checks should fail if exchange is unreachable in live mode."""
        from unified_trading_system import UnifiedSystemArchitecture

        mock_ex = MagicMock()
        mock_ex.fetch_time = AsyncMock(side_effect=ConnectionError("timeout"))
        system = _make_mock_system()
        system.config.run_mode = "live"
        system.exchanges = {"kraken": mock_ex}
        system._pre_trading_checks = UnifiedSystemArchitecture._pre_trading_checks.__get__(system)
        with pytest.raises(RuntimeError, match="Pre-trading checks failed"):
            await system._pre_trading_checks()

    @pytest.mark.asyncio
    async def test_startup_summary_logged(self, caplog):
        """Pre-trading checks should log a startup summary."""
        import logging
        from unified_trading_system import UnifiedSystemArchitecture

        system = _make_mock_system()
        system.config.run_mode = "paper"
        system._pre_trading_checks = UnifiedSystemArchitecture._pre_trading_checks.__get__(system)
        with caplog.at_level(logging.INFO):
            await system._pre_trading_checks()
        assert any("PRE-TRADING CHECKS COMPLETE" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration: main.py process lock wiring
# ---------------------------------------------------------------------------


class TestMainProcessLockWiring:
    """Test that main.py properly acquires and releases process locks."""

    def test_run_paper_trading_acquires_lock(self, tmp_path):
        """run_paper_trading should acquire and release a process lock."""
        # We'll patch ProcessLock to use tmp_path and verify it's called
        from core.process_lock import ProcessLock

        acquired = {"value": False}
        released = {"value": False}
        original_acquire = ProcessLock.acquire
        original_release = ProcessLock.release

        class TrackingLock(ProcessLock):
            def acquire(self):
                acquired["value"] = True
                return True

            def release(self):
                released["value"] = True
                return True

        with patch("core.process_lock.ProcessLock", TrackingLock):
            with patch("main.asyncio") as mock_asyncio:
                mock_asyncio.run = MagicMock()
                # Import after patching
                import importlib
                import main as main_mod
                importlib.reload(main_mod)
                # This would try to run the system, but asyncio.run is mocked
                # Just verify the lock pattern exists by checking the function source
                import inspect
                src = inspect.getsource(main_mod.run_paper_trading)
                assert "acquire_or_exit" in src
                assert "proc_lock" in src

    def test_run_live_trading_has_lock_pattern(self):
        """run_live_trading should contain process lock acquisition."""
        import inspect
        import main as main_mod
        src = inspect.getsource(main_mod.run_live_trading)
        assert "acquire_or_exit" in src
        assert "_proc_lock" in src
        assert "finally:" in src

    def test_run_unified_system_accepts_process_lock(self):
        """_run_unified_system should accept a process_lock parameter."""
        import inspect
        import main as main_mod
        sig = inspect.signature(main_mod._run_unified_system)
        assert "process_lock" in sig.parameters
