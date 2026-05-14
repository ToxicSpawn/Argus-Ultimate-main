"""
Tests for system lifecycle features:
  1. Graceful shutdown sequence (ComponentRegistry.shutdown)
  2. Per-component health checks (ComponentRegistry.health_check_all + /health/components)
  3. Component dependency graph (ComponentRegistry.validate_dependencies)
  4. Config reload timeout (HotReloadManager timeout wrapper)
  5. Process lock crash cleanup (ProcessLock stale PID detection)
  6. Regime store TTL (RegimeStore.max_age_seconds)
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal config stub for ComponentRegistry."""
    starting_capital_aud = 1000.0
    primary_exchange = "kraken"
    aud_to_usd = 0.65
    trading_pairs = ["BTC/USD", "ETH/USD"]
    llm_signal_enabled = False
    entity_name = "TEST"
    multi_venue_min_notional_aud = 200.0


class _MockComponent:
    """A component with close() for shutdown tests."""
    def __init__(self, name: str = "mock"):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


class _SlowComponent:
    """A component whose close() is deliberately slow."""
    def __init__(self):
        self._stop = threading.Event()

    def close(self):
        self._stop.wait(timeout=2.0)

    def release(self):
        self._stop.set()


class _ErrorComponent:
    """A component whose close() raises."""
    def close(self):
        raise RuntimeError("shutdown error")


class _HealthyComponent:
    """A component that reports healthy."""
    def health(self):
        return True


class _UnhealthyComponent:
    """A component that reports unhealthy."""
    def is_healthy(self):
        return False


class _DictHealthComponent:
    """A component that returns a dict from health()."""
    def health(self):
        return {"status": "degraded", "detail": "high latency"}


class _ErrorHealthComponent:
    """A component whose health() raises."""
    def health(self):
        raise RuntimeError("health check failed")


# ---------------------------------------------------------------------------
# 1. Graceful shutdown
# ---------------------------------------------------------------------------

class TestGracefulShutdown:
    """Tests for ComponentRegistry.shutdown()."""

    def _make_registry(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(_FakeConfig())
        return reg

    async def test_shutdown_sets_flag(self):
        reg = self._make_registry()
        assert reg._shutting_down is False
        result = await reg.shutdown(timeout=5.0)
        assert reg._shutting_down is True
        assert isinstance(result, dict)

    async def test_shutdown_calls_close(self):
        reg = self._make_registry()
        comp = _MockComponent("test_comp")
        reg.fill_tracker = comp
        reg._init_order = ["fill_tracker"]
        result = await reg.shutdown(timeout=5.0)
        assert comp.closed is True
        assert result["fill_tracker"] == "ok"

    async def test_shutdown_reverse_order(self):
        reg = self._make_registry()
        order = []
        comp_a = MagicMock()
        comp_a.close = lambda: order.append("a")
        comp_b = MagicMock()
        comp_b.close = lambda: order.append("b")
        comp_c = MagicMock()
        comp_c.close = lambda: order.append("c")
        reg.fill_tracker = comp_a
        reg.maker_enforcement = comp_b
        reg.rate_limit_manager = comp_c
        reg._init_order = ["fill_tracker", "maker_enforcement", "rate_limit_manager"]
        await reg.shutdown(timeout=5.0)
        assert order == ["c", "b", "a"]

    async def test_shutdown_timeout_handling(self):
        reg = self._make_registry()
        slow = _SlowComponent()
        reg.fill_tracker = slow
        reg._init_order = ["fill_tracker"]
        result = await reg.shutdown(timeout=0.05)
        slow.release()  # Let the thread finish so it cleans up
        assert result["fill_tracker"] == "timeout"

    async def test_shutdown_error_handling(self):
        reg = self._make_registry()
        bad = _ErrorComponent()
        reg.fill_tracker = bad
        reg._init_order = ["fill_tracker"]
        result = await reg.shutdown(timeout=5.0)
        assert result["fill_tracker"] == "error"

    async def test_shutdown_no_close_method(self):
        """Components without close/shutdown should get 'ok'."""
        reg = self._make_registry()
        reg.fill_tracker = object()  # no close method
        reg._init_order = ["fill_tracker"]
        result = await reg.shutdown(timeout=5.0)
        assert result["fill_tracker"] == "ok"

    async def test_shutdown_empty_registry(self):
        reg = self._make_registry()
        reg._init_order = []
        result = await reg.shutdown(timeout=5.0)
        assert result == {}

    async def test_shutdown_mixed_results(self):
        reg = self._make_registry()
        ok_comp = _MockComponent()
        err_comp = _ErrorComponent()
        reg.fill_tracker = ok_comp
        reg.maker_enforcement = err_comp
        reg._init_order = ["fill_tracker", "maker_enforcement"]
        result = await reg.shutdown(timeout=5.0)
        assert result["fill_tracker"] == "ok"
        assert result["maker_enforcement"] == "error"


# ---------------------------------------------------------------------------
# 2. Per-component health checks
# ---------------------------------------------------------------------------

class TestHealthChecks:
    """Tests for ComponentRegistry.health_check_all()."""

    def _make_registry(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(_FakeConfig())
        return reg

    def test_healthy_component(self):
        reg = self._make_registry()
        reg.fill_tracker = _HealthyComponent()
        reg._init_order = ["fill_tracker"]
        result = reg.health_check_all()
        assert result["fill_tracker"]["status"] == "healthy"

    def test_unhealthy_component(self):
        reg = self._make_registry()
        reg.fill_tracker = _UnhealthyComponent()
        reg._init_order = ["fill_tracker"]
        result = reg.health_check_all()
        assert result["fill_tracker"]["status"] == "unhealthy"

    def test_dict_health_response(self):
        reg = self._make_registry()
        reg.fill_tracker = _DictHealthComponent()
        reg._init_order = ["fill_tracker"]
        result = reg.health_check_all()
        assert result["fill_tracker"]["status"] == "degraded"

    def test_error_health_response(self):
        reg = self._make_registry()
        reg.fill_tracker = _ErrorHealthComponent()
        reg._init_order = ["fill_tracker"]
        result = reg.health_check_all()
        assert result["fill_tracker"]["status"] == "unhealthy"

    def test_no_health_method_fresh_cycle(self):
        """Components without health method use cycle time staleness."""
        reg = self._make_registry()
        reg.fill_tracker = object()
        reg._init_order = ["fill_tracker"]
        reg._last_cycle_time = time.time()
        result = reg.health_check_all()
        assert result["fill_tracker"]["status"] == "healthy"

    def test_no_health_method_stale_cycle(self):
        reg = self._make_registry()
        reg.fill_tracker = object()
        reg._init_order = ["fill_tracker"]
        reg._last_cycle_time = time.time() - 120  # 2 min stale
        result = reg.health_check_all()
        assert result["fill_tracker"]["status"] == "degraded"

    def test_health_check_all_returns_last_active(self):
        reg = self._make_registry()
        reg.fill_tracker = _HealthyComponent()
        reg._init_order = ["fill_tracker"]
        result = reg.health_check_all()
        assert "last_active" in result["fill_tracker"]
        assert isinstance(result["fill_tracker"]["last_active"], float)

    def test_empty_registry_health(self):
        reg = self._make_registry()
        reg._init_order = []
        result = reg.health_check_all()
        assert result == {}


# ---------------------------------------------------------------------------
# 2b. Health server /health/components endpoint
# ---------------------------------------------------------------------------

class TestHealthComponentsEndpoint:
    """Test the /health/components endpoint in HealthServer."""

    def test_endpoint_without_registry_returns_503(self):
        from core.health_server import HealthServer
        import urllib.request
        import urllib.error

        server = HealthServer(host="127.0.0.1", port=0)
        # Use port=0 to get a random free port
        from http.server import HTTPServer
        handler = server._make_handler()
        httpd = HTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health/components")
            try:
                resp = urllib.request.urlopen(req, timeout=3)
                # 503 would raise an error in urlopen
                assert False, "Expected HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 503
                body = json.loads(e.read())
                assert "error" in body
        finally:
            httpd.shutdown()
            thread.join(timeout=3)

    def test_endpoint_with_registry_returns_200(self):
        from core.health_server import HealthServer
        from core.component_registry import ComponentRegistry
        import urllib.request

        server = HealthServer(host="127.0.0.1", port=0)
        reg = ComponentRegistry(_FakeConfig())
        reg._init_order = ["fill_tracker"]
        reg.fill_tracker = _HealthyComponent()
        server.set_component_registry(reg)

        from http.server import HTTPServer
        handler = server._make_handler()
        httpd = HTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health/components", timeout=3
            )
            assert resp.status == 200
            data = json.loads(resp.read())
            assert "fill_tracker" in data
            assert data["fill_tracker"]["status"] == "healthy"
        finally:
            httpd.shutdown()
            thread.join(timeout=3)


# ---------------------------------------------------------------------------
# 3. Component dependency graph
# ---------------------------------------------------------------------------

class TestDependencyGraph:
    """Tests for ComponentRegistry.validate_dependencies()."""

    def _make_registry(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(_FakeConfig())
        return reg

    def test_correct_order_no_violations(self):
        reg = self._make_registry()
        # Simulate correct init order: risk before execution before monitoring
        reg._init_order = [
            "intraday_var", "stress_tester",
            "fill_tracker", "maker_enforcement",
            "latency_tracker", "discord",
        ]
        violations = reg.validate_dependencies()
        assert violations == []

    def test_violation_detected(self):
        """Monitoring before risk should be flagged."""
        reg = self._make_registry()
        reg._init_order = [
            "discord",           # monitoring (group 5)
            "intraday_var",      # risk (group 2)
        ]
        violations = reg.validate_dependencies()
        assert len(violations) > 0
        assert "discord" in violations[0]

    def test_empty_init_order(self):
        reg = self._make_registry()
        reg._init_order = []
        violations = reg.validate_dependencies()
        assert violations == []

    def test_dependency_order_class_attribute(self):
        from core.component_registry import ComponentRegistry
        assert hasattr(ComponentRegistry, "_DEPENDENCY_ORDER")
        assert isinstance(ComponentRegistry._DEPENDENCY_ORDER, list)
        assert len(ComponentRegistry._DEPENDENCY_ORDER) == 6
        assert ComponentRegistry._DEPENDENCY_ORDER[0] == ["process_lock"]

    def test_validate_partial_init(self):
        """Only initialized components are checked — missing ones skipped."""
        reg = self._make_registry()
        reg._init_order = ["fill_tracker"]
        violations = reg.validate_dependencies()
        assert violations == []


# ---------------------------------------------------------------------------
# 4. Config reload timeout
# ---------------------------------------------------------------------------

class TestConfigReloadTimeout:
    """Tests for HotReloadManager timeout wrapper."""

    def _make_manager(self, config_path=None):
        from core.hot_reload import HotReloadManager
        if config_path is None:
            config_path = "nonexistent_config.yaml"
        mgr = HotReloadManager(
            config_path=config_path,
            trading_system=None,
        )
        return mgr

    def test_reload_timeout_attribute(self):
        mgr = self._make_manager()
        assert mgr._reload_timeout == 10.0

    def test_fast_reload_succeeds(self):
        """A reload that completes quickly should succeed."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("strategies:\n  min_signal_confidence: 0.5\n")
            f.flush()
            config_path = f.name

        try:
            system = MagicMock()
            system.config = MagicMock()
            system.config.strategies = {"min_signal_confidence": 0.3}
            from core.hot_reload import HotReloadManager
            mgr = HotReloadManager(
                config_path=config_path,
                trading_system=system,
            )
            result = mgr._do_reload()
            assert result is True
        finally:
            os.unlink(config_path)

    def test_slow_reload_times_out(self):
        """A reload that hangs should time out and revert."""
        from core.hot_reload import HotReloadManager

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("strategies:\n  min_signal_confidence: 0.9\n")
            f.flush()
            config_path = f.name

        try:
            system = MagicMock()
            system.config = MagicMock()
            system.config.strategies = {"min_signal_confidence": 0.3}
            mgr = HotReloadManager(
                config_path=config_path,
                trading_system=system,
            )
            mgr._reload_timeout = 0.05  # Very short timeout

            # Monkey-patch inner reload to be slow (but not too slow for test cleanup)
            stop_event = threading.Event()
            original = mgr._do_reload_inner

            def slow_reload():
                stop_event.wait(timeout=1.0)
                return original()

            mgr._do_reload_inner = slow_reload
            result = mgr._do_reload()
            stop_event.set()  # Release the thread so it cleans up
            assert result is False
        finally:
            os.unlink(config_path)

    def test_snapshot_and_revert(self):
        """Config should be reverted on timeout."""
        from core.hot_reload import HotReloadManager

        system = MagicMock()
        config = MagicMock()
        config.strategies = {"min_signal_confidence": 0.5}
        system.config = config

        mgr = HotReloadManager(
            config_path="nonexistent.yaml",
            trading_system=system,
        )

        snapshot = mgr._snapshot_config()
        assert snapshot is not None
        assert "strategies" in snapshot
        assert snapshot["strategies"]["min_signal_confidence"] == 0.5


# ---------------------------------------------------------------------------
# 5. Process lock crash cleanup
# ---------------------------------------------------------------------------

class TestProcessLockCrashCleanup:
    """Tests for ProcessLock stale PID detection and cleanup."""

    def test_stale_lock_from_dead_pid(self):
        from core.process_lock import ProcessLock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock = ProcessLock(
                name="test_stale", lock_dir=lock_dir, timeout=1.0
            )

            # Write a fake PID that doesn't exist
            lock_path = lock_dir / "test_stale.lock"
            lock_path.write_text("999999999")  # PID that almost certainly doesn't exist

            # Acquire should succeed by cleaning up the stale lock
            assert lock.acquire() is True
            assert lock._acquired is True
            assert lock.get_owner_pid() == os.getpid()
            lock.release()

    def test_live_pid_blocks_acquire(self):
        from core.process_lock import ProcessLock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)

            # Write our own PID — it IS running
            lock_path = lock_dir / "test_live.lock"
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(str(os.getpid()))

            lock = ProcessLock(
                name="test_live", lock_dir=lock_dir, timeout=0.0
            )

            # Should fail — our PID is running and holds the lock
            assert lock.acquire() is False

    def test_pid_is_running_check(self):
        from core.process_lock import ProcessLock

        # Our own PID is definitely running
        assert ProcessLock._pid_is_running(os.getpid()) is True

        # PID 0 should be False
        assert ProcessLock._pid_is_running(0) is False

        # Very large PID should be False (not running)
        assert ProcessLock._pid_is_running(999999999) is False

    def test_context_manager_cleanup(self):
        from core.process_lock import ProcessLock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            with ProcessLock(name="ctx_test", lock_dir=lock_dir) as lock:
                assert lock._acquired is True
                lock_path = lock_dir / "ctx_test.lock"
                assert lock_path.exists()
            # After exit, lock file should be gone
            assert not lock_path.exists()

    def test_malformed_lock_file_treated_as_stale(self):
        from core.process_lock import ProcessLock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_path = lock_dir / "malformed.lock"
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path.write_text("not_a_number")

            lock = ProcessLock(
                name="malformed", lock_dir=lock_dir, timeout=1.0
            )
            assert lock.acquire() is True
            lock.release()


# ---------------------------------------------------------------------------
# 6. Regime store TTL
# ---------------------------------------------------------------------------

def _safe_unlink(path: str) -> None:
    """Best-effort file removal; ignore Windows locking errors."""
    try:
        os.unlink(path)
    except (PermissionError, OSError):
        pass  # Windows may hold SQLite WAL locks


class TestRegimeStoreTTL:
    """Tests for RegimeStore max_age_seconds parameter."""

    def _make_store(self, max_age_seconds=3600):
        from core.regime_store import RegimeStore
        with tempfile.NamedTemporaryFile(
            suffix=".db", delete=False
        ) as f:
            db_path = f.name
        store = RegimeStore(db_path=db_path, max_age_seconds=max_age_seconds)
        return store, db_path

    def test_default_ttl(self):
        from core.regime_store import RegimeStore, MAX_AGE_SECONDS
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = RegimeStore(db_path=db_path)
        assert store._max_age_seconds == MAX_AGE_SECONDS
        _safe_unlink(db_path)

    def test_custom_ttl(self):
        store, db_path = self._make_store(max_age_seconds=120)
        try:
            assert store._max_age_seconds == 120.0
        finally:
            _safe_unlink(db_path)

    def test_fresh_regime_returned(self):
        store, db_path = self._make_store(max_age_seconds=3600)
        try:
            store.save("BTC/USD", "TREND_UP", confidence=0.8, source="test")
            regime, meta = store.load("BTC/USD")
            assert regime == "TREND_UP"
            assert meta["confidence"] == 0.8
        finally:
            _safe_unlink(db_path)

    def test_stale_regime_returns_unknown(self):
        store, db_path = self._make_store(max_age_seconds=1)
        try:
            store.save("BTC/USD", "TREND_UP", confidence=0.8, source="test")

            # Manually backdate the entry in SQLite
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE regime_cache SET ts = ? WHERE symbol = ?",
                (time.time() - 100, "BTC/USD"),
            )
            conn.commit()
            conn.close()

            regime, meta = store.load("BTC/USD")
            assert regime == "UNKNOWN"
            assert meta == {}
        finally:
            _safe_unlink(db_path)

    def test_missing_symbol_returns_unknown(self):
        store, db_path = self._make_store()
        try:
            regime, meta = store.load("DOGE/USD")
            assert regime == "UNKNOWN"
            assert meta == {}
        finally:
            _safe_unlink(db_path)

    def test_ttl_override_in_load(self):
        """Passing max_age_seconds to load() overrides instance default."""
        store, db_path = self._make_store(max_age_seconds=3600)
        try:
            store.save("ETH/USD", "HIGH_VOL", confidence=0.9, source="test")

            # Backdate to 10 seconds ago
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE regime_cache SET ts = ? WHERE symbol = ?",
                (time.time() - 10, "ETH/USD"),
            )
            conn.commit()
            conn.close()

            # With instance TTL of 3600, this is fresh
            regime, meta = store.load("ETH/USD")
            assert regime == "HIGH_VOL"

            # With explicit TTL of 5, this is stale
            regime, meta = store.load("ETH/USD", max_age_seconds=5)
            assert regime == "UNKNOWN"
        finally:
            _safe_unlink(db_path)

    def test_stale_warning_logged(self):
        """Stale regime should log a warning."""
        import logging
        store, db_path = self._make_store(max_age_seconds=1)
        try:
            store.save("BTC/USD", "CRISIS", confidence=0.95, source="test")

            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE regime_cache SET ts = ? WHERE symbol = ?",
                (time.time() - 100, "BTC/USD"),
            )
            conn.commit()
            conn.close()

            with patch("core.regime_store.logger") as mock_logger:
                regime, _ = store.load("BTC/USD")
                assert regime == "UNKNOWN"
                mock_logger.warning.assert_called_once()
                assert "stale" in mock_logger.warning.call_args[0][0].lower()
        finally:
            _safe_unlink(db_path)


# ---------------------------------------------------------------------------
# Integration: init order tracking
# ---------------------------------------------------------------------------

class TestInitOrderTracking:
    """Test that _init_order is populated during _try_init."""

    def test_try_init_appends_to_order(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(_FakeConfig())

        def fake_init():
            reg.fill_tracker = object()

        reg._try_init("fill_tracker", fake_init)
        assert "fill_tracker" in reg._init_order

    def test_failed_init_not_in_order(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(_FakeConfig())

        def bad_init():
            raise ImportError("module not found")

        reg._try_init("nonexistent", bad_init)
        assert "nonexistent" not in reg._init_order


# ---------------------------------------------------------------------------
# Integration: _get_component_by_name
# ---------------------------------------------------------------------------

class TestGetComponentByName:
    def test_regular_component(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(_FakeConfig())
        obj = object()
        reg.fill_tracker = obj
        assert reg._get_component_by_name("fill_tracker") is obj

    def test_multi_venue_executor(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(_FakeConfig())
        obj = object()
        reg._multi_venue_executor = obj
        assert reg._get_component_by_name("multi_venue_executor") is obj

    def test_missing_component(self):
        from core.component_registry import ComponentRegistry
        reg = ComponentRegistry(_FakeConfig())
        assert reg._get_component_by_name("nonexistent_xyz") is None


# ---------------------------------------------------------------------------
# 3b. Dependency graph: get_init_order + get_dependency_status
# ---------------------------------------------------------------------------

class TestDependencyGraphExtended:
    """Tests for get_init_order() and get_dependency_status()."""

    def _make_registry(self):
        from core.component_registry import ComponentRegistry
        return ComponentRegistry(_FakeConfig())

    def test_get_init_order_returns_list(self):
        reg = self._make_registry()
        order = reg.get_init_order()
        assert isinstance(order, list)
        assert len(order) > 0

    def test_get_init_order_topological(self):
        """process_lock must come before regime_store, risk_managers before execution."""
        reg = self._make_registry()
        order = reg.get_init_order()
        assert order.index("process_lock") < order.index("regime_store")
        assert order.index("risk_managers") < order.index("execution")
        assert order.index("execution") < order.index("monitoring")

    def test_get_init_order_all_groups(self):
        reg = self._make_registry()
        order = reg.get_init_order()
        expected = {"process_lock", "regime_store", "position_registry",
                    "risk_managers", "ml_models", "execution", "monitoring"}
        assert set(order) == expected

    def test_get_dependency_status_known_component(self):
        reg = self._make_registry()
        reg._init_order = ["fill_tracker"]
        reg.fill_tracker = _HealthyComponent()
        status = reg.get_dependency_status("fill_tracker")
        assert status["component"] == "fill_tracker"
        assert status["group"] == "execution"
        assert isinstance(status["dependencies"], dict)

    def test_get_dependency_status_unknown_component(self):
        reg = self._make_registry()
        status = reg.get_dependency_status("nonexistent_xyz")
        assert status["group"] is None
        assert status["dependencies"] == {}

    def test_get_dependency_status_shows_dep_health(self):
        reg = self._make_registry()
        reg.intraday_var = _HealthyComponent()
        reg.fill_tracker = _HealthyComponent()
        reg._init_order = ["intraday_var", "fill_tracker"]
        status = reg.get_dependency_status("fill_tracker")
        # fill_tracker is in execution group, depends on risk_managers and ml_models
        assert "risk_managers" in status["dependencies"]

    def test_group_components_class_attribute(self):
        from core.component_registry import ComponentRegistry
        assert hasattr(ComponentRegistry, "_GROUP_COMPONENTS")
        assert "execution" in ComponentRegistry._GROUP_COMPONENTS
        assert "fill_tracker" in ComponentRegistry._GROUP_COMPONENTS["execution"]


# ---------------------------------------------------------------------------
# 4b. Config reload deadlock guard
# ---------------------------------------------------------------------------

class TestConfigReloadDeadlockGuard:
    """Tests for HotReloadManager lock timeout and re-entrancy guard."""

    def test_reload_lock_exists(self):
        from core.hot_reload import HotReloadManager
        mgr = HotReloadManager(config_path="nonexistent.yaml")
        assert hasattr(mgr, "_reload_lock")
        assert hasattr(mgr, "_reload_in_progress")
        assert mgr._reload_in_progress is False

    def test_reload_lock_timeout_default(self):
        from core.hot_reload import HotReloadManager
        mgr = HotReloadManager(config_path="nonexistent.yaml")
        assert mgr._reload_lock_timeout == 5.0

    def test_reentrant_reload_skipped(self):
        from core.hot_reload import HotReloadManager
        mgr = HotReloadManager(config_path="nonexistent.yaml")
        # Simulate a reload already in progress
        mgr._reload_in_progress = True
        result = mgr._do_reload()
        assert result is False

    def test_lock_contention_skips_reload(self):
        from core.hot_reload import HotReloadManager
        mgr = HotReloadManager(config_path="nonexistent.yaml")
        mgr._reload_lock_timeout = 0.01  # Very short timeout
        # Hold the lock from another thread
        mgr._reload_lock.acquire()
        try:
            result = mgr._do_reload()
            assert result is False
        finally:
            mgr._reload_lock.release()

    def test_reload_in_progress_cleared_after_completion(self):
        from core.hot_reload import HotReloadManager
        mgr = HotReloadManager(config_path="nonexistent.yaml")
        # Even failed reloads should clear the flag
        mgr._do_reload()
        assert mgr._reload_in_progress is False


# ---------------------------------------------------------------------------
# 5b. Process lock force_release
# ---------------------------------------------------------------------------

class TestProcessLockForceRelease:
    """Tests for ProcessLock.force_release()."""

    def test_force_release_removes_lock(self):
        from core.process_lock import ProcessLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock = ProcessLock(name="force_test", lock_dir=lock_dir)
            assert lock.acquire() is True
            lock_path = lock_dir / "force_test.lock"
            assert lock_path.exists()
            result = lock.force_release()
            assert result is True
            assert not lock_path.exists()

    def test_force_release_no_file(self):
        from core.process_lock import ProcessLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock = ProcessLock(name="nofile_test", lock_dir=lock_dir)
            result = lock.force_release()
            assert result is False

    def test_force_release_stale_lock(self):
        from core.process_lock import ProcessLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_path = lock_dir / "stale_force.lock"
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path.write_text("999999999")
            lock = ProcessLock(name="stale_force", lock_dir=lock_dir)
            result = lock.force_release()
            assert result is True
            assert not lock_path.exists()

    def test_force_release_then_acquire(self):
        from core.process_lock import ProcessLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock = ProcessLock(name="reacquire", lock_dir=lock_dir)
            lock.acquire()
            lock.force_release()
            # Should be able to acquire again
            lock2 = ProcessLock(name="reacquire", lock_dir=lock_dir)
            assert lock2.acquire() is True
            lock2.release()


# ---------------------------------------------------------------------------
# 6b. Regime store is_stale
# ---------------------------------------------------------------------------

class TestRegimeStoreIsStale:
    """Tests for RegimeStore.is_stale()."""

    def _make_store(self, max_age_seconds=3600):
        from core.regime_store import RegimeStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = RegimeStore(db_path=db_path, max_age_seconds=max_age_seconds)
        return store, db_path

    def test_is_stale_missing_symbol(self):
        store, db_path = self._make_store()
        try:
            assert store.is_stale("NONEXISTENT/USD") is True
        finally:
            _safe_unlink(db_path)

    def test_is_stale_fresh_entry(self):
        store, db_path = self._make_store(max_age_seconds=3600)
        try:
            store.save("BTC/USD", "TREND_UP", confidence=0.8, source="test")
            assert store.is_stale("BTC/USD") is False
        finally:
            _safe_unlink(db_path)

    def test_is_stale_old_entry(self):
        store, db_path = self._make_store(max_age_seconds=1)
        try:
            store.save("BTC/USD", "TREND_UP", confidence=0.8, source="test")
            # Backdate the entry
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE regime_cache SET ts = ? WHERE symbol = ?",
                (time.time() - 100, "BTC/USD"),
            )
            conn.commit()
            conn.close()
            assert store.is_stale("BTC/USD") is True
        finally:
            _safe_unlink(db_path)

    def test_is_stale_with_override(self):
        store, db_path = self._make_store(max_age_seconds=3600)
        try:
            store.save("ETH/USD", "HIGH_VOL", confidence=0.9, source="test")
            # Backdate to 10 seconds ago
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE regime_cache SET ts = ? WHERE symbol = ?",
                (time.time() - 10, "ETH/USD"),
            )
            conn.commit()
            conn.close()
            # With default TTL (3600), not stale
            assert store.is_stale("ETH/USD") is False
            # With short TTL, stale
            assert store.is_stale("ETH/USD", max_age_seconds=5) is True
        finally:
            _safe_unlink(db_path)


# ---------------------------------------------------------------------------
# 1b. Graceful shutdown — unified_trading_system signal handlers
# ---------------------------------------------------------------------------

class TestShutdownSignalHandlers:
    """Tests for _install_signal_handlers and enhanced shutdown in UTS."""

    def test_install_signal_handlers_attribute(self):
        """The unified trading system class should have _install_signal_handlers."""
        # We can't easily import the full UTS without heavy deps, so
        # just verify the method exists on the module level.
        import importlib
        mod = importlib.import_module("unified_trading_system")
        cls_names = [n for n in dir(mod) if "Trading" in n or "System" in n]
        # At minimum, the function should exist somewhere in the module
        source = Path("unified_trading_system.py").read_text(encoding="utf-8")
        assert "_install_signal_handlers" in source
        assert "signal.SIGINT" in source
        assert "signal.SIGTERM" in source

    def test_shutdown_logs_complete(self):
        """shutdown() should log 'shutdown complete'."""
        source = Path("unified_trading_system.py").read_text(encoding="utf-8")
        assert "shutdown complete" in source

    def test_shutdown_cancels_orders(self):
        """shutdown() should attempt to cancel open orders."""
        source = Path("unified_trading_system.py").read_text(encoding="utf-8")
        assert "cancel_all_orders" in source

    def test_shutdown_flushes_ledger(self):
        """shutdown() should flush the trade ledger."""
        source = Path("unified_trading_system.py").read_text(encoding="utf-8")
        assert "Flushed pending fills to trade ledger" in source

    def test_shutdown_saves_regime(self):
        """shutdown() should save regime state."""
        source = Path("unified_trading_system.py").read_text(encoding="utf-8")
        assert "Saved regime state" in source
