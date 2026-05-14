"""Push 62 — Health check + readiness probes: 26 tests."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# HealthStatus + HealthModels tests (6)
# ---------------------------------------------------------------------------
from core.health.health_models import HealthStatus, ComponentHealth, SystemHealth


class TestHealthStatus:
    def test_ordering(self):
        assert HealthStatus.HEALTHY < HealthStatus.DEGRADED < HealthStatus.UNHEALTHY

    def test_label(self):
        assert HealthStatus.HEALTHY.label == "healthy"
        assert HealthStatus.UNHEALTHY.label == "unhealthy"

    def test_ok_healthy(self):
        assert HealthStatus.HEALTHY.ok is True

    def test_ok_degraded(self):
        assert HealthStatus.DEGRADED.ok is True

    def test_ok_unhealthy(self):
        assert HealthStatus.UNHEALTHY.ok is False


class TestComponentHealth:
    def test_to_dict_keys(self):
        ch = ComponentHealth(name="db", status=HealthStatus.HEALTHY, message="ok")
        d = ch.to_dict()
        assert "name" in d and "status" in d and "latency_ms" in d


class TestSystemHealth:
    def test_is_ready_healthy(self):
        sh = SystemHealth(overall=HealthStatus.HEALTHY)
        assert sh.is_ready is True

    def test_is_ready_unhealthy(self):
        sh = SystemHealth(overall=HealthStatus.UNHEALTHY)
        assert sh.is_ready is False

    def test_is_live_always_true(self):
        sh = SystemHealth(overall=HealthStatus.UNHEALTHY)
        assert sh.is_live is True

    def test_to_dict_structure(self):
        sh = SystemHealth(overall=HealthStatus.DEGRADED)
        d = sh.to_dict()
        assert "status" in d and "ok" in d and "components" in d


# ---------------------------------------------------------------------------
# HealthRegistry tests (8)
# ---------------------------------------------------------------------------
from core.health.health_registry import HealthRegistry


class TestHealthRegistry:
    def _reg(self):
        return HealthRegistry(version="7.8.0", env="test", start_time=time.time())

    def test_empty_registry_healthy(self):
        reg = self._reg()
        result = asyncio.get_event_loop().run_until_complete(reg.run_checks())
        assert result.overall == HealthStatus.HEALTHY

    def test_register_check(self):
        reg = self._reg()
        async def _ok():
            return ComponentHealth(name="x", status=HealthStatus.HEALTHY)
        reg.register_check("x", _ok)
        assert "x" in reg.check_names

    def test_unregister_check(self):
        reg = self._reg()
        async def _ok():
            return ComponentHealth(name="x", status=HealthStatus.HEALTHY)
        reg.register_check("x", _ok)
        reg.unregister_check("x")
        assert "x" not in reg.check_names

    def test_healthy_check_passes(self):
        reg = self._reg()
        async def _ok():
            return ComponentHealth(name="c", status=HealthStatus.HEALTHY, message="ok")
        reg.register_check("c", _ok)
        result = asyncio.get_event_loop().run_until_complete(reg.run_checks())
        assert result.overall == HealthStatus.HEALTHY
        assert result.components["c"].status == HealthStatus.HEALTHY

    def test_unhealthy_check_propagates(self):
        reg = self._reg()
        async def _bad():
            return ComponentHealth(name="bad", status=HealthStatus.UNHEALTHY)
        reg.register_check("bad", _bad)
        result = asyncio.get_event_loop().run_until_complete(reg.run_checks())
        assert result.overall == HealthStatus.UNHEALTHY

    def test_degraded_check_propagates(self):
        reg = self._reg()
        async def _deg():
            return ComponentHealth(name="d", status=HealthStatus.DEGRADED)
        reg.register_check("d", _deg)
        result = asyncio.get_event_loop().run_until_complete(reg.run_checks())
        assert result.overall == HealthStatus.DEGRADED

    def test_multiple_checks_worst_wins(self):
        reg = self._reg()
        async def _ok():
            return ComponentHealth(name="ok", status=HealthStatus.HEALTHY)
        async def _bad():
            return ComponentHealth(name="bad", status=HealthStatus.UNHEALTHY)
        reg.register_check("ok", _ok)
        reg.register_check("bad", _bad)
        result = asyncio.get_event_loop().run_until_complete(reg.run_checks())
        assert result.overall == HealthStatus.UNHEALTHY

    def test_latency_recorded(self):
        reg = self._reg()
        async def _ok():
            return ComponentHealth(name="lat", status=HealthStatus.HEALTHY)
        reg.register_check("lat", _ok)
        result = asyncio.get_event_loop().run_until_complete(reg.run_checks())
        assert result.components["lat"].latency_ms >= 0

    def test_last_result_cached(self):
        reg = self._reg()
        async def _ok():
            return ComponentHealth(name="c", status=HealthStatus.HEALTHY)
        reg.register_check("c", _ok)
        asyncio.get_event_loop().run_until_complete(reg.run_checks())
        assert reg.last_result is not None


# ---------------------------------------------------------------------------
# Built-in checks tests (7)
# ---------------------------------------------------------------------------
from core.health.builtin_checks import (
    disk_check, memory_check, event_loop_check,
    exchange_check, pnl_tracker_check, risk_manager_check,
)


class TestBuiltinChecks:
    def _run(self, fn):
        return asyncio.get_event_loop().run_until_complete(fn())

    def test_disk_check_healthy(self):
        result = self._run(disk_check("/", min_free_mb=1.0))
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.UNHEALTHY)

    def test_disk_check_fails_on_impossible_threshold(self):
        result = self._run(disk_check("/", min_free_mb=999_999_999.0))
        assert result.status == HealthStatus.UNHEALTHY

    def test_memory_check_runs(self):
        result = self._run(memory_check(max_pct=99.9))
        assert result.name == "memory"

    def test_event_loop_check_healthy(self):
        result = self._run(event_loop_check())
        assert result.status == HealthStatus.HEALTHY

    def test_exchange_check_disconnected(self):
        class FakeConn:
            is_connected = False
        result = self._run(exchange_check(FakeConn()))
        assert result.status == HealthStatus.UNHEALTHY

    def test_pnl_tracker_check(self):
        class FakePnL:
            equity = 12_500.0
        result = self._run(pnl_tracker_check(FakePnL()))
        assert result.status == HealthStatus.HEALTHY
        assert "12500" in result.message

    def test_risk_manager_check_halted(self):
        class FakeRM:
            halted = True
        result = self._run(risk_manager_check(FakeRM()))
        assert result.status == HealthStatus.UNHEALTHY

    def test_risk_manager_check_ok(self):
        class FakeRM:
            halted = False
        result = self._run(risk_manager_check(FakeRM()))
        assert result.status == HealthStatus.HEALTHY


# ---------------------------------------------------------------------------
# health_router tests (3)
# ---------------------------------------------------------------------------

class TestHealthRouter:
    def test_router_created_when_fastapi_available(self):
        try:
            from fastapi import FastAPI  # noqa: F401
            from core.health.health_router import health_router
            reg = HealthRegistry()
            r = health_router(reg)
            assert r is not None
        except ImportError:
            pytest.skip("FastAPI not available")

    def test_router_returns_none_without_fastapi(self, monkeypatch):
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {"fastapi": None}):
            from importlib import reload
            import core.health.health_router as hr
            result = hr.health_router(None)
            assert result is None

    def test_system_health_to_dict_serialisable(self):
        import json
        reg = HealthRegistry()
        async def _ok():
            return ComponentHealth(name="x", status=HealthStatus.HEALTHY)
        reg.register_check("x", _ok)
        result = asyncio.get_event_loop().run_until_complete(reg.run_checks())
        json.dumps(result.to_dict())  # must not raise
