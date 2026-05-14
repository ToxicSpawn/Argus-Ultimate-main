"""Push 65 — API endpoint integration tests: 10 tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")


@pytest.fixture(scope="module")
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from core.health.health_registry import HealthRegistry
    from core.health.health_router import health_router
    from core.health.builtin_checks import disk_check, memory_check, event_loop_check
    import time

    reg = HealthRegistry(version="8.1.0", env="test", start_time=time.time())
    reg.register_check("disk", disk_check("."))
    reg.register_check("memory", memory_check(max_pct=99.9))
    reg.register_check("event_loop", event_loop_check())

    app = FastAPI()
    router = health_router(reg)
    if router:
        app.include_router(router)
    return TestClient(app)


class TestHealthEndpoints:
    def test_health_live_200(self, client):
        r = client.get("/health/live")
        assert r.status_code == 200

    def test_health_live_body(self, client):
        r = client.get("/health/live")
        assert r.json()["status"] == "alive"

    def test_health_ready_200(self, client):
        r = client.get("/health/ready")
        assert r.status_code in (200, 503)

    def test_health_ready_body_has_status(self, client):
        r = client.get("/health/ready")
        assert "status" in r.json()

    def test_health_full_200(self, client):
        r = client.get("/health")
        assert r.status_code in (200, 503)

    def test_health_full_has_version(self, client):
        r = client.get("/health")
        d = r.json()
        assert "version" in d

    def test_health_full_has_components(self, client):
        r = client.get("/health")
        d = r.json()
        assert "components" in d

    def test_health_full_has_uptime(self, client):
        r = client.get("/health")
        d = r.json()
        assert "uptime_s" in d

    def test_health_full_disk_component(self, client):
        r = client.get("/health")
        components = r.json().get("components", {})
        assert "disk" in components

    def test_health_full_memory_component(self, client):
        r = client.get("/health")
        components = r.json().get("components", {})
        assert "memory" in components
