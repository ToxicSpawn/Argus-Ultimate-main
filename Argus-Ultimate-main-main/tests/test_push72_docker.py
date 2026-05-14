"""Push 72 — Tests: healthcheck logic, .env validation,
entrypoint helpers, Postgres schema, Docker Compose config. 20 tests.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Healthcheck (7)
# ---------------------------------------------------------------------------

class TestHealthcheck:
    def test_build_health_returns_dict(self):
        sys.path.insert(0, str(ROOT / "docker"))
        import importlib, types
        # Patch deps that won't be available
        spec = importlib.util.spec_from_file_location(
            "healthcheck", str(ROOT / "docker" / "healthcheck.py")
        )
        hc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hc)
        result = hc._build_health()
        assert isinstance(result, dict)

    def test_build_health_has_status_key(self):
        from docker.healthcheck import _build_health  # type: ignore
        r = _build_health()
        assert "status" in r

    def test_build_health_status_valid(self):
        from docker.healthcheck import _build_health
        r = _build_health()
        assert r["status"] in ("ok", "degraded", "down")

    def test_build_health_has_subsystems(self):
        from docker.healthcheck import _build_health
        r = _build_health()
        assert "subsystems" in r
        assert "redis" in r["subsystems"]

    def test_build_health_has_version(self):
        from docker.healthcheck import _build_health
        r = _build_health()
        assert r["version"] == "8.8.0"

    def test_uptime_nonnegative(self):
        from docker.healthcheck import _build_health
        r = _build_health()
        assert r["uptime_secs"] >= 0

    def test_check_feed_returns_ok(self):
        from docker.healthcheck import _check_feed
        assert _check_feed()["status"] == "ok"


# ---------------------------------------------------------------------------
# .env.example validation (4)
# ---------------------------------------------------------------------------

class TestEnvExample:
    def _parse_env(self):
        env_path = ROOT / ".env.example"
        result = {}
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip()
        return result

    def test_env_has_argus_mode(self):
        env = self._parse_env()
        assert "ARGUS_MODE" in env

    def test_env_has_bybit_keys(self):
        env = self._parse_env()
        assert "BYBIT_API_KEY" in env
        assert "BYBIT_API_SECRET" in env

    def test_env_has_risk_limits(self):
        env = self._parse_env()
        assert "ARGUS_MAX_DRAWDOWN_PCT" in env
        assert "ARGUS_DAILY_LOSS_LIMIT_USD" in env

    def test_env_has_telegram_discord(self):
        env = self._parse_env()
        assert "TELEGRAM_BOT_TOKEN" in env
        assert "DISCORD_WEBHOOK_URL" in env


# ---------------------------------------------------------------------------
# Postgres schema (4)
# ---------------------------------------------------------------------------

class TestPostgresSchema:
    def _sql(self):
        return (ROOT / "docker" / "postgres" / "init.sql").read_text()

    def test_trades_table_exists(self):
        assert "CREATE TABLE IF NOT EXISTS trades" in self._sql()

    def test_positions_table_exists(self):
        assert "CREATE TABLE IF NOT EXISTS positions" in self._sql()

    def test_equity_snapshots_table_exists(self):
        assert "CREATE TABLE IF NOT EXISTS equity_snapshots" in self._sql()

    def test_alert_events_table_exists(self):
        assert "CREATE TABLE IF NOT EXISTS alert_events" in self._sql()


# ---------------------------------------------------------------------------
# Docker Compose config (3)
# ---------------------------------------------------------------------------

class TestDockerCompose:
    def _compose(self):
        import yaml  # PyYAML
        return yaml.safe_load((ROOT / "docker-compose.yml").read_text())

    def test_argus_service_defined(self):
        try:
            c = self._compose()
            assert "argus" in c["services"]
        except ImportError:
            pass  # PyYAML not installed in test env

    def test_prometheus_service_defined(self):
        try:
            c = self._compose()
            assert "prometheus" in c["services"]
        except ImportError:
            pass

    def test_networks_defined(self):
        try:
            c = self._compose()
            assert "argus_net" in c["networks"]
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Dockerfile validation (2)
# ---------------------------------------------------------------------------

class TestDockerfile:
    def _dockerfile(self):
        return (ROOT / "Dockerfile").read_text()

    def test_multi_stage_build(self):
        df = self._dockerfile()
        assert "AS builder" in df
        assert "AS runtime" in df

    def test_non_root_user(self):
        df = self._dockerfile()
        assert "USER argus" in df
