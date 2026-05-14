"""Push 61 — Config Manager: 26 tests."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# ArgusConfig / schema tests (7)
# ---------------------------------------------------------------------------
from core.config.config_schema import (
    ArgusConfig, ServerConfig, ExchangeConfig,
    RiskSection, AlertsConfig, BroadcastConfig, LoggingConfig,
)


class TestArgusConfig:
    def test_default_creates_instance(self):
        cfg = ArgusConfig.default()
        assert isinstance(cfg, ArgusConfig)

    def test_default_env_is_production(self):
        cfg = ArgusConfig.default()
        assert cfg.env == "production"

    def test_from_dict_roundtrip(self):
        cfg = ArgusConfig.default()
        d = cfg.to_dict()
        cfg2 = ArgusConfig.from_dict(d)
        assert cfg2.env == cfg.env
        assert cfg2.server.port == cfg.server.port

    def test_exchange_secrets_redacted(self):
        cfg = ExchangeConfig(api_key="secret123", api_secret="supersecret")
        d = cfg.to_dict()
        assert d["api_key"] == "***"
        assert d["api_secret"] == "***"

    def test_server_from_dict(self):
        s = ServerConfig.from_dict({"port": 9090, "debug": True})
        assert s.port == 9090
        assert s.debug is True

    def test_risk_to_risk_config(self):
        from core.risk.risk_config import RiskConfig
        r = RiskSection(max_position_usd=5000)
        rc = r.to_risk_config()
        assert isinstance(rc, RiskConfig)
        assert rc.max_position_usd == 5000

    def test_logging_config_keys(self):
        l = LoggingConfig()
        d = l.to_dict()
        assert "level" in d and "rotate_mb" in d


# ---------------------------------------------------------------------------
# EnvResolver tests (6)
# ---------------------------------------------------------------------------
from core.config.env_resolver import EnvResolver


class TestEnvResolver:
    def test_resolve_plain_string(self):
        r = EnvResolver({})
        assert r.resolve("hello") == "hello"

    def test_resolve_env_var(self):
        r = EnvResolver({"MY_VAR": "world"})
        assert r.resolve("${MY_VAR}") == "world"

    def test_resolve_default(self):
        r = EnvResolver({})
        assert r.resolve("${MISSING:-fallback}") == "fallback"

    def test_resolve_env_overrides_default(self):
        r = EnvResolver({"PORT": "9090"})
        assert r.resolve("${PORT:-8080}") == "9090"

    def test_resolve_dict(self):
        r = EnvResolver({"HOST": "localhost"})
        d = r.resolve_dict({"host": "${HOST}", "port": 8080})
        assert d["host"] == "localhost"
        assert d["port"] == 8080

    def test_resolve_nested_list(self):
        r = EnvResolver({"S1": "BTCUSDT"})
        result = r.resolve(["${S1}", "ETHUSDT"])
        assert result == ["BTCUSDT", "ETHUSDT"]

    def test_unresolved_placeholder_preserved(self):
        r = EnvResolver({})
        result = r.resolve("${UNDEFINED_VAR}")
        assert result == "${UNDEFINED_VAR}"


# ---------------------------------------------------------------------------
# ConfigLoader tests (8)
# ---------------------------------------------------------------------------
from core.config.config_loader import ConfigLoader


class TestConfigLoader:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "argus.yaml"
        p.write_text(content)
        return p

    def test_load_defaults_no_file(self):
        loader = ConfigLoader()
        cfg = loader.load(None)
        assert isinstance(cfg, ArgusConfig)

    def test_load_yaml_file(self, tmp_path):
        yaml_content = """
env: paper
server:
  port: 9999
"""
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not available")
        p = self._write_yaml(tmp_path, yaml_content)
        loader = ConfigLoader()
        cfg = loader.load(p)
        assert cfg.env == "paper"
        assert cfg.server.port == 9999

    def test_cached_config_set(self):
        loader = ConfigLoader()
        cfg = loader.load(None)
        assert loader.cached is cfg

    def test_env_override_port(self, monkeypatch):
        monkeypatch.setenv("ARGUS_SERVER_PORT", "7777")
        loader = ConfigLoader()
        cfg = loader.load(None)
        assert cfg.server.port == 7777

    def test_env_override_exchange_name(self, monkeypatch):
        monkeypatch.setenv("ARGUS_EXCHANGE_NAME", "mexc")
        loader = ConfigLoader()
        cfg = loader.load(None)
        assert cfg.exchange.name == "mexc"

    def test_env_override_risk_float(self, monkeypatch):
        monkeypatch.setenv("ARGUS_RISK_MAX_POSITION_USD", "25000")
        loader = ConfigLoader()
        cfg = loader.load(None)
        assert cfg.risk.max_position_usd == pytest.approx(25000.0)

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("ARGUS_LOG_LEVEL", "DEBUG")
        loader = ConfigLoader()
        cfg = loader.load(None)
        assert cfg.logging.level == "DEBUG"

    def test_reload_returns_new_config(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not available")
        p = self._write_yaml(tmp_path, "env: paper\n")
        loader = ConfigLoader()
        cfg1 = loader.load(p)
        p.write_text("env: backtest\n")
        cfg2 = loader.reload()
        assert cfg2.env == "backtest"


# ---------------------------------------------------------------------------
# ConfigWatcher tests (5)
# ---------------------------------------------------------------------------
from core.config.config_watcher import ConfigWatcher


class TestConfigWatcher:
    def test_add_callback(self):
        loader = ConfigLoader()
        loader.load(None)
        watcher = ConfigWatcher(loader)
        called = []
        watcher.add_callback(lambda cfg: called.append(cfg))
        assert len(watcher._callbacks) == 1

    def test_remove_callback(self):
        loader = ConfigLoader()
        loader.load(None)
        watcher = ConfigWatcher(loader)
        fn = lambda cfg: None
        watcher.add_callback(fn)
        watcher.remove_callback(fn)
        assert len(watcher._callbacks) == 0

    def test_reload_count_zero_initially(self):
        loader = ConfigLoader()
        loader.load(None)
        watcher = ConfigWatcher(loader)
        assert watcher.reload_count == 0

    def test_stop_sets_running_false(self):
        loader = ConfigLoader()
        loader.load(None)
        watcher = ConfigWatcher(loader)
        watcher._running = True
        watcher.stop()
        assert watcher._running is False

    def test_no_path_watch_exits_cleanly(self):
        loader = ConfigLoader()
        loader.load(None)  # no path
        watcher = ConfigWatcher(loader, interval=0.01)
        # Should return without hanging
        asyncio.get_event_loop().run_until_complete(
            asyncio.wait_for(watcher.watch(), timeout=1.0)
        )
