"""ConfigLoader — loads, merges, and validates Argus config — Push 61.

Load order (later overrides earlier)::

    1. Built-in defaults (ArgusConfig.default())
    2. YAML file (if path provided)
    3. Environment variable overrides (ARGUS_* prefix)

Environment variable mapping::

    ARGUS_ENV                   -> config.env
    ARGUS_SERVER_HOST           -> config.server.host
    ARGUS_SERVER_PORT           -> config.server.port  (cast to int)
    ARGUS_EXCHANGE_NAME         -> config.exchange.name
    ARGUS_EXCHANGE_API_KEY      -> config.exchange.api_key
    ARGUS_EXCHANGE_API_SECRET   -> config.exchange.api_secret
    ARGUS_EXCHANGE_TESTNET      -> config.exchange.testnet (bool)
    ARGUS_RISK_MAX_POSITION_USD -> config.risk.max_position_usd
    ARGUS_RISK_MAX_DAILY_LOSS   -> config.risk.max_daily_loss_usd
    ARGUS_LOG_LEVEL             -> config.logging.level
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from core.config.config_schema import ArgusConfig
from core.config.env_resolver import EnvResolver

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads ArgusConfig from YAML file + environment overrides."""

    def __init__(
        self,
        resolver: Optional[EnvResolver] = None,
    ) -> None:
        self._resolver = resolver or EnvResolver()
        self._cached: Optional[ArgusConfig] = None
        self._path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Optional[Path] = None) -> ArgusConfig:
        """Load config from file + env overrides. Cache result."""
        self._path = Path(path) if path else None
        raw = self._load_yaml(self._path) if self._path else {}
        raw = self._resolver.resolve_dict(raw)
        raw = self._apply_env_overrides(raw)
        config = ArgusConfig.from_dict(raw)
        self._cached = config
        logger.info(
            "ConfigLoader: loaded config env=%s exchange=%s",
            config.env, config.exchange.name,
        )
        return config

    def reload(self) -> ArgusConfig:
        """Re-load from the same path (hot-reload)."""
        return self.load(self._path)

    @property
    def cached(self) -> Optional[ArgusConfig]:
        return self._cached

    # ------------------------------------------------------------------
    # YAML
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore
        except ImportError:
            logger.warning("ConfigLoader: PyYAML not installed; using empty config")
            return {}
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.debug("ConfigLoader: read YAML from %s", path)
        return data

    # ------------------------------------------------------------------
    # Environment overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Overlay ARGUS_* environment variables onto the raw dict."""
        env = os.environ

        def _set(d: dict, *keys: str, val: Any) -> None:
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val

        mappings = [
            ("ARGUS_ENV",                    ("env",), str),
            ("ARGUS_SERVER_HOST",             ("server", "host"), str),
            ("ARGUS_SERVER_PORT",             ("server", "port"), int),
            ("ARGUS_SERVER_DEBUG",            ("server", "debug"), lambda v: v.lower() in {"1", "true", "yes"}),
            ("ARGUS_EXCHANGE_NAME",           ("exchange", "name"), str),
            ("ARGUS_EXCHANGE_API_KEY",        ("exchange", "api_key"), str),
            ("ARGUS_EXCHANGE_API_SECRET",     ("exchange", "api_secret"), str),
            ("ARGUS_EXCHANGE_TESTNET",        ("exchange", "testnet"), lambda v: v.lower() in {"1", "true", "yes"}),
            ("ARGUS_RISK_MAX_POSITION_USD",   ("risk", "max_position_usd"), float),
            ("ARGUS_RISK_MAX_DAILY_LOSS",     ("risk", "max_daily_loss_usd"), float),
            ("ARGUS_RISK_MAX_DRAWDOWN_PCT",   ("risk", "max_drawdown_pct"), float),
            ("ARGUS_RISK_MIN_CONFIDENCE",     ("risk", "min_confidence"), float),
            ("ARGUS_LOG_LEVEL",              ("logging", "level"), str),
            ("ARGUS_LOG_FILE",               ("logging", "file"), str),
            ("ARGUS_LOG_JSON",               ("logging", "json_logs"), lambda v: v.lower() in {"1", "true", "yes"}),
        ]

        for env_key, path_keys, cast in mappings:
            if env_key in env:
                try:
                    _set(raw, *path_keys, val=cast(env[env_key]))
                except (ValueError, KeyError) as exc:
                    logger.warning("ConfigLoader: invalid env var %s: %s", env_key, exc)

        return raw

    # ------------------------------------------------------------------
    # Convenience: load from env-specified path
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "ConfigLoader":
        """Create loader and immediately load from ARGUS_CONFIG_PATH if set."""
        loader = cls()
        config_path = os.getenv("ARGUS_CONFIG_PATH")
        loader.load(Path(config_path) if config_path else None)
        return loader
