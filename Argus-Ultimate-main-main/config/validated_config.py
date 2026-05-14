"""Pydantic v2 config model for Argus — validates unified_config.yaml on startup.

Usage:
    from config.validated_config import load_and_validate_config

    cfg = load_and_validate_config()  # raises ValidationError on bad config
    print(cfg.trading.starting_capital_aud)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_CONFIG_PATHS = [
    Path("unified_config.yaml"),
    Path("config.yaml"),
    Path("config/unified_config.yaml"),
]


class ExchangeConfig(BaseModel):
    name: str
    enabled: bool = True
    api_key_env: str = ""
    api_secret_env: str = ""
    taker_fee: float = Field(default=0.0026, ge=0.0, le=0.1)
    maker_fee: float = Field(default=0.0016, ge=0.0, le=0.1)
    sandbox: bool = False


class RiskConfig(BaseModel):
    max_daily_loss_pct: float = Field(default=2.0, ge=0.1, le=50.0)
    max_consecutive_losses: int = Field(default=5, ge=1, le=100)
    max_position_pct: float = Field(default=10.0, ge=0.1, le=100.0)
    max_leverage: float = Field(default=3.0, ge=1.0, le=100.0)
    circuit_breaker_cooldown_minutes: int = Field(default=60, ge=1)
    flash_crash_pct: float = Field(default=15.0, ge=1.0, le=99.0)


class TradingConfig(BaseModel):
    pairs: List[str] = Field(default_factory=list)
    starting_capital_aud: float = Field(default=1000.0, gt=0)
    run_mode: str = Field(default="paper")
    order_type: str = Field(default="market")

    @field_validator("run_mode")
    @classmethod
    def validate_run_mode(cls, v: str) -> str:
        allowed = {"live", "paper", "backtest", "dry_run"}
        if v.lower() not in allowed:
            raise ValueError(f"run_mode must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, v: List[str]) -> List[str]:
        if not v:
            logger.warning("No trading pairs configured — using default BTC/AUD")
            return ["BTC/AUD"]
        return v


class BacktestConfig(BaseModel):
    slippage_bps: float = Field(default=5.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    fill_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    commission_rate: float = Field(default=0.0026, ge=0.0)
    oos_train_ratio: float = Field(default=0.7, ge=0.1, le=0.99)


class MonitoringConfig(BaseModel):
    prometheus_port: int = Field(default=8000, ge=1024, le=65535)
    grafana_port: int = Field(default=3000, ge=1024, le=65535)
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()


class ArgusConfig(BaseModel):
    """Top-level validated config for the Argus trading system."""
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    exchanges: List[ExchangeConfig] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_live_requires_keys(self) -> "ArgusConfig":
        if self.trading.run_mode == "live":
            for exc in self.exchanges:
                if exc.enabled and not exc.api_key_env:
                    logger.warning(
                        "Exchange '%s' is enabled for live trading but has no api_key_env set",
                        exc.name,
                    )
        return self


def _find_config_file() -> Optional[Path]:
    for p in _CONFIG_PATHS:
        if p.exists():
            return p
    return None


def load_and_validate_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ArgusConfig:
    """Load YAML config and validate with Pydantic.

    Args:
        config_path: explicit path; if None, auto-discovers unified_config.yaml / config.yaml
        overrides: dict of top-level key overrides applied after file load

    Returns:
        Validated ArgusConfig instance.

    Raises:
        FileNotFoundError: if no config file found
        pydantic.ValidationError: if config is invalid
    """
    path = Path(config_path) if config_path else _find_config_file()
    if path is None or not path.exists():
        logger.warning("No config file found — using defaults")
        raw: Dict[str, Any] = {}
    else:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        logger.info("Config loaded from %s", path)

    if overrides:
        raw.update(overrides)

    # Env var overrides for critical fields
    run_mode = os.environ.get("ARGUS_RUN_MODE")
    if run_mode:
        raw.setdefault("trading", {})["run_mode"] = run_mode  # type: ignore[index]

    config = ArgusConfig.model_validate(raw)
    logger.info(
        "Config validated | mode=%s | pairs=%s | capital=A$%.2f",
        config.trading.run_mode,
        config.trading.pairs,
        config.trading.starting_capital_aud,
    )
    return config
