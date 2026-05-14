"""config/schema.py — Pydantic v2 validated config schema for Argus.

Covers the runtime fields that actually affect live trading behaviour.
All other config.yaml fields are loaded into `extra` for forward compat.

Usage:
    from config.schema import load_config
    cfg = load_config("config.yaml")
    print(cfg.risk.max_drawdown)     # 0.08 (validated float 0-1)
    print(cfg.system.initial_capital)  # 1000.0 (> 0)
    print(cfg.exchanges.enabled_names)  # ['coinbase', 'kraken']
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic import ValidationError
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    logger.warning(
        "pydantic not installed — config validation disabled. "
        "Run: pip install pydantic>=2.0"
    )


if _PYDANTIC_AVAILABLE:

    class RiskConfig(BaseModel):
        """Validated risk management settings."""
        model_config = {"extra": "allow"}

        max_drawdown: float = Field(default=0.15, ge=0.0, le=1.0,
            description="Max drawdown before halt (0-1)")
        max_daily_loss: float = Field(default=0.05, ge=0.0, le=1.0,
            description="Max daily loss as fraction of equity")
        risk_per_trade: float = Field(default=0.005, ge=0.0, le=0.5,
            description="Risk per trade as fraction of equity")
        stop_loss_default: float = Field(default=0.005, ge=0.0, le=0.5,
            description="Default stop-loss distance")
        max_portfolio_leverage: float = Field(default=3.0, ge=0.0, le=20.0,
            description="Maximum portfolio leverage")
        daily_loss_limit: float = Field(default=0.025, ge=0.0, le=1.0)
        max_drawdown_limit: float = Field(default=0.15, ge=0.0, le=1.0)

        @field_validator("max_drawdown", "max_daily_loss", mode="before")
        @classmethod
        def _coerce_pct(cls, v: Any) -> float:
            """Accept both 0.15 and 15.0 — normalise to fraction."""
            v = float(v)
            if v > 1.0:
                logger.warning("Config: converting %s%% -> %s (fraction)", v, v / 100)
                return v / 100
            return v


    class TradingConfig(BaseModel):
        model_config = {"extra": "allow"}

        capital: float = Field(default=1000.0, gt=0,
            description="Starting capital")
        max_positions: int = Field(default=20, ge=1, le=500)
        min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
        max_daily_trades: int = Field(default=1000, ge=1)
        compound_returns: bool = True


    class SystemConfig(BaseModel):
        model_config = {"extra": "allow"}

        initial_capital: float = Field(default=1000.0, gt=0)
        mode: str = Field(default="dry_run",
            description="'live', 'dry_run', or 'paper'")
        log_level: str = Field(default="INFO")

        @field_validator("mode")
        @classmethod
        def _validate_mode(cls, v: str) -> str:
            allowed = {"live", "dry_run", "paper"}
            if v.lower() not in allowed:
                raise ValueError(f"system.mode must be one of {allowed}, got '{v}'")
            return v.lower()

        @field_validator("log_level")
        @classmethod
        def _validate_log_level(cls, v: str) -> str:
            allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if v.upper() not in allowed:
                raise ValueError(f"log_level must be one of {allowed}")
            return v.upper()


    class SingleExchangeConfig(BaseModel):
        model_config = {"extra": "allow"}

        enabled: bool = True
        api_key: str = Field(default="")
        api_secret: str = Field(default="")
        symbols: List[str] = Field(default_factory=list)

        @field_validator("api_key", "api_secret", mode="before")
        @classmethod
        def _resolve_env(cls, v: Any) -> str:
            """Resolve ${ENV_VAR} references."""
            s = str(v) if v is not None else ""
            if s.startswith("${") and s.endswith("}"):
                env_key = s[2:-1]
                resolved = os.environ.get(env_key, "")
                if not resolved:
                    logger.debug("Config: env var %s not set", env_key)
                return resolved
            return s


    class ExchangesConfig(BaseModel):
        model_config = {"extra": "allow"}

        binance: Optional[SingleExchangeConfig] = None
        coinbase: Optional[SingleExchangeConfig] = None
        kraken: Optional[SingleExchangeConfig] = None
        bybit: Optional[SingleExchangeConfig] = None
        okx: Optional[SingleExchangeConfig] = None
        gate: Optional[SingleExchangeConfig] = None
        huobi: Optional[SingleExchangeConfig] = None

        @property
        def enabled_names(self) -> List[str]:
            names = []
            for field_name in ["binance", "coinbase", "kraken", "bybit", "okx", "gate", "huobi"]:
                cfg = getattr(self, field_name)
                if cfg and cfg.enabled:
                    names.append(field_name)
            return names


    class ArgusConfig(BaseModel):
        """Top-level validated Argus configuration."""
        model_config = {"extra": "allow"}

        system: SystemConfig = Field(default_factory=SystemConfig)
        risk: RiskConfig = Field(default_factory=RiskConfig)
        trading: TradingConfig = Field(default_factory=TradingConfig)
        exchanges: ExchangesConfig = Field(default_factory=ExchangesConfig)

        @model_validator(mode="after")
        def _cross_validate(self) -> "ArgusConfig":
            # Warn if live mode but capital is suspiciously low
            if self.system.mode == "live" and self.trading.capital < 10.0:
                logger.warning(
                    "Config: system.mode=live but capital=%.2f — "
                    "did you mean dry_run?",
                    self.trading.capital,
                )
            return self


else:
    # Stub classes when pydantic is absent
    class ArgusConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class RiskConfig(ArgusConfig): pass  # type: ignore[no-redef]
    class TradingConfig(ArgusConfig): pass  # type: ignore[no-redef]
    class SystemConfig(ArgusConfig): pass  # type: ignore[no-redef]
    class ExchangesConfig(ArgusConfig): pass  # type: ignore[no-redef]


def load_config(path: str | Path = "config.yaml") -> "ArgusConfig":
    """
    Load and validate config.yaml -> ArgusConfig.

    Returns ArgusConfig on success.
    Falls back to a dict-backed stub if pydantic is absent or file missing.
    """
    import yaml  # soft dep — already in requirements.txt

    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file not found: %s — using defaults", path)
        raw: Dict[str, Any] = {}
    else:
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}

    if not _PYDANTIC_AVAILABLE:
        return ArgusConfig(**raw)  # type: ignore[call-arg]

    try:
        cfg = ArgusConfig(
            system=raw.get("system", {}),
            risk=raw.get("risk", {}),
            trading=raw.get("trading", {}),
            exchanges=raw.get("exchanges", {}),
        )
        logger.info(
            "Config loaded: mode=%s capital=%.0f risk.max_dd=%.1f%%",
            cfg.system.mode,
            cfg.trading.capital,
            cfg.risk.max_drawdown * 100,
        )
        return cfg
    except Exception as exc:  # ValidationError or yaml.YAMLError
        logger.error("Config validation error: %s", exc)
        raise
