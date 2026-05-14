"""Argus configuration schema — Push 61.

All sections are plain dataclasses with from_dict / to_dict.
The top-level ArgusConfig composes all sections.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    debug: bool = False
    reload: bool = False
    log_level: str = "info"

    @classmethod
    def from_dict(cls, d: Dict) -> "ServerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExchangeConfig:
    name: str = "bybit"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    ws_url: str = ""
    rest_url: str = ""
    rate_limit_per_sec: int = 10
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])

    @classmethod
    def from_dict(cls, d: Dict) -> "ExchangeConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Redact secrets in serialised output
        if d.get("api_key"):
            d["api_key"] = "***"
        if d.get("api_secret"):
            d["api_secret"] = "***"
        return d


@dataclass
class RiskSection:
    max_position_usd: float = 10_000.0
    max_daily_loss_usd: float = 500.0
    max_drawdown_pct: float = 10.0
    min_confidence: float = 0.6
    max_open_positions: int = 5
    halt_on_drawdown: bool = True

    @classmethod
    def from_dict(cls, d: Dict) -> "RiskSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_risk_config(self):
        """Convert to core.risk.RiskConfig."""
        from core.risk.risk_config import RiskConfig
        return RiskConfig(
            max_position_usd=self.max_position_usd,
            max_daily_loss_usd=self.max_daily_loss_usd,
            max_drawdown_pct=self.max_drawdown_pct,
            min_confidence=self.min_confidence,
            max_open_positions=self.max_open_positions,
            halt_on_drawdown=self.halt_on_drawdown,
        )


@dataclass
class AlertsConfig:
    enabled: bool = True
    min_level: str = "WARNING"
    telegram_enabled: bool = False
    discord_enabled: bool = False
    email_enabled: bool = False
    webhook_enabled: bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> "AlertsConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BroadcastConfig:
    enabled: bool = True
    heartbeat_interval: float = 10.0
    max_clients: int = 50

    @classmethod
    def from_dict(cls, d: Dict) -> "BroadcastConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    file: Optional[str] = None
    json_logs: bool = False
    rotate_mb: int = 50
    backup_count: int = 5

    @classmethod
    def from_dict(cls, d: Dict) -> "LoggingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)

    def apply(self) -> None:
        """Apply logging configuration to the root logger."""
        import logging
        import logging.handlers
        level = getattr(logging, self.level.upper(), logging.INFO)
        handlers = [logging.StreamHandler()]
        if self.file:
            rh = logging.handlers.RotatingFileHandler(
                self.file,
                maxBytes=self.rotate_mb * 1024 * 1024,
                backupCount=self.backup_count,
            )
            handlers.append(rh)
        logging.basicConfig(level=level, format=self.format, handlers=handlers, force=True)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class ArgusConfig:
    """Top-level Argus configuration."""
    version: str = "1"
    env: str = "production"          # production | paper | backtest
    server: ServerConfig = field(default_factory=ServerConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskSection = field(default_factory=RiskSection)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    broadcast: BroadcastConfig = field(default_factory=BroadcastConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> "ArgusConfig":
        return cls(
            version=str(d.get("version", "1")),
            env=d.get("env", "production"),
            server=ServerConfig.from_dict(d.get("server", {})),
            exchange=ExchangeConfig.from_dict(d.get("exchange", {})),
            risk=RiskSection.from_dict(d.get("risk", {})),
            alerts=AlertsConfig.from_dict(d.get("alerts", {})),
            broadcast=BroadcastConfig.from_dict(d.get("broadcast", {})),
            logging=LoggingConfig.from_dict(d.get("logging", {})),
            extra=d.get("extra", {}),
        )

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "env": self.env,
            "server": self.server.to_dict(),
            "exchange": self.exchange.to_dict(),
            "risk": self.risk.to_dict(),
            "alerts": self.alerts.to_dict(),
            "broadcast": self.broadcast.to_dict(),
            "logging": self.logging.to_dict(),
            "extra": self.extra,
        }

    @classmethod
    def default(cls) -> "ArgusConfig":
        """Return a default configuration instance."""
        return cls()
