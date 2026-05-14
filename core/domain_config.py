"""
Domain-specific configuration dataclasses.

Splits the monolithic UnifiedConfig (200+ fields) into focused,
validated domain configs. Each can be independently validated.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapitalConfig:
    """Capital and position sizing configuration."""
    starting_capital_aud: float = 1000.0
    aud_to_usd: float = 0.65
    max_position_pct: float = 0.15
    min_position_size_aud: float = 10.0
    max_concurrent_positions: int = 5
    max_total_exposure_pct: float = 0.80

    def validate(self) -> List[str]:
        errors = []
        if self.starting_capital_aud <= 0:
            errors.append("starting_capital_aud must be > 0")
        if self.aud_to_usd <= 0 or self.aud_to_usd > 5:
            errors.append(f"aud_to_usd={self.aud_to_usd} — expected 0.3-2.0")
        if self.max_position_pct <= 0 or self.max_position_pct > 1:
            errors.append(f"max_position_pct={self.max_position_pct} — expected 0-1")
        if self.max_position_pct > self.max_total_exposure_pct:
            errors.append(f"max_position_pct ({self.max_position_pct}) > max_total_exposure_pct ({self.max_total_exposure_pct})")
        if self.max_concurrent_positions < 1:
            errors.append("max_concurrent_positions must be >= 1")
        return errors


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration."""
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    circuit_breaker_consecutive_losses: int = 5
    var_limit_pct: float = 0.05

    def validate(self) -> List[str]:
        errors = []
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > 1:
            errors.append(f"max_daily_loss_pct={self.max_daily_loss_pct} — expected 0-1")
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 1:
            errors.append(f"max_drawdown_pct={self.max_drawdown_pct} — expected 0-1")
        if self.stop_loss_pct >= self.take_profit_pct:
            errors.append(f"stop_loss_pct ({self.stop_loss_pct}) >= take_profit_pct ({self.take_profit_pct}) — R:R <= 1:1")
        if self.max_daily_loss_pct > self.max_drawdown_pct:
            errors.append(f"max_daily_loss_pct ({self.max_daily_loss_pct}) > max_drawdown_pct ({self.max_drawdown_pct})")
        return errors


@dataclass(frozen=True)
class ExecutionConfig:
    """Order execution configuration."""
    order_type: str = "limit"
    retry_attempts: int = 3
    max_slippage_pct: float = 0.01
    max_spread_bps: float = 50.0
    use_twap_for_large_orders: bool = False
    twap_min_notional_usd: float = 250.0
    twap_duration_minutes: float = 5.0
    multi_venue_enabled: bool = True
    multi_venue_min_notional_aud: float = 200.0

    def validate(self) -> List[str]:
        errors = []
        if self.order_type not in ("market", "limit", "vwap", "twap"):
            errors.append(f"order_type={self.order_type} — expected market/limit/vwap/twap")
        if self.max_slippage_pct < 0 or self.max_slippage_pct > 0.10:
            errors.append(f"max_slippage_pct={self.max_slippage_pct} — expected 0-10%")
        if self.retry_attempts < 0:
            errors.append("retry_attempts must be >= 0")
        return errors


@dataclass(frozen=True)
class ExchangeConfig:
    """Exchange connection configuration."""
    primary_exchange: str = "kraken"
    secondary_exchange: str = "coinbase"
    trading_pairs: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    health_check_interval: float = 60.0

    def validate(self) -> List[str]:
        errors = []
        if not self.trading_pairs:
            errors.append("trading_pairs is empty — no symbols to trade")
        if self.primary_exchange == self.secondary_exchange:
            errors.append("primary and secondary exchanges are the same")
        return errors


def validate_all(*configs) -> List[str]:
    """Validate multiple domain configs. Returns list of all errors."""
    errors = []
    for cfg in configs:
        if hasattr(cfg, "validate"):
            errors.extend(cfg.validate())
    return errors
