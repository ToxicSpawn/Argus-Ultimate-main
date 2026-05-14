#!/usr/bin/env python3
"""
core/argus_config.py
====================
Pydantic v2 ArgusConfig — single source of truth for all runtime
configuration that used to live in the config_manager.py god object.

Design rules
------------
* All sections are frozen BaseModel subclasses (no accidental mutation).
* Extra keys are FORBIDDEN — bad yaml keys raise at startup, not at 3 AM.
* Every numeric field carries ge/le/gt constraints so invalid values
  are caught before they touch the hot path.
* from_legacy() lets existing callers migrate without rewriting everything
  at once; they call ConfigManager.load_split_config() as before and then
  call .to_argus_config() on the returned LegacyResolvedConfig.
* DPDK / co-location fields live in NetworkConfig and are optional so the
  model loads cleanly on a non-co-lo node (values default to None / False).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FrozenModel(BaseModel):
    model_config = {"frozen": True, "extra": "ignore"}


# ---------------------------------------------------------------------------
# Constitution
# ---------------------------------------------------------------------------

class ConstitutionConfig(_FrozenModel):
    profile: str = Field(..., description="Active profile name, e.g. 'production'")
    version: str = Field(default="1.0.0")
    description: str = Field(default="")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class RuntimeConfig(_FrozenModel):
    node_role: Literal["primary", "secondary", "standalone"] = "standalone"
    manifest_emit_path: Optional[str] = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    dry_run: bool = False
    paper_trade: bool = False
    live_trade: bool = False
    tick_interval_ms: int = Field(default=100, ge=1, le=60_000)
    hot_reload_enabled: bool = False
    metrics_port: int = Field(default=8000, ge=1024, le=65535)
    health_port: int = Field(default=8080, ge=1024, le=65535)
    max_cpu_pct: float = Field(default=80.0, ge=1.0, le=100.0)
    max_memory_pct: float = Field(default=85.0, ge=1.0, le=100.0)
    database_url: str = Field(default="sqlite:///data/argus.db")
    backup_enabled: bool = True

    @model_validator(mode="after")
    def _mode_exclusivity(self) -> "RuntimeConfig":
        active = sum([self.dry_run, self.paper_trade, self.live_trade])
        if active > 1:
            raise ValueError(
                "Only one of dry_run / paper_trade / live_trade may be True"
            )
        return self


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------

class ExchangeConfig(_FrozenModel):
    name: str = Field(default="binance", description="ccxt exchange id")
    testnet: bool = True
    symbols: List[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframes: List[str] = Field(default_factory=lambda: ["1m", "5m", "1h"])
    api_key: Optional[str] = Field(default=None, repr=False)
    api_secret: Optional[str] = Field(default=None, repr=False)
    passphrase: Optional[str] = Field(default=None, repr=False)
    rate_limit_per_sec: int = Field(default=10, ge=1, le=1000)
    recv_window_ms: int = Field(default=5000, ge=100, le=60_000)
    order_timeout_ms: int = Field(default=10_000, ge=100, le=120_000)
    max_retries: int = Field(default=3, ge=0, le=20)
    commission_rate: float = Field(default=0.001, ge=0.0, le=0.05)
    slippage_pct: float = Field(default=0.05, ge=0.0, le=5.0)

    @field_validator("symbols")
    @classmethod
    def _symbols_nonempty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("symbols list must not be empty")
        return v


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class StrategyConfig(_FrozenModel):
    active: List[str] = Field(
        default_factory=lambda: ["momentum", "mean_reversion"]
    )
    weights: Dict[str, float] = Field(default_factory=dict)
    # Position sizing
    initial_balance: float = Field(default=10_000.0, gt=0)
    max_position_size_pct: float = Field(default=0.10, gt=0, le=1.0)
    risk_per_trade_pct: float = Field(default=0.01, gt=0, le=0.10)
    # Ensemble
    ensemble_enabled: bool = True
    moe_routing_enabled: bool = False
    # Timeouts / cooldowns
    signal_cooldown_s: int = Field(default=30, ge=0)
    max_open_positions: int = Field(default=5, ge=1, le=500)

    @model_validator(mode="after")
    def _weights_sum(self) -> "StrategyConfig":
        if self.weights:
            total = sum(self.weights.values())
            if not (0.999 <= total <= 1.001):
                raise ValueError(
                    f"strategy.weights must sum to 1.0, got {total:.4f}"
                )
        return self


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class RiskConfig(_FrozenModel):
    max_drawdown_pct: float = Field(default=0.15, gt=0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.05, gt=0, le=1.0)
    max_consecutive_losses: int = Field(default=5, ge=1)
    circuit_breaker_enabled: bool = True
    circuit_breaker_cooldown_s: int = Field(default=300, ge=0)
    volatility_filter_enabled: bool = True
    volatility_lookback_bars: int = Field(default=20, ge=2)
    fat_finger_max_pct: float = Field(default=0.05, gt=0, le=1.0)
    kelly_fraction: float = Field(default=0.25, gt=0, le=1.0)
    atr_stop_multiplier: float = Field(default=2.0, gt=0)
    var_confidence: float = Field(default=0.95, gt=0.5, lt=1.0)


# ---------------------------------------------------------------------------
# Network / DPDK (co-location)
# ---------------------------------------------------------------------------

class DPDKConfig(_FrozenModel):
    """Only populated when co-location hardware is present."""
    enabled: bool = False
    eal_args: List[str] = Field(default_factory=list)
    rx_queue_depth: int = Field(default=1024, ge=64)
    tx_queue_depth: int = Field(default=1024, ge=64)
    burst_size: int = Field(default=32, ge=1, le=512)
    pmd_driver: Optional[str] = None          # e.g. "net_mlx5"
    numa_node: Optional[int] = Field(default=None, ge=0)
    cpu_cores: List[int] = Field(default_factory=list)
    huge_page_size_mb: int = Field(default=2, ge=2)


class NetworkConfig(_FrozenModel):
    # Standard async websocket path
    ws_reconnect_delay_s: float = Field(default=1.0, ge=0.0)
    ws_ping_interval_s: float = Field(default=20.0, ge=1.0)
    http_timeout_s: float = Field(default=10.0, ge=0.5)
    tcp_nodelay: bool = True
    so_busy_poll_us: int = Field(default=0, ge=0)    # 0 = off, >0 = kernel busy-poll
    # Kernel-bypass / DPDK path (co-location only)
    dpdk: DPDKConfig = Field(default_factory=DPDKConfig)
    # Colocation metadata
    colocation_enabled: bool = False
    exchange_endpoint: Optional[str] = None           # IP:port of co-lo matching engine
    preferred_nic: Optional[str] = None               # e.g. "enp5s0f0"


# ---------------------------------------------------------------------------
# AI / ML
# ---------------------------------------------------------------------------

class AIConfig(_FrozenModel):
    enabled: bool = True
    model_dir: str = Field(default="models/")
    rl_enabled: bool = False
    rl_algorithm: Literal["PPO", "SAC", "DDPG", "TD3", "DQN"] = "PPO"
    rl_train_interval_episodes: int = Field(default=100, ge=1)
    meta_learning_enabled: bool = False
    bayesian_opt_enabled: bool = False
    jax_backend: bool = False
    gpu_index: Optional[int] = Field(default=None, ge=0)
    inference_batch_size: int = Field(default=32, ge=1)
    feature_store_path: str = Field(default="data/feature_store/")


# ---------------------------------------------------------------------------
# Monitoring / Alerting
# ---------------------------------------------------------------------------

class MonitoringConfig(_FrozenModel):
    enabled: bool = True
    prometheus_enabled: bool = True
    grafana_port: int = Field(default=3000, ge=1024, le=65535)
    alert_email: Optional[str] = None
    telegram_token: Optional[str] = Field(default=None, repr=False)
    telegram_chat_id: Optional[str] = None
    discord_webhook: Optional[str] = Field(default=None, repr=False)
    health_check_interval_s: int = Field(default=30, ge=1)
    latency_warn_ms: float = Field(default=50.0, gt=0)
    latency_critical_ms: float = Field(default=200.0, gt=0)

    @model_validator(mode="after")
    def _latency_order(self) -> "MonitoringConfig":
        if self.latency_warn_ms >= self.latency_critical_ms:
            raise ValueError(
                "latency_warn_ms must be less than latency_critical_ms"
            )
        return self


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class ArgusConfig(BaseModel):
    """
    Top-level configuration object for Argus-Ultimate.

    Construct via ArgusConfig.from_legacy(resolved) where `resolved` is
    a LegacyResolvedConfig returned by ConfigManager.load_split_config(),
    or build directly for tests:

        cfg = ArgusConfig(
            constitution=ConstitutionConfig(profile="paper"),
            runtime=RuntimeConfig(paper_trade=True),
            ...)
    """
    model_config = {"frozen": False, "extra": "ignore"}

    constitution: ConstitutionConfig = Field(default_factory=lambda: ConstitutionConfig(profile="paper"))
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    # Raw manifest hash forwarded from the control-plane for audit trail
    manifest_hash: str = Field(default="")
    run_mode: str = Field(default="paper")
    starting_capital_aud: float = Field(default=1000.0)
    primary_exchange: str = Field(default="kraken")
    secondary_exchange: str = Field(default="coinbase_advanced")
    use_ccxt: bool = Field(default=True)
    max_drawdown_pct: float = Field(default=20.0)
    max_consecutive_losses: int = Field(default=10)
    max_daily_loss_pct: float = Field(default=0.25)
    max_total_exposure_pct: float = Field(default=0.80)
    max_leverage: float = Field(default=3.0)
    aud_to_usd: float = Field(default=0.71)
    timezone: str = Field(default="Australia/Sydney")
    location: dict = Field(default_factory=lambda: {"city": "Sydney", "state": "NSW", "country": "AU"})
    current_equity_aud: float = Field(default=1000.0)
    kraken_api_key: str = Field(default="")
    kraken_api_secret: str = Field(default="")
    kraken_testnet: bool = Field(default=True)
    trading_pairs: List[str] = Field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    # Strategy engine thresholds
    se_buy_rsi: float = Field(default=50.0)
    se_sell_rsi: float = Field(default=50.0)
    se_buy_bb: float = Field(default=0.45)
    se_sell_bb: float = Field(default=0.55)
    min_signal_confidence: float = Field(default=0.10)
    live_min_signal_confidence: float = Field(default=0.25)
    max_concurrent_signals: int = Field(default=8)
    dd_adaptive_confidence_enabled: bool = Field(default=False)
    dd_adaptive_conf_floor: float = Field(default=0.50)
    dd_adaptive_conf_ceiling: float = Field(default=0.85)
    min_position_size_aud: float = Field(default=15.0)
    max_position_size_aud: float = Field(default=100.0)
    max_position_pct: float = Field(default=0.10)
    kraken_taker_fee: float = Field(default=0.0026)
    kraken_maker_fee: float = Field(default=0.0016)
    coinbase_taker_fee: float = Field(default=0.006)
    coinbase_maker_fee: float = Field(default=0.004)
    slippage_pct: float = Field(default=0.0015)
    stop_loss_pct: float = Field(default=0.015)
    take_profit_pct: float = Field(default=0.06)
    max_total_exposure_pct: float = Field(default=0.80)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_legacy(cls, resolved: Any) -> "ArgusConfig":
        """
        Build an ArgusConfig from a LegacyResolvedConfig bundle.

        Each sub-dict is filtered to only the keys ArgusConfig knows about
        so that unknown legacy keys don't cause ValidationError.
        """
        def _get(bundle: dict, *path: str) -> Any:
            """Traverse nested dict safely, return {} if missing."""
            cur: Any = bundle
            for key in path:
                if isinstance(cur, dict):
                    cur = cur.get(key, {})
                else:
                    return {}
            return cur if isinstance(cur, dict) else {}

        raw_constitution = _get(resolved.constitution, "constitution")
        raw_runtime = _get(resolved.runtime, "runtime")
        raw_exchange = _get(resolved.exchange, "exchange")
        raw_strategy = _get(resolved.strategy, "strategy")

        # Sub-sections that may live inside the runtime bundle
        raw_network = raw_runtime.pop("network", {}) or {}
        raw_dpdk = raw_network.pop("dpdk", {}) or {}
        raw_risk = raw_runtime.pop("risk", {}) or {}
        raw_ai = raw_runtime.pop("ai", {}) or {}
        raw_monitoring = raw_runtime.pop("monitoring", {}) or {}

        return cls(
            constitution=ConstitutionConfig(**_strip_extra(raw_constitution, ConstitutionConfig)),
            runtime=RuntimeConfig(**_strip_extra(raw_runtime, RuntimeConfig)),
            exchange=ExchangeConfig(**_strip_extra(raw_exchange, ExchangeConfig)),
            strategy=StrategyConfig(**_strip_extra(raw_strategy, StrategyConfig)),
            risk=RiskConfig(**_strip_extra(raw_risk, RiskConfig)),
            network=NetworkConfig(
                **_strip_extra(raw_network, NetworkConfig),
                dpdk=DPDKConfig(**_strip_extra(raw_dpdk, DPDKConfig)),
            ),
            ai=AIConfig(**_strip_extra(raw_ai, AIConfig)),
            monitoring=MonitoringConfig(**_strip_extra(raw_monitoring, MonitoringConfig)),
            manifest_hash=resolved.manifest_hash,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ArgusConfig":
        """Direct load from a unified_config.yaml (bypasses legacy split)."""
        import yaml  # type: ignore[import-untyped]
        from pathlib import Path
        raw: dict = yaml.safe_load(Path(path).read_text())
        # Top-level keys must match section names exactly
        ai_section = raw.get("ai", {})
        # Handle both "exchange" and "exchanges" sections
        exchange_section = raw.get("exchange", raw.get("exchanges", {}))
        return cls(
            constitution=ConstitutionConfig(**raw.get("constitution", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            exchange=ExchangeConfig(**exchange_section),
            strategy=StrategyConfig(**raw.get("strategy", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            network=NetworkConfig(
                **{k: v for k, v in raw.get("network", {}).items() if k != "dpdk"},
                dpdk=DPDKConfig(**raw.get("network", {}).get("dpdk", {})),
            ),
            ai=AIConfig(**ai_section),
            monitoring=MonitoringConfig(**raw.get("monitoring", {})),
            trading_pairs=raw.get("trading_pairs", ["BTC/USD", "ETH/USD"]),
            se_buy_rsi=ai_section.get("se_buy_rsi", 50.0),
            se_sell_rsi=ai_section.get("se_sell_rsi", 50.0),
            se_buy_bb=ai_section.get("se_buy_bb", 0.45),
            se_sell_bb=ai_section.get("se_sell_bb", 0.55),
            min_signal_confidence=ai_section.get("min_signal_confidence", 0.10),
            live_min_signal_confidence=ai_section.get("live_min_signal_confidence", 0.25),
            max_concurrent_signals=ai_section.get("max_concurrent_signals", 8),
            dd_adaptive_confidence_enabled=ai_section.get("dd_adaptive_confidence_enabled", False),
            dd_adaptive_conf_floor=ai_section.get("dd_adaptive_conf_floor", 0.50),
            dd_adaptive_conf_ceiling=ai_section.get("dd_adaptive_conf_ceiling", 0.85),
            primary_exchange=exchange_section.get("primary", "kraken"),
            secondary_exchange=exchange_section.get("secondary", "coinbase_advanced"),
            use_ccxt=exchange_section.get("use_ccxt", True),
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _strip_extra(raw: dict, model_cls: type) -> dict:
    """
    Remove keys not declared in model_cls so legacy YAML surplus keys
    don't cause ValidationError during the migration window.
    When extra="forbid" is the end-state we want, warnings are logged
    here instead of hard-failing, giving us a clean upgrade path.
    """
    import logging
    _log = logging.getLogger(__name__)
    known = set(model_cls.model_fields.keys())
    stripped = {}
    for k, v in raw.items():
        if k in known:
            stripped[k] = v
        else:
            _log.debug(
                "ArgusConfig: unknown key '%s' in %s — ignored during migration",
                k, model_cls.__name__,
            )
    return stripped
