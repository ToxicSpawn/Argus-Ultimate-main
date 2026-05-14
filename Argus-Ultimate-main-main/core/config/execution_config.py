"""
core/config/execution_config.py
================================
Pydantic-validated execution layer configuration.
Extracted from the monolithic config_manager.py god object.

All fields have sensible defaults so existing code importing
config_manager still works — this module provides a clean migration path.
"""

from __future__ import annotations

from typing import Literal, Optional

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseModel, Field, validator as field_validator  # type: ignore
        PYDANTIC_V2 = False
    except ImportError:
        from dataclasses import dataclass as BaseModel  # type: ignore
        Field = lambda *a, **kw: None  # type: ignore
        PYDANTIC_V2 = False


class ExecutionConfig(BaseModel):
    """All settings governing the execution engine and order routing."""

    # Order routing
    order_router_mode: Literal["live", "paper", "dry_run"] = "dry_run"
    kernel_bypass_enabled: bool = False
    kernel_bypass_mode: Literal["dpdk", "rdma", "stub"] = "stub"
    kernel_bypass_socket: str = "/tmp/argus_bypass.sock"

    # Execution engine
    max_concurrency: int = Field(default=4, ge=1, le=64)
    dry_run: bool = True
    execution_timeout_s: float = Field(default=5.0, gt=0)

    # Slippage / fill model
    slippage_model: Literal["zero", "fixed", "lob_aware"] = "fixed"
    fixed_slippage_bps: float = Field(default=2.0, ge=0)

    # ArgusAI adapter
    min_signal_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    batch_dispatch_enabled: bool = True
    batch_max_concurrency: int = Field(default=4, ge=1, le=32)

    # Logging
    log_fills: bool = True
    log_rejections: bool = True

    class Config:
        extra = "forbid"
        validate_assignment = True


class RiskConfig(BaseModel):
    """Risk management configuration — extracted from config_manager."""

    # Position limits
    max_position_usd: float = Field(default=10_000.0, gt=0)
    max_open_orders: int = Field(default=10, ge=1)
    max_daily_drawdown_pct: float = Field(default=5.0, gt=0, le=100)
    max_single_trade_pct: float = Field(default=2.0, gt=0, le=100)

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    cb_loss_threshold_usd: float = Field(default=500.0, gt=0)
    cb_cooldown_s: float = Field(default=300.0, ge=0)

    # Fat finger
    fat_finger_enabled: bool = True
    fat_finger_max_qty_multiplier: float = Field(default=10.0, gt=1)
    fat_finger_max_price_deviation_pct: float = Field(default=5.0, gt=0)

    # Kelly sizing
    kelly_enabled: bool = True
    kelly_fraction: float = Field(default=0.25, gt=0, le=1.0)
    kelly_lookback_trades: int = Field(default=50, ge=10)

    class Config:
        extra = "forbid"
        validate_assignment = True


class AIConfig(BaseModel):
    """AI/ML subsystem configuration."""

    # RL trainer
    jax_enabled: bool = False
    jax_num_envs: int = Field(default=512, ge=1)
    jax_learning_rate: float = Field(default=3e-4, gt=0)
    jax_gamma: float = Field(default=0.99, gt=0, le=1.0)
    jax_clip_eps: float = Field(default=0.2, gt=0)

    # Q-teacher bootstrap
    q_teacher_enabled: bool = True
    q_teacher_min_snapshots: int = Field(default=500, ge=100)
    q_teacher_dp_iterations: int = Field(default=3, ge=1, le=20)
    q_teacher_kl_threshold: float = Field(default=0.05, gt=0)

    # EWC continual learning
    ewc_enabled: bool = True
    ewc_lambda: float = Field(default=400.0, gt=0)
    ewc_fisher_samples: int = Field(default=200, ge=10)

    # LOB feed
    lob_feed_enabled: bool = True
    lob_depth: int = Field(default=20, ge=5, le=200)
    lob_snapshot_buffer: int = Field(default=2000, ge=100)

    # Confidence gating
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    confidence_decay_enabled: bool = False
    confidence_decay_halflife_hours: float = Field(default=24.0, gt=0)

    class Config:
        extra = "forbid"
        validate_assignment = True


class ArgusConfig(BaseModel):
    """Top-level config — composes all domain configs."""
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ai: AIConfig = Field(default_factory=AIConfig)

    class Config:
        extra = "forbid"
        validate_assignment = True

    @classmethod
    def from_dict(cls, d: dict) -> "ArgusConfig":
        return cls(
            execution=ExecutionConfig(**d.get("execution", {})),
            risk=RiskConfig(**d.get("risk", {})),
            ai=AIConfig(**d.get("ai", {})),
        )

    @classmethod
    def from_env(cls) -> "ArgusConfig":
        """Bootstrap config from environment variables (12-factor style)."""
        import os
        return cls.from_dict({
            "execution": {
                "order_router_mode": os.getenv("ARGUS_ORDER_MODE", "dry_run"),
                "kernel_bypass_enabled": os.getenv("ARGUS_KERNEL_BYPASS", "0") == "1",
                "min_signal_confidence": float(os.getenv("ARGUS_MIN_CONFIDENCE", "0.55")),
                "dry_run": os.getenv("ARGUS_DRY_RUN", "1") == "1",
            },
            "risk": {
                "max_position_usd": float(os.getenv("ARGUS_MAX_POS_USD", "10000")),
                "max_daily_drawdown_pct": float(os.getenv("ARGUS_MAX_DD_PCT", "5.0")),
            },
            "ai": {
                "jax_enabled": os.getenv("ARGUS_JAX", "0") == "1",
                "lob_feed_enabled": os.getenv("ARGUS_LOB_FEED", "1") == "1",
                "min_confidence": float(os.getenv("ARGUS_MIN_CONFIDENCE", "0.55")),
            },
        })
