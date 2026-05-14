"""argus_settings.py — Single unified config source (Pydantic v2).

Replaces the fragmented config.py / config_manager.py / config_unified.py /
argus_config.py / domain_config.py with ONE Pydantic Settings class.
All other config files should import from here and re-export for compat.

Install: pip install pydantic-settings
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ArgusSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARGUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Exchange ---
    exchange_id:          str   = "binance"
    api_key:              str   = ""
    api_secret:           str   = ""
    paper_trading:        bool  = True
    testnet:              bool  = False

    # --- Risk ---
    max_position_pct:     float = 0.10
    max_daily_loss_pct:   float = 0.03
    kelly_fraction:       float = 0.25
    initial_capital:      float = 1000.0
    session_timezone:     str   = "Australia/Sydney"

    # --- Strategy ---
    symbols:              List[str] = Field(default_factory=lambda: ["BTC/USDT"])
    timeframe:            str   = "1h"
    regime_min_hold_bars: int   = 6
    bandit_epsilon:       float = 0.05
    ensemble_sharpe_window: int = 30

    # --- Execution ---
    slip_atr_factor:      float = 0.05
    slip_max_pct:         float = 0.003
    taker_fee:            float = 0.001
    maker_fee:            float = 0.0008
    twap_slices:          int   = 4

    # --- Infrastructure ---
    use_uvloop:           bool  = True
    log_level:            str   = "INFO"
    db_path:              str   = "argus.db"
    checkpoint_dir:       str   = "checkpoints"

    @field_validator("max_position_pct", "max_daily_loss_pct", "kelly_fraction")
    @classmethod
    def must_be_positive_fraction(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Must be between 0 and 1, got {v}")
        return v


# Module-level singleton — import this everywhere
settings = ArgusSettings()
