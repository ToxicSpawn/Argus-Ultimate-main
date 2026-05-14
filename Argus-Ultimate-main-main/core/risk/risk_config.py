"""RiskConfig — central risk parameter dataclass — Push 55.

All limits are soft-configurable at runtime or via environment variables.

Environment variables::

    ARGUS_RISK_MAX_POSITION_PCT     float  0.0-1.0  (default 0.10)
    ARGUS_RISK_MAX_DRAWDOWN_HALT    float  0.0-1.0  (default 0.15)
    ARGUS_RISK_MAX_DAILY_LOSS       float  quote $  (default 200.0)
    ARGUS_RISK_MAX_OPEN_POSITIONS   int             (default 4)
    ARGUS_RISK_KELLY_FRACTION       float  0.0-1.0  (default 0.25)
    ARGUS_RISK_MIN_CONFIDENCE       float  0.0-1.0  (default 0.55)
    ARGUS_RISK_FEE_BPS              float           (default 2.0)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskConfig:
    """Centralised risk parameters for Argus."""

    # Position sizing
    max_position_pct: float = 0.10      # max % of equity per trade
    kelly_fraction: float = 0.25        # fractional Kelly multiplier

    # Halt triggers
    max_drawdown_halt: float = 0.15     # halt trading at this drawdown
    max_daily_loss: float = 200.0       # max daily loss in quote currency

    # Concentration limits
    max_open_positions: int = 4
    max_symbol_exposure_pct: float = 0.20  # max % equity in one symbol

    # Signal quality gate
    min_confidence: float = 0.55

    # Fees (for sizing calculations)
    fee_bps: float = 2.0

    @classmethod
    def from_env(cls) -> "RiskConfig":
        """Load RiskConfig from environment variables."""
        return cls(
            max_position_pct=float(os.getenv("ARGUS_RISK_MAX_POSITION_PCT", "0.10")),
            max_drawdown_halt=float(os.getenv("ARGUS_RISK_MAX_DRAWDOWN_HALT", "0.15")),
            max_daily_loss=float(os.getenv("ARGUS_RISK_MAX_DAILY_LOSS", "200.0")),
            max_open_positions=int(os.getenv("ARGUS_RISK_MAX_OPEN_POSITIONS", "4")),
            kelly_fraction=float(os.getenv("ARGUS_RISK_KELLY_FRACTION", "0.25")),
            min_confidence=float(os.getenv("ARGUS_RISK_MIN_CONFIDENCE", "0.55")),
            fee_bps=float(os.getenv("ARGUS_RISK_FEE_BPS", "2.0")),
        )

    def validate(self) -> None:
        """Raise ValueError if any parameter is out of range."""
        assert 0.0 < self.max_position_pct <= 1.0, "max_position_pct out of range"
        assert 0.0 < self.max_drawdown_halt <= 1.0, "max_drawdown_halt out of range"
        assert self.max_daily_loss > 0, "max_daily_loss must be positive"
        assert self.max_open_positions >= 1, "max_open_positions must be >= 1"
        assert 0.0 < self.kelly_fraction <= 1.0, "kelly_fraction out of range"
        assert 0.0 <= self.min_confidence <= 1.0, "min_confidence out of range"
