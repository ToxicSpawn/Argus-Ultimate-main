"""
core/capital_tier.py

CapitalTier enum + classify_tier(equity_aud) helper.

Tier boundaries (AUD):
  NANO   :  0 – 499
  MICRO  :  500 – 2 499
  SMALL  :  2 500 – 9 999
  MID    :  10 000 – 49 999
  LARGE  :  50 000+
"""
from __future__ import annotations

from enum import Enum


class CapitalTier(str, Enum):
    NANO   = "NANO"
    MICRO  = "MICRO"
    SMALL  = "SMALL"
    MID    = "MID"
    LARGE  = "LARGE"


_THRESHOLDS: list[tuple[float, CapitalTier]] = [
    (50_000.0, CapitalTier.LARGE),
    (10_000.0, CapitalTier.MID),
    ( 2_500.0, CapitalTier.SMALL),
    (   500.0, CapitalTier.MICRO),
    (     0.0, CapitalTier.NANO),
]


def classify_tier(equity_aud: float) -> CapitalTier:
    """Return the CapitalTier for the given AUD equity value."""
    equity_aud = max(0.0, float(equity_aud))
    for threshold, tier in _THRESHOLDS:
        if equity_aud >= threshold:
            return tier
    return CapitalTier.NANO
