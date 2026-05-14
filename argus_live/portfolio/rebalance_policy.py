from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RebalancePolicy:
    min_weight_delta: float = 0.0025
    max_turnover_pct: float = 0.20
    cooldown_seconds: int = 60
