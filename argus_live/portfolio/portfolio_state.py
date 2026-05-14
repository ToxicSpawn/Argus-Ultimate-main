from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioState:
    equity: float
    weights: dict[str, float] = field(default_factory=dict)
