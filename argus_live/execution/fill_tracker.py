from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Fill:
    intent_id: str
    quantity: float
    price: float
    latency_ms: float = 0.0
    rejected: bool = False
    partial_fill: bool = False
    reason: str = ""
    adverse_price_move_bps: float = 0.0


_FILL_SIMULATOR: Optional[Callable[[str, float, float, dict[str, Any] | None], Fill]] = None


def set_fill_simulator(simulator: Callable[[str, float, float, dict[str, Any] | None], Fill] | None) -> None:
    global _FILL_SIMULATOR
    _FILL_SIMULATOR = simulator


def clear_fill_simulator() -> None:
    set_fill_simulator(None)


def simulate_fill(intent_id: str, quantity: float, price: float, metadata: dict[str, Any] | None = None) -> Fill:
    if _FILL_SIMULATOR is not None:
        return _FILL_SIMULATOR(intent_id, quantity, price, metadata)
    return Fill(intent_id, quantity, price)
