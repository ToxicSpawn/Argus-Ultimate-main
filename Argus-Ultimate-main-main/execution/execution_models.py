from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


class DCASignal:
    """Wrapper that overrides quantity for DCA multi-level entries; delegates other attrs to inner signal."""

    def __init__(self, inner: Any, quantity: float) -> None:
        self._inner = inner
        self.quantity = quantity

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class ExecutionResult:
    """Trade execution result."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str  # filled, partial, rejected, pending
    exchange: str
    commission: float
    slippage: float
    timestamp: datetime
    pnl: float = 0.0
    error: Optional[str] = None


def build_idempotency_key(signal: Any) -> str:
    """Generate a deterministic idempotency key for a signal to prevent duplicate execution.

    Includes cycle_id (if present on the signal) so that the same technical signal
    generated in two different trading cycles is NOT treated as a duplicate.
    Without this, valid signals from cycle N+1 were permanently blocked because
    cycle N already inserted the same hash into the idempotency table.
    """
    # cycle_id disambiguates cross-cycle signals with identical indicators
    cycle_id = str(getattr(signal, "cycle_id", "") or getattr(signal, "_cycle_id", "") or "")
    parts = [
        str(getattr(signal, "symbol", "") or ""),
        str(getattr(signal, "action", "") or ""),
        str(round(float(getattr(signal, "entry_price", 0) or 0), 2)),
        str(round(float(getattr(signal, "confidence", 0) or 0), 4)),
        str(getattr(signal, "strategy", "") or getattr(signal, "source_strategy", "") or ""),
        cycle_id,
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
