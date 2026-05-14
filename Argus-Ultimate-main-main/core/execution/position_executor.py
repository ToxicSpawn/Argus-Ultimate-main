"""Push 67 — PositionExecutor: unified order lifecycle manager.

Handles the full position lifecycle:
  PENDING -> OPEN -> (SL_HIT | TP1_HIT | TP_HIT | MANUAL_CLOSE)

Features:
  - ATR-based trailing stop (updates only in favour)
  - Partial close at 1R (50% of position)
  - Break-even stop after partial close
  - Full TP at configurable R-multiple
  - Per-position P&L tracking
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time


class PositionStatus(str, Enum):
    PENDING  = "PENDING"
    OPEN     = "OPEN"
    SL_HIT   = "SL_HIT"
    TP1_HIT  = "TP1_HIT"   # partial close fired
    TP_HIT   = "TP_HIT"    # full take-profit
    MANUAL   = "MANUAL"
    EXPIRED  = "EXPIRED"


@dataclass
class PositionExecutor:
    """Tracks a single position from entry to exit."""

    # Identity
    symbol: str
    side: str                        # "buy" | "sell"
    entry_price: float
    size_usd: float
    strategy_name: str = "unknown"
    position_id: str = field(default_factory=lambda: str(int(time.time() * 1e6)))

    # Risk parameters
    stop_loss_pct: float = 0.02      # 2% hard stop
    take_profit_pct: float = 0.04    # 4% full TP (2R)
    partial_tp_pct: float = 0.02     # 1R partial close (50%)
    trailing_stop_pct: float = 0.015 # trailing stop distance
    use_trailing: bool = True
    use_partial_tp: bool = True

    # Runtime state
    status: PositionStatus = PositionStatus.PENDING
    current_price: float = field(default=0.0, repr=False)
    peak_favourable_price: float = field(default=0.0, repr=False)
    realised_pnl: float = field(default=0.0, repr=False)
    partial_closed: bool = field(default=False, repr=False)
    open_time: float = field(default_factory=time.time, repr=False)
    close_time: Optional[float] = field(default=None, repr=False)

    def __post_init__(self):
        self.current_price = self.entry_price
        self.peak_favourable_price = self.entry_price
        self.status = PositionStatus.OPEN

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, price: float) -> PositionStatus:
        """Update with latest price. Returns current status."""
        if self.status not in (PositionStatus.OPEN, PositionStatus.TP1_HIT):
            return self.status

        self.current_price = price
        pnl_pct = self._pnl_pct(price)

        # Update trailing stop peak
        if self.use_trailing:
            if self.side == "buy" and price > self.peak_favourable_price:
                self.peak_favourable_price = price
            elif self.side == "sell" and price < self.peak_favourable_price:
                self.peak_favourable_price = price

        # Check hard stop
        if pnl_pct <= -self.stop_loss_pct:
            return self._close(PositionStatus.SL_HIT, price)

        # Check trailing stop (only active after partial TP)
        if self.use_trailing and self.partial_closed:
            trail_breach = self._trailing_breach(price)
            if trail_breach:
                return self._close(PositionStatus.SL_HIT, price)

        # Check partial TP
        if self.use_partial_tp and not self.partial_closed:
            if pnl_pct >= self.partial_tp_pct:
                self.partial_closed = True
                self.realised_pnl += self.size_usd * 0.5 * pnl_pct
                # Move hard stop to break-even
                self.stop_loss_pct = 0.0
                self.status = PositionStatus.TP1_HIT
                return self.status

        # Check full TP
        if pnl_pct >= self.take_profit_pct:
            return self._close(PositionStatus.TP_HIT, price)

        return self.status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pnl_pct(self, price: float) -> float:
        if self.side == "buy":
            return (price - self.entry_price) / self.entry_price
        return (self.entry_price - price) / self.entry_price

    def _trailing_breach(self, price: float) -> bool:
        if self.side == "buy":
            return price < self.peak_favourable_price * (1 - self.trailing_stop_pct)
        return price > self.peak_favourable_price * (1 + self.trailing_stop_pct)

    def _close(self, status: PositionStatus, price: float) -> PositionStatus:
        self.status = status
        self.close_time = time.time()
        remaining = 0.5 if self.partial_closed else 1.0
        self.realised_pnl += self.size_usd * remaining * self._pnl_pct(price)
        return self.status

    @property
    def unrealised_pnl(self) -> float:
        if self.status not in (PositionStatus.OPEN, PositionStatus.TP1_HIT):
            return 0.0
        remaining = 0.5 if self.partial_closed else 1.0
        return self.size_usd * remaining * self._pnl_pct(self.current_price)

    @property
    def total_pnl(self) -> float:
        return self.realised_pnl + self.unrealised_pnl

    @property
    def is_open(self) -> bool:
        return self.status in (PositionStatus.OPEN, PositionStatus.TP1_HIT)

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "size_usd": self.size_usd,
            "status": self.status.value,
            "realised_pnl": self.realised_pnl,
            "unrealised_pnl": self.unrealised_pnl,
            "total_pnl": self.total_pnl,
            "partial_closed": self.partial_closed,
            "strategy": self.strategy_name,
        }


class PositionExecutorEngine:
    """Manages multiple concurrent PositionExecutors."""

    def __init__(self, max_positions: int = 5):
        self.max_positions = max_positions
        self._positions: Dict[str, PositionExecutor] = {}

    def open(self, executor: PositionExecutor) -> bool:
        """Register a new position. Returns False if at capacity."""
        open_count = sum(1 for p in self._positions.values() if p.is_open)
        if open_count >= self.max_positions:
            return False
        self._positions[executor.position_id] = executor
        return True

    def evaluate_all(self, prices: Dict[str, float]) -> List[PositionExecutor]:
        """Evaluate all open positions against current prices. Returns closed ones."""
        closed = []
        for pos in list(self._positions.values()):
            if not pos.is_open:
                continue
            price = prices.get(pos.symbol)
            if price is None:
                continue
            pos.evaluate(price)
            if not pos.is_open:
                closed.append(pos)
        return closed

    def get_open(self) -> List[PositionExecutor]:
        return [p for p in self._positions.values() if p.is_open]

    def get_all(self) -> List[PositionExecutor]:
        return list(self._positions.values())

    def total_unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self._positions.values() if p.is_open)

    def total_realised_pnl(self) -> float:
        return sum(p.realised_pnl for p in self._positions.values())
