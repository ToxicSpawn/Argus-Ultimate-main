"""PositionTracker — real-time P&L, drawdown, exposure tracking.

Extracted from unified_trading_system.py.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single open position."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    opened_at: float = field(default_factory=time.time)

    def update_price(self, price: float) -> None:
        self.current_price = float(price)
        self.unrealised_pnl = (self.current_price - self.avg_entry_price) * self.quantity


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: float
    cash: float
    positions_value: float
    total_equity: float
    unrealised_pnl: float
    realised_pnl: float
    drawdown_pct: float
    position_count: int


class PositionTracker:
    """
    Thread-safe real-time position and P&L tracker.

    Extracted from unified_trading_system.py.
    """

    def __init__(self, starting_cash: float = 1000.0) -> None:
        self._lock = threading.RLock()
        self._cash = float(starting_cash)
        self._positions: Dict[str, Position] = {}
        self._peak_equity = float(starting_cash)
        self._realised_pnl_total = 0.0
        self._trade_log: List[Dict[str, Any]] = []

    def apply_fill(self, symbol: str, side: str, quantity: float, price: float) -> None:
        """Apply an executed fill to positions."""
        with self._lock:
            side = side.upper()
            qty = float(quantity)
            px = float(price)

            if side == "BUY":
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    total_qty = pos.quantity + qty
                    pos.avg_entry_price = (pos.avg_entry_price * pos.quantity + px * qty) / total_qty
                    pos.quantity = total_qty
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol, quantity=qty, avg_entry_price=px, current_price=px
                    )
                self._cash -= qty * px

            elif side == "SELL":
                pos = self._positions.get(symbol)
                if pos is None:
                    logger.warning("PositionTracker: SELL on unknown position %s", symbol)
                    return
                sell_qty = min(qty, pos.quantity)
                pnl = (px - pos.avg_entry_price) * sell_qty
                self._realised_pnl_total += pnl
                self._cash += sell_qty * px
                pos.quantity -= sell_qty
                if pos.quantity <= 1e-10:
                    del self._positions[symbol]
                self._trade_log.append({
                    "symbol": symbol, "side": "SELL", "qty": sell_qty,
                    "price": px, "pnl": pnl, "ts": time.time(),
                })

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Batch-update current prices for all positions."""
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._positions:
                    self._positions[symbol].update_price(price)

    def snapshot(self) -> PortfolioSnapshot:
        """Return a point-in-time portfolio snapshot."""
        with self._lock:
            pos_value = sum(p.quantity * p.current_price for p in self._positions.values())
            unrealised = sum(p.unrealised_pnl for p in self._positions.values())
            total_equity = self._cash + pos_value
            if total_equity > self._peak_equity:
                self._peak_equity = total_equity
            dd_pct = (self._peak_equity - total_equity) / max(self._peak_equity, 1e-9) * 100.0
            return PortfolioSnapshot(
                timestamp=time.time(),
                cash=self._cash,
                positions_value=pos_value,
                total_equity=total_equity,
                unrealised_pnl=unrealised,
                realised_pnl=self._realised_pnl_total,
                drawdown_pct=max(0.0, dd_pct),
                position_count=len(self._positions),
            )

    @property
    def positions(self) -> Dict[str, Position]:
        with self._lock:
            return dict(self._positions)
