"""Typed trade record dataclass — Push 54."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TradeRecord:
    """Represents a completed round-trip trade.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. 'BTCUSDT'.
    side : str
        'long' or 'short'.
    entry_price : float
        Fill price at entry.
    exit_price : float
        Fill price at exit.
    qty : float
        Position size in base currency.
    entry_time : datetime
        UTC timestamp of entry fill.
    exit_time : datetime
        UTC timestamp of exit fill.
    fee_bps : float
        Round-trip fee in basis points (default 2 bps).
    trade_id : str
        Optional unique identifier.
    """

    symbol: str
    side: str           # 'long' | 'short'
    entry_price: float
    exit_price: float
    qty: float
    entry_time: datetime
    exit_time: datetime
    fee_bps: float = 2.0
    trade_id: str = ""

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def gross_pnl(self) -> float:
        """Gross P&L before fees (in quote currency)."""
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.qty
        else:  # short
            return (self.entry_price - self.exit_price) * self.qty

    @property
    def fee_cost(self) -> float:
        """Total round-trip fee cost in quote currency."""
        notional = self.entry_price * self.qty + self.exit_price * self.qty
        return notional * (self.fee_bps / 10_000)

    @property
    def net_pnl(self) -> float:
        """Net P&L after fees."""
        return self.gross_pnl - self.fee_cost

    @property
    def return_pct(self) -> float:
        """Return as percentage of entry notional."""
        notional = self.entry_price * self.qty
        if notional == 0:
            return 0.0
        return self.net_pnl / notional * 100.0

    @property
    def duration_seconds(self) -> float:
        """Trade duration in seconds."""
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "fee_bps": self.fee_bps,
            "gross_pnl": round(self.gross_pnl, 8),
            "fee_cost": round(self.fee_cost, 8),
            "net_pnl": round(self.net_pnl, 8),
            "return_pct": round(self.return_pct, 6),
            "duration_seconds": self.duration_seconds,
            "is_winner": self.is_winner,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
        }
