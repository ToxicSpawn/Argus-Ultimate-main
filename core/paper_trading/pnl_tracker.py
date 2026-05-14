"""Push 70 — Real-time PnL tracker.

Tracks per-symbol and aggregate PnL:
  - Realised PnL (from fills)
  - Unrealised PnL (from mark prices)
  - Total PnL = realised + unrealised
  - Rolling windows: session, daily (UTC), hourly
  - Peak equity + max drawdown (live)
  - Win/loss counts + largest win/loss
  - Trade log with timestamps
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class TradeRecord:
    symbol: str
    side: str
    qty: float
    fill_price: float
    pnl: float
    commission: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PnLSnapshot:
    timestamp: float
    realised_pnl: float
    unrealised_pnl: float
    total_pnl: float
    equity: float
    drawdown_pct: float
    n_trades: int
    win_rate: float


class RealTimePnLTracker:
    """Real-time PnL tracker for paper/live trading.

    Args:
        initial_equity: Starting equity for drawdown calculation
        snapshot_interval_secs: Min seconds between auto-snapshots
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        snapshot_interval_secs: float = 60.0,
    ):
        self.initial_equity = initial_equity
        self.snapshot_interval = snapshot_interval_secs

        # Aggregate PnL
        self._realised_pnl: float = 0.0
        self._unrealised_pnl: float = 0.0

        # Per-symbol
        self._symbol_realised: Dict[str, float] = defaultdict(float)
        self._symbol_unrealised: Dict[str, float] = defaultdict(float)

        # Equity tracking
        self._peak_equity: float = initial_equity
        self._current_equity: float = initial_equity
        self._max_drawdown_pct: float = 0.0

        # Trade stats
        self._trade_log: List[TradeRecord] = []
        self._wins: int = 0
        self._losses: int = 0
        self._largest_win: float = 0.0
        self._largest_loss: float = 0.0

        # Rolling windows (timestamp, pnl_delta)
        self._hourly_window:  Deque[Tuple[float, float]] = deque()
        self._daily_window:   Deque[Tuple[float, float]] = deque()

        # Snapshots
        self._snapshots: List[PnLSnapshot] = []
        self._last_snapshot_at: float = 0.0

    def record_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        pnl: float,
        commission: float,
    ) -> None:
        """Record a fill event and update PnL."""
        now = time.time()
        self._realised_pnl += pnl
        self._symbol_realised[symbol] += pnl

        rec = TradeRecord(symbol=symbol, side=side, qty=qty,
                           fill_price=fill_price, pnl=pnl,
                           commission=commission, timestamp=now)
        self._trade_log.append(rec)

        if pnl > 0:
            self._wins += 1
            self._largest_win = max(self._largest_win, pnl)
        elif pnl < 0:
            self._losses += 1
            self._largest_loss = min(self._largest_loss, pnl)

        # Rolling windows
        self._hourly_window.append((now, pnl))
        self._daily_window.append((now, pnl))
        self._trim_windows(now)

    def update_mark_price(self, symbol: str, mark_price: float,
                           position_qty: float, position_avg_entry: float,
                           position_side: str) -> None:
        """Update unrealised PnL for a symbol."""
        if position_side == "long":
            upnl = (mark_price - position_avg_entry) * position_qty
        else:
            upnl = (position_avg_entry - mark_price) * position_qty
        old = self._symbol_unrealised.get(symbol, 0.0)
        self._unrealised_pnl += (upnl - old)
        self._symbol_unrealised[symbol] = upnl

    def update_equity(self, equity: float) -> None:
        """Update current equity and recalculate drawdown."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        dd = (self._peak_equity - equity) / max(self._peak_equity, 1e-9) * 100.0
        self._max_drawdown_pct = max(self._max_drawdown_pct, dd)
        self._maybe_snapshot(equity)

    def _trim_windows(self, now: float) -> None:
        hour_ago = now - 3600
        day_ago  = now - 86400
        while self._hourly_window and self._hourly_window[0][0] < hour_ago:
            self._hourly_window.popleft()
        while self._daily_window and self._daily_window[0][0] < day_ago:
            self._daily_window.popleft()

    def _maybe_snapshot(self, equity: float) -> None:
        now = time.time()
        if now - self._last_snapshot_at < self.snapshot_interval:
            return
        self._last_snapshot_at = now
        snap = PnLSnapshot(
            timestamp=now,
            realised_pnl=self._realised_pnl,
            unrealised_pnl=self._unrealised_pnl,
            total_pnl=self.total_pnl,
            equity=equity,
            drawdown_pct=self.current_drawdown_pct,
            n_trades=self.n_trades,
            win_rate=self.win_rate,
        )
        self._snapshots.append(snap)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def realised_pnl(self) -> float:
        return self._realised_pnl

    @property
    def unrealised_pnl(self) -> float:
        return self._unrealised_pnl

    @property
    def total_pnl(self) -> float:
        return self._realised_pnl + self._unrealised_pnl

    @property
    def current_drawdown_pct(self) -> float:
        equity = self._current_equity
        return (self._peak_equity - equity) / max(self._peak_equity, 1e-9) * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_drawdown_pct

    @property
    def n_trades(self) -> int:
        return len(self._trade_log)

    @property
    def win_rate(self) -> float:
        total = self._wins + self._losses
        return self._wins / total if total > 0 else 0.0

    @property
    def hourly_pnl(self) -> float:
        return sum(p for _, p in self._hourly_window)

    @property
    def daily_pnl(self) -> float:
        return sum(p for _, p in self._daily_window)

    @property
    def largest_win(self) -> float:
        return self._largest_win

    @property
    def largest_loss(self) -> float:
        return self._largest_loss

    @property
    def snapshots(self) -> List[PnLSnapshot]:
        return self._snapshots

    def symbol_pnl(self, symbol: str) -> dict:
        return {
            "realised": self._symbol_realised.get(symbol, 0.0),
            "unrealised": self._symbol_unrealised.get(symbol, 0.0),
        }
