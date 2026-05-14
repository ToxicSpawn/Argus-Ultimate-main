"""Live P&L tracker — Push 54.

Thread-safe, asyncio-compatible accumulator for open and closed positions.

Usage::

    tracker = PnLTracker(fee_bps=2.0)
    tracker.open_position("BTCUSDT", side="long", price=65000.0, qty=0.01)
    tracker.close_position("BTCUSDT", exit_price=65500.0)
    stats = tracker.session_stats()
    print(stats.pretty_str())

Prometheus gauges emitted (if prometheus_client installed)::

    argus_session_pnl          — cumulative net P&L
    argus_session_trades       — total closed trades
    argus_session_win_rate     — win rate 0–1
    argus_session_drawdown     — current max drawdown fraction
    argus_unrealised_pnl       — mark-to-market unrealised P&L
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core.pnl.trade_record import TradeRecord
from core.pnl.session_stats import SessionStats
from core.pnl.drawdown import RunningDrawdown

try:
    from prometheus_client import Gauge
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Prometheus gauges
if _PROM_AVAILABLE:
    _G_PNL = Gauge("argus_session_pnl", "Cumulative net session P&L")
    _G_TRADES = Gauge("argus_session_trades", "Total closed trades this session")
    _G_WIN_RATE = Gauge("argus_session_win_rate", "Session win rate")
    _G_DD = Gauge("argus_session_drawdown", "Current max drawdown fraction")
    _G_UNREAL = Gauge("argus_unrealised_pnl", "Mark-to-market unrealised P&L")
else:
    _G_PNL = _G_TRADES = _G_WIN_RATE = _G_DD = _G_UNREAL = None


# Placeholder for dataclass-like decorator compatibility
_dataclass_like = None  # noqa — not used, just silence


class _OpenPosition:
    """Tracks an open (not yet closed) position."""
    __slots__ = ("symbol", "side", "entry_price", "qty", "entry_time", "fee_bps")

    def __init__(self, symbol: str, side: str, entry_price: float,
                 qty: float, entry_time: datetime, fee_bps: float) -> None:
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.qty = qty
        self.entry_time = entry_time
        self.fee_bps = fee_bps

    def unrealised_pnl(self, mid_price: float) -> float:
        if self.side == "long":
            return (mid_price - self.entry_price) * self.qty
        return (self.entry_price - mid_price) * self.qty


class PnLTracker:
    """Thread-safe live P&L accumulator.

    Parameters
    ----------
    fee_bps : float
        Default round-trip fee in basis points (default 2.0).
    """

    def __init__(self, fee_bps: float = 2.0) -> None:
        self._fee_bps = fee_bps
        self._lock = threading.Lock()
        self._closed_trades: List[TradeRecord] = []
        self._open_positions: Dict[str, _OpenPosition] = {}
        self._drawdown = RunningDrawdown()
        self._equity = 0.0

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        fee_bps: Optional[float] = None,
        entry_time: Optional[datetime] = None,
    ) -> None:
        """Record an entry fill."""
        with self._lock:
            if symbol in self._open_positions:
                logger.warning(
                    "PnLTracker: overwriting existing open position for %s", symbol
                )
            self._open_positions[symbol] = _OpenPosition(
                symbol=symbol,
                side=side.lower(),
                entry_price=price,
                qty=qty,
                entry_time=entry_time or datetime.now(timezone.utc),
                fee_bps=fee_bps if fee_bps is not None else self._fee_bps,
            )
            logger.debug("PnLTracker: opened %s %s @ %.4f qty=%.6f", side, symbol, price, qty)

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
    ) -> Optional[TradeRecord]:
        """Record an exit fill. Returns the completed TradeRecord."""
        with self._lock:
            pos = self._open_positions.pop(symbol, None)
            if pos is None:
                logger.warning("PnLTracker: no open position for %s to close", symbol)
                return None

            record = TradeRecord(
                symbol=symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                qty=pos.qty,
                entry_time=pos.entry_time,
                exit_time=exit_time or datetime.now(timezone.utc),
                fee_bps=pos.fee_bps,
            )
            self._closed_trades.append(record)
            self._equity += record.net_pnl
            self._drawdown.update(self._equity)
            self._emit_prometheus()

            logger.info(
                "PnLTracker: closed %s %s @ %.4f net_pnl=%+.4f",
                pos.side, symbol, exit_price, record.net_pnl,
            )
            return record

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def running_unrealised_pnl(self, prices: Dict[str, float]) -> float:
        """Compute total unrealised P&L given a dict of {symbol: mid_price}."""
        with self._lock:
            total = sum(
                pos.unrealised_pnl(prices[sym])
                for sym, pos in self._open_positions.items()
                if sym in prices
            )
        if _G_UNREAL:
            _G_UNREAL.set(total)
        return total

    # ------------------------------------------------------------------
    # Stats & queries
    # ------------------------------------------------------------------

    def session_stats(self) -> SessionStats:
        """Return an immutable snapshot of session performance."""
        with self._lock:
            trades = list(self._closed_trades)
        return SessionStats.from_trades(trades)

    @property
    def closed_trades(self) -> List[TradeRecord]:
        with self._lock:
            return list(self._closed_trades)

    @property
    def open_symbols(self) -> List[str]:
        with self._lock:
            return list(self._open_positions.keys())

    @property
    def equity(self) -> float:
        with self._lock:
            return self._equity

    @property
    def max_drawdown(self) -> float:
        return self._drawdown.max_dd

    def reset(self) -> None:
        """Reset all session data."""
        with self._lock:
            self._closed_trades.clear()
            self._open_positions.clear()
            self._drawdown.reset()
            self._equity = 0.0
        logger.info("PnLTracker: session reset")

    # ------------------------------------------------------------------
    # Prometheus
    # ------------------------------------------------------------------

    def _emit_prometheus(self) -> None:
        if not _PROM_AVAILABLE:
            return
        n = len(self._closed_trades)
        winners = sum(1 for t in self._closed_trades if t.is_winner)
        if _G_PNL:
            _G_PNL.set(self._equity)
        if _G_TRADES:
            _G_TRADES.set(n)
        if _G_WIN_RATE:
            _G_WIN_RATE.set(winners / n if n > 0 else 0.0)
        if _G_DD:
            _G_DD.set(self._drawdown.max_dd)
