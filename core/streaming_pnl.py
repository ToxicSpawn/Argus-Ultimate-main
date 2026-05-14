"""
core/streaming_pnl.py
~~~~~~~~~~~~~~~~~~~~~
Real-time unrealised and realised PnL per open position.

Design
------
- Positions keyed by (symbol, exchange).
- Mid prices updated on every book event.
- Background asyncio task fires registered callbacks every `update_interval_ms`.
- Thread-safe: public API uses a threading.RLock; async task uses the same lock
  via asyncio.get_event_loop().run_in_executor where needed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    symbol: str
    exchange: str
    net_size: float = 0.0
    avg_cost_basis: float = 0.0
    last_mid: float = 0.0
    realised_pnl: float = 0.0
    timestamp_ns: int = 0

    def unrealised_pnl(self) -> float:
        """Mark-to-market: (mid - cost_basis) × net_size."""
        if self.last_mid <= 0.0 or self.avg_cost_basis <= 0.0:
            return 0.0
        return (self.last_mid - self.avg_cost_basis) * self.net_size

    def total_pnl(self) -> float:
        return self.realised_pnl + self.unrealised_pnl()


# ---------------------------------------------------------------------------
# StreamingPnL
# ---------------------------------------------------------------------------

class StreamingPnL:
    """
    Maintains live unrealised and realised PnL for a portfolio of positions.

    Parameters
    ----------
    update_interval_ms : float
        How often (in milliseconds) the background task fires PnL callbacks.
    """

    def __init__(self, update_interval_ms: float = 100.0) -> None:
        self._update_interval_ms = update_interval_ms
        self._lock = threading.RLock()

        # (symbol, exchange) -> _Position
        self._positions: Dict[Tuple[str, str], _Position] = {}

        # Session-level peak PnL tracking (for drawdown)
        self._session_peak_pnl: float = 0.0
        self._session_start_pnl: float = 0.0
        self._max_drawdown: float = 0.0  # most negative drawdown seen

        # Callbacks
        self._callbacks: List[Callable[[dict], None]] = []

        # Background task handles
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Position & mid update
    # ------------------------------------------------------------------

    def update_position(
        self,
        symbol: str,
        exchange: str,
        net_size: float,
        avg_cost_basis: float,
        timestamp_ns: int,
    ) -> None:
        """Update or create a position entry.

        When a position is fully closed (net_size == 0), the unrealised PnL
        at the last known mid is transferred to realised PnL.
        """
        key = (symbol, exchange)
        with self._lock:
            pos = self._positions.get(key)
            if pos is None:
                pos = _Position(symbol=symbol, exchange=exchange)
                self._positions[key] = pos

            old_unrealised = pos.unrealised_pnl()

            # If sizing down to zero, crystallise the unrealised PnL
            if net_size == 0.0 and pos.net_size != 0.0:
                pos.realised_pnl += old_unrealised
                pos.net_size = 0.0
                pos.avg_cost_basis = 0.0
            else:
                pos.net_size = net_size
                pos.avg_cost_basis = avg_cost_basis

            pos.timestamp_ns = timestamp_ns

    def add_realised_pnl(
        self,
        symbol: str,
        exchange: str,
        amount: float,
    ) -> None:
        """Directly credit realised PnL (e.g. from a confirmed trade settlement)."""
        key = (symbol, exchange)
        with self._lock:
            pos = self._positions.get(key)
            if pos is None:
                pos = _Position(symbol=symbol, exchange=exchange)
                self._positions[key] = pos
            pos.realised_pnl += amount

    def update_mid(
        self,
        symbol: str,
        exchange: str,
        mid_price: float,
        timestamp_ns: int,
    ) -> None:
        """Update the mid price for (symbol, exchange). Called on every book event."""
        if mid_price <= 0:
            return
        key = (symbol, exchange)
        with self._lock:
            pos = self._positions.get(key)
            if pos is None:
                # Create a placeholder so mid is stored even before a position exists
                pos = _Position(symbol=symbol, exchange=exchange)
                self._positions[key] = pos
            pos.last_mid = mid_price
            pos.timestamp_ns = timestamp_ns
            # Maintain session peak for drawdown tracking
            current_total = self._total_pnl_locked()
            if current_total > self._session_peak_pnl:
                self._session_peak_pnl = current_total
            drawdown = self._session_peak_pnl - current_total
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown

    # ------------------------------------------------------------------
    # PnL queries
    # ------------------------------------------------------------------

    def get_unrealised_pnl(
        self,
        symbol: str,
        exchange: Optional[str] = None,
    ) -> float:
        """Mark-to-market unrealised PnL: (mid − cost_basis) × net_size."""
        with self._lock:
            return self._unrealised_pnl_locked(symbol, exchange)

    def _unrealised_pnl_locked(
        self,
        symbol: Optional[str],
        exchange: Optional[str],
    ) -> float:
        total = 0.0
        for (sym, exch), pos in self._positions.items():
            if symbol is not None and sym != symbol:
                continue
            if exchange is not None and exch != exchange:
                continue
            total += pos.unrealised_pnl()
        return total

    def get_realised_pnl(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> float:
        """Accumulated realised PnL, optionally filtered by symbol and/or exchange."""
        with self._lock:
            return self._realised_pnl_locked(symbol, exchange)

    def _realised_pnl_locked(
        self,
        symbol: Optional[str],
        exchange: Optional[str],
    ) -> float:
        total = 0.0
        for (sym, exch), pos in self._positions.items():
            if symbol is not None and sym != symbol:
                continue
            if exchange is not None and exch != exchange:
                continue
            total += pos.realised_pnl
        return total

    def get_total_pnl(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> float:
        """Realised + unrealised PnL, optionally filtered."""
        with self._lock:
            return self._total_pnl_locked(symbol, exchange)

    def _total_pnl_locked(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> float:
        return (
            self._realised_pnl_locked(symbol, exchange)
            + self._unrealised_pnl_locked(symbol, exchange)
        )

    def get_drawdown(self, symbol: Optional[str] = None) -> float:
        """Maximum drawdown from session peak PnL.

        If *symbol* is specified, computes the drawdown for that symbol's
        running unrealised PnL only (using a simple current-vs-peak heuristic).
        If no symbol is given, returns the portfolio-level max drawdown.
        """
        with self._lock:
            if symbol is None:
                return self._max_drawdown
            # Per-symbol: compare current total against a rough peak
            sym_positions = {
                k: v for k, v in self._positions.items() if k[0] == symbol
            }
            if not sym_positions:
                return 0.0
            current = sum(p.total_pnl() for p in sym_positions.values())
            # We store per-symbol peak lazily via a dict
            peak = self._sym_peak.get(symbol, current)
            if current > peak:
                self._sym_peak[symbol] = current
                peak = current
            return max(0.0, peak - current)

    # Lazy per-symbol peak tracking (used only by per-symbol drawdown)
    @property
    def _sym_peak(self) -> dict:
        if not hasattr(self, "_sym_peak_store"):
            self._sym_peak_store: Dict[str, float] = {}
        return self._sym_peak_store

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------

    def get_session_stats(self) -> dict:
        """Return a snapshot of session-level PnL statistics.

        Returned dict includes:
          - total_unrealised
          - total_realised
          - total_pnl
          - peak_pnl
          - max_drawdown
          - current_drawdown_pct  (% of peak)
          - best_symbol           (highest total PnL)
          - worst_symbol          (lowest total PnL)
          - positions             (list of per-position dicts)
        """
        with self._lock:
            total_unreal = self._unrealised_pnl_locked(None, None)
            total_real = self._realised_pnl_locked(None, None)
            total = total_unreal + total_real

            # Update peak
            if total > self._session_peak_pnl:
                self._session_peak_pnl = total
            current_dd = max(0.0, self._session_peak_pnl - total)
            if current_dd > self._max_drawdown:
                self._max_drawdown = current_dd

            current_dd_pct = (
                current_dd / self._session_peak_pnl * 100.0
                if self._session_peak_pnl > 0
                else 0.0
            )

            positions_list = []
            sym_totals: Dict[str, float] = defaultdict(float)
            for (sym, exch), pos in self._positions.items():
                unreal = pos.unrealised_pnl()
                tp = pos.realised_pnl + unreal
                sym_totals[sym] += tp
                positions_list.append(
                    {
                        "symbol": sym,
                        "exchange": exch,
                        "size": pos.net_size,
                        "cost_basis": pos.avg_cost_basis,
                        "mid": pos.last_mid,
                        "unrealised_pnl": unreal,
                        "realised_pnl": pos.realised_pnl,
                        "total_pnl": tp,
                    }
                )

            best_sym = max(sym_totals, key=sym_totals.get) if sym_totals else None
            worst_sym = min(sym_totals, key=sym_totals.get) if sym_totals else None

        return {
            "total_unrealised": total_unreal,
            "total_realised": total_real,
            "total_pnl": total,
            "peak_pnl": self._session_peak_pnl,
            "max_drawdown": self._max_drawdown,
            "current_drawdown_pct": current_dd_pct,
            "best_symbol": best_sym,
            "worst_symbol": worst_sym,
            "positions": positions_list,
        }

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_pnl_callback(self, callback: Callable[[dict], None]) -> None:
        """Register a callable that receives the session stats dict periodically.

        The callback is invoked from the background async task every
        `update_interval_ms` milliseconds.  It must not block; use
        asyncio.create_task or run_in_executor for heavy work.
        """
        with self._lock:
            self._callbacks.append(callback)

    def _fire_callbacks(self) -> None:
        stats = self.get_session_stats()
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(stats)
            except Exception:
                pass  # never let a bad callback kill the loop

    # ------------------------------------------------------------------
    # Background async task
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        interval_s = self._update_interval_ms / 1_000.0
        while self._running:
            await asyncio.sleep(interval_s)
            self._fire_callbacks()

    def start(self) -> None:
        """Start the background PnL broadcast task.

        Safe to call from a sync context.  Creates a new event loop in a
        daemon thread if no running loop is available.
        """
        if self._running:
            return
        self._running = True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            self._loop = loop
            self._task = loop.create_task(self._run_loop())
        else:
            # Spin up a daemon thread with its own event loop
            def _thread_main() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                self._loop = new_loop
                self._task = new_loop.create_task(self._run_loop())
                new_loop.run_forever()

            t = threading.Thread(target=_thread_main, daemon=True, name="StreamingPnL-bg")
            t.start()

    def stop(self) -> None:
        """Stop the background task gracefully."""
        self._running = False
        if self._task is not None:
            try:
                self._task.cancel()
            except Exception:
                pass
        if self._loop is not None and self._loop.is_running():
            try:
                self._loop.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Reset session-level peak and drawdown tracking."""
        with self._lock:
            self._session_peak_pnl = 0.0
            self._max_drawdown = 0.0

    def get_positions(self) -> List[dict]:
        """Return a snapshot list of all current position dicts."""
        return self.get_session_stats()["positions"]

    def symbol_list(self) -> List[str]:
        """Return unique symbols currently tracked."""
        with self._lock:
            return list({sym for (sym, _) in self._positions})

    def __repr__(self) -> str:  # pragma: no cover
        stats = self.get_session_stats()
        return (
            f"<StreamingPnL total_pnl={stats['total_pnl']:.2f} "
            f"max_drawdown={stats['max_drawdown']:.2f} "
            f"positions={len(stats['positions'])}>"
        )
