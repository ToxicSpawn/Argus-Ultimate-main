"""
BotState — mutable live-loop state container.

Centralises all runtime counters that were previously scattered as bare
attributes on UnifiedTradingSystem. Protected by a threading.Lock so it
can be safely read by the Prometheus metrics exporter thread.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List


@dataclass
class BotState:
    """Mutable runtime state for the live/paper trading loop."""

    # ── Capital ──────────────────────────────────────────────────────────────
    equity_aud: Decimal = Decimal("0")
    """Current account equity in AUD."""

    cash_aud: Decimal = Decimal("0")
    """Free cash not committed to open positions."""

    peak_equity_aud: Decimal = Decimal("0")
    """All-time peak equity (used for drawdown calculation)."""

    # ── Trade counters ────────────────────────────────────────────────────────
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0

    # ── Cycle state ───────────────────────────────────────────────────────────
    cycle_number: int = 0
    last_cycle_duration_s: float = 0.0
    is_paper: bool = True

    # ── Circuit breakers ─────────────────────────────────────────────────────
    circuit_breaker_tripped: bool = False
    emergency_shutdown_triggered: bool = False

    # ── Open positions ────────────────────────────────────────────────────────
    open_positions: Dict[str, dict] = field(default_factory=dict)
    """Keyed by symbol. Each value is a position dict (pending migration to typed Position)."""

    # ── Equity history (for vol estimation) ──────────────────────────────────
    equity_history: List[Decimal] = field(default_factory=list)

    # ── Thread safety ────────────────────────────────────────────────────────
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ── Derived metrics ──────────────────────────────────────────────────────
    @property
    def win_rate(self) -> float:
        """Win rate in [0.0, 1.0]; 0.0 if no trades yet."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as a fraction of peak equity; 0.0 if no history."""
        if self.peak_equity_aud == 0:
            return 0.0
        return float((self.peak_equity_aud - self.equity_aud) / self.peak_equity_aud)

    def record_equity(self, equity: Decimal) -> None:
        """Thread-safe equity snapshot append + peak update."""
        with self._lock:
            self.equity_aud = equity
            if equity > self.peak_equity_aud:
                self.peak_equity_aud = equity
            self.equity_history.append(equity)
            # Keep rolling window at 500 samples to bound memory
            if len(self.equity_history) > 500:
                self.equity_history = self.equity_history[-500:]
