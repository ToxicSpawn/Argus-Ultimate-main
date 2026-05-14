"""Immutable session statistics snapshot — Push 54."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from core.pnl.trade_record import TradeRecord
from core.pnl.drawdown import RunningDrawdown


@dataclass(frozen=True)
class SessionStats:
    """Immutable performance snapshot computed from a list of TradeRecords."""

    n_trades: int
    n_winners: int
    win_rate: float          # 0.0–1.0
    gross_pnl: float
    net_pnl: float
    total_fees: float
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float
    max_drawdown: float      # fraction, 0.0–1.0
    sharpe_ratio: float
    profit_factor: float
    avg_duration_s: float

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_trades(cls, trades: List[TradeRecord]) -> "SessionStats":
        """Compute SessionStats from a list of completed TradeRecords."""
        if not trades:
            return cls(
                n_trades=0, n_winners=0, win_rate=0.0,
                gross_pnl=0.0, net_pnl=0.0, total_fees=0.0,
                avg_trade_pnl=0.0, best_trade=0.0, worst_trade=0.0,
                max_drawdown=0.0, sharpe_ratio=0.0, profit_factor=0.0,
                avg_duration_s=0.0,
            )

        net_pnls = [t.net_pnl for t in trades]
        gross_pnls = [t.gross_pnl for t in trades]
        fees = [t.fee_cost for t in trades]
        durations = [t.duration_seconds for t in trades]

        n = len(trades)
        winners = [p for p in net_pnls if p > 0]
        losers = [p for p in net_pnls if p <= 0]

        # Sharpe (annualised daily, approximation)
        mean_pnl = sum(net_pnls) / n
        std_pnl = math.sqrt(sum((p - mean_pnl) ** 2 for p in net_pnls) / n) if n > 1 else 0.0
        sharpe = (mean_pnl / (std_pnl + 1e-9)) * math.sqrt(252)

        # Profit factor
        gross_wins = sum(winners)
        gross_loss = abs(sum(losers))
        pf = gross_wins / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown via equity curve
        dd = RunningDrawdown()
        equity = 0.0
        for p in net_pnls:
            equity += p
            dd.update(equity)

        return cls(
            n_trades=n,
            n_winners=len(winners),
            win_rate=len(winners) / n,
            gross_pnl=sum(gross_pnls),
            net_pnl=sum(net_pnls),
            total_fees=sum(fees),
            avg_trade_pnl=mean_pnl,
            best_trade=max(net_pnls),
            worst_trade=min(net_pnls),
            max_drawdown=dd.max_dd,
            sharpe_ratio=sharpe,
            profit_factor=pf,
            avg_duration_s=sum(durations) / n,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "n_winners": self.n_winners,
            "win_rate": round(self.win_rate, 4),
            "gross_pnl": round(self.gross_pnl, 6),
            "net_pnl": round(self.net_pnl, 6),
            "total_fees": round(self.total_fees, 6),
            "avg_trade_pnl": round(self.avg_trade_pnl, 6),
            "best_trade": round(self.best_trade, 6),
            "worst_trade": round(self.worst_trade, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "profit_factor": round(self.profit_factor, 4) if math.isfinite(self.profit_factor) else None,
            "avg_duration_s": round(self.avg_duration_s, 2),
        }

    def pretty_str(self) -> str:
        pf = f"{self.profit_factor:.3f}" if math.isfinite(self.profit_factor) else "∞"
        return (
            f"Session Stats\n"
            f"  Trades       : {self.n_trades} ({self.n_winners}W / {self.n_trades - self.n_winners}L)\n"
            f"  Win Rate     : {self.win_rate * 100:.1f}%\n"
            f"  Net P&L      : {self.net_pnl:+.4f}\n"
            f"  Gross P&L    : {self.gross_pnl:+.4f}\n"
            f"  Total Fees   : {self.total_fees:.4f}\n"
            f"  Avg Trade    : {self.avg_trade_pnl:+.4f}\n"
            f"  Best Trade   : {self.best_trade:+.4f}\n"
            f"  Worst Trade  : {self.worst_trade:+.4f}\n"
            f"  Max Drawdown : {self.max_drawdown * 100:.2f}%\n"
            f"  Sharpe       : {self.sharpe_ratio:.3f}\n"
            f"  Profit Factor: {pf}\n"
            f"  Avg Duration : {self.avg_duration_s:.1f}s\n"
        )
