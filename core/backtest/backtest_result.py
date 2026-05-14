"""BacktestResult dataclass — Push 59."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EquityPoint:
    """Single point on the equity curve."""
    ts: float
    equity: float


@dataclass
class BacktestResult:
    """Holds the full output of a backtest run."""

    config: Dict[str, Any] = field(default_factory=dict)
    equity_curve: List[EquityPoint] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)

    # Summary metrics
    total_return: float = 0.0
    annualised_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar: float = 0.0
    n_trades: int = 0
    initial_equity: float = 10_000.0
    final_equity: float = 10_000.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "metrics": {
                "total_return": self.total_return,
                "annualised_return": self.annualised_return,
                "sharpe": self.sharpe,
                "sortino": self.sortino,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "calmar": self.calmar,
                "n_trades": self.n_trades,
                "initial_equity": self.initial_equity,
                "final_equity": self.final_equity,
            },
            "equity_curve_length": len(self.equity_curve),
            "trade_log_length": len(self.trade_log),
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        data = self.to_dict()
        js = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(js)
            logger.info("BacktestResult: saved JSON -> %s", path)
        return js

    def to_csv(self, path: Path) -> None:
        """Write equity curve to CSV."""
        path = Path(path)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "equity"])
            for pt in self.equity_curve:
                writer.writerow([pt.ts, pt.equity])
        logger.info("BacktestResult: saved equity CSV -> %s", path)

    def plot_equity_curve(
        self,
        path: Optional[Path] = None,
        show: bool = False,
    ) -> None:
        """Plot and optionally save the equity curve as PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime, timezone
        except ImportError:
            logger.warning("BacktestResult: matplotlib not available, skipping plot")
            return

        if not self.equity_curve:
            return

        dates = [datetime.fromtimestamp(p.ts, tz=timezone.utc) for p in self.equity_curve]
        equities = [p.equity for p in self.equity_curve]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, equities, linewidth=1.5, color="#00bfff")
        ax.fill_between(dates, equities, alpha=0.15, color="#00bfff")
        ax.set_title(
            f"Argus Backtest — Equity Curve  "
            f"(Sharpe {self.sharpe:.2f} | MDD {self.max_drawdown*100:.1f}%)",
            fontsize=12,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (USD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if path:
            fig.savefig(path, dpi=150)
            logger.info("BacktestResult: saved equity plot -> %s", path)
        if show:
            plt.show()
        plt.close(fig)
