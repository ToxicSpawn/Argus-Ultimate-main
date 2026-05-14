"""Lightweight CLI performance dashboard snapshots."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import numpy as np


@dataclass
class DashboardSnapshot:
    timestamp: str
    equity: float
    pnl: float
    drawdown: float
    win_rate: float
    sharpe: float
    open_positions: int
    alerts: int


class PerformanceDashboard:
    def __init__(self, initial_equity: float = 10000):
        self.initial_equity = initial_equity
        self.equity_curve: list[float] = [initial_equity]
        self.trades: list[float] = []
        self.alert_count = 0

    def update(self, equity: float, trade_pnl: float | None = None, alerts: int = 0) -> DashboardSnapshot:
        self.equity_curve.append(equity)
        if trade_pnl is not None:
            self.trades.append(trade_pnl)
        self.alert_count += alerts
        return self.snapshot(open_positions=0)

    def snapshot(self, open_positions: int = 0) -> DashboardSnapshot:
        equity = self.equity_curve[-1]
        peak = max(self.equity_curve)
        returns = np.diff(self.equity_curve) / np.maximum(self.equity_curve[:-1], 1e-9)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)) if len(returns) else 0.0
        wins = sum(1 for trade in self.trades if trade > 0)
        return DashboardSnapshot(
            datetime.now(timezone.utc).isoformat(),
            float(equity),
            float(equity - self.initial_equity),
            float((peak - equity) / max(peak, 1e-9)),
            float(wins / len(self.trades)) if self.trades else 0.0,
            sharpe,
            open_positions,
            self.alert_count,
        )

    def render_text(self, open_positions: int = 0) -> str:
        snap = self.snapshot(open_positions)
        return json.dumps(asdict(snap), indent=2)


def _demo() -> None:
    dashboard = PerformanceDashboard()
    dashboard.update(10080, 80)
    dashboard.update(10020, -60, alerts=1)
    print("Performance dashboard ready")
    print(dashboard.render_text(open_positions=2))


if __name__ == "__main__":
    _demo()
