"""Push 66 — Per-strategy drawdown and daily loss limits.

Each strategy gets its own independent risk budget:
  - max_drawdown_pct: strategy-level peak-to-trough halt
  - max_daily_loss_usd: daily loss cap per strategy
  - cooldown_bars: bars to wait after halt before reactivating
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StrategyRiskBudget:
    name: str
    max_drawdown_pct: float = 3.0     # % drawdown to halt this strategy
    max_daily_loss_usd: float = 100.0 # USD daily loss cap
    cooldown_bars: int = 50           # bars to wait after halt

    # Runtime state
    peak_equity: float = field(default=0.0, repr=False)
    session_pnl: float = field(default=0.0, repr=False)
    halted: bool = field(default=False, repr=False)
    bars_since_halt: int = field(default=0, repr=False)

    def __post_init__(self):
        self.peak_equity = 0.0  # set on first update


class PerStrategyRisk:
    """Independent risk tracking per strategy name."""

    def __init__(self, default_max_dd: float = 3.0,
                 default_max_daily: float = 100.0):
        self.default_max_dd = default_max_dd
        self.default_max_daily = default_max_daily
        self._budgets: Dict[str, StrategyRiskBudget] = {}

    def register(self, name: str, budget: StrategyRiskBudget | None = None):
        if budget is None:
            budget = StrategyRiskBudget(
                name=name,
                max_drawdown_pct=self.default_max_dd,
                max_daily_loss_usd=self.default_max_daily,
            )
        self._budgets[name] = budget

    def update(self, name: str, current_equity: float, trade_pnl: float) -> bool:
        """Update strategy equity + PnL. Returns True if strategy is active."""
        if name not in self._budgets:
            self.register(name)
        b = self._budgets[name]

        if b.peak_equity == 0.0:
            b.peak_equity = current_equity

        # Uncount halt cooldown
        if b.halted:
            b.bars_since_halt += 1
            if b.bars_since_halt >= b.cooldown_bars:
                b.halted = False
                b.bars_since_halt = 0
                b.session_pnl = 0.0
            return False

        b.peak_equity = max(b.peak_equity, current_equity)
        b.session_pnl += trade_pnl

        drawdown_pct = ((b.peak_equity - current_equity) / b.peak_equity) * 100.0
        if drawdown_pct >= b.max_drawdown_pct:
            b.halted = True
            b.bars_since_halt = 0
            return False

        if b.session_pnl <= -b.max_daily_loss_usd:
            b.halted = True
            b.bars_since_halt = 0
            return False

        return True

    def is_active(self, name: str) -> bool:
        if name not in self._budgets:
            return True
        return not self._budgets[name].halted

    def get_budget(self, name: str) -> StrategyRiskBudget | None:
        return self._budgets.get(name)

    def reset_session(self) -> None:
        for b in self._budgets.values():
            b.session_pnl = 0.0
