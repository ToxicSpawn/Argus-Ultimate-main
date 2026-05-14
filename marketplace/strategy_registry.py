"""
Strategy Marketplace Registry — Cryptohopper-style strategy self-registration.

How Cryptohopper does it:
  Strategies are published by vendors with static backtest stats.
  Users subscribe for $5-30/month with zero code ownership.

How Argus Registry improves on it:
  1. Strategies self-register with LIVE stats, not backtest.
  2. Composite ranking score = 0.4*sharpe + 0.3*calmar + 0.2*win_rate + 0.1*profit_factor
  3. Hot-swap loading: swap to a better strategy mid-session without restart.
  4. Full source ownership: every strategy is your code, no vendor platform fees.
  5. Auto-deprecation: strategies with drawdown > threshold are flagged and
     suspended automatically (Cryptohopper never auto-suspends bad strategies).
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DRAWDOWN_SUSPEND_THRESHOLD = 0.15   # auto-suspend if live drawdown > 15%
MIN_LIVE_DAYS_FOR_RANKING = 3       # must have 3 days live data to be ranked


@dataclass
class StrategyRecord:
    strategy_id: str
    module_path: str            # e.g. 'strategies.sol_dca_superior'
    class_name: str             # e.g. 'SolSafeDCAStrategy'
    author: str
    description: str
    # Live performance metrics (updated by live_performance_ranker.py)
    sharpe: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    max_drawdown: float = 0.0
    live_days: float = 0.0
    total_trades: int = 0
    pnl_pct: float = 0.0
    registered_at: float = field(default_factory=time.time)
    suspended: bool = False
    suspension_reason: str = ""

    def composite_score(self) -> float:
        """Weighted ranking score (higher = better)."""
        if self.live_days < MIN_LIVE_DAYS_FOR_RANKING:
            return -999.0  # unranked until enough live data
        return (
            0.40 * self.sharpe
            + 0.30 * self.calmar
            + 0.20 * self.win_rate
            + 0.10 * self.profit_factor
        )

    def should_suspend(self) -> bool:
        return self.max_drawdown > DRAWDOWN_SUSPEND_THRESHOLD


class StrategyRegistry:
    """
    Central strategy marketplace registry.

    Usage:
        registry = StrategyRegistry()
        registry.register(StrategyRecord(strategy_id='sol_safe_dca', ...))
        best = registry.top_strategies(n=3)
        instance = registry.load(strategy_id='sol_safe_dca', capital_aud=1000)
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, StrategyRecord] = {}

    # ------------------------------------------------------------------
    def register(self, record: StrategyRecord) -> None:
        """Register or update a strategy."""
        self._strategies[record.strategy_id] = record

    def update_metrics(
        self,
        strategy_id: str,
        sharpe: float,
        calmar: float,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
        live_days: float,
        total_trades: int,
        pnl_pct: float,
    ) -> None:
        """Update live performance metrics. Called by live_performance_ranker."""
        if strategy_id not in self._strategies:
            return
        rec = self._strategies[strategy_id]
        rec.sharpe = sharpe
        rec.calmar = calmar
        rec.win_rate = win_rate
        rec.profit_factor = profit_factor
        rec.max_drawdown = max_drawdown
        rec.live_days = live_days
        rec.total_trades = total_trades
        rec.pnl_pct = pnl_pct
        # Auto-suspend if drawdown threshold breached
        if rec.should_suspend() and not rec.suspended:
            rec.suspended = True
            rec.suspension_reason = (
                f"auto_suspended: drawdown={max_drawdown:.2%} > "
                f"threshold={DRAWDOWN_SUSPEND_THRESHOLD:.2%}"
            )

    def top_strategies(
        self,
        n: int = 5,
        exclude_suspended: bool = True,
    ) -> List[StrategyRecord]:
        """Return top N strategies ranked by composite score."""
        candidates = [
            r for r in self._strategies.values()
            if not (exclude_suspended and r.suspended)
        ]
        return sorted(candidates, key=lambda r: r.composite_score(), reverse=True)[:n]

    def get(self, strategy_id: str) -> Optional[StrategyRecord]:
        return self._strategies.get(strategy_id)

    def all_records(self) -> List[StrategyRecord]:
        return list(self._strategies.values())

    def load(self, strategy_id: str, **kwargs: Any) -> Any:
        """
        Hot-load and instantiate a strategy class by ID.
        kwargs are passed to the strategy constructor.
        """
        rec = self._strategies.get(strategy_id)
        if rec is None:
            raise KeyError(f"Strategy '{strategy_id}' not registered")
        if rec.suspended:
            raise RuntimeError(
                f"Strategy '{strategy_id}' is suspended: {rec.suspension_reason}"
            )
        module = importlib.import_module(rec.module_path)
        cls = getattr(module, rec.class_name)
        return cls(**kwargs)

    def leaderboard_str(self) -> str:
        """Human-readable leaderboard for Telegram dashboard."""
        ranked = self.top_strategies(n=10)
        if not ranked:
            return "No ranked strategies yet (all suspended or awaiting live data)."
        lines = ["=== Strategy Leaderboard ==="]
        for i, r in enumerate(ranked, 1):
            status = "\u26d4" if r.suspended else "\u2705"
            lines.append(
                f"{i}. {status} {r.strategy_id} | "
                f"Score={r.composite_score():.2f} | "
                f"Sharpe={r.sharpe:.2f} | "
                f"DD={r.max_drawdown:.1%} | "
                f"PnL={r.pnl_pct:.1%} | "
                f"Trades={r.total_trades}"
            )
        return "\n".join(lines)
