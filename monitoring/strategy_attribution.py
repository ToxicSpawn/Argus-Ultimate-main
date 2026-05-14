"""
Strategy Attribution Engine — per-strategy P&L, factor decomposition, alpha measurement.

Answers the questions every quant asks:
1. Which strategy is actually making money? (per-strategy P&L)
2. Is the alpha real or just beta? (factor decomposition)
3. How much did execution cost me? (transaction cost attribution)
4. What regime generates the most alpha? (regime-conditional alpha)
5. Is each strategy earning its allocation? (capital efficiency)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StrategyAttribution:
    """Complete attribution for one strategy."""
    strategy: str
    total_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    avg_return: float = 0.0
    max_drawdown: float = 0.0
    # Factor decomposition
    alpha: float = 0.0              # return above market beta
    beta_contribution: float = 0.0  # return from market exposure
    execution_cost: float = 0.0     # total slippage + fees
    # Regime analysis
    best_regime: str = ""
    worst_regime: str = ""
    regime_pnl: Dict[str, float] = field(default_factory=dict)
    # Capital efficiency
    capital_allocated: float = 0.0
    return_on_capital: float = 0.0


@dataclass
class PortfolioAttribution:
    """Portfolio-level attribution."""
    total_pnl: float
    strategies: Dict[str, StrategyAttribution]
    top_contributor: str
    worst_contributor: str
    total_execution_cost: float
    total_alpha: float
    total_beta: float
    pnl_by_regime: Dict[str, float]
    pnl_by_symbol: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class StrategyAttributionEngine:
    """
    Real-time strategy attribution.

    Records every trade with its strategy, then computes attribution
    metrics on demand. Feeds back into the self-optimizer.
    """

    def __init__(self, market_benchmark_returns: Optional[List[float]] = None):
        self._trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._symbol_pnl: Dict[str, float] = defaultdict(float)
        self._regime_pnl: Dict[str, float] = defaultdict(float)
        self._execution_costs: Dict[str, float] = defaultdict(float)
        self._capital_allocated: Dict[str, float] = defaultdict(float)
        self._benchmark: List[float] = market_benchmark_returns or []

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        pnl: float,
        slippage_bps: float = 0.0,
        fee_usd: float = 0.0,
        regime: str = "normal",
        capital_used: float = 0.0,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
    ) -> None:
        """Record a completed trade for attribution."""
        trade = {
            "strategy": strategy,
            "symbol": symbol,
            "pnl": pnl,
            "slippage_bps": slippage_bps,
            "fee_usd": fee_usd,
            "regime": regime,
            "capital_used": capital_used,
            "timestamp": time.time(),
        }
        self._trades[strategy].append(trade)
        self._symbol_pnl[symbol] += pnl
        self._regime_pnl[regime] += pnl
        exec_cost = slippage_bps / 10000 * capital_used + fee_usd
        self._execution_costs[strategy] += exec_cost
        self._capital_allocated[strategy] = max(
            self._capital_allocated[strategy], capital_used
        )

        # Keep last 500 per strategy
        if len(self._trades[strategy]) > 500:
            self._trades[strategy] = self._trades[strategy][-500:]

    def record_benchmark_return(self, ret: float) -> None:
        """Record market benchmark return (e.g. BTC return per cycle)."""
        self._benchmark.append(ret)
        if len(self._benchmark) > 1000:
            self._benchmark = self._benchmark[-1000:]

    def compute(self) -> PortfolioAttribution:
        """Compute full portfolio attribution."""
        strategy_attrs: Dict[str, StrategyAttribution] = {}
        total_pnl = 0.0
        total_exec_cost = 0.0
        total_alpha = 0.0
        total_beta = 0.0

        for strategy, trades in self._trades.items():
            if not trades:
                continue

            pnls = [t["pnl"] for t in trades]
            n = len(pnls)
            total = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            mean_ret = total / n
            std_ret = (sum((p - mean_ret) ** 2 for p in pnls) / max(n - 1, 1)) ** 0.5
            sharpe = mean_ret / max(std_ret, 1e-9)

            # Max drawdown
            eq = [0.0]
            for p in pnls:
                eq.append(eq[-1] + p)
            peak = eq[0]
            max_dd = 0.0
            for e in eq:
                peak = max(peak, e)
                max_dd = max(max_dd, peak - e)

            # Factor decomposition: alpha = return - beta * benchmark_return
            beta = 0.0
            alpha_val = mean_ret
            if self._benchmark and len(self._benchmark) >= n:
                bench = self._benchmark[-n:]
                bench_mean = sum(bench) / len(bench)
                bench_var = sum((b - bench_mean) ** 2 for b in bench) / max(len(bench) - 1, 1)
                if bench_var > 1e-12:
                    cov = sum((p - mean_ret) * (b - bench_mean)
                              for p, b in zip(pnls, bench)) / max(n - 1, 1)
                    beta = cov / bench_var
                alpha_val = mean_ret - beta * bench_mean

            # Regime breakdown
            regime_pnl: Dict[str, float] = defaultdict(float)
            for t in trades:
                regime_pnl[t.get("regime", "normal")] += t["pnl"]

            best_regime = max(regime_pnl, key=regime_pnl.get) if regime_pnl else ""
            worst_regime = min(regime_pnl, key=regime_pnl.get) if regime_pnl else ""

            # Capital efficiency
            capital = self._capital_allocated.get(strategy, 1.0)
            roc = total / max(capital, 1.0)

            attr = StrategyAttribution(
                strategy=strategy, total_pnl=total, trade_count=n,
                win_count=wins, win_rate=wins / n if n > 0 else 0,
                sharpe=sharpe, avg_return=mean_ret, max_drawdown=max_dd,
                alpha=alpha_val, beta_contribution=beta * (sum(self._benchmark[-n:]) / n if self._benchmark else 0),
                execution_cost=self._execution_costs.get(strategy, 0),
                best_regime=best_regime, worst_regime=worst_regime,
                regime_pnl=dict(regime_pnl),
                capital_allocated=capital, return_on_capital=roc,
            )
            strategy_attrs[strategy] = attr
            total_pnl += total
            total_exec_cost += attr.execution_cost
            total_alpha += alpha_val * n
            total_beta += attr.beta_contribution * n

        top = max(strategy_attrs.values(), key=lambda a: a.total_pnl).strategy if strategy_attrs else ""
        worst = min(strategy_attrs.values(), key=lambda a: a.total_pnl).strategy if strategy_attrs else ""

        return PortfolioAttribution(
            total_pnl=total_pnl,
            strategies=strategy_attrs,
            top_contributor=top,
            worst_contributor=worst,
            total_execution_cost=total_exec_cost,
            total_alpha=total_alpha / max(sum(len(t) for t in self._trades.values()), 1),
            total_beta=total_beta / max(sum(len(t) for t in self._trades.values()), 1),
            pnl_by_regime=dict(self._regime_pnl),
            pnl_by_symbol=dict(self._symbol_pnl),
        )

    def get_stats(self) -> Dict[str, Any]:
        total_trades = sum(len(t) for t in self._trades.values())
        return {
            "strategies_tracked": len(self._trades),
            "total_trades": total_trades,
            "total_pnl": sum(self._symbol_pnl.values()),
            "pnl_by_regime": dict(self._regime_pnl),
            "top_strategy": max(self._trades, key=lambda k: sum(t["pnl"] for t in self._trades[k]))
            if self._trades else "",
        }
