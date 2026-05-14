"""
Self-Optimization Meta-Engine — ARGUS continuously improves itself.

This is the brain that watches ARGUS trade and identifies what to fix.
It runs every N cycles and produces actionable optimization directives.

What it monitors and optimizes:
1. Per-strategy P&L → disable losers, boost winners
2. Per-symbol edge → stop trading symbols with no edge
3. Gate effectiveness → identify gates that always reduce and never help
4. Sizing accuracy → compare intended size vs actual fill size
5. Signal quality → track which signals produce winners vs losers
6. Regime accuracy → does the detected regime match actual market behavior
7. Evolver progress → is the evolver finding better strategies over time
8. Execution quality → slippage, fill rate, latency trends
9. Correlation drift → are position correlations increasing risk
10. Recovery speed → how fast does ARGUS recover from drawdowns

Every optimization produces a directive that the trading system can act on:
- DISABLE_STRATEGY, ENABLE_STRATEGY
- REDUCE_SYMBOL_ALLOCATION, INCREASE_SYMBOL_ALLOCATION
- ADJUST_GATE_WEIGHT
- RETRAIN_MODEL
- TRIGGER_EVOLUTION
- REBALANCE_PORTFOLIO
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OptimizationDirective:
    """A single optimization action to take."""
    action: str                 # e.g. "DISABLE_STRATEGY", "RETRAIN_MODEL"
    target: str                 # e.g. strategy name, symbol, model name
    reason: str                 # human-readable explanation
    priority: float             # 0.0 (low) to 1.0 (urgent)
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationReport:
    """Result of one self-optimization cycle."""
    cycle: int
    directives: List[OptimizationDirective]
    strategy_scores: Dict[str, float]       # strategy → rolling Sharpe
    symbol_scores: Dict[str, float]         # symbol → edge score
    system_health: float                    # 0-100
    improvements_applied: int
    duration_ms: float


class SelfOptimizer:
    """
    Autonomous self-improvement engine.

    Watches trading performance and produces optimization directives.
    Runs every optimization_interval cycles (default 100).
    """

    def __init__(
        self,
        optimization_interval: int = 100,
        strategy_disable_sharpe: float = -0.5,
        strategy_enable_sharpe: float = 0.3,
        min_trades_to_judge: int = 10,
        symbol_min_edge: float = -0.02,     # disable symbol if avg return < -2%
        correlation_threshold: float = 0.85,
        max_drawdown_recovery_cycles: int = 500,
    ):
        self._interval = optimization_interval
        self._disable_sharpe = strategy_disable_sharpe
        self._enable_sharpe = strategy_enable_sharpe
        self._min_trades = min_trades_to_judge
        self._symbol_min_edge = symbol_min_edge
        self._corr_threshold = correlation_threshold
        self._max_recovery = max_drawdown_recovery_cycles

        # Performance tracking
        self._strategy_trades: Dict[str, List[float]] = defaultdict(list)  # strategy → list of returns
        self._symbol_trades: Dict[str, List[float]] = defaultdict(list)    # symbol → list of returns
        self._signal_outcomes: Dict[str, List[bool]] = defaultdict(list)   # source → list of correct predictions
        self._gate_hits: Dict[str, int] = defaultdict(int)                 # gate_name → times triggered
        self._gate_helped: Dict[str, int] = defaultdict(int)               # gate_name → times trade was a loser (gate was right)
        self._slippage_history: List[float] = []
        self._fill_rates: List[float] = []
        self._drawdown_start_cycle: Optional[int] = None
        self._cycle_count = 0
        self._directives_history: List[OptimizationDirective] = []

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        pnl_pct: float,
        signal_source: str = "",
        slippage_bps: float = 0.0,
        fill_rate: float = 1.0,
    ) -> None:
        """Record a completed trade for optimization analysis."""
        self._strategy_trades[strategy].append(pnl_pct)
        self._symbol_trades[symbol].append(pnl_pct)
        if signal_source:
            self._signal_outcomes[signal_source].append(pnl_pct > 0)
        self._slippage_history.append(slippage_bps)
        self._fill_rates.append(fill_rate)

        # Keep last 200 per category
        if len(self._strategy_trades[strategy]) > 200:
            self._strategy_trades[strategy] = self._strategy_trades[strategy][-200:]
        if len(self._symbol_trades[symbol]) > 200:
            self._symbol_trades[symbol] = self._symbol_trades[symbol][-200:]
        if len(self._slippage_history) > 500:
            self._slippage_history = self._slippage_history[-500:]

    def record_gate_hit(self, gate_name: str, trade_was_loser: bool) -> None:
        """Record when a gate reduced position size, and whether the trade lost."""
        self._gate_hits[gate_name] += 1
        if trade_was_loser:
            self._gate_helped[gate_name] += 1

    def record_drawdown_start(self, cycle: int) -> None:
        if self._drawdown_start_cycle is None:
            self._drawdown_start_cycle = cycle

    def record_drawdown_end(self) -> None:
        self._drawdown_start_cycle = None

    def optimize(self, cycle: int, advisory: Optional[Dict[str, Any]] = None) -> OptimizationReport:
        """
        Run one optimization cycle. Returns directives to improve performance.
        Called every optimization_interval cycles.
        """
        t0 = time.time()
        self._cycle_count = cycle
        directives: List[OptimizationDirective] = []
        strategy_scores: Dict[str, float] = {}
        symbol_scores: Dict[str, float] = {}

        if advisory is None:
            advisory = {}

        # ── 1. Per-strategy analysis ──
        for strategy, trades in self._strategy_trades.items():
            if len(trades) < self._min_trades:
                continue
            mean_ret = sum(trades) / len(trades)
            std_ret = (sum((t - mean_ret) ** 2 for t in trades) / max(len(trades) - 1, 1)) ** 0.5
            sharpe = mean_ret / max(std_ret, 1e-9)
            strategy_scores[strategy] = sharpe

            if sharpe < self._disable_sharpe:
                directives.append(OptimizationDirective(
                    action="DISABLE_STRATEGY", target=strategy,
                    reason=f"Sharpe={sharpe:.2f} < {self._disable_sharpe} over {len(trades)} trades",
                    priority=0.9,
                ))
            elif sharpe > self._enable_sharpe * 2:
                directives.append(OptimizationDirective(
                    action="BOOST_STRATEGY", target=strategy,
                    reason=f"Strong edge: Sharpe={sharpe:.2f} over {len(trades)} trades",
                    priority=0.6,
                    params={"boost_multiplier": min(1.5, 1.0 + sharpe * 0.2)},
                ))

        # ── 2. Per-symbol analysis ──
        for symbol, trades in self._symbol_trades.items():
            if len(trades) < self._min_trades:
                continue
            mean_ret = sum(trades) / len(trades)
            symbol_scores[symbol] = mean_ret

            if mean_ret < self._symbol_min_edge:
                directives.append(OptimizationDirective(
                    action="REDUCE_SYMBOL_ALLOCATION", target=symbol,
                    reason=f"Avg return={mean_ret:.3f}% over {len(trades)} trades — no edge",
                    priority=0.7,
                    params={"reduce_by": 0.5},
                ))

        # ── 3. Signal quality analysis ──
        for source, outcomes in self._signal_outcomes.items():
            if len(outcomes) < 20:
                continue
            accuracy = sum(outcomes) / len(outcomes)
            if accuracy < 0.40:
                directives.append(OptimizationDirective(
                    action="REDUCE_SIGNAL_WEIGHT", target=source,
                    reason=f"Signal accuracy={accuracy:.0%} (below 40%) over {len(outcomes)} signals",
                    priority=0.5,
                    params={"accuracy": accuracy},
                ))
            elif accuracy > 0.65:
                directives.append(OptimizationDirective(
                    action="BOOST_SIGNAL_WEIGHT", target=source,
                    reason=f"Signal accuracy={accuracy:.0%} — high quality",
                    priority=0.4,
                    params={"accuracy": accuracy},
                ))

        # ── 4. Gate effectiveness analysis ──
        for gate, hits in self._gate_hits.items():
            if hits < 20:
                continue
            helped = self._gate_helped.get(gate, 0)
            effectiveness = helped / hits
            if effectiveness < 0.30:
                directives.append(OptimizationDirective(
                    action="RELAX_GATE", target=gate,
                    reason=f"Gate only helped {effectiveness:.0%} of {hits} times — mostly hurting P&L",
                    priority=0.4,
                    params={"effectiveness": effectiveness},
                ))

        # ── 5. Execution quality ──
        if len(self._slippage_history) >= 20:
            avg_slip = sum(self._slippage_history[-50:]) / len(self._slippage_history[-50:])
            if avg_slip > 5.0:  # > 5 bps average slippage
                directives.append(OptimizationDirective(
                    action="IMPROVE_EXECUTION", target="slippage",
                    reason=f"Avg slippage={avg_slip:.1f}bps — switch to limit orders",
                    priority=0.6,
                    params={"avg_slippage_bps": avg_slip},
                ))

        if len(self._fill_rates) >= 20:
            avg_fill = sum(self._fill_rates[-50:]) / len(self._fill_rates[-50:])
            if avg_fill < 0.85:
                directives.append(OptimizationDirective(
                    action="IMPROVE_EXECUTION", target="fill_rate",
                    reason=f"Avg fill rate={avg_fill:.0%} — orders too aggressive",
                    priority=0.5,
                    params={"avg_fill_rate": avg_fill},
                ))

        # ── 6. Evolver progress ──
        evolver_adv = advisory.get("strategy_evolver", {})
        if isinstance(evolver_adv, dict):
            stagnation = int(evolver_adv.get("stagnation_counter", 0) or 0)
            if stagnation >= 8:
                directives.append(OptimizationDirective(
                    action="TRIGGER_EVOLUTION", target="strategy_evolver",
                    reason=f"Evolver stagnant for {stagnation} generations — inject chaos",
                    priority=0.7,
                ))

        # ── 7. Recovery speed ──
        if self._drawdown_start_cycle is not None:
            recovery_cycles = cycle - self._drawdown_start_cycle
            if recovery_cycles > self._max_recovery:
                directives.append(OptimizationDirective(
                    action="REBALANCE_PORTFOLIO", target="all",
                    reason=f"Drawdown recovery taking {recovery_cycles} cycles (max={self._max_recovery})",
                    priority=0.8,
                ))

        # ── 8. System health from advisory ──
        health_adv = advisory.get("health_score", {})
        system_health = 50.0
        if isinstance(health_adv, dict):
            system_health = float(health_adv.get("score", 50) or 50)
            if system_health < 30:
                directives.append(OptimizationDirective(
                    action="REDUCE_EXPOSURE", target="all",
                    reason=f"System health={system_health:.0f}/100 — critical",
                    priority=0.95,
                    params={"health": system_health},
                ))

        # Sort by priority descending
        directives.sort(key=lambda d: d.priority, reverse=True)
        self._directives_history.extend(directives)
        if len(self._directives_history) > 500:
            self._directives_history = self._directives_history[-500:]

        duration_ms = (time.time() - t0) * 1000

        if directives:
            logger.info(
                "SelfOptimizer cycle %d: %d directives (top: %s %s — %s)",
                cycle, len(directives),
                directives[0].action, directives[0].target, directives[0].reason,
            )

        return OptimizationReport(
            cycle=cycle,
            directives=directives,
            strategy_scores=strategy_scores,
            symbol_scores=symbol_scores,
            system_health=system_health,
            improvements_applied=0,  # caller tracks
            duration_ms=duration_ms,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_directives": len(self._directives_history),
            "strategies_tracked": len(self._strategy_trades),
            "symbols_tracked": len(self._symbol_trades),
            "signals_tracked": len(self._signal_outcomes),
            "avg_slippage_bps": sum(self._slippage_history[-50:]) / max(len(self._slippage_history[-50:]), 1) if self._slippage_history else 0,
            "avg_fill_rate": sum(self._fill_rates[-50:]) / max(len(self._fill_rates[-50:]), 1) if self._fill_rates else 1.0,
        }
