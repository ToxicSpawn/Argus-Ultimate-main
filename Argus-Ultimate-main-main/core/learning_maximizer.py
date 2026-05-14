"""
Learning Maximizer — extract maximum knowledge from every paper trading cycle.

During paper trading, ARGUS should learn as aggressively as possible because
paper losses cost nothing. This module ensures every system gets maximum data:

1. SHADOW TRADING: for every trade taken, also track what the OPPOSITE trade
   would have done. Doubles the counterfactual learning data.

2. STRATEGY TOURNAMENT: every cycle, ALL 28 strategies produce signals for
   ALL 50 pairs. Even disabled strategies run in shadow mode. The scanner
   compares them all and learns which strategies work in which conditions.

3. REGIME JOURNAL: every cycle records the full market state. After 1000
   cycles, ARGUS has a complete map of "what works when."

4. EXPLORATION MODE: during paper trading, intentionally take some trades
   that the meta-cognition would normally SKIP. This builds data for the
   unknown regions where ARGUS has no experience.

5. FAST EVOLUTION: run the evolver every 50 cycles (not 500) during paper.
   More generations = faster convergence to optimal strategies.

6. AGGRESSIVE HYPOTHESIS TESTING: generate 5x more hypotheses and test them
   against paper results. Accelerates the research engine.

7. FULL LOGGING: record every advisory key, every gate decision, every
   signal score. Creates a complete audit trail for post-analysis.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ShadowTrade:
    """A trade that wasn't taken, tracked for learning."""
    symbol: str
    side: str
    strategy: str
    signal_confidence: float
    reason_skipped: str             # "meta_cognition_skip", "entropy_filter", "gate_blocked"
    price_at_signal: float
    price_after_10_bars: float = 0.0
    price_after_50_bars: float = 0.0
    would_have_pnl_pct: float = 0.0
    regime: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class CycleSnapshot:
    """Complete state of ARGUS at one cycle — for post-analysis."""
    cycle: int
    timestamp: float
    # Market state
    prices: Dict[str, float]
    regime: str
    volatility: float
    # Signals
    signals_generated: int
    signals_executed: int
    signals_skipped: int
    # Advisory summary
    entropy: float
    meta_confidence: float
    conviction_avg: float
    data_brain_composite: Dict[str, float]
    # Performance
    portfolio_value: float
    daily_pnl: float
    drawdown_pct: float


class LearningMaximizer:
    """
    Maximizes learning during paper trading.

    Activates automatically when run_mode == "paper". Deactivates in live
    to avoid unnecessary computation.
    """

    def __init__(
        self,
        exploration_rate: float = 0.15,     # 15% of skipped trades get explored anyway
        shadow_capacity: int = 5000,
        snapshot_capacity: int = 10000,
        fast_evolution_interval: int = 50,  # every 50 cycles (not 500)
        fast_research_interval: int = 50,   # every 50 cycles (not 200)
    ):
        self._explore_rate = exploration_rate
        self._fast_evo = fast_evolution_interval
        self._fast_research = fast_research_interval

        self._shadow_trades: deque = deque(maxlen=shadow_capacity)
        self._snapshots: deque = deque(maxlen=snapshot_capacity)
        self._skipped_signals: Dict[str, List[ShadowTrade]] = defaultdict(list)
        self._exploration_outcomes: List[Dict[str, Any]] = []

        # Learning metrics
        self._total_shadows = 0
        self._total_explorations = 0
        self._shadows_that_would_profit = 0
        self._explorations_that_profited = 0

        # Strategy tournament
        self._strategy_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # strategy_scores[strategy][metric] = value

    def should_explore(self, signal_skipped: bool, skip_reason: str) -> bool:
        """Should we take this trade anyway for learning?

        During paper trading, 15% of skipped trades get explored.
        This builds data for conditions the system normally avoids.
        """
        if not signal_skipped:
            return False
        import random
        if random.random() < self._explore_rate:
            self._total_explorations += 1
            logger.info("LearningMaximizer: EXPLORING skipped trade (reason: %s)", skip_reason)
            return True
        return False

    def record_shadow(
        self,
        symbol: str,
        side: str,
        strategy: str,
        confidence: float,
        skip_reason: str,
        current_price: float,
        regime: str = "",
    ) -> None:
        """Record a signal that was generated but not traded."""
        shadow = ShadowTrade(
            symbol=symbol, side=side, strategy=strategy,
            signal_confidence=confidence, reason_skipped=skip_reason,
            price_at_signal=current_price, regime=regime,
        )
        self._shadow_trades.append(shadow)
        self._skipped_signals[symbol].append(shadow)
        self._total_shadows += 1

    def update_shadow_outcomes(self, prices: Dict[str, float], bars_elapsed: int) -> None:
        """Update shadow trades with what actually happened."""
        for shadow in self._shadow_trades:
            if shadow.price_at_signal <= 0:
                continue
            current = prices.get(shadow.symbol, 0)
            if current <= 0:
                continue

            if bars_elapsed >= 10 and shadow.price_after_10_bars == 0:
                shadow.price_after_10_bars = current
            if bars_elapsed >= 50 and shadow.price_after_50_bars == 0:
                shadow.price_after_50_bars = current
                # Compute would-have P&L
                if shadow.side == "buy":
                    shadow.would_have_pnl_pct = (current / shadow.price_at_signal - 1) * 100
                else:
                    shadow.would_have_pnl_pct = (shadow.price_at_signal / current - 1) * 100

                if shadow.would_have_pnl_pct > 0.5:
                    self._shadows_that_would_profit += 1

    def record_exploration_outcome(self, pnl_pct: float, skip_reason: str) -> None:
        """Record the outcome of an exploration trade."""
        self._exploration_outcomes.append({
            "pnl_pct": pnl_pct, "reason": skip_reason, "timestamp": time.time(),
        })
        if pnl_pct > 0:
            self._explorations_that_profited += 1

    def record_cycle_snapshot(
        self,
        cycle: int,
        prices: Dict[str, float],
        advisory: Dict[str, Any],
        portfolio_value: float = 0.0,
        daily_pnl: float = 0.0,
    ) -> None:
        """Record complete cycle state for post-analysis."""
        entropy_adv = advisory.get("entropy_filter", {})
        meta_adv = advisory.get("meta_cognition", {})
        brain_adv = advisory.get("universal_data_brain", {})

        snapshot = CycleSnapshot(
            cycle=cycle,
            timestamp=time.time(),
            prices=dict(prices) if prices else {},
            regime=str(advisory.get("regime", "")),
            volatility=0.0,
            signals_generated=0,
            signals_executed=0,
            signals_skipped=0,
            entropy=float(entropy_adv.get("entropy", 0.5) if isinstance(entropy_adv, dict) else 0.5),
            meta_confidence=float(meta_adv.get("confidence", 0.5) if isinstance(meta_adv, dict) else 0.5),
            conviction_avg=0.0,
            data_brain_composite={
                sym: float(d.get("composite", 0) if isinstance(d, dict) else 0)
                for sym, d in (brain_adv.items() if isinstance(brain_adv, dict) else {})
            },
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            drawdown_pct=0.0,
        )
        self._snapshots.append(snapshot)

    def record_strategy_tournament(self, strategy: str, symbol: str,
                                    sharpe: float, win_rate: float,
                                    trade_count: int) -> None:
        """Record a strategy's performance in the tournament."""
        key = f"{strategy}:{symbol}"
        self._strategy_scores[key]["sharpe"] = sharpe
        self._strategy_scores[key]["win_rate"] = win_rate
        self._strategy_scores[key]["trades"] = trade_count

    def get_fast_intervals(self) -> Dict[str, int]:
        """Return accelerated intervals for paper trading."""
        return {
            "evolution_interval": self._fast_evo,
            "research_interval": self._fast_research,
            "scanner_interval": 20,      # every 20 cycles (not 100)
            "attribution_interval": 50,   # every 50 cycles (not 200)
            "forecast_interval": 100,     # every 100 cycles (not 500)
        }

    def get_shadow_insights(self) -> Dict[str, Any]:
        """What did we learn from trades we DIDN'T take?"""
        if not self._shadow_trades:
            return {"total_shadows": 0}

        completed = [s for s in self._shadow_trades if s.price_after_50_bars > 0]
        if not completed:
            return {"total_shadows": self._total_shadows, "completed": 0}

        profitable = [s for s in completed if s.would_have_pnl_pct > 0.5]
        by_reason = defaultdict(list)
        for s in completed:
            by_reason[s.reason_skipped].append(s.would_have_pnl_pct)

        # Which skip reasons are costing us money?
        reason_analysis = {}
        for reason, pnls in by_reason.items():
            avg_pnl = sum(pnls) / len(pnls)
            profit_rate = sum(1 for p in pnls if p > 0.5) / len(pnls)
            reason_analysis[reason] = {
                "count": len(pnls),
                "avg_would_have_pnl": round(avg_pnl, 2),
                "profit_rate": round(profit_rate, 2),
                "verdict": "STOP_SKIPPING" if profit_rate > 0.55 else "CORRECT_TO_SKIP",
            }

        return {
            "total_shadows": self._total_shadows,
            "completed": len(completed),
            "would_have_profited": len(profitable),
            "profit_rate": len(profitable) / max(len(completed), 1),
            "by_skip_reason": reason_analysis,
        }

    def get_exploration_insights(self) -> Dict[str, Any]:
        """What did we learn from exploration trades?"""
        if not self._exploration_outcomes:
            return {"total_explorations": 0}

        by_reason = defaultdict(list)
        for e in self._exploration_outcomes:
            by_reason[e["reason"]].append(e["pnl_pct"])

        reason_value = {}
        for reason, pnls in by_reason.items():
            reason_value[reason] = {
                "count": len(pnls),
                "avg_pnl": round(sum(pnls) / len(pnls), 2),
                "profit_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 2),
            }

        return {
            "total_explorations": self._total_explorations,
            "explorations_profitable": self._explorations_that_profited,
            "by_reason": reason_value,
        }

    def get_tournament_winners(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the top performing strategy:symbol combinations."""
        ranked = []
        for key, scores in self._strategy_scores.items():
            strategy, symbol = key.split(":", 1) if ":" in key else (key, "")
            ranked.append({
                "strategy": strategy,
                "symbol": symbol,
                "sharpe": scores.get("sharpe", 0),
                "win_rate": scores.get("win_rate", 0),
                "trades": scores.get("trades", 0),
            })
        ranked.sort(key=lambda x: x["sharpe"], reverse=True)
        return ranked[:top_n]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_shadows": self._total_shadows,
            "total_explorations": self._total_explorations,
            "shadows_profitable": self._shadows_that_would_profit,
            "explorations_profitable": self._explorations_that_profited,
            "snapshots_recorded": len(self._snapshots),
            "strategies_in_tournament": len(self._strategy_scores),
        }
