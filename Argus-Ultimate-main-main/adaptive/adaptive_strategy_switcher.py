"""Adaptive Strategy Switcher.

Features:
- Strategy performance tracking
- Automatic strategy switching based on market conditions
- Strategy ranking and selection
- Multi-strategy portfolio
- Smooth transitions between strategies
- Strategy health monitoring
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    ACTIVE = "active"
    WARM_UP = "warm_up"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"
    FAILED = "failed"


@dataclass
class StrategyPerformance:
    strategy_id: str
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_count: int = 0
    avg_trade_duration: float = 0.0
    last_signal_time: float = 0.0
    uptime_pct: float = 0.0
    score: float = 0.0


@dataclass
class StrategyMetrics:
    strategy_id: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_time: float = 0.0
    timestamp: float = 0.0


class StrategyMetricsTracker:
    def __init__(self):
        self._trades: deque = deque(maxlen=500)
        self._open_positions: Dict[str, Dict] = {}

    def open_position(
        self,
        strategy_id: str,
        symbol: str,
        entry_price: float,
        qty: float,
        side: str,
    ) -> None:
        key = f"{strategy_id}:{symbol}"
        self._open_positions[key] = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "qty": qty,
            "side": side,
            "entry_time": time.time(),
        }

    def close_position(
        self,
        strategy_id: str,
        symbol: str,
        exit_price: float,
    ) -> Optional[float]:
        key = f"{strategy_id}:{symbol}"
        if key not in self._open_positions:
            return None

        pos = self._open_positions.pop(key)
        pnl = (exit_price - pos["entry_price"]) * pos["qty"]
        
        if pos["side"] == "SELL":
            pnl = -pnl

        self._trades.append({
            "strategy_id": strategy_id,
            "symbol": symbol,
            "pnl": pnl,
            "holding_time": time.time() - pos["entry_time"],
            "timestamp": time.time(),
        })

        return pnl

    def get_metrics(self, strategy_id: str) -> StrategyMetrics:
        strategy_trades = [t for t in self._trades if t["strategy_id"] == strategy_id]

        if not strategy_trades:
            return StrategyMetrics(
                strategy_id=strategy_id,
                timestamp=time.time(),
            )

        pnls = [t["pnl"] for t in strategy_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return StrategyMetrics(
            strategy_id=strategy_id,
            realized_pnl=sum(pnls),
            trade_count=len(strategy_trades),
            win_count=len(wins),
            loss_count=len(losses),
            avg_win=np.mean(wins) if wins else 0.0,
            avg_loss=np.mean(losses) if losses else 0.0,
            largest_win=max(wins) if wins else 0.0,
            largest_loss=min(losses) if losses else 0.0,
            avg_holding_time=np.mean([t["holding_time"] for t in strategy_trades]),
            timestamp=time.time(),
        )


class StrategySelector:
    def __init__(self):
        self._strategies: Dict[str, StrategyPerformance] = {}
        self._rankings: List[Tuple[str, float]] = []

    def add_strategy(
        self,
        strategy_id: str,
        initial_score: float = 0.0,
    ) -> None:
        self._strategies[strategy_id] = StrategyPerformance(
            strategy_id=strategy_id,
            score=initial_score,
        )

    def update_performance(
        self,
        strategy_id: str,
        metrics: StrategyMetrics,
    ) -> None:
        if strategy_id not in self._strategies:
            self.add_strategy(strategy_id)

        perf = self._strategies[strategy_id]

        perf.trade_count = metrics.trade_count
        perf.total_return_pct = metrics.realized_pnl
        perf.avg_trade_duration = metrics.avg_holding_time

        if metrics.trade_count > 0:
            perf.win_rate_pct = metrics.win_count / metrics.trade_count * 100

            avg_win = metrics.avg_win
            avg_loss = abs(metrics.avg_loss)
            if avg_loss > 0:
                perf.sharpe_ratio = avg_win / avg_loss

        perf.score = self._calculate_score(perf)

    def _calculate_score(self, perf: StrategyPerformance) -> float:
        score = 0.0

        score += max(0, perf.sharpe_ratio) * 20

        score += perf.win_rate_pct * 0.3

        score += min(20, perf.total_return_pct * 0.5)

        decay = min(1.0, perf.trade_count / 20)
        score *= decay

        return max(0, score)

    def get_best_strategy(
        self,
        exclude: List[str] = None,
    ) -> Optional[str]:
        exclude = exclude or []

        available = [
            (sid, perf.score)
            for sid, perf in self._strategies.items()
            if sid not in exclude and perf.score > 0
        ]

        if not available:
            return None

        return max(available, key=lambda x: x[1])[0]

    def get_top_n(self, n: int = 3) -> List[Tuple[str, float]]:
        ranked = sorted(
            [(sid, p.score) for sid, p in self._strategies.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]

    def rank(self) -> List[str]:
        return [sid for sid, _ in self.get_top_n(10)]


class StrategySwitcher:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._tracker = StrategyMetricsTracker()
        self._selector = StrategySelector()

        self._active_strategy: Optional[str] = None
        self._strategy_states: Dict[str, StrategyState] = {}

        self._switch_history: deque = deque(maxlen=100)
        self._switch_callbacks: List[Callable] = []

        self._min_trades = self.config.get("min_trades_for_switch", 10)
        self._cooldown_secs = self.config.get("cooldown_seconds", 300)
        self._score_threshold = self.config.get("score_threshold", 0.5)
        self._warm_up_trades = self.config.get("warm_up_trades", 5)
        self._max_strategies = self.config.get("max_active_strategies", 3)

    def register_strategy(
        self,
        strategy_id: str,
        initial_score: float = 0.0,
    ) -> None:
        self._selector.add_strategy(strategy_id, initial_score)
        self._strategy_states[strategy_id] = StrategyState.WARM_UP

    def on_signal(
        self,
        strategy_id: str,
        symbol: str,
        entry_price: float,
        qty: float,
        side: str,
    ) -> None:
        self._tracker.open_position(strategy_id, symbol, entry_price, qty, side)

        perf = self._selector._strategies.get(strategy_id)
        if perf:
            perf.last_signal_time = time.time()

    def on_fill(
        self,
        strategy_id: str,
        symbol: str,
        exit_price: float,
    ) -> None:
        pnl = self._tracker.close_position(strategy_id, symbol, exit_price)

        if pnl is not None:
            metrics = self._tracker.get_metrics(strategy_id)
            self._selector.update_performance(strategy_id, metrics)

    def select_strategy(
        self,
        market_regime: Optional[str] = None,
        forced: Optional[str] = None,
    ) -> Optional[str]:
        if forced and forced in self._strategy_states:
            return forced

        if self._active_strategy:
            current_state = self._strategy_states.get(
                self._active_strategy, StrategyState.ACTIVE
            )
            if current_state == StrategyState.COOLDOWN:
                pass

        if len(self._selector._strategies) < 2:
            return self._active_strategy

        best = self._selector.get_best_strategy(
            exclude=self._get_excluded_strategies()
        )

        if best and best != self._active_strategy:
            self._switch_to(best, reason="better_performance")

        return self._active_strategy

    def _switch_to(
        self,
        strategy_id: str,
        reason: str = "",
    ) -> None:
        old_strategy = self._active_strategy
        self._active_strategy = strategy_id

        self._switch_history.append({
            "from": old_strategy,
            "to": strategy_id,
            "reason": reason,
            "timestamp": time.time(),
        })

        if old_strategy:
            self._strategy_states[old_strategy] = StrategyState.COOLDOWN

        self._strategy_states[strategy_id] = StrategyState.ACTIVE

        for callback in self._switch_callbacks:
            try:
                callback(old_strategy, strategy_id, reason)
            except Exception as e:
                logger.warning(f"Switch callback error: {e}")

        logger.info(f"Switched strategy: {old_strategy} -> {strategy_id} ({reason})")

    def _get_excluded_strategies(self) -> List[str]:
        excluded = []

        for sid, state in self._strategy_states.items():
            if state in [StrategyState.DISABLED, StrategyState.FAILED]:
                excluded.append(sid)
            elif state == StrategyState.COOLDOWN:
                if time.time() - self._get_last_switch_time(sid) < self._cooldown_secs:
                    excluded.append(sid)

        return excluded

    def _get_last_switch_time(self, strategy_id: str) -> float:
        for switch in reversed(list(self._switch_history)):
            if switch["to"] == strategy_id:
                return switch["timestamp"]
        return 0.0

    def register_switch_callback(
        self,
        callback: Callable[[str, str, str], None],
    ) -> None:
        self._switch_callbacks.append(callback)

    def get_active_strategy(self) -> Optional[str]:
        return self._active_strategy

    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        return self._tracker.get_metrics(strategy_id)

    def get_strategy_performance(
        self,
        strategy_id: str,
    ) -> Optional[StrategyPerformance]:
        return self._selector._strategies.get(strategy_id)

    def get_all_performances(self) -> Dict[str, StrategyPerformance]:
        return self._selector._strategies.copy()

    def get_rankings(self) -> List[Tuple[str, float]]:
        return self._selector.get_top_n()

    def disable_strategy(self, strategy_id: str) -> None:
        self._strategy_states[strategy_id] = StrategyState.DISABLED

    def enable_strategy(self, strategy_id: str) -> None:
        if strategy_id in self._strategy_states:
            self._strategy_states[strategy_id] = StrategyState.WARM_UP

    def force_switch(self, strategy_id: str, reason: str = "manual") -> None:
        if strategy_id in self._strategy_states:
            self._switch_to(strategy_id, reason)

    def get_switch_history(self) -> List[Dict]:
        return list(self._switch_history)


class MultiStrategyPortfolio:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._switcher = StrategySwitcher(config)

        self._allocations: Dict[str, float] = {}
        self._total_allocation = 0.0

        self._max_strategies = self.config.get("max_strategies", 3)
        self._rebalance_threshold = self.config.get("rebalance_threshold", 0.2)

    def add_strategy(
        self,
        strategy_id: str,
        allocation_pct: float = 0.0,
    ) -> None:
        self._switcher.register_strategy(strategy_id)

        if allocation_pct > 0:
            self._allocations[strategy_id] = allocation_pct
            self._total_allocation += allocation_pct

    def allocate(
        self,
        strategy_id: str,
        allocation_pct: float,
    ) -> None:
        old_allocation = self._allocations.get(strategy_id, 0)
        self._allocations[strategy_id] = allocation_pct
        self._total_allocation += allocation_pct - old_allocation

    def get_allocation(self, strategy_id: str) -> float:
        return self._allocations.get(strategy_id, 0.0) / max(0.01, self._total_allocation)

    def rebalance(self, market_regime: Optional[str] = None) -> Dict[str, float]:
        rankings = self._switcher.get_rankings()

        new_allocations = {}
        top_n = rankings[:self._max_strategies]

        if not top_n:
            return self._allocations.copy()

        total_score = sum(score for _, score in top_n)

        for strategy_id, score in top_n:
            weight = score / total_score if total_score > 0 else 1.0 / len(top_n)
            new_allocations[strategy_id] = weight

        for sid in self._allocations:
            if sid not in new_allocations:
                new_allocations[sid] = 0.0

        for strategy_id, new_alloc in new_allocations.items():
            old_alloc = self._allocations.get(strategy_id, 0.0)
            if abs(new_alloc - old_alloc) > self._rebalance_threshold:
                self._switcher.force_switch(strategy_id, "rebalance")

        self._allocations = new_allocations
        self._total_allocation = sum(new_allocations.values())

        return self.get_all_allocations()

    def get_all_allocations(self) -> Dict[str, float]:
        total = max(0.01, self._total_allocation)
        return {
            sid: alloc / total
            for sid, alloc in self._allocations.items()
        }

    def get_active_strategies(self) -> List[str]:
        return [
            sid for sid, alloc in self._allocations.items()
            if alloc > 0
        ]

    def process_signal(
        self,
        signal,
        market_regime: Optional[str] = None,
    ) -> Optional[str]:
        return self._switcher.select_strategy(market_regime)