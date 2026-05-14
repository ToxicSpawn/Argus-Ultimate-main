"""
Batch 3 – Rolling Performance Feeder
======================================
Feeds rolling per-strategy PnL metrics into the Self-Optimising Meta Engine
and Evaluation Engine so weight evolution keeps pace with live performance.

Called from the trading loop after each closed trade via feed_trade().
Also runs a periodic tick that pushes aggregated snapshots every N seconds.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from unified_trading_system import UnifiedSystemArchitecture

logger = logging.getLogger(__name__)

_WINDOW = 100   # rolling window of trades per strategy
_TICK_S = 30    # background aggregation interval (seconds)


class RollingPerfFeeder:
    """
    Aggregates closed-trade PnL per (strategy, symbol) and periodically pushes
    metrics into the strategy evaluation and self-optimising meta engines.
    """

    def __init__(self, system: "UnifiedSystemArchitecture") -> None:
        self._sys = system
        # (strategy, symbol) -> deque of trade dicts
        self._buckets: Dict[tuple, deque] = defaultdict(lambda: deque(maxlen=_WINDOW))
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_push_ts = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._task = loop.create_task(
                    self._tick_loop(), name="rolling_perf_feeder"
                )
        except Exception as exc:
            logger.debug("RollingPerfFeeder could not create async task: %s", exc)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    # ------------------------------------------------------------------
    # Public: called from trade-close path
    # ------------------------------------------------------------------

    def feed_trade(self, trade: Dict[str, Any]) -> None:
        """Record a closed trade. Called from the trading loop."""
        strategy = str(trade.get("strategy") or trade.get("source_strategy") or "unknown")
        symbol = str(trade.get("symbol") or "")
        key = (strategy, symbol)
        self._buckets[key].append(trade)
        # Opportunistic push (lightweight, no async required)
        if (time.time() - self._last_push_ts) >= _TICK_S:
            self._push_metrics()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(_TICK_S)
                self._push_metrics()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("RollingPerfFeeder tick error: %s", exc)

    def _push_metrics(self) -> None:
        self._last_push_ts = time.time()
        metrics_by_strategy: Dict[str, Dict[str, Any]] = {}

        for (strategy, symbol), trades in self._buckets.items():
            if not trades:
                continue
            pnl_list = [float(t.get("pnl") or t.get("pnl_aud") or 0.0) for t in trades]
            wins = sum(1 for p in pnl_list if p > 0)
            losses = sum(1 for p in pnl_list if p < 0)
            total = len(pnl_list)
            avg_pnl = sum(pnl_list) / total if total else 0.0
            gross_profit = sum(p for p in pnl_list if p > 0)
            gross_loss = abs(sum(p for p in pnl_list if p < 0))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
            win_rate = wins / total if total else 0.0
            expectancy = avg_pnl

            entry = metrics_by_strategy.setdefault(strategy, {
                "total": 0, "wins": 0, "losses": 0,
                "total_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            })
            entry["total"] += total
            entry["wins"] += wins
            entry["losses"] += losses
            entry["total_pnl"] += sum(pnl_list)
            entry["gross_profit"] += gross_profit
            entry["gross_loss"] += gross_loss

        for strategy, agg in metrics_by_strategy.items():
            total = agg["total"]
            if total == 0:
                continue
            gross_loss = agg["gross_loss"]
            profit_factor = (
                agg["gross_profit"] / gross_loss if gross_loss > 0 else float("inf")
            )
            win_rate = agg["wins"] / total
            expectancy = agg["total_pnl"] / total
            metric_payload = {
                "strategy": strategy,
                "total_trades": total,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "total_pnl": agg["total_pnl"],
            }
            self._push_to_evaluation(metric_payload)
            self._push_to_meta(metric_payload)

    def _push_to_evaluation(self, payload: Dict[str, Any]) -> None:
        engine = getattr(self._sys, "strategy_evaluation_engine", None)
        if engine is None:
            return
        try:
            if hasattr(engine, "ingest_perf_snapshot"):
                engine.ingest_perf_snapshot(payload)
        except Exception as exc:
            logger.debug("RollingPerfFeeder → evaluation engine error: %s", exc)

    def _push_to_meta(self, payload: Dict[str, Any]) -> None:
        engine = getattr(self._sys, "self_optimizing_meta_engine", None)
        if engine is None:
            return
        try:
            if hasattr(engine, "ingest_perf_snapshot"):
                engine.ingest_perf_snapshot(payload)
        except Exception as exc:
            logger.debug("RollingPerfFeeder → meta engine error: %s", exc)

    def summary(self) -> Dict[str, Any]:
        return {
            "buckets": len(self._buckets),
            "total_trades_tracked": sum(len(v) for v in self._buckets.values()),
            "last_push_ts": self._last_push_ts,
        }
