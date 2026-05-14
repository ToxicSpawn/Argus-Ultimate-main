#!/usr/bin/env python3
"""
Rolling Performance Feeder — pushes per-strategy rolling PnL windows into
SelfOptimizingMetaEngine so weights evolve in near-real-time.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RollingPerformanceFeeder:
    """
    Accumulates closed-trade PnL per strategy and pushes rolling windows
    (default last 50 trades) into the meta engine on each feed() call.
    """

    def __init__(
        self,
        meta_engine: Any,
        *,
        window: int = 50,
        push_every_n: int = 5,
        regime_label: str = "",
    ):
        self.meta_engine = meta_engine
        self.window = max(5, int(window))
        self.push_every_n = max(1, int(push_every_n))
        self.regime_label = str(regime_label or "")
        self._pnl_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._feed_count = 0
        self._last_push_ts = 0.0

    def record_trade(self, *, strategy: str, pnl: float, symbol: str = "", regime: str = "") -> None:
        """Call whenever a trade closes with its realized PnL."""
        key = str(strategy or "unknown")
        self._pnl_windows[key].append(float(pnl))
        self._feed_count += 1
        if self._feed_count % self.push_every_n == 0:
            self._push()

    def feed(self, trades: List[Dict[str, Any]]) -> None:
        """Batch-feed a list of closed trades (dicts with strategy + pnl keys)."""
        for t in list(trades or []):
            strategy = str(t.get("strategy") or t.get("source_strategy") or "unknown")
            pnl = float(t.get("pnl") or t.get("realized_pnl") or 0.0)
            self._pnl_windows[strategy].append(pnl)
        self._feed_count += len(trades or [])
        if trades:
            self._push()

    def _push(self) -> None:
        """Push all non-empty rolling windows to the meta engine."""
        if self.meta_engine is None:
            return
        now = time.time()
        self._last_push_ts = now
        for strategy, window in list(self._pnl_windows.items()):
            trades = list(window)
            if not trades:
                continue
            try:
                if hasattr(self.meta_engine, "update_from_trades"):
                    self.meta_engine.update_from_trades(
                        strategy_name=strategy,
                        trades=trades,
                        regime_label=self.regime_label or None,
                    )
                elif hasattr(self.meta_engine, "record_trade_outcome"):
                    for pnl in trades:
                        self.meta_engine.record_trade_outcome(
                            strategy_name=strategy,
                            pnl=pnl,
                            regime_label=self.regime_label or None,
                        )
            except Exception as e:
                logger.debug("RollingPerformanceFeeder push %s: %s", strategy, e)

    def update_regime(self, regime_label: str) -> None:
        """Update current regime label used when pushing to meta engine."""
        self.regime_label = str(regime_label or "")

    def summary(self) -> Dict[str, Any]:
        return {
            "strategies": len(self._pnl_windows),
            "feed_count": self._feed_count,
            "last_push_ts": self._last_push_ts,
            "window": self.window,
        }
