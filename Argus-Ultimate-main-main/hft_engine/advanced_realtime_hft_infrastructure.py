"""Advanced real-time HFT infrastructure — migrated from hft/ into canonical hft_engine/.

Original source: hft/advanced_realtime_hft_infrastructure.py
Migrated: 2026-04-15
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class MarketMicrostructure:
    """Snapshot of current order-book microstructure."""
    symbol: str
    bid: float
    ask: float
    mid: float
    spread: float
    spread_bps: float
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


class AdvancedRealtimeHFTInfrastructure:
    """
    Real-time HFT infrastructure layer.

    Responsibilities:
    - Sub-millisecond order-book tick processing
    - Spread / microstructure monitoring
    - Latency budget tracking per cycle
    - Hot-path signal dispatch to HFTScalpingEngine
    """

    MAX_TICK_HISTORY = 10_000
    LATENCY_BUDGET_MS = 5.0

    def __init__(
        self,
        symbol: str,
        scalping_engine: Optional[Any] = None,
        latency_budget_ms: float = 5.0,
    ) -> None:
        self.symbol = str(symbol)
        self.scalping_engine = scalping_engine
        self.latency_budget_ms = float(latency_budget_ms)
        self._tick_history: Deque[MarketMicrostructure] = deque(maxlen=self.MAX_TICK_HISTORY)
        self._cycle_latencies_ms: Deque[float] = deque(maxlen=1_000)
        self._total_ticks = 0
        self._budget_breaches = 0
        logger.info(
            "AdvancedRealtimeHFTInfrastructure ready | symbol=%s | budget=%.1fms",
            self.symbol, self.latency_budget_ms,
        )

    def process_tick(self, bid: float, ask: float) -> MarketMicrostructure:
        """Process a single order-book tick. Returns a MarketMicrostructure snapshot."""
        t0_ns = time.time_ns()

        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = (spread / mid * 10_000.0) if mid > 0 else 0.0

        snap = MarketMicrostructure(
            symbol=self.symbol,
            bid=float(bid),
            ask=float(ask),
            mid=float(mid),
            spread=float(spread),
            spread_bps=float(spread_bps),
            timestamp_ns=t0_ns,
        )
        self._tick_history.append(snap)
        self._total_ticks += 1

        # Latency tracking
        elapsed_ms = (time.time_ns() - t0_ns) / 1_000_000.0
        self._cycle_latencies_ms.append(elapsed_ms)
        if elapsed_ms > self.latency_budget_ms:
            self._budget_breaches += 1
            logger.warning(
                "HFT latency budget breach: %.3fms > %.1fms (breach #%d)",
                elapsed_ms, self.latency_budget_ms, self._budget_breaches,
            )

        return snap

    async def process_tick_async(self, bid: float, ask: float) -> MarketMicrostructure:
        """Async wrapper for process_tick — dispatches to scalping engine if attached."""
        snap = self.process_tick(bid, ask)
        if self.scalping_engine is not None:
            try:
                await self.scalping_engine.on_tick(snap)
            except Exception as exc:
                logger.error("HFT scalping engine tick error: %s", exc)
        return snap

    def get_avg_spread_bps(self, n: int = 100) -> float:
        """Average spread in BPS over last n ticks."""
        recent = list(self._tick_history)[-n:]
        if not recent:
            return 0.0
        return sum(s.spread_bps for s in recent) / len(recent)

    def get_p99_latency_ms(self) -> float:
        """P99 tick-processing latency in milliseconds."""
        lats = sorted(self._cycle_latencies_ms)
        if not lats:
            return 0.0
        idx = max(0, int(len(lats) * 0.99) - 1)
        return lats[idx]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_ticks": self._total_ticks,
            "budget_breaches": self._budget_breaches,
            "breach_rate_pct": (self._budget_breaches / max(self._total_ticks, 1)) * 100.0,
            "avg_spread_bps": self.get_avg_spread_bps(),
            "p99_latency_ms": self.get_p99_latency_ms(),
        }


# ---------------------------------------------------------------------------
# Compatibility function (replaces hft/ stub)
# ---------------------------------------------------------------------------

def get_hft_infrastructure(config: Any = None, *, hft_engine: Any = None) -> AdvancedRealtimeHFTInfrastructure:
    """Return HFT infrastructure instance.

    This replaces the stub in hft/advanced_realtime_hft_infrastructure.py
    with the real implementation.
    """
    cfg = config if isinstance(config, dict) else (getattr(config, "__dict__", {}) or {})
    return AdvancedRealtimeHFTInfrastructure(
        symbol=cfg.get("symbol", "BTC/USDT"),
        tick_budget_ms=cfg.get("tick_budget_ms", 0.5),
    )
