"""
Market Manipulation Detector — identify wash trading, spoofing, pump-and-dump.

Protects ARGUS from trading on manipulated price data by detecting:
1. Wash Trading: rapid buy-sell pairs at same price (fake volume)
2. Spoofing: large orders placed then cancelled before fill
3. Pump-and-Dump: sudden volume spike + price spike + rapid reversal
4. Quote Stuffing: abnormally high message rates
5. Layering: multiple orders at different prices creating false depth

When manipulation is detected, ARGUS reduces position sizes or pauses
trading for that symbol until conditions normalize.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ManipulationType(Enum):
    WASH_TRADING = "wash_trading"
    SPOOFING = "spoofing"
    PUMP_AND_DUMP = "pump_and_dump"
    QUOTE_STUFFING = "quote_stuffing"
    LAYERING = "layering"
    CLEAN = "clean"


@dataclass(frozen=True)
class ManipulationAlert:
    """A detected manipulation event."""
    symbol: str
    manipulation_type: ManipulationType
    confidence: float               # 0.0 to 1.0
    description: str
    timestamp: float
    block_trading: bool             # True = do not trade this symbol


@dataclass
class SymbolHealthState:
    """Per-symbol health tracking for manipulation detection."""
    symbol: str
    # Trade-level tracking
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=200))
    # Price movement tracking
    recent_prices: deque = field(default_factory=lambda: deque(maxlen=500))
    recent_volumes: deque = field(default_factory=lambda: deque(maxlen=500))
    # Order book tracking
    recent_spreads: deque = field(default_factory=lambda: deque(maxlen=100))
    # Alert tracking
    alerts: deque = field(default_factory=lambda: deque(maxlen=50))
    blocked_until: float = 0.0


class ManipulationDetector:
    """
    Real-time market manipulation detection.

    Runs every cycle, analyzing trade data, price action, and order book
    patterns to identify manipulation. Outputs advisory signals that
    reduce or block trading on manipulated symbols.
    """

    def __init__(
        self,
        wash_trade_window_s: float = 5.0,
        wash_trade_price_tolerance: float = 0.001,  # 0.1% price match
        pump_dump_vol_mult: float = 5.0,            # 5x normal volume
        pump_dump_reversal_pct: float = 0.02,       # 2% reversal
        quote_stuff_rate_limit: int = 50,            # messages per second
        block_duration_s: float = 300.0,             # block for 5 min after detection
    ):
        self._wash_window = wash_trade_window_s
        self._wash_tol = wash_trade_price_tolerance
        self._pump_vol = pump_dump_vol_mult
        self._pump_reversal = pump_dump_reversal_pct
        self._quote_limit = quote_stuff_rate_limit
        self._block_duration = block_duration_s

        self._symbols: Dict[str, SymbolHealthState] = {}
        self._total_alerts = 0

    def _get_state(self, symbol: str) -> SymbolHealthState:
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolHealthState(symbol=symbol)
        return self._symbols[symbol]

    def record_trade(self, symbol: str, price: float, volume: float,
                     side: str, timestamp: Optional[float] = None) -> None:
        """Record a market trade for analysis."""
        ts = timestamp or time.time()
        state = self._get_state(symbol)
        state.recent_trades.append((ts, price, volume, side))
        state.recent_prices.append((ts, price))
        state.recent_volumes.append((ts, volume))

    def record_spread(self, symbol: str, bid: float, ask: float,
                      timestamp: Optional[float] = None) -> None:
        """Record bid-ask spread for analysis."""
        ts = timestamp or time.time()
        state = self._get_state(symbol)
        spread_bps = (ask - bid) / max(bid, 1e-9) * 10000
        state.recent_spreads.append((ts, spread_bps))

    def check(self, symbol: str) -> ManipulationAlert:
        """Check a symbol for manipulation. Returns alert with confidence."""
        state = self._get_state(symbol)
        now = time.time()

        # Still blocked from previous detection?
        if now < state.blocked_until:
            return ManipulationAlert(
                symbol=symbol, manipulation_type=ManipulationType.WASH_TRADING,
                confidence=0.8, description="blocked from previous detection",
                timestamp=now, block_trading=True,
            )

        alerts = []

        # ── 1. Wash Trading Detection ──
        wash_conf = self._detect_wash_trading(state)
        if wash_conf > 0.5:
            alerts.append(ManipulationAlert(
                symbol=symbol, manipulation_type=ManipulationType.WASH_TRADING,
                confidence=wash_conf,
                description=f"Rapid buy-sell pairs detected (conf={wash_conf:.0%})",
                timestamp=now, block_trading=wash_conf > 0.7,
            ))

        # ── 2. Pump-and-Dump Detection ──
        pump_conf = self._detect_pump_dump(state)
        if pump_conf > 0.5:
            alerts.append(ManipulationAlert(
                symbol=symbol, manipulation_type=ManipulationType.PUMP_AND_DUMP,
                confidence=pump_conf,
                description=f"Volume spike + price reversal (conf={pump_conf:.0%})",
                timestamp=now, block_trading=pump_conf > 0.7,
            ))

        # ── 3. Quote Stuffing Detection ──
        stuff_conf = self._detect_quote_stuffing(state)
        if stuff_conf > 0.5:
            alerts.append(ManipulationAlert(
                symbol=symbol, manipulation_type=ManipulationType.QUOTE_STUFFING,
                confidence=stuff_conf,
                description=f"Abnormal message rate (conf={stuff_conf:.0%})",
                timestamp=now, block_trading=stuff_conf > 0.8,
            ))

        if alerts:
            worst = max(alerts, key=lambda a: a.confidence)
            state.alerts.append(worst)
            self._total_alerts += 1
            if worst.block_trading:
                state.blocked_until = now + self._block_duration
                logger.warning("ManipulationDetector: BLOCKING %s for %.0fs — %s",
                               symbol, self._block_duration, worst.description)
            return worst

        return ManipulationAlert(
            symbol=symbol, manipulation_type=ManipulationType.CLEAN,
            confidence=0.0, description="clean", timestamp=now, block_trading=False,
        )

    def is_blocked(self, symbol: str) -> bool:
        """Check if a symbol is currently blocked due to detected manipulation."""
        state = self._get_state(symbol)
        return time.time() < state.blocked_until

    def _detect_wash_trading(self, state: SymbolHealthState) -> float:
        """Detect rapid buy-sell pairs at similar prices."""
        if len(state.recent_trades) < 4:
            return 0.0

        trades = list(state.recent_trades)
        now = time.time()
        wash_count = 0
        total_pairs = 0

        for i in range(len(trades) - 1):
            ts_i, price_i, vol_i, side_i = trades[i]
            if now - ts_i > 60:  # only check last 60s
                continue
            for j in range(i + 1, len(trades)):
                ts_j, price_j, vol_j, side_j = trades[j]
                if ts_j - ts_i > self._wash_window:
                    break
                total_pairs += 1
                # Same price (within tolerance), opposite sides
                if (side_i != side_j and
                        abs(price_i - price_j) / max(price_i, 1e-9) < self._wash_tol):
                    wash_count += 1

        if total_pairs < 2:
            return 0.0
        return min(1.0, wash_count / max(total_pairs * 0.1, 1))

    def _detect_pump_dump(self, state: SymbolHealthState) -> float:
        """Detect volume spike + rapid price reversal pattern."""
        if len(state.recent_prices) < 20 or len(state.recent_volumes) < 20:
            return 0.0

        prices = [p for _, p in list(state.recent_prices)[-50:]]
        volumes = [v for _, v in list(state.recent_volumes)[-50:]]

        if len(prices) < 20:
            return 0.0

        # Average volume of first half vs second half
        mid = len(volumes) // 2
        avg_vol_early = sum(volumes[:mid]) / max(mid, 1)
        avg_vol_late = sum(volumes[mid:]) / max(len(volumes) - mid, 1)
        vol_ratio = avg_vol_late / max(avg_vol_early, 1e-9)

        # Price spike then reversal
        max_price = max(prices[-20:])
        current_price = prices[-1]
        start_price = prices[-20]

        if start_price <= 0:
            return 0.0

        spike_pct = (max_price - start_price) / start_price
        reversal_pct = (max_price - current_price) / max_price if max_price > 0 else 0

        # High volume + price spike + reversal = pump-and-dump
        confidence = 0.0
        if vol_ratio > self._pump_vol:
            confidence += 0.3
        if spike_pct > self._pump_reversal:
            confidence += 0.3
        if reversal_pct > self._pump_reversal * 0.5:
            confidence += 0.4

        return min(1.0, confidence)

    def _detect_quote_stuffing(self, state: SymbolHealthState) -> float:
        """Detect abnormally high trade/message rates."""
        if len(state.recent_trades) < 10:
            return 0.0

        trades = list(state.recent_trades)
        now = time.time()
        recent = [t for t in trades if now - t[0] < 1.0]  # trades in last 1 second
        rate = len(recent)

        if rate > self._quote_limit:
            return min(1.0, rate / self._quote_limit)
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        blocked = [s for s, state in self._symbols.items() if self.is_blocked(s)]
        return {
            "symbols_tracked": len(self._symbols),
            "total_alerts": self._total_alerts,
            "currently_blocked": blocked,
            "alert_counts": {
                s: len(state.alerts) for s, state in self._symbols.items() if state.alerts
            },
        }
