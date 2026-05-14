"""
Wash Trade Detection — detects and blocks same-symbol buy→sell reversals
within a configurable time window on the same venue.

Required for regulated exchange compliance. Logs all detections.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WashTradeAlert:
    """Alert when a potential wash trade is detected."""
    symbol: str
    venue: str
    side_a: str
    side_b: str
    time_between_seconds: float
    quantity_a: float
    quantity_b: float
    blocked: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class FillRecord:
    """Minimal fill record for wash trade checking."""
    symbol: str
    venue: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    timestamp: float
    strategy: str = ""


class WashTradeDetector:
    """
    Detects potential wash trades: same-symbol opposite-side trades
    within a time window on the same venue.

    A wash trade is when you buy and then immediately sell (or vice versa)
    the same asset, creating artificial volume without real economic change.

    Config:
        window_seconds: Time window to check for reversals (default 300 = 5 min)
        min_overlap_pct: Minimum quantity overlap to flag (default 0.5 = 50%)
        block_mode: If True, blocks the trade. If False, logs warning only.
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        min_overlap_pct: float = 0.50,
        block_mode: bool = True,
        max_history: int = 500,
    ):
        self._window = window_seconds
        self._min_overlap = min_overlap_pct
        self._block_mode = block_mode
        # Recent fills per (symbol, venue)
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._alerts: List[WashTradeAlert] = []
        self._total_checked = 0
        self._total_blocked = 0

    def check(self, fill: FillRecord) -> Optional[WashTradeAlert]:
        """
        Check if a fill constitutes a wash trade.

        Returns WashTradeAlert if detected, None if clean.
        """
        self._total_checked += 1
        key = f"{fill.symbol}:{fill.venue}"
        now = fill.timestamp or time.time()

        # Check against recent fills for opposite-side trades
        for prior in self._history[key]:
            age = now - prior.timestamp
            if age > self._window:
                continue
            if age < 0:
                continue

            # Same side = not a wash (just adding to position)
            if prior.side == fill.side:
                continue

            # Opposite side within window — check quantity overlap
            smaller = min(prior.quantity, fill.quantity)
            larger = max(prior.quantity, fill.quantity)
            overlap_pct = smaller / max(larger, 1e-9)

            if overlap_pct >= self._min_overlap:
                alert = WashTradeAlert(
                    symbol=fill.symbol,
                    venue=fill.venue,
                    side_a=prior.side,
                    side_b=fill.side,
                    time_between_seconds=age,
                    quantity_a=prior.quantity,
                    quantity_b=fill.quantity,
                    blocked=self._block_mode,
                    reason=f"opposite-side trade within {age:.1f}s, {overlap_pct:.0%} qty overlap",
                )
                self._alerts.append(alert)
                if self._block_mode:
                    self._total_blocked += 1
                    logger.warning(
                        "WASH TRADE BLOCKED: %s %s→%s on %s within %.1fs (qty overlap %.0f%%)",
                        fill.symbol, prior.side, fill.side, fill.venue, age, overlap_pct * 100,
                    )
                else:
                    logger.warning(
                        "WASH TRADE WARNING: %s %s→%s on %s within %.1fs (qty overlap %.0f%%)",
                        fill.symbol, prior.side, fill.side, fill.venue, age, overlap_pct * 100,
                    )
                return alert

        # No wash trade — record this fill
        self._history[key].append(fill)
        return None

    def get_alerts(self, last_n: int = 50) -> List[WashTradeAlert]:
        """Get recent wash trade alerts."""
        return self._alerts[-last_n:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_checked": self._total_checked,
            "total_blocked": self._total_blocked,
            "total_alerts": len(self._alerts),
            "window_seconds": self._window,
            "block_mode": self._block_mode,
        }
