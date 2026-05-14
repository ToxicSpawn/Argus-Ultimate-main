"""
Trader profiler — classifies market participants by archetype.

Observed features per order/fill:
  - Order size (relative to typical book depth)
  - Lifetime (cancel rate / time-to-fill distribution)
  - Spread placement (inside, at, or wide of best)
  - Frequency (orders per minute)
  - Direction persistence

Archetypes:
  - MARKET_MAKER: high frequency, tight spreads, high cancel rate, balanced direction
  - WHALE: large size, held long, wide spreads, strong direction
  - RETAIL: small size, poor spread placement, sparse frequency
  - HFT: extreme frequency, nano-second lifetime, small size
  - ARBITRAGEUR: fast execution, cross-venue timing, moderate size

The profiler maintains rolling counters per archetype and exposes a
"caution factor" to downstream risk modules — e.g. if whale-archetype
activity spikes on the same side as ARGUS's planned trade, reduce size.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Archetypes
# ═════════════════════════════════════════════════════════════════════════════


class TraderArchetype(Enum):
    MARKET_MAKER = "market_maker"
    WHALE = "whale"
    RETAIL = "retail"
    HFT = "hft"
    ARBITRAGEUR = "arbitrageur"
    UNKNOWN = "unknown"


# ═════════════════════════════════════════════════════════════════════════════
# Feature dataclass
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class OrderFootprint:
    """Microstructure features for a single order or fill."""

    timestamp: float
    side: str  # "buy" | "sell"
    size_relative: float  # size / book_depth_at_best
    lifetime_ms: float  # time from submit to fill or cancel (ms)
    spread_placement_bps: float  # distance from best (bps)
    cancelled: bool = False
    venue: str = ""

    def classify(self) -> TraderArchetype:
        """Heuristic classification of a single order."""
        # HFT: very fast (<100ms) and small
        if self.lifetime_ms < 100 and self.size_relative < 0.1:
            return TraderArchetype.HFT
        # Whale: large size
        if self.size_relative > 1.0:
            return TraderArchetype.WHALE
        # Market maker: tight spread, high cancel rate
        if self.cancelled and self.spread_placement_bps < 2.0:
            return TraderArchetype.MARKET_MAKER
        # Arbitrageur: cross-venue heuristic (requires venue info)
        if self.lifetime_ms < 500 and 0.1 <= self.size_relative <= 0.5:
            return TraderArchetype.ARBITRAGEUR
        # Retail: small size, loose placement
        if self.size_relative < 0.1 and self.spread_placement_bps > 5.0:
            return TraderArchetype.RETAIL
        return TraderArchetype.UNKNOWN


# ═════════════════════════════════════════════════════════════════════════════
# Snapshot
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ProfilerSnapshot:
    timestamp: float
    total_observed: int
    archetype_counts: Dict[str, int]
    archetype_share: Dict[str, float]
    dominant_archetype: str
    recent_whale_direction: str  # "buy" | "sell" | "balanced"
    caution_factor: float  # 0 = no caution, 1 = max caution


# ═════════════════════════════════════════════════════════════════════════════
# TraderProfiler
# ═════════════════════════════════════════════════════════════════════════════


class TraderProfiler:
    """
    Rolling profiler of market participant archetypes.

    Parameters
    ----------
    window : int, default 200
        Rolling window size for archetype counters.
    whale_window : int, default 50
        Window for whale direction tracking.
    """

    def __init__(
        self,
        window: int = 200,
        whale_window: int = 50,
    ) -> None:
        self.window = int(window)
        self.whale_window = int(whale_window)
        self._footprints: Deque[OrderFootprint] = deque(maxlen=self.window)
        self._archetypes: Deque[TraderArchetype] = deque(maxlen=self.window)
        self._whale_sides: Deque[str] = deque(maxlen=self.whale_window)

    # ── Record an observation ────────────────────────────────────────────────

    def record_order(self, footprint: OrderFootprint) -> TraderArchetype:
        """Record a single observed order footprint and return its archetype."""
        archetype = footprint.classify()
        self._footprints.append(footprint)
        self._archetypes.append(archetype)
        if archetype == TraderArchetype.WHALE:
            self._whale_sides.append(footprint.side)
        return archetype

    def record_order_dict(self, order: Dict[str, Any]) -> TraderArchetype:
        """Convenience: accept a dict and convert to OrderFootprint."""
        try:
            fp = OrderFootprint(
                timestamp=float(order.get("timestamp", time.time())),
                side=str(order.get("side", "buy")).lower(),
                size_relative=float(order.get("size_relative", 0.1)),
                lifetime_ms=float(order.get("lifetime_ms", 1000.0)),
                spread_placement_bps=float(order.get("spread_placement_bps", 5.0)),
                cancelled=bool(order.get("cancelled", False)),
                venue=str(order.get("venue", "")),
            )
            return self.record_order(fp)
        except (TypeError, ValueError) as exc:
            logger.debug("record_order_dict failed: %s", exc)
            return TraderArchetype.UNKNOWN

    # ── Read state ───────────────────────────────────────────────────────────

    def archetype_counts(self) -> Dict[str, int]:
        """Count occurrences of each archetype in the rolling window."""
        counts: Dict[str, int] = defaultdict(int)
        for a in self._archetypes:
            counts[a.value] += 1
        return dict(counts)

    def dominant_archetype(self) -> str:
        counts = self.archetype_counts()
        if not counts:
            return TraderArchetype.UNKNOWN.value
        return max(counts, key=counts.get)

    def recent_whale_direction(self) -> str:
        """Return 'buy', 'sell', or 'balanced' based on recent whale orders."""
        if not self._whale_sides:
            return "balanced"
        buy_count = sum(1 for s in self._whale_sides if s == "buy")
        sell_count = sum(1 for s in self._whale_sides if s == "sell")
        if buy_count > 1.5 * sell_count:
            return "buy"
        if sell_count > 1.5 * buy_count:
            return "sell"
        return "balanced"

    def caution_factor(self, intended_side: str) -> float:
        """
        Compute a caution factor in [0, 1].

        Returns 0 when no whale activity; up to 1 when whales dominate
        the same side as ``intended_side``.
        """
        counts = self.archetype_counts()
        n = sum(counts.values())
        if n == 0:
            return 0.0

        whale_share = counts.get(TraderArchetype.WHALE.value, 0) / n
        whale_dir = self.recent_whale_direction()

        # Map intended side to canonical
        intended = str(intended_side).lower()
        if intended in ("long", "buy"):
            intended = "buy"
        elif intended in ("short", "sell"):
            intended = "sell"

        # Whales on same side = high caution (they might dump on us)
        if whale_dir == intended and whale_share > 0.1:
            return min(1.0, whale_share * 2.5)
        # Whales on opposite side = moderate caution (they might pin price)
        if whale_dir != "balanced" and whale_dir != intended and whale_share > 0.1:
            return min(0.3, whale_share)
        return 0.0

    def snapshot(self) -> ProfilerSnapshot:
        counts = self.archetype_counts()
        total = sum(counts.values())
        share = {k: v / max(total, 1) for k, v in counts.items()}
        return ProfilerSnapshot(
            timestamp=time.time(),
            total_observed=total,
            archetype_counts=counts,
            archetype_share=share,
            dominant_archetype=self.dominant_archetype(),
            recent_whale_direction=self.recent_whale_direction(),
            caution_factor=0.0,  # computed per-side on demand
        )
