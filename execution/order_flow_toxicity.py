"""
Order Flow Toxicity Detector — measures adverse selection in fills.

VPIN (Volume-synchronised Probability of Informed Trading) adapted for crypto.
High toxicity indicates informed traders are on the other side of your trades,
leading to poor fills and adverse price movement post-execution.

Additionally tracks:
  - Fill rate degradation (ratio of filled quantity to expected depth)
  - Spread widening at fill time relative to baseline
  - Size vs market depth

Signal outputs:
  toxicity_score  0.0–1.0    (0 = clean flow, 1 = highly toxic)
  adverse_selection_bps      estimated cost in bps from informed flow
  recommendation             REDUCE_SIZE | WIDEN_LIMIT | PAUSE | OK

VPIN algorithm (simplified):
  1. Accumulate fills into volume buckets of ``bucket_size`` total volume.
  2. For each complete bucket compute:
         imbalance = buy_volume / total_volume
  3. VPIN = rolling mean of |imbalance − 0.5| × 2  over the last ``window``
     complete buckets.  Range [0, 1]; 0.5 = perfectly balanced flow.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOXICITY_THRESHOLD_PAUSE: float = 0.80
_TOXICITY_THRESHOLD_WIDEN: float = 0.60
_TOXICITY_THRESHOLD_REDUCE: float = 0.40
_BASELINE_SPREAD_BPS: float = 5.0        # assumed baseline when no history yet


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FillContext:
    """Context captured at the moment a fill is received."""

    symbol: str
    side: str               # "buy" or "sell"
    quantity: float         # traded quantity in base asset units
    price: float            # fill price
    spread_bps: float       # best ask − best bid at fill time, in bps
    depth_at_price: float   # available quantity at the fill price level
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToxicityReading:
    """Current toxicity assessment for a single symbol."""

    symbol: str
    toxicity_score: float       # 0.0–1.0
    adverse_selection_bps: float
    recommendation: str         # "OK" | "REDUCE_SIZE" | "WIDEN_LIMIT" | "PAUSE"
    vpin: float                 # raw VPIN value
    fill_rate: float            # ratio of fill qty to depth_at_price (1.0 = 100 %)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Internal per-symbol state
# ---------------------------------------------------------------------------

@dataclass
class _Bucket:
    """One volume-synchronised bucket."""
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    @property
    def total(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def imbalance(self) -> float:
        """Fraction of buy volume; 0.5 = balanced."""
        tot = self.total
        return self.buy_volume / tot if tot > 0 else 0.5


@dataclass
class _SymbolState:
    """Rolling state maintained per symbol."""

    window: int
    bucket_size: float

    # Completed buckets used for VPIN calculation
    completed_buckets: Deque[_Bucket] = field(default_factory=deque)
    # Partially-filled current bucket
    current_bucket: _Bucket = field(default_factory=_Bucket)

    # Raw fills for adverse selection computation
    fills: Deque[FillContext] = field(default_factory=deque)
    max_fills: int = 500

    def add_fill(self, ctx: FillContext) -> None:
        """Ingest a fill into the current bucket, advancing when full."""
        qty = ctx.quantity
        if ctx.side.lower() == "buy":
            self.current_bucket.buy_volume += qty
        else:
            self.current_bucket.sell_volume += qty

        # Advance bucket if full
        while self.current_bucket.total >= self.bucket_size:
            self.completed_buckets.append(self.current_bucket)
            if len(self.completed_buckets) > self.window:
                self.completed_buckets.popleft()
            self.current_bucket = _Bucket()

        # Store raw fill
        self.fills.append(ctx)
        while len(self.fills) > self.max_fills:
            self.fills.popleft()


# ---------------------------------------------------------------------------
# OrderFlowToxicity
# ---------------------------------------------------------------------------

class OrderFlowToxicity:
    """
    Tracks adverse selection and order flow toxicity per trading symbol.

    Call ``record_fill()`` for every completed fill.
    Call ``get_toxicity()`` to obtain the current assessment.
    """

    def __init__(self, window: int = 50, bucket_size: float = 100.0) -> None:
        """
        Parameters
        ----------
        window:
            Number of completed volume buckets to include in the VPIN rolling
            average.  Larger windows are more stable but slower to react.
        bucket_size:
            Volume (base-asset units) to accumulate before closing a bucket.
            For BTC pairs on Kraken, 0.5–2.0 BTC is typical.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be > 0, got {bucket_size}")

        self._window = window
        self._bucket_size = bucket_size
        self._states: Dict[str, _SymbolState] = {}
        logger.info(
            "OrderFlowToxicity initialised: window=%d bucket_size=%.2f",
            window, bucket_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_fill(self, ctx: FillContext) -> None:
        """
        Ingest a completed fill event.

        Must be called for every fill on every symbol this detector tracks.
        """
        state = self._get_state(ctx.symbol)
        state.add_fill(ctx)
        logger.debug(
            "OrderFlowToxicity: fill recorded symbol=%s side=%s qty=%.4f price=%.4f",
            ctx.symbol, ctx.side, ctx.quantity, ctx.price,
        )

    def get_toxicity(self, symbol: str) -> ToxicityReading:
        """
        Return the current toxicity reading for ``symbol``.

        Returns an "OK" reading with zero scores if no fills have been recorded.
        """
        state = self._states.get(symbol)
        if state is None or not state.fills:
            return ToxicityReading(
                symbol=symbol,
                toxicity_score=0.0,
                adverse_selection_bps=0.0,
                recommendation="OK",
                vpin=0.0,
                fill_rate=1.0,
            )

        vpin = self._compute_vpin(symbol)
        adv_sel_bps = self._compute_adverse_selection(symbol)
        fill_rate = self._compute_fill_rate(symbol)

        # Composite toxicity score: weighted combination
        # VPIN already in [0,1]; we normalise adverse selection by a 10-bps cap
        adv_sel_norm = min(1.0, adv_sel_bps / 10.0)
        fill_rate_penalty = max(0.0, 1.0 - fill_rate)   # high fill-rate = low penalty
        toxicity = 0.5 * vpin + 0.35 * adv_sel_norm + 0.15 * fill_rate_penalty
        toxicity = min(1.0, max(0.0, toxicity))

        recommendation = self._recommend(toxicity)

        return ToxicityReading(
            symbol=symbol,
            toxicity_score=toxicity,
            adverse_selection_bps=adv_sel_bps,
            recommendation=recommendation,
            vpin=vpin,
            fill_rate=fill_rate,
        )

    def snapshot(self) -> Dict[str, ToxicityReading]:
        """Return current toxicity readings for all tracked symbols."""
        return {sym: self.get_toxicity(sym) for sym in self._states}

    # ------------------------------------------------------------------
    # Private computation methods
    # ------------------------------------------------------------------

    def _compute_vpin(self, symbol: str) -> float:
        """
        Compute VPIN from completed buckets.

        VPIN = mean of |imbalance − 0.5| × 2 over the rolling window.
        Returns 0.0 when fewer than 2 complete buckets are available.
        """
        state = self._states.get(symbol)
        if state is None or len(state.completed_buckets) < 2:
            return 0.0

        buckets = list(state.completed_buckets)
        scores = [abs(b.imbalance - 0.5) * 2.0 for b in buckets]
        return sum(scores) / len(scores)

    def _compute_adverse_selection(self, symbol: str) -> float:
        """
        Estimate adverse selection cost in bps.

        Methodology: compare spread_bps at fill time versus the rolling median
        spread.  Widening spread at fill time is a proxy for informed flow.
        Additionally penalise fills where depth_at_price is smaller than the
        fill quantity (size > available liquidity).
        """
        state = self._states.get(symbol)
        if state is None or not state.fills:
            return 0.0

        fills = list(state.fills)
        spreads = [f.spread_bps for f in fills]

        # Robust baseline: median spread
        spreads_sorted = sorted(spreads)
        n = len(spreads_sorted)
        if n % 2 == 0:
            median_spread = (spreads_sorted[n // 2 - 1] + spreads_sorted[n // 2]) / 2.0
        else:
            median_spread = spreads_sorted[n // 2]

        baseline = max(median_spread, _BASELINE_SPREAD_BPS)

        # Most recent fill's spread vs baseline
        recent = fills[-1]
        spread_penalty = max(0.0, recent.spread_bps - baseline)

        # Depth penalty: how much quantity exceeded available depth
        depth_ratio = recent.quantity / max(recent.depth_at_price, 1e-10)
        depth_penalty = max(0.0, (depth_ratio - 1.0)) * baseline

        return spread_penalty + depth_penalty

    def _compute_fill_rate(self, symbol: str) -> float:
        """
        Average ratio of fill_quantity / depth_at_price over recent fills.

        Values > 1.0 indicate fills that consumed more than the visible depth
        (possible hidden liquidity absorption).  We cap at 1.0 for the metric.
        """
        state = self._states.get(symbol)
        if state is None or not state.fills:
            return 1.0

        fills = list(state.fills)[-20:]  # most recent 20 fills
        rates: List[float] = []
        for f in fills:
            if f.depth_at_price > 0:
                rates.append(min(1.0, f.quantity / f.depth_at_price))

        return sum(rates) / len(rates) if rates else 1.0

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend(toxicity: float) -> str:
        """Map toxicity score to a trading recommendation."""
        if toxicity >= _TOXICITY_THRESHOLD_PAUSE:
            return "PAUSE"
        if toxicity >= _TOXICITY_THRESHOLD_WIDEN:
            return "WIDEN_LIMIT"
        if toxicity >= _TOXICITY_THRESHOLD_REDUCE:
            return "REDUCE_SIZE"
        return "OK"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self, symbol: str) -> _SymbolState:
        """Return or create per-symbol state."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(
                window=self._window,
                bucket_size=self._bucket_size,
            )
        return self._states[symbol]
