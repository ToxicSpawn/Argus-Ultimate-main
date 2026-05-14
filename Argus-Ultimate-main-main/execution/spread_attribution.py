"""
execution/spread_attribution.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-fill PnL decomposition into spread capture vs adverse selection.

Terminology
-----------
spread_captured_bps  : positive means we earned the spread (bought below mid or sold above mid).
adverse_selection_bps: positive means the market moved against us after the fill.
net_pnl_bps          : spread_captured - adverse_selection  (positive = net winner).
is_toxic             : adverse_selection_bps_5s > spread_captured_bps
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FillAttribution:
    """Decomposed per-fill PnL attribution for a single market-making fill."""

    fill_id: str
    symbol: str
    side: str  # "buy" or "sell"

    fill_price: float
    fill_size: float
    quote_mid_at_fill: float
    timestamp_ns: int

    # Spread capture: positive = earned the spread
    spread_captured_bps: float = 0.0

    # Adverse selection: positive = market moved against us
    adverse_selection_bps_500ms: float = 0.0
    adverse_selection_bps_5s: float = 0.0

    # Net: spread earned minus adverse drift
    net_pnl_bps: float = 0.0

    # True when the adverse drift exceeds what we captured
    is_toxic: bool = False

    # Populated as post-fill mid observations arrive
    post_mid_500ms: Optional[float] = None
    post_mid_5s: Optional[float] = None

    # Resolved flag: True once 5 s observation has arrived
    resolved: bool = False


@dataclass
class _SymbolStats:
    """Running aggregates for one symbol."""

    fill_count: int = 0
    toxic_fill_count: int = 0
    total_spread_pnl_bps: float = 0.0
    total_adverse_selection_bps: float = 0.0

    def adverse_selection_ratio(self) -> float:
        denom = self.total_spread_pnl_bps + self.total_adverse_selection_bps
        if denom <= 0.0:
            return 0.0
        return self.total_adverse_selection_bps / denom

    def net_mm_pnl_bps(self) -> float:
        return self.total_spread_pnl_bps - self.total_adverse_selection_bps

    def avg_spread_captured_bps(self) -> float:
        if self.fill_count == 0:
            return 0.0
        return self.total_spread_pnl_bps / self.fill_count


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class SpreadAttributionEngine:
    """
    Records market-making fills and decomposes PnL into:
      - Spread capture  (immediate edge from posting liquidity)
      - Adverse selection (mid-price drift working against our position post-fill)

    Thread-safe via a single re-entrant lock.

    Parameters
    ----------
    attribution_window_ms : float
        Maximum age (ms) after which a fill is considered stale and will not
        receive further post-fill observations. Default 5 000 ms.
    should_widen_threshold : float
        Adverse-selection ratio above which `should_widen_spread` returns True.
        Default 0.6.
    """

    WIDEN_THRESHOLD_DEFAULT: float = 0.6
    POST_DELAY_TOLERANCE_MS: float = 100.0  # slack when matching delay buckets

    def __init__(
        self,
        attribution_window_ms: float = 5000.0,
        should_widen_threshold: float = WIDEN_THRESHOLD_DEFAULT,
    ) -> None:
        self._attribution_window_ms = attribution_window_ms
        self._widen_threshold = should_widen_threshold

        self._fills: Dict[str, FillAttribution] = {}
        self._symbol_stats: Dict[str, _SymbolStats] = defaultdict(_SymbolStats)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_fill(
        self,
        fill_id: str,
        symbol: str,
        side: str,
        fill_price: float,
        fill_size: float,
        quote_mid_at_fill: float,
        timestamp_ns: int,
    ) -> None:
        """Register a new fill.

        The spread capture is computed immediately using the mid price at the
        time of the fill:
          - For a buy: we captured spread when fill_price < mid  (we paid less)
          - For a sell: we captured spread when fill_price > mid  (we received more)

        spread_captured_bps = |fill_price - mid| / mid × 10 000

        The sign convention keeps this value positive when we genuinely captured
        the spread and negative when we were adversely filled (e.g. crossed spread
        at market).
        """
        if not symbol or not side or fill_size <= 0:
            raise ValueError("symbol, side, and fill_size must be non-empty/positive")
        if quote_mid_at_fill <= 0:
            raise ValueError("quote_mid_at_fill must be positive")

        side_l = side.lower()
        if side_l not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        # Spread capture in bps
        if side_l == "buy":
            # Positive when we bought below mid
            spread_bps = (quote_mid_at_fill - fill_price) / quote_mid_at_fill * 10_000.0
        else:
            # Positive when we sold above mid
            spread_bps = (fill_price - quote_mid_at_fill) / quote_mid_at_fill * 10_000.0

        attr = FillAttribution(
            fill_id=fill_id,
            symbol=symbol,
            side=side_l,
            fill_price=fill_price,
            fill_size=fill_size,
            quote_mid_at_fill=quote_mid_at_fill,
            timestamp_ns=timestamp_ns,
            spread_captured_bps=spread_bps,
        )

        with self._lock:
            self._fills[fill_id] = attr

    def record_post_fill_mid(
        self,
        fill_id: str,
        post_mid: float,
        delay_ms: float,
    ) -> None:
        """Record the mid-price observed *delay_ms* after the fill.

        Adverse selection is the mid-price drift that works against our position:
          - For a buy fill: mid falling is bad → adverse = (mid_at_fill - post_mid) / mid_at_fill
          - For a sell fill: mid rising is bad → adverse = (post_mid - mid_at_fill) / mid_at_fill

        A positive adverse_selection_bps means the market moved against us.

        Calling convention: call once at ~500 ms and once at ~5 000 ms.  The
        bucket is chosen by proximity to these two target delays.
        """
        if post_mid <= 0:
            raise ValueError("post_mid must be positive")

        with self._lock:
            attr = self._fills.get(fill_id)
            if attr is None:
                return  # unknown fill; ignore
            if attr.resolved:
                return  # already finalised

            mid0 = attr.quote_mid_at_fill
            side = attr.side

            # Adverse selection bps: positive = bad for us
            if side == "buy":
                adv_bps = (mid0 - post_mid) / mid0 * 10_000.0
            else:
                adv_bps = (post_mid - mid0) / mid0 * 10_000.0

            # Bucket assignment by proximity
            tol = self.POST_DELAY_TOLERANCE_MS
            is_500ms = abs(delay_ms - 500.0) < tol or delay_ms < 1000.0
            is_5s = abs(delay_ms - 5000.0) < tol or delay_ms >= 1000.0

            if is_500ms and attr.post_mid_500ms is None:
                attr.post_mid_500ms = post_mid
                attr.adverse_selection_bps_500ms = max(adv_bps, 0.0)  # clip at 0

            if is_5s and delay_ms >= 1000.0:
                attr.post_mid_5s = post_mid
                attr.adverse_selection_bps_5s = max(adv_bps, 0.0)
                self._finalise(attr)

    def _finalise(self, attr: FillAttribution) -> None:
        """Compute net PnL, toxicity flag, and update symbol aggregates.

        Must be called with self._lock held.
        """
        attr.net_pnl_bps = attr.spread_captured_bps - attr.adverse_selection_bps_5s
        attr.is_toxic = attr.adverse_selection_bps_5s > attr.spread_captured_bps
        attr.resolved = True

        stats = self._symbol_stats[attr.symbol]
        stats.fill_count += 1
        if attr.is_toxic:
            stats.toxic_fill_count += 1
        # Only accumulate positive spread capture
        stats.total_spread_pnl_bps += max(attr.spread_captured_bps, 0.0)
        stats.total_adverse_selection_bps += attr.adverse_selection_bps_5s

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_attribution(self, fill_id: str) -> Optional[FillAttribution]:
        """Return the FillAttribution for fill_id, or None if not found."""
        with self._lock:
            return self._fills.get(fill_id)

    def get_symbol_stats(self, symbol: str) -> dict:
        """Return aggregated attribution statistics for *symbol*.

        Returns a dict with:
          - total_spread_pnl        : sum of spread_captured_bps across resolved fills
          - total_adverse_selection : cumulative adverse movement post-fill (bps)
          - net_mm_pnl              : spread_pnl - adverse_selection
          - avg_spread_captured_bps : mean spread capture per fill
          - adverse_selection_ratio : adverse / (spread + adverse)
          - fill_count
          - toxic_fill_count        : fills where adverse > spread capture
        """
        with self._lock:
            stats = self._symbol_stats.get(symbol)
            if stats is None:
                return {
                    "symbol": symbol,
                    "total_spread_pnl": 0.0,
                    "total_adverse_selection": 0.0,
                    "net_mm_pnl": 0.0,
                    "avg_spread_captured_bps": 0.0,
                    "adverse_selection_ratio": 0.0,
                    "fill_count": 0,
                    "toxic_fill_count": 0,
                }
            return {
                "symbol": symbol,
                "total_spread_pnl": stats.total_spread_pnl_bps,
                "total_adverse_selection": stats.total_adverse_selection_bps,
                "net_mm_pnl": stats.net_mm_pnl_bps(),
                "avg_spread_captured_bps": stats.avg_spread_captured_bps(),
                "adverse_selection_ratio": stats.adverse_selection_ratio(),
                "fill_count": stats.fill_count,
                "toxic_fill_count": stats.toxic_fill_count,
            }

    def get_worst_symbols(self) -> List[Tuple[str, float]]:
        """Return symbols sorted by adverse_selection_ratio descending.

        Returns a list of (symbol, ratio) tuples.
        """
        with self._lock:
            ranked = [
                (sym, stats.adverse_selection_ratio())
                for sym, stats in self._symbol_stats.items()
            ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def should_widen_spread(self, symbol: str) -> bool:
        """Return True if the symbol's adverse selection ratio exceeds the threshold.

        A ratio > 0.6 (default) means more than 60 % of the combined
        spread-plus-adverse is adverse, indicating systematic toxic flow that
        warrants wider quoted spreads.
        """
        with self._lock:
            stats = self._symbol_stats.get(symbol)
            if stats is None:
                return False
            return stats.adverse_selection_ratio() > self._widen_threshold

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def get_all_fills(self, symbol: Optional[str] = None) -> List[FillAttribution]:
        """Return all stored fills, optionally filtered by symbol."""
        with self._lock:
            fills = list(self._fills.values())
        if symbol is not None:
            fills = [f for f in fills if f.symbol == symbol]
        return fills

    def get_resolved_fills(self, symbol: Optional[str] = None) -> List[FillAttribution]:
        """Return fills for which the 5 s post-fill observation has arrived."""
        return [f for f in self.get_all_fills(symbol) if f.resolved]

    def get_pending_fills(self) -> List[FillAttribution]:
        """Return fills still awaiting the 5 s post-fill observation."""
        with self._lock:
            return [f for f in self._fills.values() if not f.resolved]

    def flush_stale(self, now_ns: int) -> int:
        """Remove fills older than attribution_window_ms that are still unresolved.

        Returns the count of flushed entries.
        """
        cutoff_ns = now_ns - int(self._attribution_window_ms * 1_000_000)
        with self._lock:
            stale = [
                fid
                for fid, attr in self._fills.items()
                if not attr.resolved and attr.timestamp_ns < cutoff_ns
            ]
            for fid in stale:
                del self._fills[fid]
        return len(stale)

    def clear(self) -> None:
        """Reset all state (useful between sessions)."""
        with self._lock:
            self._fills.clear()
            self._symbol_stats.clear()

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def get_global_stats(self) -> dict:
        """Aggregate statistics across all symbols."""
        with self._lock:
            all_stats = list(self._symbol_stats.values())

        if not all_stats:
            return {
                "total_fills": 0,
                "total_toxic_fills": 0,
                "total_spread_pnl_bps": 0.0,
                "total_adverse_selection_bps": 0.0,
                "net_mm_pnl_bps": 0.0,
                "global_adverse_selection_ratio": 0.0,
                "symbol_count": 0,
            }

        total_fills = sum(s.fill_count for s in all_stats)
        total_toxic = sum(s.toxic_fill_count for s in all_stats)
        total_spread = sum(s.total_spread_pnl_bps for s in all_stats)
        total_adv = sum(s.total_adverse_selection_bps for s in all_stats)
        denom = total_spread + total_adv
        ratio = total_adv / denom if denom > 0 else 0.0

        return {
            "total_fills": total_fills,
            "total_toxic_fills": total_toxic,
            "total_spread_pnl_bps": total_spread,
            "total_adverse_selection_bps": total_adv,
            "net_mm_pnl_bps": total_spread - total_adv,
            "global_adverse_selection_ratio": ratio,
            "symbol_count": len(all_stats),
        }

    def print_summary(self) -> None:  # pragma: no cover
        """Pretty-print a summary table to stdout."""
        g = self.get_global_stats()
        print(
            f"\n{'='*60}\n"
            f"SpreadAttributionEngine — Global Summary\n"
            f"{'='*60}\n"
            f"  Symbols tracked   : {g['symbol_count']}\n"
            f"  Total fills       : {g['total_fills']}\n"
            f"  Toxic fills       : {g['total_toxic_fills']}\n"
            f"  Total spread PnL  : {g['total_spread_pnl_bps']:.2f} bps\n"
            f"  Total adverse sel : {g['total_adverse_selection_bps']:.2f} bps\n"
            f"  Net MM PnL        : {g['net_mm_pnl_bps']:.2f} bps\n"
            f"  Adv sel ratio     : {g['global_adverse_selection_ratio']:.2%}\n"
            f"{'='*60}"
        )
        worst = self.get_worst_symbols()
        if worst:
            print("  Worst symbols (by adverse selection ratio):")
            for sym, ratio in worst[:5]:
                print(f"    {sym:<20} {ratio:.2%}")
        print()
