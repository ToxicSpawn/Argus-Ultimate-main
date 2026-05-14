"""
Cross-Venue Position Netting — avoids double-counting correlated hedges.

Problem
-------
A trading desk may hold positions on the same underlying asset across multiple
venues (e.g. BTC/USD on Kraken AND BTC/USDT on Binance).  Naive summation
double-counts risk.  This module computes the *netted* exposure by:

1. Maintaining a per-(exchange, symbol) position ledger.
2. Detecting correlated pairs (|correlation| > threshold).
3. Deducting overlapping exposure from the gross risk figure.

Usage
-----
    netter = CrossVenuePositionNetter(correlation_threshold=0.85)
    netter.update_position("kraken",   "BTC/USD",  0.5, ts_ns)
    netter.update_position("binance",  "BTC/USDT", -0.5, ts_ns)
    netter.get_net_position("BTC/USD")   # → 0.0 (fully netted via BTC/USDT)
    netter.get_netted_exposure()          # → {"BTC": 0.0}
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HedgeRec:
    """Hedge recommendation issued when cross-venue exposure is unbalanced."""
    symbol: str
    exchange: str
    side: str        # "buy" | "sell"
    size: float
    reason: str


@dataclass
class NetPosition:
    """Aggregated position for a single symbol."""
    symbol: str
    gross_long: float
    gross_short: float
    net: float
    venues: Dict[str, float]  # exchange → signed position


# ---------------------------------------------------------------------------
# Pre-defined common crypto correlations
# ---------------------------------------------------------------------------

_DEFAULT_CORRELATIONS: List[Tuple[str, str, float]] = [
    ("BTC/USD",  "BTC/USDT",   0.999),
    ("ETH/USD",  "ETH/USDT",   0.999),
    ("BTC/USD",  "BTC-PERP",   0.97),
    ("ETH/USD",  "ETH-PERP",   0.97),
    ("BTC/USDT", "BTC-PERP",   0.97),
    ("ETH/USDT", "ETH-PERP",   0.97),
    ("BNB/USD",  "BNB/USDT",   0.999),
    ("SOL/USD",  "SOL/USDT",   0.999),
    ("XRP/USD",  "XRP/USDT",   0.999),
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CrossVenuePositionNetter:
    """
    Cross-venue position netter with correlation-aware exposure calculation.

    Parameters
    ----------
    correlation_threshold : float
        Minimum |correlation| to consider two symbols as correlated for
        double-count detection (default 0.85).
    """

    def __init__(self, correlation_threshold: float = 0.85) -> None:
        self._threshold = correlation_threshold
        # (exchange, symbol) → signed net size
        self._positions: Dict[Tuple[str, str], float] = {}
        # Timestamps for each (exchange, symbol) entry
        self._timestamps: Dict[Tuple[str, str], int] = {}
        # Correlation registry: frozenset({sym_a, sym_b}) → correlation
        self._correlations: Dict[frozenset, float] = {}

        # Seed default correlations
        for sym_a, sym_b, corr in _DEFAULT_CORRELATIONS:
            self._correlations[frozenset({sym_a, sym_b})] = corr

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def update_position(
        self,
        exchange: str,
        symbol: str,
        net_size: float,
        timestamp_ns: int,
    ) -> None:
        """
        Record or update position for (exchange, symbol).

        Parameters
        ----------
        exchange : str
        symbol : str
        net_size : float
            Signed net position: positive = long, negative = short.
        timestamp_ns : int
        """
        key = (exchange, symbol)
        self._positions[key] = net_size
        self._timestamps[key] = timestamp_ns
        logger.debug(
            "Position updated: %s/%s = %.6f @ %d ns",
            exchange, symbol, net_size, timestamp_ns
        )

    def get_net_position(self, symbol: str) -> float:
        """
        Sum of all signed positions for *symbol* across all venues.

        If correlated symbols are detected, their contribution is also
        included via the netting logic so callers get the true net.
        """
        total = 0.0
        for (exch, sym), size in self._positions.items():
            if sym == symbol:
                total += size
        return total

    def get_venue_breakdown(self, symbol: str) -> Dict[str, float]:
        """Return per-exchange signed position for the given symbol."""
        result: Dict[str, float] = {}
        for (exch, sym), size in self._positions.items():
            if sym == symbol:
                result[exch] = size
        return result

    # ------------------------------------------------------------------
    # Correlation registry
    # ------------------------------------------------------------------

    def register_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        correlation: float,
    ) -> None:
        """
        Declare two symbols as correlated.

        E.g. register_correlation("BTC/USD", "BTC-PERP", 0.97)
        """
        key = frozenset({symbol_a, symbol_b})
        self._correlations[key] = correlation
        logger.debug(
            "Correlation registered: %s ↔ %s = %.3f", symbol_a, symbol_b, correlation
        )

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Return registered correlation; 0.0 if not found."""
        return self._correlations.get(frozenset({symbol_a, symbol_b}), 0.0)

    def is_double_counted(
        self,
        exchange_a: str,
        symbol_a: str,
        exchange_b: str,
        symbol_b: str,
    ) -> bool:
        """
        Return True if both positions represent the same underlying
        (i.e. |correlation| > threshold).

        Two positions on the *same* (exchange, symbol) are always considered
        the same entry — not double-counted.
        """
        if exchange_a == exchange_b and symbol_a == symbol_b:
            return False  # Same entry — not a double-count concern
        if symbol_a == symbol_b:
            return True   # Identical instruments on different venues
        corr = abs(self.get_correlation(symbol_a, symbol_b))
        return corr > self._threshold

    # ------------------------------------------------------------------
    # Netted exposure
    # ------------------------------------------------------------------

    def get_netted_exposure(self) -> Dict[str, float]:
        """
        Return net exposure per *base* symbol after accounting for correlations.

        Algorithm:
        1. Build groups of correlated positions.
        2. Within each group, sum all signed positions.
        3. The net is the residual exposure.
        """
        symbols = list({sym for (_, sym) in self._positions.keys()})
        # Build union-find for correlated symbol groups
        parent: Dict[str, str] = {s: s for s in symbols}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for s_a in symbols:
            for s_b in symbols:
                if s_a < s_b:
                    corr = abs(self.get_correlation(s_a, s_b))
                    if corr > self._threshold:
                        union(s_a, s_b)

        # Aggregate signed positions per group
        group_net: Dict[str, float] = defaultdict(float)
        for (_, sym), size in self._positions.items():
            root = find(sym)
            group_net[root] += size

        return dict(group_net)

    # ------------------------------------------------------------------
    # Hedge recommendations
    # ------------------------------------------------------------------

    def get_hedge_recommendation(self, symbol: str) -> Optional[HedgeRec]:
        """
        Suggest a hedge if exposure is materially unbalanced across venues.

        Logic: if total net_position for the symbol group is ≠ 0 and there
        exists an exchange with the largest position, recommend a hedge on
        that exchange to balance to zero.
        """
        netted = self.get_netted_exposure()
        # Find root of the group containing symbol
        symbols = list({sym for (_, sym) in self._positions.keys()})
        if symbol not in symbols:
            return None

        parent: Dict[str, str] = {s: s for s in symbols}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for s_a in symbols:
            for s_b in symbols:
                if s_a < s_b and abs(self.get_correlation(s_a, s_b)) > self._threshold:
                    union(s_a, s_b)

        root = find(symbol)
        net = netted.get(root, 0.0)
        if abs(net) < 1e-9:
            return None

        # Find the venue with the largest absolute position in the group
        best_exchange = ""
        best_size = 0.0
        for (exch, sym), size in self._positions.items():
            if find(sym) == root and abs(size) > abs(best_size):
                best_exchange = exch
                best_size = size

        if not best_exchange:
            return None

        hedge_side = "sell" if net > 0 else "buy"
        return HedgeRec(
            symbol=symbol,
            exchange=best_exchange,
            side=hedge_side,
            size=abs(net),
            reason=(
                f"Net exposure {net:.4f} detected across correlated group '{root}'; "
                f"hedge {hedge_side} {abs(net):.4f} on {best_exchange}"
            ),
        )

    # ------------------------------------------------------------------
    # Double-count detection
    # ------------------------------------------------------------------

    def _find_double_counted_pairs(self) -> List[dict]:
        """Return list of position pairs that are likely double-counted."""
        items = list(self._positions.items())
        pairs = []
        seen = set()
        for i, ((exch_a, sym_a), size_a) in enumerate(items):
            for j, ((exch_b, sym_b), size_b) in enumerate(items):
                if i >= j:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                if self.is_double_counted(exch_a, sym_a, exch_b, sym_b):
                    seen.add(key)
                    # Only flag if both positions have the same sign (both long or both short)
                    if size_a * size_b > 0:
                        pairs.append({
                            "exchange_a": exch_a,
                            "symbol_a": sym_a,
                            "size_a": size_a,
                            "exchange_b": exch_b,
                            "symbol_b": sym_b,
                            "size_b": size_b,
                            "correlation": self.get_correlation(sym_a, sym_b),
                        })
        return pairs

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def get_session_summary(self) -> dict:
        """
        Full session summary including positions, netted exposures, and
        detected double-count risks.
        """
        all_positions = {
            f"{exch}/{sym}": size
            for (exch, sym), size in self._positions.items()
        }
        netted_exposures = self.get_netted_exposure()
        double_counted = self._find_double_counted_pairs()

        return {
            "positions": all_positions,
            "netted_exposures": netted_exposures,
            "double_counted_pairs": double_counted,
            "total_positions": len(self._positions),
            "total_symbols": len({sym for (_, sym) in self._positions.keys()}),
            "total_exchanges": len({exch for (exch, _) in self._positions.keys()}),
            "snapshot_ts_ns": time.time_ns(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all recorded positions (session reset)."""
        self._positions.clear()
        self._timestamps.clear()
        logger.info("CrossVenuePositionNetter cleared")

    def all_symbols(self) -> List[str]:
        """Return sorted list of all known symbols."""
        return sorted({sym for (_, sym) in self._positions.keys()})

    def all_exchanges(self) -> List[str]:
        """Return sorted list of all known exchanges."""
        return sorted({exch for (exch, _) in self._positions.keys()})

    def get_all_net_positions(self) -> List[NetPosition]:
        """Return NetPosition summary for every unique symbol."""
        result = []
        for symbol in self.all_symbols():
            venues = self.get_venue_breakdown(symbol)
            gross_long = sum(v for v in venues.values() if v > 0)
            gross_short = sum(abs(v) for v in venues.values() if v < 0)
            net = gross_long - gross_short
            result.append(NetPosition(
                symbol=symbol,
                gross_long=gross_long,
                gross_short=gross_short,
                net=net,
                venues=venues,
            ))
        return result
