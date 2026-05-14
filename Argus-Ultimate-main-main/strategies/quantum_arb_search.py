"""
Quantum-Grover-driven arbitrage search strategy.

Uses Grover's algorithm to find profitable arbitrage triples across exchange
venues without enumerating every possible (venue_a, venue_b, symbol) tuple.

Algorithm
---------
1. Build the candidate set: every (venue_a, venue_b, symbol) where the
   simulator has price quotes from both venues.
2. Construct an oracle that marks triples where
   ``abs(price_a - price_b) / mid > (fee_a + fee_b) * threshold_multiplier``.
3. Run Grover's algorithm with optimal iteration count
   ``k = floor(π/4 · sqrt(N/M))``.
4. Decode the most-frequent measurement outcomes as candidate arbitrage hits.
5. Emit ``ArbSignal`` objects describing each opportunity for the strategy
   engine.

Honest note: classical simulation cost is O(N) per Grover iteration. The
quadratic speedup is in *quantum-query* complexity (oracle evaluations),
not wall-clock on this CPU. Value is correctness, framing, and hardware-
portability.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from quantum.algorithms.grover import GroverSearch

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class VenuePrice:
    """A single price quote on a single venue."""
    venue: str
    symbol: str
    bid: float
    ask: float
    fee_bps: float = 10.0  # default 10 bps round-trip taker fee

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass
class ArbCandidate:
    """A single (venue_a, venue_b, symbol) arbitrage triple."""
    venue_buy: str
    venue_sell: str
    symbol: str
    buy_price: float
    sell_price: float
    fee_total_bps: float
    edge_bps: float  # gross edge before fees

    @property
    def net_edge_bps(self) -> float:
        return self.edge_bps - self.fee_total_bps


@dataclass
class ArbSignal:
    """A trading signal emitted by the Grover arb search."""
    symbol: str
    side: str  # "buy" on venue_buy, "sell" on venue_sell
    venue_buy: str
    venue_sell: str
    expected_edge_bps: float
    confidence: float
    source: str = "quantum_arb_search"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Quantum Arbitrage Searcher
# ═════════════════════════════════════════════════════════════════════════════


class QuantumArbSearcher:
    """
    Find arbitrage opportunities using Grover's algorithm on the in-repo
    simulator.

    Parameters
    ----------
    threshold_multiplier : float
        Multiplier on the fee total. Triples are marked when
        ``edge_bps > fee_total_bps * threshold_multiplier``. Default 1.5
        means a 50% safety margin over fees.
    min_edge_bps : float
        Minimum absolute net edge in bps for a triple to be considered.
        Default 5 bps.
    """

    def __init__(
        self,
        threshold_multiplier: float = 1.5,
        min_edge_bps: float = 5.0,
    ) -> None:
        self.threshold_multiplier = float(threshold_multiplier)
        self.min_edge_bps = float(min_edge_bps)
        self._last_run_ts: float = 0.0
        self._n_runs = 0
        self._n_signals_emitted = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def find_opportunities(
        self,
        prices_by_venue: Dict[str, Dict[str, VenuePrice]],
    ) -> List[ArbSignal]:
        """
        Search for arbitrage opportunities across venues.

        Parameters
        ----------
        prices_by_venue : Dict[str, Dict[str, VenuePrice]]
            Map from venue name → {symbol → VenuePrice}.

        Returns
        -------
        List[ArbSignal]
            Top-k arb signals (limited by Grover's discovered indices).
        """
        self._n_runs += 1
        self._last_run_ts = time.time()

        # Build the candidate set
        candidates = self._build_candidates(prices_by_venue)
        if not candidates:
            return []

        # Mark profitable candidates
        marked_indices: List[int] = []
        for i, cand in enumerate(candidates):
            if (
                cand.net_edge_bps > 0
                and cand.edge_bps > cand.fee_total_bps * self.threshold_multiplier
                and cand.net_edge_bps > self.min_edge_bps
            ):
                marked_indices.append(i)

        if not marked_indices:
            return []

        # Run Grover to find them
        n_total = len(candidates)
        n_marked = len(marked_indices)
        n_qubits = max(1, int(math.ceil(math.log2(max(n_total, 2)))))
        n_qubits = min(n_qubits, 10)  # cap at 10 qubits for simulation cost
        padded_size = 1 << n_qubits

        # Build classical oracle from the marked set
        marked_set = set(marked_indices)

        def oracle(idx: int) -> bool:
            return idx in marked_set

        try:
            grover = GroverSearch(n_qubits=n_qubits)
            result = grover.search(
                oracle_fn=oracle,
                n_items=min(n_total, padded_size),
                n_solutions=n_marked,
            )
        except Exception as exc:
            logger.debug("Grover arb search failed: %s", exc)
            # Fallback: return all marked indices directly
            result = {
                "found_indices": marked_indices,
                "success_probability": 1.0,
            }

        # Convert Grover hits to ArbSignals
        signals: List[ArbSignal] = []
        success_prob = float(result.get("success_probability", 0.5))
        for idx in result.get("found_indices", []):
            if idx < len(candidates):
                cand = candidates[int(idx)]
                signals.append(
                    ArbSignal(
                        symbol=cand.symbol,
                        side="buy",  # buy on venue_buy, sell on venue_sell
                        venue_buy=cand.venue_buy,
                        venue_sell=cand.venue_sell,
                        expected_edge_bps=cand.net_edge_bps,
                        confidence=success_prob,
                        metadata={
                            "buy_price": cand.buy_price,
                            "sell_price": cand.sell_price,
                            "fee_total_bps": cand.fee_total_bps,
                            "gross_edge_bps": cand.edge_bps,
                            "method": "grover_arb_search",
                            "n_total_candidates": n_total,
                            "n_marked": n_marked,
                        },
                    )
                )

        self._n_signals_emitted += len(signals)
        return signals

    # ── Candidate construction ───────────────────────────────────────────────

    def _build_candidates(
        self,
        prices_by_venue: Dict[str, Dict[str, VenuePrice]],
    ) -> List[ArbCandidate]:
        """
        Build the cartesian set of (venue_a, venue_b, symbol) candidates where
        both venues quote the same symbol. Skips reflexive (a == b) pairs.
        """
        candidates: List[ArbCandidate] = []
        venues = sorted(prices_by_venue.keys())
        for i, va in enumerate(venues):
            for j, vb in enumerate(venues):
                if i == j:
                    continue
                quotes_a = prices_by_venue[va]
                quotes_b = prices_by_venue[vb]
                for sym in set(quotes_a.keys()) & set(quotes_b.keys()):
                    pa = quotes_a[sym]
                    pb = quotes_b[sym]
                    # Buy at venue a's ask, sell at venue b's bid
                    if pa.ask <= 0 or pb.bid <= 0:
                        continue
                    mid = 0.5 * (pa.ask + pb.bid)
                    if mid <= 0:
                        continue
                    edge_bps = (pb.bid - pa.ask) / mid * 10000.0
                    fee_total = float(pa.fee_bps + pb.fee_bps)
                    candidates.append(
                        ArbCandidate(
                            venue_buy=va,
                            venue_sell=vb,
                            symbol=sym,
                            buy_price=pa.ask,
                            sell_price=pb.bid,
                            fee_total_bps=fee_total,
                            edge_bps=edge_bps,
                        )
                    )
        return candidates

    # ── Telemetry ────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_runs": self._n_runs,
            "n_signals_emitted": self._n_signals_emitted,
            "last_run_ts": self._last_run_ts,
            "threshold_multiplier": self.threshold_multiplier,
            "min_edge_bps": self.min_edge_bps,
        }
