"""
Position Netter — aggregates positions across all exchanges into a unified view.

Answers: "What is total BTC exposure across Kraken + Coinbase + Bybit?"
Used before placing orders to avoid inadvertently doubling up on positions.

Net position per asset:
    net_quantity = sum(long quantities) - sum(short quantities)

Positive net_quantity → net long; negative → net short; ~zero → flat.

Usage:
    netter = PositionNetter()
    netter.update("kraken",   "BTC/USD", "long",  0.5, 65000.0)
    netter.update("coinbase", "BTC/USD", "short", 0.2, 64800.0)

    net = netter.get_net("BTC/USD")
    # net.net_quantity == 0.3  (net long 0.3 BTC)

    total_usd = netter.get_total_exposure_usd({"BTC/USD": 65000.0})
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VenuePosition:
    """A single open position on a specific exchange."""

    exchange: str
    symbol: str
    side: str               # "long" or "short"
    quantity: float         # absolute (always >= 0)
    entry_price: float
    unrealised_pnl: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class NetPosition:
    """Aggregated view of a single asset across all venues."""

    symbol: str
    net_quantity: float           # positive = net long, negative = net short
    long_quantity: float          # sum of all long legs
    short_quantity: float         # sum of all short legs
    venues: List[str]             # exchanges with open positions
    weighted_avg_price: float     # notional-weighted average entry price
    total_unrealised_pnl: float


# ---------------------------------------------------------------------------
# PositionNetter
# ---------------------------------------------------------------------------

class PositionNetter:
    """
    Thread-safe aggregator of positions across multiple exchanges.

    Internal storage: keyed by (exchange, symbol).
    All public methods acquire no explicit lock — callers in a single-threaded
    asyncio context are safe.  For multi-threaded use, wrap with an external
    ``threading.Lock``.
    """

    def __init__(self) -> None:
        # (exchange, symbol) → VenuePosition
        self._positions: Dict[tuple, VenuePosition] = {}
        logger.info("PositionNetter initialised")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(
        self,
        exchange: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        unrealised_pnl: float = 0.0,
    ) -> None:
        """
        Insert or replace a position for (exchange, symbol).

        ``side`` must be "long" or "short" (case-insensitive).
        ``quantity`` must be a positive absolute value.
        Passing quantity=0 is equivalent to calling remove().
        """
        side_norm = side.lower()
        if side_norm not in {"long", "short"}:
            logger.warning(
                "PositionNetter.update: unexpected side '%s' for %s on %s; treating as long",
                side, symbol, exchange,
            )
            side_norm = "long"

        if quantity < 0:
            logger.warning(
                "PositionNetter.update: negative quantity %.6f for %s on %s; using abs()",
                quantity, symbol, exchange,
            )
            quantity = abs(quantity)

        key = (exchange, symbol)
        if quantity == 0.0:
            self._positions.pop(key, None)
            logger.debug("PositionNetter: removed zero-quantity position %s@%s", symbol, exchange)
            return

        pos = VenuePosition(
            exchange=exchange,
            symbol=symbol,
            side=side_norm,
            quantity=quantity,
            entry_price=entry_price,
            unrealised_pnl=unrealised_pnl,
        )
        self._positions[key] = pos
        logger.debug(
            "PositionNetter: updated %s@%s side=%s qty=%.6f price=%.4f",
            symbol, exchange, side_norm, quantity, entry_price,
        )

    def remove(self, exchange: str, symbol: str) -> None:
        """Remove a specific venue+symbol position if it exists."""
        key = (exchange, symbol)
        if key in self._positions:
            del self._positions[key]
            logger.debug("PositionNetter: removed position %s@%s", symbol, exchange)
        else:
            logger.debug(
                "PositionNetter.remove: no position found for %s@%s (no-op)", symbol, exchange
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_net(self, symbol: str) -> Optional[NetPosition]:
        """
        Return the aggregated net position for ``symbol`` across all exchanges.

        Returns None if no open positions exist for the symbol.
        """
        relevant = [
            pos for pos in self._positions.values() if pos.symbol == symbol
        ]
        if not relevant:
            return None

        return self._aggregate(symbol, relevant)

    def get_all_nets(self) -> Dict[str, NetPosition]:
        """Return net positions for every symbol that has at least one open leg."""
        # Group by symbol
        by_symbol: Dict[str, List[VenuePosition]] = {}
        for pos in self._positions.values():
            by_symbol.setdefault(pos.symbol, []).append(pos)

        return {
            sym: self._aggregate(sym, legs)
            for sym, legs in by_symbol.items()
        }

    def is_flat(self, symbol: str, threshold: float = 0.0001) -> bool:
        """
        True if the net quantity for ``symbol`` is within ``threshold`` of zero.

        Also returns True when no positions exist for the symbol.
        """
        net = self.get_net(symbol)
        if net is None:
            return True
        return abs(net.net_quantity) <= threshold

    def get_total_exposure_usd(self, prices: Dict[str, float]) -> float:
        """
        Sum of abs(net_quantity * price) across all symbols with a known price.

        ``prices`` is a mapping of symbol → current mid-price in USD.
        Symbols missing from ``prices`` are skipped with a warning.
        """
        total = 0.0
        for sym, net in self.get_all_nets().items():
            price = prices.get(sym)
            if price is None:
                logger.warning(
                    "PositionNetter.get_total_exposure_usd: no price for %s; skipping", sym
                )
                continue
            total += abs(net.net_quantity) * price
        return total

    def snapshot(self) -> Dict:
        """
        Return a serialisable snapshot of all positions and net aggregates.

        Suitable for logging, monitoring, or persistence.
        """
        nets = self.get_all_nets()
        return {
            "timestamp": time.time(),
            "venue_positions": [
                {
                    "exchange": p.exchange,
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "unrealised_pnl": p.unrealised_pnl,
                }
                for p in self._positions.values()
            ],
            "net_positions": {
                sym: {
                    "net_quantity": n.net_quantity,
                    "long_quantity": n.long_quantity,
                    "short_quantity": n.short_quantity,
                    "venues": n.venues,
                    "weighted_avg_price": n.weighted_avg_price,
                    "total_unrealised_pnl": n.total_unrealised_pnl,
                }
                for sym, n in nets.items()
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(symbol: str, legs: List[VenuePosition]) -> NetPosition:
        """
        Compute a NetPosition from a list of VenuePosition objects for a single symbol.
        """
        long_qty = 0.0
        short_qty = 0.0
        total_pnl = 0.0
        venues: List[str] = []
        weighted_price_num = 0.0   # numerator of weighted average price
        total_qty = 0.0

        for pos in legs:
            qty = abs(pos.quantity)
            if pos.side == "long":
                long_qty += qty
            else:
                short_qty += qty
            total_pnl += pos.unrealised_pnl
            venues.append(pos.exchange)
            weighted_price_num += qty * pos.entry_price
            total_qty += qty

        net_qty = long_qty - short_qty
        wav_price = weighted_price_num / total_qty if total_qty > 0 else 0.0

        return NetPosition(
            symbol=symbol,
            net_quantity=net_qty,
            long_quantity=long_qty,
            short_quantity=short_qty,
            venues=sorted(set(venues)),
            weighted_avg_price=wav_price,
            total_unrealised_pnl=total_pnl,
        )
