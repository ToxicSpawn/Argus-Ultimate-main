"""
micro_capital_mm.py — Micro-Capital Market Maker for ~$1k Accounts.

Optimised for zero-fee maker venues (Bybit spot) where any spread captured
is pure profit. Posts two-sided limit orders on altcoin pairs with naturally
wide spreads (30–100+ bps), skews quotes based on inventory, enforces
per-pair position limits, and halts automatically on drawdown breach.

Design philosophy
-----------------
- Primary venue: Bybit spot (0% maker fee) → profitable at 1-tick spread.
- Fallback venues: Kraken (0.16% maker) / Coinbase (0.40% maker) → need
  wider spreads to cover the fee before sizing in.
- Capital: $620 USD split across ≤3 pairs; $200/pair; ~$50/side per order.
- Inventory skew: linear, half-spread intensity  →  encourages fill on the
  overstocked side without sacrificing too much spread income.
- Kill switch: session PnL < -(max_drawdown_pct × total_capital) → cancel all.

Integration points
------------------
- QuoteThrottleFilter  from execution/quote_throttle.py
- CancelReplaceManager from execution/cancel_replace.py
- AltcoinPairScanner   from alpha/pair_scanner.py

Usage::

    config = MicroMMConfig()
    mm = MicroCapitalMM(config)
    await mm.run({"bybit": bybit_client})
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Internal imports — use absolute paths matching project layout
from execution.quote_throttle import QuoteThrottleFilter
from execution.cancel_replace import CancelReplaceManager
from alpha.pair_scanner import AltcoinPairScanner, PairOpportunity

log = logging.getLogger("argus.micro_mm")

# ---------------------------------------------------------------------------
# Exchange fee constants (maker fees)
# ---------------------------------------------------------------------------

EXCHANGE_MAKER_FEES: Dict[str, float] = {
    "bybit":    0.0000,   # Zero maker fee on spot
    "kraken":   0.0016,   # 0.16% maker
    "coinbase": 0.0040,   # 0.40% maker
}

# Minimum profitable spread for each exchange (need spread > 2× fee to profit
# after paying fee on both sides, though MM only pays once per round-trip)
def min_profitable_spread_bps(exchange: str) -> float:
    """Return the minimum spread (bps) required for a round-trip to profit."""
    fee = EXCHANGE_MAKER_FEES.get(exchange.lower(), 0.0016)
    # Round-trip cost = fee on sell side (we capture spread, pay fee once)
    return fee * 10_000  # convert to bps (e.g. 0.0016 → 16 bps)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MicroMMConfig:
    """
    Configuration for MicroCapitalMM.

    All dollar amounts are in USD.  Bybit converts AUD→USD at ~0.62 rate,
    so $1,000 AUD ≈ $620 USD.
    """

    # ── Capital ────────────────────────────────────────────────────────────
    total_capital_usd: float = 620.0
    max_pairs: int = 3
    per_pair_capital_usd: float = 200.0

    # ── Order sizing ───────────────────────────────────────────────────────
    # 25% of per_pair_capital → ~$50 per side per order
    order_size_pct: float = 0.25

    # ── Spread / fee filter ────────────────────────────────────────────────
    # Minimum spread to consider a pair (bps)
    min_spread_bps: int = 30

    # ── Inventory / risk ──────────────────────────────────────────────────
    # Never hold more than this fraction of per_pair_capital in a single asset
    max_position_pct: float = 0.50
    max_drawdown_pct: float = 15.0   # % of total_capital_usd

    # ── Timing ────────────────────────────────────────────────────────────
    refresh_interval_ms: float = 200.0   # quote refresh period
    scanner_rescan_interval_s: float = 60.0  # how often to re-rank pairs

    # ── Venues ─────────────────────────────────────────────────────────────
    exchanges: List[str] = field(default_factory=lambda: ["bybit"])
    fallback_exchanges: List[str] = field(
        default_factory=lambda: ["kraken", "coinbase"]
    )

    # ── Quote throttle tuning ─────────────────────────────────────────────
    # Min price tick for throttle (varies by asset; 0.0001 is safe default)
    min_tick: float = 0.0001
    throttle_min_age_ms: float = 50.0
    throttle_max_rate_per_sec: int = 20

    def order_size_usd(self) -> float:
        """Dollar value per order side."""
        return self.per_pair_capital_usd * self.order_size_pct

    def max_position_usd(self) -> float:
        """Maximum inventory (USD value) allowed per pair."""
        return self.per_pair_capital_usd * self.max_position_pct

    def drawdown_limit_usd(self) -> float:
        """Absolute drawdown limit in USD before kill switch triggers."""
        return -(self.total_capital_usd * self.max_drawdown_pct / 100.0)


# ---------------------------------------------------------------------------
# Per-pair inventory / order state
# ---------------------------------------------------------------------------

class PairSide(Enum):
    BID = "bid"
    ASK = "ask"


@dataclass
class OpenOrder:
    order_id: str
    symbol: str
    exchange: str
    side: PairSide
    price: float
    size: float
    placed_at: float = field(default_factory=time.time)


@dataclass
class PairState:
    """All runtime state for one actively-quoted pair."""
    symbol: str
    exchange: str
    fee_rate: float

    # Live market data (updated each refresh)
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid: float = 0.0
    spread_bps: float = 0.0

    # Inventory tracking (units of base asset, positive = long)
    inventory_base: float = 0.0
    inventory_value_usd: float = 0.0   # mark-to-market

    # Session PnL for this pair (realised)
    realised_pnl_usd: float = 0.0

    # Open quotes
    bid_order: Optional[OpenOrder] = None
    ask_order: Optional[OpenOrder] = None

    # Quote prices we most recently sent
    last_sent_bid: float = 0.0
    last_sent_ask: float = 0.0

    # Statistics
    total_fills: int = 0
    total_bid_fills: int = 0
    total_ask_fills: int = 0
    session_start: float = field(default_factory=time.time)

    def inventory_pct(self, per_pair_capital_usd: float) -> float:
        """Signed inventory as a fraction of per-pair capital. Range: [-1, +1]."""
        if per_pair_capital_usd == 0:
            return 0.0
        return self.inventory_value_usd / per_pair_capital_usd

    def unrealised_pnl_usd(self) -> float:
        """Mark-to-market PnL on current inventory (simplified: mid × inventory)."""
        if self.mid <= 0:
            return 0.0
        # inventory_value_usd is already at last-trade price; compare to mid
        # This is a simplified MTM; a full implementation would track avg cost
        return 0.0  # placeholder — realised PnL is what matters for kill switch

    def session_pnl(self) -> float:
        return self.realised_pnl_usd

    def __str__(self) -> str:
        return (
            f"PairState({self.symbol}@{self.exchange} "
            f"inv={self.inventory_base:.6f} "
            f"pnl=${self.realised_pnl_usd:.4f})"
        )


# ---------------------------------------------------------------------------
# Inventory skew formula
# ---------------------------------------------------------------------------

def compute_inventory_skew(
    inventory_pct: float,
    base_spread: float,
    skew_strength: float = 0.5,
) -> float:
    """
    Compute additive price skew based on current inventory.

    Formula
    -------
    skew = -inventory_pct × base_spread × skew_strength

    Interpretation
    --------------
    - Long inventory (positive inventory_pct):
      skew is negative → ask price moves DOWN → encourages sells.
    - Short inventory (negative inventory_pct):
      skew is positive → bid price moves UP → encourages buys.
    - skew_strength = 0.5 means at 100% inventory we shift quotes by half
      the natural spread.

    Returns
    -------
    float
        Skew amount in price units (same units as base_spread).
    """
    return -inventory_pct * base_spread * skew_strength


def compute_quotes(
    best_bid: float,
    best_ask: float,
    inventory_pct: float,
    config: MicroMMConfig,
    exchange: str,
) -> Tuple[float, float]:
    """
    Derive bid/ask quote prices with inventory skew.

    On a zero-fee venue (Bybit) we can sit at best bid/ask because every
    fill is pure profit.  On fee venues we need to be inside (or tighter than)
    the natural spread by at least the round-trip fee cost.

    Parameters
    ----------
    best_bid, best_ask : float
        Current top-of-book prices.
    inventory_pct : float
        Signed inventory as fraction of per-pair capital.
    config : MicroMMConfig
    exchange : str

    Returns
    -------
    (bid_price, ask_price)
    """
    fee = EXCHANGE_MAKER_FEES.get(exchange.lower(), 0.0)
    natural_spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2.0

    # On zero-fee venues: post at exact best bid/ask for maximum fill priority
    # On fee venues: quote inside by enough to profit after fees
    fee_offset = fee * mid  # absolute price equivalent of fee

    if fee == 0.0:
        # Sit on the best bid/ask — any fill is profitable
        base_bid = best_bid
        base_ask = best_ask
    else:
        # Need spread ≥ 2 × fee_offset to profit on both sides
        # Quote one tick inside best bid/ask so we're still at top of book
        # while ensuring the captured spread exceeds fees
        min_half_spread = fee_offset * 1.1  # 10% buffer above breakeven
        half_spread = max(natural_spread / 2.0, min_half_spread)
        base_bid = mid - half_spread
        base_ask = mid + half_spread

    # Apply inventory skew
    skew = compute_inventory_skew(
        inventory_pct=inventory_pct,
        base_spread=natural_spread,
        skew_strength=0.5,
    )
    bid_price = base_bid + skew
    ask_price = base_ask + skew

    # Sanity: ensure bid < ask always
    if bid_price >= ask_price:
        half = (bid_price + ask_price) / 2.0
        min_gap = mid * 0.0001  # 1 basis point minimum
        bid_price = half - min_gap
        ask_price = half + min_gap

    return round(bid_price, 8), round(ask_price, 8)


# ---------------------------------------------------------------------------
# MicroCapitalMM — main class
# ---------------------------------------------------------------------------

class MicroCapitalMM:
    """
    Micro-capital market maker for $1k accounts on zero-fee venues.

    Lifecycle
    ---------
    1. ``run()``   — starts the main loop; call with exchange_clients dict.
    2. ``stop()``  — cancels all open orders and logs final PnL.

    The main loop::

        while running:
            scan for top pairs (every scanner_rescan_interval_s)
            for each active pair:
                fetch latest ticker
                check fills (compare open orders vs exchange state)
                update inventory
                compute new quotes (with skew)
                throttle check → send if needed
                check kill switch
            sleep refresh_interval_ms
    """

    def __init__(self, config: MicroMMConfig) -> None:
        self.config = config
        self._running: bool = False
        self._halted: bool = False

        # Active pair states, keyed by "SYMBOL@exchange"
        self._pairs: Dict[str, PairState] = {}

        # Scanner instance
        self._scanner = AltcoinPairScanner(
            min_spread_bps=config.min_spread_bps,
            min_24h_volume_usd=50_000.0,
            max_24h_volume_usd=5_000_000.0,
        )

        # Quote throttle
        self._throttle = QuoteThrottleFilter(
            min_tick=config.min_tick,
            min_age_ms=config.throttle_min_age_ms,
            max_rate_per_sec=config.throttle_max_rate_per_sec,
        )

        # Cancel-replace manager (populated on run() with real clients)
        self._cr_manager: Optional[CancelReplaceManager] = None

        # Exchange clients (populated on run())
        self._clients: Dict[str, Any] = {}

        # Session PnL tracking (across all pairs)
        self._session_realised_pnl: float = 0.0
        self._session_start: float = time.time()
        self._last_scan_time: float = 0.0

        log.info(
            "MicroCapitalMM initialised — capital=$%.0f, max_pairs=%d, "
            "order_size=$%.0f, min_spread=%dbps",
            config.total_capital_usd,
            config.max_pairs,
            config.order_size_usd(),
            config.min_spread_bps,
        )

    # ── Public entry points ────────────────────────────────────────────────

    async def run(self, exchange_clients: Dict[str, Any]) -> None:
        """
        Main event loop.

        Parameters
        ----------
        exchange_clients : dict
            Maps exchange name (lowercase) to an async exchange client that
            implements the CCXT-style interface (fetch_tickers, create_order,
            cancel_order, fetch_open_orders, amend_order, etc.).
        """
        self._clients = {k.lower(): v for k, v in exchange_clients.items()}
        self._cr_manager = CancelReplaceManager(self._clients)
        self._running = True
        self._halted = False
        self._session_start = time.time()

        log.info(
            "MicroCapitalMM starting — exchanges: %s",
            list(self._clients.keys()),
        )

        try:
            while self._running and not self._halted:
                loop_start = time.monotonic()

                # ── 1. Re-scan for pairs if due ────────────────────────
                await self._maybe_rescan_pairs()

                # ── 2. Process each active pair ────────────────────────
                tasks = [
                    self._refresh_pair(pair_key, ps)
                    for pair_key, ps in list(self._pairs.items())
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # ── 3. Kill switch check ───────────────────────────────
                if self._check_kill_switch():
                    log.critical(
                        "MicroCapitalMM KILL SWITCH TRIGGERED — "
                        "session PnL $%.4f < limit $%.4f",
                        self._session_realised_pnl,
                        self.config.drawdown_limit_usd(),
                    )
                    self._halted = True
                    break

                # ── 4. Sleep for remainder of refresh interval ─────────
                elapsed_ms = (time.monotonic() - loop_start) * 1_000.0
                sleep_s = max(
                    0.0, (self.config.refresh_interval_ms - elapsed_ms) / 1_000.0
                )
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

        except asyncio.CancelledError:
            log.info("MicroCapitalMM run() cancelled")
        except Exception as exc:
            log.exception("MicroCapitalMM fatal error: %s", exc)
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Cancel all open orders and log final PnL."""
        self._running = False

        log.info("MicroCapitalMM stopping — cancelling all open orders …")
        cancel_tasks = []
        for ps in self._pairs.values():
            if ps.bid_order is not None:
                cancel_tasks.append(self._cancel_order(ps, ps.bid_order))
            if ps.ask_order is not None:
                cancel_tasks.append(self._cancel_order(ps, ps.ask_order))

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

        runtime_s = time.time() - self._session_start
        log.info(
            "MicroCapitalMM stopped. Session PnL: $%.4f | Runtime: %.1fs | Pairs: %d",
            self._session_realised_pnl,
            runtime_s,
            len(self._pairs),
        )

        for ps in self._pairs.values():
            log.info(
                "  [%s] fills=%d realised=$%.4f inventory=%.6f",
                ps.symbol, ps.total_fills, ps.realised_pnl_usd, ps.inventory_base,
            )

    def get_status(self) -> dict:
        """
        Return a snapshot of current running state.

        Returns
        -------
        dict with keys:
            running, halted, pairs, session_pnl_usd, drawdown_pct,
            drawdown_limit_usd, uptime_s
        """
        pairs_info = {}
        for key, ps in self._pairs.items():
            pairs_info[key] = {
                "symbol": ps.symbol,
                "exchange": ps.exchange,
                "best_bid": ps.best_bid,
                "best_ask": ps.best_ask,
                "spread_bps": ps.spread_bps,
                "inventory_base": ps.inventory_base,
                "inventory_value_usd": ps.inventory_value_usd,
                "inventory_pct": ps.inventory_pct(self.config.per_pair_capital_usd),
                "realised_pnl_usd": ps.realised_pnl_usd,
                "total_fills": ps.total_fills,
                "bid_order_id": ps.bid_order.order_id if ps.bid_order else None,
                "ask_order_id": ps.ask_order.order_id if ps.ask_order else None,
            }

        drawdown_pct = 0.0
        if self.config.total_capital_usd > 0:
            drawdown_pct = (
                self._session_realised_pnl / self.config.total_capital_usd * 100.0
            )

        return {
            "running": self._running,
            "halted": self._halted,
            "pairs": pairs_info,
            "session_pnl_usd": self._session_realised_pnl,
            "drawdown_pct": drawdown_pct,
            "drawdown_limit_usd": self.config.drawdown_limit_usd(),
            "uptime_s": time.time() - self._session_start,
        }

    # ── Internal: pair management ──────────────────────────────────────────

    async def _maybe_rescan_pairs(self) -> None:
        """Re-scan for top pairs if the rescan interval has elapsed."""
        now = time.time()
        if now - self._last_scan_time < self.config.scanner_rescan_interval_s:
            return

        self._last_scan_time = now
        log.info("MicroCapitalMM: scanning for top pairs …")

        opportunities = await self._scanner.scan_all_exchanges(self._clients)
        if not opportunities:
            log.warning("MicroCapitalMM: pair scanner returned no opportunities")
            return

        # Limit to max_pairs; prefer zero-fee venues
        sorted_opps = sorted(
            opportunities,
            key=lambda o: (
                -(EXCHANGE_MAKER_FEES.get(o.exchange, 0.001)),  # prefer low fee
                -o.score,
            ),
        )
        selected = sorted_opps[: self.config.max_pairs]

        # Build set of desired pair keys
        desired = {f"{o.symbol}@{o.exchange}" for o in selected}
        current = set(self._pairs.keys())

        # Drop pairs we no longer want
        for key in current - desired:
            await self._retire_pair(key)

        # Add new pairs
        for opp in selected:
            key = f"{opp.symbol}@{opp.exchange}"
            if key not in self._pairs:
                await self._activate_pair(opp)

        log.info(
            "MicroCapitalMM: active pairs: %s",
            list(self._pairs.keys()),
        )

    async def _activate_pair(self, opp: PairOpportunity) -> None:
        """Start market making on a new pair."""
        key = f"{opp.symbol}@{opp.exchange}"
        fee = EXCHANGE_MAKER_FEES.get(opp.exchange.lower(), 0.0016)
        ps = PairState(
            symbol=opp.symbol,
            exchange=opp.exchange,
            fee_rate=fee,
            best_bid=opp.bid,
            best_ask=opp.ask,
            mid=(opp.bid + opp.ask) / 2.0,
            spread_bps=opp.spread_bps,
        )
        self._pairs[key] = ps
        log.info(
            "MicroCapitalMM: activating pair %s — spread=%.1fbps fee=%.4f score=%.2f",
            key, opp.spread_bps, fee, opp.score,
        )

    async def _retire_pair(self, pair_key: str) -> None:
        """Cancel open orders for a pair and remove from active set."""
        ps = self._pairs.get(pair_key)
        if ps is None:
            return

        log.info("MicroCapitalMM: retiring pair %s", pair_key)
        if ps.bid_order:
            await self._cancel_order(ps, ps.bid_order)
        if ps.ask_order:
            await self._cancel_order(ps, ps.ask_order)

        self._session_realised_pnl += ps.realised_pnl_usd
        del self._pairs[pair_key]

    # ── Internal: quote refresh ────────────────────────────────────────────

    async def _refresh_pair(self, pair_key: str, ps: PairState) -> None:
        """
        One refresh cycle for a single pair:
        1. Fetch latest ticker.
        2. Check if any open orders have been filled.
        3. Re-quote with inventory skew.
        """
        try:
            # ── 1. Fetch ticker ────────────────────────────────────────
            ticker = await self._fetch_ticker(ps.symbol, ps.exchange)
            if ticker is None:
                return

            ps.best_bid = float(ticker.get("bid", 0.0) or 0.0)
            ps.best_ask = float(ticker.get("ask", 0.0) or 0.0)
            if ps.best_bid <= 0 or ps.best_ask <= 0:
                return

            ps.mid = (ps.best_bid + ps.best_ask) / 2.0
            ps.spread_bps = (
                (ps.best_ask - ps.best_bid) / ps.mid * 10_000
                if ps.mid > 0 else 0.0
            )

            # ── 2. Check fills ─────────────────────────────────────────
            await self._check_fills(ps)

            # ── 3. Check if this pair is still viable ──────────────────
            if ps.spread_bps < self.config.min_spread_bps:
                log.debug(
                    "QuoteRefresh[%s]: spread=%.1fbps below min=%dbps — skipping",
                    ps.symbol, ps.spread_bps, self.config.min_spread_bps,
                )
                return

            # Verify spread is profitable after fees
            required = min_profitable_spread_bps(ps.exchange)
            if ps.spread_bps < required:
                log.debug(
                    "QuoteRefresh[%s]: spread=%.1fbps < fee breakeven %.1fbps",
                    ps.symbol, ps.spread_bps, required,
                )
                return

            # ── 4. Compute new quotes ──────────────────────────────────
            inv_pct = ps.inventory_pct(self.config.per_pair_capital_usd)
            new_bid, new_ask = compute_quotes(
                best_bid=ps.best_bid,
                best_ask=ps.best_ask,
                inventory_pct=inv_pct,
                config=self.config,
                exchange=ps.exchange,
            )

            skew = compute_inventory_skew(
                inventory_pct=inv_pct,
                base_spread=ps.best_ask - ps.best_bid,
            )

            log.debug(
                "QuoteRefresh[%s]: bid=%.6f ask=%.6f spread=%.1fbps "
                "inv=%.4f (%.1f%%) skew=%.6f",
                ps.symbol, new_bid, new_ask, ps.spread_bps,
                ps.inventory_base, inv_pct * 100.0, skew,
            )

            # ── 5. Throttle check ──────────────────────────────────────
            if not self._throttle.should_refresh(
                symbol=pair_key,
                new_bid=new_bid,
                new_ask=new_ask,
                last_bid=ps.last_sent_bid or None,
                last_ask=ps.last_sent_ask or None,
            ):
                return

            # ── 6. Send / amend quotes ─────────────────────────────────
            await self._send_quotes(ps, new_bid, new_ask)
            self._throttle.record_refresh(pair_key, bid=new_bid, ask=new_ask)
            ps.last_sent_bid = new_bid
            ps.last_sent_ask = new_ask

        except Exception as exc:
            log.warning(
                "QuoteRefresh[%s]: unhandled error: %s", pair_key, exc
            )

    async def _send_quotes(
        self, ps: PairState, new_bid: float, new_ask: float
    ) -> None:
        """
        Place or amend the two-sided quote for a pair.

        Uses CancelReplaceManager when orders already exist.
        Places fresh orders if no open orders on either side.
        Skips the side whose position limit would be breached.
        """
        order_size_usd = self.config.order_size_usd()
        order_size_base = order_size_usd / ps.mid if ps.mid > 0 else 0.0

        # ── Check position limits ──────────────────────────────────────
        # Bid side: buying more would increase long inventory
        long_after_fill = ps.inventory_value_usd + order_size_usd
        can_bid = long_after_fill <= self.config.max_position_usd()

        # Ask side: selling would reduce long (or increase short) inventory
        short_after_fill = ps.inventory_value_usd - order_size_usd
        can_ask = short_after_fill >= -self.config.max_position_usd()

        client = self._clients.get(ps.exchange)
        if client is None:
            return

        assert self._cr_manager is not None

        # ── Bid side ───────────────────────────────────────────────────
        if can_bid:
            if ps.bid_order is not None:
                # Amend existing bid
                try:
                    result = await self._cr_manager.amend_order(
                        exchange=ps.exchange,
                        order_id=ps.bid_order.order_id,
                        symbol=ps.symbol,
                        new_price=new_bid,
                        new_size=order_size_base,
                    )
                    ps.bid_order.price = new_bid
                    log.debug(
                        "Quote[%s]: bid amended → %.6f (order %s)",
                        ps.symbol, new_bid, ps.bid_order.order_id,
                    )
                except Exception as exc:
                    log.warning("Quote[%s]: bid amend failed: %s — placing fresh", ps.symbol, exc)
                    ps.bid_order = None

            if ps.bid_order is None:
                # Place fresh bid
                try:
                    order = await client.create_order(
                        symbol=ps.symbol,
                        side="buy",
                        price=new_bid,
                        size=order_size_base,
                        order_type="limit",
                    )
                    ps.bid_order = OpenOrder(
                        order_id=str(order.get("id", "")),
                        symbol=ps.symbol,
                        exchange=ps.exchange,
                        side=PairSide.BID,
                        price=new_bid,
                        size=order_size_base,
                    )
                    log.info(
                        "Quote[%s]: bid placed %.6f × %.6f (id=%s)",
                        ps.symbol, new_bid, order_size_base, ps.bid_order.order_id,
                    )
                except Exception as exc:
                    log.warning("Quote[%s]: bid placement failed: %s", ps.symbol, exc)
        else:
            # Position limit hit on bid side — cancel if outstanding
            if ps.bid_order is not None:
                await self._cancel_order(ps, ps.bid_order)
                ps.bid_order = None
                log.info(
                    "Quote[%s]: bid cancelled — position limit (%.1f%% of capital)",
                    ps.symbol,
                    ps.inventory_pct(self.config.per_pair_capital_usd) * 100.0,
                )

        # ── Ask side ───────────────────────────────────────────────────
        if can_ask:
            if ps.ask_order is not None:
                try:
                    result = await self._cr_manager.amend_order(
                        exchange=ps.exchange,
                        order_id=ps.ask_order.order_id,
                        symbol=ps.symbol,
                        new_price=new_ask,
                        new_size=order_size_base,
                    )
                    ps.ask_order.price = new_ask
                    log.debug(
                        "Quote[%s]: ask amended → %.6f (order %s)",
                        ps.symbol, new_ask, ps.ask_order.order_id,
                    )
                except Exception as exc:
                    log.warning("Quote[%s]: ask amend failed: %s — placing fresh", ps.symbol, exc)
                    ps.ask_order = None

            if ps.ask_order is None:
                try:
                    order = await client.create_order(
                        symbol=ps.symbol,
                        side="sell",
                        price=new_ask,
                        size=order_size_base,
                        order_type="limit",
                    )
                    ps.ask_order = OpenOrder(
                        order_id=str(order.get("id", "")),
                        symbol=ps.symbol,
                        exchange=ps.exchange,
                        side=PairSide.ASK,
                        price=new_ask,
                        size=order_size_base,
                    )
                    log.info(
                        "Quote[%s]: ask placed %.6f × %.6f (id=%s)",
                        ps.symbol, new_ask, order_size_base, ps.ask_order.order_id,
                    )
                except Exception as exc:
                    log.warning("Quote[%s]: ask placement failed: %s", ps.symbol, exc)
        else:
            if ps.ask_order is not None:
                await self._cancel_order(ps, ps.ask_order)
                ps.ask_order = None
                log.info(
                    "Quote[%s]: ask cancelled — short position limit (%.1f%%)",
                    ps.symbol,
                    ps.inventory_pct(self.config.per_pair_capital_usd) * 100.0,
                )

    # ── Internal: fill detection ───────────────────────────────────────────

    async def _check_fills(self, ps: PairState) -> None:
        """
        Compare open order state with exchange.  If an order is gone
        (filled or cancelled), update inventory and PnL.

        This is a lightweight polling approach.  A production system would
        use websocket order update streams; this is compatible with CCXT's
        ``watch_orders`` when available.
        """
        client = self._clients.get(ps.exchange)
        if client is None:
            return

        try:
            open_orders: List[dict] = await client.fetch_open_orders(ps.symbol)
            open_ids = {str(o.get("id", "")) for o in open_orders}
        except Exception as exc:
            log.debug("FillCheck[%s]: fetch_open_orders failed: %s", ps.symbol, exc)
            return

        # ── Check bid fill ─────────────────────────────────────────────
        if ps.bid_order is not None and ps.bid_order.order_id not in open_ids:
            # Bid order is gone — assume filled
            fill_price = ps.bid_order.price
            fill_size = ps.bid_order.size
            fill_value_usd = fill_price * fill_size

            ps.inventory_base += fill_size
            ps.inventory_value_usd += fill_value_usd
            ps.total_fills += 1
            ps.total_bid_fills += 1

            # Cost of this fill (fee × fill_value; zero on Bybit)
            fee_cost = fill_value_usd * ps.fee_rate
            # No spread captured yet on buy — profit realises when ask fills
            ps.realised_pnl_usd -= fee_cost

            log.info(
                "Fill[%s]: BID filled %.6f @ %.6f ($%.2f) fee=$%.4f inv=%.6f",
                ps.symbol, fill_size, fill_price, fill_value_usd,
                fee_cost, ps.inventory_base,
            )
            ps.bid_order = None

        # ── Check ask fill ─────────────────────────────────────────────
        if ps.ask_order is not None and ps.ask_order.order_id not in open_ids:
            fill_price = ps.ask_order.price
            fill_size = ps.ask_order.size
            fill_value_usd = fill_price * fill_size

            # Realise PnL: spread = ask_price - bid_price (last bid fill)
            # Simplified: mark the full USD proceeds as reducing inventory
            ps.inventory_base -= fill_size
            ps.inventory_value_usd -= fill_value_usd
            ps.total_fills += 1
            ps.total_ask_fills += 1

            # For a matched round-trip: profit = (ask - bid) × size − fees
            # Without tracking exact cost basis per lot, we approximate by
            # crediting the spread on the ask side fill
            if ps.last_sent_bid > 0 and ps.last_sent_ask > 0:
                captured_spread = ps.last_sent_ask - ps.last_sent_bid
                round_trip_pnl = captured_spread * fill_size
                fee_cost = fill_value_usd * ps.fee_rate
                net_pnl = round_trip_pnl - fee_cost
                ps.realised_pnl_usd += net_pnl
                self._session_realised_pnl += net_pnl
                log.info(
                    "Fill[%s]: ASK filled %.6f @ %.6f ($%.2f) "
                    "spread=$%.4f fee=$%.4f net=$%.4f",
                    ps.symbol, fill_size, fill_price, fill_value_usd,
                    round_trip_pnl, fee_cost, net_pnl,
                )
            else:
                fee_cost = fill_value_usd * ps.fee_rate
                ps.realised_pnl_usd -= fee_cost
                log.info(
                    "Fill[%s]: ASK filled %.6f @ %.6f (no bid reference)",
                    ps.symbol, fill_size, fill_price,
                )

            ps.ask_order = None

    # ── Internal: cancel helper ────────────────────────────────────────────

    async def _cancel_order(self, ps: PairState, order: OpenOrder) -> None:
        """Cancel a single order on the exchange, ignoring errors."""
        client = self._clients.get(ps.exchange)
        if client is None:
            return
        try:
            await client.cancel_order(order.order_id, ps.symbol)
            log.debug(
                "Cancel[%s]: cancelled %s order %s @ %.6f",
                ps.symbol, order.side.value, order.order_id, order.price,
            )
        except Exception as exc:
            log.debug(
                "Cancel[%s]: cancel failed for order %s: %s",
                ps.symbol, order.order_id, exc,
            )

    # ── Internal: ticker fetch ─────────────────────────────────────────────

    async def _fetch_ticker(
        self, symbol: str, exchange: str
    ) -> Optional[dict]:
        """Fetch current best bid/ask for symbol from exchange."""
        client = self._clients.get(exchange)
        if client is None:
            return None
        try:
            ticker = await client.fetch_ticker(symbol)
            return ticker if isinstance(ticker, dict) else None
        except Exception as exc:
            log.debug("TickerFetch[%s@%s]: %s", symbol, exchange, exc)
            return None

    # ── Internal: kill switch ──────────────────────────────────────────────

    def _check_kill_switch(self) -> bool:
        """
        Return True if the session PnL has breached the drawdown limit.

        Uses realised PnL only (conservative — we don't mark open inventory).
        """
        if self._session_realised_pnl < self.config.drawdown_limit_usd():
            return True
        return False

    # ── Fee calculation helper (public utility) ────────────────────────────

    @staticmethod
    def calculate_fee_for_order(
        price: float,
        size: float,
        exchange: str,
    ) -> float:
        """
        Calculate the dollar fee for one maker order on the given exchange.

        Parameters
        ----------
        price : float
            Limit price of the order.
        size : float
            Order size in base asset units.
        exchange : str
            Exchange name (case-insensitive).

        Returns
        -------
        float
            Fee in USD.  Zero for Bybit spot.
        """
        fee_rate = EXCHANGE_MAKER_FEES.get(exchange.lower(), 0.0016)
        notional = price * size
        return notional * fee_rate
