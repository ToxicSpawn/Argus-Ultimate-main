"""
pair_scanner_extended.py — Extended Pair Scanner for MEXC and BTC Markets.

Extends the base ``AltcoinPairScanner`` (alpha/pair_scanner.py) with two new
exchange integrations:

  * **MEXC** — 1000+ USDT spot pairs, 0% maker fee.  Any captured spread is
    pure gross profit.  Scoring multiplier ×1.5 vs Bybit baseline.

  * **BTC Markets** — AUD-quoted pairs, -0.05% maker rebate.  Exchange pays
    you *per fill*, so even a zero-spread quote is profitable.  Scoring
    multiplier ×2.0.

Design notes
------------
- ``ExtendedPairScanner`` inherits from ``AltcoinPairScanner``.  All original
  Bybit / Kraken / Coinbase logic is preserved.
- MEXC is filtered aggressively (1000+ pairs available) but with a *lower*
  volume minimum than Bybit because there is less competition and mid-cap
  pairs are less liquid.
- BTC Markets AUD prices are normalised to USD before scoring so all
  opportunities can be ranked on a common scale.
- ``scan_all`` merges results from all venues and returns the top 20 (up from
  top 10 in the base scanner).

Scoring formula (inherited, then scaled per venue)
--------------------------------------------------
  base_score = spread_bps × sqrt(volume_24h_usd)
  mexc_score   = base_score × 1.5
  btcm_score   = base_score × 2.0

Usage::

    scanner = ExtendedPairScanner(min_spread_bps=20)
    opps = await scanner.scan_all({
        "bybit":      bybit_client,
        "mexc":       mexc_client,
        "btcmarkets": btcm_client,
    })
    recommended = scanner.get_recommended_by_exchange()
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Base scanner import
# ---------------------------------------------------------------------------
from alpha.pair_scanner import (
    AltcoinPairScanner,
    PairOpportunity,
    STABLECOINS,
    TOO_LIQUID_BASES,
    estimate_daily_profit,
    ESTIMATED_FILLS_PER_DAY,
)

log = logging.getLogger("argus.pair_scanner_extended")

# ---------------------------------------------------------------------------
# Fee constants for the new venues
# ---------------------------------------------------------------------------

MEXC_SPOT_MAKER_FEE: float = 0.0       # zero maker fee
MEXC_SPOT_TAKER_FEE: float = 0.0005
BTCM_MAKER_FEE: float = -0.0005        # negative rebate — exchange pays maker

# ---------------------------------------------------------------------------
# Volume / spread filters for MEXC
# (lower min volume than Bybit because less competition on mid-cap pairs)
# ---------------------------------------------------------------------------

MEXC_MIN_VOLUME_USD: float = 20_000.0   # $20k min 24h volume
MEXC_MAX_VOLUME_USD: float = 2_000_000.0  # $2M max (beyond this spreads compress)
MEXC_MIN_SPREAD_BPS: float = 20.0       # 20 bps minimum (lower bar than Bybit)

# ---------------------------------------------------------------------------
# BTC Markets pairs of interest (AUD-quoted)
# ---------------------------------------------------------------------------

BTCM_DEFAULT_PAIRS: List[str] = [
    "BTC-AUD",
    "ETH-AUD",
    "SOL-AUD",
    "XRP-AUD",
    "DOGE-AUD",
]

# ---------------------------------------------------------------------------
# Scoring multipliers — reward zero/negative fee venues
# ---------------------------------------------------------------------------

MEXC_SCORE_MULTIPLIER: float = 1.5    # 0% fee: all spread is profit
BTCM_SCORE_MULTIPLIER: float = 2.0   # negative fee: rebate = free money

# How many top opportunities to return from the combined scan
TOP_N_COMBINED: int = 20
TOP_N_PER_EXCHANGE: int = 5


# ---------------------------------------------------------------------------
# ExtendedPairScanner
# ---------------------------------------------------------------------------

class ExtendedPairScanner(AltcoinPairScanner):
    """
    Extends ``AltcoinPairScanner`` to include MEXC and BTC Markets.

    Parameters
    ----------
    min_spread_bps : int
        Global minimum spread filter applied to Bybit / Kraken / Coinbase.
        MEXC uses its own lower threshold (MEXC_MIN_SPREAD_BPS).
    order_size_usd : float
        Order size used for daily profit estimation.
    aud_usd_rate : float
        AUD→USD conversion rate for normalising BTC Markets prices.
        Defaults to 0.62.
    """

    def __init__(
        self,
        min_spread_bps: int = 30,
        order_size_usd: float = 50.0,
        aud_usd_rate: float = 0.62,
    ) -> None:
        super().__init__(min_spread_bps=min_spread_bps, order_size_usd=order_size_usd)
        self.aud_usd_rate: float = aud_usd_rate
        self._last_scan_results: Dict[str, List[PairOpportunity]] = {}

    # ------------------------------------------------------------------
    # MEXC scanning
    # ------------------------------------------------------------------

    async def scan_mexc(self, mexc_client: Any) -> List[PairOpportunity]:
        """
        Scan MEXC spot tickers and return scored opportunities.

        Filters
        -------
        - USDT-quoted pairs only.
        - Base asset not in STABLECOINS or TOO_LIQUID_BASES.
        - 24h volume between MEXC_MIN_VOLUME_USD and MEXC_MAX_VOLUME_USD.
        - Spread ≥ MEXC_MIN_SPREAD_BPS.

        Scoring
        -------
        ``spread_bps × sqrt(volume_24h_usd) × MEXC_SCORE_MULTIPLIER``

        Because the maker fee is exactly zero, every basis point of captured
        spread is gross profit — hence the ×1.5 bonus over the Bybit baseline.

        Parameters
        ----------
        mexc_client : MEXCClient
            Authenticated MEXC client with a ``get_all_tickers()`` coroutine
            returning a list of dicts with keys:
            ``symbol``, ``bid``, ``ask``, ``volume_24h_usdt``.

        Returns
        -------
        list[PairOpportunity]
            Filtered, scored, sorted opportunities (best first).
        """
        log.info("scan_mexc: fetching all tickers from MEXC …")
        try:
            tickers: List[Dict[str, Any]] = await mexc_client.get_all_tickers()
        except Exception as exc:
            log.error("scan_mexc: failed to fetch tickers — %s", exc)
            return []

        opportunities: List[PairOpportunity] = []

        for ticker in tickers:
            symbol: str = ticker.get("symbol", "")

            # USDT pairs only
            if not symbol.endswith("USDT"):
                continue

            base = symbol.replace("USDT", "")
            if base in STABLECOINS or base in TOO_LIQUID_BASES:
                continue

            try:
                bid: float = float(ticker["bid"])
                ask: float = float(ticker["ask"])
                volume_usd: float = float(ticker.get("volume_24h_usdt", 0))
            except (KeyError, ValueError, TypeError):
                continue

            if bid <= 0 or ask <= 0 or ask <= bid:
                continue

            # Volume filter (lower minimum — less competition on MEXC)
            if not (MEXC_MIN_VOLUME_USD <= volume_usd <= MEXC_MAX_VOLUME_USD):
                continue

            # Spread filter
            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / mid) * 10_000.0
            if spread_bps < MEXC_MIN_SPREAD_BPS:
                continue

            # Score with MEXC bonus multiplier
            base_score = spread_bps * math.sqrt(volume_usd)
            score = base_score * MEXC_SCORE_MULTIPLIER

            # Profit estimate (fee_rate=0 → all spread is profit)
            est_profit = estimate_daily_profit(
                spread_bps=spread_bps,
                order_size_usd=self.order_size_usd,
                fee_rate=MEXC_SPOT_MAKER_FEE,
                fills_per_day=ESTIMATED_FILLS_PER_DAY,
            )

            opp = PairOpportunity(
                symbol=symbol,
                exchange="mexc",
                bid=bid,
                ask=ask,
                spread_bps=spread_bps,
                volume_24h_usd=volume_usd,
                score=score,
                fee_rate=MEXC_SPOT_MAKER_FEE,
                estimated_daily_profit_usd=est_profit,
                recommended=(spread_bps >= MEXC_MIN_SPREAD_BPS),
            )
            opportunities.append(opp)

        opportunities.sort(key=lambda o: o.score, reverse=True)
        log.info(
            "scan_mexc: %d raw tickers → %d opportunities",
            len(tickers),
            len(opportunities),
        )
        self._last_scan_results["mexc"] = opportunities[:TOP_N_PER_EXCHANGE]
        return opportunities

    # ------------------------------------------------------------------
    # BTC Markets scanning
    # ------------------------------------------------------------------

    async def scan_btcmarkets(
        self,
        btcm_client: Any,
        aud_usd_rate: Optional[float] = None,
    ) -> List[PairOpportunity]:
        """
        Scan BTC Markets AUD-quoted pairs and return scored opportunities.

        All prices are normalised to USD (using ``aud_usd_rate``) before
        scoring so that opportunities can be compared with USDT-quoted venues.

        Key properties of BTC Markets
        ------------------------------
        - Maker fee is *negative* (-0.0005) — the exchange pays you per fill.
        - AUD pairs face less automated competition than USDT equivalents.
        - Wider spreads are common even on BTC-AUD and ETH-AUD.

        Scoring
        -------
        ``spread_bps × sqrt(volume_24h_usd_equiv) × BTCM_SCORE_MULTIPLIER``

        The ×2.0 multiplier reflects that a maker rebate generates revenue
        even with a zero-spread quote — any additional spread is pure bonus.
        ``fee_rate`` is set to -0.0005 in the returned ``PairOpportunity``;
        a negative value signals to downstream strategies that this venue pays
        a rebate.

        Parameters
        ----------
        btcm_client : BTCMarketsClient
            Authenticated BTC Markets client with a ``get_markets()`` coroutine
            returning a list of dicts with keys:
            ``marketId``, ``bestBid``, ``bestAsk``, ``volume24h``.
        aud_usd_rate : float | None
            Override the instance AUD/USD rate for this call.

        Returns
        -------
        list[PairOpportunity]
            Filtered, scored, sorted opportunities (best first).
        """
        rate = aud_usd_rate if aud_usd_rate is not None else self.aud_usd_rate
        log.info(
            "scan_btcmarkets: fetching markets (AUD/USD=%.4f) …", rate
        )
        try:
            markets: List[Dict[str, Any]] = await btcm_client.get_markets()
        except Exception as exc:
            log.error("scan_btcmarkets: failed to fetch markets — %s", exc)
            return []

        opportunities: List[PairOpportunity] = []

        for market in markets:
            market_id: str = market.get("marketId", "")

            # Only AUD-quoted pairs we target
            if not market_id.endswith("-AUD"):
                continue

            try:
                bid_aud: float = float(market["bestBid"])
                ask_aud: float = float(market["bestAsk"])
                volume_aud: float = float(market.get("volume24h", 0))
            except (KeyError, ValueError, TypeError):
                continue

            if bid_aud <= 0 or ask_aud <= 0 or ask_aud <= bid_aud:
                continue

            # Convert to USD equivalent for cross-venue comparison
            bid_usd = bid_aud * rate
            ask_usd = ask_aud * rate
            volume_usd = volume_aud * rate

            mid_usd = (bid_usd + ask_usd) / 2.0
            spread_bps = ((ask_usd - bid_usd) / mid_usd) * 10_000.0

            # BTC Markets pairs typically have wider spreads; lower filter
            if spread_bps < 10.0 or volume_usd < 5_000.0:
                continue

            # Score with BTC Markets bonus multiplier (negative fee = free money)
            base_score = spread_bps * math.sqrt(volume_usd)
            score = base_score * BTCM_SCORE_MULTIPLIER

            # Profit estimate: negative fee_rate means rebate is income even at 0 spread
            est_profit = estimate_daily_profit(
                spread_bps=spread_bps,
                order_size_usd=self.order_size_usd,
                fee_rate=BTCM_MAKER_FEE,   # negative → rebate boosts profit
                fills_per_day=ESTIMATED_FILLS_PER_DAY,
            )

            # Use USD-normalised prices in the opportunity object
            opp = PairOpportunity(
                symbol=market_id,           # e.g. "BTC-AUD"
                exchange="btcmarkets",
                bid=bid_usd,                # USD-equivalent bid
                ask=ask_usd,                # USD-equivalent ask
                spread_bps=spread_bps,
                volume_24h_usd=volume_usd,
                score=score,
                fee_rate=BTCM_MAKER_FEE,   # negative signals rebate to strategies
                estimated_daily_profit_usd=est_profit,
                recommended=True,           # BTC Markets is always preferred (rebate)
            )
            opportunities.append(opp)

        opportunities.sort(key=lambda o: o.score, reverse=True)
        log.info(
            "scan_btcmarkets: %d markets → %d opportunities",
            len(markets),
            len(opportunities),
        )
        self._last_scan_results["btcmarkets"] = opportunities[:TOP_N_PER_EXCHANGE]
        return opportunities

    # ------------------------------------------------------------------
    # Unified scan_all (overrides base)
    # ------------------------------------------------------------------

    async def scan_all(
        self,
        exchange_clients: Dict[str, Any],
    ) -> List[PairOpportunity]:
        """
        Scan all configured exchanges and return the top combined opportunities.

        Calls the original ``scan_all_exchanges`` (for Bybit / Kraken /
        Coinbase) then adds MEXC and BTC Markets if their clients are provided.
        Results are merged and re-ranked by score descending, returning the
        top ``TOP_N_COMBINED`` (20) opportunities.

        Parameters
        ----------
        exchange_clients : dict[str, Any]
            Map of ``exchange_name`` → client instance.  Recognised keys:
            ``"bybit"``, ``"mexc"``, ``"btcmarkets"``, ``"kraken"``,
            ``"coinbase"``.  Unknown keys are ignored.

        Returns
        -------
        list[PairOpportunity]
            Top 20 opportunities across all venues, ranked by score.
        """
        tasks = []

        # Base scanner handles bybit / kraken / coinbase
        base_clients = {
            k: v for k, v in exchange_clients.items()
            if k in ("bybit", "kraken", "coinbase")
        }

        # New venue clients
        mexc_client = exchange_clients.get("mexc")
        btcm_client = exchange_clients.get("btcmarkets")

        # Build coroutines to run concurrently
        coros = []

        if base_clients:
            coros.append(self._scan_base_venues(base_clients))

        if mexc_client is not None:
            coros.append(self.scan_mexc(mexc_client))

        if btcm_client is not None:
            coros.append(self.scan_btcmarkets(btcm_client))

        if not coros:
            log.warning("scan_all: no exchange clients provided")
            return []

        results = await asyncio.gather(*coros, return_exceptions=True)

        all_opps: List[PairOpportunity] = []
        for result in results:
            if isinstance(result, Exception):
                log.error("scan_all: a scanner raised — %s", result)
            elif isinstance(result, list):
                all_opps.extend(result)

        # Re-rank everything by score
        all_opps.sort(key=lambda o: o.score, reverse=True)

        # Tag recommended flag on top-scoring opportunities
        seen_exchanges: Dict[str, int] = {}
        for opp in all_opps:
            count = seen_exchanges.get(opp.exchange, 0)
            opp.recommended = count < TOP_N_PER_EXCHANGE
            seen_exchanges[opp.exchange] = count + 1

        top = all_opps[:TOP_N_COMBINED]
        log.info(
            "scan_all: merged %d opportunities across %d venues → top %d",
            len(all_opps),
            len(exchange_clients),
            len(top),
        )
        return top

    # ------------------------------------------------------------------
    # Helper: wrap base scanner's async scan into a flat list
    # ------------------------------------------------------------------

    async def _scan_base_venues(
        self, clients: Dict[str, Any]
    ) -> List[PairOpportunity]:
        """
        Call the original scan_all_exchanges from the base class and return
        all results as a flat list.
        """
        try:
            # AltcoinPairScanner.scan_all_exchanges returns list[PairOpportunity]
            result = await super().scan_all_exchanges(clients)
            return result if isinstance(result, list) else []
        except AttributeError:
            # Fallback: if base class method is named differently, try scan()
            all_opps: List[PairOpportunity] = []
            for exchange, client in clients.items():
                try:
                    opps = await super().scan(client, exchange)
                    all_opps.extend(opps)
                except Exception as exc:
                    log.error("_scan_base_venues(%s): %s", exchange, exc)
            return all_opps

    # ------------------------------------------------------------------
    # Per-exchange breakdown
    # ------------------------------------------------------------------

    def get_recommended_by_exchange(self) -> Dict[str, List[PairOpportunity]]:
        """
        Return top-5 opportunities grouped by exchange from the last scan.

        This is populated by ``scan_all``, ``scan_mexc``, and
        ``scan_btcmarkets`` — call one of those first.

        Returns
        -------
        dict[str, list[PairOpportunity]]
            Keys are exchange names; values are up to 5 opportunities,
            ranked by score.

        Example
        -------
        >>> recs = scanner.get_recommended_by_exchange()
        >>> for exchange, opps in recs.items():
        ...     print(exchange, [o.symbol for o in opps])
        """
        return {
            exchange: sorted(opps, key=lambda o: o.score, reverse=True)[
                :TOP_N_PER_EXCHANGE
            ]
            for exchange, opps in self._last_scan_results.items()
            if opps
        }

    # ------------------------------------------------------------------
    # Convenience: update AUD/USD rate
    # ------------------------------------------------------------------

    def set_aud_usd_rate(self, rate: float) -> None:
        """
        Update the AUD→USD conversion rate used for BTC Markets price
        normalisation.

        Parameters
        ----------
        rate : float
            Current AUD/USD exchange rate (e.g. 0.62 → $1 AUD ≈ $0.62 USD).
        """
        if rate <= 0:
            raise ValueError(f"aud_usd_rate must be positive, got {rate}")
        self.aud_usd_rate = rate
        log.debug("AUD/USD rate updated to %.4f", rate)
