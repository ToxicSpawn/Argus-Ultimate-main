"""
services/liquidity_scanner.py — Market Liquidity Scanner for ARGUS.

Scans an exchange for the most liquid trading pairs by combining:
  - 24-hour volume (in USD)
  - Bid-ask spread (percentage)
  - Order book depth (total bid + ask value in top N levels)
  - Order book imbalance (abs(bid_value - ask_value) / total)

A composite liquidity_score is computed as:
    score = w_volume * vol_score
          + w_spread * (1 - spread_score)   # lower spread = better
          + w_depth * depth_score
          + w_imbalance * (1 - imbalance_score)  # lower imbalance = better

All scores are normalised to [0, 1] relative to the scanned universe.

Usage (async)::

    scanner = LiquidityScanner(exchange_id="kraken", max_pairs=20)
    results = await scanner.scan()
    for r in results:
        print(r.symbol, r.liquidity_score)

Usage (sync helper for CLI)::

    results = run_scan_sync(exchange_id="kraken", max_pairs=20)
    print_scan_table(results)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LiquidityResult:
    """Per-symbol liquidity metrics and composite score."""

    symbol: str
    volume_usd: float          # 24-h quote volume in USD equivalent
    spread_pct: float          # (ask - bid) / mid * 100
    depth_usd: float           # sum of (price * qty) across top N bid+ask levels
    imbalance: float           # |bid_value - ask_value| / (bid_value + ask_value), in [0,1]
    liquidity_score: float     # composite, higher = more liquid
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# LiquidityScanner
# ---------------------------------------------------------------------------

class LiquidityScanner:
    """
    Async liquidity scanner.

    Parameters
    ----------
    exchange_id : str
        ccxt exchange id (e.g. "kraken", "bybit").
    quote_currencies : sequence of str
        Only scan pairs with these quote currencies.
    max_pairs : int
        Maximum number of pairs to scan (pre-filtered by 24h volume).
    min_volume_usd : float
        Minimum 24-h USD volume to include a pair.
    depth_levels : int
        Number of order book levels to use for depth/imbalance.
    batch_size : int
        Pairs to fetch per batch (throttles API usage).
    batch_delay_s : float
        Seconds to wait between batches.
    cache_ttl_s : float
        Seconds to cache results before re-scanning.
    w_volume, w_spread, w_depth, w_imbalance : float
        Weights for each component in the composite score (must sum to 1).
    """

    def __init__(
        self,
        exchange_id: str = "kraken",
        quote_currencies: Sequence[str] = ("USD",),  # Kraken uses /USD not /USDT
        max_pairs: int = 20,
        min_volume_usd: float = 50_000.0,
        depth_levels: int = 10,
        batch_size: int = 5,
        batch_delay_s: float = 1.0,
        cache_ttl_s: float = 60.0,
        w_volume: float = 0.40,
        w_spread: float = 0.30,
        w_depth: float = 0.20,
        w_imbalance: float = 0.10,
        exchange: Optional[Any] = None,  # injected exchange (for testing)
    ) -> None:
        self.exchange_id = exchange_id
        self.quote_currencies = tuple(q.upper() for q in quote_currencies)
        self.max_pairs = int(max_pairs)
        self.min_volume_usd = float(min_volume_usd)
        self.depth_levels = int(depth_levels)
        self.batch_size = int(batch_size)
        self.batch_delay_s = float(batch_delay_s)
        self.cache_ttl_s = float(cache_ttl_s)
        self.w_volume = float(w_volume)
        self.w_spread = float(w_spread)
        self.w_depth = float(w_depth)
        self.w_imbalance = float(w_imbalance)
        self._exchange = exchange  # optional pre-built exchange (for tests)
        self._cache: Optional[List[LiquidityResult]] = None
        self._cache_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan(self, force: bool = False) -> List[LiquidityResult]:
        """
        Scan the exchange for liquid pairs.

        Returns results sorted by liquidity_score descending.
        Caches results for cache_ttl_s seconds unless force=True.
        """
        now = time.time()
        if not force and self._cache and (now - self._cache_ts) < self.cache_ttl_s:
            logger.debug("LiquidityScanner: returning cached results (%d pairs)", len(self._cache))
            return self._cache

        exchange = await self._get_exchange()
        try:
            results = await self._do_scan(exchange)
        finally:
            if self._exchange is None:
                # Only close exchanges we created ourselves
                try:
                    await exchange.close()
                except Exception:
                    pass

        self._cache = results
        self._cache_ts = now
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_exchange(self) -> Any:
        """Return the injected exchange or create a new ccxt async exchange."""
        if self._exchange is not None:
            return self._exchange
        # Lazy import to avoid hard dep at module load
        try:
            import ccxt.pro as ccxt_module  # type: ignore
        except ImportError:
            import ccxt.async_support as ccxt_module  # type: ignore
        cls = getattr(ccxt_module, self.exchange_id.lower())
        return cls({"enableRateLimit": True})

    async def _do_scan(self, exchange: Any) -> List[LiquidityResult]:
        """Perform the actual scan against the exchange."""
        # --- Step 1: load markets ---
        try:
            markets = await exchange.load_markets()
        except Exception as exc:
            logger.warning("LiquidityScanner: load_markets failed: %s", exc)
            # Fall back to synchronous fetch if available
            try:
                markets = exchange.markets or {}
            except Exception:
                markets = {}

        # --- Step 2: filter to quote currency pairs ---
        candidates = [
            sym for sym, info in markets.items()
            if (
                info.get("active", True)
                and info.get("quote", "").upper() in self.quote_currencies
            )
        ]
        logger.debug("LiquidityScanner: %d candidate symbols after quote filter", len(candidates))

        if not candidates:
            return []

        # --- Step 3: fetch tickers in one call ---
        raw_metrics: List[Tuple[str, float, float, float, float]] = []
        try:
            tickers = await exchange.fetch_tickers(candidates)
        except Exception as exc:
            logger.warning("LiquidityScanner: fetch_tickers failed (%s) — falling back to individual calls", exc)
            tickers = {}
            for sym in candidates[:self.max_pairs]:
                try:
                    tickers[sym] = await exchange.fetch_ticker(sym)
                except Exception as e2:
                    logger.debug("fetch_ticker(%s) error: %s", sym, e2)

        # Filter by minimum volume and sort
        vol_filtered: List[Tuple[str, float]] = []
        for sym, ticker in tickers.items():
            quoteVolume = ticker.get("quoteVolume") or ticker.get("volume") or 0.0
            try:
                quoteVolume = float(quoteVolume)
            except (TypeError, ValueError):
                quoteVolume = 0.0
            if quoteVolume >= self.min_volume_usd:
                vol_filtered.append((sym, quoteVolume))

        vol_filtered.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in vol_filtered[: self.max_pairs]]
        logger.debug("LiquidityScanner: %d symbols above min_volume threshold", len(top_symbols))

        if not top_symbols:
            return []

        # --- Step 4: fetch order books in batches ---
        ob_data: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(top_symbols), self.batch_size):
            batch = top_symbols[i: i + self.batch_size]
            for sym in batch:
                try:
                    ob = await exchange.fetch_order_book(sym, limit=self.depth_levels)
                    ob_data[sym] = ob
                except Exception as exc:
                    logger.debug("fetch_order_book(%s) error: %s", sym, exc)
            if i + self.batch_size < len(top_symbols):
                await asyncio.sleep(self.batch_delay_s)

        # --- Step 5: compute raw metrics ---
        raw: List[Tuple[str, float, float, float, float]] = []
        for sym in top_symbols:
            ticker = tickers.get(sym, {})
            vol_usd = float(ticker.get("quoteVolume") or ticker.get("volume") or 0.0)

            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            mid = (bid + ask) / 2 if bid and ask else float(ticker.get("last") or 1.0)
            spread_pct = ((ask - bid) / mid * 100) if mid > 0 and ask > bid else 999.0

            ob = ob_data.get(sym, {})
            bids_raw = ob.get("bids", [])
            asks_raw = ob.get("asks", [])
            bid_value = sum(float(p) * float(q) for p, q in bids_raw[:self.depth_levels] if p and q)
            ask_value = sum(float(p) * float(q) for p, q in asks_raw[:self.depth_levels] if p and q)
            depth_usd = bid_value + ask_value
            total = bid_value + ask_value
            imbalance = abs(bid_value - ask_value) / total if total > 0 else 1.0

            raw.append((sym, vol_usd, spread_pct, depth_usd, imbalance))

        if not raw:
            return []

        # --- Step 6: normalise each dimension to [0, 1] ---
        vols = [r[1] for r in raw]
        spreads = [r[2] for r in raw]
        depths = [r[3] for r in raw]
        imbalances = [r[4] for r in raw]

        def _norm(values: List[float]) -> List[float]:
            lo, hi = min(values), max(values)
            if hi == lo:
                return [0.5] * len(values)
            return [(v - lo) / (hi - lo) for v in values]

        norm_vol = _norm(vols)
        norm_spread = _norm(spreads)   # lower is better → invert below
        norm_depth = _norm(depths)
        norm_imb = _norm(imbalances)   # lower is better → invert below

        results: List[LiquidityResult] = []
        for idx, (sym, vol_usd, spread_pct, depth_usd, imbalance) in enumerate(raw):
            score = (
                self.w_volume    * norm_vol[idx]
                + self.w_spread  * (1.0 - norm_spread[idx])
                + self.w_depth   * norm_depth[idx]
                + self.w_imbalance * (1.0 - norm_imb[idx])
            )
            results.append(LiquidityResult(
                symbol=sym,
                volume_usd=round(vol_usd, 2),
                spread_pct=round(spread_pct, 6),
                depth_usd=round(depth_usd, 2),
                imbalance=round(imbalance, 6),
                liquidity_score=round(score, 6),
            ))

        results.sort(key=lambda r: r.liquidity_score, reverse=True)
        logger.info(
            "LiquidityScanner: scan complete — %d results, top=%s (%.4f)",
            len(results), results[0].symbol, results[0].liquidity_score,
        )
        return results


# ---------------------------------------------------------------------------
# Convenience sync wrapper and CLI printer
# ---------------------------------------------------------------------------

def run_scan_sync(
    exchange_id: str = "kraken",
    max_pairs: int = 20,
    min_volume_usd: float = 50_000.0,
    depth_levels: int = 10,
    **kwargs: Any,
) -> List[LiquidityResult]:
    """Synchronous wrapper around LiquidityScanner.scan(). Useful for CLI scripts."""
    scanner = LiquidityScanner(
        exchange_id=exchange_id,
        max_pairs=max_pairs,
        min_volume_usd=min_volume_usd,
        depth_levels=depth_levels,
        **kwargs,
    )
    return asyncio.run(scanner.scan())


def print_scan_table(results: List[LiquidityResult], top_n: int = 20) -> None:
    """Print a simple ASCII table of liquidity scan results."""
    header = f"{'#':>4}  {'Symbol':<14}  {'Vol USD':>14}  {'Spread%':>9}  {'Depth USD':>12}  {'Imbalance':>10}  {'Score':>8}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results[:top_n], 1):
        print(
            f"{i:>4}  {r.symbol:<14}  {r.volume_usd:>14,.0f}"
            f"  {r.spread_pct:>9.4f}  {r.depth_usd:>12,.0f}"
            f"  {r.imbalance:>10.4f}  {r.liquidity_score:>8.4f}"
        )
