"""
Universe builder – select top liquid USD pairs by real volume (CCXT).

Used by adaptive universe and self-improver to expand or rank the trading universe.
Fetches markets and tickers from the exchange, filters USD-quoted pairs, sorts by
quote volume (USD) and returns top_n. Falls back to DEFAULT_LIQUID_USD on failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# Default liquid pairs when exchange fetch is not used or fails
DEFAULT_LIQUID_USD = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "LTC/USD",
    "AVAX/USD",
    "DOT/USD",
    "LINK/USD",
    "UNI/USD",
    "ADA/USD",
    "DOGE/USD",
]


@dataclass
class UniverseSelection:
    """Result of select_top_liquid_usd_pairs: list of symbol strings."""
    symbols: List[str]


def select_top_liquid_usd_pairs(
    exchange_id: str = "kraken",
    top_n: int = 10,
) -> UniverseSelection:
    """
    Return top N liquid USD pairs for the given exchange by quote (USD) volume.
    Uses CCXT fetch_markets + fetch_tickers when available; falls back to
    DEFAULT_LIQUID_USD[:top_n] on failure or when volume data is missing.
    """
    top_n = max(1, int(top_n))
    try:
        import ccxt
        Exchange = getattr(ccxt, exchange_id, None)
        if not Exchange or not callable(Exchange):
            logger.debug("universe_builder: exchange %s not found", exchange_id)
            return UniverseSelection(symbols=[s for s in DEFAULT_LIQUID_USD[:top_n] if s])
        exchange = Exchange({"enableRateLimit": True})
        markets = exchange.fetch_markets()
        # Filter: spot, active, quote = USD (or USDT if you want; we stick to USD)
        usd_pairs = []
        for m in markets:
            if not m.get("active", True):
                continue
            if m.get("type") != "spot":
                continue
            quote = str(m.get("quote", "") or "").upper()
            # Kraken uses /USD not /USDT - exclude stablecoin quotes
            if exchange_id and exchange_id.lower() == "kraken":
                if quote != "USD":
                    continue
            elif quote not in ("USD", "USDT"):
                continue
            base = str(m.get("base", "") or "").upper()
            sym = m.get("symbol") or f"{base}/{quote}"
            usd_pairs.append(sym)
        if not usd_pairs:
            return UniverseSelection(symbols=[s for s in DEFAULT_LIQUID_USD[:top_n] if s])
        # Fetch tickers for volume ranking
        tickers = exchange.fetch_tickers(usd_pairs) if usd_pairs else {}
        # Sort by quote volume (USD); fallback to base volume then symbol
        def volume_key(sym: str) -> float:
            t = tickers.get(sym) or {}
            quote_vol = float(t.get("quoteVolume") or t.get("quote_volume") or 0)
            if quote_vol > 0:
                return quote_vol
            return float(t.get("baseVolume") or t.get("base_volume") or 0)
        usd_pairs.sort(key=volume_key, reverse=True)
        symbols = [s for s in usd_pairs[:top_n] if s]
        if symbols:
            logger.debug("universe_builder: top %s by volume from %s: %s", top_n, exchange_id, symbols[:5])
            return UniverseSelection(symbols=symbols)
    except Exception as e:
        logger.debug("universe_builder select_top_liquid_usd_pairs: %s", e)
    symbols = [s for s in DEFAULT_LIQUID_USD[:top_n] if s]
    return UniverseSelection(symbols=symbols)
