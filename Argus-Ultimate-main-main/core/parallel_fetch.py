"""
Parallel OHLCV fetcher — fetches multiple symbols and timeframes
concurrently using asyncio.gather, replacing the sequential per-symbol loop.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


async def fetch_ohlcv_parallel(
    exchange: Any,
    symbols: List[str],
    timeframe: str = "1m",
    limit: int = 150,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch OHLCV for all symbols concurrently.

    Returns
    -------
    dict mapping symbol -> DataFrame (or None on error)
    """
    async def _fetch(symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
        try:
            raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw or len(raw) < 80:
                return symbol, None
            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            return symbol, df
        except Exception as exc:
            logger.debug("fetch_ohlcv_parallel %s %s: %s", symbol, timeframe, exc)
            return symbol, None

    results = await asyncio.gather(*[_fetch(s) for s in symbols])
    return dict(results)


async def fetch_mtf_parallel(
    exchange: Any,
    symbol: str,
    timeframes: List[Tuple[str, int]],
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch multiple timeframes for a single symbol concurrently.

    Parameters
    ----------
    timeframes : list of (timeframe_str, limit) tuples
        e.g. [("1h", 100), ("4h", 60)]

    Returns
    -------
    dict mapping timeframe_str -> DataFrame (or None on error)
    """
    async def _fetch(tf: str, lim: int) -> Tuple[str, Optional[pd.DataFrame]]:
        try:
            raw = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
            if not raw or len(raw) < 51:
                return tf, None
            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            return tf, df
        except Exception as exc:
            logger.debug("fetch_mtf_parallel %s %s: %s", symbol, tf, exc)
            return tf, None

    results = await asyncio.gather(*[_fetch(tf, lim) for tf, lim in timeframes])
    return dict(results)
