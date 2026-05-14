"""Batch 1 — Async parallel OHLCV fetcher.

Fetches OHLCV candles for multiple symbols/timeframes concurrently using
asyncio + ccxt.async_support, with per-symbol semaphore throttling and
exponential-backoff retries.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt_async
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class AsyncOHLCVFetcher:
    """Fetches OHLCV data for multiple symbols in parallel."""

    def __init__(
        self,
        exchange_id: str = "kraken",
        api_key: str = "",
        api_secret: str = "",
        max_concurrent: int = 8,
        max_retries: int = 4,
        base_delay: float = 1.0,
    ) -> None:
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._exchange: Optional[ccxt_async.Exchange] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _get_exchange(self) -> ccxt_async.Exchange:
        if self._exchange is None:
            cls = getattr(ccxt_async, self._exchange_id)
            self._exchange = cls(
                {
                    "apiKey": self._api_key,
                    "secret": self._api_secret,
                    "enableRateLimit": True,
                }
            )
            await self._exchange.load_markets()
        return self._exchange

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    async def _fetch_one(
        self,
        exchange: ccxt_async.Exchange,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[int],
    ) -> Tuple[str, str, pd.DataFrame]:
        async with self._sem:
            for attempt in range(self._max_retries):
                try:
                    raw = await exchange.fetch_ohlcv(
                        symbol, timeframe, since=since, limit=limit
                    )
                    df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df.set_index("timestamp", inplace=True)
                    return symbol, timeframe, df
                except Exception as exc:  # noqa: BLE001
                    if attempt == self._max_retries - 1:
                        logger.error(
                            "OHLCV fetch failed %s/%s after %d attempts: %s",
                            symbol,
                            timeframe,
                            self._max_retries,
                            exc,
                        )
                        return symbol, timeframe, pd.DataFrame(columns=OHLCV_COLUMNS[1:])
                    delay = self._base_delay * (2**attempt)
                    logger.warning(
                        "Retrying %s/%s (attempt %d) in %.1fs", symbol, timeframe, attempt + 1, delay
                    )
                    await asyncio.sleep(delay)
        # unreachable but satisfies type checker
        return symbol, timeframe, pd.DataFrame(columns=OHLCV_COLUMNS[1:])

    async def fetch_many(
        self,
        pairs: List[Tuple[str, str]],
        limit: int = 500,
        since: Optional[int] = None,
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Fetch (symbol, timeframe) pairs concurrently.

        Returns a dict keyed by (symbol, timeframe).
        """
        exchange = await self._get_exchange()
        tasks = [
            asyncio.create_task(self._fetch_one(exchange, sym, tf, limit, since))
            for sym, tf in pairs
        ]
        results: Dict[Tuple[str, str], pd.DataFrame] = {}
        for coro in asyncio.as_completed(tasks):
            sym, tf, df = await coro
            results[(sym, tf)] = df
            logger.debug("Fetched %d candles for %s/%s", len(df), sym, tf)
        return results

    # ------------------------------------------------------------------
    # Convenience sync wrapper
    # ------------------------------------------------------------------

    def fetch_many_sync(
        self,
        pairs: List[Tuple[str, str]],
        limit: int = 500,
        since: Optional[int] = None,
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Blocking wrapper for use from synchronous contexts."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.fetch_many(pairs, limit, since))
        finally:
            loop.run_until_complete(self.close())
            loop.close()
