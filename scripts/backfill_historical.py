#!/usr/bin/env python
"""
ARGUS Historical Data Backfill — fetch OHLCV from exchanges for all trading pairs.

Downloads historical candles from Kraken (primary) and Coinbase (secondary)
for all 11 ARGUS trading pairs, storing as Parquet via HistoricalDataIngester.

Usage:
    py scripts/backfill_historical.py [--exchange kraken] [--timeframes 1h 15m] [--days 365]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.ccxt_data_provider import get_ccxt_exchange
from data.historical_ingester import HistoricalDataIngester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_historical")

DEFAULT_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LTC/USD",
    "AVAX/USD", "DOT/USD", "LINK/USD", "UNI/USD", "ADA/USD", "DOGE/USD",
]

DEFAULT_TIMEFRAMES = ["1h", "15m"]

# ccxt timeframe → milliseconds
TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def fetch_ohlcv_chunked(
    exchange: Any,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 720,
    rate_limit_s: float = 1.0,
) -> pd.DataFrame:
    """Fetch OHLCV in chunks, respecting rate limits."""
    tf_ms = TF_MS.get(timeframe, 3_600_000)
    all_rows: List[list] = []
    cursor = since_ms

    while cursor < until_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        except Exception as e:
            logger.warning("fetch_ohlcv failed for %s %s at %d: %s", symbol, timeframe, cursor, e)
            time.sleep(rate_limit_s * 3)
            # Skip this chunk
            cursor += tf_ms * limit
            continue

        if not candles:
            break

        all_rows.extend(candles)
        last_ts = candles[-1][0]

        # If we got fewer than requested, we've reached the end
        if len(candles) < limit:
            break

        cursor = last_ts + tf_ms
        time.sleep(rate_limit_s)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Filter to requested range
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] <= until_ms)]
    return df


def backfill(
    exchange_id: str = "kraken",
    pairs: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    days: int = 365,
    output_dir: str = "data/historical",
) -> Dict[str, int]:
    """Run the backfill for all pairs and timeframes. Returns row counts."""
    pairs = pairs or DEFAULT_PAIRS
    timeframes = timeframes or DEFAULT_TIMEFRAMES

    exchange = get_ccxt_exchange(exchange_id)
    exchange.load_markets()

    ingester = HistoricalDataIngester(exchange_id=exchange_id, data_dir=output_dir)

    until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    results: Dict[str, int] = {}

    for symbol in pairs:
        if symbol not in exchange.markets:
            logger.warning("Symbol %s not available on %s, skipping", symbol, exchange_id)
            results[symbol] = 0
            continue

        for tf in timeframes:
            key = f"{symbol}_{tf}"
            logger.info("Backfilling %s %s from %s (%d days)...", symbol, tf, exchange_id, days)

            # Check existing data to avoid re-downloading
            existing = ingester.load(symbol, tf)
            effective_since = since_ms
            if existing is not None and not existing.empty and "timestamp" in existing.columns:
                last_existing = int(existing["timestamp"].max())
                if last_existing > effective_since:
                    effective_since = last_existing + TF_MS.get(tf, 3_600_000)
                    logger.info("  Resuming from %s (have %d existing rows)",
                                datetime.fromtimestamp(effective_since / 1000, tz=timezone.utc).isoformat(),
                                len(existing))

            if effective_since >= until_ms:
                logger.info("  Already up to date (%d rows)", len(existing) if existing is not None else 0)
                results[key] = len(existing) if existing is not None else 0
                continue

            df = fetch_ohlcv_chunked(exchange, symbol, tf, effective_since, until_ms)

            if df.empty:
                logger.warning("  No data returned for %s %s", symbol, tf)
                results[key] = len(existing) if existing is not None else 0
                continue

            # Merge with existing
            if existing is not None and not existing.empty:
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"], keep="first")
                df = df.sort_values("timestamp").reset_index(drop=True)

            ingester.save(symbol, tf, df)
            results[key] = len(df)
            logger.info("  Saved %d candles for %s %s", len(df), symbol, tf)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS Historical Data Backfill")
    parser.add_argument("--exchange", default="kraken", help="Exchange to fetch from (default: kraken)")
    parser.add_argument("--pairs", nargs="*", default=None, help="Trading pairs (default: all 11)")
    parser.add_argument("--timeframes", nargs="*", default=None, help="Timeframes (default: 1h 15m)")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    parser.add_argument("--output-dir", default="data/historical", help="Output directory")
    args = parser.parse_args()

    logger.info("ARGUS Historical Data Backfill")
    logger.info("Exchange: %s | Days: %d | Pairs: %s | Timeframes: %s",
                args.exchange, args.days, args.pairs or "all 11", args.timeframes or "1h,15m")

    results = backfill(
        exchange_id=args.exchange,
        pairs=args.pairs,
        timeframes=args.timeframes,
        days=args.days,
        output_dir=args.output_dir,
    )

    total = sum(results.values())
    logger.info("Backfill complete: %d total candles across %d datasets", total, len(results))
    for key, count in sorted(results.items()):
        logger.info("  %s: %d candles", key, count)


if __name__ == "__main__":
    main()
