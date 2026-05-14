"""
Historical OHLCV Data Loader — fetches months of real candle data on startup.

This is the single most impactful change for ARGUS performance:
without real historical data, the evolver/scanner optimize on synthetic
market microstructure (close±0.1%, constant volume) which produces
strategies that don't work in the real market.

Features:
- Paginated fetch from any ccxt exchange (Kraken default)
- Disk cache in data/historical/ to survive restarts
- Staleness check: re-fetches if cache >24h old
- Returns numpy arrays ready for scanner/evolver consumption
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/historical")
_CACHE_STALENESS_HOURS = 24


def _cache_path(symbol: str, timeframe: str) -> Path:
    """Get disk cache path for a symbol/timeframe pair."""
    safe_sym = symbol.replace("/", "_").replace(":", "_")
    return _CACHE_DIR / f"{safe_sym}_{timeframe}.json"


def _load_from_cache(symbol: str, timeframe: str) -> Optional[Dict[str, np.ndarray]]:
    """Load cached OHLCV from disk. Returns None if stale or missing."""
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        cached_at = data.get("cached_at", 0)
        age_hours = (time.time() - cached_at) / 3600
        if age_hours > _CACHE_STALENESS_HOURS:
            logger.info("Historical cache stale for %s (%.1fh old)", symbol, age_hours)
            return None

        candles = data.get("candles", [])
        if not candles:
            return None

        return {
            "timestamp": np.array([c[0] for c in candles], dtype=float),
            "open": np.array([c[1] for c in candles], dtype=float),
            "high": np.array([c[2] for c in candles], dtype=float),
            "low": np.array([c[3] for c in candles], dtype=float),
            "close": np.array([c[4] for c in candles], dtype=float),
            "volume": np.array([c[5] for c in candles], dtype=float),
        }
    except Exception as e:
        logger.warning("Failed to load historical cache for %s: %s", symbol, e)
        return None


def _save_to_cache(symbol: str, timeframe: str, candles: List[List[float]]) -> None:
    """Save OHLCV candles to disk cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(symbol, timeframe)
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "cached_at": time.time(),
            "candle_count": len(candles),
            "candles": candles,
        }
        path.write_text(json.dumps(data))
        logger.info("Cached %d candles for %s to %s", len(candles), symbol, path)
    except Exception as e:
        logger.warning("Failed to cache historical data for %s: %s", symbol, e)


async def fetch_historical_ohlcv(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    lookback_hours: int = 4380,
    max_per_request: int = 720,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Fetch historical OHLCV with pagination. Returns numpy arrays.

    Args:
        exchange: ccxt exchange instance (async)
        symbol: e.g. "BTC/USD"
        timeframe: "1h", "4h", "1d" etc.
        lookback_hours: how far back to fetch (default 6 months)
        max_per_request: max candles per API call (Kraken allows 720)

    Returns:
        Dict with keys: timestamp, open, high, low, close, volume (numpy arrays)
        or None on failure.
    """
    # Try disk cache first
    cached = _load_from_cache(symbol, timeframe)
    if cached is not None:
        logger.info("Loaded %d cached candles for %s", len(cached["close"]), symbol)
        return cached

    # Calculate pagination
    tf_ms = _timeframe_to_ms(timeframe)
    if tf_ms <= 0:
        return None

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (lookback_hours * 3600 * 1000)
    all_candles: List[List[float]] = []

    logger.info("Fetching %d hours of %s candles for %s...", lookback_hours, timeframe, symbol)

    since = start_ms
    retries = 0
    max_retries = 3

    while since < now_ms:
        try:
            fetch_fn = getattr(exchange, "fetch_ohlcv", None)
            if fetch_fn is None:
                # Try unwrapping
                inner = getattr(exchange, "_exchange", None)
                if inner:
                    fetch_fn = getattr(inner, "fetch_ohlcv", None)
            if fetch_fn is None:
                logger.warning("Exchange has no fetch_ohlcv for %s", symbol)
                break

            candles = await asyncio.wait_for(
                fetch_fn(symbol, timeframe=timeframe, since=since, limit=max_per_request),
                timeout=15.0,
            )

            if not candles or not isinstance(candles, list):
                break

            all_candles.extend(candles)
            # Move since to after last candle
            last_ts = candles[-1][0]
            since = last_ts + tf_ms

            if len(candles) < max_per_request:
                break  # no more data

            retries = 0
            await asyncio.sleep(0.5)  # rate limit courtesy

        except asyncio.TimeoutError:
            retries += 1
            if retries >= max_retries:
                logger.warning("Timeout fetching historical %s for %s after %d retries",
                               timeframe, symbol, retries)
                break
            await asyncio.sleep(2.0)
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logger.warning("Error fetching historical %s for %s: %s", timeframe, symbol, e)
                break
            await asyncio.sleep(2.0)

    if not all_candles:
        logger.warning("No historical candles fetched for %s", symbol)
        return None

    # Deduplicate by timestamp (pagination can overlap)
    seen_ts = set()
    unique_candles = []
    for c in all_candles:
        ts = c[0]
        if ts not in seen_ts:
            seen_ts.add(ts)
            unique_candles.append(c)
    unique_candles.sort(key=lambda c: c[0])

    # Save to disk cache
    _save_to_cache(symbol, timeframe, unique_candles)

    result = {
        "timestamp": np.array([c[0] for c in unique_candles], dtype=float),
        "open": np.array([c[1] for c in unique_candles], dtype=float),
        "high": np.array([c[2] for c in unique_candles], dtype=float),
        "low": np.array([c[3] for c in unique_candles], dtype=float),
        "close": np.array([c[4] for c in unique_candles], dtype=float),
        "volume": np.array([c[5] for c in unique_candles], dtype=float),
    }

    logger.info("Loaded %d historical %s candles for %s (%.1f months)",
                len(unique_candles), timeframe, symbol,
                len(unique_candles) / (24 * 30))
    return result


async def load_all_historical(
    exchange,
    symbols: List[str],
    timeframe: str = "1h",
    lookback_hours: int = 4380,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load historical OHLCV for all symbols. Returns {symbol: {open,high,low,close,volume}}."""
    result = {}
    for symbol in symbols:
        try:
            data = await fetch_historical_ohlcv(exchange, symbol, timeframe, lookback_hours)
            if data is not None and len(data["close"]) >= 50:
                result[symbol] = data
        except Exception as e:
            logger.warning("Historical load failed for %s: %s", symbol, e)
    logger.info("Historical data loaded for %d/%d symbols", len(result), len(symbols))
    return result


def load_all_from_cache(
    symbols: List[str],
    timeframe: str = "1h",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Synchronous cache-only loader (for non-async contexts like ComponentRegistry init)."""
    result = {}
    for symbol in symbols:
        cached = _load_from_cache(symbol, timeframe)
        if cached is not None and len(cached["close"]) >= 50:
            result[symbol] = cached
    if result:
        logger.info("Loaded historical cache for %d symbols", len(result))
    return result


def _timeframe_to_ms(tf: str) -> int:
    """Convert timeframe string to milliseconds."""
    multipliers = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
        "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
        "1d": 86_400_000, "1w": 604_800_000,
    }
    return multipliers.get(tf, 0)
