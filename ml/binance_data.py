"""
Binance Historical Data Downloader for Backtesting.

Downloads real OHLCV data from Binance API for backtesting.
Free: 1500 candles per request.
"""

import asyncio
import json
import logging
import os
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Binance API
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_UKLINES = "https://api.binance.com/fapi/v1/klines"  # USDT futures
BINANCE_DLINES = "https://api.binance.com/dapi/v1/klines"  # COIN-M futures


@dataclass
class OHLCV:
    """OHLCV candle."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BinanceData:
    """Download historical data from Binance."""

    TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w", "1mo"]

    def __init__(self):
        self.cache_dir = Path("data/binance_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start: int = None,
        end: int = None,
        limit: int = 1500,
        futures: bool = False,
    ) -> list[OHLCV]:
        """
        Fetch klines from Binance.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Timeframe (1m, 5m, 1h, 4h, 1d, etc.)
            start: Start timestamp (ms)
            end: End timestamp (ms) 
            limit: Max candles (max 1500)
            futures: Use USDT futures

        Returns:
            List of OHLCV candles
        """
        if interval not in self.TIMEFRAMES:
            raise ValueError(f"Invalid interval: {interval}")

        # Default: last 30 days
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        if end is None:
            end = now
        if start is None:
            start = end - (30 * 24 * 60 * 60 * 1000)  # 30 days

        url = BINANCE_UKLINES if futures else BINANCE_KLINES
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start,
            "endTime": end,
            "limit": limit,
        }

        logger.info(f"Fetching {symbol} {interval} from {start} to {end}...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        raise Exception(f"API error: {resp.status}")

                    data = await resp.json()

            candles = []
            for d in data:
                candles.append(OHLCV(
                    timestamp=int(d[0]),
                    open=float(d[1]),
                    high=float(d[2]),
                    low=float(d[3]),
                    close=float(d[4]),
                    volume=float(d[5]),
                    close_time=int(d[6]),
                    quote_volume=float(d[7]),
                    trades=int(d[8]),
                ))

            logger.info(f"Got {len(candles)} candles")
            return candles

        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return []

    async def download_symbol(
        self,
        symbol: str = "BTCUSDT",
        intervals: list = None,
        days: int = 365,
    ) -> dict:
        """
        Download multiple timeframes for a symbol.
        
        Args:
            symbol: Trading pair
            intervals: List of timeframes
            days: Days of history
            
        Returns:
            Dict of interval -> list of OHLCV
        """
        if intervals is None:
            intervals = ["1h", "4h", "1d"]

        results = {}

        for interval in intervals:
            # Calculate timestamps
            now = int(datetime.now(timezone.utc).timestamp() * 1000)
            start = now - (days * 24 * 60 * 60 * 1000)

            candles = await self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start=start,
                end=now,
                limit=1500,
            )

            if candles:
                results[interval] = candles

            await asyncio.sleep(0.5)  # Rate limit

        return results

    def save_cache(self, symbol: str, data: dict):
        """Save to cache."""
        path = self.cache_dir / f"{symbol}.json"
        
        serialized = {}
        for interval, candles in data.items():
            serialized[interval] = [
                c.to_dict() for c in candles
            ]

        with open(path, "w") as f:
            json.dump(serialized, f)
        
        logger.info(f"Cached {symbol} to {path}")

    def load_cache(self, symbol: str) -> dict:
        """Load from cache."""
        path = self.cache_dir / f"{symbol}.json"
        
        if not path.exists():
            return {}

        with open(path) as f:
            data = json.load(f)

        result = {}
        for interval, candles in data.items():
            result[interval] = [
                OHLCV(
                    timestamp=c["timestamp"],
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c["volume"],
                    close_time=0,
                    quote_volume=0,
                    trades=0,
                )
                for c in candles
            ]

        return result


async def download_backtest_data():
    """Download data for backtesting."""
    from ml.multi_agent_voting import get_multi_agent_signal

    binance = BinanceData()

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    all_data = {}

    for symbol in symbols:
        print(f"\n=== Downloading {symbol} ===")
        data = await binance.download_symbol(symbol, ["1h", "4h"], days=180)
        
        if data:
            all_data[symbol] = data
            binance.save_cache(symbol, data)

    # Test signals on BTC data
    print("\n=== Testing Multi-Agent Voting ===")
    
    btc_data = all_data.get("BTCUSDT", {})
    ohlcv = btc_data.get("1h", [])
    
    if ohlcv:
        # Convert to list format
        ohlcv_list = [
            {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in ohlcv[-100:]
        ]

        market_data = {
            "headlines": ["BTC shows strong momentum", "Institutional interest grows"],
            "fear_greed": 65,
            "funding_rate": 0.0003,
            "oi_change": 0.05,
        }

        result = await get_multi_agent_signal("BTCUSDT", ohlcv_list, market_data)

        print(f"Direction: {result.direction}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Voting: {result.agreement_count}/3 agents agree")


if __name__ == "__main__":
    asyncio.run(download_backtest_data())