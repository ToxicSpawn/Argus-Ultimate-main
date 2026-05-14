#!/usr/bin/env python3
"""
Fetch extended market data: 6+ months of 15-min candles + order book + funding rates + open interest.

This script fetches:
1. 6+ months of 15-min candles (more granular than 4h)
2. Order book snapshots (bid/ask depth)
3. Funding rates (perpetual futures)
4. Open interest data
"""

import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Bybit API
BASE_URL = "https://api.bybit.com"

# Top symbols to fetch
SYMBOLS = [
    # Major
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT",
    # Large Cap
    "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "FILUSDT", "ICPUSDT", "AAVEUSDT",
    # Mid Cap
    "GRTUSDT", "SNXUSDT", "COMPUSDT", "SANDUSDT", "MANAUSDT",
    "CHZUSDT", "AXSUSDT", "THETAUSDT", "SUIUSDT", "TIAUSDT",
    # Newer
    "SEIUSDT", "JUPUSDT", "JTOUSDT", "WIFUSDT", "POPCATUSDT",
]


def fetch_klines(symbol: str, interval: str = "15", start_ts: Optional[int] = None, 
                 end_ts: Optional[int] = None, limit: int = 1000) -> List:
    """Fetch kline/candle data from Bybit."""
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_ts:
        params["start"] = start_ts
    if end_ts:
        params["end"] = end_ts
    
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
        data = resp.json()
        
        if data["retCode"] != 0:
            logger.warning(f"API error for {symbol}: {data.get('retMsg', 'Unknown error')}")
            return []
        
        return data["result"]["list"]
    except Exception as e:
        logger.warning(f"Failed to fetch klines for {symbol}: {e}")
        return []


def fetch_extended_klines(symbol: str, interval: str = "15", months: int = 6) -> List:
    """Fetch extended kline data spanning multiple months."""
    all_klines = []
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=months * 30)
    
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    current_start = start_ts
    
    while current_start < end_ts:
        klines = fetch_klines(symbol, interval, current_start, end_ts, limit=1000)
        
        if not klines:
            break
        
        all_klines.extend(klines)
        
        # Move to next batch (klines are returned newest first)
        oldest_ts = int(klines[-1][0])
        if oldest_ts <= current_start:
            break
        current_start = oldest_ts
        
        time.sleep(0.1)  # Rate limiting
    
    # Sort by timestamp (oldest first)
    all_klines.sort(key=lambda x: int(x[0]))
    
    # Remove duplicates
    seen = set()
    unique_klines = []
    for k in all_klines:
        ts = k[0]
        if ts not in seen:
            seen.add(ts)
            unique_klines.append(k)
    
    return unique_klines


def fetch_order_book(symbol: str, limit: int = 200) -> Optional[Dict]:
    """Fetch order book snapshot."""
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": limit,
    }
    
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/orderbook", params=params, timeout=10)
        data = resp.json()
        
        if data["retCode"] != 0:
            return None
        
        return data["result"]
    except Exception as e:
        logger.warning(f"Failed to fetch order book for {symbol}: {e}")
        return None


def fetch_funding_rate_history(symbol: str, limit: int = 100) -> List:
    """Fetch funding rate history."""
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": limit,
    }
    
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/funding/history", params=params, timeout=10)
        data = resp.json()
        
        if data["retCode"] != 0:
            return []
        
        return data["result"]["list"]
    except Exception as e:
        logger.warning(f"Failed to fetch funding rates for {symbol}: {e}")
        return []


def fetch_open_interest(symbol: str, interval: str = "1h", limit: int = 100) -> List:
    """Fetch open interest history."""
    params = {
        "category": "linear",
        "symbol": symbol,
        "intervalTime": interval,
        "limit": limit,
    }
    
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/open-interest", params=params, timeout=10)
        data = resp.json()
        
        if data["retCode"] != 0:
            return []
        
        return data["result"]["list"]
    except Exception as e:
        logger.warning(f"Failed to fetch open interest for {symbol}: {e}")
        return []


def fetch_tickers(symbol: str) -> Optional[Dict]:
    """Fetch ticker information (includes mark price, index price, etc.)."""
    params = {
        "category": "linear",
        "symbol": symbol,
    }
    
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers", params=params, timeout=10)
        data = resp.json()
        
        if data["retCode"] != 0:
            return None
        
        return data["result"]["list"][0] if data["result"]["list"] else None
    except Exception as e:
        logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
        return None


def main():
    """Fetch all extended market data."""
    logger.info("="*60)
    logger.info("FETCHING EXTENDED MARKET DATA (6+ months, 15-min)")
    logger.info("="*60)
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    all_data = {}
    
    for i, symbol in enumerate(SYMBOLS):
        logger.info(f"\n[{i+1}/{len(SYMBOLS)}] {symbol}")
        
        # Fetch 6+ months of 15-min candles
        logger.info(f"  Fetching 15-min candles (6+ months)...")
        klines = fetch_extended_klines(symbol, interval="15", months=6)
        logger.info(f"    Got {len(klines)} candles")
        
        if len(klines) < 1000:
            logger.warning(f"    Skipping {symbol}: insufficient data")
            continue
        
        # Convert to DataFrame format
        ohlcv = []
        for k in klines:
            ohlcv.append({
                'timestamp': int(k[0]),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'turnover': float(k[6]) if len(k) > 6 else 0,
            })
        
        # Fetch order book
        logger.info(f"  Fetching order book...")
        order_book = fetch_order_book(symbol, limit=200)
        
        # Fetch funding rates
        logger.info(f"  Fetching funding rates...")
        funding_rates = fetch_funding_rate_history(symbol, limit=100)
        
        # Fetch open interest
        logger.info(f"  Fetching open interest...")
        open_interest = fetch_open_interest(symbol, interval="1h", limit=100)
        
        # Fetch ticker
        logger.info(f"  Fetching ticker...")
        ticker = fetch_tickers(symbol)
        
        # Store all data
        all_data[symbol] = {
            'ohlcv_15m': ohlcv,
            'order_book': order_book,
            'funding_rates': funding_rates,
            'open_interest': open_interest,
            'ticker': ticker,
            'interval': '15',
            'fetched_at': datetime.now().isoformat(),
            'source': 'bybit',
        }
        
        logger.info(f"  Order book: {'OK' if order_book else 'FAILED'}")
        logger.info(f"  Funding rates: {len(funding_rates)} entries")
        logger.info(f"  Open interest: {len(open_interest)} entries")
        logger.info(f"  Ticker: {'OK' if ticker else 'FAILED'}")
        
        time.sleep(0.2)  # Rate limiting
    
    # Save all data
    output_path = output_dir / "extended_market_data.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FETCH COMPLETE")
    logger.info("="*60)
    logger.info(f"Symbols: {len(all_data)}")
    
    for symbol, data in all_data.items():
        candles = len(data['ohlcv_15m'])
        logger.info(f"  {symbol}: {candles} candles")
    
    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
