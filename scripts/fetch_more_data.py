#!/usr/bin/env python3
"""
Fetch additional market data from Bybit API.

Adds more symbols to the training dataset for better ML models.

Usage:
    py scripts/fetch_more_data.py
"""

import sys
import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# Additional symbols to fetch (major cryptos + alts)
ADDITIONAL_SYMBOLS = [
    # Major
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT",
    # Large cap
    "MATICUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "FILUSDT", "ICPUSDT",
    # Mid cap
    "AAVEUSDT", "GRTUSDT", "SNXUSDT", "COMPUSDT", "MKRUSDT",
    "SANDUSDT", "MANAUSDT", "CHZUSDT", "AXSUSDT", "THETAUSDT",
    # Newer
    "SUIUSDT", "TIAUSDT", "SEIUSDT", "JUPUSDT", "JTOUSDT",
    "WIFUSDT", "BONKUSDT", "POPCATUSDT", "FLOKIUSDT", "PEPEUSDT",
]

# Existing symbols to avoid duplicates
EXISTING_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LINK/USD", "AVAX/USD", "DOT/USD", "ADA/USD"]


def fetch_bybit_klines(symbol: str, interval: str = "240", limit: int = 1000):
    """
    Fetch kline data from Bybit API.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval in minutes (60=1h, 240=4h, 1440=1d)
        limit: Number of candles to fetch (max 1000)
    
    Returns:
        List of OHLCV data or None if failed
    """
    import urllib.request
    import json
    
    url = f"https://api.bybit.com/v5/market/kline"
    params = f"category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    full_url = f"{url}?{params}"
    
    try:
        req = urllib.request.Request(full_url)
        req.add_header('User-Agent', 'Argus-Training/1.0')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        if data.get("retCode") != 0:
            logger.warning("Bybit API error for %s: %s", symbol, data.get("retMsg"))
            return None
        
        klines = data.get("result", {}).get("list", [])
        if not klines:
            logger.warning("No data for %s", symbol)
            return None
        
        # Convert to OHLCV format: [timestamp, open, high, low, close, volume]
        ohlcv = []
        for k in reversed(klines):  # Bybit returns newest first
            ohlcv.append([
                int(k[0]),           # timestamp (ms)
                float(k[1]),         # open
                float(k[2]),         # high
                float(k[3]),         # low
                float(k[4]),         # close
                float(k[5]),         # volume
            ])
        
        return ohlcv
        
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", symbol, e)
        return None


def load_existing_data():
    """Load existing training data."""
    data_path = Path("data/training_market_data.pkl")
    
    if data_path.exists():
        with open(data_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_data(data):
    """Save training data."""
    data_path = Path("data/training_market_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Saved data to %s", data_path)


def main():
    """Main fetch function."""
    logger.info("=" * 60)
    logger.info("FETCHING ADDITIONAL MARKET DATA")
    logger.info("=" * 60)
    
    # Load existing data
    existing_data = load_existing_data()
    logger.info("Existing symbols: %d", len(existing_data))
    
    # Fetch new data
    new_symbols = []
    failed_symbols = []
    total_new_candles = 0
    
    for i, symbol in enumerate(ADDITIONAL_SYMBOLS):
        logger.info("[%d/%d] Fetching %s...", i + 1, len(ADDITIONAL_SYMBOLS), symbol)
        
        # Fetch 4h candles
        ohlcv = fetch_bybit_klines(symbol, interval="240", limit=1000)
        
        if ohlcv and len(ohlcv) >= 100:
            # Convert symbol format (BTCUSDT -> BTC/USD)
            base = symbol.replace("USDT", "")
            display_name = f"{base}/USD"
            
            existing_data[display_name] = {
                "ohlcv": ohlcv,
                "interval": "4h",
                "fetched_at": datetime.now().isoformat(),
                "source": "bybit",
            }
            
            new_symbols.append(display_name)
            total_new_candles += len(ohlcv)
            logger.info("  OK: %d candles", len(ohlcv))
        else:
            failed_symbols.append(symbol)
            logger.warning("  FAILED: insufficient data")
        
        # Rate limiting
        time.sleep(0.2)
    
    # Save updated data
    save_data(existing_data)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 60)
    logger.info("New symbols added: %d", len(new_symbols))
    logger.info("Failed symbols: %d", len(failed_symbols))
    logger.info("Total new candles: %d", total_new_candles)
    logger.info("Total symbols now: %d", len(existing_data))
    
    if new_symbols:
        logger.info("\nNew symbols:")
        for s in new_symbols:
            logger.info("  - %s", s)
    
    if failed_symbols:
        logger.info("\nFailed symbols:")
        for s in failed_symbols:
            logger.info("  - %s", s)


if __name__ == "__main__":
    main()
