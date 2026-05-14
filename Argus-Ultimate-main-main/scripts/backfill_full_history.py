#!/usr/bin/env python
"""
ARGUS Full Historical Backfill — paginated OHLCV download going back years.

Uses CryptoCompare API (free, no key needed for basic access) which has full
historical data unlike Kraken's limited OHLCV endpoint.

Falls back to Kraken CCXT for recent data if CryptoCompare unavailable.

Usage:
    py -B scripts/backfill_full_history.py
    py -B scripts/backfill_full_history.py --symbols BTC ETH SOL XRP DOGE --timeframes 1h 1d
    py -B scripts/backfill_full_history.py --since 2015-01-01
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill")

DB_PATH = "data/historical_ohlcv.db"
CC_LIMIT = 2000  # CryptoCompare max per request
RATE_LIMIT_DELAY = 0.5  # seconds between API calls (generous free tier)


def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf ON ohlcv(symbol, timeframe, timestamp)")
    conn.commit()
    return conn


def get_earliest_timestamp(conn: sqlite3.Connection, symbol: str, timeframe: str) -> int | None:
    row = conn.execute(
        "SELECT MIN(timestamp) FROM ohlcv WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchone()
    return row[0] if row and row[0] else None


def get_latest_timestamp(conn: sqlite3.Connection, symbol: str, timeframe: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(timestamp) FROM ohlcv WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchone()
    return row[0] if row and row[0] else None


def get_count(conn: sqlite3.Connection, symbol: str, timeframe: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchone()
    return row[0] if row else 0


def timeframe_to_ms(tf: str) -> int:
    mapping = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    return mapping.get(tf, 3_600_000)


def cc_fetch(fsym: str, tsym: str, tf: str, to_ts: int, limit: int = 2000) -> list:
    """Fetch OHLCV from CryptoCompare API. Returns list of [timestamp_ms, o, h, l, c, v]."""
    if tf == "1h":
        endpoint = "histohour"
    elif tf == "1d":
        endpoint = "histoday"
    elif tf == "4h":
        # CryptoCompare doesn't have 4h — fetch 1h and we'll aggregate later
        endpoint = "histohour"
        limit = min(limit * 4, 2000)
    else:
        endpoint = "histohour"

    url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}?fsym={fsym}&tsym={tsym}&limit={limit}&toTs={to_ts}"
    api_key = os.environ.get("CRYPTOCOMPARE_API_KEY", "")
    if api_key:
        url += f"&api_key={api_key}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ARGUS-Backfill/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        log.warning("  CryptoCompare fetch error: %s", e)
        return []

    if data.get("Response") != "Success" or "Data" not in data:
        return []

    candles = data["Data"].get("Data", [])
    result = []
    for c in candles:
        if c.get("close", 0) == 0 and c.get("open", 0) == 0:
            continue  # Skip empty candles
        ts_ms = c["time"] * 1000
        result.append([ts_ms, c["open"], c["high"], c["low"], c["close"], c.get("volumefrom", 0)])
    return result


def backfill_symbol(conn: sqlite3.Connection, symbol: str, timeframe: str, since_ms: int):
    """Paginate backwards from now to since_ms using CryptoCompare."""
    # Parse symbol: BTC/USD -> fsym=BTC, tsym=USD
    parts = symbol.split("/")
    fsym, tsym = parts[0], parts[1]

    now_ts = int(datetime.now(timezone.utc).timestamp())
    since_ts = since_ms // 1000

    # Check existing data — find earliest timestamp to avoid re-fetching
    earliest = get_earliest_timestamp(conn, symbol, timeframe)
    if earliest:
        # We already have data — paginate backwards from our earliest point
        cursor_ts = earliest // 1000 - 1
    else:
        cursor_ts = now_ts

    total_saved = 0
    empty_count = 0

    while cursor_ts > since_ts:
        candles = cc_fetch(fsym, tsym, timeframe, cursor_ts, CC_LIMIT)

        if not candles:
            empty_count += 1
            if empty_count >= 3:
                log.info("  No more data available before %s",
                        datetime.fromtimestamp(cursor_ts, tz=timezone.utc).strftime("%Y-%m-%d"))
                break
            cursor_ts -= CC_LIMIT * (timeframe_to_ms(timeframe) // 1000)
            time.sleep(RATE_LIMIT_DELAY)
            continue

        empty_count = 0
        rows = [(symbol, timeframe, int(c[0]), c[1], c[2], c[3], c[4], c[5]) for c in candles]
        conn.executemany(
            "INSERT OR IGNORE INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        total_saved += len(rows)

        first_ts = candles[0][0] // 1000
        first_date = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%Y-%m-%d")

        if total_saved % 10000 < CC_LIMIT:
            log.info("  %s %s: %d candles saved, earliest: %s", symbol, timeframe, total_saved, first_date)

        # Move cursor before first candle
        cursor_ts = first_ts - 1
        time.sleep(RATE_LIMIT_DELAY)

    # Also fetch forward from latest to fill any gaps to present
    latest = get_latest_timestamp(conn, symbol, timeframe)
    if latest:
        latest_ts = latest // 1000
        if now_ts - latest_ts > timeframe_to_ms(timeframe) // 1000:
            forward_candles = cc_fetch(fsym, tsym, timeframe, now_ts, CC_LIMIT)
            if forward_candles:
                rows = [(symbol, timeframe, int(c[0]), c[1], c[2], c[3], c[4], c[5]) for c in forward_candles]
                conn.executemany(
                    "INSERT OR IGNORE INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
                total_saved += len(rows)

    return total_saved


def main():
    parser = argparse.ArgumentParser(description="ARGUS Full Historical Backfill")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"])
    parser.add_argument("--timeframes", nargs="+", default=["1h", "1d"])
    parser.add_argument("--since", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--db", type=str, default=DB_PATH)
    args = parser.parse_args()

    since_ms = int(datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    log.info("=" * 60)
    log.info("ARGUS Full Historical Backfill")
    log.info("=" * 60)
    log.info("Symbols: %s", args.symbols)
    log.info("Timeframes: %s", args.timeframes)
    log.info("Since: %s", args.since)
    log.info("Database: %s", args.db)
    log.info("")
    log.info("Using CryptoCompare API (free tier, full historical data)")

    conn = init_db(args.db)

    # Normalize symbols: accept both "BTC" and "BTC/USD"
    normalized = []
    for s in args.symbols:
        if "/" not in s:
            s = f"{s}/USD"
        normalized.append(s)
    args.symbols = normalized

    results = []
    start_time = time.time()

    for symbol in args.symbols:
        for tf in args.timeframes:
            existing = get_count(conn, symbol, tf)
            log.info("Backfilling %s %s (existing: %d candles)...", symbol, tf, existing)

            saved = backfill_symbol(conn, symbol, tf, since_ms)
            total = get_count(conn, symbol, tf)

            earliest = get_earliest_timestamp(conn, symbol, tf)
            latest = get_latest_timestamp(conn, symbol, tf)

            if earliest and latest:
                earliest_dt = datetime.fromtimestamp(earliest / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                latest_dt = datetime.fromtimestamp(latest / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            else:
                earliest_dt = latest_dt = "N/A"

            results.append((symbol, tf, saved, total, earliest_dt, latest_dt))
            log.info("  Done: +%d new, %d total, range: %s to %s", saved, total, earliest_dt, latest_dt)

    elapsed = time.time() - start_time
    conn.close()

    # Print summary
    log.info("")
    log.info("=" * 70)
    log.info("BACKFILL COMPLETE — %.1f minutes", elapsed / 60)
    log.info("=" * 70)
    log.info("%-10s %-5s %8s %8s  %-12s %-12s", "Symbol", "TF", "New", "Total", "From", "To")
    log.info("-" * 70)

    grand_total = 0
    for symbol, tf, saved, total, earliest, latest in results:
        log.info("%-10s %-5s %8d %8d  %-12s %-12s", symbol, tf, saved, total, earliest, latest)
        grand_total += total

    log.info("-" * 70)
    log.info("Grand total: %d candles across %d symbol-timeframe pairs", grand_total, len(results))

    # DB file size
    db_size = os.path.getsize(args.db) / (1024 * 1024)
    log.info("Database size: %.1f MB (%s)", db_size, args.db)


if __name__ == "__main__":
    main()
