#!/usr/bin/env python3
"""
Fetch 30 days of OHLCV and run the unified backtester to estimate 30-day earnings.
Usage: python scripts/run_30d_backtest.py [--symbol BTC/USD] [--capital 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fetch_ohlcv_ccxt(symbol: str = "BTC/USD", days: int = 30, timeframe: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV via CCXT (Kraken public, no keys). 1h = 24*30 = 720 bars."""
    import ccxt
    limit = min(720, days * 24) if timeframe == "1h" else min(4320, days * 24 * 6)
    ex = ccxt.kraken({"enableRateLimit": True})
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"Fetch failed: {e}. Using synthetic 30d data for demo.")
        import numpy as np
        n = 720
        t = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="1h")
        close = 97000.0 + np.cumsum(np.random.randn(n) * 200)
        high = close + np.abs(np.random.randn(n) * 100)
        low = close - np.abs(np.random.randn(n) * 100)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        vol = np.random.rand(n) * 1e6 + 1e5
        return pd.DataFrame({
            "timestamp": t,
            "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        })
    if not ohlcv:
        raise RuntimeError("No OHLCV data returned")
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="30-day backtest: fetch data + run unified backtester")
    ap.add_argument("--symbol", default="BTC/USD", help="Trading pair")
    ap.add_argument("--capital", type=float, default=1000.0, help="Starting capital AUD")
    ap.add_argument("--days", type=int, default=30, help="Days of history")
    ap.add_argument("--config", default=None, help="Config file (default: unified_config.yaml)")
    ap.add_argument("--csv", default=None, help="Use existing CSV instead of fetching")
    args = ap.parse_args()

    config_file = args.config or str(ROOT / "unified_config.yaml")
    csv_path = Path(args.csv) if args.csv else ROOT / "data" / "ohlcv_30d_backtest.csv"

    if args.csv and Path(args.csv).exists():
        print(f"Loading OHLCV from {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        print(f"Fetching {args.days} days of {args.symbol} OHLCV (1h)...")
        try:
            df = fetch_ohlcv_ccxt(symbol=args.symbol, days=args.days, timeframe="1h")
        except Exception as e:
            print(f"Fetch error: {e}")
            return 1
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} bars to {csv_path}")

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        except Exception:
            pass
    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]
    if "volume" not in df.columns and "Volume" in df.columns:
        df["volume"] = df["Volume"]

    print(f"Running unified backtest: {args.symbol}, ${args.capital:,.2f} AUD, {len(df)} bars...")
    from core.config_manager import load_unified_trading_config
    from backtesting.unified_event_backtester import run_backtest_sync

    cfg = load_unified_trading_config(config_file)
    try:
        setattr(cfg, "starting_capital_aud", float(args.capital))
        setattr(cfg, "run_mode", "backtest")
    except Exception:
        pass

    result = run_backtest_sync(config=cfg, symbol=args.symbol, ohlcv=df)

    print("\n" + "=" * 60)
    print("30-DAY BACKTEST RESULT")
    print("=" * 60)
    print(f"  Symbol:           {result.symbol}")
    print(f"  Start equity:     ${result.start_equity_aud:,.2f} AUD")
    print(f"  End equity:       ${result.end_equity_aud:,.2f} AUD")
    profit = result.end_equity_aud - result.start_equity_aud
    print(f"  Profit/Loss:      ${profit:+,.2f} AUD")
    print(f"  Return:           {result.total_return_pct:+.2f}%")
    print(f"  Max drawdown:     {result.max_drawdown_pct:.2f}%")
    closed = result.wins + result.losses
    print(f"  Trades:           {result.trades} (closed: {closed}, wins: {result.wins}, losses: {result.losses})")
    if closed > 0:
        print(f"  Win rate:         {100.0 * result.wins / closed:.1f}%")
    else:
        print(f"  Win rate:         N/A (no closed trades)")
    print("=" * 60)
    print(f"\nEstimated 30-day earnings at ${args.capital:,.0f} capital: ${profit:+,.2f} AUD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
