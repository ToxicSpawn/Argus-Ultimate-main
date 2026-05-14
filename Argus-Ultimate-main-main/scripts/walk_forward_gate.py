#!/usr/bin/env python3
"""
Walk-forward gate: run backtest on train window, then test window; report train vs test metrics.
Use for periodic validation that strategy edge holds out-of-sample.
Usage:
  python scripts/walk_forward_gate.py --csv data/ohlcv.csv [--train-days 21] [--test-days 7] [--config unified_config.yaml]
Expects CSV with columns: timestamp, open, high, low, close, volume (or similar).
"""

import argparse
import asyncio
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtest: train then test window")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV (timestamp, open, high, low, close, volume)")
    parser.add_argument("--train-days", type=int, default=21, help="Training window days")
    parser.add_argument("--test-days", type=int, default=7, help="Test (out-of-sample) window days")
    parser.add_argument("--config", default="unified_config.yaml", help="Config file path")
    parser.add_argument("--symbol", default="BTC/USD", help="Symbol name for backtest")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("ERROR: CSV not found:", csv_path)
        return 1

    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print("ERROR: pandas/numpy required:", e)
        return 1

    # Load CSV
    df = pd.read_csv(csv_path)
    for col in ["close", "timestamp"]:
        if col not in df.columns and col == "timestamp":
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**9
            elif "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**9
            else:
                print("ERROR: CSV needs timestamp/date column")
                return 1
        elif col not in df.columns:
            print("ERROR: CSV needs column:", col)
            return 1

    # Normalise timestamp to numeric (Unix seconds) regardless of source format
    if df["timestamp"].dtype == object or str(df["timestamp"].dtype).startswith("datetime"):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 10**9

    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < args.train_days + args.test_days:
        print("WARN: Not enough rows for train+test; using full series as single window")

    # Split: last (train_days + test_days) bars; train = first train_days, test = next test_days
    bars_per_day = max(1, len(df) // max(1, int((int(df["timestamp"].max()) - int(df["timestamp"].min())) / 86400)))
    train_bars = args.train_days * bars_per_day
    test_bars = args.test_days * bars_per_day
    if train_bars + test_bars > len(df):
        train_bars = max(100, len(df) // 2)
        test_bars = max(50, len(df) - train_bars)

    train_df = df.iloc[:train_bars]
    test_df = df.iloc[train_bars : train_bars + test_bars]

    if train_df.empty or test_df.empty:
        print("ERROR: Train or test window empty")
        return 1

    # Load config — flatten all nested sections into a single namespace so components
    # that access config.kraken_taker_fee, config.min_position_size_aud, etc. work.
    try:
        import yaml
        from types import SimpleNamespace
        cfg_path = repo_root / args.config
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            flat: dict = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    flat.update(v)
                else:
                    flat[k] = v
            flat["trading_pairs"] = [args.symbol]
            flat["run_mode"] = "backtest"
            # Ensure critical defaults are present
            flat.setdefault("starting_capital_aud", 1000.0)
            flat.setdefault("min_position_size_aud", 15.0)
            flat.setdefault("max_position_size_aud", 100.0)
            flat.setdefault("max_position_pct", 0.1)
            flat.setdefault("max_total_exposure_pct", 0.4)
            flat.setdefault("max_concurrent_positions", 4)
            flat.setdefault("kraken_taker_fee", 0.0026)
            flat.setdefault("coinbase_taker_fee", 0.006)
            flat.setdefault("exchange_fee", 0.001)
            config = SimpleNamespace(**flat)
        else:
            config = SimpleNamespace(
                starting_capital_aud=1000.0, min_position_size_aud=15.0,
                max_position_size_aud=100.0, max_position_pct=0.1,
                max_total_exposure_pct=0.4, max_concurrent_positions=4,
                kraken_taker_fee=0.0026, coinbase_taker_fee=0.006,
                exchange_fee=0.001, trading_pairs=[args.symbol], run_mode="backtest",
            )
    except Exception as e:
        print("WARN: Config load failed, using defaults:", e)
        from types import SimpleNamespace
        config = SimpleNamespace(
            starting_capital_aud=1000.0, min_position_size_aud=15.0,
            max_position_size_aud=100.0, max_position_pct=0.1,
            max_total_exposure_pct=0.4, max_concurrent_positions=4,
            kraken_taker_fee=0.0026, coinbase_taker_fee=0.006,
            exchange_fee=0.001, trading_pairs=[args.symbol], run_mode="backtest",
        )

    async def run_one(data: pd.DataFrame, label: str):
        from backtesting.unified_event_backtester import UnifiedEventBacktester
        bt = UnifiedEventBacktester(config)
        await bt.initialize()
        result = await bt.run(symbol=args.symbol, ohlcv=data)
        ret = (result.end_equity_aud - result.start_equity_aud) / max(result.start_equity_aud, 1e-9) * 100.0
        print(f"  {label}: return={ret:.2f}% max_dd={result.max_drawdown_pct:.2f}% trades={result.trades} wins={result.wins} losses={result.losses}")
        return result

    async def main_async():
        print("Train window:", len(train_df), "bars | Test window:", len(test_df), "bars")
        print("Running backtest on train then test...")
        train_result = await run_one(train_df, "Train")
        test_result = await run_one(test_df, "Test")
        train_ret = (train_result.end_equity_aud - train_result.start_equity_aud) / max(train_result.start_equity_aud, 1e-9) * 100.0
        test_ret = (test_result.end_equity_aud - test_result.start_equity_aud) / max(test_result.start_equity_aud, 1e-9) * 100.0
        if train_result.trades < 5:
            print("WARN: Few trades on train; walk-forward may be noisy")
        if test_ret < -20 and train_ret > 5:
            print("WARN: Test return much worse than train (possible overfitting)")
        # Write result for evolution or reporting (data/walk_forward_result.json)
        try:
            import json
            out_path = repo_root / "data" / "walk_forward_result.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "train_return_pct": train_ret,
                    "test_return_pct": test_ret,
                    "train_trades": train_result.trades,
                    "test_trades": test_result.trades,
                    "train_max_dd_pct": getattr(train_result, "max_drawdown_pct", 0),
                    "test_max_dd_pct": getattr(test_result, "max_drawdown_pct", 0),
                    "train_days": args.train_days,
                    "test_days": args.test_days,
                    "symbol": args.symbol,
                }, f, indent=2)
            print("Wrote", out_path)
        except Exception as e:
            print("WARN: Could not write walk_forward_result.json:", e)
        return 0

    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
