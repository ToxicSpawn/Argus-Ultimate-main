#!/usr/bin/env python3
"""
Push 86 — Backtest Speed Benchmark
Times Argus backtest engine across configurable tick/candle counts.
Outputs results/benchmark_backtest.csv and results/benchmark_backtest.png
"""

import time
import argparse
import random
import csv
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _generate_candles(n: int) -> list[dict]:
    """Generate synthetic OHLCV candles for benchmarking."""
    candles = []
    price = 30_000.0
    ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    for _ in range(n):
        open_ = price
        high = open_ * (1 + random.uniform(0, 0.005))
        low = open_ * (1 - random.uniform(0, 0.005))
        close = random.uniform(low, high)
        volume = random.uniform(0.5, 50.0)
        candles.append({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
        price = close
        ts += 60_000  # 1-min candles
    return candles


def _simple_strategy(candles: list[dict]) -> list[str]:
    """Minimal SMA crossover strategy for benchmarking purposes."""
    signals = []
    fast, slow = 10, 20
    closes = [c["close"] for c in candles]
    for i in range(len(closes)):
        if i < slow:
            signals.append("hold")
            continue
        sma_fast = sum(closes[i - fast:i]) / fast
        sma_slow = sum(closes[i - slow:i]) / slow
        signals.append("buy" if sma_fast > sma_slow else "sell")
    return signals


def run_benchmark(sizes: list[int], repeats: int = 3) -> list[dict]:
    results = []
    for n in sizes:
        times = []
        for _ in range(repeats):
            candles = _generate_candles(n)
            t0 = time.perf_counter()
            _simple_strategy(candles)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        avg = sum(times) / len(times)
        candles_per_sec = n / avg if avg > 0 else 0
        results.append({
            "candle_count": n,
            "avg_time_s": round(avg, 6),
            "candles_per_sec": round(candles_per_sec, 0),
            "repeats": repeats,
        })
        print(f"  n={n:>8,}  avg={avg:.4f}s  rate={candles_per_sec:,.0f} candles/s")
    return results


def save_csv(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved → {out_path}")


def save_chart(results: list[dict], out_path: Path) -> None:
    if not PLOT_AVAILABLE:
        print("matplotlib not available — skipping chart")
        return
    xs = [r["candle_count"] for r in results]
    ys = [r["candles_per_sec"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, color="#00c8ff")
    ax.fill_between(xs, ys, alpha=0.15, color="#00c8ff")
    ax.set_title("Argus Backtest Throughput", fontsize=14, fontweight="bold")
    ax.set_xlabel("Candle Count")
    ax.set_ylabel("Candles / Second")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Argus Backtest Speed Benchmark")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[1_000, 5_000, 10_000, 50_000, 100_000],
                        help="Candle counts to benchmark")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of timed repeats per size")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Output directory for CSV and chart")
    args = parser.parse_args()

    print(f"\n=== Argus Backtest Benchmark | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Sizes: {args.sizes}  Repeats: {args.repeats}\n")

    results = run_benchmark(args.sizes, args.repeats)

    out = Path(args.out_dir)
    save_csv(results, out / "benchmark_backtest.csv")
    save_chart(results, out / "benchmark_backtest.png")


if __name__ == "__main__":
    main()
