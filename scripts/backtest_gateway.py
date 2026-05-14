"""backtest_gateway.py — Push 41.

Replays historical OHLCV data through all 8 SignalGateway sources and
scores consensus signals against forward returns.

Sources replayed
----------------
  VOID_BREAKER  — MatrixEvaluator conviction + MTF boost
  CROSS_ASSET   — CrossAssetRegime scalar
  RL_AGENT      — OnlineAdapter win_rate (updated online during replay)
  DEEPLOB       — DeepLOBLiveBridge LOB snapshot
  OFI_STREAM    — LiveOFIStream bar-by-bar z-score
  VPIN_STREAM   — LiveVPINStream synthetic trade-tape VPIN
  LLM_OVERLAY   — MultiTimeframeFeatures aggregate bias
  FUNDING_ARB   — FundingRateScanner latest_signal (static mock in BT)

Scoring
-------
  Forward return windows: +1, +5, +15 bars (close-to-close).
  hit_rate_N  = fraction of signals where sign(consensus) == sign(fwd_N)
  edge_N      = mean(sign(consensus) * fwd_N)   i.e. signed alignment
  sharpe_like = edge_1 / std(edge_1) * sqrt(252)

Output
------
  results/backtest_gateway_<ts>.csv          per-signal detail
  results/backtest_gateway_<ts>_summary.json aggregate stats

CLI
---
  python scripts/backtest_gateway.py
  python scripts/backtest_gateway.py --symbol BTCUSDT --bars 2000
  python scripts/backtest_gateway.py --csv data/ohlcv.csv --forward 5
  python scripts/backtest_gateway.py --min-confidence 0.55 --quiet
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional CCXT for live data seeding
# ---------------------------------------------------------------------------
try:
    import ccxt
    _HAS_CCXT = True
except ImportError:
    _HAS_CCXT = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("backtest_gateway")

_DEFAULT_BARS       = 2000
_DEFAULT_SYMBOL     = "BTC/USDT"
_DEFAULT_EXCHANGE   = "binance"
_DEFAULT_TIMEFRAME  = "1m"
_DEFAULT_FORWARD    = [1, 5, 15]
_MIN_WARMUP_BARS    = 60
_DEFAULT_MIN_CONF   = 0.50
_MTF_BOOST          = 0.15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SignalRecord:
    bar_idx:      int
    timestamp:    float
    direction:    str
    confidence:   float
    sources:      str          # comma-joined source names
    n_sources:    int
    close_price:  float
    fwd_ret_1:    float = 0.0
    fwd_ret_5:    float = 0.0
    fwd_ret_15:   float = 0.0
    hit_1:        int   = 0
    hit_5:        int   = 0
    hit_15:       int   = 0


@dataclass
class BacktestSummary:
    symbol:           str
    timeframe:        str
    total_bars:       int
    warmup_bars:      int
    n_signals:        int
    min_confidence:   float
    hit_rate_1:       float = 0.0
    hit_rate_5:       float = 0.0
    hit_rate_15:      float = 0.0
    edge_1:           float = 0.0
    edge_5:           float = 0.0
    edge_15:          float = 0.0
    sharpe_like_1:    float = 0.0
    long_signals:     int   = 0
    short_signals:    int   = 0
    source_counts:    Dict[str, int] = field(default_factory=dict)
    runtime_sec:      float = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv_csv(path: str) -> np.ndarray:
    """Load OHLCV from CSV. Expected columns: timestamp,open,high,low,close,volume."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([
                float(row.get("timestamp", row.get("ts", 0))),
                float(row.get("open",  row.get("o", 0))),
                float(row.get("high",  row.get("h", 0))),
                float(row.get("low",   row.get("l", 0))),
                float(row.get("close", row.get("c", 0))),
                float(row.get("volume",row.get("v", 0))),
            ])
    return np.array(rows, dtype=np.float64)


def fetch_ohlcv_ccxt(
    symbol: str = _DEFAULT_SYMBOL,
    exchange_id: str = _DEFAULT_EXCHANGE,
    timeframe: str = _DEFAULT_TIMEFRAME,
    bars: int = _DEFAULT_BARS,
) -> np.ndarray:
    """Fetch OHLCV via CCXT. Falls back to synthetic if unavailable."""
    if not _HAS_CCXT:
        logger.warning("ccxt not installed — using synthetic data")
        return _synthetic_ohlcv(bars)
    try:
        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        logger.info("Fetching %d bars of %s %s from %s", bars, symbol, timeframe, exchange_id)
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=bars)
        arr = np.array(raw, dtype=np.float64)
        logger.info("Fetched %d candles", len(arr))
        return arr
    except Exception as exc:
        logger.warning("CCXT fetch failed (%s) — using synthetic data", exc)
        return _synthetic_ohlcv(bars)


def _synthetic_ohlcv(n: int = 2000, seed: int = 42) -> np.ndarray:
    """Generate synthetic OHLCV for testing."""
    rng  = np.random.default_rng(seed)
    base = 50_000.0
    rets = rng.normal(0, 0.001, n)
    close = base * np.cumprod(1 + rets)
    ts    = np.arange(n, dtype=np.float64) * 60_000  # 1-min bars in ms
    noise = rng.uniform(0.0, 0.001, n)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = np.roll(close, 1); open_[0] = base
    vol   = rng.exponential(10.0, n)
    return np.column_stack([ts, open_, high, low, close, vol])


# ---------------------------------------------------------------------------
# Synthetic trade tape generator (for OFI / VPIN replay)
# ---------------------------------------------------------------------------

def _bar_to_trades(
    bar: np.ndarray,
    n_trades: int = 20,
    seed: int = 0,
) -> List[Dict]:
    """Decompose a single OHLCV bar into synthetic trades for OFI/VPIN."""
    rng   = np.random.default_rng(seed)
    o, h, l, c, v = bar[1], bar[2], bar[3], bar[4], bar[5]
    prices = np.linspace(o, c, n_trades) + rng.normal(0, (h - l) * 0.1, n_trades)
    amounts = rng.exponential(v / n_trades, n_trades)
    # Bias side toward close direction
    buy_prob = 0.5 + 0.4 * np.sign(c - o) * 0.5
    sides = ["buy" if rng.random() < buy_prob else "sell" for _ in range(n_trades)]
    return [
        {"price": float(prices[i]), "amount": float(amounts[i]), "side": sides[i]}
        for i in range(n_trades)
    ]


# ---------------------------------------------------------------------------
# Simple consensus engine (mirrors SignalGateway logic, no async)
# ---------------------------------------------------------------------------

def _consensus(
    signals: List[Tuple[str, str, float]],  # (source, direction, confidence)
    min_confidence: float = _DEFAULT_MIN_CONF,
) -> Optional[Tuple[str, float, List[str]]]:
    """Return (direction, aggregate_confidence, sources) or None."""
    long_weight  = sum(c for _, d, c in signals if d == "long")
    short_weight = sum(c for _, d, c in signals if d == "short")
    total_weight = long_weight + short_weight
    if total_weight < 1e-9:
        return None
    direction  = "long" if long_weight >= short_weight else "short"
    confidence = max(long_weight, short_weight) / total_weight
    if confidence < min_confidence:
        return None
    sources = [s for s, d, _ in signals if d == direction]
    return direction, confidence, sources


# ---------------------------------------------------------------------------
# Forward return helpers
# ---------------------------------------------------------------------------

def _forward_return(closes: np.ndarray, idx: int, horizon: int) -> float:
    future_idx = idx + horizon
    if future_idx >= len(closes):
        return float("nan")
    return float((closes[future_idx] - closes[idx]) / closes[idx])


def _hit(direction: str, fwd_ret: float) -> int:
    if math.isnan(fwd_ret):
        return 0
    return int((direction == "long" and fwd_ret > 0) or
               (direction == "short" and fwd_ret < 0))


# ---------------------------------------------------------------------------
# Source signal extractors
# ---------------------------------------------------------------------------

def _signal_void_breaker(candles, mtf, matrix, mtf_boost=_MTF_BOOST):
    """VOID_BREAKER: MatrixEvaluator + MTF boost."""
    try:
        from strategies.tentacles.matrix_evaluator import MatrixEvaluator, Action
        result = matrix.evaluate(candles)
        direction = ("long"  if result.action == Action.BUY
                     else "short" if result.action == Action.SELL else None)
        if direction is None:
            return None
        conf = float(np.clip(result.conviction, 0, 1))
        try:
            mtf_r = mtf.compute(candles)
            mtf_dir = mtf_r.direction
            if (direction == "long"  and mtf_dir == "long")  or \
               (direction == "short" and mtf_dir == "short"):
                conf = min(conf + mtf_boost, 1.0)
        except Exception:
            pass
        return ("VOID_BREAKER", direction, conf)
    except Exception as exc:
        logger.debug("VOID_BREAKER failed: %s", exc)
        return None


def _signal_cross_asset(candles, regime):
    """CROSS_ASSET: CrossAssetRegime scalar."""
    try:
        scalar = regime.get_scalar(candles)
        direction = "long" if scalar > 1.0 else ("short" if scalar < 0.8 else None)
        if direction is None:
            return None
        conf = min(abs(scalar - 1.0) + 0.5, 1.0)
        return ("CROSS_ASSET", direction, conf)
    except Exception as exc:
        logger.debug("CROSS_ASSET failed: %s", exc)
        return None


def _signal_rl_agent(adapter):
    """RL_AGENT: OnlineAdapter win_rate."""
    try:
        wr = float(getattr(adapter, "win_rate", 0.5) or 0.5)
        if 0.45 < wr < 0.55:
            return None
        direction = "long" if wr >= 0.55 else "short"
        conf = min(abs(wr - 0.5) * 2.0, 1.0)
        return ("RL_AGENT", direction, max(conf, 0.1))
    except Exception as exc:
        logger.debug("RL_AGENT failed: %s", exc)
        return None


def _signal_deeplob(bridge, candles):
    """DEEPLOB: LOB inference."""
    try:
        sig = bridge.get_signal()
        if not sig or sig.get("direction", "flat") == "flat":
            return None
        return ("DEEPLOB", sig["direction"], float(sig["confidence"]))
    except Exception as exc:
        logger.debug("DEEPLOB failed: %s", exc)
        return None


def _signal_ofi(ofi_stream):
    """OFI_STREAM: z-score threshold."""
    try:
        z = ofi_stream.ofi_zscore
        if abs(z) < 0.5:
            return None
        direction = "long" if z > 0 else "short"
        conf = min(abs(z) / 3.0, 1.0)
        return ("OFI_STREAM", direction, conf)
    except Exception as exc:
        logger.debug("OFI_STREAM failed: %s", exc)
        return None


def _signal_vpin(vpin_stream):
    """VPIN_STREAM: bucket VPIN."""
    try:
        v = vpin_stream.vpin
        if 0.35 <= v <= 0.65:
            return None
        direction = "short" if v > 0.65 else "long"
        conf = min(abs(v - 0.5) * 2.0, 1.0)
        return ("VPIN_STREAM", direction, max(conf, 0.1))
    except Exception as exc:
        logger.debug("VPIN_STREAM failed: %s", exc)
        return None


def _signal_llm_overlay(mtf, candles):
    """LLM_OVERLAY: MultiTimeframeFeatures."""
    try:
        r = mtf.compute(candles)
        if r.direction == "flat":
            return None
        return ("LLM_OVERLAY", r.direction, float(r.confidence))
    except Exception as exc:
        logger.debug("LLM_OVERLAY failed: %s", exc)
        return None


def _signal_funding_arb(scanner):
    """FUNDING_ARB: latest scanner signal."""
    try:
        sig = scanner.latest_signal()
        if not sig or sig.get("direction", "flat") == "flat":
            return None
        conf = min(max(float(sig.get("stability", 0.5)), 0.1), 1.0)
        return ("FUNDING_ARB", sig["direction"], conf)
    except Exception as exc:
        logger.debug("FUNDING_ARB failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    candles:        np.ndarray,
    symbol:         str  = _DEFAULT_SYMBOL,
    timeframe:      str  = _DEFAULT_TIMEFRAME,
    forward_windows: List[int] = None,
    min_confidence: float = _DEFAULT_MIN_CONF,
    quiet:          bool  = False,
) -> Tuple[List[SignalRecord], BacktestSummary]:
    if forward_windows is None:
        forward_windows = list(_DEFAULT_FORWARD)

    t0 = time.time()

    # Lazy imports (allow partial installs)
    try:
        from strategies.tentacles.matrix_evaluator import MatrixEvaluator, AggregationMode
        matrix = MatrixEvaluator(mode=MatrixEvaluator.get_configured_mode(),
                                 buy_threshold=0.15, sell_threshold=0.15, type_weights={})
    except Exception as exc:
        logger.warning("MatrixEvaluator unavailable (%s) — VOID_BREAKER disabled", exc)
        matrix = None

    try:
        from alpha.cross_asset import CrossAssetRegime
        regime = CrossAssetRegime()
    except Exception:
        regime = None

    try:
        from learning.online_adapter import OnlineAdapter
        adapter = OnlineAdapter()
    except Exception:
        adapter = None

    try:
        from alpha.microstructure.deeplob_live_bridge import DeepLOBLiveBridge
        bridge = DeepLOBLiveBridge()
    except Exception:
        bridge = None

    try:
        from core.mtf_features import MultiTimeframeFeatures
        mtf = MultiTimeframeFeatures()
    except Exception:
        mtf = None

    try:
        from alpha.funding_rate_scanner import FundingRateScanner
        # Static mock during backtest — no live polling
        funding = FundingRateScanner(symbols=["BTCUSDT"], poll_interval=999_999)
    except Exception:
        funding = None

    from alpha.microstructure.live_ofi_stream import LiveOFIStream
    from alpha.microstructure.live_vpin_stream import LiveVPINStream
    ofi_stream  = LiveOFIStream(window=20)
    vpin_stream = LiveVPINStream(bucket_size=50, n_buckets=50)

    closes  = candles[:, 4]
    n_bars  = len(candles)
    records: List[SignalRecord] = []
    source_counts: Dict[str, int] = {}

    logger.info("Backtest start | bars=%d warmup=%d min_conf=%.2f fwd=%s",
                n_bars, _MIN_WARMUP_BARS, min_confidence, forward_windows)

    for i in range(_MIN_WARMUP_BARS, n_bars):
        window = candles[max(0, i - 500): i + 1]

        # Feed OFI / VPIN with synthetic trades for this bar
        bar_trades = _bar_to_trades(candles[i], n_trades=20, seed=i)
        for t in bar_trades:
            ofi_stream.on_trade(t)
            vpin_stream.on_trade(t)
        # Also feed a synthetic book delta
        bar = candles[i]
        spread = float(bar[2] - bar[3])  # high - low as proxy
        ofi_stream.on_book_delta({
            "bid_delta":  spread * 0.5 if bar[4] > bar[1] else -spread * 0.5,
            "ask_delta": -spread * 0.5 if bar[4] > bar[1] else  spread * 0.5,
        })
        ofi_stream.close_bar()

        # Collect signals from all 8 sources
        signals = []
        if matrix:
            s = _signal_void_breaker(window, mtf, matrix)
            if s: signals.append(s)
        if regime:
            s = _signal_cross_asset(window, regime)
            if s: signals.append(s)
        if adapter:
            s = _signal_rl_agent(adapter)
            if s: signals.append(s)
        if bridge:
            s = _signal_deeplob(bridge, window)
            if s: signals.append(s)
        s = _signal_ofi(ofi_stream)
        if s: signals.append(s)
        s = _signal_vpin(vpin_stream)
        if s: signals.append(s)
        if mtf:
            s = _signal_llm_overlay(mtf, window)
            if s: signals.append(s)
        if funding:
            s = _signal_funding_arb(funding)
            if s: signals.append(s)

        if not signals:
            continue

        result = _consensus(signals, min_confidence=min_confidence)
        if result is None:
            continue

        direction, confidence, src_names = result

        # Score forward returns
        fwd = {h: _forward_return(closes, i, h) for h in forward_windows}

        rec = SignalRecord(
            bar_idx     = i,
            timestamp   = float(candles[i, 0]),
            direction   = direction,
            confidence  = confidence,
            sources     = ",".join(src_names),
            n_sources   = len(src_names),
            close_price = float(closes[i]),
            fwd_ret_1   = fwd.get(1,  float("nan")),
            fwd_ret_5   = fwd.get(5,  float("nan")),
            fwd_ret_15  = fwd.get(15, float("nan")),
            hit_1       = _hit(direction, fwd.get(1,  float("nan"))),
            hit_5       = _hit(direction, fwd.get(5,  float("nan"))),
            hit_15      = _hit(direction, fwd.get(15, float("nan"))),
        )
        records.append(rec)

        for src, _, _ in signals:
            source_counts[src] = source_counts.get(src, 0) + 1

        # Online RL update
        if adapter and not math.isnan(fwd.get(1, float("nan"))):
            won = (direction == "long" and fwd[1] > 0) or (direction == "short" and fwd[1] < 0)
            adapter.update(won)

        if not quiet and i % 200 == 0:
            logger.info("Bar %d/%d | signals_so_far=%d", i, n_bars, len(records))

    runtime = time.time() - t0

    # Aggregate stats
    n = len(records)
    if n == 0:
        logger.warning("No consensus signals generated.")

    def _safe_mean(vals):
        v = [x for x in vals if not math.isnan(x)]
        return float(np.mean(v)) if v else 0.0

    def _safe_std(vals):
        v = [x for x in vals if not math.isnan(x)]
        return float(np.std(v)) if len(v) > 1 else 1.0

    edge1_vals = [
        (1 if r.direction == "long" else -1) * r.fwd_ret_1
        for r in records if not math.isnan(r.fwd_ret_1)
    ]
    edge1_mean = _safe_mean(edge1_vals)
    edge1_std  = _safe_std(edge1_vals)
    sharpe     = edge1_mean / edge1_std * math.sqrt(252) if edge1_std > 1e-10 else 0.0

    summary = BacktestSummary(
        symbol         = symbol,
        timeframe      = timeframe,
        total_bars     = n_bars,
        warmup_bars    = _MIN_WARMUP_BARS,
        n_signals      = n,
        min_confidence = min_confidence,
        hit_rate_1     = _safe_mean([r.hit_1  for r in records]),
        hit_rate_5     = _safe_mean([r.hit_5  for r in records]),
        hit_rate_15    = _safe_mean([r.hit_15 for r in records]),
        edge_1         = edge1_mean,
        edge_5         = _safe_mean([(1 if r.direction=="long" else -1)*r.fwd_ret_5  for r in records]),
        edge_15        = _safe_mean([(1 if r.direction=="long" else -1)*r.fwd_ret_15 for r in records]),
        sharpe_like_1  = sharpe,
        long_signals   = sum(1 for r in records if r.direction == "long"),
        short_signals  = sum(1 for r in records if r.direction == "short"),
        source_counts  = source_counts,
        runtime_sec    = runtime,
    )

    logger.info(
        "Backtest complete | signals=%d hit@1=%.3f hit@5=%.3f hit@15=%.3f "
        "edge@1=%.5f sharpe_like=%.3f runtime=%.1fs",
        n, summary.hit_rate_1, summary.hit_rate_5, summary.hit_rate_15,
        summary.edge_1, summary.sharpe_like_1, runtime,
    )
    return records, summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_results(
    records: List[SignalRecord],
    summary: BacktestSummary,
    out_dir: str = "results",
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts  = int(time.time())
    csv_path  = os.path.join(out_dir, f"backtest_gateway_{ts}.csv")
    json_path = os.path.join(out_dir, f"backtest_gateway_{ts}_summary.json")

    if records:
        fieldnames = list(asdict(records[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(r) for r in records)
        logger.info("Signal detail CSV → %s", csv_path)

    with open(json_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    logger.info("Summary JSON    → %s", json_path)

    return csv_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Argus Gateway Backtest (Push 41)")
    p.add_argument("--symbol",     default=_DEFAULT_SYMBOL)
    p.add_argument("--exchange",   default=_DEFAULT_EXCHANGE)
    p.add_argument("--timeframe",  default=_DEFAULT_TIMEFRAME)
    p.add_argument("--bars",       default=_DEFAULT_BARS, type=int)
    p.add_argument("--csv",        default=None, help="Load OHLCV from CSV instead of fetching")
    p.add_argument("--forward",    default=1,    type=int,
                   help="Primary forward window for Sharpe calc (default 1)")
    p.add_argument("--min-confidence", default=_DEFAULT_MIN_CONF, type=float)
    p.add_argument("--out-dir",    default="results")
    p.add_argument("--quiet",      action="store_true")
    p.add_argument("--no-write",   action="store_true", help="Skip writing output files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.csv:
        candles = load_ohlcv_csv(args.csv)
    else:
        candles = fetch_ohlcv_ccxt(
            symbol=args.symbol, exchange_id=args.exchange,
            timeframe=args.timeframe, bars=args.bars,
        )

    records, summary = run_backtest(
        candles, symbol=args.symbol, timeframe=args.timeframe,
        min_confidence=args.min_confidence, quiet=args.quiet,
    )

    if not args.no_write:
        write_results(records, summary, out_dir=args.out_dir)
    else:
        import pprint
        pprint.pprint(asdict(summary))
