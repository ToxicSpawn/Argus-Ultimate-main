"""optimize_gateway.py — Push 42.

Grid search over SignalGateway parameters using a saved backtest CSV
(output of scripts/backtest_gateway.py).

Parameters searched
-------------------
  min_confidence : float in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  source_weights : per-source weight multipliers in {0.5, 1.0, 1.5}
    Sources: VOID_BREAKER, CROSS_ASSET, RL_AGENT, DEEPLOB,
             OFI_STREAM, VPIN_STREAM, LLM_OVERLAY, FUNDING_ARB

Scoring
-------
  Primary:   Sharpe-like = mean(signed_edge@1) / std(signed_edge@1) * sqrt(252)
  Secondary: hit_rate@1, n_signals

  Signed edge = (1 if long else -1) * fwd_ret_1
  Only signals matching the weighted consensus are rescored per combo.

Output
------
  results/optimize_gateway_<ts>.json  — ranked top-20 configs + best config
  Prints ranked table to stdout.

CLI
---
  python scripts/optimize_gateway.py --csv results/backtest_gateway_<ts>.csv
  python scripts/optimize_gateway.py --csv results/bt.csv --top 10 --forward fwd_ret_5
  python scripts/optimize_gateway.py --csv results/bt.csv --sources OFI_STREAM VPIN_STREAM
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("optimize_gateway")

_ALL_SOURCES = [
    "VOID_BREAKER", "CROSS_ASSET", "RL_AGENT", "DEEPLOB",
    "OFI_STREAM", "VPIN_STREAM", "LLM_OVERLAY", "FUNDING_ARB",
]
_CONFIDENCE_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
_WEIGHT_GRID     = [0.5, 1.0, 1.5]
_DEFAULT_FORWARD = "fwd_ret_1"
_DEFAULT_TOP     = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class BacktestRow:
    bar_idx:    int
    direction:  str
    confidence: float
    sources:    List[str]
    fwd_ret_1:  float
    fwd_ret_5:  float
    fwd_ret_15: float


def load_backtest_csv(path: str) -> List[BacktestRow]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(BacktestRow(
                    bar_idx    = int(row["bar_idx"]),
                    direction  = row["direction"],
                    confidence = float(row["confidence"]),
                    sources    = [s.strip() for s in row["sources"].split(",") if s.strip()],
                    fwd_ret_1  = float(row["fwd_ret_1"]),
                    fwd_ret_5  = float(row["fwd_ret_5"]),
                    fwd_ret_15 = float(row["fwd_ret_15"]),
                ))
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed row: %s", exc)
    logger.info("Loaded %d signal records from %s", len(rows), path)
    return rows


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------

def rescore(
    rows:           List[BacktestRow],
    min_confidence: float,
    source_weights: Dict[str, float],
    forward_col:    str = _DEFAULT_FORWARD,
) -> Tuple[float, float, int]:
    """Re-evaluate rows under new params. Returns (sharpe, hit_rate, n_signals)."""
    edges: List[float] = []
    hits:  List[int]   = []

    for row in rows:
        if not row.sources:
            continue

        # Reweight confidence: scale by mean weight of participating sources
        weights = [source_weights.get(s, 1.0) for s in row.sources]
        adj_conf = row.confidence * (sum(weights) / len(weights))

        if adj_conf < min_confidence:
            continue

        fwd = getattr(row, forward_col, row.fwd_ret_1)
        if math.isnan(fwd):
            continue

        sign  = 1 if row.direction == "long" else -1
        edge  = sign * fwd
        edges.append(edge)
        hits.append(1 if edge > 0 else 0)

    n = len(edges)
    if n == 0:
        return 0.0, 0.0, 0

    mean_e = float(np.mean(edges))
    std_e  = float(np.std(edges))
    sharpe = mean_e / std_e * math.sqrt(252) if std_e > 1e-10 else 0.0
    hit    = float(np.mean(hits))
    return sharpe, hit, n


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

@dataclass
class GridResult:
    rank:           int
    sharpe:         float
    hit_rate:       float
    n_signals:      int
    min_confidence: float
    source_weights: Dict[str, float]


def grid_search(
    rows:        List[BacktestRow],
    sources:     List[str] = None,
    forward_col: str       = _DEFAULT_FORWARD,
    top_n:       int       = _DEFAULT_TOP,
    quiet:       bool      = False,
) -> List[GridResult]:
    if sources is None:
        sources = _ALL_SOURCES

    # Build weight combos only for specified sources; others fixed at 1.0
    weight_combos = list(itertools.product(_WEIGHT_GRID, repeat=len(sources)))
    total = len(_CONFIDENCE_GRID) * len(weight_combos)
    logger.info(
        "Grid search | sources=%d weight_combos=%d conf_levels=%d total=%d",
        len(sources), len(weight_combos), len(_CONFIDENCE_GRID), total,
    )

    results: List[GridResult] = []
    evaluated = 0

    for conf in _CONFIDENCE_GRID:
        for w_combo in weight_combos:
            sw: Dict[str, float] = {s: 1.0 for s in _ALL_SOURCES}
            for s, w in zip(sources, w_combo):
                sw[s] = w

            sharpe, hit, n = rescore(rows, conf, sw, forward_col)
            results.append(GridResult(
                rank=0, sharpe=sharpe, hit_rate=hit, n_signals=n,
                min_confidence=conf, source_weights=dict(sw),
            ))
            evaluated += 1
            if not quiet and evaluated % 500 == 0:
                logger.info("Progress %d/%d", evaluated, total)

    # Rank by Sharpe descending
    results.sort(key=lambda r: r.sharpe, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    top = results[:top_n]
    logger.info("Grid search complete | best_sharpe=%.4f best_conf=%.2f",
                top[0].sharpe if top else 0.0,
                top[0].min_confidence if top else 0.0)
    return top


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results: List[GridResult]) -> None:
    header = f"{'Rank':>4}  {'Sharpe':>8}  {'Hit@1':>7}  {'N':>6}  {'MinConf':>7}  Source Weights"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        weights_str = "  ".join(
            f"{s[:4]}={w:.1f}" for s, w in r.source_weights.items() if w != 1.0
        ) or "all=1.0"
        print(
            f"{r.rank:>4}  {r.sharpe:>8.4f}  {r.hit_rate:>7.3f}  "
            f"{r.n_signals:>6}  {r.min_confidence:>7.2f}  {weights_str}"
        )


def write_results(
    results: List[GridResult],
    out_dir: str = "results",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts   = int(time.time())
    path = os.path.join(out_dir, f"optimize_gateway_{ts}.json")
    payload = {
        "generated_at": ts,
        "top_results": [asdict(r) for r in results],
        "best_config": asdict(results[0]) if results else {},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Optimizer results → %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Argus Gateway Optimiser (Push 42)")
    p.add_argument("--csv",     required=True, help="Backtest CSV from backtest_gateway.py")
    p.add_argument("--forward", default=_DEFAULT_FORWARD,
                   choices=["fwd_ret_1", "fwd_ret_5", "fwd_ret_15"])
    p.add_argument("--sources", nargs="+", default=None,
                   help="Sources to vary weights for (default: all 8)")
    p.add_argument("--top",     default=_DEFAULT_TOP, type=int)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--quiet",   action="store_true")
    p.add_argument("--no-write", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = load_backtest_csv(args.csv)
    if not rows:
        print("No valid rows in CSV. Exiting.")
        raise SystemExit(1)

    top = grid_search(
        rows, sources=args.sources,
        forward_col=args.forward,
        top_n=args.top,
        quiet=args.quiet,
    )
    print_table(top)

    if not args.no_write:
        write_results(top, out_dir=args.out_dir)
