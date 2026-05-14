#!/usr/bin/env python3
"""
Deterministic paper-edge evaluator using local OHLCV snapshots.

Writes data/paper_results.json in a normalized format used by pre-live checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_loop_30d_unified import run_paper_loop


def _discover_symbols(data_dir: Path) -> List[str]:
    out: List[str] = []
    if not data_dir.exists():
        return out
    for path in sorted(data_dir.glob("*.json")):
        sym = path.stem.replace("_", "/").upper()
        if sym and sym not in out:
            out.append(sym)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate deterministic paper edge from local OHLCV data")
    p.add_argument("--config", default="unified_config.yaml")
    p.add_argument("--profile", default=None, help="Config profile name or YAML path overlay")
    p.add_argument("--data-dir", default="data/ohlcv_15m")
    p.add_argument("--symbols", default="", help="Comma-separated symbols (default: auto-discover from data-dir)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--output", default="data/paper_results.json")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    symbols = str(args.symbols or "").strip()
    if not symbols:
        discovered = _discover_symbols(data_dir)
        symbols = ",".join(discovered)
    if not symbols:
        print(f"ERROR: no symbols found in {data_dir}")
        return 1

    result = run_paper_loop(
        config_path=str(args.config),
        profile=(str(args.profile) if args.profile else None),
        days=int(args.days),
        timeframe=str(args.timeframe),
        symbols_csv=symbols,
        fetch=False,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "deterministic_local_ohlcv",
        "config": str(args.config),
        "profile": str(args.profile or ""),
        "symbols": [x.strip() for x in symbols.split(",") if x.strip()],
        "days": int(args.days),
        "timeframe": str(args.timeframe),
        "trades": int(result.get("trades", 0) or 0),
        "wins": int(result.get("wins", 0) or 0),
        "losses": int(result.get("losses", 0) or 0),
        "closed_trades": int(result.get("closed_trades", 0) or 0),
        "win_rate_pct": float(result.get("win_rate_pct", 0.0) or 0.0),
        "return_pct": float(result.get("return_pct", 0.0) or 0.0),
        "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0) or 0.0),
        "pnl_aud": float(result.get("pnl_aud", 0.0) or 0.0),
        "sharpe": float(result.get("sharpe", 0.0) or 0.0),
        "sortino": float(result.get("sortino", 0.0) or 0.0),
        "returns_volatility": float(result.get("returns_volatility", 0.0) or 0.0),
        "status": str(result.get("status", "ok")),
        "error": str(result.get("error", "")),
        "symbol_results": list(result.get("symbol_results") or []),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        "paper_edge_eval: status=%s trades=%s closed=%s win_rate=%.2f%% return=%.2f%%"
        % (
            payload["status"],
            payload["trades"],
            payload["closed_trades"],
            payload["win_rate_pct"],
            payload["return_pct"],
        )
    )
    print(f"paper_edge_eval: wrote {out_path.resolve()}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
