#!/usr/bin/env python3
"""
ARGUS System Status Dashboard (CLI) — read from health server or local state.

Displays:
  - Uptime, current cycle, capital, P&L
  - Active strategies, regime, last trade, errors
  - Model status (loaded / stale / missing)
  - Exchange connectivity

Usage:
    py -B scripts/status.py
    py -B scripts/status.py --port 8080
    py -B scripts/status.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Model files with expected staleness threshold (7 days)
MODEL_FILES = {
    "regime_classifier.pkl": 7,
    "rl_agent.zip": 7,
    "alpha_model.pkl": 7,
    "hmm_regime.pkl": 7,
    "signal_stacker.pkl": 7,
    "volatility_forecaster.pkl": 7,
    "vol_forecaster_v2.pkl": 7,
}


def _fetch_health_server(port: int) -> Optional[Dict[str, Any]]:
    """Try to fetch status from the running health server."""
    try:
        url = f"http://127.0.0.1:{port}/health"
        resp = urlopen(url, timeout=3)
        data = json.loads(resp.read().decode("utf-8"))
        return data
    except (URLError, OSError, json.JSONDecodeError, Exception):
        return None


def _get_model_status() -> List[Dict[str, Any]]:
    """Check model files for existence and staleness."""
    results: List[Dict[str, Any]] = []
    now = time.time()
    for name, max_days in MODEL_FILES.items():
        path = MODELS_DIR / name
        if not path.exists():
            results.append({"name": name, "status": "missing", "age_days": None})
        else:
            mtime = path.stat().st_mtime
            age_days = (now - mtime) / 86400
            if age_days > max_days:
                status = "stale"
            else:
                status = "loaded"
            results.append({"name": name, "status": status, "age_days": round(age_days, 1)})
    return results


def _get_exchange_status() -> List[Dict[str, str]]:
    """Quick TCP check for exchange endpoints."""
    import socket
    exchanges = [
        ("Kraken", "api.kraken.com", 443),
        ("Coinbase", "api.coinbase.com", 443),
    ]
    results: List[Dict[str, str]] = []
    for name, host, port in exchanges:
        try:
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
            results.append({"name": name, "status": "reachable"})
        except OSError:
            results.append({"name": name, "status": "unreachable"})

    # Check API keys
    key_map = {"Kraken": "KRAKEN_API_KEY", "Coinbase": "COINBASE_API_KEY"}
    for r in results:
        key_var = key_map.get(r["name"], "")
        val = os.getenv(key_var, "")
        if val and "your_" not in val.lower() and "_here" not in val.lower():
            r["api_key"] = "configured"
        else:
            r["api_key"] = "not configured"
    return results


def _get_local_trade_info() -> Dict[str, Any]:
    """Read latest trade info from local databases."""
    info: Dict[str, Any] = {"last_trade": None, "total_paper_trades": 0, "total_pnl": 0.0}
    trades_db = DATA_DIR / "unified_trades.db"
    if not trades_db.exists():
        return info
    try:
        conn = sqlite3.connect(str(trades_db))
        conn.row_factory = sqlite3.Row
        # Last trade
        try:
            row = conn.execute(
                "SELECT * FROM trades ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if row:
                info["last_trade"] = dict(row)
        except sqlite3.OperationalError:
            pass
        # Trade count + P&L
        try:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(pnl), 0) FROM trades WHERE mode='paper'"
            ).fetchone()
            if row:
                info["total_paper_trades"] = row[0]
                info["total_pnl"] = round(float(row[1]), 2)
        except sqlite3.OperationalError:
            pass
        conn.close()
    except Exception:
        pass
    return info


def build_status(port: int = 8080) -> Dict[str, Any]:
    """Assemble full status from health server + local state."""
    status: Dict[str, Any] = {}

    # Try health server first
    health = _fetch_health_server(port)
    if health:
        status["health_server"] = "running"
        status["system_status"] = health.get("status", "unknown")
        status["uptime_seconds"] = health.get("uptime_seconds", 0)
        status["is_trading"] = health.get("is_trading", False)
        status["active_strategies"] = health.get("active_strategies", 0)
        status["open_positions"] = health.get("open_positions", 0)
        status["daily_pnl_usd"] = health.get("daily_pnl_usd", 0.0)
        status["error_count"] = health.get("error_count", 0)
        last_sig = health.get("last_signal_ts")
        if last_sig:
            status["last_signal"] = datetime.fromtimestamp(last_sig, tz=timezone.utc).isoformat()
        else:
            status["last_signal"] = None
    else:
        status["health_server"] = "not running"
        status["system_status"] = "offline"

    # Model status
    status["models"] = _get_model_status()

    # Exchange status
    status["exchanges"] = _get_exchange_status()

    # Local trade info
    status["trade_info"] = _get_local_trade_info()

    status["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

    return status


def print_dashboard(status: Dict[str, Any]) -> None:
    """Pretty-print the status dashboard to terminal."""
    print("=" * 60)
    print("  ARGUS ULTIMATE -- System Status Dashboard")
    print("=" * 60)
    print()

    # System status
    srv = status.get("health_server", "unknown")
    sys_status = status.get("system_status", "unknown")
    print(f"  Health Server:   {srv}")
    print(f"  System Status:   {sys_status}")

    if srv == "running":
        uptime = status.get("uptime_seconds", 0)
        uptime_str = str(timedelta(seconds=int(uptime)))
        print(f"  Uptime:          {uptime_str}")
        print(f"  Trading:         {'YES' if status.get('is_trading') else 'NO'}")
        print(f"  Strategies:      {status.get('active_strategies', 0)} active")
        print(f"  Open Positions:  {status.get('open_positions', 0)}")
        print(f"  Daily P&L:       ${status.get('daily_pnl_usd', 0.0):.2f}")
        print(f"  Errors:          {status.get('error_count', 0)}")
        last_sig = status.get("last_signal")
        print(f"  Last Signal:     {last_sig or 'none'}")

    print()

    # Trade info from local DB
    ti = status.get("trade_info", {})
    print(f"  Paper Trades:    {ti.get('total_paper_trades', 0)}")
    print(f"  Paper P&L:       ${ti.get('total_pnl', 0.0):.2f}")
    lt = ti.get("last_trade")
    if lt and isinstance(lt, dict):
        symbol = lt.get("symbol", lt.get("pair", "?"))
        print(f"  Last Trade:      {symbol}")

    print()

    # Models
    print("  Models:")
    models = status.get("models", [])
    for m in models:
        name = m["name"]
        st = m["status"]
        age = m.get("age_days")
        if st == "missing":
            marker = "MISS"
        elif st == "stale":
            marker = "STALE"
        else:
            marker = " OK "
        age_str = f" ({age:.0f}d old)" if age is not None else ""
        print(f"    [{marker}] {name}{age_str}")

    print()

    # Exchanges
    print("  Exchanges:")
    exchanges = status.get("exchanges", [])
    for ex in exchanges:
        name = ex["name"]
        net = ex["status"]
        key = ex.get("api_key", "?")
        print(f"    {name}: {net}, API key: {key}")

    print()
    print(f"  Timestamp: {status.get('timestamp', 'unknown')}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS system status dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Health server port")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    status = build_status(port=args.port)

    if args.json:
        print(json.dumps(status, indent=2, default=str))
    else:
        print_dashboard(status)

    return 0


if __name__ == "__main__":
    sys.exit(main())
