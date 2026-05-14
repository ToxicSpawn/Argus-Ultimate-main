#!/usr/bin/env python3
"""
Clean Paper Trading State — safe reset for a fresh paper run.

Clears stale paper positions, checkpoints, strategy states, and order intents
from SQLite databases under data/. Never touches model files or config.

Usage:
    py -B scripts/clean_paper_state.py
    py -B scripts/clean_paper_state.py --dry-run     # show what would be cleaned
    py -B scripts/clean_paper_state.py --all          # also remove paperloop state DBs
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _clean_table(db_path: Path, table: str, where: str = "", dry_run: bool = False) -> int:
    """Delete rows from a table. Returns row count deleted."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # Check table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not cursor.fetchone():
            conn.close()
            return 0
        # Count rows
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        cursor.execute(sql)
        count = cursor.fetchone()[0]
        if count > 0 and not dry_run:
            del_sql = f"DELETE FROM {table}"
            if where:
                del_sql += f" WHERE {where}"
            cursor.execute(del_sql)
            conn.commit()
        conn.close()
        return count
    except sqlite3.OperationalError:
        return 0


def _vacuum_db(db_path: Path, dry_run: bool = False) -> None:
    """Reclaim space after deletions."""
    if not db_path.exists() or dry_run:
        return
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("VACUUM")
        conn.close()
    except Exception:
        pass


def clean_paper_state(dry_run: bool = False, clean_all: bool = False) -> List[Tuple[str, str, int]]:
    """
    Clean paper trading state. Returns list of (db_file, description, rows_affected).
    """
    actions: List[Tuple[str, str, int]] = []
    prefix = "[DRY RUN] " if dry_run else ""

    # 1. unified_trades.db — paper trades and positions
    trades_db = DATA_DIR / "unified_trades.db"
    if trades_db.exists():
        n = _clean_table(trades_db, "trades", "mode='paper'", dry_run)
        if n:
            actions.append(("unified_trades.db", f"{prefix}Cleared {n} paper trades", n))
        n = _clean_table(trades_db, "positions", "mode='paper'", dry_run)
        if n:
            actions.append(("unified_trades.db", f"{prefix}Cleared {n} paper positions", n))
        # Also try generic position/order tables
        for table in ["open_positions", "paper_positions", "order_intents", "pending_orders"]:
            n = _clean_table(trades_db, table, dry_run=dry_run)
            if n:
                actions.append(("unified_trades.db", f"{prefix}Cleared {n} rows from {table}", n))
        _vacuum_db(trades_db, dry_run)

    # 2. paper_trades.db — dedicated paper trade DB
    paper_db = DATA_DIR / "paper_trades.db"
    if paper_db.exists():
        for table in ["trades", "positions", "orders", "fills"]:
            n = _clean_table(paper_db, table, dry_run=dry_run)
            if n:
                actions.append(("paper_trades.db", f"{prefix}Cleared {n} rows from {table}", n))
        _vacuum_db(paper_db, dry_run)

    # 3. checkpoints.db — stale checkpoint data
    ckpt_db = DATA_DIR / "checkpoints.db"
    if ckpt_db.exists():
        for table in ["checkpoints", "state_snapshots", "checkpoint_data"]:
            n = _clean_table(ckpt_db, table, dry_run=dry_run)
            if n:
                actions.append(("checkpoints.db", f"{prefix}Cleared {n} rows from {table}", n))
        _vacuum_db(ckpt_db, dry_run)

    # 4. strategy_states.db — stale strategy state
    strat_db = DATA_DIR / "strategy_states.db"
    if strat_db.exists():
        for table in ["strategy_states", "strategy_metrics", "signals"]:
            n = _clean_table(strat_db, table, dry_run=dry_run)
            if n:
                actions.append(("strategy_states.db", f"{prefix}Cleared {n} rows from {table}", n))
        _vacuum_db(strat_db, dry_run)

    # 5. fills.db — execution fill records
    fills_db = DATA_DIR / "fills.db"
    if fills_db.exists():
        for table in ["fills", "fill_events"]:
            n = _clean_table(fills_db, table, dry_run=dry_run)
            if n:
                actions.append(("fills.db", f"{prefix}Cleared {n} rows from {table}", n))
        _vacuum_db(fills_db, dry_run)

    # 6. command_bus.db — stale commands
    cmd_db = DATA_DIR / "command_bus.db"
    if cmd_db.exists():
        for table in ["commands", "command_log"]:
            n = _clean_table(cmd_db, table, dry_run=dry_run)
            if n:
                actions.append(("command_bus.db", f"{prefix}Cleared {n} rows from {table}", n))
        _vacuum_db(cmd_db, dry_run)

    # 7. Paper equity curve CSV
    equity_csv = DATA_DIR / "paper_equity_curve.csv"
    if equity_csv.exists():
        if not dry_run:
            equity_csv.unlink()
        actions.append(("paper_equity_curve.csv", f"{prefix}Removed paper equity curve", 1))

    # 8. Paper results JSON
    results_json = DATA_DIR / "paper_results.json"
    if results_json.exists():
        if not dry_run:
            results_json.unlink()
        actions.append(("paper_results.json", f"{prefix}Removed paper results", 1))

    # 9. Paper trades CSV
    paper_csv = DATA_DIR / "paper_trades.csv"
    if paper_csv.exists():
        if not dry_run:
            paper_csv.unlink()
        actions.append(("paper_trades.csv", f"{prefix}Removed paper trades CSV", 1))

    # 10. Stale paperloop state DBs (unified_paperloop_state_*.db)
    if clean_all:
        pattern = str(DATA_DIR / "unified_paperloop_state_*.db")
        for f in glob.glob(pattern):
            p = Path(f)
            if not dry_run:
                p.unlink()
            actions.append((p.name, f"{prefix}Removed stale paperloop state DB", 1))

    # 11. Reconciliation freeze
    recon_freeze = DATA_DIR / "reconciliation_freeze.json"
    if recon_freeze.exists():
        if not dry_run:
            recon_freeze.unlink()
        actions.append(("reconciliation_freeze.json", f"{prefix}Removed recon freeze", 1))

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean paper trading state for fresh run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without doing it")
    parser.add_argument("--all", action="store_true", help="Also remove stale paperloop state DBs")
    args = parser.parse_args()

    print(f"[{_ts()}] ARGUS Paper State Cleaner")
    print(f"[{_ts()}] Data directory: {DATA_DIR}")
    print()

    actions = clean_paper_state(dry_run=args.dry_run, clean_all=args.all)

    if not actions:
        print(f"[{_ts()}] Nothing to clean -- state is already fresh.")
    else:
        for db, desc, count in actions:
            print(f"  {desc}")
        print()
        total = sum(a[2] for a in actions)
        verb = "would be" if args.dry_run else "were"
        print(f"[{_ts()}] {len(actions)} actions, {total} items {verb} cleaned.")

    print(f"[{_ts()}] Model files and config: UNTOUCHED (safe).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
