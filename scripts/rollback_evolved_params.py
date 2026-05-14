#!/usr/bin/env python3
"""
Rollback evolved params to a previous version from history.

Usage:
  python scripts/rollback_evolved_params.py [--index 0] [--path data/evolved_params.json] [--dry-run]
When evolution_version_history_size > 0, write_evolved_params writes copies to data/evolved_history/.
Index 0 = most recent in history (i.e. the version before current file). Index 1 = two versions back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback evolved params to previous version from history")
    parser.add_argument("--index", type=int, default=0, help="History index (0 = previous version)")
    parser.add_argument("--path", default="data/evolved_params.json", help="Evolved params file path")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be rolled back")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / args.path if not Path(args.path).is_absolute() else Path(args.path)

    try:
        from evolution.apply_evolved_strategies import get_version_history, rollback_to_previous
    except ImportError as e:
        print("ERROR: evolution.apply_evolved_strategies not found:", e, file=sys.stderr)
        return 1

    hist_dir = path.parent / "evolved_history"
    history = get_version_history(hist_dir, limit=args.index + 5)
    if args.index >= len(history):
        print("No history at index %s (have %s versions in %s). Enable evolution_version_history_size when running evolution."
              % (args.index, len(history), hist_dir))
        return 1

    payload = history[args.index]
    best = payload.get("best_params") or {}
    ts = payload.get("timestamp_utc", "?")
    print("Version at index %s: timestamp_utc=%s, params=%s" % (args.index, ts, list(best.keys())[:8]))

    if args.dry_run:
        print("Dry-run: would write to %s" % path)
        return 0

    result = rollback_to_previous(path=path, version_history_dir=str(hist_dir), index=args.index)
    if result is None:
        print("Rollback failed.", file=sys.stderr)
        return 1
    print("Rolled back to version index=%s. Restart the bot to use rolled-back params." % args.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
