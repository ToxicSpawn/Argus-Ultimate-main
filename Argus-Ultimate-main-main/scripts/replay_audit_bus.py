#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def _iter_events(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt:
                continue
            try:
                row = json.loads(txt)
                if isinstance(row, dict):
                    yield row
            except Exception:
                continue


def _event_ts(row: Dict[str, Any]) -> Optional[float]:
    v = row.get("timestamp")
    if isinstance(v, (int, float)):
        return float(v)
    payload = row.get("payload")
    if isinstance(payload, dict):
        pv = payload.get("timestamp")
        if isinstance(pv, (int, float)):
            return float(pv)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Replay JSONL audit bus events with optional timing")
    p.add_argument("--input", default="logs/audit_bus_latest.jsonl")
    p.add_argument("--speed", type=float, default=0.0, help="0 = no sleep; 1 = realtime; 2 = 2x")
    p.add_argument("--max-events", type=int, default=0, help="0 = all")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    prev_ts: Optional[float] = None
    count = 0
    speed = float(args.speed)
    for row in _iter_events(path):
        ts = _event_ts(row)
        if speed > 0 and prev_ts is not None and ts is not None:
            delay = max(0.0, (ts - prev_ts) / speed)
            if delay > 0:
                time.sleep(min(delay, 2.0))
        print(json.dumps(row, ensure_ascii=True, default=str))
        prev_ts = ts if ts is not None else prev_ts
        count += 1
        if args.max_events and count >= int(args.max_events):
            break

    print(f"replayed_events={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
