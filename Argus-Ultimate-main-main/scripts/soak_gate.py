#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_manager import load_unified_yaml
from monitoring.soak_gate import (
    evaluate_soak_gate,
    load_thresholds_from_runtime,
    write_soak_gate_report,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate soak promotion gate from recent run telemetry")
    p.add_argument("--config", default="unified_config.yaml")
    p.add_argument("--profile", default=None, help="Config profile name or YAML path overlay")
    p.add_argument("--db", default="data/unified_trades.db")
    p.add_argument("--start-ts", type=float, default=0.0, help="Unix epoch seconds at soak start")
    p.add_argument("--output-dir", default="reports")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_unified_yaml(args.config, profile=args.profile)
    runtime = dict((cfg or {}).get("runtime") or {})
    thresholds = load_thresholds_from_runtime(runtime)

    start_ts = float(args.start_ts or 0.0)
    if start_ts <= 0:
        # Fallback: evaluate recent window aligned with configured min duration.
        start_ts = datetime.now(timezone.utc).timestamp() - max(60.0, float(thresholds.min_duration_seconds))

    report = evaluate_soak_gate(
        thresholds=thresholds,
        db_path=str(args.db),
        start_ts=start_ts,
    )

    report["config_path"] = str(Path(args.config))
    written = write_soak_gate_report(report, args.output_dir)
    report["report_paths"] = written

    # Keep output deterministic and machine-friendly.
    print(json.dumps(report, ensure_ascii=True))
    print(written["latest"])
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
