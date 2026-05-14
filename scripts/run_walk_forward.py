#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.walk_forward_harness import run_walk_forward


def main() -> int:
    p = argparse.ArgumentParser(description="Walk-forward harness skeleton + deploy bundle builder")
    p.add_argument("--config", default="unified_config.yaml")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--root-dir", default=".")
    args = p.parse_args()

    result = run_walk_forward(config_path=args.config, report_dir=args.report_dir, root_dir=args.root_dir)
    print(result.report_path)
    print(result.bundle_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
