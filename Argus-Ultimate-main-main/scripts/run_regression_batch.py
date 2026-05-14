from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from argus_live.simulation.batch_runner import ScenarioBatchRunner
from argus_live.simulation.regression_library import build_regression_library


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ARGUS hostile regression batch")
    parser.add_argument("--artifacts-root", default="artifacts/regression")
    parser.add_argument("--batch-name", default="ci-regression")
    parser.add_argument("--strict-governance", action="store_true", default=True)
    args = parser.parse_args()

    result = ScenarioBatchRunner(Path(args.artifacts_root), strict_governance=args.strict_governance).run(
        batch_name=args.batch_name,
        scenarios=build_regression_library(),
    )
    summary = {
        "batch_name": result.batch_name,
        "overall_score": result.overall_score,
        "pass_rate": result.pass_rate,
        "summary_path": result.summary_path,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.pass_rate >= 100.0 and result.overall_score >= 70.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
