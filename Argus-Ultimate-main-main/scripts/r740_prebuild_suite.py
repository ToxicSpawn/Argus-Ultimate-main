#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "ok": proc.returncode == 0,
        "stdout_tail": out[-4000:],
        "stderr_tail": err[-4000:],
    }


def _extract_last_line(stdout_tail: str) -> str:
    lines = [x.strip() for x in str(stdout_tail or "").splitlines() if x.strip()]
    return lines[-1] if lines else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R740 prebuild readiness suite")
    parser.add_argument("--config", default="unified_config.yaml")
    parser.add_argument("--profile", default="", help="Optional profile name/path for main.py validate")
    parser.add_argument("--manifest", default="docs/hardware/R740_PREBUILD_MANIFEST.yaml")
    parser.add_argument("--bundle-root", default="deploy/r740_bundle")
    parser.add_argument("--bundle-check-output", default="reports/infra/r740_bundle_check_latest.json")
    parser.add_argument("--readiness-manifest", default="docs/institutional/evidence_manifest.json")
    parser.add_argument("--run-readiness", action="store_true")
    parser.add_argument("--suite-output", default="reports/infra/r740_prebuild_suite_latest.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable
    profile_args = ["--profile", args.profile] if str(args.profile).strip() else []

    steps: List[Dict[str, Any]] = []
    overall_ok = True
    bundle_path = ""

    validate_cmd = [py, "main.py", "validate", "--config", args.config] + profile_args
    step = _run(validate_cmd, repo_root)
    step["step_id"] = "validate"
    steps.append(step)
    if not step["ok"]:
        overall_ok = False

    if overall_ok:
        prep_cmd = [
            py,
            "scripts/r740_prepare_bundle.py",
            "--manifest",
            args.manifest,
            "--output-root",
            args.bundle_root,
        ]
        step = _run(prep_cmd, repo_root)
        step["step_id"] = "bundle_build"
        bundle_path = _extract_last_line(step.get("stdout_tail", ""))
        step["bundle_path"] = bundle_path
        steps.append(step)
        if not step["ok"] or not bundle_path:
            overall_ok = False
            if not bundle_path:
                step["ok"] = False
                step["stderr_tail"] = (step.get("stderr_tail") or "") + "\nfailed to detect bundle path from output"

    if overall_ok:
        check_cmd = [
            py,
            "scripts/r740_bundle_check.py",
            "--bundle",
            bundle_path,
            "--output",
            args.bundle_check_output,
        ]
        step = _run(check_cmd, repo_root)
        step["step_id"] = "bundle_check"
        steps.append(step)
        if not step["ok"]:
            overall_ok = False

    if overall_ok and args.run_readiness:
        readiness_cmd = [
            py,
            "scripts/institutional_readiness_check.py",
            "--config",
            args.config,
            "--manifest",
            args.readiness_manifest,
            "--output",
            "reports/institutional_readiness_latest.json",
            "--allow-manual-unverified",
            "--skip-pre-live",
        ] + profile_args
        step = _run(readiness_cmd, repo_root)
        step["step_id"] = "institutional_readiness_soft"
        steps.append(step)
        if not step["ok"]:
            overall_ok = False

    status = "PASS" if overall_ok else "FAIL"
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "bundle_path": bundle_path,
        "steps": steps,
    }
    out_path = (repo_root / args.suite_output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(status)
    print(str(out_path))
    if bundle_path:
        print(bundle_path)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
