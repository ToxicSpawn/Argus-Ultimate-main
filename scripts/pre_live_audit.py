#!/usr/bin/env python3
"""
One-command pre-live artifact audit.

Runs the same pre-live integrity gates as live startup and writes a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(str(raw or "").strip())
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _run_check(name: str, fn: Callable[[], None]) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        fn()
        return {
            "name": name,
            "status": "PASS",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "details": "",
        }
    except SystemExit as e:
        code = int(getattr(e, "code", 1) or 1)
        return {
            "name": name,
            "status": "FAIL",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "details": f"SystemExit({code})",
        }
    except Exception as e:
        return {
            "name": name,
            "status": "FAIL",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "details": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict pre-live artifact audit")
    parser.add_argument("--config", default="unified_config.yaml", help="Config path")
    parser.add_argument("--profile", default=None, help="Optional profile name/path")
    parser.add_argument("--output", default="reports/pre_live_audit_latest.json", help="Audit report JSON path")
    args = parser.parse_args()

    from core.config_manager import load_unified_yaml, validate_unified_config_dict
    from main import _kill_switch_enabled, enforce_pre_live_artifact_gates_or_exit

    cfg_path = _resolve_path(ROOT, str(args.config))
    out_path = _resolve_path(ROOT, str(args.output))

    checks: List[Dict[str, Any]] = []

    def _check_config() -> None:
        y = load_unified_yaml(str(cfg_path), profile=args.profile)
        validate_unified_config_dict(y, source=str(cfg_path))

    def _check_kill_switch_clear() -> None:
        if _kill_switch_enabled():
            raise RuntimeError("KILL_SWITCH file is present")

    def _check_pre_live_gates() -> None:
        enforce_pre_live_artifact_gates_or_exit(str(cfg_path), profile=args.profile)

    checks.append(_run_check("config_validation", _check_config))
    checks.append(_run_check("kill_switch_clear", _check_kill_switch_clear))
    checks.append(_run_check("pre_live_artifact_gates", _check_pre_live_gates))

    overall_pass = all(str(c.get("status")) == "PASS" for c in checks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": bool(overall_pass),
        "config_file": str(cfg_path),
        "profile": str(args.profile or ""),
        "checks": checks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(out_path))
    if overall_pass:
        print("PRE_LIVE_AUDIT: PASS")
        return 0
    print("PRE_LIVE_AUDIT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

