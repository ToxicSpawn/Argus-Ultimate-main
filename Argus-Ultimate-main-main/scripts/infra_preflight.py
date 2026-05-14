#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_checked_at(raw: Any) -> datetime | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    try:
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _bool(v: Any) -> bool:
    return bool(v is True)


def evaluate(report: Dict[str, Any], *, max_age_hours: float, max_clock_offset_us: float, require_onload: bool, require_hugepages: bool) -> Dict[str, Any]:
    fail_reasons: List[str] = []
    checks = dict(report.get("checks") or {})
    details = dict(report.get("details") or {})

    checked_at = _parse_checked_at(report.get("checked_at"))
    now = datetime.now(timezone.utc)
    if checked_at is None:
        fail_reasons.append("verification report has missing/invalid checked_at")
        age_hours = None
    else:
        age_hours = (now - checked_at).total_seconds() / 3600.0
        if age_hours > float(max_age_hours):
            fail_reasons.append(f"verification report stale ({age_hours:.2f}h > {max_age_hours:.2f}h)")

    required = [
        ("platform_linux", _bool(checks.get("platform_linux"))),
        ("kernel_isolation_configured", _bool(checks.get("kernel_isolation_configured"))),
        ("cpu_governor_all_performance", _bool(checks.get("cpu_governor_all_performance"))),
        ("irqbalance_disabled", _bool(checks.get("irqbalance_disabled"))),
        ("ntp_synced", _bool(checks.get("ntp_synced"))),
        ("hardware_timestamping", _bool(checks.get("hardware_timestamping"))),
    ]
    for name, ok in required:
        if not ok:
            fail_reasons.append(f"required check failed: {name}")

    clock = dict(details.get("clock_sync") or {})
    offset = clock.get("clock_offset_us")
    if offset is None:
        fail_reasons.append("clock offset unavailable (chrony not reporting)")
    else:
        if abs(float(offset)) > float(max_clock_offset_us):
            fail_reasons.append(f"clock offset {float(offset):.2f}us exceeds threshold {float(max_clock_offset_us):.2f}us")

    if require_onload and not _bool(checks.get("onload_loaded")):
        fail_reasons.append("onload kernel-bypass module not loaded")
    if require_hugepages and not _bool(checks.get("hugepages_configured")):
        fail_reasons.append("hugepages not configured")

    status = "PASS" if not fail_reasons else "FAIL"
    return {
        "checked_at": now.isoformat(),
        "status": status,
        "source_status": str(report.get("status") or "UNKNOWN"),
        "thresholds": {
            "max_age_hours": float(max_age_hours),
            "max_clock_offset_us": float(max_clock_offset_us),
            "require_onload": bool(require_onload),
            "require_hugepages": bool(require_hugepages),
        },
        "age_hours": age_hours,
        "fail_reasons": fail_reasons,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Fail-closed infra preflight gate from verification report")
    p.add_argument("--report", default="reports/infra/verification_latest.json")
    p.add_argument("--output", default="reports/infra/infra_preflight_latest.json")
    p.add_argument("--max-age-hours", type=float, default=24.0)
    p.add_argument("--max-clock-offset-us", type=float, default=250.0)
    p.add_argument("--require-onload", action="store_true")
    p.add_argument("--require-hugepages", action="store_true")
    args = p.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"FAIL: verification report not found: {report_path}")
        return 1

    report = _load_json(report_path)
    result = evaluate(
        report,
        max_age_hours=float(args.max_age_hours),
        max_clock_offset_us=float(args.max_clock_offset_us),
        require_onload=bool(args.require_onload),
        require_hugepages=bool(args.require_hugepages),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")

    print(result["status"])
    print(str(out.resolve()))
    if result["status"] != "PASS":
        for reason in result["fail_reasons"]:
            print(f"- {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
