#!/usr/bin/env python3
"""
Institutional readiness gate.

Checks required artifacts, report freshness/status, and manual attestations.
Writes a machine-readable report and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DISALLOWED_DRAFT_MARKERS = ("TODO", "TBD", "DRAFT", "<FILL>")


@dataclass
class CheckResult:
    check_id: str
    category: str
    ok: bool
    message: str
    details: Dict[str, Any]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
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
        pass
    try:
        dt = datetime.strptime(txt, "%Y%m%d_%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _check_required_artifact(repo_root: Path, spec: Dict[str, Any]) -> CheckResult:
    check_id = str(spec.get("id") or "artifact")
    rel = str(spec.get("path") or "")
    path = (repo_root / rel).resolve()
    min_bytes = int(spec.get("min_bytes") or 1)
    allow_draft = bool(spec.get("allow_draft", False))
    required_markers = [str(x) for x in (spec.get("required_markers") or [])]
    if not path.exists():
        return CheckResult(check_id, "required_artifact", False, f"missing file: {rel}", {"path": rel})
    size = path.stat().st_size
    if size < min_bytes:
        return CheckResult(
            check_id,
            "required_artifact",
            False,
            f"file too small: {rel} ({size} < {min_bytes})",
            {"path": rel, "size": size, "min_bytes": min_bytes},
        )
    text = _read_text(path)
    upper = text.upper()
    if not allow_draft:
        for marker in DISALLOWED_DRAFT_MARKERS:
            if marker in upper:
                return CheckResult(
                    check_id,
                    "required_artifact",
                    False,
                    f"draft marker '{marker}' found: {rel}",
                    {"path": rel, "marker": marker},
                )
    for marker in required_markers:
        if marker not in text:
            return CheckResult(
                check_id,
                "required_artifact",
                False,
                f"required marker missing '{marker}': {rel}",
                {"path": rel, "marker": marker},
            )
    return CheckResult(check_id, "required_artifact", True, f"ok: {rel}", {"path": rel, "size": size})


def _check_report(repo_root: Path, spec: Dict[str, Any]) -> CheckResult:
    check_id = str(spec.get("id") or "report")
    rel = str(spec.get("path") or "")
    path = (repo_root / rel).resolve()
    if not path.exists():
        return CheckResult(check_id, "report", False, f"missing report: {rel}", {"path": rel})
    try:
        payload = json.loads(_read_text(path))
    except Exception as exc:
        return CheckResult(check_id, "report", False, f"invalid JSON report: {rel}", {"path": rel, "error": str(exc)})

    required_status = str(spec.get("required_status") or "").strip().upper()
    if required_status:
        got = str(payload.get("status") or "").strip().upper()
        if got != required_status:
            return CheckResult(
                check_id,
                "report",
                False,
                f"status mismatch in {rel}: got {got or 'UNKNOWN'}, need {required_status}",
                {"path": rel, "got": got, "required": required_status},
            )

    max_age_h = spec.get("max_age_hours")
    if max_age_h is not None:
        ts_keys = [str(x) for x in (spec.get("timestamp_keys") or [])]
        stamp = None
        for key in ts_keys:
            if payload.get(key) is not None:
                stamp = payload.get(key)
                break
        dt = _parse_timestamp(stamp)
        if dt is None:
            return CheckResult(
                check_id,
                "report",
                False,
                f"timestamp missing/invalid in {rel}",
                {"path": rel, "timestamp_keys": ts_keys},
            )
        age_h = (_now_utc() - dt).total_seconds() / 3600.0
        if age_h > float(max_age_h):
            return CheckResult(
                check_id,
                "report",
                False,
                f"report stale: {rel} age {age_h:.2f}h > {float(max_age_h):.2f}h",
                {"path": rel, "age_hours": age_h, "max_age_hours": float(max_age_h)},
            )
    return CheckResult(check_id, "report", True, f"ok: {rel}", {"path": rel})


def _check_manual_attestation(repo_root: Path, spec: Dict[str, Any], allow_unverified: bool) -> CheckResult:
    check_id = str(spec.get("id") or "manual")
    rel = str(spec.get("path") or "")
    path = (repo_root / rel).resolve()
    required_markers = [str(x) for x in (spec.get("required_markers") or [])]
    if not path.exists():
        if allow_unverified:
            return CheckResult(check_id, "manual_attestation", True, f"skipped missing manual attestation: {rel}", {"path": rel, "skipped": True})
        return CheckResult(check_id, "manual_attestation", False, f"missing manual attestation: {rel}", {"path": rel})
    text = _read_text(path)
    if allow_unverified:
        return CheckResult(check_id, "manual_attestation", True, f"manual attestation check skipped: {rel}", {"path": rel, "skipped": True})
    for marker in required_markers:
        if marker not in text:
            return CheckResult(
                check_id,
                "manual_attestation",
                False,
                f"manual attestation marker missing '{marker}': {rel}",
                {"path": rel, "marker": marker},
            )
    return CheckResult(check_id, "manual_attestation", True, f"ok: {rel}", {"path": rel})


def _run_pre_live_check(repo_root: Path, config: str, profile: str | None = None) -> CheckResult:
    script = repo_root / "scripts" / "pre_live_check.py"
    if not script.exists():
        return CheckResult("PRELIVE-001", "pre_live", False, "missing pre_live_check.py", {"path": str(script)})
    cmd = [sys.executable, str(script), "--config", str(config)]
    if profile:
        cmd.extend(["--profile", str(profile)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return CheckResult("PRELIVE-001", "pre_live", False, "pre_live_check failed", {"returncode": proc.returncode, "output": output[-4000:]})
    return CheckResult("PRELIVE-001", "pre_live", True, "pre_live_check passed", {"returncode": proc.returncode, "output": output[-4000:]})


def _refresh_soak_gate_report(repo_root: Path, config: str, profile: str | None = None) -> CheckResult:
    script = repo_root / "scripts" / "soak_gate.py"
    if not script.exists():
        return CheckResult(
            "SOAK-REFRESH-001",
            "report_refresh",
            False,
            "missing soak_gate.py",
            {"path": str(script)},
        )
    cmd = [sys.executable, str(script), "--config", str(config)]
    if profile:
        cmd.extend(["--profile", str(profile)])
    report_path = (repo_root / "reports" / "soak_gate_latest.json").resolve()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as exc:
        return CheckResult(
            "SOAK-REFRESH-001",
            "report_refresh",
            False,
            "soak gate refresh failed",
            {"error": str(exc)},
        )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and not report_path.exists():
        return CheckResult(
            "SOAK-REFRESH-001",
            "report_refresh",
            False,
            "soak gate refresh returned non-zero and no report was produced",
            {
                "returncode": int(proc.returncode),
                "output": output[-4000:],
                "report_path": str(report_path),
            },
        )
    message = "soak gate refreshed"
    if proc.returncode != 0:
        message = "soak gate refreshed with non-zero status (report generated)"
    return CheckResult(
        "SOAK-REFRESH-001",
        "report_refresh",
        True,
        message,
        {
            "returncode": int(proc.returncode),
            "output": output[-4000:],
            "report_path": str(report_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Institutional readiness gate")
    parser.add_argument("--manifest", default="docs/institutional/evidence_manifest.json")
    parser.add_argument("--config", default="unified_config.yaml")
    parser.add_argument("--profile", default=None, help="Config profile name or YAML path overlay")
    parser.add_argument("--output", default="reports/institutional_readiness_latest.json")
    parser.add_argument("--allow-manual-unverified", action="store_true")
    parser.add_argument("--skip-pre-live", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        print(f"FAIL missing manifest: {manifest_path}")
        return 1

    try:
        manifest = json.loads(_read_text(manifest_path))
    except Exception as exc:
        print(f"FAIL invalid manifest JSON: {manifest_path} ({exc})")
        return 1

    checks: List[CheckResult] = []
    checks.append(_refresh_soak_gate_report(repo_root, args.config, args.profile))
    if not args.skip_pre_live:
        checks.append(_run_pre_live_check(repo_root, args.config, args.profile))
    for item in manifest.get("required_artifacts") or []:
        checks.append(_check_required_artifact(repo_root, item))
    for item in manifest.get("report_checks") or []:
        checks.append(_check_report(repo_root, item))
    for item in manifest.get("manual_attestations") or []:
        checks.append(_check_manual_attestation(repo_root, item, bool(args.allow_manual_unverified)))

    total = len(checks)
    passed = sum(1 for c in checks if c.ok)
    failed = [c for c in checks if not c.ok]
    status = "PASS" if not failed else "FAIL"

    out = {
        "checked_at": _now_utc().isoformat(),
        "status": status,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": len(failed),
            "score_pct": round((100.0 * passed / total), 2) if total > 0 else 0.0,
            "allow_manual_unverified": bool(args.allow_manual_unverified),
            "skip_pre_live": bool(args.skip_pre_live),
        },
        "checks": [
            {
                "id": c.check_id,
                "category": c.category,
                "ok": c.ok,
                "message": c.message,
                "details": c.details,
            }
            for c in checks
        ],
    }

    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for c in checks:
        prefix = "PASS" if c.ok else "FAIL"
        print(f"{prefix} [{c.category}] {c.check_id}: {c.message}")
    print(f"Readiness: {status} ({passed}/{total})")
    print(f"Report: {out_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
