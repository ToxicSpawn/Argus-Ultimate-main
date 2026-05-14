#!/usr/bin/env python3
"""
Dual-approval promotion workflow for restricted_live -> live.

Commands:
- approve: append an approval/reject vote to artifact approvals log.
- finalize: produce promotion_certificate.json when thresholds are met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _append_approval(
    *,
    artifact_dir: Path,
    approver_id: str,
    outcome: str,
    note: str,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    approvals_path = artifact_dir / "promotion_approvals.jsonl"
    previous_hash = ""
    if approvals_path.exists():
        lines = [ln for ln in approvals_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            previous_hash = _sha256_bytes(lines[-1].encode("utf-8"))

    row = {
        "ts": _now_iso(),
        "approver_id": str(approver_id).strip(),
        "outcome": str(outcome).strip().lower(),
        "note": str(note or ""),
        "previous_hash": previous_hash,
    }
    line = json.dumps(row, ensure_ascii=True)
    with approvals_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return approvals_path


def _read_approvals(approvals_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not approvals_path.exists():
        return rows
    for raw in approvals_path.read_text(encoding="utf-8").splitlines():
        txt = str(raw or "").strip()
        if not txt:
            continue
        try:
            row = json.loads(txt)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _finalize(
    *,
    artifact_dir: Path,
    min_approvals: int,
    required_stage: str,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    decision_path = artifact_dir / "promotion_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(f"missing promotion decision artifact: {decision_path}")

    decision = _load_json(decision_path, {})
    approvals = _read_approvals(artifact_dir / "promotion_approvals.jsonl")
    approves = [
        a for a in approvals
        if str(a.get("outcome", "")).strip().lower() == "approve"
        and str(a.get("approver_id", "")).strip()
    ]
    rejects = [
        a for a in approvals
        if str(a.get("outcome", "")).strip().lower() == "reject"
    ]
    unique_approvers = sorted(
        {
            str(a.get("approver_id", "")).strip()
            for a in approves
            if str(a.get("approver_id", "")).strip()
        }
    )

    status = "APPROVED" if (len(unique_approvers) >= int(min_approvals) and not rejects) else "HOLD"
    reasons: List[str] = []
    if len(unique_approvers) < int(min_approvals):
        reasons.append(f"insufficient_unique_approvals:{len(unique_approvers)}/{int(min_approvals)}")
    if rejects:
        reasons.append(f"reject_votes:{len(rejects)}")

    cert = {
        "issued_at": _now_iso(),
        "status": status,
        "stage": str(required_stage or "restricted_live_to_live"),
        "min_approvals": int(min_approvals),
        "unique_approvers": list(unique_approvers),
        "approvals": approvals,
        "reject_count": len(rejects),
        "reasons": reasons,
        "decision_path": str(decision_path),
        "decision_sha256": _sha256_file(decision_path),
        "approvals_log_path": str((artifact_dir / "promotion_approvals.jsonl")),
        "approvals_log_sha256": _sha256_file(artifact_dir / "promotion_approvals.jsonl")
        if (artifact_dir / "promotion_approvals.jsonl").exists()
        else "",
        "decision": decision,
    }
    cert_path = artifact_dir / "promotion_certificate.json"
    cert_path.write_text(json.dumps(cert, ensure_ascii=True, indent=2), encoding="utf-8")
    return cert_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Promotion dual-approval workflow")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_approve = sub.add_parser("approve", help="append approval/reject vote")
    p_approve.add_argument("--artifact-dir", required=True)
    p_approve.add_argument("--approver-id", required=True)
    p_approve.add_argument("--outcome", choices=["approve", "reject"], default="approve")
    p_approve.add_argument("--note", default="")

    p_finalize = sub.add_parser("finalize", help="finalize certificate")
    p_finalize.add_argument("--artifact-dir", required=True)
    p_finalize.add_argument("--min-approvals", type=int, default=2)
    p_finalize.add_argument("--required-stage", default="restricted_live_to_live")

    args = parser.parse_args()
    artifact_dir = Path(str(args.artifact_dir)).resolve()

    if args.cmd == "approve":
        if not str(args.approver_id).strip():
            print("ERROR: --approver-id is required")
            return 1
        out = _append_approval(
            artifact_dir=artifact_dir,
            approver_id=str(args.approver_id).strip(),
            outcome=str(args.outcome).strip().lower(),
            note=str(args.note or ""),
        )
        print(str(out))
        return 0

    if args.cmd == "finalize":
        try:
            out = _finalize(
                artifact_dir=artifact_dir,
                min_approvals=max(1, int(args.min_approvals or 2)),
                required_stage=str(args.required_stage or "restricted_live_to_live"),
            )
        except Exception as e:
            print(f"ERROR: {e}")
            return 1
        print(str(out))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

