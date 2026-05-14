from __future__ import annotations

import json
from pathlib import Path

from argus_live.evidence.session_summary import summarize_jsonl


def _read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    return json.loads(p.read_text(encoding="utf-8"))


def build_evidence_pack(*, output_path: str | Path, manifest_path: str | Path, operator_state_path: str | Path, journal_path: str | Path, audit_path: str | Path) -> None:
    payload = {
        "manifest": _read_json(manifest_path),
        "operator_state": _read_json(operator_state_path),
        "journal_summary": summarize_jsonl(journal_path),
        "audit_summary": summarize_jsonl(audit_path),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
