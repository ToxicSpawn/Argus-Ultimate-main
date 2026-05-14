#!/usr/bin/env python3
"""
Nightly institutional controls rollup with hash-chain retention.

Collects key reports, snapshots immutable copies, and appends a chain-linked
ledger entry under reports/control_rollups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_SOURCES = [
    "reports/soak_gate_latest.json",
    "reports/institutional_readiness_latest.json",
    "reports/walk_forward_latest.json",
    "reports/daily_runtime_summary.json",
    "reports/overnight_verification_summary.json",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _set_readonly(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode & 0o555)
    except Exception:
        pass


def _load_ledger(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        txt = str(line or "").strip()
        if not txt:
            continue
        try:
            row = json.loads(txt)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly controls rollup")
    parser.add_argument("--output-root", default="reports/control_rollups")
    parser.add_argument("--source", action="append", default=[], help="Add source report path (repeatable)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_root = (repo_root / str(args.output_root)).resolve()
    now = _now_utc()
    day_dir = output_root / now.strftime("%Y%m%d")
    run_dir = day_dir / now.strftime("%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    sources: List[str] = list(DEFAULT_SOURCES)
    for item in list(args.source or []):
        rel = str(item or "").strip()
        if rel and rel not in sources:
            sources.append(rel)

    copied: List[Tuple[str, str]] = []
    missing: List[str] = []
    for rel in sources:
        raw = str(rel or "").strip()
        if not raw:
            continue
        src_path = Path(raw)
        if src_path.is_absolute():
            src = src_path.resolve()
            rel_key = f"external/{src.name}"
        else:
            src = (repo_root / raw).resolve()
            rel_key = raw.replace("\\", "/")
        dst = (run_dir / rel_key).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            missing.append(raw)
            continue
        if src == dst:
            dst = (run_dir / "external" / src.name).resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append((rel_key, _sha256_file(dst)))
        _set_readonly(dst)

    manifest = {
        "generated_at": now.isoformat(),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "copied": [{"path": rel, "sha256": sha} for rel, sha in copied],
        "missing": list(missing),
    }
    manifest_path = run_dir / "rollup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    _set_readonly(manifest_path)

    ledger_path = output_root / "ledger.jsonl"
    ledger_rows = _load_ledger(ledger_path)
    prev_hash = ""
    if ledger_rows:
        prev_hash = str(ledger_rows[-1].get("entry_hash") or "")
    entry_payload = {
        "generated_at": now.isoformat(),
        "run_dir": str(run_dir),
        "manifest_sha256": _sha256_file(manifest_path),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "previous_entry_hash": prev_hash,
    }
    entry_blob = json.dumps(entry_payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    entry_hash = hashlib.sha256(entry_blob).hexdigest()
    entry = dict(entry_payload)
    entry["entry_hash"] = entry_hash
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    latest = output_root / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "generated_at": now.isoformat(),
                "run_dir": str(run_dir),
                "ledger_path": str(ledger_path),
                "entry_hash": entry_hash,
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(latest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
