#!/usr/bin/env python3
"""
Generate HMAC-SHA256 signature manifest for live config/bundle artifacts.

The manifest is consumed by main.py runtime.signature_gate before live startup.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hmac_sha256_hex(secret: str, message: str) -> str:
    return hmac.new(
        key=str(secret).encode("utf-8"),
        msg=str(message).encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()


def _resolve_rel(base: Path, raw: str) -> Path:
    p = Path(str(raw or "").strip())
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create live signature manifest")
    parser.add_argument("--config", default="unified_config.yaml")
    parser.add_argument("--output", default="reports/live_signature_manifest.json")
    parser.add_argument("--hmac-env", default="ARGUS_SIGNING_KEY")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Relative/absolute file path to include; can be repeated.",
    )
    parser.add_argument(
        "--latest-bundle-hashes",
        action="store_true",
        help="Also include deploy/bundles/<latest>/hashes.txt if available.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = _resolve_rel(repo_root, str(args.config))
    output_path = _resolve_rel(repo_root, str(args.output))
    secret = str(os.getenv(str(args.hmac_env), "")).strip()
    if not secret:
        print(f"ERROR: env var {args.hmac_env} is not set")
        return 1

    files: List[str] = []
    files.extend(str(x) for x in list(args.file or []))
    if str(cfg_path):
        try:
            files.insert(0, str(cfg_path.relative_to(repo_root)).replace("\\", "/"))
        except Exception:
            files.insert(0, str(cfg_path))

    if args.latest_bundle_hashes:
        bundles_root = repo_root / "deploy" / "bundles"
        if bundles_root.exists():
            candidates = [p for p in bundles_root.iterdir() if p.is_dir()]
            if candidates:
                latest = sorted(candidates)[-1]
                hashes = latest / "hashes.txt"
                if hashes.exists():
                    files.append(str(hashes.relative_to(repo_root)).replace("\\", "/"))

    normalized = []
    for raw in files:
        rel = str(raw or "").strip().replace("\\", "/")
        if not rel:
            continue
        if rel not in normalized:
            normalized.append(rel)

    records: Dict[str, Dict[str, str]] = {}
    for rel in normalized:
        path = _resolve_rel(repo_root, rel)
        if not path.exists():
            print(f"ERROR: file not found: {rel}")
            return 1
        sha = _sha256_file(path).lower()
        payload = f"{rel}:{sha}"
        sig = _hmac_sha256_hex(secret, payload).lower()
        records[rel] = {"sha256": sha, "hmac_sha256": sig}

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "hmac-sha256",
        "hmac_env_var": str(args.hmac_env),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

