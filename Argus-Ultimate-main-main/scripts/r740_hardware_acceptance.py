#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml  # type: ignore[import-untyped]


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("facts JSON root must be an object")
    return payload


def _read_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("spec YAML root must be a mapping")
    return payload


def _check_cpu(spec: Dict[str, Any], facts: Dict[str, Any], fails: List[str]) -> None:
    expected_model = str(spec.get("expected_model_contains") or "").strip().lower()
    min_logical = int(spec.get("min_logical_cpus") or 0)
    got_model = str((facts.get("cpu") or {}).get("model_name") or "").strip().lower()
    got_logical = int((facts.get("cpu") or {}).get("logical_cpus") or 0)
    if expected_model and expected_model not in got_model:
        fails.append(f"cpu model mismatch: expected contains '{expected_model}', got '{got_model}'")
    if min_logical and got_logical < min_logical:
        fails.append(f"logical CPU count below minimum: {got_logical} < {min_logical}")


def _check_memory(spec: Dict[str, Any], facts: Dict[str, Any], fails: List[str]) -> None:
    min_gib = float(spec.get("min_total_gib") or 0.0)
    got = float((facts.get("memory") or {}).get("mem_total_gib") or 0.0)
    if min_gib and got < min_gib:
        fails.append(f"memory below minimum: {got:.2f} GiB < {min_gib:.2f} GiB")


def _check_network(spec: Dict[str, Any], facts: Dict[str, Any], fails: List[str]) -> None:
    req = spec.get("required_interfaces") or []
    min_speed = int(spec.get("min_speed_mbps") or 0)
    iface_rows = (facts.get("network") or {}).get("interfaces") or []
    indexed = {str(row.get("name") or ""): row for row in iface_rows if isinstance(row, dict)}
    for iface in req:
        name = str(iface).strip()
        if not name:
            continue
        row = indexed.get(name)
        if row is None:
            fails.append(f"required interface missing: {name}")
            continue
        speed = int(row.get("speed_mbps") or 0)
        if min_speed and speed < min_speed:
            fails.append(f"interface {name} speed below minimum: {speed} < {min_speed} Mbps")


def _check_storage(spec: Dict[str, Any], facts: Dict[str, Any], fails: List[str]) -> None:
    min_ssd = int(spec.get("min_ssd_count") or 0)
    min_disk = int(spec.get("min_disk_count") or 0)
    rows = (facts.get("storage") or {}).get("lsblk") or []
    disks = [row for row in rows if isinstance(row, dict) and str(row.get("type") or "") == "disk"]
    if min_disk and len(disks) < min_disk:
        fails.append(f"disk count below minimum: {len(disks)} < {min_disk}")
    if min_ssd:
        ssd_count = 0
        for d in disks:
            rota = str(d.get("rota", "")).strip()
            if rota in {"0", "false", "False"}:
                ssd_count += 1
        if ssd_count < min_ssd:
            fails.append(f"ssd count below minimum: {ssd_count} < {min_ssd}")


def run_acceptance(spec: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
    fails: List[str] = []
    _check_cpu(dict(spec.get("cpu") or {}), facts, fails)
    _check_memory(dict(spec.get("memory") or {}), facts, fails)
    _check_network(dict(spec.get("network") or {}), facts, fails)
    _check_storage(dict(spec.get("storage") or {}), facts, fails)
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not fails else "FAIL",
        "fail_reasons": fails,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check captured host facts against R740 acceptance spec")
    parser.add_argument("--spec", default="docs/hardware/R740_ACCEPTANCE_SPEC.yaml")
    parser.add_argument("--facts", required=True)
    parser.add_argument("--output", default="reports/infra/r740_acceptance_latest.json")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    facts_path = Path(args.facts).resolve()
    if not spec_path.exists():
        print(f"FAIL missing spec: {spec_path}")
        return 1
    if not facts_path.exists():
        print(f"FAIL missing facts: {facts_path}")
        return 1

    spec = _read_yaml(spec_path)
    facts = _read_json(facts_path)
    result = run_acceptance(spec, facts)

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["status"])
    print(str(out))
    if result["status"] != "PASS":
        for r in result["fail_reasons"]:
            print(f"- {r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
