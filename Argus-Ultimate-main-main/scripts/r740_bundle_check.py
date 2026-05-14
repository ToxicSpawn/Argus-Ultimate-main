#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


REQUIRED_FILES = {
    "render/grub_cmdline.txt",
    "render/99-argus-lowlatency.conf",
    "render/argus.service.override.conf",
    "render/argus-pin-irqs.sh",
    "render/argus-pin-irqs.service",
    "render/50-argus-10gbe.yaml",
    "render/chrony.argus.conf",
    "render/ptp4l.argus.conf",
    "render/phc2sys.argus.service",
    "render/argus-ethtool-apply.sh",
    "manifest.resolved.json",
    "build_info.json",
    "APPLY_STEPS.md",
    "hashes.txt",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_cpu_list(raw: str) -> Set[int]:
    out: Set[int] = set()
    txt = str(raw or "").strip()
    if not txt:
        return out
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            if hi < lo:
                raise ValueError(f"Invalid CPU range '{part}'")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _validate_ip_cidr(raw: str) -> bool:
    try:
        ipaddress.ip_interface(str(raw))
        return True
    except Exception:
        return False


def _validate_ip(raw: str) -> bool:
    try:
        ipaddress.ip_address(str(raw))
        return True
    except Exception:
        return False


def _validate_iface(raw: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9._-]+$", str(raw or "")))


def _parse_hashes(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        m = re.match(r"^([0-9a-fA-F]{64})\s{2}(.+)$", row)
        if not m:
            raise ValueError(f"invalid hashes.txt line {idx}: {line!r}")
        rows.append((m.group(2).strip().replace("\\", "/"), m.group(1).lower()))
    return rows


def _check_hashes(bundle: Path, fail: List[str]) -> None:
    hashes_path = bundle / "hashes.txt"
    if not hashes_path.exists():
        fail.append("hashes.txt missing")
        return
    try:
        rows = _parse_hashes(hashes_path)
    except Exception as exc:
        fail.append(f"invalid hashes.txt format: {exc}")
        return

    indexed = {name: digest for name, digest in rows}
    all_files = {
        str(p.relative_to(bundle)).replace("\\", "/")
        for p in bundle.rglob("*")
        if p.is_file() and p.name != "hashes.txt"
    }
    missing_entries = sorted(x for x in all_files if x not in indexed)
    for rel in missing_entries:
        fail.append(f"hash missing for file: {rel}")

    for rel, expected in rows:
        file_path = bundle / rel
        if not file_path.exists():
            fail.append(f"hash references missing file: {rel}")
            continue
        got = _sha256(file_path)
        if got != expected:
            fail.append(f"hash mismatch: {rel}")


def find_latest_bundle(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Bundle root not found: {root}")
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"No bundle directories under: {root}")
    return dirs[-1]


def check_bundle(bundle: Path) -> Dict[str, Any]:
    fail: List[str] = []
    for rel in sorted(REQUIRED_FILES):
        if not (bundle / rel).exists():
            fail.append(f"missing required file: {rel}")

    manifest_path = bundle / "manifest.resolved.json"
    values: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            values = dict(payload.get("resolved_values") or {})
        except Exception as exc:
            fail.append(f"invalid manifest.resolved.json: {exc}")
    else:
        fail.append("manifest.resolved.json missing")

    if values:
        if not _validate_iface(values.get("IFACE_PRIMARY")):
            fail.append("invalid IFACE_PRIMARY")
        if not _validate_iface(values.get("IFACE_SECONDARY")):
            fail.append("invalid IFACE_SECONDARY")
        if not _validate_ip_cidr(values.get("IP_PRIMARY_CIDR")):
            fail.append("invalid IP_PRIMARY_CIDR")
        if not _validate_ip_cidr(values.get("IP_SECONDARY_CIDR")):
            fail.append("invalid IP_SECONDARY_CIDR")
        if not _validate_ip(values.get("GATEWAY4")):
            fail.append("invalid GATEWAY4")
        if not _validate_iface(values.get("PTP_IFACE")):
            fail.append("invalid PTP_IFACE")
        try:
            mtu = int(str(values.get("MTU", "0")))
            if mtu < 1500 or mtu > 9216:
                fail.append("MTU outside expected range 1500-9216")
        except Exception:
            fail.append("invalid MTU")
        dns_csv = str(values.get("DNS_SERVERS_CSV", "")).strip()
        if not dns_csv:
            fail.append("DNS_SERVERS_CSV is empty")

        try:
            os_cpus = _parse_cpu_list(str(values.get("OS_CPUS", "")))
            exec_cpus = _parse_cpu_list(str(values.get("EXEC_CPUS", "")))
            iso_cpus = _parse_cpu_list(str(values.get("CPU_ISOLATION", "")))
            if not os_cpus:
                fail.append("OS_CPUS is empty")
            if not exec_cpus:
                fail.append("EXEC_CPUS is empty")
            if not iso_cpus:
                fail.append("CPU_ISOLATION is empty")
            overlap = os_cpus.intersection(exec_cpus)
            if overlap:
                fail.append(f"OS_CPUS and EXEC_CPUS overlap: {sorted(overlap)}")
            overlap_iso_os = os_cpus.intersection(iso_cpus)
            if overlap_iso_os:
                fail.append(f"OS_CPUS and CPU_ISOLATION overlap: {sorted(overlap_iso_os)}")
            if iso_cpus and exec_cpus and not iso_cpus.issubset(exec_cpus):
                fail.append("CPU_ISOLATION must be subset of EXEC_CPUS")
        except Exception as exc:
            fail.append(f"invalid CPU list: {exc}")

        grub_path = bundle / "render" / "grub_cmdline.txt"
        if grub_path.exists():
            grub = grub_path.read_text(encoding="utf-8", errors="ignore")
            cpu_iso = str(values.get("CPU_ISOLATION", "")).strip()
            if cpu_iso:
                expected_tokens = [
                    f"isolcpus={cpu_iso}",
                    f"nohz_full={cpu_iso}",
                    f"rcu_nocbs={cpu_iso}",
                ]
                for token in expected_tokens:
                    if token not in grub:
                        fail.append(f"grub_cmdline missing token: {token}")

    _check_hashes(bundle, fail)

    status = "PASS" if not fail else "FAIL"
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(bundle.resolve()),
        "status": status,
        "fail_reasons": fail,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Validate generated R740 prebuild bundle")
    p.add_argument("--bundle", default="", help="Path to specific bundle (default: latest under --bundle-root)")
    p.add_argument("--bundle-root", default="deploy/r740_bundle")
    p.add_argument("--output", default="reports/infra/r740_bundle_check_latest.json")
    args = p.parse_args()

    if args.bundle:
        bundle = Path(args.bundle)
    else:
        bundle = find_latest_bundle(Path(args.bundle_root))
    result = check_bundle(bundle)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(result["status"])
    print(str(out.resolve()))
    if result["status"] != "PASS":
        for r in result["fail_reasons"]:
            print(f"- {r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
