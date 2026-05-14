#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _run(cmd: list[str]) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")
    except Exception as exc:
        return 1, "", str(exc)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _parse_chrony_offset_us(raw: str) -> Optional[float]:
    for line in raw.splitlines():
        if "Last offset" not in line:
            continue
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(seconds|ms|us|ns)", line, flags=re.IGNORECASE)
        if not m:
            continue
        v = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "seconds":
            return v * 1_000_000.0
        if unit == "ms":
            return v * 1_000.0
        if unit == "us":
            return v
        if unit == "ns":
            return v / 1_000.0
    return None


def _parse_cpu_governors() -> Dict[str, Any]:
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
    governors: Dict[str, str] = {}
    for p in paths:
        cpu = p.parts[-3]
        governors[cpu] = _read_text(p).strip()
    total = len(governors)
    perf = sum(1 for g in governors.values() if g == "performance")
    return {
        "total_cpus_with_governor": total,
        "performance_count": perf,
        "all_performance": bool(total > 0 and perf == total),
        "governors": governors,
    }


def _check_irqbalance_disabled() -> Dict[str, Any]:
    rc, out, err = _run(["systemctl", "is-active", "irqbalance"])
    state = out.strip() if out.strip() else err.strip()
    return {"disabled": state != "active", "state": state or "unknown", "returncode": rc}


def _check_ntp_sync() -> Dict[str, Any]:
    rc, out, _ = _run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"])
    ntp_synced = (out.strip().lower() == "yes") if rc == 0 else False
    chr_rc, chr_out, chr_err = _run(["chronyc", "tracking"])
    offset_us = _parse_chrony_offset_us(chr_out) if chr_rc == 0 else None
    return {
        "ntp_synced": bool(ntp_synced),
        "chrony_available": bool(chr_rc == 0),
        "clock_offset_us": offset_us,
        "chrony_error": chr_err.strip(),
    }


def _check_kernel_isolation() -> Dict[str, Any]:
    cmdline = _read_text(Path("/proc/cmdline")).strip()
    has_isol = "isolcpus=" in cmdline
    has_nohz = "nohz_full=" in cmdline
    has_rcu = "rcu_nocbs=" in cmdline
    return {
        "cmdline": cmdline,
        "has_isolcpus": has_isol,
        "has_nohz_full": has_nohz,
        "has_rcu_nocbs": has_rcu,
        "configured": bool(has_isol and has_nohz and has_rcu),
    }


def _check_iface_timestamps(iface: str) -> Dict[str, Any]:
    rc, out, err = _run(["ethtool", "-T", iface])
    if rc != 0:
        return {"available": False, "hardware_timestamping": False, "error": err.strip() or out.strip()}
    txt = out.lower()
    hw = ("sof_timestamping_tx_hardware" in txt) and ("sof_timestamping_rx_hardware" in txt)
    phc = None
    m = re.search(r"ptp hardware clock:\s*(-?\d+)", txt)
    if m:
        phc = int(m.group(1))
    return {"available": True, "hardware_timestamping": bool(hw), "ptp_hardware_clock": phc}


def _check_iface_driver(iface: str) -> Dict[str, Any]:
    rc, out, err = _run(["ethtool", "-i", iface])
    if rc != 0:
        return {"available": False, "error": err.strip() or out.strip()}
    driver = ""
    bus = ""
    for line in out.splitlines():
        if line.lower().startswith("driver:"):
            driver = line.split(":", 1)[1].strip()
        elif line.lower().startswith("bus-info:"):
            bus = line.split(":", 1)[1].strip()
    return {"available": True, "driver": driver, "bus_info": bus}


def _check_onload_loaded() -> Dict[str, Any]:
    rc, out, err = _run(["lsmod"])
    if rc != 0:
        return {"loaded": False, "error": err.strip()}
    names = [line.split()[0] for line in out.splitlines()[1:] if line.strip()]
    return {"loaded": ("onload" in names), "modules": [n for n in names if n in {"onload", "sfc", "sfc_resource"}]}


def _check_hugepages() -> Dict[str, Any]:
    txt = _read_text(Path("/proc/meminfo"))
    total = 0
    free = 0
    size_kb = 0
    for line in txt.splitlines():
        if line.startswith("HugePages_Total:"):
            total = int(line.split(":", 1)[1].strip())
        elif line.startswith("HugePages_Free:"):
            free = int(line.split(":", 1)[1].strip())
        elif line.startswith("Hugepagesize:"):
            size_kb = int(line.split(":", 1)[1].strip().split()[0])
    return {
        "total": total,
        "free": free,
        "size_kb": size_kb,
        "configured": bool(total > 0),
    }


def build_report(iface: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    linux = platform.system().lower() == "linux"
    kernel = _check_kernel_isolation() if linux else {"configured": False}
    gov = _parse_cpu_governors() if linux else {"all_performance": False}
    irq = _check_irqbalance_disabled() if linux else {"disabled": False, "state": "n/a"}
    ntp = _check_ntp_sync() if linux else {"ntp_synced": False, "clock_offset_us": None}
    ts = _check_iface_timestamps(iface) if linux else {"hardware_timestamping": False}
    drv = _check_iface_driver(iface) if linux else {"available": False}
    onload = _check_onload_loaded() if linux else {"loaded": False}
    hp = _check_hugepages() if linux else {"configured": False}

    checks = {
        "platform_linux": bool(linux),
        "kernel_isolation_configured": bool(kernel.get("configured", False)),
        "cpu_governor_all_performance": bool(gov.get("all_performance", False)),
        "irqbalance_disabled": bool(irq.get("disabled", False)),
        "ntp_synced": bool(ntp.get("ntp_synced", False)),
        "hardware_timestamping": bool(ts.get("hardware_timestamping", False)),
        "hugepages_configured": bool(hp.get("configured", False)),
        "onload_loaded": bool(onload.get("loaded", False)),
    }
    required_ok = all(
        [
            checks["platform_linux"],
            checks["kernel_isolation_configured"],
            checks["cpu_governor_all_performance"],
            checks["irqbalance_disabled"],
            checks["ntp_synced"],
            checks["hardware_timestamping"],
        ]
    )
    report = {
        "checked_at": now,
        "host": socket.gethostname(),
        "iface": iface,
        "status": "PASS" if required_ok else "FAIL",
        "checks": checks,
        "details": {
            "kernel": kernel,
            "cpu_governor": gov,
            "irqbalance": irq,
            "clock_sync": ntp,
            "interface_timestamps": ts,
            "interface_driver": drv,
            "onload": onload,
            "hugepages": hp,
        },
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Verify Linux infra determinism for Argus execution island")
    p.add_argument("--iface", default=os.getenv("ARGUS_IFACE", "eth0"))
    p.add_argument("--output", default="reports/infra/verification_latest.json")
    p.add_argument("--strict", action="store_true", help="Exit non-zero when status is FAIL")
    args = p.parse_args()

    report = build_report(str(args.iface))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(out.resolve()))
    print(report["status"])
    if args.strict and report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
