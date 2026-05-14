#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _cpu_facts() -> Dict[str, Any]:
    model = ""
    physical = set()
    logical = 0
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        phys = None
        core = None
        for line in _read_text(cpuinfo).splitlines():
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
            if line.startswith("processor"):
                logical += 1
                phys = None
                core = None
            elif line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
            if phys is not None and core is not None:
                physical.add((phys, core))
    if logical == 0 and os.cpu_count():
        logical = int(os.cpu_count() or 0)
    return {
        "model_name": model,
        "logical_cpus": logical,
        "physical_cores": len(physical) if physical else None,
    }


def _memory_facts() -> Dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    mem_kib = None
    if meminfo.exists():
        for line in _read_text(meminfo).splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    mem_kib = int(parts[1])
                break
    mem_gib = (float(mem_kib) / (1024.0 * 1024.0)) if mem_kib else None
    return {"mem_total_kib": mem_kib, "mem_total_gib": round(mem_gib, 2) if mem_gib else None}


def _network_facts() -> Dict[str, Any]:
    interfaces: List[Dict[str, Any]] = []
    net_root = Path("/sys/class/net")
    if net_root.exists():
        for iface_dir in sorted(net_root.iterdir(), key=lambda p: p.name):
            iface = iface_dir.name
            if iface == "lo":
                continue
            speed = None
            speed_path = iface_dir / "speed"
            if speed_path.exists():
                try:
                    speed = int(_read_text(speed_path).strip())
                except Exception:
                    speed = None
            interfaces.append({"name": iface, "speed_mbps": speed})
    return {"interfaces": interfaces}


def _storage_facts() -> Dict[str, Any]:
    try:
        proc = subprocess.run(["lsblk", "-J", "-o", "NAME,TYPE,SIZE,ROTA,MODEL"], capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout:
            payload = json.loads(proc.stdout)
            devices = payload.get("blockdevices") or []
            return {"lsblk": devices}
    except Exception:
        pass
    return {"lsblk": []}


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture host hardware facts for R740 acceptance checks")
    parser.add_argument("--output", default="reports/infra/r740_host_facts_latest.json")
    args = parser.parse_args()

    try:
        uname = os.uname()
        sysname = uname.sysname
        release = uname.release
        machine = uname.machine
    except Exception:
        sysname = platform.system()
        release = platform.release()
        machine = platform.machine()
    facts: Dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "sysname": sysname,
            "release": release,
            "machine": machine,
        },
        "cpu": _cpu_facts(),
        "memory": _memory_facts(),
        "network": _network_facts(),
        "storage": _storage_facts(),
    }

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
