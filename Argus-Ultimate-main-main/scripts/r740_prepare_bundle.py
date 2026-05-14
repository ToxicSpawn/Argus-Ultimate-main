#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Tuple

import yaml  # type: ignore[import-untyped]


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "docs" / "hardware" / "R740_PREBUILD_MANIFEST.yaml"
TEMPLATES_DIR = ROOT / "ops" / "linux" / "templates"


@dataclass
class RenderSpec:
    src: str
    dst: str
    executable: bool = False


RENDER_SPECS: List[RenderSpec] = [
    RenderSpec("grub_cmdline.txt.tmpl", "render/grub_cmdline.txt"),
    RenderSpec("sysctl_argus_lowlatency.conf.tmpl", "render/99-argus-lowlatency.conf"),
    RenderSpec("argus.service.override.conf.tmpl", "render/argus.service.override.conf"),
    RenderSpec("argus_pin_irqs.sh.tmpl", "render/argus-pin-irqs.sh", executable=True),
    RenderSpec("argus-pin-irqs.service.tmpl", "render/argus-pin-irqs.service"),
    RenderSpec("netplan_10gbe.yaml.tmpl", "render/50-argus-10gbe.yaml"),
    RenderSpec("chrony_argus.conf.tmpl", "render/chrony.argus.conf"),
    RenderSpec("ptp4l_argus.conf.tmpl", "render/ptp4l.argus.conf"),
    RenderSpec("phc2sys.argus.service.tmpl", "render/phc2sys.argus.service"),
    RenderSpec("ethtool_apply.sh.tmpl", "render/argus-ethtool-apply.sh", executable=True),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Manifest root must be mapping")
    return data


def _flatten_manifest(m: Dict[str, Any]) -> Dict[str, str]:
    host = dict(m.get("host") or {})
    network = dict(m.get("network") or {})
    cpu = dict(m.get("cpu") or {})
    timing = dict(m.get("timing") or {})
    build = dict(m.get("build") or {})

    dns = network.get("dns_servers") or []
    if not isinstance(dns, list):
        dns = []
    dns_csv = ", ".join(str(x).strip() for x in dns if str(x).strip())

    values: Dict[str, str] = {
        "HOSTNAME": str(host.get("hostname", "argus-r740")),
        "ARGUS_USER": str(host.get("argus_user", "argus")),
        "REPO_DIR": str(host.get("repo_dir", "/opt/argus/repo")),
        "LOG_DIR": str(host.get("log_dir", "/opt/argus/logs")),
        "IFACE_PRIMARY": str(network.get("iface_primary", "enp3s0f0")),
        "IFACE_SECONDARY": str(network.get("iface_secondary", "enp3s0f1")),
        "IP_PRIMARY_CIDR": str(network.get("ip_primary_cidr", "192.168.10.100/24")),
        "IP_SECONDARY_CIDR": str(network.get("ip_secondary_cidr", "192.168.10.101/24")),
        "GATEWAY4": str(network.get("gateway4", "192.168.10.1")),
        "DNS_SERVERS_CSV": dns_csv or "1.1.1.1, 8.8.8.8",
        "MTU": str(network.get("mtu", 9000)),
        "PTP_IFACE": str(network.get("ptp_iface", network.get("iface_primary", "enp3s0f0"))),
        "OS_CPUS": str(cpu.get("os_cpus", "0-3")),
        "EXEC_CPUS": str(cpu.get("exec_cpus", "4-15")),
        "CPU_ISOLATION": str(cpu.get("cpu_isolation", "4-15")),
        "NTP_SERVER_1": str(timing.get("ntp_server_1", "time.cloudflare.com")),
        "NTP_SERVER_2": str(timing.get("ntp_server_2", "time.google.com")),
        "PROFILE_NAME": str(build.get("profile_name", "r740-default")),
    }
    return values


def _render_one(src: Path, dst: Path, values: Dict[str, str], executable: bool) -> None:
    body = src.read_text(encoding="utf-8")
    rendered = Template(body).safe_substitute(values)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(rendered, encoding="utf-8")
    if executable:
        mode = dst.stat().st_mode
        dst.chmod(mode | 0o111)


def _write_apply_steps(bundle: Path, values: Dict[str, str]) -> None:
    lines = [
        "# R740 Apply Steps (Generated)",
        "",
        "1. Review generated files under `render/`.",
        "2. Install files on host:",
        "   - `99-argus-lowlatency.conf` -> `/etc/sysctl.d/`",
        "   - `argus.service.override.conf` -> `/etc/systemd/system/argus.service.d/override.conf`",
        "   - `argus-pin-irqs.sh` -> `/usr/local/sbin/argus-pin-irqs.sh`",
        "   - `argus-pin-irqs.service` -> `/etc/systemd/system/argus-pin-irqs.service`",
        "   - `50-argus-10gbe.yaml` -> `/etc/netplan/50-argus-10gbe.yaml`",
        "   - `chrony.argus.conf` -> `/etc/chrony/chrony.conf` (merge carefully)",
        "   - `ptp4l.argus.conf` -> `/etc/linuxptp/ptp4l.conf`",
        "   - `phc2sys.argus.service` -> `/etc/systemd/system/phc2sys.argus.service`",
        "3. Apply kernel args from `grub_cmdline.txt` and run `update-grub`.",
        "4. Restart networking/time services and run host verification:",
        f"   `python3 scripts/infra_verify_host.py --iface {values['IFACE_PRIMARY']} --strict`",
        "5. Generate infra preflight report and keep live gated until PASS:",
        "   `python3 scripts/infra_preflight.py --report reports/infra/verification_latest.json --output reports/infra/infra_preflight_latest.json --max-clock-offset-us 250`",
        "",
        "Safety:",
        "- Keep live trading gated and off by default.",
        "- Prefer halting safely over uncertain recovery in ambiguous states.",
    ]
    (bundle / "APPLY_STEPS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_hashes(bundle: Path) -> List[Tuple[str, str]]:
    hashes: List[Tuple[str, str]] = []
    for p in sorted(bundle.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(bundle)).replace("\\", "/")
        if rel == "hashes.txt":
            continue
        hashes.append((rel, _sha256(p)))
    return hashes


def _write_hashes(bundle: Path, hashes: Iterable[Tuple[str, str]]) -> None:
    lines = [f"{h}  {name}" for name, h in hashes]
    (bundle / "hashes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_bundle(manifest_path: Path, output_root: Path) -> Path:
    manifest = _load_manifest(manifest_path)
    values = _flatten_manifest(manifest)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = (output_root / f"{ts}_{values['PROFILE_NAME']}").resolve()
    bundle.mkdir(parents=True, exist_ok=True)

    for spec in RENDER_SPECS:
        src = TEMPLATES_DIR / spec.src
        if not src.exists():
            raise FileNotFoundError(f"Template not found: {src}")
        dst = bundle / spec.dst
        _render_one(src, dst, values, spec.executable)

    resolved_manifest = {
        "source_manifest": str(manifest_path.resolve()),
        "resolved_values": values,
        "manifest_raw": manifest,
    }
    (bundle / "manifest.resolved.json").write_text(
        json.dumps(resolved_manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    build_info = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_type": "r740_prebuild",
        "profile_name": values["PROFILE_NAME"],
        "hostname": values["HOSTNAME"],
        "platform": os.name,
    }
    (bundle / "build_info.json").write_text(json.dumps(build_info, ensure_ascii=True, indent=2), encoding="utf-8")
    _write_apply_steps(bundle, values)
    hashes = _collect_hashes(bundle)
    _write_hashes(bundle, hashes)
    return bundle


def main() -> int:
    p = argparse.ArgumentParser(description="Generate R740 prebuild deployment bundle from manifest")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--output-root", default="deploy/r740_bundle")
    args = p.parse_args()

    bundle = build_bundle(Path(args.manifest), Path(args.output_root))
    print(str(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
