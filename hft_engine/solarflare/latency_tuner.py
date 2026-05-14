#!/usr/bin/env python3
"""
Latency Tuner — applies system-level tuning for minimum tick-to-trade latency
on the Dell R7525 + Solarflare SFN8522-PLUS setup.

Tuning applied (all require root; failures are non-fatal):
  1. IRQ affinity — pin Solarflare NIC IRQs to isolated CPUs
  2. CPU frequency — set governor to performance
  3. NIC interrupt coalescing — set rx-usecs=0 for minimum IRQ latency
  4. Huge pages — allocate 1GB huge pages for DMA buffers
  5. NUMA awareness — report if process is on wrong NUMA node
  6. Kernel network tunables — rmem_max, wmem_max, tcp_low_latency
  7. CPU isolation check — warn if trading CPUs share with kernel threads
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    applied: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_root: bool = False

    def summary(self) -> str:
        lines = [
            f"LatencyTuner results (root={self.is_root}):",
            f"  Applied  ({len(self.applied)}): {', '.join(self.applied) or 'none'}",
            f"  Skipped  ({len(self.skipped)}): {', '.join(self.skipped) or 'none'}",
        ]
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        return "\n".join(lines)


class LatencyTuner:
    """
    System-level latency tuner for the Argus HFT stack.

    Usage::
        tuner = LatencyTuner(nic_interfaces=["eth0", "eth1"])
        result = tuner.tune_all()
        print(result.summary())
    """

    def __init__(
        self,
        nic_interfaces: Optional[List[str]] = None,
        *,
        isolated_cpus: Optional[List[int]] = None,
        numa_node: int = -1,
        huge_pages_1g: int = 4,
        dry_run: bool = False,
    ):
        self.nic_interfaces = list(nic_interfaces or [])
        self.isolated_cpus = list(isolated_cpus or [])
        self.numa_node = int(numa_node)
        self.huge_pages_1g = max(0, int(huge_pages_1g))
        self.dry_run = bool(dry_run)

    def tune_all(self) -> TuningResult:
        """Run all tuning steps. Returns a TuningResult summary."""
        result = TuningResult(is_root=(os.geteuid() == 0))
        steps = [
            ("cpu_governor",            self._tune_cpu_governor),
            ("kernel_net_tunables",      self._tune_kernel_net),
            ("nic_irq_coalescing",       self._tune_irq_coalescing),
            ("nic_irq_affinity",         self._tune_irq_affinity),
            ("huge_pages",               self._tune_huge_pages),
            ("numa_check",               self._check_numa),
            ("cpu_isolation_check",      self._check_cpu_isolation),
        ]
        for name, fn in steps:
            try:
                ok, msg = fn()
                if ok:
                    result.applied.append(name)
                else:
                    result.skipped.append(name)
                if msg:
                    result.warnings.append(f"{name}: {msg}")
            except Exception as e:
                result.skipped.append(name)
                result.warnings.append(f"{name}: exception {e}")
        logger.info("LatencyTuner:\n%s", result.summary())
        return result

    # ---------------------------------------------------------------- steps

    def _tune_cpu_governor(self) -> Tuple[bool, str]:
        """Set CPU frequency governor to performance on all cores."""
        gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        if not gov_path.exists():
            return False, "cpufreq not available"
        if self.dry_run:
            return True, "dry_run"
        try:
            cpu_dirs = list(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
            for p in cpu_dirs:
                self._write(p, "performance")
            return True, ""
        except PermissionError:
            return False, "need root for cpufreq"

    def _tune_kernel_net(self) -> Tuple[bool, str]:
        """Apply sysctl network tunables."""
        sysctls = {
            "net.core.rmem_max":                 "134217728",   # 128MB
            "net.core.wmem_max":                 "134217728",
            "net.core.rmem_default":             "67108864",    # 64MB
            "net.core.wmem_default":             "67108864",
            "net.ipv4.tcp_rmem":                 "4096 87380 134217728",
            "net.ipv4.tcp_wmem":                 "4096 65536 134217728",
            "net.core.netdev_max_backlog":        "250000",
            "net.core.somaxconn":                "65535",
            "net.ipv4.tcp_low_latency":          "1",
            "net.ipv4.tcp_timestamps":           "0",
            "net.ipv4.tcp_sack":                 "1",
            "net.ipv4.tcp_no_delay_ack":         "1",
            "net.ipv4.tcp_thin_linear_timeouts": "1",
            "kernel.numa_balancing":             "0",   # disable auto NUMA balancing
        }
        if self.dry_run:
            return True, "dry_run"
        applied = 0
        for key, val in sysctls.items():
            try:
                subprocess.run(
                    ["sysctl", "-w", f"{key}={val}"],
                    capture_output=True, timeout=3,
                )
                applied += 1
            except Exception:
                pass
        return (applied > 0), ""

    def _tune_irq_coalescing(self) -> Tuple[bool, str]:
        """Set rx-usecs=0 on Solarflare NICs for minimum IRQ latency."""
        if not self.nic_interfaces:
            return False, "no NIC interfaces specified"
        if self.dry_run:
            return True, "dry_run"
        applied = 0
        for iface in self.nic_interfaces:
            try:
                subprocess.run(
                    ["ethtool", "-C", iface, "rx-usecs", "0", "tx-usecs", "0"],
                    capture_output=True, timeout=5,
                )
                applied += 1
                logger.debug("LatencyTuner: set coalescing rx-usecs=0 on %s", iface)
            except Exception as e:
                logger.debug("LatencyTuner: coalescing %s: %s", iface, e)
        return (applied > 0), ""

    def _tune_irq_affinity(self) -> Tuple[bool, str]:
        """Pin Solarflare NIC IRQs to isolated CPUs."""
        if not self.nic_interfaces or not self.isolated_cpus:
            return False, "no interfaces or isolated_cpus specified"
        if self.dry_run:
            return True, "dry_run"
        applied = 0
        for iface in self.nic_interfaces:
            irq_nums = self._get_nic_irqs(iface)
            if not irq_nums:
                continue
            # Round-robin IRQs across isolated CPUs
            cpu_iter = iter(self.isolated_cpus * (len(irq_nums) // len(self.isolated_cpus) + 1))
            for irq in irq_nums:
                cpu = next(cpu_iter)
                affinity_path = Path(f"/proc/irq/{irq}/smp_affinity_list")
                try:
                    self._write(affinity_path, str(cpu))
                    applied += 1
                except Exception as e:
                    logger.debug("IRQ affinity %d->cpu%d: %s", irq, cpu, e)
        return (applied > 0), ""

    def _tune_huge_pages(self) -> Tuple[bool, str]:
        """Allocate 1GB huge pages for ef_vi DMA buffers."""
        if self.huge_pages_1g <= 0:
            return False, "huge_pages_1g=0"
        hp_path = Path("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages")
        if not hp_path.exists():
            return False, "1GB hugepage support not in kernel"
        if self.dry_run:
            return True, "dry_run"
        try:
            current = int(hp_path.read_text().strip())
            target = max(current, self.huge_pages_1g)
            self._write(hp_path, str(target))
            return True, ""
        except PermissionError:
            return False, "need root for hugepages"

    def _check_numa(self) -> Tuple[bool, str]:
        """Warn if current process NUMA node != NIC NUMA node."""
        pid_numa = self._get_pid_numa_node()
        if self.numa_node >= 0 and pid_numa >= 0 and pid_numa != self.numa_node:
            return True, (
                f"Process on NUMA node {pid_numa} but NIC on node {self.numa_node}. "
                f"Run: numactl --cpunodebind={self.numa_node} --membind={self.numa_node} python main.py"
            )
        return True, ""

    def _check_cpu_isolation(self) -> Tuple[bool, str]:
        """Warn if no CPUs are isolated (isolcpus kernel param absent)."""
        try:
            cmdline = Path("/proc/cmdline").read_text()
            if "isolcpus" not in cmdline:
                return True, (
                    "No isolcpus kernel parameter detected. "
                    "Add isolcpus=<cpulist> to GRUB_CMDLINE_LINUX for lowest jitter. "
                    "Example: isolcpus=2-7,10-15 for 12 isolated cores on R7525."
                )
        except Exception:
            pass
        return True, ""

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _write(path: Path, value: str) -> None:
        path.write_text(value)

    @staticmethod
    def _get_nic_irqs(iface: str) -> List[int]:
        """Return IRQ numbers for a NIC interface."""
        irqs: List[int] = []
        try:
            with open("/proc/interrupts") as f:
                for line in f:
                    if iface in line:
                        m = re.match(r"\s*(\d+):", line)
                        if m:
                            irqs.append(int(m.group(1)))
        except Exception:
            pass
        return irqs

    @staticmethod
    def _get_pid_numa_node() -> int:
        """Return NUMA node of current process (-1 if unknown)."""
        try:
            out = subprocess.check_output(
                ["numactl", "--show"], stderr=subprocess.DEVNULL, text=True, timeout=3
            )
            m = re.search(r"nodebind:\s*(\d+)", out)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return -1
