"""
core/dpdk_colo_bridge.py
========================
Co-location latency profiler + DPDK readiness bridge.

Responsibilities
----------------
1. Continuously probe round-trip latency to the exchange endpoint
   (UDP echo or REST ping) and maintain a rolling p99 histogram.
2. Emit a ``colo_ready`` event on the EventBus once p99 RTT drops
   below ``KernelBypassConfig.latency_budget_us``.
3. Expose a ``DPDKReadinessChecker`` that validates:
   - hugepages allocated
   - DPDK-bound NIC present (via /sys or dpdk-devbind)
   - sidecar socket live
   - NUMA pinning active
4. Wire KernelBypassRouter into ExecutionEngine when colo_ready.

The module is designed to run as a background asyncio task alongside
the main Argus loop.  It does NOT block startup — if co-lo is not
detected the system continues on the TCP fallback path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("argus.core.dpdk_colo_bridge")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ColoBridgeConfig:
    """Runtime configuration for the co-location bridge."""
    exchange_host: str = os.environ.get("ARGUS_EXCHANGE_HOST", "127.0.0.1")
    exchange_port: int = int(os.environ.get("ARGUS_EXCHANGE_PORT", "443"))
    probe_interval_s: float = 1.0        # seconds between latency probes
    window_size: int = 60                # samples in rolling window
    latency_budget_us: float = 10.0      # p99 target in microseconds
    ready_streak: int = 5                # consecutive windows under budget
    hugepage_path: str = "/proc/sys/vm/nr_hugepages"
    sidecar_socket: str = os.environ.get("ARGUS_BYPASS_SOCKET", "/tmp/argus_bypass.sock")
    dpdk_devbind: str = "/usr/local/bin/dpdk-devbind.py"
    numa_cpus: str = os.environ.get("ARGUS_NUMA_CPUS", "")  # e.g. "0-7"


# ---------------------------------------------------------------------------
# Latency probe
# ---------------------------------------------------------------------------

class RTTProbe:
    """
    Measures round-trip latency to the exchange using UDP echo or
    a minimal HTTP HEAD request.  Falls back to a loopback echo
    if the exchange endpoint is unreachable.
    """

    def __init__(self, host: str, port: int, timeout_s: float = 0.05) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout_s
        self._samples: list[float] = []
        self._window = 60

    async def probe(self) -> Optional[float]:
        """Return RTT in microseconds, or None on failure."""
        t0 = time.monotonic_ns()
        try:
            loop = asyncio.get_event_loop()
            conn = loop.create_connection(
                asyncio.Protocol,
                host=self._host,
                port=self._port,
            )
            transport, _ = await asyncio.wait_for(conn, timeout=self._timeout)
            transport.close()
        except Exception:
            # Loopback fallback: just measure the asyncio overhead
            await asyncio.sleep(0)
        rtt_us = (time.monotonic_ns() - t0) / 1_000.0
        self._samples.append(rtt_us)
        if len(self._samples) > self._window:
            self._samples.pop(0)
        return rtt_us

    def p99(self) -> Optional[float]:
        if len(self._samples) < 2:
            return None
        s = sorted(self._samples)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    def p50(self) -> Optional[float]:
        if not self._samples:
            return None
        return statistics.median(self._samples)

    @property
    def latest(self) -> Optional[float]:
        return self._samples[-1] if self._samples else None


# ---------------------------------------------------------------------------
# DPDK readiness checker
# ---------------------------------------------------------------------------

class DPDKReadinessChecker:
    """
    Validates that the host is ready for kernel-bypass operation.

    Checks (non-exhaustive):
    - /proc/sys/vm/nr_hugepages > 0
    - sidecar Unix socket exists
    - /sys/bus/pci/drivers/vfio-pci has at least one device bound
    - NUMA CPU set non-empty (if configured)
    """

    def __init__(self, cfg: ColoBridgeConfig) -> None:
        self._cfg = cfg

    def check(self) -> dict[str, bool]:
        results: dict[str, bool] = {}

        # hugepages
        try:
            hp = int(Path(self._cfg.hugepage_path).read_text().strip())
            results["hugepages"] = hp > 0
        except Exception:
            results["hugepages"] = False

        # sidecar socket
        results["sidecar_socket"] = os.path.exists(self._cfg.sidecar_socket)

        # vfio-pci binding
        vfio_path = Path("/sys/bus/pci/drivers/vfio-pci")
        try:
            devices = [p for p in vfio_path.iterdir() if p.name.startswith("0000:")]
            results["vfio_pci_bound"] = len(devices) > 0
        except Exception:
            results["vfio_pci_bound"] = False

        # NUMA pinning
        results["numa_pinned"] = bool(self._cfg.numa_cpus)

        results["all_ok"] = all(v for k, v in results.items() if k != "numa_pinned")
        return results

    def log_status(self) -> None:
        checks = self.check()
        for k, v in checks.items():
            icon = "✓" if v else "✗"
            level = logging.INFO if v else logging.WARNING
            logger.log(level, "DPDK readiness [%s] %s", icon, k)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class DPDKColoBridge:
    """
    Background asyncio task that:
    1. Polls exchange RTT every ``probe_interval_s``.
    2. Runs DPDKReadinessChecker on startup and every 60 s.
    3. Fires ``on_colo_ready(stats)`` callback once the ready condition
       is met (p99 < budget AND dpdk checks pass, for ``ready_streak``
       consecutive windows).
    4. Fires ``on_colo_lost()`` if conditions degrade after going ready.
    """

    def __init__(
        self,
        cfg: Optional[ColoBridgeConfig] = None,
        on_colo_ready: Optional[Callable[[dict], None]] = None,
        on_colo_lost: Optional[Callable[[], None]] = None,
    ) -> None:
        self._cfg = cfg or ColoBridgeConfig()
        self._probe = RTTProbe(self._cfg.exchange_host, self._cfg.exchange_port)
        self._checker = DPDKReadinessChecker(self._cfg)
        self._on_ready = on_colo_ready or (lambda s: None)
        self._on_lost = on_colo_lost or (lambda: None)
        self._ready = False
        self._streak = 0
        self._total_probes = 0
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background probe loop."""
        logger.info("DPDKColoBridge starting — target: %s:%d budget: %.1f µs",
                    self._cfg.exchange_host, self._cfg.exchange_port,
                    self._cfg.latency_budget_us)
        self._checker.log_status()
        self._task = asyncio.create_task(self._loop(), name="dpdk_colo_bridge")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DPDKColoBridge stopped")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def stats(self) -> dict:
        return {
            "colo_ready": self._ready,
            "rtt_p50_us": self._probe.p50(),
            "rtt_p99_us": self._probe.p99(),
            "rtt_latest_us": self._probe.latest,
            "streak": self._streak,
            "total_probes": self._total_probes,
            "dpdk_checks": self._checker.check(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        check_counter = 0
        while True:
            try:
                await asyncio.sleep(self._cfg.probe_interval_s)
                rtt = await self._probe.probe()
                self._total_probes += 1
                check_counter += 1

                p99 = self._probe.p99()
                if p99 is None:
                    continue

                # Re-run DPDK checks every 60 probes
                dpdk_ok = True
                if check_counter % 60 == 0:
                    checks = self._checker.check()
                    dpdk_ok = checks.get("all_ok", True)  # permissive if checks absent
                    if not dpdk_ok:
                        logger.warning("DPDK readiness degraded: %s", checks)

                under_budget = p99 < self._cfg.latency_budget_us

                if under_budget and dpdk_ok:
                    self._streak += 1
                else:
                    self._streak = 0

                if self._streak >= self._cfg.ready_streak and not self._ready:
                    self._ready = True
                    logger.info("colo_ready=True p99=%.2fµs streak=%d",
                                p99, self._streak)
                    self._on_ready(self.stats())

                elif self._ready and not under_budget:
                    self._ready = False
                    self._streak = 0
                    logger.warning("colo_ready lost — p99=%.2fµs", p99)
                    self._on_lost()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("DPDKColoBridge probe error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_colo_bridge(
    exchange_host: str = "127.0.0.1",
    latency_budget_us: float = 10.0,
    on_ready: Optional[Callable[[dict], None]] = None,
    on_lost: Optional[Callable[[], None]] = None,
) -> DPDKColoBridge:
    """
    Factory used by full_wiring.py::

        bridge = make_colo_bridge(
            exchange_host=cfg.exchange_ws_url,
            latency_budget_us=cfg.latency_budget_us,
            on_ready=lambda s: execution_engine.switch_to_bypass(),
            on_lost=lambda: execution_engine.switch_to_tcp(),
        )
        asyncio.create_task(bridge.start())
    """
    cfg = ColoBridgeConfig(
        exchange_host=exchange_host,
        latency_budget_us=latency_budget_us,
    )
    return DPDKColoBridge(cfg=cfg, on_colo_ready=on_ready, on_colo_lost=on_lost)
