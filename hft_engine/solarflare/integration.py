#!/usr/bin/env python3
"""
Solarflare Integration — top-level class that wires NIC detection, socket
patching, and system tuning into the Argus HFT engine at startup.

Usage in main.py or UnifiedSystemArchitecture.__init__::

    from hft_engine.solarflare import SolarflareIntegration
    sf = SolarflareIntegration(config)
    await sf.initialise()  # non-blocking, all errors swallowed

After initialise():
  - sf.active         True if Solarflare NICs found and onload available
  - sf.socket_factory OnloadSocketFactory for creating accelerated sockets
  - sf.status_dict()  Full JSON-serialisable status for /health endpoint
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .nic_detector import NICDetector, NICInfo
from .onload_socket import OnloadSocketFactory, apply_onload_env
from .onload_launcher import OnloadLauncher
from .latency_tuner import LatencyTuner, TuningResult
from .efvi_bridge import EfviBridge

logger = logging.getLogger(__name__)


class SolarflareIntegration:
    """
    One-stop Solarflare integration for Argus.

    Responsibilities:
      1. Detect Solarflare NICs (NICDetector)
      2. Apply EF_* environment variables (OnloadSocketFactory.apply_env)
      3. Apply system latency tuning (LatencyTuner)
      4. Optionally open ef_vi handle for zero-copy I/O (EfviBridge)
      5. Expose status to the API /health endpoint
      6. Integrate with LatencyTelemetry to annotate journeys with bypass mode
    """

    def __init__(self, config: Any = None):
        self.config = config
        self.active = False
        self.nics: List[NICInfo] = []
        self.socket_factory: Optional[OnloadSocketFactory] = None
        self.launcher: Optional[OnloadLauncher] = None
        self.tuning_result: Optional[TuningResult] = None
        self.efvi_bridges: List[EfviBridge] = []
        self._init_ts: float = 0.0
        self._enabled = bool(getattr(config, "solarflare_enabled", True))

    async def initialise(self) -> bool:
        """
        Run full Solarflare initialisation sequence asynchronously.
        All steps are best-effort; never raises.
        Returns True if Solarflare is active.
        """
        if not self._enabled:
            logger.info("SolarflareIntegration: disabled via config")
            return False
        self._init_ts = time.time()
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._init_sync)
        except Exception as e:
            logger.warning("SolarflareIntegration.initialise: %s", e)
        return self.active

    def _init_sync(self) -> None:
        """Synchronous init (runs in executor to avoid blocking event loop)."""
        # Step 1: NIC detection
        detector = NICDetector()
        self.nics = detector.detect()
        if not self.nics:
            logger.info("SolarflareIntegration: no Solarflare NICs detected")
            return

        iface_names = [n.interface for n in self.nics]
        numa_node = self.nics[0].numa_node if self.nics else -1

        # Step 2: Apply EF_* env vars
        apply_onload_env()

        # Step 3: Create socket factory
        self.socket_factory = OnloadSocketFactory(
            busy_poll_us=int(getattr(self.config, "sf_busy_poll_us", 50) or 50),
            rcvbuf=int(getattr(self.config, "sf_socket_rcvbuf", 16 * 1024 * 1024) or 16 * 1024 * 1024),
            sndbuf=int(getattr(self.config, "sf_socket_sndbuf", 16 * 1024 * 1024) or 16 * 1024 * 1024),
        )

        # Step 4: Check onload launcher
        self.launcher = OnloadLauncher()

        # Step 5: System latency tuning
        isolated_cpus = list(getattr(self.config, "sf_isolated_cpus", []) or [])
        dry_run = bool(getattr(self.config, "sf_tuning_dry_run", False))
        tuner = LatencyTuner(
            nic_interfaces=iface_names,
            isolated_cpus=isolated_cpus,
            numa_node=numa_node,
            huge_pages_1g=int(getattr(self.config, "sf_huge_pages_1g", 4) or 4),
            dry_run=dry_run,
        )
        self.tuning_result = tuner.tune_all()

        # Step 6: Optionally open ef_vi bridges
        efvi_enabled = bool(getattr(self.config, "sf_efvi_enabled", True))
        if efvi_enabled:
            for nic in self.nics:
                if nic.efvi_capable:
                    bridge = EfviBridge(interface=nic.interface)
                    if bridge.open():
                        self.efvi_bridges.append(bridge)
                        logger.info("SolarflareIntegration: ef_vi opened on %s", nic.interface)

        self.active = True
        logger.info(
            "SolarflareIntegration: active on %d NIC(s) %s | onload=%s | efvi_bridges=%d",
            len(self.nics), iface_names,
            self.launcher.available if self.launcher else False,
            len(self.efvi_bridges),
        )

    def status_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable status for /health and /status endpoints."""
        return {
            "solarflare_active": self.active,
            "nics": [
                {
                    "interface": n.interface,
                    "pci": n.pci_address,
                    "speed_gbe": n.speed_gbe,
                    "numa_node": n.numa_node,
                    "onload_capable": n.onload_capable,
                    "efvi_capable": n.efvi_capable,
                    "capabilities": n.capabilities,
                    "firmware": n.firmware_version,
                }
                for n in self.nics
            ],
            "onload_available": self.launcher.available if self.launcher else False,
            "efvi_bridges": len(self.efvi_bridges),
            "tuning_applied": self.tuning_result.applied if self.tuning_result else [],
            "tuning_warnings": self.tuning_result.warnings if self.tuning_result else [],
            "init_ts": self._init_ts,
        }

    def annotate_telemetry(self, telemetry: Any) -> None:
        """
        Attach Solarflare status to the LatencyTelemetry singleton so
        LatencyReport.to_dict() includes kernel-bypass metadata.
        """
        try:
            telemetry._solarflare_status = self.status_dict()
        except Exception:
            pass

    def close(self) -> None:
        for bridge in self.efvi_bridges:
            try:
                bridge.close()
            except Exception:
                pass
        if self.socket_factory:
            self.socket_factory.close_all()
