#!/usr/bin/env python3
"""
NIC Detector — auto-detects Solarflare SFN8522-PLUS NICs on the system.

Detection chain (each step falls back gracefully if tool not available):
  1. /sys/class/net/<iface>/device/vendor + device sysfs attributes
  2. ethtool -i <iface>  (driver == sfc)
  3. lspci grep for Solarflare / XtremeScale

Returns a list of NICInfo dataclasses with interface name, PCI address,
driver, firmware version, and detected capabilities.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Solarflare/Xilinx PCI vendor ID
_SF_VENDOR_ID = "0x1924"
# SFN8522 device IDs (10GbE)
_SF_DEVICE_IDS = {"0x0a03", "0x0903", "0x0a13", "0x1923"}


@dataclass
class NICInfo:
    interface: str
    pci_address: str = ""
    driver: str = ""
    firmware_version: str = ""
    bus_info: str = ""
    speed_gbe: int = 0
    onload_capable: bool = False
    efvi_capable: bool = False
    numa_node: int = -1
    capabilities: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        caps = ", ".join(self.capabilities) or "none"
        return (
            f"NICInfo({self.interface} pci={self.pci_address} driver={self.driver} "
            f"fw={self.firmware_version} speed={self.speed_gbe}GbE "
            f"onload={self.onload_capable} efvi={self.efvi_capable} caps=[{caps}])"
        )


class NICDetector:
    """Detect Solarflare NICs and their capabilities."""

    def __init__(self):
        self._cache: Optional[List[NICInfo]] = None

    def detect(self, force: bool = False) -> List[NICInfo]:
        """Return list of detected Solarflare NICs. Cached after first call."""
        if self._cache is not None and not force:
            return self._cache
        nics: List[NICInfo] = []
        try:
            ifaces = self._list_interfaces()
            for iface in ifaces:
                info = self._probe_interface(iface)
                if info is not None:
                    nics.append(info)
        except Exception as e:
            logger.debug("NICDetector.detect: %s", e)
        # Fallback: lspci scan
        if not nics:
            nics = self._lspci_fallback()
        self._cache = nics
        if nics:
            logger.info("NICDetector: found %d Solarflare NIC(s): %s",
                        len(nics), [n.interface for n in nics])
        else:
            logger.debug("NICDetector: no Solarflare NICs detected")
        return nics

    def has_solarflare(self) -> bool:
        return len(self.detect()) > 0

    def primary_interface(self) -> Optional[str]:
        """Return the first detected Solarflare interface name, or None."""
        nics = self.detect()
        return nics[0].interface if nics else None

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _list_interfaces() -> List[str]:
        net_path = Path("/sys/class/net")
        if not net_path.exists():
            return []
        return [p.name for p in net_path.iterdir() if p.name != "lo"]

    def _probe_interface(self, iface: str) -> Optional[NICInfo]:
        """Return NICInfo if iface is a Solarflare device, else None."""
        # Check sysfs vendor
        vendor_path = Path(f"/sys/class/net/{iface}/device/vendor")
        if vendor_path.exists():
            vendor = vendor_path.read_text().strip().lower()
            if vendor != _SF_VENDOR_ID:
                return None
        else:
            # Try ethtool driver check
            if not self._is_sfc_driver(iface):
                return None

        info = NICInfo(interface=iface)
        info.driver = "sfc"

        # ethtool -i for firmware + bus info
        try:
            out = subprocess.check_output(
                ["ethtool", "-i", iface], stderr=subprocess.DEVNULL, text=True, timeout=3
            )
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("firmware-version:"):
                    info.firmware_version = line.split(":", 1)[1].strip()
                elif line.startswith("bus-info:"):
                    info.bus_info = line.split(":", 1)[1].strip()
                    info.pci_address = info.bus_info
        except Exception:
            pass

        # Speed via ethtool
        try:
            out = subprocess.check_output(
                ["ethtool", iface], stderr=subprocess.DEVNULL, text=True, timeout=3
            )
            m = re.search(r"Speed:\s*(\d+)([MG])b/s", out)
            if m:
                speed = int(m.group(1))
                info.speed_gbe = speed // 1000 if m.group(2) == "M" else speed
        except Exception:
            pass

        # NUMA node
        numa_path = Path(f"/sys/class/net/{iface}/device/numa_node")
        if numa_path.exists():
            try:
                info.numa_node = int(numa_path.read_text().strip())
            except Exception:
                pass

        # Capability detection
        info.onload_capable = self._check_onload_capable()
        info.efvi_capable = self._check_efvi_capable()
        caps = []
        if info.onload_capable:
            caps.append("OpenOnload")
        if info.efvi_capable:
            caps.append("ef_vi")
        if info.speed_gbe >= 10:
            caps.append(f"{info.speed_gbe}GbE")
        info.capabilities = caps

        return info

    @staticmethod
    def _is_sfc_driver(iface: str) -> bool:
        try:
            out = subprocess.check_output(
                ["ethtool", "-i", iface], stderr=subprocess.DEVNULL, text=True, timeout=3
            )
            return "driver: sfc" in out
        except Exception:
            return False

    @staticmethod
    def _check_onload_capable() -> bool:
        """True if onload binary and kernel module are present."""
        # Check for onload binary
        try:
            subprocess.check_output(["which", "onload"], stderr=subprocess.DEVNULL, timeout=2)
            binary_ok = True
        except Exception:
            binary_ok = False
        # Check for sfc_resource or onload kernel module
        module_ok = Path("/dev/onload").exists() or Path("/proc/driver/onload").exists()
        return binary_ok or module_ok

    @staticmethod
    def _check_efvi_capable() -> bool:
        """True if libefvi shared library is loadable."""
        import ctypes
        for lib_name in ("libefvi.so", "libefvi.so.1", "/usr/lib/libefvi.so"):
            try:
                ctypes.CDLL(lib_name)
                return True
            except OSError:
                pass
        return False

    @staticmethod
    def _lspci_fallback() -> List[NICInfo]:
        """Use lspci to find Solarflare devices when sysfs probing fails."""
        nics: List[NICInfo] = []
        try:
            out = subprocess.check_output(
                ["lspci", "-v"], stderr=subprocess.DEVNULL, text=True, timeout=5
            )
            # Look for Solarflare / XtremeScale entries
            for block in re.split(r"\n(?=[0-9a-f]{2}:)", out):
                if re.search(r"(solarflare|xtremescale|sfn8522|sfn852)", block, re.I):
                    pci_m = re.match(r"([0-9a-f:.]+)\s", block)
                    pci_addr = pci_m.group(1) if pci_m else ""
                    info = NICInfo(
                        interface="ethX",  # interface name unknown via lspci alone
                        pci_address=pci_addr,
                        driver="sfc",
                        onload_capable=NICDetector._check_onload_capable(),
                        efvi_capable=NICDetector._check_efvi_capable(),
                    )
                    nics.append(info)
        except Exception as e:
            logger.debug("NICDetector lspci fallback: %s", e)
        return nics
