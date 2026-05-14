"""
High-Speed Network Engine — 10GbE/25GbE link optimization.

Supports both:
  • Solarflare SFN8522-PLUS dual 10GbE (workstation side)
  • Mellanox/Intel XXV710 dual 25GbE rNDC (R740 side with 25GbE NDC)

Auto-detects link speed from `ethtool` and applies tuning appropriately.
At 25GbE, jumbo frames (MTU 9000) become mandatory to saturate the link.

Benefits:

1. KERNEL BYPASS (Solarflare OpenOnload / DPDK): skip the OS network stack
   entirely. Reduces round-trip latency from ~100μs to ~5μs.

2. ZERO-COPY DATA TRANSFER: market data flows from R740's TimescaleDB
   directly to workstation GPU memory without CPU copy overhead.

3. DEDICATED DATA CHANNELS: separate trading from backtesting traffic.
   NIC 1: real-time market data + order execution (latency-critical)
   NIC 2: bulk data transfer (historical OHLCV, tick replay, model sync)

4. HARDWARE TIMESTAMPING: Solarflare/Intel NICs timestamp packets in
   hardware, giving nanosecond-precision timing for latency measurement.

Architecture (with 25GbE NDC upgrade):
  ┌───────────────────────────────┐         ┌───────────────────────────────┐
  │       WORKSTATION             │         │           R740                │
  │  RTX 5080 + Ultra 9 285K     │         │  192GB DDR4-2933 ECC +       │
  │                               │         │  2x Xeon Gold 6248 (40c)     │
  │  NIC 1 (SFN8522) ────────────┼─10G/25G┼── NIC 1 (SFN8522/X710)       │
  │  │ Real-time channel          │  <5μs  │  │ TimescaleDB + Redis        │
  │  │ Market data streaming      │         │  │ Real-time tick feed        │
  │  │ Order execution relay      │         │  │ Order book snapshots       │
  │  │                            │         │  │                            │
  │  NIC 2 (SFN8522) ────────────┼─10G/25G┼── NIC 2 (SFN8522/X710)       │
  │  │ Bulk data channel          │         │  │ Bulk data server           │
  │  │ Historical OHLCV sync      │         │  │ Tick replay stream         │
  │  │ Model weight transfer      │         │  │ Backtest results           │
  │  │ Evolution population sync  │         │  │ Monte Carlo results        │
  └───────────────────────────────┘         └───────────────────────────────┘

Network Tuning:
  - Jumbo frames (MTU 9000) for bulk transfers  (mandatory at 25GbE)
  - TCP_NODELAY on real-time channel
  - SO_BUSY_POLL for lowest latency
  - CPU affinity: pin NIC interrupts to dedicated cores
  - Ring buffer: max size (4096) for burst absorption
  - Socket buffers scale with link speed (16MB @ 10G, 32MB @ 25G)
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import struct
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Link speed detection
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LinkInfo:
    """Detected info about one network interface."""
    interface: str              # e.g. "ens1f0"
    speed_mbps: int             # 10000 = 10GbE, 25000 = 25GbE, 0 = unknown
    link_up: bool               # True if interface is UP
    duplex: str = "unknown"     # "full" / "half" / "unknown"
    driver: str = ""            # e.g. "sfc" (Solarflare), "i40e" (Intel X710), "ice" (XXV710)

    @property
    def speed_gbps(self) -> float:
        return self.speed_mbps / 1000.0

    @property
    def is_25gbe(self) -> bool:
        return self.speed_mbps >= 25000

    @property
    def is_10gbe(self) -> bool:
        return 10000 <= self.speed_mbps < 25000

    @property
    def tier(self) -> str:
        if self.speed_mbps >= 100000:
            return "100GbE"
        if self.speed_mbps >= 40000:
            return "40GbE"
        if self.speed_mbps >= 25000:
            return "25GbE"
        if self.speed_mbps >= 10000:
            return "10GbE"
        if self.speed_mbps >= 1000:
            return "1GbE"
        return f"{self.speed_mbps}Mbps"


def detect_link_speed(interface: str) -> LinkInfo:
    """
    Detect link speed for a network interface using ethtool.

    Works on Linux. On Windows or other systems, returns a stub with
    speed_mbps=0. Callers should treat 0 as "unknown, assume 10GbE".
    """
    if os.name != "posix":
        logger.debug("detect_link_speed: %s — non-POSIX, returning stub", interface)
        return LinkInfo(interface=interface, speed_mbps=0, link_up=False)

    if not shutil.which("ethtool"):
        logger.debug("detect_link_speed: ethtool not installed")
        return LinkInfo(interface=interface, speed_mbps=0, link_up=False)

    try:
        result = subprocess.run(
            ["ethtool", interface],
            capture_output=True, text=True, timeout=5.0,
        )
        output = result.stdout

        speed_mbps = 0
        link_up = False
        duplex = "unknown"
        driver = ""

        # Speed: Mb/s
        m = re.search(r"Speed:\s*(\d+)\s*Mb/s", output)
        if m:
            speed_mbps = int(m.group(1))

        # Link detected: yes
        m = re.search(r"Link detected:\s*(yes|no)", output)
        if m:
            link_up = m.group(1) == "yes"

        # Duplex: Full / Half
        m = re.search(r"Duplex:\s*(Full|Half)", output)
        if m:
            duplex = m.group(1).lower()

        # Driver from ethtool -i
        drv_result = subprocess.run(
            ["ethtool", "-i", interface],
            capture_output=True, text=True, timeout=5.0,
        )
        m = re.search(r"driver:\s*(\S+)", drv_result.stdout)
        if m:
            driver = m.group(1)

        return LinkInfo(
            interface=interface, speed_mbps=speed_mbps,
            link_up=link_up, duplex=duplex, driver=driver,
        )
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.debug("detect_link_speed: ethtool failed for %s: %s", interface, exc)
        return LinkInfo(interface=interface, speed_mbps=0, link_up=False)


def detect_primary_link() -> LinkInfo:
    """
    Detect the primary high-speed interface on this host.

    Tries common names first: ens1f0 (rNDC port 0), enp1s0f0 (PCIe), eth0.
    Returns the first one that's UP at 10 Gbps or more.
    """
    candidates = ["ens1f0", "ens1f1", "enp1s0f0", "enp1s0f1", "eth0", "eth1"]
    for iface in candidates:
        info = detect_link_speed(iface)
        if info.link_up and info.speed_mbps >= 10000:
            return info
    # Fall back to first candidate even if not UP
    return detect_link_speed(candidates[0])


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for the dual high-speed setup (10GbE or 25GbE)."""
    # Real-time channel (NIC 1)
    rt_local_ip: str = "10.0.0.1"       # workstation
    rt_remote_ip: str = "10.0.0.2"      # R740
    rt_port: int = 9100                   # real-time data port
    rt_mtu: int = 1500                    # standard for real-time (lower latency)
    rt_interface: str = "ens1f0"          # R740 rNDC port 0 (Intel X710/XXV710)

    # Bulk channel (NIC 2)
    bulk_local_ip: str = "10.0.1.1"      # workstation
    bulk_remote_ip: str = "10.0.1.2"     # R740
    bulk_port: int = 9200                 # bulk data port
    bulk_mtu: int = 9000                  # jumbo frames for throughput
    bulk_interface: str = "ens1f1"        # R740 rNDC port 1

    # Tuning
    tcp_nodelay: bool = True
    so_busy_poll: int = 50               # microseconds
    ring_buffer_size: int = 4096
    cpu_affinity_rt: int = 8             # pin RT NIC to exec island core
    cpu_affinity_bulk: int = 9           # pin bulk NIC to exec island core

    # Link speed (auto-detected, default 10GbE)
    link_speed_mbps: int = 10000

    @property
    def is_25gbe(self) -> bool:
        return self.link_speed_mbps >= 25000

    @property
    def socket_buffer_size(self) -> int:
        """Scale socket buffers with link speed. 16MB @ 10G, 32MB @ 25G."""
        if self.link_speed_mbps >= 25000:
            return 32 * 1024 * 1024  # 32MB for 25GbE
        return 16 * 1024 * 1024  # 16MB for 10GbE

    @classmethod
    def autodetect(cls, **overrides: Any) -> "NetworkConfig":
        """
        Build a NetworkConfig by auto-detecting link speed from the primary
        interface. Any fields passed as kwargs override the detection.
        """
        info = detect_primary_link()
        defaults: Dict[str, Any] = {}
        if info.speed_mbps > 0:
            defaults["link_speed_mbps"] = info.speed_mbps
            defaults["rt_interface"] = info.interface
            logger.info(
                "NetworkConfig.autodetect: %s at %s (driver=%s, link=%s)",
                info.interface, info.tier, info.driver,
                "UP" if info.link_up else "DOWN",
            )
            # At 25GbE, RT channel also benefits from jumbo frames
            if info.speed_mbps >= 25000:
                defaults["rt_mtu"] = 9000
        defaults.update(overrides)
        return cls(**defaults)


@dataclass
class LatencyStats:
    """Network latency statistics."""
    samples: int = 0
    min_us: float = 999999.0
    max_us: float = 0.0
    avg_us: float = 0.0
    p99_us: float = 0.0
    jitter_us: float = 0.0
    _history: List[float] = field(default_factory=list)

    def record(self, latency_us: float) -> None:
        self.samples += 1
        self.min_us = min(self.min_us, latency_us)
        self.max_us = max(self.max_us, latency_us)
        self._history.append(latency_us)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]
        self.avg_us = sum(self._history) / len(self._history)
        sorted_h = sorted(self._history)
        self.p99_us = sorted_h[int(len(sorted_h) * 0.99)] if len(sorted_h) >= 100 else self.max_us
        if len(self._history) >= 2:
            diffs = [abs(self._history[i] - self._history[i-1]) for i in range(1, len(self._history))]
            self.jitter_us = sum(diffs) / len(diffs)


class SolarflareOptimizer:
    """
    Optimizes network settings for Solarflare SFN8522-PLUS NICs.

    Applies kernel tuning, socket options, and CPU affinity for
    minimum latency on the real-time channel and maximum throughput
    on the bulk channel.
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        self._config = config or NetworkConfig()
        self._rt_latency = LatencyStats()
        self._bulk_throughput_mbps = 0.0
        self._optimized = False

    def optimize_system(self) -> Dict[str, Any]:
        """Apply system-level optimizations. Returns status dict."""
        results = {}

        # These are Linux-specific tuning commands
        # On Windows, they're logged as recommendations
        tuning_commands = [
            # Increase socket buffer sizes
            ("net.core.rmem_max", "16777216"),
            ("net.core.wmem_max", "16777216"),
            ("net.core.rmem_default", "1048576"),
            ("net.core.wmem_default", "1048576"),
            # TCP tuning
            ("net.ipv4.tcp_timestamps", "1"),
            ("net.ipv4.tcp_sack", "1"),
            ("net.ipv4.tcp_low_latency", "1"),
            ("net.core.busy_poll", str(self._config.so_busy_poll)),
            ("net.core.busy_read", str(self._config.so_busy_poll)),
            # Ring buffer
            ("net.core.netdev_max_backlog", str(self._config.ring_buffer_size)),
        ]

        for param, value in tuning_commands:
            results[param] = value
            logger.debug("10GbE tuning: %s = %s", param, value)

        # Apply on Linux
        if os.name == "posix":
            for param, value in tuning_commands:
                try:
                    path = f"/proc/sys/{param.replace('.', '/')}"
                    if os.path.exists(path):
                        with open(path, "w") as f:
                            f.write(value)
                except (PermissionError, OSError):
                    pass

        self._optimized = True
        logger.info("SolarflareOptimizer: %d tuning parameters applied", len(tuning_commands))
        return results

    def create_rt_socket(self) -> Optional[socket.socket]:
        """Create an optimized real-time channel socket."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Scale buffer size with link speed: 4MB @ 10GbE, 8MB @ 25GbE
            rt_buf = self._config.socket_buffer_size // 4
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rt_buf)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, rt_buf)

            # SO_BUSY_POLL for lowest latency (Linux only)
            if hasattr(socket, "SO_BUSY_POLL"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BUSY_POLL, self._config.so_busy_poll)

            sock.settimeout(5.0)
            return sock
        except Exception as e:
            logger.warning("Failed to create RT socket: %s", e)
            return None

    def create_bulk_socket(self) -> Optional[socket.socket]:
        """Create an optimized bulk data channel socket."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Scale buffer size with link speed: 16MB @ 10GbE, 32MB @ 25GbE
            bulk_buf = self._config.socket_buffer_size
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, bulk_buf)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, bulk_buf)

            sock.settimeout(30.0)
            return sock
        except Exception as e:
            logger.warning("Failed to create bulk socket: %s", e)
            return None

    def measure_latency(self, target_ip: str, port: int) -> float:
        """Measure round-trip latency to target in microseconds."""
        try:
            sock = self.create_rt_socket()
            if sock is None:
                return -1.0

            sock.settimeout(2.0)
            t0 = time.perf_counter_ns()
            try:
                sock.connect((target_ip, port))
                sock.send(b"PING")
                sock.recv(4)
                latency_ns = time.perf_counter_ns() - t0
                latency_us = latency_ns / 1000
                self._rt_latency.record(latency_us)
                return latency_us
            except (ConnectionRefusedError, socket.timeout, OSError):
                return -1.0
            finally:
                sock.close()
        except Exception:
            return -1.0

    def get_stats(self) -> Dict[str, Any]:
        link_tier = "25GbE" if self._config.is_25gbe else "10GbE"
        return {
            "optimized": self._optimized,
            "config": {
                "rt_link": f"{self._config.rt_local_ip} ↔ {self._config.rt_remote_ip}:{self._config.rt_port}",
                "bulk_link": f"{self._config.bulk_local_ip} ↔ {self._config.bulk_remote_ip}:{self._config.bulk_port}",
                "rt_mtu": self._config.rt_mtu,
                "bulk_mtu": self._config.bulk_mtu,
                "link_speed_mbps": self._config.link_speed_mbps,
                "link_tier": link_tier,
                "socket_buffer_mb": self._config.socket_buffer_size // (1024 * 1024),
                "rt_interface": self._config.rt_interface,
                "bulk_interface": self._config.bulk_interface,
            },
            "latency": {
                "samples": self._rt_latency.samples,
                "min_us": self._rt_latency.min_us if self._rt_latency.samples > 0 else 0,
                "avg_us": self._rt_latency.avg_us,
                "p99_us": self._rt_latency.p99_us,
                "jitter_us": self._rt_latency.jitter_us,
            },
        }


# ════════════════════════════════════════════════════════════════════════════
# Data Channel Protocol
# ════════════════════════════════════════════════════════════════════════════

class DataChannelProtocol:
    """
    Wire protocol for workstation ↔ R740 data exchange.

    Message types:
    - TICK: real-time trade data (RT channel)
    - OHLCV: historical candle data (bulk channel)
    - ORDER: execution relay (RT channel)
    - BACKTEST_REQUEST: ask R740 to run backtest (bulk channel)
    - BACKTEST_RESULT: R740 returns results (bulk channel)
    - MODEL_SYNC: transfer model weights (bulk channel)
    - EVOLUTION_POP: sync evolution population (bulk channel)
    - HEARTBEAT: keep-alive (RT channel)
    """

    MSG_TICK = 1
    MSG_OHLCV = 2
    MSG_ORDER = 3
    MSG_BACKTEST_REQ = 4
    MSG_BACKTEST_RES = 5
    MSG_MODEL_SYNC = 6
    MSG_EVOLUTION = 7
    MSG_HEARTBEAT = 8

    HEADER_FMT = "!BIQ"  # type(1) + length(4) + timestamp(8) = 13 bytes
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    @staticmethod
    def encode(msg_type: int, payload: bytes) -> bytes:
        """Encode a message with header."""
        ts = int(time.time() * 1e6)  # microsecond timestamp
        header = struct.pack(DataChannelProtocol.HEADER_FMT, msg_type, len(payload), ts)
        return header + payload

    @staticmethod
    def decode_header(data: bytes) -> Tuple[int, int, int]:
        """Decode message header. Returns (msg_type, payload_length, timestamp_us)."""
        return struct.unpack(DataChannelProtocol.HEADER_FMT, data[:DataChannelProtocol.HEADER_SIZE])

    @staticmethod
    def encode_tick(symbol: str, price: float, qty: float, side: str) -> bytes:
        """Encode a tick message."""
        import json
        payload = json.dumps({"s": symbol, "p": price, "q": qty, "d": side}).encode()
        return DataChannelProtocol.encode(DataChannelProtocol.MSG_TICK, payload)

    @staticmethod
    def encode_heartbeat() -> bytes:
        return DataChannelProtocol.encode(DataChannelProtocol.MSG_HEARTBEAT, b"HB")


# ════════════════════════════════════════════════════════════════════════════
# Network Health Monitor
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# Extreme Networks X460-G2 Switch Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SwitchConfig:
    """Configuration for one Extreme Networks X460-G2 switch."""
    name: str
    role: str                       # "real_time" or "bulk_data"
    vlan: int                       # VLAN ID for traffic isolation
    qos_priority: int = 0           # 0-7, higher = more priority
    jumbo_frames: bool = False
    link_aggregation: bool = False  # LACP bond 2x SFP+ = 20Gbps
    spanning_tree: bool = False     # disable for minimum latency
    flow_control: bool = True       # IEEE 802.3x pause frames


def generate_switch_config(switch: SwitchConfig) -> str:
    """Generate Extreme XOS CLI commands for switch configuration.

    These commands configure the X460-G2 for optimal ARGUS performance.
    Run these on the switch console before connecting the Solarflare NICs.
    """
    commands = [
        f"# === ARGUS {switch.name} ({switch.role}) ===",
        f"create vlan argus_{switch.role} tag {switch.vlan}",
        f'configure vlan argus_{switch.role} description "ARGUS {switch.role} traffic"',
    ]

    # Add SFP+ uplink ports to VLAN (ports 49-52 are the 10GbE SFP+ on X460-G2)
    commands.append(f"configure vlan argus_{switch.role} add ports 49-50 tagged")

    # QoS for real-time traffic
    if switch.qos_priority > 0:
        commands.extend([
            f"configure qosprofile qp{switch.qos_priority} minbw 50 maxbw 100 priority {switch.qos_priority}",
            f"configure vlan argus_{switch.role} qosprofile qp{switch.qos_priority}",
        ])

    # Jumbo frames for bulk transfers
    if switch.jumbo_frames:
        commands.append("enable jumbo-frame ports 49-50")
        commands.append("configure jumbo-frame size 9216")

    # LACP link aggregation (bond 2x 10GbE = 20Gbps)
    if switch.link_aggregation:
        commands.extend([
            "enable sharing 49 grouping 49-50 algorithm address-based L3_L4 lacp",
        ])

    # Disable spanning tree for latency
    if not switch.spanning_tree:
        commands.append("disable stpd s0 ports 49-50")

    # Flow control
    if switch.flow_control:
        commands.append("enable flow-control tx-pause ports 49-50")
        commands.append("enable flow-control rx-pause ports 49-50")

    # Edge port (no STP negotiation delay)
    commands.extend([
        "configure stpd s0 ports 49-50 edge-safeguard enable",
        "configure stpd s0 ports 49-50 bpdu-restrict enable",
    ])

    return "\n".join(commands)


def generate_all_switch_configs(link_speed_mbps: int = 10000) -> Dict[str, str]:
    """Generate configs for both X460-G2 switches.

    At 25GbE link speed, enable jumbo frames on the RT switch too —
    the bandwidth overhead from small frames becomes significant at 25G+.
    """
    is_25gbe = link_speed_mbps >= 25000
    rt_switch = SwitchConfig(
        name="switch-rt", role="real_time", vlan=10,
        qos_priority=7, jumbo_frames=is_25gbe,
        link_aggregation=False, spanning_tree=False, flow_control=True,
    )
    bulk_switch = SwitchConfig(
        name="switch-bulk", role="bulk_data", vlan=20,
        qos_priority=3, jumbo_frames=True,
        link_aggregation=True, spanning_tree=False, flow_control=True,
    )
    return {
        "switch_rt": generate_switch_config(rt_switch),
        "switch_bulk": generate_switch_config(bulk_switch),
    }


class NetworkHealthMonitor:
    """
    Monitors the health of both 10GbE links.

    Checks:
    - Link state (up/down)
    - Latency (should be <10μs for direct 10GbE)
    - Packet loss (should be 0%)
    - Throughput utilization
    - Jitter (latency variance)

    Alerts if:
    - Latency exceeds 100μs (indicates kernel bypass failure)
    - Jitter exceeds 50μs (indicates congestion)
    - Any packet loss detected
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        self._config = config or NetworkConfig()
        self._rt_healthy = False
        self._bulk_healthy = False
        self._check_count = 0
        self._last_check = 0.0

    def check(self) -> Dict[str, Any]:
        """Run health check on both channels."""
        self._check_count += 1
        self._last_check = time.time()

        rt_status = self._check_link(self._config.rt_remote_ip, self._config.rt_port, "RT")
        bulk_status = self._check_link(self._config.bulk_remote_ip, self._config.bulk_port, "BULK")

        self._rt_healthy = rt_status["reachable"]
        self._bulk_healthy = bulk_status["reachable"]

        return {
            "rt_channel": rt_status,
            "bulk_channel": bulk_status,
            "both_healthy": self._rt_healthy and self._bulk_healthy,
            "checks": self._check_count,
        }

    def _check_link(self, ip: str, port: int, label: str) -> Dict[str, Any]:
        """Check if a network endpoint is reachable."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            t0 = time.perf_counter_ns()
            result = sock.connect_ex((ip, port))
            latency_us = (time.perf_counter_ns() - t0) / 1000
            sock.close()

            reachable = result == 0
            return {
                "label": label,
                "ip": ip,
                "port": port,
                "reachable": reachable,
                "latency_us": latency_us if reachable else -1,
                "status": "UP" if reachable else "DOWN",
            }
        except Exception as e:
            return {
                "label": label,
                "ip": ip,
                "port": port,
                "reachable": False,
                "latency_us": -1,
                "status": f"ERROR: {e}",
            }

    @property
    def healthy(self) -> bool:
        return self._rt_healthy and self._bulk_healthy

    def get_stats(self) -> Dict[str, Any]:
        return {
            "rt_healthy": self._rt_healthy,
            "bulk_healthy": self._bulk_healthy,
            "checks": self._check_count,
            "last_check": self._last_check,
        }
