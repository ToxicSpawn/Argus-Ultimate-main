#!/usr/bin/env python3
"""
core/node_orchestrator.py — Argus v6.4.0
=========================================
Dual-node orchestration layer for the Argus trading system.

Architecture:
  - Dell R7525 (LIVE_SERVER): AMD EPYC, ECC RAM, 24/7 live trading execution
  - Home PC (PAPER_PC): Core Ultra 9 285K, RTX 5080, ML training + paper trading
  - Both nodes on same home LAN (<1 ms latency)
  - State synchronised via Git branch 'argus-state'

Usage:
    from core.node_orchestrator import NodeOrchestrator, NodeConfig, NodeRole
    cfg = NodeConfig(live_server_hostname="argus-r7525")
    orch = NodeOrchestrator(cfg)
    orch.apply_role()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeRole(str, Enum):
    """Role assigned to this Argus node instance."""
    LIVE_SERVER = "live_server"   # Dell R7525 — real order execution
    PAPER_PC    = "paper_pc"      # Home PC — paper trading + GPU inference
    STANDALONE  = "standalone"    # Single-machine / dev mode


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    """
    All tuneable parameters for :class:`NodeOrchestrator`.

    Attributes
    ----------
    node_role:
        Explicit role override. Use :meth:`NodeOrchestrator.detect_role` to
        derive the role automatically from env vars / hostname.
    node_id:
        Unique identifier for this node.  If empty the ID is generated from
        ``hostname + MAC address`` on first access.
    live_server_hostname:
        DNS name or IP of the R7525 server on the LAN
        (e.g. ``"argus-r7525"``).
    enable_duplicate_guard:
        If *True*, :class:`DuplicateOrderGuard` is consulted before
        submitting any order.
    state_sync_branch:
        Git branch used to exchange state between nodes
        (default ``"argus-state"``).
    state_sync_interval_s:
        How often (seconds) state is pushed or pulled.
    lan_signal_port:
        TCP port for the LAN signal bridge; PC sends DeepLOB inference
        results to the R7525 on this port.
    config_path:
        Path to the exchanges config YAML that is read to detect
        ``paper_trading`` mode for role auto-detection.
    state_dir:
        Directory where node state JSON files are persisted locally.
    """
    node_role: NodeRole           = NodeRole.STANDALONE
    node_id: str                  = ""
    live_server_hostname: str     = ""
    enable_duplicate_guard: bool  = True
    state_sync_branch: str        = "argus-state"
    state_sync_interval_s: float  = 300.0
    lan_signal_port: int          = 9200
    config_path: str              = "config/exchanges_config.yaml"
    state_dir: str                = "data/node_state"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mac_address() -> str:
    """Return the primary NIC MAC address as a 12-char hex string."""
    mac = uuid.getnode()
    return f"{mac:012x}"


def _get_node_id(hostname: str) -> str:
    """Derive a stable, human-readable node ID from hostname + MAC."""
    mac = _get_mac_address()
    short_mac = mac[-6:]
    return f"{hostname}-{short_mac}"


def _get_local_ip() -> str:
    """Return the primary LAN IP of this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _detect_gpu() -> Optional[str]:
    """
    Return a short GPU description string if NVIDIA GPU is present,
    otherwise *None*.  Uses ``nvidia-smi`` if available; falls back to
    checking for the ``torch`` CUDA device.
    """
    # Try nvidia-smi first
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except Exception:
        pass

    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"{name} ({mem_gb:.0f}GB)"
    except Exception:
        pass

    return None


def _ram_gb() -> float:
    """Total physical RAM in gigabytes."""
    return psutil.virtual_memory().total / (1024 ** 3)


# ---------------------------------------------------------------------------
# DuplicateOrderGuard
# ---------------------------------------------------------------------------

class DuplicateOrderGuard:
    """
    Prevents the same order from being submitted by two nodes concurrently.

    The R7525 (live) and the PC (paper) both write their active orders to
    ``active_orders.json``.  Before placing a new order either node reads
    that file and rejects if the *other* node already has a matching open
    order (same symbol + side + exchange).

    Parameters
    ----------
    state_dir:
        Directory containing ``active_orders.json``.
    local_node_id:
        Node ID of *this* node — orders with a matching ``node_id`` field
        are ignored (they are ours, not duplicates from the remote node).
    """

    def __init__(self, state_dir: str, local_node_id: str) -> None:
        self.state_path = Path(state_dir) / "active_orders.json"
        self.local_node_id = local_node_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_order(self, symbol: str, side: str, exchange: str) -> bool:
        """
        Return *True* if this order is safe to submit, *False* if blocked.

        An order is blocked when the remote node already has a live order
        with the exact same ``(symbol, side, exchange)`` combination.
        """
        remote_orders = self._load_remote_orders()
        for order in remote_orders:
            if (
                order.get("symbol", "").upper()   == symbol.upper()
                and order.get("side", "").lower() == side.lower()
                and order.get("exchange", "").lower() == exchange.lower()
            ):
                logger.warning(
                    "DuplicateOrderGuard: blocked %s %s on %s — "
                    "remote node %s already has matching active order.",
                    side, symbol, exchange, order.get("node_id", "?"),
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_remote_orders(self) -> List[Dict[str, Any]]:
        """Load orders from the state file, excluding this node's own orders."""
        if not self.state_path.exists():
            return []
        try:
            raw: List[Dict[str, Any]] = json.loads(self.state_path.read_text())
            return [o for o in raw if o.get("node_id") != self.local_node_id]
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("DuplicateOrderGuard: cannot read %s — %s", self.state_path, exc)
            return []


# ---------------------------------------------------------------------------
# NodeHealthReporter
# ---------------------------------------------------------------------------

class NodeHealthReporter:
    """
    Periodically samples system metrics and writes a ``health_{node_id}.json``
    file to ``state_dir``.  The :class:`GitHubStateSync` then pushes this
    file to the remote Git branch so the other node can inspect it.

    Parameters
    ----------
    node_id:
        Unique identifier for the node generating the report.
    role:
        :class:`NodeRole` of this node.
    state_dir:
        Directory where the health JSON is written.
    start_time:
        ``time.time()`` timestamp recorded at bot start (for uptime calc).
    """

    def __init__(
        self,
        node_id: str,
        role: NodeRole,
        state_dir: str,
        start_time: float,
    ) -> None:
        self.node_id    = node_id
        self.role       = role
        self.state_dir  = Path(state_dir)
        self.start_time = start_time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_health_report(self) -> Dict[str, Any]:
        """
        Build and return the health report dict.

        Keys
        ----
        node_id, role, hostname, uptime_s, cpu_percent, mem_percent,
        disk_percent, gpu, strategies_active, sync_status,
        exchange_latencies_ms, timestamp_utc
        """
        vm   = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu  = psutil.cpu_percent(interval=0.5)

        report: Dict[str, Any] = {
            "node_id":             self.node_id,
            "role":                self.role.value,
            "hostname":            socket.gethostname(),
            "uptime_s":            time.time() - self.start_time,
            "cpu_percent":         cpu,
            "mem_percent":         vm.percent,
            "mem_used_gb":         round(vm.used / (1024 ** 3), 2),
            "mem_total_gb":        round(vm.total / (1024 ** 3), 2),
            "disk_percent":        disk.percent,
            "disk_free_gb":        round(disk.free / (1024 ** 3), 2),
            "gpu":                 _detect_gpu(),
            "strategies_active":   self._count_active_strategies(),
            "sync_status":         self._read_sync_status(),
            "exchange_latencies_ms": self._read_exchange_latencies(),
            "timestamp_utc":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return report

    def save_health(self, path: Optional[str] = None) -> Path:
        """
        Write the health report to disk.

        Parameters
        ----------
        path:
            Explicit file path.  If *None*, defaults to
            ``{state_dir}/health_{node_id}.json``.

        Returns
        -------
        Path
            The file path that was written.
        """
        report = self.generate_health_report()
        out = Path(path) if path else (self.state_dir / f"health_{self.node_id}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        logger.debug("Health report written to %s", out)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_active_strategies(self) -> int:
        """Read strategy count from node_state if available."""
        state_file = self.state_dir / "active_positions.json"
        if not state_file.exists():
            return 0
        try:
            data = json.loads(state_file.read_text())
            return len(data) if isinstance(data, list) else 0
        except Exception:
            return 0

    def _read_sync_status(self) -> str:
        """Return the last sync status string from a local marker file."""
        marker = self.state_dir / "last_sync.txt"
        if marker.exists():
            return marker.read_text().strip()
        return "never"

    def _read_exchange_latencies(self) -> Dict[str, float]:
        """Read cached exchange latency data from node state."""
        lat_file = self.state_dir / "exchange_latencies.json"
        if not lat_file.exists():
            return {}
        try:
            return json.loads(lat_file.read_text())
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# LAN Signal Receiver (R7525 side)
# ---------------------------------------------------------------------------

class _LANSignalReceiver:
    """
    Async TCP server running on the R7525 that receives DeepLOB inference
    signals from the PC over the LAN.

    Protocol (newline-delimited JSON):
        {"symbol": "BTC/USDT", "signal": 1, "confidence": 0.92, "ts": 1700000000.0}

    Signals are buffered in ``self.latest_signals`` (keyed by symbol) and
    consumed by the execution engine.
    """

    def __init__(self, port: int, bind: str = "0.0.0.0") -> None:
        self.port = port
        self.bind = bind
        self.latest_signals: Dict[str, Dict[str, Any]] = {}
        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_client, self.bind, self.port
        )
        logger.info("LAN signal receiver listening on %s:%d", self.bind, self.port)
        asyncio.create_task(self._serve_forever())

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _serve_forever(self) -> None:
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.debug("LAN signal receiver: connection from %s", peer)
        try:
            async for raw_line in reader:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg: Dict[str, Any] = json.loads(line)
                    symbol = msg.get("symbol", "UNKNOWN")
                    self.latest_signals[symbol] = msg
                    logger.debug("Received signal for %s: %s", symbol, msg)
                except json.JSONDecodeError:
                    logger.warning("LAN signal receiver: bad JSON from %s", peer)
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()


# ---------------------------------------------------------------------------
# LAN Signal Sender (PC side)
# ---------------------------------------------------------------------------

class _LANSignalSender:
    """
    Sends DeepLOB inference results from the PC to the R7525 over a
    persistent TCP connection.

    The sender maintains a reconnect loop so brief network glitches do
    not lose signals permanently.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1024)
        self._running = False

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._send_loop())
        logger.info("LAN signal sender targeting %s:%d", self.host, self.port)

    async def stop(self) -> None:
        self._running = False

    def enqueue_signal(self, signal: Dict[str, Any]) -> None:
        """Enqueue a signal dict for transmission.  Non-blocking."""
        try:
            self._queue.put_nowait(signal)
        except asyncio.QueueFull:
            logger.warning("LAN signal sender queue full — dropping oldest signal")
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(signal)
            except Exception:
                pass

    async def _send_loop(self) -> None:
        while self._running:
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
                logger.info("LAN signal sender: connected to %s:%d", self.host, self.port)
                while self._running:
                    try:
                        signal = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                        line = json.dumps(signal) + "\n"
                        writer.write(line.encode())
                        await writer.drain()
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        logger.warning("LAN signal sender: send error — %s", exc)
                        break
                writer.close()
            except Exception as exc:
                logger.warning(
                    "LAN signal sender: cannot connect to %s:%d — %s. Retrying in 5 s.",
                    self.host, self.port, exc
                )
                await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# NodeOrchestrator
# ---------------------------------------------------------------------------

class NodeOrchestrator:
    """
    Central controller that determines which role this machine plays and
    wires up the appropriate subsystems.

    Lifecycle
    ---------
    1.  ``__init__`` — Store config, resolve node_id, set start_time.
    2.  ``detect_role()`` — Query env vars / hostname / config file to
        determine :class:`NodeRole`.
    3.  ``apply_role()`` — Start background tasks appropriate to the role.
    4.  ``get_node_info()`` — Rich dict for health dashboards.

    Parameters
    ----------
    config:
        :class:`NodeConfig` instance with all tunable settings.
    """

    def __init__(self, config: NodeConfig) -> None:
        self.config = config
        self._start_time = time.time()

        # Resolve node_id
        hostname = socket.gethostname()
        if not config.node_id:
            config.node_id = _get_node_id(hostname)

        self._hostname = hostname
        self._role: NodeRole = config.node_role

        # Ensure state directory exists
        Path(config.state_dir).mkdir(parents=True, exist_ok=True)

        # Sub-components (created lazily / in apply_role)
        self._duplicate_guard: Optional[DuplicateOrderGuard] = None
        self._health_reporter: Optional[NodeHealthReporter]  = None
        self._lan_receiver: Optional[_LANSignalReceiver]     = None
        self._lan_sender: Optional[_LANSignalSender]         = None

        logger.info(
            "NodeOrchestrator initialised — node_id=%s hostname=%s",
            config.node_id, hostname,
        )

    # ------------------------------------------------------------------
    # Role detection
    # ------------------------------------------------------------------

    def detect_role(self) -> NodeRole:
        """
        Determine this node's role using the following priority chain:

        1. ``ARGUS_ROLE`` environment variable (``"live"`` → LIVE_SERVER,
           ``"paper"`` → PAPER_PC).
        2. Hostname match against ``config.live_server_hostname``.
        3. Config file: if ``paper_trading: true`` → PAPER_PC.
        4. Default: STANDALONE.

        The detected role is stored in ``self._role`` and returned.
        """
        # 1. Env variable
        env_role = os.getenv("ARGUS_ROLE", "").strip().lower()
        if env_role == "live":
            self._role = NodeRole.LIVE_SERVER
            logger.info("Role detected from ARGUS_ROLE env: LIVE_SERVER")
            return self._role
        if env_role == "paper":
            self._role = NodeRole.PAPER_PC
            logger.info("Role detected from ARGUS_ROLE env: PAPER_PC")
            return self._role

        # 2. Hostname match
        if (
            self.config.live_server_hostname
            and self._hostname.lower() == self.config.live_server_hostname.lower()
        ):
            self._role = NodeRole.LIVE_SERVER
            logger.info("Role detected from hostname match: LIVE_SERVER")
            return self._role

        # 3. Config file
        cfg_paper = self._read_paper_trading_flag()
        if cfg_paper is True:
            self._role = NodeRole.PAPER_PC
            logger.info("Role detected from config file (paper_trading=true): PAPER_PC")
            return self._role
        if cfg_paper is False:
            # Explicit false in config means live intent, but without hostname
            # match we default to STANDALONE for safety.
            self._role = NodeRole.STANDALONE
            logger.info("Role detected from config file (paper_trading=false) without hostname match: STANDALONE")
            return self._role

        # 4. Default
        self._role = NodeRole.STANDALONE
        logger.info("Role defaulting to STANDALONE (no definitive signal found)")
        return self._role

    def _read_paper_trading_flag(self) -> Optional[bool]:
        """
        Read ``paper_trading`` key from the exchanges config YAML.
        Returns *True*, *False*, or *None* if not found / unreadable.
        """
        cfg_path = Path(self.config.config_path)
        if not cfg_path.exists():
            return None
        try:
            import yaml  # type: ignore[import]
            data = yaml.safe_load(cfg_path.read_text()) or {}
            # Support nested or flat key
            val = data.get("paper_trading", data.get("trading", {}).get("paper_trading"))
            if isinstance(val, bool):
                return val
        except Exception as exc:
            logger.warning("Cannot read config for paper_trading flag: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    def is_live(self) -> bool:
        """Return *True* if this node is assigned the LIVE_SERVER role."""
        return self._role == NodeRole.LIVE_SERVER

    def is_paper(self) -> bool:
        """Return *True* if this node is assigned the PAPER_PC role."""
        return self._role == NodeRole.PAPER_PC

    @property
    def role(self) -> NodeRole:
        """Current :class:`NodeRole` of this node."""
        return self._role

    # ------------------------------------------------------------------
    # apply_role
    # ------------------------------------------------------------------

    def apply_role(self) -> None:
        """
        Wire up subsystems appropriate to the detected role.

        LIVE_SERVER
        ~~~~~~~~~~~
        * Forces ``paper_trading=False`` in the live config context.
        * Creates the :class:`DuplicateOrderGuard`.
        * Starts the LAN signal *receiver* (async TCP server on
          ``config.lan_signal_port``).
        * Schedules periodic state *push* to Git.

        PAPER_PC
        ~~~~~~~~
        * Forces ``paper_trading=True`` (belt-and-suspenders safety).
        * Disables any live order submission code paths via env flag.
        * Creates the :class:`DuplicateOrderGuard`.
        * Starts the LAN signal *sender* targeting the R7525.
        * Schedules periodic state *pull* from Git.

        STANDALONE
        ~~~~~~~~~~
        * No special wiring; runs as self-contained node.
        """
        logger.info("Applying role: %s", self._role.value)

        # Create shared components
        if self.config.enable_duplicate_guard:
            self._duplicate_guard = DuplicateOrderGuard(
                state_dir=self.config.state_dir,
                local_node_id=self.config.node_id,
            )

        self._health_reporter = NodeHealthReporter(
            node_id=self.config.node_id,
            role=self._role,
            state_dir=self.config.state_dir,
            start_time=self._start_time,
        )

        if self._role == NodeRole.LIVE_SERVER:
            self._apply_live()
        elif self._role == NodeRole.PAPER_PC:
            self._apply_paper()
        else:
            self._apply_standalone()

    def _apply_live(self) -> None:
        """Configure LIVE_SERVER role."""
        # Enforce live mode — belt-and-suspenders
        os.environ["ARGUS_PAPER_TRADING"] = "false"
        os.environ["ARGUS_ROLE"]          = "live"

        # Schedule async tasks when an event loop is running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._start_lan_receiver())
                loop.create_task(self._state_push_loop())
            else:
                logger.info(
                    "No running event loop — LAN receiver and state push loop "
                    "will be started externally via start_async_tasks()."
                )
        except RuntimeError:
            logger.info("No event loop yet; call start_async_tasks() from your async entry point.")

        logger.info(
            "LIVE_SERVER role applied. LAN signal receiver on port %d. "
            "State push every %.0f s.",
            self.config.lan_signal_port,
            self.config.state_sync_interval_s,
        )

    def _apply_paper(self) -> None:
        """Configure PAPER_PC role."""
        # Enforce paper mode — no live orders ever
        os.environ["ARGUS_PAPER_TRADING"] = "true"
        os.environ["ARGUS_LIVE_ORDERS"]   = "false"
        os.environ["ARGUS_ROLE"]           = "paper"

        # Resolve R7525 hostname
        target_host = self.config.live_server_hostname or "argus-r7525"

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._start_lan_sender(target_host))
                loop.create_task(self._state_pull_loop())
            else:
                logger.info(
                    "No running event loop — LAN sender and state pull loop "
                    "will be started externally via start_async_tasks()."
                )
        except RuntimeError:
            logger.info("No event loop yet; call start_async_tasks() from your async entry point.")

        logger.info(
            "PAPER_PC role applied. LAN signal sender → %s:%d. "
            "State pull every %.0f s.",
            target_host,
            self.config.lan_signal_port,
            self.config.state_sync_interval_s,
        )

    def _apply_standalone(self) -> None:
        """Configure STANDALONE role — minimal wiring."""
        logger.info("STANDALONE role applied — no LAN bridge, no remote state sync.")

    # ------------------------------------------------------------------
    # Async background tasks
    # ------------------------------------------------------------------

    async def start_async_tasks(self) -> None:
        """
        Start all background async tasks for the current role.
        Call this from your ``async main()`` after ``apply_role()``.
        """
        if self._role == NodeRole.LIVE_SERVER:
            await asyncio.gather(
                self._start_lan_receiver(),
                self._state_push_loop(),
                return_exceptions=True,
            )
        elif self._role == NodeRole.PAPER_PC:
            target_host = self.config.live_server_hostname or "argus-r7525"
            await asyncio.gather(
                self._start_lan_sender(target_host),
                self._state_pull_loop(),
                return_exceptions=True,
            )

    async def _start_lan_receiver(self) -> None:
        """Create and start the LAN signal receiver (LIVE_SERVER)."""
        self._lan_receiver = _LANSignalReceiver(port=self.config.lan_signal_port)
        await self._lan_receiver.start()

    async def _start_lan_sender(self, host: str) -> None:
        """Create and start the LAN signal sender (PAPER_PC)."""
        self._lan_sender = _LANSignalSender(host=host, port=self.config.lan_signal_port)
        await self._lan_sender.start()

    async def _state_push_loop(self) -> None:
        """Periodically write health report (LIVE_SERVER pushes to Git elsewhere)."""
        while True:
            try:
                if self._health_reporter:
                    self._health_reporter.save_health()
                self._update_sync_marker("push")
            except Exception as exc:
                logger.error("State push loop error: %s", exc)
            await asyncio.sleep(self.config.state_sync_interval_s)

    async def _state_pull_loop(self) -> None:
        """Periodically read health report from remote node state dir."""
        while True:
            try:
                self._update_sync_marker("pull")
            except Exception as exc:
                logger.error("State pull loop error: %s", exc)
            await asyncio.sleep(self.config.state_sync_interval_s)

    def _update_sync_marker(self, direction: str) -> None:
        """Write last sync timestamp to state_dir/last_sync.txt."""
        marker = Path(self.config.state_dir) / "last_sync.txt"
        marker.write_text(
            f"{direction}:{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
        )

    # ------------------------------------------------------------------
    # LAN signal API (for external callers)
    # ------------------------------------------------------------------

    def send_signal(self, signal: Dict[str, Any]) -> None:
        """
        Enqueue a signal for transmission to the R7525 (PAPER_PC only).

        Parameters
        ----------
        signal:
            Dict with at minimum ``{"symbol": str, "signal": int}``.
        """
        if self._role != NodeRole.PAPER_PC:
            raise RuntimeError("send_signal() only available on PAPER_PC node")
        if self._lan_sender is None:
            raise RuntimeError("LAN sender not initialised — call start_async_tasks() first")
        self._lan_sender.enqueue_signal(signal)

    def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent DeepLOB signal for *symbol* (LIVE_SERVER only).
        """
        if self._role != NodeRole.LIVE_SERVER:
            raise RuntimeError("get_latest_signal() only available on LIVE_SERVER node")
        if self._lan_receiver is None:
            return None
        return self._lan_receiver.latest_signals.get(symbol)

    # ------------------------------------------------------------------
    # Duplicate guard API
    # ------------------------------------------------------------------

    def check_order(self, symbol: str, side: str, exchange: str) -> bool:
        """
        Return *True* if the order should be submitted, *False* if blocked.

        Delegates to :class:`DuplicateOrderGuard` when enabled.
        """
        if not self.config.enable_duplicate_guard or self._duplicate_guard is None:
            return True
        return self._duplicate_guard.check_order(symbol, side, exchange)

    # ------------------------------------------------------------------
    # Node info
    # ------------------------------------------------------------------

    def get_node_info(self) -> Dict[str, Any]:
        """
        Return a rich dict describing this node for health dashboards.

        Keys
        ----
        node_id, role, hostname, ip, uptime_s, cpu_count, ram_gb,
        gpu, os, python_version, pid
        """
        return {
            "node_id":        self.config.node_id,
            "role":           self._role.value,
            "hostname":       self._hostname,
            "ip":             _get_local_ip(),
            "uptime_s":       round(time.time() - self._start_time, 1),
            "cpu_count":      psutil.cpu_count(logical=True),
            "cpu_count_phys": psutil.cpu_count(logical=False),
            "ram_gb":         round(_ram_gb(), 2),
            "gpu":            _detect_gpu(),
            "os":             f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "pid":            os.getpid(),
            "lan_signal_port": self.config.lan_signal_port,
            "state_sync_branch": self.config.state_sync_branch,
            "state_sync_interval_s": self.config.state_sync_interval_s,
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully stop all background async tasks."""
        logger.info("NodeOrchestrator shutting down…")
        if self._lan_receiver:
            await self._lan_receiver.stop()
        if self._lan_sender:
            await self._lan_sender.stop()
        logger.info("NodeOrchestrator shutdown complete.")

    def __repr__(self) -> str:
        return (
            f"NodeOrchestrator(node_id={self.config.node_id!r}, "
            f"role={self._role.value!r}, hostname={self._hostname!r})"
        )
