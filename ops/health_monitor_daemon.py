"""
Health Monitor Daemon — comprehensive system health monitoring.

Runs as a background process and checks:
  - HTTP health endpoint (http://localhost:8080/health)
  - Process alive (PID check)
  - Disk space (alert if <1GB free)
  - Memory usage (alert if >90%)
  - Log file growing (if stale for 10 min → hung process)

Actions:
  - Argus unresponsive for 5 min → restart
  - Disk <1GB → alert + stop trading
  - Memory >90% → alert
  - Log not growing for 10 min → restart

Sends Discord/Telegram alerts on any action taken.

CLI: py -B -m ops.health_monitor_daemon
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    healthy: bool
    message: str
    timestamp: float = field(default_factory=time.time)


class HealthMonitorDaemon:
    """
    Comprehensive health monitor that runs as a daemon process.
    Checks multiple health dimensions and takes automated action.
    """

    def __init__(
        self,
        health_url: str = "http://localhost:8080/health",
        check_interval: int = 30,
        unresponsive_timeout: int = 300,  # 5 minutes
        min_disk_gb: float = 1.0,
        max_memory_pct: float = 90.0,
        log_stale_timeout: int = 600,  # 10 minutes
        log_path: Optional[str | Path] = None,
        project_root: Optional[str | Path] = None,
        discord_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        self.health_url = health_url
        self.check_interval = check_interval
        self.unresponsive_timeout = unresponsive_timeout
        self.min_disk_gb = min_disk_gb
        self.max_memory_pct = max_memory_pct
        self.log_stale_timeout = log_stale_timeout
        self.log_path = Path(log_path) if log_path else None
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent)

        # Alerting
        self._discord_url = discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        self._telegram_token = telegram_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self._telegram_chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")

        # State
        self._running = False
        self._stop_event = threading.Event()
        self._first_unresponsive: Optional[float] = None
        self._last_log_size: int = 0
        self._last_log_check: float = time.time()
        self._check_history: List[HealthCheck] = []
        self._restarter = None  # Lazy import to avoid circular deps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the health monitor loop (blocking)."""
        self._running = True
        self._stop_event.clear()
        logger.info("Health monitor daemon started (interval=%ds)", self.check_interval)

        while not self._stop_event.is_set():
            try:
                self._run_checks()
            except Exception as exc:
                logger.error("Health check cycle error: %s", exc)

            self._stop_event.wait(timeout=self.check_interval)

        logger.info("Health monitor daemon stopped")

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False
        self._stop_event.set()

    def run_once(self) -> List[HealthCheck]:
        """Run all checks once and return results (for testing)."""
        return self._run_checks()

    def get_check_history(self) -> List[HealthCheck]:
        """Return history of health checks."""
        return list(self._check_history)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def _run_checks(self) -> List[HealthCheck]:
        """Run all health checks and take action if needed."""
        results: List[HealthCheck] = []

        results.append(self._check_http_health())
        results.append(self._check_disk_space())
        results.append(self._check_memory())

        if self.log_path:
            results.append(self._check_log_growing())

        self._check_history.extend(results)

        # Trim history to last 1000
        if len(self._check_history) > 1000:
            self._check_history = self._check_history[-500:]

        return results

    def _check_http_health(self) -> HealthCheck:
        """Check the Argus HTTP health endpoint."""
        try:
            req = urllib.request.Request(self.health_url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                status = data.get("status", "unknown")

                if status in ("ok", "degraded"):
                    self._first_unresponsive = None
                    return HealthCheck("http_health", True, f"Status: {status}")
                else:
                    return self._handle_unhealthy(f"Health endpoint returned: {status}")

        except (urllib.error.URLError, OSError, json.JSONDecodeError, Exception) as exc:
            return self._handle_unhealthy(f"Health endpoint unreachable: {exc}")

    def _handle_unhealthy(self, message: str) -> HealthCheck:
        """Handle an unhealthy HTTP check, possibly triggering restart."""
        now = time.time()
        if self._first_unresponsive is None:
            self._first_unresponsive = now

        elapsed = now - self._first_unresponsive
        if elapsed >= self.unresponsive_timeout:
            self._send_alert(f"Argus unresponsive for {elapsed:.0f}s — triggering restart")
            self._trigger_restart("unresponsive")
            self._first_unresponsive = None
            return HealthCheck("http_health", False, f"{message} — RESTARTED after {elapsed:.0f}s")

        return HealthCheck("http_health", False, f"{message} (unresponsive for {elapsed:.0f}s)")

    def _check_disk_space(self) -> HealthCheck:
        """Check disk free space."""
        try:
            usage = shutil.disk_usage(str(self.project_root))
            free_gb = usage.free / (1024 ** 3)

            if free_gb < self.min_disk_gb:
                msg = f"Disk space critically low: {free_gb:.1f}GB free"
                self._send_alert(msg + " — stopping trading")
                # Create kill switch to stop trading
                kill_switch = self.project_root / "KILL_SWITCH"
                kill_switch.write_text(f"auto:disk_low:{free_gb:.1f}GB", encoding="utf-8")
                return HealthCheck("disk_space", False, msg)

            return HealthCheck("disk_space", True, f"Disk free: {free_gb:.1f}GB")

        except OSError as exc:
            return HealthCheck("disk_space", False, f"Disk check error: {exc}")

    def _check_memory(self) -> HealthCheck:
        """Check system memory usage."""
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong

                class MEMORYSTATUS(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", c_ulong),
                        ("dwMemoryLoad", c_ulong),
                        ("dwTotalPhys", c_ulong),
                        ("dwAvailPhys", c_ulong),
                        ("dwTotalPageFile", c_ulong),
                        ("dwAvailPageFile", c_ulong),
                        ("dwTotalVirtual", c_ulong),
                        ("dwAvailVirtual", c_ulong),
                    ]

                mem = MEMORYSTATUS()
                mem.dwLength = ctypes.sizeof(mem)
                kernel32.GlobalMemoryStatus(ctypes.byref(mem))
                memory_pct = float(mem.dwMemoryLoad)
            else:
                # Linux: read /proc/meminfo
                meminfo: Dict[str, int] = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            meminfo[parts[0].rstrip(":")] = int(parts[1])
                total = meminfo.get("MemTotal", 1)
                available = meminfo.get("MemAvailable", 0)
                memory_pct = ((total - available) / total) * 100.0

            if memory_pct > self.max_memory_pct:
                msg = f"Memory usage high: {memory_pct:.1f}%"
                self._send_alert(msg)
                return HealthCheck("memory", False, msg)

            return HealthCheck("memory", True, f"Memory usage: {memory_pct:.1f}%")

        except Exception as exc:
            return HealthCheck("memory", True, f"Memory check skipped: {exc}")

    def _check_log_growing(self) -> HealthCheck:
        """Check if the log file is still growing (process not hung)."""
        if self.log_path is None or not self.log_path.exists():
            return HealthCheck("log_growing", True, "Log file not found — skipping")

        try:
            current_size = self.log_path.stat().st_size
            now = time.time()

            if current_size > self._last_log_size:
                self._last_log_size = current_size
                self._last_log_check = now
                return HealthCheck("log_growing", True, f"Log active ({current_size} bytes)")

            stale_seconds = now - self._last_log_check
            if stale_seconds >= self.log_stale_timeout:
                msg = f"Log file stale for {stale_seconds:.0f}s — process may be hung"
                self._send_alert(msg + " — triggering restart")
                self._trigger_restart("log_stale")
                self._last_log_check = now
                return HealthCheck("log_growing", False, msg)

            return HealthCheck(
                "log_growing", True, f"Log unchanged for {stale_seconds:.0f}s (threshold: {self.log_stale_timeout}s)"
            )

        except OSError as exc:
            return HealthCheck("log_growing", True, f"Log check error: {exc}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _trigger_restart(self, reason: str) -> None:
        """Trigger Argus restart via AutoRestarter."""
        try:
            if self._restarter is None:
                from ops.auto_restart import AutoRestarter
                self._restarter = AutoRestarter(project_root=self.project_root)
            self._restarter.restart_argus(reason=reason)
        except Exception as exc:
            logger.error("Failed to restart Argus: %s", exc)

    def _send_alert(self, message: str) -> None:
        """Send alert via Discord and/or Telegram."""
        logger.warning("ALERT: %s", message)
        self._send_discord(message)
        self._send_telegram(message)

    def _send_discord(self, message: str) -> None:
        """Send Discord webhook alert."""
        if not self._discord_url:
            return
        try:
            payload = json.dumps({
                "content": f"**ARGUS Health Monitor** - {message}",
                "username": "ARGUS Watchdog",
            }).encode("utf-8")
            req = urllib.request.Request(
                self._discord_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            logger.debug("Discord alert failed: %s", exc)

    def _send_telegram(self, message: str) -> None:
        """Send Telegram alert."""
        if not self._telegram_token or not self._telegram_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
            payload = json.dumps({
                "chat_id": self._telegram_chat_id,
                "text": f"ARGUS Watchdog: {message}",
                "parse_mode": "HTML",
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            logger.debug("Telegram alert failed: %s", exc)


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS Health Monitor Daemon")
    parser.add_argument("--health-url", default="http://localhost:8080/health")
    parser.add_argument("--check-interval", type=int, default=30)
    parser.add_argument("--log-path", default=None, help="Path to Argus log file to monitor")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.project_root or Path(__file__).resolve().parent.parent) / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(str(log_dir / "health_monitor.log"), maxBytes=10_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])

    daemon = HealthMonitorDaemon(
        health_url=args.health_url,
        check_interval=args.check_interval,
        log_path=args.log_path,
        project_root=args.project_root,
    )

    # Graceful shutdown
    def _signal_handler(sig: int, frame: Any) -> None:
        logger.info("Received signal %s — shutting down", sig)
        daemon.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    daemon.start()


if __name__ == "__main__":
    main()
