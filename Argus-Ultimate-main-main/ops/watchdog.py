"""
ARGUS Watchdog Daemon — main self-healing orchestrator.

Monitors Argus health and takes automated corrective action:
  - Health check every 30s via HTTP endpoint
  - 3 consecutive failures → restart
  - KILL_SWITCH auto-clear in paper mode → restart
  - Process not running → auto-start
  - Log anomaly detection (delegates to LogAnomalyDetector)
  - All actions logged to data/watchdog.log

CLI: py -B -m ops.watchdog --mode paper --check-interval 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ArgusWatchdog:
    """
    Main watchdog daemon that monitors and self-heals the Argus trading system.
    Runs as a separate process alongside Argus.
    """

    def __init__(
        self,
        mode: str = "paper",
        health_url: str = "http://localhost:8080/health",
        check_interval: int = 30,
        max_consecutive_failures: int = 3,
        project_root: Optional[str | Path] = None,
        py_executable: str = "py",
    ) -> None:
        self.mode = mode
        self.health_url = health_url
        self.check_interval = check_interval
        self.max_consecutive_failures = max_consecutive_failures
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent)
        self.py_executable = py_executable

        # Lazy-loaded components
        self._restarter = None
        self._anomaly_detector = None
        self._config_tuner = None

        # State
        self._running = False
        self._consecutive_failures = 0
        self._total_restarts = 0
        self._last_health_status: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the watchdog main loop (blocking)."""
        self._running = True
        logger.info(
            "ARGUS Watchdog started — mode=%s, health=%s, interval=%ds",
            self.mode, self.health_url, self.check_interval,
        )

        # Ensure Argus is running
        if not self._get_restarter().is_running():
            logger.info("Argus not running — starting in %s mode", self.mode)
            self._get_restarter().start_argus(mode=self.mode)

        # Start log anomaly detector
        log_path = self.project_root / "data" / "argus.log"
        if log_path.exists():
            self._get_anomaly_detector().watch(log_path)

        try:
            while self._running:
                self._check_cycle()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("Watchdog interrupted by Ctrl+C")
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the watchdog to stop."""
        self._running = False

    def check_health(self) -> bool:
        """
        Check Argus health endpoint. Returns True if healthy.
        Can be called directly for testing.
        """
        try:
            req = urllib.request.Request(self.health_url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                status = data.get("status", "unknown")
                self._last_health_status = status
                return status in ("ok", "degraded")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, Exception) as exc:
            self._last_health_status = f"error: {exc}"
            return False

    def check_kill_switch(self) -> bool:
        """
        Check for KILL_SWITCH file. If in paper mode, auto-clear and restart.
        Returns True if kill switch was found and handled.
        """
        kill_switch = self.project_root / "KILL_SWITCH"
        if not kill_switch.exists():
            return False

        logger.warning("KILL_SWITCH detected")

        if self.mode == "paper":
            try:
                content = kill_switch.read_text(encoding="utf-8").strip()
                logger.info("Kill switch content: %s", content)
                kill_switch.unlink()
                logger.info("KILL_SWITCH cleared (paper mode) — restarting")
                self._get_restarter().restart_argus(reason="kill_switch_paper_clear")
                self._total_restarts += 1
                return True
            except OSError as exc:
                logger.error("Failed to clear KILL_SWITCH: %s", exc)
                return True
        else:
            # Live mode: do NOT auto-clear — respect the kill switch
            logger.critical("KILL_SWITCH active in LIVE mode — NOT clearing. Manual intervention required.")
            return True

    def get_status(self) -> dict:
        """Return current watchdog status."""
        return {
            "mode": self.mode,
            "running": self._running,
            "consecutive_failures": self._consecutive_failures,
            "total_restarts": self._total_restarts,
            "last_health_status": self._last_health_status,
            "argus_running": self._get_restarter().is_running(),
            "argus_uptime": self._get_restarter().get_uptime(),
            "anomaly_stats": self._get_anomaly_detector().get_anomaly_stats() if self._anomaly_detector else {},
        }

    # ------------------------------------------------------------------
    # Main check cycle
    # ------------------------------------------------------------------

    def _check_cycle(self) -> None:
        """Run one complete check cycle."""
        # Check kill switch first
        if self.check_kill_switch():
            self._consecutive_failures = 0
            return

        # Check if process is alive
        if not self._get_restarter().is_running():
            logger.warning("Argus process not running — restarting")
            self._get_restarter().start_argus(mode=self.mode)
            self._total_restarts += 1
            self._consecutive_failures = 0
            return

        # Check health endpoint
        if self.check_health():
            if self._consecutive_failures > 0:
                logger.info(
                    "Health restored after %d failures", self._consecutive_failures
                )
                self._get_restarter().reset_restart_counter()
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            logger.warning(
                "Health check failed (%d/%d)",
                self._consecutive_failures,
                self.max_consecutive_failures,
            )

            if self._consecutive_failures >= self.max_consecutive_failures:
                logger.error(
                    "Health check failed %d times — restarting Argus",
                    self._consecutive_failures,
                )
                self._get_restarter().restart_argus(reason="health_check_failures")
                self._total_restarts += 1
                self._consecutive_failures = 0

        # Check for hung process via anomaly detector
        if self._anomaly_detector is not None:
            self._anomaly_detector.check_hung_process()

    # ------------------------------------------------------------------
    # Component access (lazy init)
    # ------------------------------------------------------------------

    def _get_restarter(self):
        if self._restarter is None:
            from ops.auto_restart import AutoRestarter
            self._restarter = AutoRestarter(
                project_root=self.project_root,
                py_executable=self.py_executable,
            )
        return self._restarter

    def _get_anomaly_detector(self):
        if self._anomaly_detector is None:
            from ops.log_anomaly_detector import LogAnomalyDetector
            self._anomaly_detector = LogAnomalyDetector(
                on_restart=lambda reason: self._get_restarter().restart_argus(reason=reason),
                on_clear_kill_switch=lambda: self._clear_kill_switch_file(),
                on_clear_positions=lambda: logger.warning("Position drift — clear stale positions"),
                on_increase_timeout=lambda: logger.info("Timeout flood — should increase timeout"),
                on_increase_delay=lambda: logger.info("Rate limit — should increase delay"),
            )
        return self._anomaly_detector

    def _clear_kill_switch_file(self) -> None:
        """Clear kill switch file if in paper mode."""
        ks = self.project_root / "KILL_SWITCH"
        if ks.exists() and self.mode == "paper":
            try:
                ks.unlink()
                logger.info("KILL_SWITCH cleared by anomaly detector")
            except OSError:
                pass

    def _shutdown(self) -> None:
        """Clean up on shutdown."""
        logger.info("Watchdog shutting down")
        if self._anomaly_detector is not None:
            self._anomaly_detector.stop()


# ======================================================================
# CLI
# ======================================================================


def _setup_logging(project_root: Path) -> None:
    """Configure logging to both file and console."""
    log_dir = project_root / "data"
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        str(log_dir / "watchdog.log"),
        maxBytes=10_000_000,
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS Watchdog Daemon")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"],
                        help="Trading mode (default: paper)")
    parser.add_argument("--check-interval", type=int, default=30,
                        help="Health check interval in seconds (default: 30)")
    parser.add_argument("--health-url", default="http://localhost:8080/health",
                        help="Argus health endpoint URL")
    parser.add_argument("--max-failures", type=int, default=3,
                        help="Consecutive health failures before restart")
    parser.add_argument("--project-root", default=None,
                        help="Project root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root or Path(__file__).resolve().parent.parent)
    _setup_logging(project_root)

    watchdog = ArgusWatchdog(
        mode=args.mode,
        health_url=args.health_url,
        check_interval=args.check_interval,
        max_consecutive_failures=args.max_failures,
        project_root=project_root,
    )

    # Graceful shutdown on signals
    def _signal_handler(sig: int, frame: Any) -> None:
        logger.info("Received signal %s — stopping watchdog", sig)
        watchdog.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    watchdog.start()


if __name__ == "__main__":
    main()
