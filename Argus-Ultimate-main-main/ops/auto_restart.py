"""
Auto-Restart with State Preservation — manages Argus process lifecycle.

Handles:
  - Starting Argus as a subprocess (paper or live mode)
  - Graceful stop (SIGTERM → wait → SIGKILL)
  - Restart with exponential backoff (10s, 30s, 60s, 120s, 300s)
  - Max 5 restarts within 1 hour to prevent restart loops
  - Preserves checkpoint data across restarts (checkpoints.db untouched)

Usage:
  restarter = AutoRestarter(project_root=Path("F:/Argus-Ultimate-main"))
  restarter.start_argus(mode="paper")
  restarter.restart_argus()
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Exponential backoff delays (seconds) for consecutive restarts
_BACKOFF_DELAYS = [10, 30, 60, 120, 300]

# Max restarts within the window before refusing to restart
_MAX_RESTARTS = 5
_RESTART_WINDOW_SECONDS = 3600  # 1 hour


@dataclass
class RestartRecord:
    """Record of a single restart event."""
    timestamp: float
    reason: str
    success: bool


class AutoRestarter:
    """Manages Argus process lifecycle with state preservation and backoff."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        py_executable: str = "py",
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent)
        self.py_executable = py_executable
        self._process: Optional[subprocess.Popen] = None
        self._start_time: Optional[float] = None
        self._mode: str = "paper"
        self._restart_history: List[RestartRecord] = []
        self._consecutive_restarts: int = 0

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def start_argus(self, mode: str = "paper") -> bool:
        """Start Argus as a subprocess. Returns True if started successfully."""
        if self.is_running():
            logger.warning("Argus is already running (PID %s)", self._process.pid)
            return True

        self._mode = mode
        cmd = [self.py_executable, "-B", "main.py", mode]
        logger.info("Starting Argus: %s (cwd=%s)", " ".join(cmd), self.project_root)

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            self._start_time = time.time()
            logger.info("Argus started with PID %s", self._process.pid)
            return True
        except FileNotFoundError:
            logger.error("Could not find executable: %s", self.py_executable)
            return False
        except OSError as exc:
            logger.error("Failed to start Argus: %s", exc)
            return False

    def stop_argus(self, timeout: float = 30.0) -> bool:
        """
        Stop Argus gracefully.
        Sends SIGTERM (or CTRL_BREAK on Windows), waits *timeout* seconds,
        then SIGKILL if still running.  Returns True if process is dead.
        """
        if not self.is_running():
            logger.info("Argus is not running — nothing to stop")
            self._process = None
            return True

        pid = self._process.pid
        logger.info("Stopping Argus (PID %s) — sending SIGTERM, waiting %ss", pid, timeout)

        try:
            if sys.platform == "win32":
                # On Windows, CTRL_BREAK_EVENT is the closest to SIGTERM
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                self._process.terminate()
        except OSError as exc:
            logger.warning("SIGTERM failed: %s", exc)

        try:
            self._process.wait(timeout=timeout)
            logger.info("Argus (PID %s) terminated gracefully", pid)
            self._process = None
            return True
        except subprocess.TimeoutExpired:
            logger.warning("Argus did not stop in %ss — sending SIGKILL", timeout)

        try:
            self._process.kill()
            self._process.wait(timeout=10)
            logger.info("Argus (PID %s) killed", pid)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.error("Failed to kill Argus (PID %s): %s", pid, exc)
            return False

        self._process = None
        return True

    def restart_argus(self, reason: str = "manual") -> bool:
        """
        Stop + start Argus.  Applies exponential backoff and respects
        the max-restart-per-hour limit.  Returns True on success.
        """
        if not self._can_restart():
            logger.error(
                "Max restarts (%d) within %d seconds exceeded — refusing to restart",
                _MAX_RESTARTS,
                _RESTART_WINDOW_SECONDS,
            )
            self._restart_history.append(
                RestartRecord(time.time(), reason, success=False)
            )
            return False

        delay = self._get_backoff_delay()
        if delay > 0:
            logger.info("Backoff: waiting %ds before restart (attempt %d)", delay, self._consecutive_restarts + 1)
            time.sleep(delay)

        self.stop_argus()
        self._clean_paper_state()

        success = self.start_argus(mode=self._mode)
        self._restart_history.append(
            RestartRecord(time.time(), reason, success=success)
        )

        if success:
            self._consecutive_restarts += 1
            logger.info("Argus restarted successfully (attempt %d)", self._consecutive_restarts)
        else:
            self._consecutive_restarts += 1
            logger.error("Argus restart failed (attempt %d)", self._consecutive_restarts)

        return success

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Check if the Argus subprocess is still alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_uptime(self) -> float:
        """Get uptime in seconds (0.0 if not running)."""
        if not self.is_running() or self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_pid(self) -> Optional[int]:
        """Return PID of Argus process (None if not running)."""
        if self._process is not None and self.is_running():
            return self._process.pid
        return None

    def get_restart_history(self) -> List[RestartRecord]:
        """Return list of past restart records."""
        return list(self._restart_history)

    def reset_restart_counter(self) -> None:
        """Reset consecutive restart counter (e.g., after a sustained healthy period)."""
        self._consecutive_restarts = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _can_restart(self) -> bool:
        """Check if we're allowed to restart (max N within window)."""
        now = time.time()
        cutoff = now - _RESTART_WINDOW_SECONDS
        recent = [r for r in self._restart_history if r.timestamp >= cutoff]
        return len(recent) < _MAX_RESTARTS

    def _get_backoff_delay(self) -> int:
        """Return backoff delay in seconds for the current consecutive restart count."""
        if self._consecutive_restarts == 0:
            return 0
        idx = min(self._consecutive_restarts - 1, len(_BACKOFF_DELAYS) - 1)
        return _BACKOFF_DELAYS[idx]

    def _clean_paper_state(self) -> None:
        """
        Clean transient paper-trading state files.
        Preserves checkpoints.db and other persistent data.
        """
        kill_switch = self.project_root / "KILL_SWITCH"
        if kill_switch.exists():
            try:
                kill_switch.unlink()
                logger.info("Cleared KILL_SWITCH file before restart")
            except OSError as exc:
                logger.warning("Failed to clear KILL_SWITCH: %s", exc)

        # Clean stale PID lock if present
        pid_lock = self.project_root / "data" / "argus.pid"
        if pid_lock.exists():
            try:
                pid_lock.unlink()
                logger.info("Cleared stale PID lock")
            except OSError:
                pass
