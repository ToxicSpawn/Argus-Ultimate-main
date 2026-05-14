"""
Log File Anomaly Detector — real-time log analysis for self-healing.

Tails log files and detects patterns that require automated intervention:
  - Emergency stop / kill switch → clear + restart
  - Consecutive losses → warning (paper continues)
  - Timeout floods → increase timeout config + restart
  - Rate limit / throttle → increase inter-request delay
  - Position drift → clear stale positions
  - Coroutine never awaited → log as known issue
  - No cycle completion in 10 min → restart (hung process)

Thread-safe, non-blocking. Designed to run inside the watchdog process.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Record of a detected anomaly."""
    timestamp: float
    pattern_name: str
    line: str
    action_taken: str


class LogAnomalyDetector:
    """
    Real-time log anomaly detector.

    Call ``watch(log_path)`` to start tailing the log file in a background
    thread.  Detected anomalies are recorded and callbacks are fired for
    actions (restart, config change, etc.).
    """

    # Sliding window for rate-based detections
    _TIMEOUT_WINDOW = 600  # 10 minutes
    _TIMEOUT_THRESHOLD = 5
    _CYCLE_TIMEOUT = 600   # 10 minutes without a cycle → hung

    def __init__(
        self,
        on_restart: Optional[Callable[[str], None]] = None,
        on_clear_kill_switch: Optional[Callable[[], None]] = None,
        on_clear_positions: Optional[Callable[[], None]] = None,
        on_increase_timeout: Optional[Callable[[], None]] = None,
        on_increase_delay: Optional[Callable[[], None]] = None,
    ) -> None:
        # Callbacks for self-healing actions
        self._on_restart = on_restart
        self._on_clear_kill_switch = on_clear_kill_switch
        self._on_clear_positions = on_clear_positions
        self._on_increase_timeout = on_increase_timeout
        self._on_increase_delay = on_increase_delay

        # Pattern match counts
        self._stats: Dict[str, int] = defaultdict(int)
        self._anomaly_history: List[AnomalyEvent] = []
        self._lock = threading.Lock()

        # Timeout tracking (sliding window)
        self._timeout_timestamps: Deque[float] = deque()

        # Cycle completion tracking
        self._last_cycle_time: float = time.time()

        # Thread management
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def watch(self, log_path: str | Path) -> None:
        """Start tailing the log file in a background thread."""
        if self._watch_thread is not None and self._watch_thread.is_alive():
            logger.warning("Already watching a log file")
            return

        self._stop_event.clear()
        self._last_cycle_time = time.time()
        self._watch_thread = threading.Thread(
            target=self._tail_loop,
            args=(Path(log_path),),
            name="LogAnomalyDetector",
            daemon=True,
        )
        self._watch_thread.start()
        logger.info("Log anomaly detector started on %s", log_path)

    def stop(self) -> None:
        """Stop the background watcher thread."""
        self._stop_event.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=5)
            self._watch_thread = None

    def analyze_line(self, line: str) -> Optional[str]:
        """
        Analyze a single log line. Returns the action name if one was taken,
        None otherwise. Can be called directly for testing.
        """
        stripped = line.strip()
        if not stripped:
            return None

        # Check for cycle completion (reset hung-process timer)
        if re.search(r"Cycle.*complete", stripped, re.IGNORECASE):
            self._last_cycle_time = time.time()
            return None

        # Pattern: Emergency stop / kill switch
        if "Emergency stop triggered" in stripped or "KILL_SWITCH detected" in stripped:
            return self._handle_kill_switch(stripped)

        # Pattern: Maximum consecutive losses
        if "Maximum consecutive losses" in stripped:
            return self._handle_consecutive_losses(stripped)

        # Pattern: TimeoutError
        if "TimeoutError" in stripped:
            return self._handle_timeout(stripped)

        # Pattern: Rate limit / throttle
        if re.search(r"rate.?limit|throttle", stripped, re.IGNORECASE):
            return self._handle_rate_limit(stripped)

        # Pattern: Position drift
        if "POSITION DRIFT" in stripped:
            return self._handle_position_drift(stripped)

        # Pattern: coroutine never awaited (known issue, just log)
        if re.search(r"coroutine.*never awaited", stripped, re.IGNORECASE):
            return self._handle_coroutine_warning(stripped)

        return None

    def get_anomaly_stats(self) -> Dict[str, int]:
        """Return dict of pattern_name → count."""
        with self._lock:
            return dict(self._stats)

    def get_anomaly_history(self) -> List[AnomalyEvent]:
        """Return list of all anomaly events."""
        with self._lock:
            return list(self._anomaly_history)

    def check_hung_process(self) -> bool:
        """
        Check if the process appears hung (no cycle completion in _CYCLE_TIMEOUT).
        Returns True if hung and restart was triggered.
        """
        elapsed = time.time() - self._last_cycle_time
        if elapsed > self._CYCLE_TIMEOUT:
            self._record("hung_process", "No cycle completion for {:.0f}s".format(elapsed), "restart")
            if self._on_restart:
                self._on_restart("hung_process")
            return True
        return False

    # ------------------------------------------------------------------
    # Pattern handlers
    # ------------------------------------------------------------------

    def _handle_kill_switch(self, line: str) -> str:
        self._record("kill_switch", line, "clear_and_restart")
        if self._on_clear_kill_switch:
            self._on_clear_kill_switch()
        if self._on_restart:
            self._on_restart("kill_switch")
        return "clear_and_restart"

    def _handle_consecutive_losses(self, line: str) -> str:
        self._record("consecutive_losses", line, "warning_only")
        logger.warning("Consecutive losses detected — continuing in paper mode")
        return "warning_only"

    def _handle_timeout(self, line: str) -> str:
        now = time.time()
        self._timeout_timestamps.append(now)

        # Prune old entries outside the window
        cutoff = now - self._TIMEOUT_WINDOW
        while self._timeout_timestamps and self._timeout_timestamps[0] < cutoff:
            self._timeout_timestamps.popleft()

        self._record("timeout_error", line, "logged")

        if len(self._timeout_timestamps) >= self._TIMEOUT_THRESHOLD:
            self._record("timeout_flood", line, "increase_timeout_and_restart")
            if self._on_increase_timeout:
                self._on_increase_timeout()
            if self._on_restart:
                self._on_restart("timeout_flood")
            self._timeout_timestamps.clear()
            return "increase_timeout_and_restart"

        return "logged"

    def _handle_rate_limit(self, line: str) -> str:
        self._record("rate_limit", line, "increase_delay")
        if self._on_increase_delay:
            self._on_increase_delay()
        return "increase_delay"

    def _handle_position_drift(self, line: str) -> str:
        self._record("position_drift", line, "clear_positions")
        if self._on_clear_positions:
            self._on_clear_positions()
        return "clear_positions"

    def _handle_coroutine_warning(self, line: str) -> str:
        self._record("coroutine_never_awaited", line, "known_issue")
        return "known_issue"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, pattern: str, line: str, action: str) -> None:
        with self._lock:
            self._stats[pattern] += 1
            self._anomaly_history.append(
                AnomalyEvent(
                    timestamp=time.time(),
                    pattern_name=pattern,
                    line=line[:500],
                    action_taken=action,
                )
            )

    def _tail_loop(self, log_path: Path) -> None:
        """Background thread: tail the log file and analyze each line."""
        last_check = time.time()

        # Wait for file to exist
        while not self._stop_event.is_set() and not log_path.exists():
            time.sleep(1.0)

        if self._stop_event.is_set():
            return

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                # Seek to end — we only care about new lines
                fh.seek(0, os.SEEK_END)

                while not self._stop_event.is_set():
                    line = fh.readline()
                    if line:
                        self.analyze_line(line)
                    else:
                        # No new data — check for hung process periodically
                        now = time.time()
                        if now - last_check > 60:
                            self.check_hung_process()
                            last_check = now
                        time.sleep(0.5)

        except OSError as exc:
            logger.error("Log tail error: %s", exc)
