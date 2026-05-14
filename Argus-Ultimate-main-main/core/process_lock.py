"""
Process Lock — prevents multiple ARGUS instances from running simultaneously.

Uses a PID file lock to ensure only one instance runs at a time.
Prevents double-trading and account confusion.

Usage (context manager):
    with ProcessLock("argus_paper") as lock:
        # Only one process runs here
        run_trading()

Or manual:
    lock = ProcessLock("argus_live")
    if not lock.acquire():
        sys.exit("Another instance is already running")
    try:
        run_trading()
    finally:
        lock.release()
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default lock directory: ~/.argus/locks
_DEFAULT_LOCK_DIR = Path.home() / ".argus" / "locks"


class ProcessLock:
    """
    Advisory PID-file lock for ARGUS trading system instances.

    The lock file contains only the owning process's PID as a plain integer.
    When the owning process exits (cleanly or not), the PID file is left
    behind.  The next caller detects that the recorded PID is dead, removes
    the stale file, and acquires the lock for itself.

    Thread-safety note: this is a *process*-level lock.  Multiple threads
    within the same process share the lock; the intention is to prevent a
    second *process* from starting.
    """

    def __init__(
        self,
        name: str = "argus",
        lock_dir: Optional[Path] = None,
        timeout: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        name:
            Logical name for this lock (used as the file stem).
            E.g. "argus_paper", "argus_live".
        lock_dir:
            Directory where the PID file is stored.
            Defaults to ~/.argus/locks/.
        timeout:
            Seconds to wait when attempting to acquire a contested lock
            before giving up.  Set to 0 for an instant non-blocking attempt.
        """
        self._name = name
        self._lock_dir: Path = lock_dir or _DEFAULT_LOCK_DIR
        self._timeout = timeout
        self._acquired = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _lock_path(self) -> Path:
        """Absolute path of the PID lock file."""
        return self._lock_dir / f"{self._name}.lock"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> bool:
        """
        Attempt to acquire the lock.

        Returns True if this process now owns the lock, False if another
        live process holds it and the timeout expired.

        Algorithm:
          1. Ensure the lock directory exists.
          2. If a PID file exists and the PID is alive → locked by another
             process; poll until timeout.
          3. If a PID file exists but the PID is dead → stale; remove it.
          4. Write our own PID.
        """
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._timeout

        while True:
            if not self._lock_path.exists():
                if self._write_pid():
                    self._acquired = True
                    logger.debug(
                        "ProcessLock '%s' acquired by PID %d",
                        self._name,
                        os.getpid(),
                    )
                    return True
                # Another process slipped in between our check and write —
                # fall through and re-evaluate.

            else:
                owner = self.get_owner_pid()
                if owner is None:
                    # File is unreadable / malformed → treat as stale.
                    self._remove_stale()
                elif not self._pid_is_running(owner):
                    logger.info(
                        "ProcessLock '%s': removing stale lock from dead PID %d",
                        self._name,
                        owner,
                    )
                    self._remove_stale()
                else:
                    # Another live process holds the lock.
                    if time.monotonic() >= deadline:
                        logger.warning(
                            "ProcessLock '%s': timeout waiting for PID %d to release",
                            self._name,
                            owner,
                        )
                        return False
                    time.sleep(0.1)
                    continue

        # Unreachable — loop exits via return statements above.

    def release(self) -> bool:
        """
        Release the lock by removing the PID file.

        Returns True if the lock was successfully released, False if we
        did not own it or the file was already gone.
        """
        if not self._acquired:
            logger.debug(
                "ProcessLock '%s': release() called but we don't own the lock",
                self._name,
            )
            return False

        try:
            pid_in_file = self.get_owner_pid()
            if pid_in_file == os.getpid():
                self._lock_path.unlink(missing_ok=True)
                self._acquired = False
                logger.debug(
                    "ProcessLock '%s' released by PID %d",
                    self._name,
                    os.getpid(),
                )
                return True
            else:
                # Someone else's PID is in the file — do not remove it.
                logger.warning(
                    "ProcessLock '%s': lock file contains PID %s, not ours (%d); not removing",
                    self._name,
                    pid_in_file,
                    os.getpid(),
                )
                self._acquired = False
                return False
        except OSError as exc:
            logger.error(
                "ProcessLock '%s': error releasing lock: %s", self._name, exc
            )
            return False

    def force_release(self) -> bool:
        """
        Forcibly remove the lock file regardless of ownership.

        Use this to clean up stale locks when the owning process has crashed
        and the lock file remains. Returns True if the file was removed,
        False if it was already gone or an error occurred.

        WARNING: This bypasses all safety checks. Only use when you are
        certain the lock holder is dead.
        """
        try:
            if self._lock_path.exists():
                owner = self.get_owner_pid()
                self._lock_path.unlink(missing_ok=True)
                self._acquired = False
                logger.info(
                    "ProcessLock '%s': force-released (was owned by PID %s)",
                    self._name,
                    owner,
                )
                return True
            else:
                logger.debug(
                    "ProcessLock '%s': force_release() — lock file does not exist",
                    self._name,
                )
                return False
        except OSError as exc:
            logger.error(
                "ProcessLock '%s': force_release() failed: %s", self._name, exc
            )
            return False

    def is_locked(self) -> bool:
        """
        Return True if a *live* process currently holds the lock.

        Does not consider whether *we* hold it; an external caller can use
        this to check whether ARGUS is already running before starting.
        """
        if not self._lock_path.exists():
            return False
        owner = self.get_owner_pid()
        if owner is None:
            return False
        return self._pid_is_running(owner)

    def get_owner_pid(self) -> Optional[int]:
        """
        Read the PID stored in the lock file.

        Returns None if the file does not exist, is empty, or is malformed.
        """
        try:
            raw = self._lock_path.read_text(encoding="utf-8").strip()
            return int(raw) if raw else None
        except (FileNotFoundError, ValueError):
            return None
        except OSError as exc:
            logger.debug(
                "ProcessLock '%s': could not read lock file: %s",
                self._name,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ProcessLock":
        if not self.acquire():
            owner = self.get_owner_pid()
            raise RuntimeError(
                f"Could not acquire ProcessLock '{self._name}': "
                f"another instance is running (PID {owner})."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.release()
        return None  # Do not suppress exceptions.

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_pid(self) -> bool:
        """
        Atomically write our PID to the lock file.

        Uses os.open with O_CREAT | O_EXCL to fail if the file already
        exists, providing a best-effort atomic create on POSIX systems.
        On Windows, this flag combination is also supported since Python 3.3.

        Returns True on success, False if the file already existed.
        """
        pid_str = str(os.getpid()).encode()
        try:
            fd = os.open(
                str(self._lock_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            try:
                os.write(fd, pid_str)
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            return False
        except OSError as exc:
            logger.error(
                "ProcessLock '%s': failed to write PID file: %s",
                self._name,
                exc,
            )
            return False

    def _remove_stale(self) -> None:
        """Remove a stale lock file, ignoring errors if already gone."""
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug(
                "ProcessLock '%s': could not remove stale lock: %s",
                self._name,
                exc,
            )

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        """
        Return True if the given PID refers to a running process.

        Cross-platform implementation:
          - POSIX: os.kill(pid, 0) raises ProcessLookupError if dead.
          - Windows: same call works; raises PermissionError if the process
            exists but is owned by another user (still "running").
        """
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # We lack permission to signal the process, but it exists.
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Module-level convenience: check from a script without instantiating
# ---------------------------------------------------------------------------

def is_argus_running(name: str = "argus", lock_dir: Optional[Path] = None) -> bool:
    """
    Quick check: is an ARGUS instance with the given lock name already running?

    Does not acquire the lock.
    """
    return ProcessLock(name=name, lock_dir=lock_dir).is_locked()


def acquire_or_exit(
    name: str = "argus",
    lock_dir: Optional[Path] = None,
    timeout: float = 5.0,
) -> ProcessLock:
    """
    Acquire the named lock or exit the process with a clear error message.

    Intended for use in entrypoints (main.py) that should not start if
    another instance is already running.

    Returns the acquired ProcessLock (caller must call .release() or use as
    a context manager).
    """
    lock = ProcessLock(name=name, lock_dir=lock_dir, timeout=timeout)
    if not lock.acquire():
        owner = lock.get_owner_pid()
        logger.error(
            "Another ARGUS instance '%s' is already running (PID %s). Exiting.",
            name,
            owner,
        )
        sys.exit(1)
    return lock
