#!/usr/bin/env python3
"""
OpenOnload Launcher — wraps any command with `onload` binary + optimal EF_*
environment so the child process gets full kernel-bypass acceleration.

Usage::
    launcher = OnloadLauncher()
    if launcher.available:
        proc = launcher.launch(["python", "main.py", "--mode", "live"])
        proc.wait()
    else:
        # Fallback: run without onload
        subprocess.run(["python", "main.py", "--mode", "live"])

Or for the current process (re-exec with onload)::
    launcher.reexec_self()  # replaces current process with onload-wrapped self
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_ONLOAD_BINARY_CANDIDATES = [
    "onload",
    "/usr/bin/onload",
    "/usr/local/bin/onload",
    "/opt/onload/bin/onload",
]

# Optimal EF_* environment for minimum-latency trading
_LAUNCH_ENV: dict[str, str] = {
    "EF_POLL_USEC":             "100000",
    "EF_SPIN_USEC":             "100000",
    "EF_HUGE_PAGES":            "1",
    "EF_TCP_SEND_SPIN":         "1",
    "EF_INT_DRIVEN":            "1",
    "EF_RECV_SPIN":             "1",
    "EF_DELACK_THRESH":         "1",
    "EF_TX_PUSH":               "1",
    "EF_TCP_CONNECT_SPIN":      "1",
    "EF_SOCKET_RECV_BUFFER":    "16777216",
    "EF_SOCKET_SEND_BUFFER":    "16777216",
    "EF_CLUSTER_SIZE":          "1",
    # Log level 0 = silent in production
    "EF_LOG_VIA_IOCTL":         "1",
    # UDP acceleration
    "EF_UDP_RECV_SPIN":         "1",
    "EF_UDP_SEND_SPIN":         "1",
    # Epoll spin — critical for asyncio event loop
    "EF_EPOLL_SPIN":            "1",
    "EF_EPOLL_CTL_FAST":        "1",
}


class OnloadLauncher:
    """Launch or re-exec processes under OpenOnload kernel bypass."""

    def __init__(self):
        self._onload_path: Optional[str] = self._find_onload()
        self.available = self._onload_path is not None
        if self.available:
            logger.info("OnloadLauncher: found onload at %s", self._onload_path)
        else:
            logger.debug("OnloadLauncher: onload binary not found — kernel bypass unavailable")

    @staticmethod
    def _find_onload() -> Optional[str]:
        for candidate in _ONLOAD_BINARY_CANDIDATES:
            try:
                result = subprocess.run(
                    ["which", candidate] if not candidate.startswith("/") else ["test", "-x", candidate],
                    capture_output=True, timeout=2,
                )
                if result.returncode == 0:
                    path = result.stdout.decode().strip() if not candidate.startswith("/") else candidate
                    if path and Path(path).exists():
                        return path
            except Exception:
                pass
        # Direct existence check
        for candidate in _ONLOAD_BINARY_CANDIDATES:
            if Path(candidate).exists():
                return candidate
        return None

    def build_env(self) -> dict:
        """Return merged environment dict with EF_* vars applied."""
        env = dict(os.environ)
        for k, v in _LAUNCH_ENV.items():
            env.setdefault(k, v)
        return env

    def build_command(self, cmd: List[str]) -> List[str]:
        """Prepend onload binary to cmd if available."""
        if self.available and self._onload_path:
            return [self._onload_path] + cmd
        return cmd

    def launch(
        self,
        cmd: List[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        **kwargs,
    ) -> subprocess.Popen:
        """
        Launch cmd under onload with optimal EF_* environment.
        Returns the Popen object. Caller must call .wait() or .communicate().
        """
        full_cmd = self.build_command(cmd)
        env = self.build_env()
        logger.info(
            "OnloadLauncher: launching %s (onload=%s)",
            " ".join(cmd[:3]), self.available,
        )
        return subprocess.Popen(
            full_cmd,
            env=env,
            **kwargs,
        )

    def reexec_self(self) -> None:
        """
        Replace the current process with itself wrapped in onload.

        Call this early in main() — it will os.execve() into:
            onload <EF_*env> python <original argv>

        If already running under onload (detected via /proc/self/environ),
        this is a no-op.
        """
        if not self.available or self._onload_path is None:
            logger.debug("OnloadLauncher.reexec_self: onload not available, skipping")
            return
        # Detect if already running under onload
        try:
            environ_raw = Path("/proc/self/environ").read_bytes()
            if b"EF_ONLOAD" in environ_raw or b"LD_PRELOAD" in environ_raw:
                logger.debug("OnloadLauncher.reexec_self: already under onload")
                return
        except Exception:
            pass
        env = self.build_env()
        cmd = [self._onload_path] + sys.argv
        logger.info("OnloadLauncher: re-exec with onload: %s", " ".join(cmd[:4]))
        try:
            os.execve(self._onload_path, cmd, env)
        except Exception as e:
            logger.warning("OnloadLauncher.reexec_self failed: %s", e)

    def print_status(self) -> str:
        lines = [
            f"OnloadLauncher status:",
            f"  available    : {self.available}",
            f"  onload_path  : {self._onload_path or 'not found'}",
            f"  EF_* vars    : {len(_LAUNCH_ENV)}",
        ]
        return "\n".join(lines)
