"""
rust_lob_bridge.py — Zero-copy bridge to the Rust LOB hot-path worker.

The Rust worker reads JSON lines from stdin and writes JSON lines to stdout,
keeping state (the order books) across calls — ideal for a persistent subprocess.

Usage
-----
    bridge = RustLOBBridge()
    bridge.start()

    sigs = await bridge.update("BTC-USD", "bid", 45000.0, 1.5)
    # {'obi': 0.23, 'weighted_mid': 45000.12, ..., 'timestamp_ns': ...}

    snap = await bridge.snapshot("BTC-USD")
    result = await bridge.benchmark()
    bridge.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default binary path relative to project root.
_DEFAULT_BINARY = "multilang/workers/rust_engine/target/release/rust_engine"

# On Windows the binary has an .exe suffix.
if sys.platform == "win32":
    _DEFAULT_BINARY += ".exe"


class RustLOBBridge:
    """Zero-copy bridge to the Rust LOB worker via stdin/stdout pipes.

    Thread-safe: all async operations are serialised with an asyncio.Lock so
    that concurrent coroutines do not interleave JSON lines.

    Falls back gracefully when the binary is not available — all async methods
    return None and log a warning instead of raising.
    """

    def __init__(self, worker_binary: str = _DEFAULT_BINARY) -> None:
        self._binary = Path(worker_binary)
        self._proc: Optional[subprocess.Popen] = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._available: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Spawn the Rust subprocess.  Returns True if successful."""
        if not self._binary.exists():
            logger.warning(
                "RustLOBBridge: binary not found at %s — bridge disabled. "
                "Run `cargo build --release` in multilang/workers/rust_engine/",
                self._binary,
            )
            self._available = False
            return False

        try:
            self._proc = subprocess.Popen(
                [str(self._binary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,           # unbuffered — lowest latency
            )
            self._available = True
            logger.info("RustLOBBridge: started PID %d", self._proc.pid)
            return True
        except OSError as exc:
            logger.warning("RustLOBBridge: failed to start process: %s", exc)
            self._available = False
            return False

    def stop(self) -> None:
        """Terminate the Rust subprocess cleanly."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()       # signals EOF → Rust main loop exits
                self._proc.wait(timeout=2.0)
            except Exception:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except Exception:
                    self._proc.kill()
            finally:
                self._proc = None
                self._available = False
                logger.info("RustLOBBridge: stopped")

    def is_available(self) -> bool:
        """Return True if the binary exists and the subprocess is running."""
        if not self._available:
            return False
        if self._proc is None or self._proc.poll() is not None:
            self._available = False
            logger.warning("RustLOBBridge: process has exited unexpectedly")
            return False
        return True

    # ── Internal I/O ─────────────────────────────────────────────────────────

    def _send_recv(self, payload: dict) -> Optional[dict]:
        """Synchronous send/receive — must be called while holding the lock."""
        if not self.is_available():
            return None

        line = (json.dumps(payload) + "\n").encode()
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            response_line = self._proc.stdout.readline()
            if not response_line:
                logger.warning("RustLOBBridge: empty response (process may have exited)")
                self._available = False
                return None
            return json.loads(response_line.decode().strip())
        except (OSError, json.JSONDecodeError, BrokenPipeError) as exc:
            logger.warning("RustLOBBridge: I/O error: %s", exc)
            self._available = False
            return None

    # ── Async public API ─────────────────────────────────────────────────────

    async def update(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        count: int = 1,
    ) -> Optional[dict]:
        """Send a LOB level update, return alpha signals.

        Parameters
        ----------
        symbol : e.g. "BTC-USD"
        side   : "bid" or "ask"
        price  : price of the level
        size   : quantity at the level (0.0 → delete level)
        count  : number of orders at the level (default 1)

        Returns
        -------
        dict with keys: obi, weighted_mid, microprice, spread_bps, pressure,
                        timestamp_ns — or None if bridge unavailable.
        """
        if not self.is_available():
            return None

        payload = {
            "task_type": "lob_update",
            "data": {
                "symbol": symbol,
                "side": side,
                "price": price,
                "size": size,
                "count": count,
            },
        }
        async with self._lock:
            resp = await asyncio.get_running_loop().run_in_executor(
                None, self._send_recv, payload
            )
        if resp is None:
            return None
        if not resp.get("ok"):
            logger.warning("RustLOBBridge.update error: %s", resp.get("error"))
            return None
        return resp.get("result")

    async def snapshot(self, symbol: str) -> Optional[dict]:
        """Return the current full book state for *symbol*.

        Returns
        -------
        dict with keys: symbol, bids, asks, best_bid, best_ask, mid,
                        spread_bps, last_update_ns — or None if unavailable.
        """
        if not self.is_available():
            return None

        payload = {"task_type": "lob_snapshot", "data": {"symbol": symbol}}
        async with self._lock:
            resp = await asyncio.get_running_loop().run_in_executor(
                None, self._send_recv, payload
            )
        if resp is None:
            return None
        if not resp.get("ok"):
            logger.warning("RustLOBBridge.snapshot error: %s", resp.get("error"))
            return None
        return resp.get("result")

    async def compute_signals(self, symbol: str) -> Optional[dict]:
        """Return all alpha signals for *symbol* without mutating the book."""
        if not self.is_available():
            return None

        payload = {"task_type": "compute_signals", "data": {"symbol": symbol}}
        async with self._lock:
            resp = await asyncio.get_running_loop().run_in_executor(
                None, self._send_recv, payload
            )
        if resp is None:
            return None
        if not resp.get("ok"):
            logger.warning("RustLOBBridge.compute_signals error: %s", resp.get("error"))
            return None
        return resp.get("result")

    async def benchmark(self) -> Optional[dict]:
        """Run 1 M update+compute cycles inside the Rust process.

        Returns
        -------
        dict: iterations, elapsed_ns, ns_per_op, ops_per_sec, final_mid, final_spread_bps
        """
        if not self.is_available():
            return None

        payload = {"task_type": "benchmark", "data": {}}
        async with self._lock:
            resp = await asyncio.get_running_loop().run_in_executor(
                None, self._send_recv, payload
            )
        if resp is None:
            return None
        if not resp.get("ok"):
            logger.warning("RustLOBBridge.benchmark error: %s", resp.get("error"))
            return None
        return resp.get("result")

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    async def __aenter__(self):
        self.start()
        return self

    async def __aexit__(self, *_):
        self.stop()

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        return f"RustLOBBridge(binary={self._binary!r}, status={status!r})"
