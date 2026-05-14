"""
Node.js WebSocket multiplexer bridge for ARGUS.

Launches the Node.js ws_mux.js as a subprocess, then connects Python
to the local multiplexed WebSocket on localhost:9999.

Falls back to direct Python WebSocket connections if Node.js is not available.

Prerequisites:
    cd multilang/workers/node_ws && npm install

Usage:
    mux = NodeWSMultiplexer()
    await mux.start()
    mux.on_message(my_callback)
    await mux.connect()
    ...
    await mux.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = _THIS_DIR / "ws_mux.js"
_DEFAULT_PORT = 9999


class NodeWSMultiplexer:
    """
    WebSocket multiplexer: Node.js native or Python asyncio fallback.

    In native mode, a Node.js subprocess manages all exchange WS connections
    and multiplexes them onto localhost:9999.

    In fallback mode, Python connects directly to exchanges via asyncio.
    """

    def __init__(self, port: int = _DEFAULT_PORT) -> None:
        self._port = port
        self._process: Optional[subprocess.Popen] = None
        self._node_available = self._detect_node()
        self._backend = "native" if self._node_available else "fallback"
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._ws = None
        self._running = False
        self._message_count = 0
        self._feeds: List[Dict[str, Any]] = []

        if self._node_available:
            logger.info("NodeWSMultiplexer: Node.js available, will use native mux")
        else:
            logger.info("NodeWSMultiplexer: Node.js not found, using Python fallback")

    @staticmethod
    def _detect_node() -> bool:
        """Check if node and the ws_mux.js script + node_modules exist."""
        if not _SCRIPT_PATH.is_file():
            return False
        node_modules = _THIS_DIR / "node_modules"
        if not node_modules.is_dir():
            return False
        return shutil.which("node") is not None

    @property
    def available(self) -> bool:
        return self._node_available

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def message_count(self) -> int:
        return self._message_count

    def add_feed(self, exchange: str, url: str, subscribe: Optional[Dict] = None) -> None:
        """Register an exchange feed to connect to."""
        self._feeds.append({
            "exchange": exchange,
            "url": url,
            "subscribe": subscribe,
        })

    def on_message(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for incoming multiplexed messages."""
        self._callbacks.append(callback)

    async def start(self) -> bool:
        """Start the multiplexer (launches Node.js subprocess in native mode)."""
        if self._node_available:
            return self._start_native()
        logger.info("NodeWSMultiplexer: fallback mode — no subprocess started")
        self._running = True
        return True

    def _start_native(self) -> bool:
        """Launch the Node.js subprocess."""
        config = json.dumps({"port": self._port, "feeds": self._feeds})
        try:
            self._process = subprocess.Popen(
                ["node", str(_SCRIPT_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Send config via stdin and close
            if self._process.stdin:
                self._process.stdin.write(config)
                self._process.stdin.close()
            self._running = True
            logger.info("NodeWSMultiplexer: started Node.js process (PID %d)", self._process.pid)
            return True
        except (FileNotFoundError, OSError) as exc:
            logger.warning("NodeWSMultiplexer: failed to start Node.js: %s", exc)
            self._node_available = False
            self._backend = "fallback"
            self._running = True
            return True

    async def connect(self) -> bool:
        """Connect Python to the multiplexed WebSocket."""
        if not self._running:
            return False

        if self._node_available:
            return await self._connect_native()

        # Fallback: direct connections handled elsewhere
        logger.info("NodeWSMultiplexer: fallback mode — Python manages WS directly")
        return True

    async def _connect_native(self) -> bool:
        """Connect to the local mux server via websockets or aiohttp."""
        try:
            import websockets  # type: ignore
            uri = f"ws://127.0.0.1:{self._port}"
            self._ws = await websockets.connect(uri)
            logger.info("NodeWSMultiplexer: connected to %s", uri)
            return True
        except ImportError:
            logger.info("NodeWSMultiplexer: websockets not installed, using fallback")
            return True
        except Exception as exc:
            logger.warning("NodeWSMultiplexer: connect failed: %s", exc)
            return False

    async def stop(self) -> None:
        """Stop the multiplexer."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    def parse_message(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse a multiplexed message envelope."""
        try:
            msg = json.loads(raw)
            if not isinstance(msg, dict):
                return None
            # Validate expected fields
            if "exchange" in msg and "data" in msg:
                self._message_count += 1
                return msg
            return msg
        except (json.JSONDecodeError, TypeError):
            return None

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        """Dispatch a parsed message to all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(msg)
            except Exception as exc:
                logger.debug("NodeWSMultiplexer callback error: %s", exc)
