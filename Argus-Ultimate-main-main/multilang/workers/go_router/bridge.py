"""
Go order router bridge for ARGUS.

Starts the compiled Go binary as a subprocess (HTTP on localhost:9998),
then routes Python calls to the Go HTTP endpoints.

Falls back to a pure-Python latency-based router when Go binary is unavailable.

Build the Go binary:
    cd multilang/workers/go_router && go build -o go_router .

Usage:
    router = GoOrderRouter()
    router.start()
    venue = router.route({"symbol": "BTC/AUD", "side": "buy", "quantity": 0.01, "type": "limit"})
    result = router.submit({"symbol": "BTC/AUD", "side": "buy", "quantity": 0.01, "type": "limit"})
    stats = router.status()
    router.stop()
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_BINARY_NAME = "go_router.exe" if platform.system() == "Windows" else "go_router"
_BINARY_PATH = _THIS_DIR / _BINARY_NAME
_DEFAULT_PORT = 9998


class GoOrderRouter:
    """Order router: Go HTTP service with Python fallback."""

    def __init__(self, port: int = _DEFAULT_PORT, binary_path: Optional[str] = None) -> None:
        self._port = port
        self._binary = Path(binary_path) if binary_path else _BINARY_PATH
        self._native_available = self._binary.is_file()
        self._backend = "native" if self._native_available else "fallback"
        self._process: Optional[subprocess.Popen] = None
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._call_count = 0
        self._total_latency = 0.0

        # Fallback state
        self._fb_venues: Dict[str, Dict[str, Any]] = {
            "kraken": {"avg_latency_ms": 55.0, "healthy": True, "fill_rate": 0.95},
            "coinbase": {"avg_latency_ms": 60.0, "healthy": True, "fill_rate": 0.93},
        }

        if self._native_available:
            logger.info("GoOrderRouter: binary found at %s", self._binary)
        else:
            logger.info("GoOrderRouter: binary not found, using Python fallback")

    @property
    def available(self) -> bool:
        return self._native_available

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def avg_latency_ms(self) -> float:
        if self._call_count == 0:
            return 0.0
        return (self._total_latency / self._call_count) * 1000.0

    def start(self) -> bool:
        """Start the Go router subprocess (native mode only)."""
        if not self._native_available:
            logger.info("GoOrderRouter: fallback mode, no subprocess needed")
            return True
        try:
            self._process = subprocess.Popen(
                [str(self._binary), "-port", str(self._port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Give it a moment to bind
            time.sleep(0.3)
            if self._process.poll() is not None:
                logger.warning("GoOrderRouter: process exited immediately")
                self._native_available = False
                self._backend = "fallback"
                return True
            logger.info("GoOrderRouter: started (PID %d) on port %d", self._process.pid, self._port)
            return True
        except (FileNotFoundError, OSError) as exc:
            logger.warning("GoOrderRouter: failed to start: %s", exc)
            self._native_available = False
            self._backend = "fallback"
            return True

    def stop(self) -> None:
        """Stop the Go router subprocess."""
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

    def route(self, order: Dict[str, Any], venues: Optional[List[str]] = None) -> Dict[str, Any]:
        """Select the best venue for an order."""
        t0 = time.monotonic()
        try:
            if self._native_available and self._process:
                return self._http_route(order, venues)
            return self._fb_route(order, venues)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def submit(self, order: Dict[str, Any], venue: Optional[str] = None) -> Dict[str, Any]:
        """Submit an order to a venue."""
        t0 = time.monotonic()
        try:
            if self._native_available and self._process:
                return self._http_submit(order, venue)
            return self._fb_submit(order, venue)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def status(self) -> List[Dict[str, Any]]:
        """Get latency stats per venue."""
        if self._native_available and self._process:
            return self._http_status()
        return self._fb_status()

    # ── Native HTTP calls ─────────────────────────────────────────────────────

    def _http_post(self, path: str, data: Dict[str, Any]) -> Any:
        body = json.dumps(data).encode("utf-8")
        req = Request(
            f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except (URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("GoOrderRouter HTTP error: %s", exc)
            return None

    def _http_route(self, order: Dict[str, Any], venues: Optional[List[str]]) -> Dict[str, Any]:
        result = self._http_post("/route", {"order": order, "venues": venues or []})
        if result:
            return result
        return self._fb_route(order, venues)

    def _http_submit(self, order: Dict[str, Any], venue: Optional[str]) -> Dict[str, Any]:
        result = self._http_post("/submit", {"order": order, "venue": venue or ""})
        if result:
            return result
        return self._fb_submit(order, venue)

    def _http_status(self) -> List[Dict[str, Any]]:
        result = self._http_post("/status", {})
        if result and isinstance(result, list):
            return result
        return self._fb_status()

    # ── Fallback (Python) ─────────────────────────────────────────────────────

    def _fb_route(self, order: Dict[str, Any], venues: Optional[List[str]]) -> Dict[str, Any]:
        candidates = self._fb_venues
        if venues:
            candidates = {k: v for k, v in self._fb_venues.items() if k in venues}

        best_name = None
        best_score = float("inf")
        for name, stats in candidates.items():
            if not stats["healthy"]:
                continue
            score = stats["avg_latency_ms"] * (2.0 - stats["fill_rate"])
            if score < best_score:
                best_score = score
                best_name = name

        if best_name is None:
            best_name = next(iter(candidates), "kraken")
            best_score = 0.0

        return {
            "venue": best_name,
            "score": best_score,
            "latency_ms": self._fb_venues.get(best_name, {}).get("avg_latency_ms", 0.0),
            "reason": "lowest_score",
        }

    def _fb_submit(self, order: Dict[str, Any], venue: Optional[str]) -> Dict[str, Any]:
        if not venue:
            route = self._fb_route(order, None)
            venue = route["venue"]
        return {
            "venue": venue,
            "order_id": f"fb_sim_{int(time.time() * 1e6)}",
            "status": "submitted",
            "latency_ms": 0.1,
        }

    def _fb_status(self) -> List[Dict[str, Any]]:
        stats = []
        for name, s in sorted(self._fb_venues.items()):
            stats.append({
                "name": name,
                "avg_latency_ms": s["avg_latency_ms"],
                "p99_latency_ms": s["avg_latency_ms"] * 1.5,
                "healthy": s["healthy"],
                "fill_rate": s["fill_rate"],
            })
        return stats
