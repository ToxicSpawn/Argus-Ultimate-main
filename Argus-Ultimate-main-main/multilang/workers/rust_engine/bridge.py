"""
Rust computation engine bridge for ARGUS.

Tries the compiled Rust binary first (JSON stdin/stdout protocol).
Falls back to pure-numpy implementations when the binary is not available.

Build the Rust binary:
    cd multilang/workers/rust_engine && cargo build --release

Usage:
    engine = RustEngine()
    result = engine.compute("correlation_matrix", {"series": [[1,2,3], [4,5,6]]})
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_BINARY_NAME = "rust_engine.exe" if platform.system() == "Windows" else "rust_engine"
_BINARY_PATH = _THIS_DIR / "target" / "release" / _BINARY_NAME


class RustEngine:
    """High-performance math via Rust subprocess, with numpy fallback."""

    def __init__(self, binary_path: Optional[str] = None) -> None:
        self._binary = Path(binary_path) if binary_path else _BINARY_PATH
        self._native_available = self._binary.is_file()
        self._backend = "native" if self._native_available else "fallback"
        self._call_count = 0
        self._total_latency = 0.0
        if self._native_available:
            logger.info("RustEngine: native binary found at %s", self._binary)
        else:
            logger.info("RustEngine: binary not found, using numpy fallback")

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

    def compute(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a command. Returns the result dict."""
        t0 = time.monotonic()
        try:
            if self._native_available:
                return self._call_native(command, data)
            return self._call_fallback(command, data)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    # ── Native (subprocess) ───────────────────────────────────────────────────

    def _call_native(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps({"command": command, "data": data})
        try:
            proc = subprocess.run(
                [str(self._binary)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                logger.warning("Rust engine stderr: %s", proc.stderr)
                return self._call_fallback(command, data)
            resp = json.loads(proc.stdout)
            if not resp.get("ok"):
                logger.warning("Rust engine error: %s", resp.get("error"))
                return self._call_fallback(command, data)
            return resp["result"]
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Rust engine call failed (%s), falling back to numpy", exc)
            return self._call_fallback(command, data)

    # ── Fallback (numpy) ──────────────────────────────────────────────────────

    def _call_fallback(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        dispatch = {
            "correlation_matrix": self._fb_correlation_matrix,
            "portfolio_var": self._fb_portfolio_var,
            "kelly_fraction": self._fb_kelly_fraction,
            "signal_zscore": self._fb_signal_zscore,
        }
        handler = dispatch.get(command)
        if handler is None:
            raise ValueError(f"Unknown command: {command}")
        return handler(data)

    @staticmethod
    def _fb_correlation_matrix(data: Dict[str, Any]) -> Dict[str, Any]:
        series = data.get("series", [])
        if not series:
            return []
        arr = np.array(series, dtype=np.float64)
        # np.corrcoef: each row is a variable, each column an observation
        corr = np.corrcoef(arr)
        # Handle single-series edge case
        if corr.ndim == 0:
            return [[1.0]]
        return corr.tolist()

    @staticmethod
    def _fb_portfolio_var(data: Dict[str, Any]) -> Dict[str, Any]:
        returns_flat = np.array(data.get("returns", []), dtype=np.float64)
        weights = np.array(data.get("weights", []), dtype=np.float64)
        confidence = float(data.get("confidence", 0.95))

        n_assets = len(weights)
        if n_assets == 0:
            return {"var": 0.0, "confidence": confidence, "n_periods": 0}
        n_periods = len(returns_flat) // n_assets
        returns_2d = returns_flat.reshape(n_periods, n_assets)
        portfolio_returns = returns_2d @ weights
        var_value = float(-np.percentile(portfolio_returns, (1.0 - confidence) * 100))
        return {"var": var_value, "confidence": confidence, "n_periods": n_periods}

    @staticmethod
    def _fb_kelly_fraction(data: Dict[str, Any]) -> Dict[str, Any]:
        win_rate = float(data.get("win_rate", 0.0))
        avg_win = float(data.get("avg_win", 0.0))
        avg_loss = float(data.get("avg_loss", 0.0))

        if abs(avg_loss) < 1e-15:
            return {"kelly": 0.0, "kelly_raw": 0.0, "win_rate": win_rate, "payoff_ratio": 0.0}
        b = avg_win / abs(avg_loss)
        kelly_raw = win_rate - (1.0 - win_rate) / b
        kelly = max(0.0, min(1.0, kelly_raw))
        return {"kelly": kelly, "kelly_raw": kelly_raw, "win_rate": win_rate, "payoff_ratio": b}

    @staticmethod
    def _fb_signal_zscore(data: Dict[str, Any]) -> Dict[str, Any]:
        values = np.array(data.get("values", []), dtype=np.float64)
        if len(values) == 0:
            return {"zscores": [], "mean": 0.0, "std": 0.0}
        m = float(np.mean(values))
        s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if s < 1e-15:
            zscores = [0.0] * len(values)
        else:
            zscores = ((values - m) / s).tolist()
        return {"zscores": zscores, "mean": m, "std": s}
