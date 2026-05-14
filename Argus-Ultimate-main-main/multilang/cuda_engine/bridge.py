"""
CUDA GPU computation engine bridge for ARGUS.

Tries the compiled CUDA binary first (JSON stdin/stdout protocol).
Falls back to pure-numpy implementations when the binary is not available.

Build the CUDA binary:
    cd multilang/cuda_engine && nvcc -O3 -o cuda_engine kernels.cu -lcurand

Usage:
    engine = CudaEngine()
    var_result = engine.monte_carlo_var(returns, n_scenarios=100000)
    emas = engine.batch_ema(price_arrays, periods)
    corr = engine.correlation_matrix(price_matrix)
    scores = engine.batch_signal_score(signals, features)
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_BINARY_NAME = "cuda_engine.exe" if platform.system() == "Windows" else "cuda_engine"
_BINARY_PATH = _THIS_DIR / _BINARY_NAME


class CudaEngine:
    """GPU-accelerated trading computations via CUDA, with numpy fallback."""

    def __init__(self, binary_path: Optional[str] = None) -> None:
        self._binary = Path(binary_path) if binary_path else _BINARY_PATH
        self._native_available = self._binary.is_file()
        self._backend = "native" if self._native_available else "fallback"
        self._call_count = 0
        self._total_latency = 0.0
        if self._native_available:
            logger.info("CudaEngine: native binary found at %s", self._binary)
        else:
            logger.info("CudaEngine: binary not found, using numpy fallback")

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

    # ── Native (subprocess) ───────────────────────────────────────────

    def _call_native(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps({"command": command, "data": data})
        try:
            proc = subprocess.run(
                [str(self._binary)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                logger.warning("CUDA engine stderr: %s", proc.stderr)
                return self._call_fallback(command, data)
            resp = json.loads(proc.stdout)
            if not resp.get("ok"):
                logger.warning("CUDA engine error: %s", resp.get("error"))
                return self._call_fallback(command, data)
            return resp["result"]
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("CUDA engine call failed (%s), falling back to numpy", exc)
            return self._call_fallback(command, data)

    def _call_fallback(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        dispatch = {
            "monte_carlo_var": self._fb_monte_carlo_var,
            "batch_ema": self._fb_batch_ema,
            "correlation_matrix": self._fb_correlation_matrix,
            "batch_signal_score": self._fb_batch_signal_score,
        }
        handler = dispatch.get(command)
        if handler is None:
            raise ValueError(f"Unknown CUDA command: {command}")
        return handler(data)

    def _timed(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.monotonic()
        try:
            if self._native_available:
                return self._call_native(command, data)
            return self._call_fallback(command, data)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    # ── Public API ────────────────────────────────────────────────────

    def monte_carlo_var(
        self,
        returns: Union[List[List[float]], np.ndarray],
        n_scenarios: int = 100_000,
        confidence: float = 0.95,
        weights: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Monte Carlo Value-at-Risk via GPU simulation.

        Args:
            returns: 2D array (n_periods x n_assets) of historical returns.
            n_scenarios: Number of random scenarios to generate.
            confidence: VaR confidence level (e.g. 0.95 or 0.99).
            weights: Portfolio weights. Defaults to equal weight.

        Returns:
            {"var": float, "cvar": float, "confidence": float, "n_scenarios": int}
        """
        arr = np.asarray(returns, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_assets = arr.shape[1]
        if weights is None:
            w = np.ones(n_assets) / n_assets
        else:
            w = np.asarray(weights, dtype=np.float64)
        data = {
            "returns": arr.tolist(),
            "weights": w.tolist(),
            "n_scenarios": n_scenarios,
            "confidence": confidence,
        }
        return self._timed("monte_carlo_var", data)

    def batch_ema(
        self,
        price_arrays: Union[List[List[float]], np.ndarray],
        periods: Union[List[int], int],
    ) -> Dict[str, Any]:
        """
        Compute EMA for multiple symbols in parallel on GPU.

        Args:
            price_arrays: 2D array (n_symbols x n_periods).
            periods: EMA period per symbol, or single int for all.

        Returns:
            {"emas": [[float, ...], ...]}
        """
        arr = np.asarray(price_arrays, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_symbols = arr.shape[0]
        if isinstance(periods, int):
            p_list = [periods] * n_symbols
        else:
            p_list = list(periods)
        data = {"prices": arr.tolist(), "periods": p_list}
        return self._timed("batch_ema", data)

    def correlation_matrix(
        self,
        price_matrix: Union[List[List[float]], np.ndarray],
    ) -> Dict[str, Any]:
        """
        GPU-accelerated correlation matrix.

        Args:
            price_matrix: 2D array (n_assets x n_periods).

        Returns:
            {"matrix": [[float, ...], ...]}
        """
        arr = np.asarray(price_matrix, dtype=np.float64)
        data = {"matrix": arr.tolist()}
        return self._timed("correlation_matrix", data)

    def batch_signal_score(
        self,
        signals: Union[List[List[float]], np.ndarray],
        features: Union[List[float], np.ndarray],
    ) -> Dict[str, Any]:
        """
        Batch score signals against a feature vector (dot product) on GPU.

        Args:
            signals: 2D array (n_signals x n_features).
            features: 1D feature vector.

        Returns:
            {"scores": [float, ...]}
        """
        sig = np.asarray(signals, dtype=np.float64)
        feat = np.asarray(features, dtype=np.float64)
        data = {"signals": sig.tolist(), "features": feat.tolist()}
        return self._timed("batch_signal_score", data)

    # ── Fallback implementations (numpy) ──────────────────────────────

    @staticmethod
    def _fb_monte_carlo_var(data: Dict[str, Any]) -> Dict[str, Any]:
        returns = np.array(data["returns"], dtype=np.float64)
        weights = np.array(data["weights"], dtype=np.float64)
        n_scenarios = int(data.get("n_scenarios", 100_000))
        confidence = float(data.get("confidence", 0.95))

        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        n_periods = returns.shape[0]
        if n_periods == 0:
            return {"var": 0.0, "cvar": 0.0, "confidence": confidence, "n_scenarios": 0}

        # Sample random periods with replacement
        rng = np.random.default_rng()
        indices = rng.integers(0, n_periods, size=n_scenarios)
        sampled = returns[indices]  # (n_scenarios, n_assets)
        portfolio_returns = sampled @ weights  # (n_scenarios,)

        # Sort for VaR/CVaR
        sorted_returns = np.sort(portfolio_returns)
        cutoff_idx = int((1.0 - confidence) * n_scenarios)
        cutoff_idx = max(1, cutoff_idx)
        var_value = float(-sorted_returns[cutoff_idx])
        cvar_value = float(-np.mean(sorted_returns[:cutoff_idx]))

        return {
            "var": var_value,
            "cvar": cvar_value,
            "confidence": confidence,
            "n_scenarios": n_scenarios,
        }

    @staticmethod
    def _fb_batch_ema(data: Dict[str, Any]) -> Dict[str, Any]:
        prices = np.array(data["prices"], dtype=np.float64)
        periods = data["periods"]

        if prices.ndim == 1:
            prices = prices.reshape(1, -1)

        n_symbols, n_periods = prices.shape
        emas = np.empty_like(prices)

        for s in range(n_symbols):
            period = periods[s] if s < len(periods) else periods[-1]
            alpha = 2.0 / (period + 1.0)
            emas[s, 0] = prices[s, 0]
            for i in range(1, n_periods):
                emas[s, i] = alpha * prices[s, i] + (1.0 - alpha) * emas[s, i - 1]

        return {"emas": emas.tolist()}

    @staticmethod
    def _fb_correlation_matrix(data: Dict[str, Any]) -> Dict[str, Any]:
        matrix = np.array(data["matrix"], dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] < 2:
            n = matrix.shape[0]
            return {"matrix": np.eye(n).tolist()}

        # np.corrcoef: each row is a variable
        corr = np.corrcoef(matrix)
        if corr.ndim == 0:
            return {"matrix": [[1.0]]}
        # Replace NaN with 0 for constant series
        corr = np.nan_to_num(corr, nan=0.0)
        return {"matrix": corr.tolist()}

    @staticmethod
    def _fb_batch_signal_score(data: Dict[str, Any]) -> Dict[str, Any]:
        signals = np.array(data["signals"], dtype=np.float64)
        features = np.array(data["features"], dtype=np.float64)

        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        scores = signals @ features
        return {"scores": scores.tolist()}
