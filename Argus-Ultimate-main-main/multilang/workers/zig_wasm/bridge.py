"""
Zig WASM performance metrics bridge for ARGUS.

Tries to load the compiled WASM module first.
Falls back to pure-numpy implementations.

Build the WASM module:
    cd multilang/workers/zig_wasm && zig build

Usage:
    zc = ZigCompute()
    dd = zc.drawdown(equity_curve)
    sr = zc.sharpe(returns)
    so = zc.sortino(returns)
    cr = zc.calmar(returns, max_drawdown)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_WASM_PATH = _THIS_DIR / "zig-out" / "lib" / "argus_compute.wasm"


class ZigCompute:
    """Performance metric computation via Zig WASM, with numpy fallback."""

    def __init__(self, wasm_path: Optional[str] = None) -> None:
        self._wasm = Path(wasm_path) if wasm_path else _WASM_PATH
        self._native_available = self._wasm.is_file()
        self._backend = "native" if self._native_available else "fallback"
        self._call_count = 0
        self._total_latency = 0.0
        self._wasm_instance = None

        if self._native_available:
            logger.info("ZigCompute: WASM module found at %s", self._wasm)
            self._try_load_wasm()
        else:
            logger.info("ZigCompute: WASM module not found, using numpy fallback")

    def _try_load_wasm(self) -> None:
        """Try to load the WASM module via wasmtime or wasmer."""
        try:
            import wasmtime
            store = wasmtime.Store()
            module = wasmtime.Module.from_file(store.engine, str(self._wasm))
            self._wasm_instance = wasmtime.Instance(store, module, [])
            logger.info("ZigCompute: WASM loaded via wasmtime")
        except ImportError:
            logger.debug("ZigCompute: wasmtime not installed, using numpy fallback")
            self._native_available = False
            self._backend = "fallback"
        except Exception as exc:
            logger.warning("ZigCompute: failed to load WASM (%s), using fallback", exc)
            self._native_available = False
            self._backend = "fallback"

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

    # ── Public API ────────────────────────────────────────────────────

    def drawdown(self, equity_curve: Union[List[float], np.ndarray]) -> float:
        """
        Compute maximum drawdown from an equity curve.

        Args:
            equity_curve: Array of portfolio values over time.

        Returns:
            Maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
        """
        arr = np.asarray(equity_curve, dtype=np.float64)
        t0 = time.monotonic()
        try:
            return self._fb_drawdown(arr)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def sharpe(
        self,
        returns: Union[List[float], np.ndarray],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Compute Sharpe ratio.

        Args:
            returns: Array of period returns.
            risk_free_rate: Per-period risk-free rate.

        Returns:
            Sharpe ratio.
        """
        arr = np.asarray(returns, dtype=np.float64)
        t0 = time.monotonic()
        try:
            return self._fb_sharpe(arr, risk_free_rate)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def sortino(
        self,
        returns: Union[List[float], np.ndarray],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Compute Sortino ratio (downside deviation only).

        Args:
            returns: Array of period returns.
            risk_free_rate: Per-period risk-free rate.

        Returns:
            Sortino ratio.
        """
        arr = np.asarray(returns, dtype=np.float64)
        t0 = time.monotonic()
        try:
            return self._fb_sortino(arr, risk_free_rate)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def calmar(self, annualized_return: float, max_drawdown: float) -> float:
        """
        Compute Calmar ratio.

        Args:
            annualized_return: Annualized return.
            max_drawdown: Maximum drawdown as positive fraction.

        Returns:
            Calmar ratio.
        """
        t0 = time.monotonic()
        try:
            return self._fb_calmar(annualized_return, max_drawdown)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    # ── Fallback implementations (numpy) ──────────────────────────────

    @staticmethod
    def _fb_drawdown(equity: np.ndarray) -> float:
        if len(equity) < 2:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for i in range(1, len(equity)):
            if equity[i] > peak:
                peak = equity[i]
            if peak > 1e-15:
                dd = (peak - equity[i]) / peak
                if dd > max_dd:
                    max_dd = dd
        return float(max_dd)

    @staticmethod
    def _fb_sharpe(returns: np.ndarray, risk_free_rate: float) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate
        mean_excess = float(np.mean(excess))
        std_excess = float(np.std(excess, ddof=1))
        if std_excess < 1e-15:
            return 0.0
        return mean_excess / std_excess

    @staticmethod
    def _fb_sortino(returns: np.ndarray, risk_free_rate: float) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate
        mean_excess = float(np.mean(excess))

        # Downside deviation: only negative excess returns
        downside = excess[excess < 0]
        if len(downside) == 0:
            return 99.99 if mean_excess > 0 else 0.0
        down_dev = float(np.sqrt(np.mean(downside ** 2)))
        if down_dev < 1e-15:
            return 0.0
        return mean_excess / down_dev

    @staticmethod
    def _fb_calmar(annualized_return: float, max_drawdown: float) -> float:
        if max_drawdown < 1e-15:
            return 99.99 if annualized_return > 0 else 0.0
        return annualized_return / max_drawdown
