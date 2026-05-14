"""
Julia optimization solver bridge for ARGUS.

Tries the Julia runtime first (JSON stdin/stdout protocol).
Falls back to scipy.optimize / numpy implementations.

Requirements for native mode:
    Julia installed and on PATH, with solver.jl in this directory.

Usage:
    solver = JuliaSolver()
    result = solver.optimize_portfolio(returns_matrix, risk_aversion=2.0)
    rp = solver.risk_parity(covariance_matrix)
    kelly = solver.kelly_optimal(win_rates, payoff_ratios, corr)
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_SOLVER_SCRIPT = _THIS_DIR / "solver.jl"


class JuliaSolver:
    """Portfolio optimization via Julia, with numpy/scipy fallback."""

    def __init__(self) -> None:
        self._julia_path = shutil.which("julia")
        self._script = _SOLVER_SCRIPT
        self._native_available = (
            self._julia_path is not None and self._script.is_file()
        )
        self._backend = "native" if self._native_available else "fallback"
        self._call_count = 0
        self._total_latency = 0.0
        if self._native_available:
            logger.info("JuliaSolver: Julia found at %s", self._julia_path)
        else:
            logger.info("JuliaSolver: Julia not available, using numpy/scipy fallback")

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
                [self._julia_path, str(self._script)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode != 0:
                logger.warning("Julia solver stderr: %s", proc.stderr)
                return self._call_fallback(command, data)
            resp = json.loads(proc.stdout)
            if not resp.get("ok"):
                logger.warning("Julia solver error: %s", resp.get("error"))
                return self._call_fallback(command, data)
            return resp["result"]
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Julia solver call failed (%s), falling back", exc)
            return self._call_fallback(command, data)

    def _call_fallback(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        dispatch = {
            "optimize_portfolio": self._fb_optimize_portfolio,
            "risk_parity": self._fb_risk_parity,
            "kelly_optimal": self._fb_kelly_optimal,
        }
        handler = dispatch.get(command)
        if handler is None:
            raise ValueError(f"Unknown Julia command: {command}")
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

    def optimize_portfolio(
        self,
        returns_matrix: Union[List[List[float]], np.ndarray],
        risk_aversion: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Mean-variance portfolio optimization.

        Args:
            returns_matrix: 2D array (n_periods x n_assets).
            risk_aversion: Risk aversion parameter (higher = more conservative).

        Returns:
            {"weights": [float, ...], "expected_return": float, "expected_risk": float}
        """
        arr = np.asarray(returns_matrix, dtype=np.float64)
        data = {"returns": arr.tolist(), "risk_aversion": risk_aversion}
        return self._timed("optimize_portfolio", data)

    def risk_parity(
        self,
        covariance_matrix: Union[List[List[float]], np.ndarray],
    ) -> Dict[str, Any]:
        """
        Risk parity allocation: each asset contributes equally to risk.

        Args:
            covariance_matrix: 2D array (n_assets x n_assets).

        Returns:
            {"weights": [float, ...], "risk_contributions": [float, ...]}
        """
        arr = np.asarray(covariance_matrix, dtype=np.float64)
        data = {"covariance": arr.tolist()}
        return self._timed("risk_parity", data)

    def kelly_optimal(
        self,
        win_rates: Union[List[float], np.ndarray],
        payoff_ratios: Union[List[float], np.ndarray],
        correlation_matrix: Optional[Union[List[List[float]], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-asset Kelly criterion with correlation adjustment.

        Args:
            win_rates: Win probability per asset.
            payoff_ratios: Average win / average loss per asset.
            correlation_matrix: Optional inter-asset correlation.

        Returns:
            {"fractions": [float, ...], "total_allocation": float}
        """
        wr = np.asarray(win_rates, dtype=np.float64)
        pr = np.asarray(payoff_ratios, dtype=np.float64)
        n = len(wr)
        if correlation_matrix is None:
            corr = np.eye(n)
        else:
            corr = np.asarray(correlation_matrix, dtype=np.float64)
        data = {
            "win_rates": wr.tolist(),
            "payoff_ratios": pr.tolist(),
            "correlation": corr.tolist(),
        }
        return self._timed("kelly_optimal", data)

    # ── Fallback implementations (numpy/scipy) ────────────────────────

    @staticmethod
    def _fb_optimize_portfolio(data: Dict[str, Any]) -> Dict[str, Any]:
        returns = np.array(data["returns"], dtype=np.float64)
        risk_aversion = float(data.get("risk_aversion", 2.0))

        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        n_assets = returns.shape[1]
        mu = np.mean(returns, axis=0)
        cov = np.cov(returns, rowvar=False) if returns.shape[0] > 1 else np.eye(n_assets) * 0.01

        # Ensure 2D covariance
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Gradient descent: max(w'μ - λ/2 * w'Σw), s.t. sum(w)=1, w>=0
        w = np.ones(n_assets) / n_assets
        lr = 0.01

        for _ in range(1000):
            grad = mu - risk_aversion * cov @ w
            w += lr * grad
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s > 1e-15:
                w /= s
            else:
                w = np.ones(n_assets) / n_assets

        expected_return = float(mu @ w)
        expected_risk = float(np.sqrt(w @ cov @ w))

        return {
            "weights": w.tolist(),
            "expected_return": expected_return,
            "expected_risk": expected_risk,
        }

    @staticmethod
    def _fb_risk_parity(data: Dict[str, Any]) -> Dict[str, Any]:
        cov = np.array(data["covariance"], dtype=np.float64)
        if cov.ndim == 0:
            return {"weights": [1.0], "risk_contributions": [1.0]}

        n = cov.shape[0]
        w = np.ones(n) / n

        for _ in range(500):
            sigma_w = cov @ w
            port_vol = np.sqrt(w @ sigma_w)
            if port_vol < 1e-15:
                break
            mrc = sigma_w / port_vol
            target_rc = port_vol / n
            for i in range(n):
                if mrc[i] > 1e-15:
                    w[i] = target_rc / mrc[i]
            s = w.sum()
            if s > 1e-15:
                w /= s

        # Compute actual risk contributions
        sigma_w = cov @ w
        port_vol = np.sqrt(w @ sigma_w)
        if port_vol > 1e-15:
            rc = (w * sigma_w) / port_vol
        else:
            rc = np.zeros(n)

        return {
            "weights": w.tolist(),
            "risk_contributions": rc.tolist(),
        }

    @staticmethod
    def _fb_kelly_optimal(data: Dict[str, Any]) -> Dict[str, Any]:
        win_rates = np.array(data["win_rates"], dtype=np.float64)
        payoff_ratios = np.array(data["payoff_ratios"], dtype=np.float64)
        corr = np.array(data["correlation"], dtype=np.float64)

        n = len(win_rates)
        fractions = np.zeros(n)

        for i in range(n):
            if payoff_ratios[i] > 1e-15:
                kelly_raw = win_rates[i] - (1.0 - win_rates[i]) / payoff_ratios[i]
                fractions[i] = np.clip(kelly_raw, 0.0, 1.0)

        # Correlation adjustment
        for i in range(n):
            penalty = 0.0
            for j in range(n):
                if i != j:
                    penalty += abs(corr[i, j]) * fractions[j]
            if n > 1:
                avg_penalty = penalty / (n - 1)
                fractions[i] *= max(0.0, 1.0 - avg_penalty * 0.5)

        total = float(fractions.sum())
        if total > 1.0:
            fractions /= total
            total = 1.0

        return {
            "fractions": fractions.tolist(),
            "total_allocation": total,
        }
