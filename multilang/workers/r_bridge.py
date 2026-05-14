"""Optional R statistics bridge for Argus polyglot compute.

The R worker is useful for statistics/econometrics tasks such as volatility,
correlation, VaR/CVaR, skew, and confidence calibration. This bridge keeps R
fully optional: if ``Rscript`` or required R packages are unavailable, Python
fallbacks preserve the same response shape.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class RStatsBridge:
    """Call ``multilang/workers/r_worker.R`` with fallback statistics."""

    def __init__(self, worker_path: str | Path | None = None, timeout_s: float = 2.0) -> None:
        root = Path(__file__).resolve().parents[2]
        self.worker_path = Path(worker_path) if worker_path else root / "multilang" / "workers" / "r_worker.R"
        self.timeout_s = timeout_s
        self.available = shutil.which("Rscript") is not None and self.worker_path.exists()
        self.backend = "rscript" if self.available else "fallback"
        self.avg_latency_ms = 0.0
        self._calls = 0

    def compute(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            if self.available:
                result = self._call_r(task_type, data)
            else:
                result = self._fallback(task_type, data)
        finally:
            elapsed = (time.monotonic() - start) * 1000.0
            self._calls += 1
            self.avg_latency_ms += (elapsed - self.avg_latency_ms) / self._calls
        return result

    def volatility_estimate(self, prices: List[float] | None = None, returns: List[float] | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if prices is not None:
            payload["prices"] = prices
        if returns is not None:
            payload["returns"] = returns
        return self.compute("volatility_estimate", payload)

    def correlation_estimate(self, series_a: List[float], series_b: List[float]) -> Dict[str, Any]:
        return self.compute("correlation_estimate", {"series_a": series_a, "series_b": series_b})

    def var_estimate(self, returns: List[float], confidence_level: float = 0.95) -> Dict[str, Any]:
        return self.compute("var_estimate", {"returns": returns, "confidence_level": confidence_level})

    def skew_estimate(self, returns: List[float]) -> Dict[str, Any]:
        return self.compute("skew_estimate", {"returns": returns})

    def confidence_calibration(self, confidences: List[float], win_rate: float = 0.5) -> Dict[str, Any]:
        return self.compute("confidence_calibration", {"confidences": confidences, "win_rate": win_rate})

    def _call_r(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        request = json.dumps({"task_type": task_type, "data": data}) + "\n"
        proc = subprocess.run(
            ["Rscript", str(self.worker_path)],
            input=request,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            self.available = False
            self.backend = "fallback"
            return self._fallback(task_type, data)
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        if not parsed.get("ok", False):
            return self._fallback(task_type, data)
        return parsed.get("result", {})

    def _fallback(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if task_type == "volatility_estimate":
            returns = self._returns(data)
            vol = float(np.std(returns, ddof=1) * math.sqrt(252) * 10000) if len(returns) > 1 else 0.0
            return {"volatility_annual_bps": round(vol, 4), "volatility_weight": 1.2, "language": "r", "ok": True}
        if task_type == "correlation_estimate":
            a = np.asarray(data.get("series_a", []), dtype=float)
            b = np.asarray(data.get("series_b", []), dtype=float)
            n = min(len(a), len(b))
            corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 2 else 0.0
            return {"correlation": round(0.0 if math.isnan(corr) else corr, 6), "language": "r", "ok": True}
        if task_type == "var_estimate":
            returns = np.sort(np.asarray(data.get("returns", []), dtype=float))
            if len(returns) == 0:
                return {"var_pct": 0.0, "cvar_pct": 0.0, "language": "r", "ok": True}
            confidence = float(data.get("confidence_level", 0.95) or 0.95)
            idx = max(0, min(len(returns) - 1, int(len(returns) * (1.0 - confidence))))
            tail = returns[: idx + 1]
            return {"var_pct": round(float(-returns[idx]), 6), "cvar_pct": round(float(-np.mean(tail)), 6), "language": "r", "ok": True}
        if task_type == "skew_estimate":
            values = np.asarray(data.get("returns", []), dtype=float)
            if len(values) < 3:
                return {"skew": 0.0, "language": "r", "ok": True}
            centered = values - np.mean(values)
            std = np.std(values)
            skew = float(np.mean(centered ** 3) / (std ** 3)) if std > 0 else 0.0
            return {"skew": round(skew, 6), "language": "r", "ok": True}
        if task_type == "confidence_calibration":
            confidences = np.asarray(data.get("confidences", []), dtype=float)
            avg_conf = float(np.mean(confidences)) if len(confidences) else 0.5
            win_rate = float(data.get("win_rate", 0.5) or 0.5)
            return {"calibrated_confidence": round(0.5 * avg_conf + 0.5 * win_rate, 6), "language": "r", "ok": True}
        return {"language": "r", "ok": True}

    @staticmethod
    def _returns(data: Dict[str, Any]) -> np.ndarray:
        if data.get("returns"):
            return np.asarray(data["returns"], dtype=float)
        prices = np.asarray(data.get("prices", []), dtype=float)
        if len(prices) < 2:
            return np.asarray([], dtype=float)
        return np.diff(np.log(np.maximum(prices, 1e-12)))
