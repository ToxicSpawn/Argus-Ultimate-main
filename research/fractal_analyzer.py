#!/usr/bin/env python3
"""
Fractal / Hurst Exponent Analysis — market memory and trend persistence.

Estimates the Hurst exponent via Rescaled Range (R/S) analysis to determine
whether a price series exhibits:

- **Persistent** behaviour (H > 0.5): trends tend to continue.
- **Anti-persistent** behaviour (H < 0.5): mean-reverting.
- **Random walk** (H ≈ 0.5): no exploitable memory.

Also computes the fractal dimension ``D = 2 - H`` as an alternative measure
of price-path roughness.

Usage::

    fa = FractalAnalyzer()
    result = fa.compute_hurst(prices, max_lag=100)
    print(result.hurst_exponent, result.interpretation)

Pure Python with numpy acceleration when available.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HurstResult:
    """Result of Hurst exponent estimation."""

    hurst_exponent: float
    interpretation: str  # "persistent" | "anti_persistent" | "random_walk"
    r_squared: float
    confidence: str  # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Fractal Analyzer
# ---------------------------------------------------------------------------


class FractalAnalyzer:
    """Computes fractal properties of price series.

    All methods are stateless — the class serves as a namespace with
    configurable defaults.

    Parameters
    ----------
    persistent_threshold:
        Hurst values above this are labelled ``"persistent"``.
    anti_persistent_threshold:
        Hurst values below this are labelled ``"anti_persistent"``.
    """

    def __init__(
        self,
        persistent_threshold: float = 0.55,
        anti_persistent_threshold: float = 0.45,
    ) -> None:
        self.persistent_threshold = persistent_threshold
        self.anti_persistent_threshold = anti_persistent_threshold
        logger.info(
            "FractalAnalyzer initialised — thresholds=(%.2f, %.2f)",
            anti_persistent_threshold,
            persistent_threshold,
        )

    # ------------------------------------------------------------------
    # Hurst exponent (R/S analysis)
    # ------------------------------------------------------------------

    def compute_hurst(
        self, prices: List[float], max_lag: int = 100
    ) -> HurstResult:
        """Estimate the Hurst exponent of *prices* using R/S analysis.

        Parameters
        ----------
        prices:
            Raw price series (at least 20 observations recommended).
        max_lag:
            Maximum sub-series length to consider.

        Returns
        -------
        HurstResult
        """
        if len(prices) < 20:
            logger.warning(
                "FractalAnalyzer.compute_hurst: only %d prices (min 20 recommended)",
                len(prices),
            )
            return HurstResult(
                hurst_exponent=0.5,
                interpretation="random_walk",
                r_squared=0.0,
                confidence="low",
            )

        # Compute log returns
        if _HAS_NUMPY:
            arr = np.array(prices, dtype=float)
            returns = np.diff(np.log(arr[arr > 0]))
            if len(returns) < 10:
                return HurstResult(0.5, "random_walk", 0.0, "low")
            log_ns, log_rs = self._rs_analysis_numpy(returns, max_lag)
        else:
            returns = []
            for i in range(1, len(prices)):
                if prices[i] > 0 and prices[i - 1] > 0:
                    returns.append(math.log(prices[i] / prices[i - 1]))
            if len(returns) < 10:
                return HurstResult(0.5, "random_walk", 0.0, "low")
            log_ns, log_rs = self._rs_analysis_pure(returns, max_lag)

        if len(log_ns) < 2:
            return HurstResult(0.5, "random_walk", 0.0, "low")

        # Linear regression: log(R/S) = H * log(n) + c
        hurst, r_squared = self._linear_regression(log_ns, log_rs)

        # Clamp to [0, 1]
        hurst = max(0.0, min(1.0, hurst))

        # Interpret
        if hurst > self.persistent_threshold:
            interpretation = "persistent"
        elif hurst < self.anti_persistent_threshold:
            interpretation = "anti_persistent"
        else:
            interpretation = "random_walk"

        # Confidence based on R^2 and sample size
        if r_squared > 0.9 and len(returns) > 100:
            confidence = "high"
        elif r_squared > 0.7 and len(returns) > 50:
            confidence = "medium"
        else:
            confidence = "low"

        logger.debug(
            "FractalAnalyzer.compute_hurst H=%.4f R2=%.4f interp=%s conf=%s",
            hurst,
            r_squared,
            interpretation,
            confidence,
        )
        return HurstResult(
            hurst_exponent=round(hurst, 6),
            interpretation=interpretation,
            r_squared=round(r_squared, 6),
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # R/S analysis internals
    # ------------------------------------------------------------------

    def _rs_analysis_numpy(
        self, returns: "np.ndarray", max_lag: int
    ) -> Tuple[List[float], List[float]]:
        """R/S analysis using numpy for speed."""
        T = len(returns)
        min_n = 10
        lags = self._generate_lags(min_n, min(max_lag, T // 2))

        log_ns: List[float] = []
        log_rs_vals: List[float] = []

        for n in lags:
            n_segments = T // n
            if n_segments < 1:
                continue

            rs_values = []
            for seg in range(n_segments):
                chunk = returns[seg * n : (seg + 1) * n]
                mean_ret = float(np.mean(chunk))
                deviations = chunk - mean_ret
                cumulative = np.cumsum(deviations)

                R = float(np.max(cumulative) - np.min(cumulative))
                S = float(np.std(chunk, ddof=1))

                if S > 1e-12:
                    rs_values.append(R / S)

            if rs_values:
                avg_rs = float(np.mean(rs_values))
                if avg_rs > 0:
                    log_ns.append(math.log(n))
                    log_rs_vals.append(math.log(avg_rs))

        return log_ns, log_rs_vals

    def _rs_analysis_pure(
        self, returns: List[float], max_lag: int
    ) -> Tuple[List[float], List[float]]:
        """R/S analysis in pure Python."""
        T = len(returns)
        min_n = 10
        lags = self._generate_lags(min_n, min(max_lag, T // 2))

        log_ns: List[float] = []
        log_rs_vals: List[float] = []

        for n in lags:
            n_segments = T // n
            if n_segments < 1:
                continue

            rs_values: List[float] = []
            for seg in range(n_segments):
                chunk = returns[seg * n : (seg + 1) * n]
                mean_ret = sum(chunk) / len(chunk)

                # Cumulative deviations
                cumulative: List[float] = []
                cum = 0.0
                for v in chunk:
                    cum += v - mean_ret
                    cumulative.append(cum)

                R = max(cumulative) - min(cumulative)

                # Standard deviation
                var = sum((v - mean_ret) ** 2 for v in chunk) / (len(chunk) - 1)
                S = math.sqrt(var) if var > 0 else 0.0

                if S > 1e-12:
                    rs_values.append(R / S)

            if rs_values:
                avg_rs = sum(rs_values) / len(rs_values)
                if avg_rs > 0:
                    log_ns.append(math.log(n))
                    log_rs_vals.append(math.log(avg_rs))

        return log_ns, log_rs_vals

    @staticmethod
    def _generate_lags(min_n: int, max_n: int) -> List[int]:
        """Generate a geometric sequence of lag values between *min_n* and *max_n*."""
        if max_n < min_n:
            return []
        lags: List[int] = []
        n = min_n
        while n <= max_n:
            if n not in lags:
                lags.append(n)
            n = max(n + 1, int(n * 1.3))
        return lags

    @staticmethod
    def _linear_regression(
        x: List[float], y: List[float]
    ) -> Tuple[float, float]:
        """Simple OLS regression returning (slope, r_squared)."""
        n = len(x)
        if n < 2:
            return 0.5, 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xx = sum(xi * xi for xi in x)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))

        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            return 0.5, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # R^2
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        intercept = (sum_y - slope * sum_x) / n
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return slope, max(0.0, r_squared)

    # ------------------------------------------------------------------
    # Market memory (convenience)
    # ------------------------------------------------------------------

    def get_market_memory(self, prices: List[float]) -> str:
        """Classify market memory as ``"trending"``, ``"mean_reverting"``, or ``"random"``.

        Parameters
        ----------
        prices:
            Raw price series.

        Returns
        -------
        str
        """
        result = self.compute_hurst(prices)
        if result.interpretation == "persistent":
            return "trending"
        elif result.interpretation == "anti_persistent":
            return "mean_reverting"
        return "random"

    # ------------------------------------------------------------------
    # Fractal dimension
    # ------------------------------------------------------------------

    def compute_fractal_dimension(self, prices: List[float]) -> float:
        """Estimate fractal dimension via ``D = 2 - H``.

        A Brownian motion has ``D = 1.5``.  Lower values indicate smoother
        (trending) paths; higher values indicate rougher (mean-reverting) paths.

        Returns
        -------
        float
            Fractal dimension in [1.0, 2.0].
        """
        result = self.compute_hurst(prices)
        d = 2.0 - result.hurst_exponent
        logger.debug(
            "FractalAnalyzer.compute_fractal_dimension D=%.4f (H=%.4f)",
            d,
            result.hurst_exponent,
        )
        return round(d, 6)
