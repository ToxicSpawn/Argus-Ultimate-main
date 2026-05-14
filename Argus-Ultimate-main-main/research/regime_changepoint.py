#!/usr/bin/env python3
"""
Bayesian Online Changepoint Detection (BOCPD).

Implements the Adams & MacKay (2007) algorithm for real-time detection of
distributional changes in streaming time-series data.  Useful for detecting
regime shifts in crypto price returns, volatility, or order-flow metrics.

Model assumptions:
- **Likelihood**: Gaussian (conjugate Normal-Inverse-Gamma prior).
- **Hazard function**: Geometric with constant rate ``1/hazard_lambda``.

Usage::

    detector = BOCPDetector(hazard_lambda=200)
    for ret in returns:
        result = detector.update(ret)
        if result.changepoint_detected:
            print(f"Changepoint at {result.timestamp}")

Pure Python with numpy acceleration when available.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

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
class ChangePointResult:
    """Result returned by :meth:`BOCPDetector.update`."""

    changepoint_detected: bool
    run_length: int
    probability: float
    growth_probability: float
    timestamp: Optional[float] = None


@dataclass
class _ChangePointRecord:
    """Internal record for detected changepoint history."""

    timestamp: float
    run_length: int
    probability: float


# ---------------------------------------------------------------------------
# Gaussian sufficient statistics (conjugate prior)
# ---------------------------------------------------------------------------


class _GaussianSuffStats:
    """Normal-Inverse-Gamma conjugate posterior for streaming Gaussian data.

    Tracks per-run-length sufficient statistics:
    ``mu0``, ``kappa``, ``alpha``, ``beta`` (NIG parameters).
    """

    def __init__(
        self,
        mu0: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        self.mu0 = mu0
        self.kappa0 = kappa
        self.alpha0 = alpha
        self.beta0 = beta

        # Per-run-length arrays (Python lists — fast enough for online use)
        self.muT: List[float] = [mu0]
        self.kappaT: List[float] = [kappa]
        self.alphaT: List[float] = [alpha]
        self.betaT: List[float] = [beta]

    def add_run_length(self) -> None:
        """Prepend a new run-length slot (for a potential changepoint)."""
        self.muT.insert(0, self.mu0)
        self.kappaT.insert(0, self.kappa0)
        self.alphaT.insert(0, self.alpha0)
        self.betaT.insert(0, self.beta0)

    def update(self, x: float) -> None:
        """Bayesian update of all run-length slots with new observation *x*."""
        n = len(self.muT)
        for i in range(n):
            old_mu = self.muT[i]
            old_kappa = self.kappaT[i]
            old_alpha = self.alphaT[i]
            old_beta = self.betaT[i]

            new_kappa = old_kappa + 1.0
            new_mu = (old_kappa * old_mu + x) / new_kappa
            new_alpha = old_alpha + 0.5
            new_beta = old_beta + (old_kappa * (x - old_mu) ** 2) / (
                2.0 * new_kappa
            )

            self.muT[i] = new_mu
            self.kappaT[i] = new_kappa
            self.alphaT[i] = new_alpha
            self.betaT[i] = new_beta

    def predictive_log_prob(self, x: float) -> List[float]:
        """Student-t predictive log-probability of *x* for each run length.

        Returns a list of log-probabilities (one per run-length slot).
        """
        n = len(self.muT)
        log_probs: List[float] = []
        for i in range(n):
            mu = self.muT[i]
            kappa = self.kappaT[i]
            alpha = self.alphaT[i]
            beta = self.betaT[i]

            # Student-t with 2*alpha degrees of freedom
            df = 2.0 * alpha
            scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
            scale = math.sqrt(scale_sq) if scale_sq > 0 else 1e-10

            # Student-t log PDF
            z = (x - mu) / scale
            log_p = (
                math.lgamma((df + 1.0) / 2.0)
                - math.lgamma(df / 2.0)
                - 0.5 * math.log(df * math.pi * scale_sq)
                - ((df + 1.0) / 2.0) * math.log(1.0 + z * z / df)
            )
            log_probs.append(log_p)

        return log_probs

    def truncate(self, max_len: int) -> None:
        """Keep only the first *max_len* run-length slots to bound memory."""
        if len(self.muT) > max_len:
            self.muT = self.muT[:max_len]
            self.kappaT = self.kappaT[:max_len]
            self.alphaT = self.alphaT[:max_len]
            self.betaT = self.betaT[:max_len]


# ---------------------------------------------------------------------------
# BOCPD Detector
# ---------------------------------------------------------------------------


class BOCPDetector:
    """Bayesian Online Changepoint Detector.

    Parameters
    ----------
    hazard_lambda:
        Expected run length between changepoints.  Higher values mean the
        detector expects fewer changepoints (more conservative).
    mu0:
        Prior mean of the Gaussian likelihood.
    kappa:
        Prior precision scaling for the mean.
    alpha:
        Prior shape for the inverse-gamma variance.
    beta:
        Prior rate for the inverse-gamma variance.
    threshold:
        Probability threshold above which a changepoint is declared.
    max_run_length:
        Maximum run-length to track (bounds memory).
    """

    def __init__(
        self,
        hazard_lambda: float = 200.0,
        mu0: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        threshold: float = 0.5,
        max_run_length: int = 500,
    ) -> None:
        self.hazard_lambda = hazard_lambda
        self.threshold = threshold
        self.max_run_length = max_run_length

        # Hazard probability (geometric prior)
        self._H = 1.0 / hazard_lambda

        # Run-length distribution (start with run_length=0 having prob 1)
        self._run_length_probs: List[float] = [1.0]

        # Sufficient statistics
        self._stats = _GaussianSuffStats(mu0, kappa, alpha, beta)

        # History
        self._changepoint_history: List[_ChangePointRecord] = []
        self._n_observations = 0

        logger.info(
            "BOCPDetector initialised — hazard_lambda=%.0f threshold=%.2f",
            hazard_lambda,
            threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, value: float, timestamp: Optional[float] = None
    ) -> ChangePointResult:
        """Process a new observation and return changepoint detection result.

        Parameters
        ----------
        value:
            The new scalar observation (e.g. a return, a volatility reading).
        timestamp:
            Optional Unix timestamp.  Defaults to ``time.time()``.

        Returns
        -------
        ChangePointResult
        """
        if timestamp is None:
            timestamp = time.time()

        self._n_observations += 1

        # 1. Predictive probabilities under each run length
        log_preds = self._stats.predictive_log_prob(value)

        # 2. Growth probabilities  (run length increases by 1)
        n = len(self._run_length_probs)
        growth_probs = [0.0] * n
        for i in range(n):
            growth_probs[i] = self._run_length_probs[i] * math.exp(log_preds[i]) * (
                1.0 - self._H
            )

        # 3. Changepoint probability  (run length resets to 0)
        cp_prob = 0.0
        for i in range(n):
            cp_prob += self._run_length_probs[i] * math.exp(log_preds[i]) * self._H

        # 4. New run-length distribution = [cp_prob] + growth_probs
        new_probs = [cp_prob] + growth_probs

        # Normalise
        total = sum(new_probs)
        if total > 0:
            new_probs = [p / total for p in new_probs]

        # Truncate to bound memory
        if len(new_probs) > self.max_run_length:
            new_probs = new_probs[: self.max_run_length]
            # Re-normalise after truncation
            total = sum(new_probs)
            if total > 0:
                new_probs = [p / total for p in new_probs]

        self._run_length_probs = new_probs

        # 5. Update sufficient statistics
        self._stats.add_run_length()
        self._stats.update(value)
        self._stats.truncate(self.max_run_length)

        # 6. Determine MAP run length and changepoint flag
        map_rl = 0
        map_prob = 0.0
        for i, p in enumerate(self._run_length_probs):
            if p > map_prob:
                map_prob = p
                map_rl = i

        # Changepoint detected when the reset-to-zero probability is high
        changepoint_detected = new_probs[0] > self.threshold

        if changepoint_detected:
            record = _ChangePointRecord(
                timestamp=timestamp,
                run_length=map_rl,
                probability=new_probs[0],
            )
            self._changepoint_history.append(record)
            logger.info(
                "BOCPDetector changepoint detected — prob=%.3f run_length=%d obs=%d",
                new_probs[0],
                map_rl,
                self._n_observations,
            )

        return ChangePointResult(
            changepoint_detected=changepoint_detected,
            run_length=map_rl,
            probability=map_prob,
            growth_probability=new_probs[0],
            timestamp=timestamp,
        )

    def get_run_length_distribution(self) -> List[float]:
        """Return the current run-length probability distribution.

        ``distribution[i]`` is the probability that the current run length is *i*.
        """
        return list(self._run_length_probs)

    def get_changepoint_history(self, lookback: int = 100) -> List[dict]:
        """Return the last *lookback* detected changepoints.

        Returns
        -------
        list[dict]
            Each dict has keys ``timestamp``, ``run_length``, ``probability``.
        """
        records = self._changepoint_history[-lookback:]
        return [
            {
                "timestamp": r.timestamp,
                "run_length": r.run_length,
                "probability": r.probability,
            }
            for r in records
        ]

    @property
    def n_observations(self) -> int:
        """Total number of observations processed."""
        return self._n_observations
