"""
ml/model_rolling/drift_detector.py
====================================
Statistical drift detection for model predictions.

Supports:
  - ADWIN (Adaptive Windowing) for concept drift
  - Page-Hinkley test for abrupt drift
  - Rolling window comparison (PSI / KL-divergence)
  - Early drift warnings before significant degradation

Each monitored model signal gets its own detector instance.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DriftStatus(Enum):
    NO_DRIFT    = "no_drift"
    WARNING     = "warning"
    DRIFT       = "drift"
    RECOVERING  = "recovering"


@dataclass
class DriftAlert:
    """Fired when drift is detected on a monitored signal."""
    model_name    : str
    signal        : str           # e.g. "sharpe", "accuracy", "mse"
    status        : DriftStatus
    p_value       : float
    drift_magnitude: float        # e.g. PSI value or PH statistic
    window_start  : float         # epoch seconds
    window_end    : float
    message       : str
    should_rollback: bool         # True if auto-rollback threshold crossed

    def to_dict(self) -> Dict:
        return {
            "model_name"     : self.model_name,
            "signal"        : self.signal,
            "status"        : self.status.value,
            "p_value"       : self.p_value,
            "drift_magnitude": self.drift_magnitude,
            "window_start"  : self.window_start,
            "window_end"    : self.window_end,
            "message"       : self.message,
            "should_rollback": self.should_rollback,
            "timestamp"     : datetime.utcnow().isoformat(),
        }


@dataclass
class PredictionSample:
    """One (prediction, actual) pair for a model signal."""
    timestamp    : float
    prediction   : float
    actual       : float
    weight       : float = 1.0    # sample importance


# ---------------------------------------------------------------------------
# ADWIN — Adaptive Windowing for concept drift
# ---------------------------------------------------------------------------

class ADWIN:
    """
    ADWIN detects concept drift by maintaining a variable-length window
    of recent items. When the window contains two sub-windows with
    statistically different means, drift is declared.

    Reference: Bifet & Gavalda (2007).
    """

    def __init__(self, confidence: float = 0.002, min_window: int = 30) -> None:
        self._confidence   = confidence
        self._min_window   = min_window
        self._window: Deque[float] = deque()
        self._total        = 0.0
        self._total_sq     = 0.0

    def add(self, value: float) -> bool:
        """Return True if drift was detected."""
        self._window.append(value)
        self._total     += value
        self._total_sq  += value * value
        return self._detect()

    def _detect(self) -> bool:
        n = len(self._window)
        if n < self._min_window * 2:
            return False

        # Try all split points
        for split in range(self._min_window, n - self._min_window):
            left  = list(self._window)[:split]
            right = list(self._window)[split:]
            n0, n1 = len(left), len(right)
            if n0 < self._min_window or n1 < self._min_window:
                continue

            mean0 = sum(left) / n0
            mean1 = sum(right) / n1
            var0  = sum((x - mean0)**2 for x in left)  / n0
            var1  = sum((x - mean1)**2 for x in right) / n1

            epsilon_cut = math.sqrt(
                (1.0 / (2.0 * n0)) * math.log(4.0 / self._confidence)
            ) + math.sqrt(
                (1.0 / (2.0 * n1)) * math.log(4.0 / self._confidence)
            )

            if abs(mean0 - mean1) > epsilon_cut:
                # Drift detected — trim window to right side only
                for _ in range(split):
                    v = self._window.popleft()
                    self._total    -= v
                    self._total_sq -= v * v
                return True
        return False

    @property
    def mean(self) -> float:
        return self._total / len(self._window) if self._window else 0.0

    @property
    def variance(self) -> float:
        n = len(self._window)
        if n < 2:
            return 0.0
        return (self._total_sq / n) - (self._total / n) ** 2

    def reset(self) -> None:
        self._window.clear()
        self._total = 0.0
        self._total_sq = 0.0


# ---------------------------------------------------------------------------
# Page-Hinkley test for abrupt drift
# ---------------------------------------------------------------------------

class PageHinkley:
    """
    Page-Hinkley sequential change detection.
    Detects abrupt changes in the mean of a signal.

    Reference: Page (1954).
    """

    def __init__(
        self,
        threshold   : float = 50.0,
        alpha      : float = 0.9995,  # forgetting factor
        min_obs    : int   = 100,
    ) -> None:
        self._threshold  = threshold
        self._alpha     = alpha
        self._min_obs   = min_obs
        self._mean      = 0.0
        self._cumulative: float = 0.0
        self._n         = 0
        self._in_drift  = False

    def add(self, value: float) -> bool:
        """Return True if a change point is detected."""
        self._n += 1
        if self._n == 1:
            self._mean = value
            return False

        # Update running mean with exponential decay
        self._mean = self._alpha * self._mean + (1 - self._alpha) * value

        # Cumulative sum of deviations from the running mean
        self._cumulative += value - self._mean

        if self._n < self._min_obs:
            return False

        if abs(self._cumulative) > self._threshold:
            self._cumulative = 0.0
            self._in_drift = True
            return True
        return False

    def reset(self) -> None:
        self._mean = 0.0
        self._cumulative = 0.0
        self._n = 0
        self._in_drift = False


# ---------------------------------------------------------------------------
# Rolling comparison — PSI / KL-divergence
# ---------------------------------------------------------------------------

class RollingWindowComparison:
    """
    Compares two rolling windows using:
      - PSI (Population Stability Index) for numerical predictions
      - Proportion shift for bounded signals
    """

    def __init__(
        self,
        window_size  : int   = 500,
        bucket_count : int   = 10,
        psi_threshold: float = 0.2,
    ) -> None:
        self._window_size   = window_size
        self._bucket_count  = bucket_count
        self._psi_threshold = psi_threshold
        self._reference     : Deque[float] = deque(maxlen=window_size)
        self._current       : Deque[float] = deque(maxlen=window_size)
        self._psi           = 0.0

    def add(self, value: float, is_reference: bool = False) -> None:
        if is_reference:
            self._reference.append(value)
        else:
            self._current.append(value)

    def compute_psi(self) -> Tuple[float, bool]:
        """
        Compute PSI between reference and current windows.
        Returns (psi_value, is_drift_detected).
        Drift is True when PSI > psi_threshold.
        """
        if len(self._reference) < self._window_size // 2:
            return 0.0, False
        if len(self._current) < self._window_size // 2:
            return 0.0, False

        ref_arr = np.array(self._reference)
        cur_arr = np.array(self._current)

        # Bucket into equal-width bins across combined range
        all_vals = np.concatenate([ref_arr, cur_arr])
        min_v, max_v = np.min(all_vals), np.max(all_vals)
        if max_v == min_v:
            return 0.0, False

        bucket_edges = np.linspace(min_v, max_v, self._bucket_count + 1)

        ref_pcts = np.histogram(ref_arr, bins=bucket_edges)[0] / len(ref_arr)
        cur_pcts = np.histogram(cur_arr, bins=bucket_edges)[0] / len(cur_arr)

        # Clamp to avoid log(0)
        ref_pcts = np.clip(ref_pcts, 1e-6, 1.0)
        cur_pcts = np.clip(cur_pcts, 1e-6, 1.0)

        psi = np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts))
        self._psi = float(psi)
        return self._psi, self._psi > self._psi_threshold

    @property
    def psi(self) -> float:
        return self._psi


# ---------------------------------------------------------------------------
# Composite drift detector for a model's signal
# ---------------------------------------------------------------------------

class SignalDriftDetector:
    """
    Composite drift detector combining ADWIN, Page-Hinkley, and
    rolling PSI for robust detection.
    """

    def __init__(
        self,
        signal_name      : str,
        adwin_confidence : float  = 0.002,
        ph_threshold    : float  = 50.0,
        psi_threshold   : float  = 0.2,
        rollback_threshold: float = 0.5,  # PSI beyond this triggers rollback
    ) -> None:
        self.signal_name  = signal_name
        self.rollback_threshold = rollback_threshold

        self._adwin = ADWIN(confidence=adwin_confidence)
        self._ph    = PageHinkley(threshold=ph_threshold)
        self._psi   = RollingWindowComparison(psi_threshold=psi_threshold)
        self._psi_ref: Deque[float] = deque(maxlen=500)

        self._samples: Deque[PredictionSample] = deque(maxlen=2000)
        self._lock    = threading.Lock()

    def record(self, pred: float, actual: float, timestamp: Optional[float] = None) -> None:
        ts = timestamp or time.time()
        sample = PredictionSample(timestamp=ts, prediction=pred, actual=actual)
        with self._lock:
            self._samples.append(sample)
            self._psi_ref.append(pred)

    def check(self, model_name: str, now: Optional[float] = None) -> List[DriftAlert]:
        """
        Run all detectors. Returns list of alerts (may be empty).
        """
        alerts: List[DriftAlert] = []
        if not self._samples:
            return alerts

        ts = now or time.time()
        preds   = [s.prediction for s in self._samples]
        errors  = [abs(s.prediction - s.actual) for s in self._samples]
        residuals = [s.prediction - s.actual for s in self._samples]

        # --- ADWIN on prediction errors ---
        adwin_drift = False
        for e in errors:
            if self._adwin.add(e):
                adwin_drift = True
                break

        # --- Page-Hinkley on residuals ---
        ph_drift = False
        for r in residuals:
            if self._ph.add(r):
                ph_drift = True
                break

        # --- PSI ---
        psi_value, psi_drift = 0.0, False
        if len(self._psi_ref) >= 250:
            for p in self._psi_ref:
                self._psi.add(p, is_reference=True)
            for p in preds[-250:]:
                self._psi.add(p, is_reference=False)
            psi_value, psi_drift = self._psi.compute_psi()

        # Synthesise alert
        if adwin_drift or ph_drift or psi_drift:
            statuses = []
            if adwin_drift: statuses.append("ADWIN")
            if ph_drift:    statuses.append("PageHinkley")
            if psi_drift:    statuses.append(f"PSI={psi_value:.4f}")

            mag = max(psi_value, float(adwin_drift) * 0.3, float(ph_drift) * 0.5)
            should_rollback = psi_value > self.rollback_threshold

            alert = DriftAlert(
                model_name      = model_name,
                signal          = self.signal_name,
                status          = DriftStatus.DRIFT if should_rollback else DriftStatus.WARNING,
                p_value         = 0.0,
                drift_magnitude = mag,
                window_start    = self._samples[0].timestamp,
                window_end      = ts,
                message         = f"Drift detected via {'+'.join(statuses)}",
                should_rollback = should_rollback,
            )
            alerts.append(alert)

        return alerts

    def reset(self) -> None:
        with self._lock:
            self._adwin.reset()
            self._ph.reset()
            self._psi = RollingWindowComparison()
            self._psi_ref.clear()
            self._samples.clear()

    def sample_count(self) -> int:
        return len(self._samples)


# ---------------------------------------------------------------------------
# Rolling drift detector managing all signals for a model
# ---------------------------------------------------------------------------

class RollingDriftDetector:
    """
    Manages SignalDriftDetector instances per signal for a given model,
    with a global check() that returns alerts from all signals.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._detectors: Dict[str, SignalDriftDetector] = {}
        self._lock     = threading.Lock()

    def register_signal(
        self,
        signal          : str,
        adwin_confidence : float  = 0.002,
        ph_threshold    : float  = 50.0,
        psi_threshold   : float  = 0.2,
        rollback_threshold: float = 0.5,
    ) -> None:
        with self._lock:
            self._detectors[signal] = SignalDriftDetector(
                signal_name         = signal,
                adwin_confidence    = adwin_confidence,
                ph_threshold        = ph_threshold,
                psi_threshold       = psi_threshold,
                rollback_threshold  = rollback_threshold,
            )

    def record(self, signal: str, pred: float, actual: float) -> None:
        with self._lock:
            det = self._detectors.get(signal)
        if det:
            det.record(pred, actual)
        else:
            # Auto-register with defaults
            self.register_signal(signal)
            self._detectors[signal].record(pred, actual)

    def check(self) -> List[DriftAlert]:
        """Check all registered signals for drift."""
        alerts: List[DriftAlert] = []
        with self._lock:
            detectors = dict(self._detectors)
        for signal, det in detectors.items():
            alerts.extend(det.check(self.model_name))
        return alerts

    def signals(self) -> List[str]:
        return list(self._detectors.keys())

    def summary(self) -> Dict:
        with self._lock:
            return {s: d.sample_count() for s, d in self._detectors.items()}
