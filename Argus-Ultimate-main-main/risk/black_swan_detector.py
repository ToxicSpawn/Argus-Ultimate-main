"""
Black Swan / Anomaly Detector — monitors multiple market metrics for extreme
events and produces a system-wide risk level assessment.

Uses rolling Z-scores over configurable windows to detect statistical anomalies
across price, volume, funding rates, open interest, and exchange flows.

Operates entirely in-memory with thread-safe per-symbol storage.

Usage:
    detector = BlackSwanDetector(window_hours=168)
    detector.update_metrics("BTC/USD", price=65000, volume=1200,
                            funding_rate=0.001, oi_change_pct=-5.0)
    report = detector.detect_anomalies("BTC/USD")
    level = detector.get_system_risk_level()
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnomalyReport:
    """Result of anomaly detection for a single symbol."""
    symbol: str
    anomaly_detected: bool
    anomaly_score: float                # 0-1 composite score
    triggers: List[str]                 # list of triggered conditions
    severity: str                       # "low" / "medium" / "high" / "critical"
    recommendation: str                 # "monitor" / "reduce" / "hedge" / "halt"
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetricPoint:
    """Single observation of market metrics."""
    timestamp: float
    price: float
    volume: float
    funding_rate: Optional[float] = None
    oi_change_pct: Optional[float] = None
    exchange_flow: Optional[float] = None


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = {
    "price_zscore": 3.0,
    "volume_zscore": 4.0,
    "funding_rate_extreme": 0.01,       # abs funding rate > 1% per period
    "oi_drop_pct": 10.0,                # OI drops more than 10%
    "exchange_inflow_zscore": 3.5,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BlackSwanDetector:
    """
    Multi-metric anomaly detector for crypto markets.

    Parameters
    ----------
    window_hours : int
        Rolling window size in hours (default 168 = 7 days).
    thresholds : dict or None
        Override default Z-score / percentage thresholds.
    max_points_per_symbol : int
        Maximum metric points retained per symbol.
    """

    def __init__(
        self,
        window_hours: int = 168,
        thresholds: Optional[Dict[str, float]] = None,
        max_points_per_symbol: int = 10_000,
    ) -> None:
        self._window_seconds = window_hours * 3600
        self._max_points = max(100, max_points_per_symbol)
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self._thresholds.update(thresholds)

        # symbol -> deque[MetricPoint]
        self._data: Dict[str, Deque[MetricPoint]] = {}
        # Cache of most recent anomaly report per symbol
        self._reports: Dict[str, AnomalyReport] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        logger.info(
            "BlackSwanDetector initialised  window=%dh  max_points=%d  thresholds=%s",
            window_hours, self._max_points, self._thresholds,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_lock(self, symbol: str) -> threading.Lock:
        with self._global_lock:
            if symbol not in self._locks:
                self._locks[symbol] = threading.Lock()
            return self._locks[symbol]

    def _prune(self, symbol: str) -> None:
        """Remove points outside the rolling window."""
        cutoff = time.time() - self._window_seconds
        dq = self._data.get(symbol)
        if dq:
            while dq and dq[0].timestamp < cutoff:
                dq.popleft()

    @staticmethod
    def _zscore(values: List[float], current: float) -> float:
        """Compute Z-score of *current* relative to *values*."""
        if len(values) < 2:
            return 0.0
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        return (current - mean) / std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_metrics(
        self,
        symbol: str,
        price: float,
        volume: float,
        funding_rate: Optional[float] = None,
        oi_change_pct: Optional[float] = None,
        exchange_flow: Optional[float] = None,
    ) -> None:
        """
        Record a new set of market metrics for a symbol.

        Parameters
        ----------
        price : float           Current price.
        volume : float          Recent volume (e.g. 1h candle volume).
        funding_rate : float    Perpetual funding rate (optional).
        oi_change_pct : float   Open interest change percentage (optional).
        exchange_flow : float   Net exchange inflow (positive = inflow, optional).
        """
        now = time.time()
        point = MetricPoint(
            timestamp=now,
            price=price,
            volume=volume,
            funding_rate=funding_rate,
            oi_change_pct=oi_change_pct,
            exchange_flow=exchange_flow,
        )

        lock = self._get_lock(symbol)
        with lock:
            if symbol not in self._data:
                self._data[symbol] = deque(maxlen=self._max_points)
            self._data[symbol].append(point)
            self._prune(symbol)

        logger.debug(
            "Metrics update  symbol=%s  price=%.2f  vol=%.2f  funding=%s  oi_chg=%s  flow=%s",
            symbol, price, volume, funding_rate, oi_change_pct, exchange_flow,
        )

    def detect_anomalies(self, symbol: str) -> AnomalyReport:
        """
        Run anomaly detection on the latest metrics for *symbol*.

        Returns
        -------
        AnomalyReport
        """
        lock = self._get_lock(symbol)
        with lock:
            history = self._data.get(symbol)
            if not history or len(history) < 5:
                return AnomalyReport(
                    symbol=symbol,
                    anomaly_detected=False,
                    anomaly_score=0.0,
                    triggers=[],
                    severity="low",
                    recommendation="monitor",
                )
            points = list(history)

        latest = points[-1]
        triggers: List[str] = []
        scores: List[float] = []

        # --- Price Z-score ---
        prices = [p.price for p in points]
        price_z = abs(self._zscore(prices[:-1], latest.price))
        threshold = self._thresholds["price_zscore"]
        if price_z > threshold:
            triggers.append(f"price_zscore={price_z:.2f}>{threshold}")
            scores.append(min(1.0, price_z / (threshold * 2)))

        # --- Volume Z-score ---
        volumes = [p.volume for p in points]
        vol_z = abs(self._zscore(volumes[:-1], latest.volume))
        threshold = self._thresholds["volume_zscore"]
        if vol_z > threshold:
            triggers.append(f"volume_zscore={vol_z:.2f}>{threshold}")
            scores.append(min(1.0, vol_z / (threshold * 2)))

        # --- Funding rate extreme ---
        if latest.funding_rate is not None:
            fr_threshold = self._thresholds["funding_rate_extreme"]
            if abs(latest.funding_rate) > fr_threshold:
                triggers.append(f"funding_rate={latest.funding_rate:.4f} (extreme)")
                scores.append(min(1.0, abs(latest.funding_rate) / (fr_threshold * 3)))

        # --- OI drop ---
        if latest.oi_change_pct is not None:
            oi_threshold = self._thresholds["oi_drop_pct"]
            if latest.oi_change_pct < -oi_threshold:
                triggers.append(
                    f"oi_drop={latest.oi_change_pct:.1f}%<-{oi_threshold}%"
                )
                scores.append(min(1.0, abs(latest.oi_change_pct) / (oi_threshold * 3)))

        # --- Exchange inflow spike ---
        if latest.exchange_flow is not None:
            flows = [p.exchange_flow for p in points if p.exchange_flow is not None]
            if len(flows) >= 5:
                flow_z = self._zscore(flows[:-1], latest.exchange_flow)
                flow_threshold = self._thresholds["exchange_inflow_zscore"]
                if flow_z > flow_threshold:
                    triggers.append(f"exchange_inflow_zscore={flow_z:.2f}>{flow_threshold}")
                    scores.append(min(1.0, flow_z / (flow_threshold * 2)))

        # --- Composite score ---
        if scores:
            anomaly_score = min(1.0, sum(scores) / len(scores) + 0.1 * (len(scores) - 1))
        else:
            anomaly_score = 0.0

        # --- Severity and recommendation ---
        severity, recommendation = self._classify(anomaly_score, len(triggers))

        report = AnomalyReport(
            symbol=symbol,
            anomaly_detected=len(triggers) > 0,
            anomaly_score=round(anomaly_score, 4),
            triggers=triggers,
            severity=severity,
            recommendation=recommendation,
        )

        with lock:
            self._reports[symbol] = report

        if report.anomaly_detected:
            logger.warning(
                "Anomaly detected  symbol=%s  score=%.3f  severity=%s  triggers=%s  rec=%s",
                symbol, anomaly_score, severity, triggers, recommendation,
            )

        return report

    def get_system_risk_level(self) -> str:
        """
        Return the overall system risk level based on active anomalies.

        Returns
        -------
        str
            "green"  — no anomalies
            "yellow" — 1 low/medium anomaly
            "orange" — 2+ anomalies or any high severity
            "red"    — any critical anomaly or 3+ high anomalies
        """
        with self._global_lock:
            reports = list(self._reports.values())

        active = [r for r in reports if r.anomaly_detected]
        if not active:
            return "green"

        severities = [r.severity for r in active]
        critical_count = severities.count("critical")
        high_count = severities.count("high")

        if critical_count > 0 or high_count >= 3:
            level = "red"
        elif high_count > 0 or len(active) >= 2:
            level = "orange"
        else:
            level = "yellow"

        logger.debug(
            "System risk level=%s  active_anomalies=%d  critical=%d  high=%d",
            level, len(active), critical_count, high_count,
        )
        return level

    def get_active_anomalies(self) -> List[AnomalyReport]:
        """Return all currently active anomaly reports."""
        with self._global_lock:
            return [r for r in self._reports.values() if r.anomaly_detected]

    def get_symbols(self) -> List[str]:
        """Return list of monitored symbols."""
        with self._global_lock:
            return list(self._data.keys())

    # ------------------------------------------------------------------
    # Internal classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(score: float, trigger_count: int) -> Tuple[str, str]:
        """
        Map composite anomaly score to severity and recommendation.

        Returns (severity, recommendation).
        """
        if score >= 0.8 or trigger_count >= 4:
            return ("critical", "halt")
        elif score >= 0.6 or trigger_count >= 3:
            return ("high", "hedge")
        elif score >= 0.35 or trigger_count >= 2:
            return ("medium", "reduce")
        elif score > 0:
            return ("low", "monitor")
        else:
            return ("low", "monitor")
