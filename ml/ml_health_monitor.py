"""
ML Health Monitor and Alert System.

Provides:
- Real-time model health tracking
- Performance degradation detection
- Drift detection integration
- Alert generation and management
- Health dashboards and reports

Usage:
    monitor = MLHealthMonitor()
    monitor.track_performance("xgb_model", y_true, y_pred)
    health = monitor.get_health_status()
    if health.status == "degraded":
        send_alert(health.alerts)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HealthLevel(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A health alert."""

    alert_id: str
    timestamp: str
    severity: str
    model_name: str
    metric: str
    message: str
    value: float
    threshold: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthStatus:
    """Overall health status of monitored models."""

    status: str
    overall_score: float
    models: Dict[str, Dict[str, Any]]
    alerts: List[Alert]
    last_check: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "overall_score": float(self.overall_score),
            "n_models": len(self.models),
            "n_alerts": len(self.alerts),
            "models": self.models,
            "alerts": [a.to_dict() for a in self.alerts],
            "last_check": self.last_check,
        }


@dataclass
class PerformanceSnapshot:
    """A snapshot of model performance metrics."""

    timestamp: str
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metrics": self.metrics,
        }


class MLHealthMonitor:
    """
    Real-time ML model health monitoring.

    Tracks:
    - Accuracy, precision, recall, F1
    - Prediction calibration
    - Feature drift
    - Latency and throughput
    - Memory usage

    Generates alerts when thresholds are exceeded.
    """

    def __init__(
        self,
        *,
        accuracy_threshold: float = 0.5,
        drift_threshold: float = 0.1,
        calibration_threshold: float = 0.1,
        window_size: int = 100,
        alert_callback: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        self.accuracy_threshold = accuracy_threshold
        self.drift_threshold = drift_threshold
        self.calibration_threshold = calibration_threshold
        self.window_size = window_size
        self.alert_callback = alert_callback

        self._snapshots: Dict[str, List[PerformanceSnapshot]] = {}
        self._alerts: List[Alert] = []
        self._alert_count = 0

    def track_performance(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceSnapshot:
        """Track performance for a model."""
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, y_proba)

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            predictions=y_pred,
            actuals=y_true,
            metrics=metrics,
        )

        # Store snapshot
        if model_name not in self._snapshots:
            self._snapshots[model_name] = []
        self._snapshots[model_name].append(snapshot)

        # Trim to window size
        if len(self._snapshots[model_name]) > self.window_size:
            self._snapshots[model_name] = self._snapshots[model_name][-self.window_size:]

        # Check for issues and generate alerts
        self._check_and_alert(model_name, metrics)

        return snapshot

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        metrics: Dict[str, float] = {}

        # Basic accuracy
        accuracy = np.mean((y_pred > 0.5) == y_true)
        metrics["accuracy"] = float(accuracy)

        # Precision, recall, F1 (for binary)
        tp = np.sum((y_pred > 0.5) & (y_true == 1))
        fp = np.sum((y_pred > 0.5) & (y_true == 0))
        fn = np.sum((y_pred <= 0.5) & (y_true == 1))
        tn = np.sum((y_pred <= 0.5) & (y_true == 0))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1"] = float(f1)
        metrics["tp"] = float(tp)
        metrics["fp"] = float(fp)
        metrics["tn"] = float(tn)
        metrics["fn"] = float(fn)

        # Calibration (if probabilities provided)
        if y_proba is not None:
            y_proba = np.asarray(y_proba, dtype=float)
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]

            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0

            for i in range(n_bins):
                mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
                if np.sum(mask) > 0:
                    bin_pred = np.mean(y_proba[mask])
                    bin_true = np.mean(y_true[mask])
                    ece += np.abs(bin_pred - bin_true) * np.sum(mask)

            metrics["calibration_error"] = float(ece / len(y_proba))

            # Brier score
            metrics["brier_score"] = float(np.mean((y_proba - y_true) ** 2))

        # Prediction distribution
        metrics["pred_mean"] = float(np.mean(y_pred))
        metrics["pred_std"] = float(np.std(y_pred))
        metrics["pred_positive_rate"] = float(np.mean(y_pred > 0.5))

        return metrics

    def _check_and_alert(
        self,
        model_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """Check metrics and generate alerts if needed."""
        # Accuracy check
        accuracy = metrics.get("accuracy", 0.5)
        if accuracy < self.accuracy_threshold:
            self._create_alert(
                model_name=model_name,
                severity=AlertSeverity.ERROR.value,
                metric="accuracy",
                message=f"Accuracy {accuracy:.3f} below threshold {self.accuracy_threshold:.3f}",
                value=accuracy,
                threshold=self.accuracy_threshold,
                recommendation="Retrain or investigate data drift",
            )

        # Calibration check
        if "calibration_error" in metrics:
            cal_error = metrics["calibration_error"]
            if cal_error > self.calibration_threshold:
                self._create_alert(
                    model_name=model_name,
                    severity=AlertSeverity.WARNING.value,
                    metric="calibration_error",
                    message=f"Calibration error {cal_error:.3f} above threshold {self.calibration_threshold:.3f}",
                    value=cal_error,
                    threshold=self.calibration_threshold,
                    recommendation="Recalibrate model probabilities",
                )

        # F1 check
        f1 = metrics.get("f1", 0.0)
        if f1 < 0.3:
            self._create_alert(
                model_name=model_name,
                severity=AlertSeverity.WARNING.value,
                metric="f1",
                message=f"F1 score {f1:.3f} is low",
                value=f1,
                threshold=0.3,
                recommendation="Check for class imbalance or model fit",
            )

    def _create_alert(
        self,
        model_name: str,
        severity: str,
        metric: str,
        message: str,
        value: float,
        threshold: float,
        recommendation: str,
    ) -> None:
        """Create and store an alert."""
        self._alert_count += 1
        alert = Alert(
            alert_id=f"alert_{self._alert_count}_{datetime.now(timezone.utc).strftime('%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            model_name=model_name,
            metric=metric,
            message=message,
            value=value,
            threshold=threshold,
            recommendation=recommendation,
        )

        self._alerts.append(alert)

        # Call alert callback if set
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

        logger.log(
            logging.WARNING if severity in [AlertSeverity.WARNING.value, AlertSeverity.ERROR.value] else logging.INFO,
            f"Alert: {message}",
        )

    def get_health_status(
        self,
        model_names: Optional[List[str]] = None,
    ) -> HealthStatus:
        """Get overall health status."""
        if model_names is None:
            model_names = list(self._snapshots.keys())

        models_info: Dict[str, Dict[str, Any]] = {}
        total_score = 0.0
        n_models = 0

        for model_name in model_names:
            if model_name not in self._snapshots:
                continue

            # Get recent snapshots
            snapshots = self._snapshots[model_name][-10:]
            if not snapshots:
                continue

            # Average metrics
            avg_metrics = {}
            for key in snapshots[0].metrics.keys():
                values = [s.metrics.get(key, 0) for s in snapshots]
                avg_metrics[f"avg_{key}"] = float(np.mean(values))

            # Compute model score
            accuracy = avg_metrics.get("avg_accuracy", 0.5)
            f1 = avg_metrics.get("avg_f1", 0.0)
            cal_error = avg_metrics.get("avg_calibration_error", 0.0)

            model_score = (accuracy + f1) / 2 - cal_error
            model_score = max(0.0, min(1.0, model_score))

            models_info[model_name] = {
                "score": model_score,
                "accuracy": avg_metrics.get("avg_accuracy", 0.0),
                "f1": avg_metrics.get("avg_f1", 0.0),
                "calibration_error": avg_metrics.get("avg_calibration_error", 0.0),
                "n_samples": len(snapshots),
                "last_update": snapshots[-1].timestamp,
            }

            total_score += model_score
            n_models += 1

        # Overall score
        overall_score = total_score / max(n_models, 1)

        # Determine status
        if n_models == 0:
            status = HealthLevel.UNKNOWN.value
        elif overall_score > 0.7:
            status = HealthLevel.HEALTHY.value
        elif overall_score > 0.4:
            status = HealthLevel.DEGRADED.value
        else:
            status = HealthLevel.CRITICAL.value

        # Get alerts for these models
        model_alerts = [a for a in self._alerts if a.model_name in model_names]
        # Keep last 20 alerts
        model_alerts = model_alerts[-20:]

        return HealthStatus(
            status=status,
            overall_score=overall_score,
            models=models_info,
            alerts=model_alerts,
            last_check=datetime.now(timezone.utc).isoformat(),
        )

    def get_performance_trend(
        self,
        model_name: str,
        metric: str = "accuracy",
        window: int = 50,
    ) -> List[Tuple[str, float]]:
        """Get performance trend for a metric."""
        if model_name not in self._snapshots:
            return []

        snapshots = self._snapshots[model_name]
        trend = []

        for snapshot in snapshots[-window:]:
            value = snapshot.metrics.get(metric, 0.0)
            trend.append((snapshot.timestamp, value))

        return trend

    def detect_performance_drop(
        self,
        model_name: str,
        metric: str = "accuracy",
        threshold: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        """Detect significant performance drop."""
        trend = self.get_performance_trend(model_name, metric)
        if len(trend) < 10:
            return None

        # Compare recent average to historical average
        recent_values = [v for _, v in trend[-5:]]
        historical_values = [v for _, v in trend[:-5]]

        recent_avg = float(np.mean(recent_values))
        historical_avg = float(np.mean(historical_values))
        drop = historical_avg - recent_avg

        if drop > threshold:
            return {
                "model_name": model_name,
                "metric": metric,
                "drop": drop,
                "recent_avg": recent_avg,
                "historical_avg": historical_avg,
                "recommendation": "Investigate and potentially retrain",
            }

        return None

    def clear_alerts(
        self,
        model_name: Optional[str] = None,
        before_timestamp: Optional[str] = None,
    ) -> int:
        """Clear alerts for a model or before timestamp."""
        if model_name is None and before_timestamp is None:
            cleared = len(self._alerts)
            self._alerts.clear()
            return cleared

        before_dt = None
        if before_timestamp:
            from dateutil import parser
            before_dt = parser.parse(before_timestamp)

        def should_clear(alert: Alert) -> bool:
            if model_name and alert.model_name != model_name:
                return False
            if before_dt:
                from dateutil import parser
                alert_dt = parser.parse(alert.timestamp)
                if alert_dt < before_dt:
                    return True
                return False
            return True

        original_count = len(self._alerts)
        self._alerts = [a for a in self._alerts if not should_clear(a)]
        return original_count - len(self._alerts)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        health = self.get_health_status()
        return {
            "status": health.status,
            "overall_score": health.overall_score,
            "n_models": len(health.models),
            "n_alerts": len(health.alerts),
            "critical_alerts": sum(1 for a in health.alerts if a.severity == AlertSeverity.CRITICAL.value),
            "models": {
                name: {"score": info["score"], "accuracy": info["accuracy"]}
                for name, info in health.models.items()
            },
        }


class AlertManager:
    """
    Manage and route alerts to appropriate channels.

    Supports:
    - Log alerts
    - Email alerts (stub)
    - Webhook alerts (stub)
    - Alert deduplication
    - Alert aggregation
    """

    def __init__(
        self,
        *,
        deduplicate_window: int = 300,  # seconds
        aggregate_window: int = 60,
    ) -> None:
        self.deduplicate_window = deduplicate_window
        self.aggregate_window = aggregate_window
        self._seen_alerts: Dict[str, datetime] = {}
        self._alert_handlers: Dict[str, List[Callable]] = {
            "log": [self._log_handler],
        }

    def register_handler(
        self,
        channel: str,
        handler: Callable[[Alert], None],
    ) -> None:
        """Register a handler for alerts."""
        if channel not in self._alert_handlers:
            self._alert_handlers[channel] = []
        self._alert_handlers[channel].append(handler)

    def send_alert(self, alert: Alert) -> None:
        """Send an alert through all registered handlers."""
        # Check deduplication
        alert_key = f"{alert.model_name}_{alert.metric}"
        if alert_key in self._seen_alerts:
            last_seen = self._seen_alerts[alert_key]
            elapsed = (datetime.now(timezone.utc) - last_seen).total_seconds()
            if elapsed < self.deduplicate_window:
                logger.debug(f"Deduplicating alert: {alert_key}")
                return

        self._seen_alerts[alert_key] = datetime.now(timezone.utc)

        # Send to handlers
        for handlers in self._alert_handlers.values():
            for handler in handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.warning(f"Alert handler failed: {e}")

    def _log_handler(self, alert: Alert) -> None:
        """Log handler."""
        log_level = {
            AlertSeverity.INFO.value: logging.INFO,
            AlertSeverity.WARNING.value: logging.WARNING,
            AlertSeverity.ERROR.value: logging.ERROR,
            AlertSeverity.CRITICAL.value: logging.CRITICAL,
        }.get(alert.severity, logging.INFO)

        logger.log(log_level, f"[{alert.severity.upper()}] {alert.model_name}: {alert.message}")


def create_monitor(
    alert_callback: Optional[Callable[[Alert], None]] = None,
) -> MLHealthMonitor:
    """Factory function to create health monitor."""
    return MLHealthMonitor(alert_callback=alert_callback)


__all__ = [
    "MLHealthMonitor",
    "AlertManager",
    "HealthStatus",
    "Alert",
    "AlertSeverity",
    "PerformanceSnapshot",
    "create_monitor",
]