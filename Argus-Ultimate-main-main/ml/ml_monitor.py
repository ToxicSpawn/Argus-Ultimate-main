"""
ML Monitoring Dashboard — real-time model health and performance tracking.

Features:
  - Model prediction tracking
  - Drift detection alerts
  - Performance degradation alerts
  - Feature importance tracking
  - Model comparison views
  - Export to JSON for external dashboards

Usage:
    monitor = MLMonitor()
    
    # Track predictions
    monitor.track_prediction("regime_classifier", prediction, confidence)
    
    # Track actual outcomes
    monitor.track_outcome("regime_classifier", actual_label)
    
    # Get health report
    health = monitor.get_health_report("regime_classifier")
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of ML alerts."""
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_ERROR_RATE = "high_error_rate"
    MODEL_STALE = "model_stale"
    FEATURE_ANOMALY = "feature_anomaly"


@dataclass
class MLAlert:
    """ML monitoring alert."""
    alert_id: str
    model_name: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "model_name": self.model_name,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class PredictionRecord:
    """Record of a single prediction."""
    model_name: str
    prediction: Any
    confidence: float
    timestamp: datetime
    actual: Optional[Any] = None
    error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prediction": str(self.prediction),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "actual": str(self.actual) if self.actual is not None else None,
            "error": round(self.error, 4) if self.error is not None else None,
        }


@dataclass
class ModelHealth:
    """Model health summary."""
    model_name: str
    status: str  # "healthy", "degraded", "critical"
    total_predictions: int
    avg_confidence: float
    avg_error: float
    error_rate: float  # Fraction of predictions with error > threshold
    drift_score: float
    last_prediction_time: Optional[datetime]
    uptime_hours: float
    alerts: List[MLAlert]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "total_predictions": self.total_predictions,
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_error": round(self.avg_error, 4),
            "error_rate": round(self.error_rate, 4),
            "drift_score": round(self.drift_score, 4),
            "last_prediction_time": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            "uptime_hours": round(self.uptime_hours, 2),
            "active_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "alerts": [a.to_dict() for a in self.alerts[-10:]],  # Last 10 alerts
        }


class MLMonitor:
    """
    Real-time ML model monitoring and alerting.
    
    Tracks:
    - Predictions and confidence
    - Actual outcomes and errors
    - Drift detection
    - Performance degradation
    - Alert generation
    
    Args:
        window_size: Number of recent predictions to track
        alert_cooldown_seconds: Minimum time between similar alerts
        confidence_threshold: Minimum expected confidence
        error_threshold: Maximum acceptable error
        drift_window: Number of samples for drift detection
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        alert_cooldown_seconds: int = 300,
        confidence_threshold: float = 0.5,
        error_threshold: float = 0.3,
        drift_window: int = 100,
    ):
        self.window_size = window_size
        self.alert_cooldown = alert_cooldown_seconds
        self.confidence_threshold = confidence_threshold
        self.error_threshold = error_threshold
        self.drift_window = drift_window
        
        # Per-model tracking
        self._predictions: Dict[str, Deque[PredictionRecord]] = {}
        self._alerts: Dict[str, List[MLAlert]] = {}
        self._last_alert_time: Dict[str, datetime] = {}
        self._baselines: Dict[str, Dict[str, float]] = {}
        
        # Global tracking
        self._start_time = datetime.now(timezone.utc)
        self._total_predictions = 0
        self._total_alerts = 0
    
    def track_prediction(
        self,
        model_name: str,
        prediction: Any,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track a model prediction."""
        if model_name not in self._predictions:
            self._predictions[model_name] = deque(maxlen=self.window_size)
            self._alerts[model_name] = []
        
        record = PredictionRecord(
            model_name=model_name,
            prediction=prediction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
        )
        
        self._predictions[model_name].append(record)
        self._total_predictions += 1
        
        # Check for low confidence alert
        if confidence < self.confidence_threshold:
            self._maybe_alert(model_name, AlertType.LOW_CONFIDENCE, {
                "confidence": confidence,
                "threshold": self.confidence_threshold,
            })
    
    def track_outcome(
        self,
        model_name: str,
        actual: Any,
        prediction_index: int = -1,
    ) -> None:
        """Track actual outcome for a prediction."""
        if model_name not in self._predictions:
            return
        
        predictions = list(self._predictions[model_name])
        
        if not predictions:
            return
        
        # Find the prediction to match
        if prediction_index < 0:
            record = predictions[prediction_index]
        else:
            if prediction_index >= len(predictions):
                return
            record = predictions[prediction_index]
        
        # Update record
        record.actual = actual
        
        # Compute error
        try:
            if isinstance(record.prediction, (int, float)):
                record.error = abs(float(record.prediction) - float(actual))
            elif isinstance(record.prediction, np.ndarray):
                record.error = float(np.mean(np.abs(record.prediction - actual)))
            else:
                # Classification: 0 if correct, 1 if wrong
                record.error = 0.0 if record.prediction == actual else 1.0
        except Exception:
            record.error = 1.0
        
        # Check for high error alert
        if record.error > self.error_threshold:
            self._maybe_alert(model_name, AlertType.HIGH_ERROR_RATE, {
                "error": record.error,
                "threshold": self.error_threshold,
            })
    
    def set_baseline(
        self,
        model_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """Set baseline metrics for comparison."""
        self._baselines[model_name] = metrics.copy()
        logger.info("Baseline set for %s: %s", model_name, metrics)
    
    def get_health(self, model_name: str) -> ModelHealth:
        """Get health summary for a model."""
        predictions = list(self._predictions.get(model_name, []))
        alerts = self._alerts.get(model_name, [])
        
        if not predictions:
            return ModelHealth(
                model_name=model_name,
                status="unknown",
                total_predictions=0,
                avg_confidence=0.0,
                avg_error=0.0,
                error_rate=0.0,
                drift_score=0.0,
                last_prediction_time=None,
                uptime_hours=0.0,
                alerts=[],
            )
        
        # Compute stats
        confidences = [p.confidence for p in predictions]
        errors = [p.error for p in predictions if p.error is not None]
        
        avg_confidence = np.mean(confidences)
        avg_error = np.mean(errors) if errors else 0.0
        error_rate = len([e for e in errors if e > self.error_threshold]) / max(len(errors), 1)
        
        # Drift score (comparing recent vs older predictions)
        drift_score = self._compute_drift_score(predictions)
        
        # Status
        if error_rate > 0.3 or drift_score > 0.5:
            status = "critical"
        elif error_rate > 0.15 or drift_score > 0.3:
            status = "degraded"
        else:
            status = "healthy"
        
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() / 3600
        
        return ModelHealth(
            model_name=model_name,
            status=status,
            total_predictions=len(predictions),
            avg_confidence=float(avg_confidence),
            avg_error=float(avg_error),
            error_rate=float(error_rate),
            drift_score=float(drift_score),
            last_prediction_time=predictions[-1].timestamp if predictions else None,
            uptime_hours=uptime,
            alerts=alerts,
        )
    
    def get_all_health(self) -> Dict[str, ModelHealth]:
        """Get health for all tracked models."""
        return {
            name: self.get_health(name)
            for name in self._predictions.keys()
        }
    
    def get_alerts(
        self,
        model_name: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        unacknowledged_only: bool = False,
    ) -> List[MLAlert]:
        """Get alerts with optional filtering."""
        all_alerts: List[MLAlert] = []
        
        if model_name:
            all_alerts = self._alerts.get(model_name, [])
        else:
            for alerts in self._alerts.values():
                all_alerts.extend(alerts)
        
        # Filter
        if severity:
            all_alerts = [a for a in all_alerts if a.severity == severity]
        
        if unacknowledged_only:
            all_alerts = [a for a in all_alerts if not a.acknowledged]
        
        return sorted(all_alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, model_name: str, alert_id: str) -> bool:
        """Acknowledge an alert."""
        alerts = self._alerts.get(model_name, [])
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export all data for dashboard display."""
        health = self.get_all_health()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_predictions": self._total_predictions,
            "total_alerts": self._total_alerts,
            "models": {
                name: h.to_dict() for name, h in health.items()
            },
            "summary": {
                "healthy": len([h for h in health.values() if h.status == "healthy"]),
                "degraded": len([h for h in health.values() if h.status == "degraded"]),
                "critical": len([h for h in health.values() if h.status == "critical"]),
            },
        }
    
    def save_dashboard(self, path: str) -> None:
        """Save dashboard data to JSON file."""
        data = self.export_dashboard_data()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Dashboard saved to %s", path)
    
    def _compute_drift_score(self, predictions: List[PredictionRecord]) -> float:
        """Compute drift score by comparing recent vs older predictions."""
        if len(predictions) < self.drift_window * 2:
            return 0.0
        
        # Split into older and recent
        older = predictions[-self.drift_window * 2:-self.drift_window]
        recent = predictions[-self.drift_window:]
        
        # Compare confidence distributions
        older_conf = [p.confidence for p in older]
        recent_conf = [p.confidence for p in recent]
        
        # Simple drift: difference in means
        conf_drift = abs(np.mean(recent_conf) - np.mean(older_conf))
        
        # Compare error rates if we have actuals
        older_errors = [p.error for p in older if p.error is not None]
        recent_errors = [p.error for p in recent if p.error is not None]
        
        error_drift = 0.0
        if older_errors and recent_errors:
            older_error_rate = np.mean([e > self.error_threshold for e in older_errors])
            recent_error_rate = np.mean([e > self.error_threshold for e in recent_errors])
            error_drift = abs(recent_error_rate - older_error_rate)
        
        return float(min(1.0, conf_drift + error_drift))
    
    def _maybe_alert(
        self,
        model_name: str,
        alert_type: AlertType,
        details: Dict[str, Any],
    ) -> None:
        """Create alert if cooldown has passed."""
        alert_key = f"{model_name}:{alert_type.value}"
        now = datetime.now(timezone.utc)
        
        # Check cooldown
        last_time = self._last_alert_time.get(alert_key)
        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.alert_cooldown:
                return
        
        # Determine severity
        severity = self._determine_severity(alert_type, details)
        
        # Create alert
        alert = MLAlert(
            alert_id=f"alert_{int(now.timestamp())}_{self._total_alerts}",
            model_name=model_name,
            alert_type=alert_type,
            severity=severity,
            message=self._format_alert_message(alert_type, details),
            details=details,
            timestamp=now,
        )
        
        self._alerts.setdefault(model_name, []).append(alert)
        self._last_alert_time[alert_key] = now
        self._total_alerts += 1
        
        logger.warning("ML Alert: %s - %s", model_name, alert.message)
    
    def _determine_severity(
        self,
        alert_type: AlertType,
        details: Dict[str, Any],
    ) -> AlertSeverity:
        """Determine alert severity."""
        if alert_type == AlertType.DRIFT_DETECTED:
            drift_score = details.get("drift_score", 0.0)
            if drift_score > 0.5:
                return AlertSeverity.CRITICAL
            elif drift_score > 0.3:
                return AlertSeverity.ERROR
            else:
                return AlertSeverity.WARNING
        
        elif alert_type == AlertType.PERFORMANCE_DEGRADATION:
            degradation = details.get("degradation_pct", 0.0)
            if degradation > 50:
                return AlertSeverity.CRITICAL
            elif degradation > 25:
                return AlertSeverity.ERROR
            else:
                return AlertSeverity.WARNING
        
        elif alert_type == AlertType.LOW_CONFIDENCE:
            return AlertSeverity.WARNING
        
        elif alert_type == AlertType.HIGH_ERROR_RATE:
            return AlertSeverity.ERROR
        
        return AlertSeverity.INFO
    
    def _format_alert_message(
        self,
        alert_type: AlertType,
        details: Dict[str, Any],
    ) -> str:
        """Format alert message."""
        if alert_type == AlertType.LOW_CONFIDENCE:
            return f"Low confidence: {details.get('confidence', 0):.2f} < {details.get('threshold', 0.5)}"
        
        elif alert_type == AlertType.HIGH_ERROR_RATE:
            return f"High error: {details.get('error', 0):.2f} > {details.get('threshold', 0.3)}"
        
        elif alert_type == AlertType.DRIFT_DETECTED:
            return f"Drift detected: score={details.get('drift_score', 0):.2f}"
        
        elif alert_type == AlertType.PERFORMANCE_DEGRADATION:
            return f"Performance degraded by {details.get('degradation_pct', 0):.1f}%"
        
        return f"Alert: {alert_type.value}"
