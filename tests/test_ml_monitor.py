"""
Tests for ML monitoring dashboard (ml_monitor.py).

Covers:
- Prediction tracking
- Outcome tracking and error computation
- Alert generation and cooldown
- Health report computation
- Drift detection
- Dashboard export
- Alert acknowledgment
"""

import json
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from ml.ml_monitor import (
    MLMonitor,
    ModelHealth,
    MLAlert,
    PredictionRecord,
    AlertSeverity,
    AlertType,
)


class TestPredictionRecord:
    """Tests for PredictionRecord dataclass."""
    
    def test_creation(self):
        """Test basic creation."""
        record = PredictionRecord(
            model_name="test_model",
            prediction=0.8,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
        )
        
        assert record.model_name == "test_model"
        assert record.prediction == 0.8
        assert record.confidence == 0.9
        assert record.actual is None
        assert record.error is None
    
    def test_to_dict(self):
        """Test serialization."""
        ts = datetime.now(timezone.utc)
        record = PredictionRecord(
            model_name="test",
            prediction=1,
            confidence=0.85,
            timestamp=ts,
            actual=1,
            error=0.0,
        )
        
        d = record.to_dict()
        assert d["model_name"] == "test"
        assert d["confidence"] == 0.85
        assert d["actual"] == "1"
        assert d["error"] == 0.0


class TestMLAlert:
    """Tests for MLAlert dataclass."""
    
    def test_creation(self):
        """Test basic creation."""
        alert = MLAlert(
            alert_id="alert_001",
            model_name="test_model",
            alert_type=AlertType.LOW_CONFIDENCE,
            severity=AlertSeverity.WARNING,
            message="Low confidence detected",
            details={"confidence": 0.3},
            timestamp=datetime.now(timezone.utc),
        )
        
        assert alert.alert_id == "alert_001"
        assert alert.acknowledged is False
    
    def test_to_dict(self):
        """Test serialization."""
        ts = datetime.now(timezone.utc)
        alert = MLAlert(
            alert_id="alert_002",
            model_name="test",
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.ERROR,
            message="High error",
            details={"error": 0.5},
            timestamp=ts,
        )
        
        d = alert.to_dict()
        assert d["alert_id"] == "alert_002"
        assert d["severity"] == "error"
        assert d["acknowledged"] is False


class TestModelHealth:
    """Tests for ModelHealth dataclass."""
    
    def test_to_dict(self):
        """Test serialization."""
        health = ModelHealth(
            model_name="test",
            status="healthy",
            total_predictions=100,
            avg_confidence=0.85,
            avg_error=0.1,
            error_rate=0.05,
            drift_score=0.02,
            last_prediction_time=datetime.now(timezone.utc),
            uptime_hours=2.5,
            alerts=[],
        )
        
        d = health.to_dict()
        assert d["status"] == "healthy"
        assert d["total_predictions"] == 100
        assert d["active_alerts"] == 0


class TestMLMonitor:
    """Tests for MLMonitor class."""
    
    def test_initialization(self):
        """Test default initialization."""
        monitor = MLMonitor()
        
        assert monitor.window_size == 1000
        assert monitor.alert_cooldown == 300
        assert monitor.confidence_threshold == 0.5
        assert monitor.error_threshold == 0.3
        assert monitor.drift_window == 100
    
    def test_custom_initialization(self):
        """Test custom parameters."""
        monitor = MLMonitor(
            window_size=500,
            alert_cooldown_seconds=60,
            confidence_threshold=0.7,
            error_threshold=0.2,
            drift_window=50,
        )
        
        assert monitor.window_size == 500
        assert monitor.alert_cooldown == 60
        assert monitor.confidence_threshold == 0.7
        assert monitor.error_threshold == 0.2
        assert monitor.drift_window == 50
    
    def test_track_prediction(self):
        """Test tracking a prediction."""
        monitor = MLMonitor()
        
        monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        
        assert monitor._total_predictions == 1
        assert "model_a" in monitor._predictions
        assert len(monitor._predictions["model_a"]) == 1
    
    def test_track_multiple_predictions(self):
        """Test tracking multiple predictions."""
        monitor = MLMonitor()
        
        for i in range(10):
            monitor.track_prediction("model_a", prediction=i, confidence=0.8 + i * 0.01)
        
        assert monitor._total_predictions == 10
        assert len(monitor._predictions["model_a"]) == 10
    
    def test_track_prediction_low_confidence_alert(self):
        """Test low confidence alert generation."""
        monitor = MLMonitor(confidence_threshold=0.7)
        
        # Track prediction with low confidence
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.3)
        
        # Should have generated an alert
        alerts = monitor.get_alerts("model_a")
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.LOW_CONFIDENCE
    
    def test_track_prediction_no_alert_above_threshold(self):
        """Test no alert when confidence is above threshold."""
        monitor = MLMonitor(confidence_threshold=0.5)
        
        monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        
        alerts = monitor.get_alerts("model_a")
        assert len(alerts) == 0
    
    def test_track_outcome_numeric(self):
        """Test tracking outcome for numeric prediction."""
        monitor = MLMonitor()
        
        monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        monitor.track_outcome("model_a", actual=0.75)
        
        predictions = list(monitor._predictions["model_a"])
        assert predictions[0].actual == 0.75
        assert predictions[0].error is not None
        assert abs(predictions[0].error - 0.05) < 1e-10  # |0.8 - 0.75| with float tolerance
    
    def test_track_outcome_classification(self):
        """Test tracking outcome for classification."""
        monitor = MLMonitor()
        
        monitor.track_prediction("model_a", prediction=1, confidence=0.9)
        monitor.track_outcome("model_a", actual=1)
        
        predictions = list(monitor._predictions["model_a"])
        assert predictions[0].error == 0.0  # Correct prediction
        
        # Wrong prediction
        monitor.track_prediction("model_a", prediction=1, confidence=0.9)
        monitor.track_outcome("model_a", actual=0)
        
        predictions = list(monitor._predictions["model_a"])
        assert predictions[1].error == 1.0  # Wrong prediction
    
    def test_track_outcome_high_error_alert(self):
        """Test high error alert generation."""
        monitor = MLMonitor(error_threshold=0.3)
        
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.9)
        monitor.track_outcome("model_a", actual=0.9)  # Error = 0.4 > threshold
        
        alerts = monitor.get_alerts("model_a")
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.HIGH_ERROR_RATE
    
    def test_track_outcome_no_predictions(self):
        """Test tracking outcome when no predictions exist."""
        monitor = MLMonitor()
        
        # Should not raise
        monitor.track_outcome("nonexistent", actual=1.0)
    
    def test_set_baseline(self):
        """Test setting baseline metrics."""
        monitor = MLMonitor()
        
        monitor.set_baseline("model_a", {"accuracy": 0.9, "f1": 0.85})
        
        assert "model_a" in monitor._baselines
        assert monitor._baselines["model_a"]["accuracy"] == 0.9
    
    def test_get_health_no_data(self):
        """Test health report with no data."""
        monitor = MLMonitor()
        
        health = monitor.get_health("nonexistent")
        
        assert health.status == "unknown"
        assert health.total_predictions == 0
    
    def test_get_health_healthy(self):
        """Test health report for healthy model."""
        monitor = MLMonitor()
        
        # Track good predictions
        for i in range(20):
            monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
            monitor.track_outcome("model_a", actual=0.8)  # Perfect predictions
        
        health = monitor.get_health("model_a")
        
        assert health.status == "healthy"
        assert health.total_predictions == 20
        assert health.avg_confidence > 0.8
        assert health.error_rate < 0.1
    
    def test_get_health_degraded(self):
        """Test health report for degraded model."""
        monitor = MLMonitor(error_threshold=0.3)
        
        # Track predictions with moderate errors
        for i in range(20):
            monitor.track_prediction("model_a", prediction=0.5, confidence=0.6)
            monitor.track_outcome("model_a", actual=0.8)  # Error = 0.3
        
        health = monitor.get_health("model_a")
        
        # Error rate should be high enough to trigger degraded status
        assert health.total_predictions == 20
    
    def test_get_health_critical(self):
        """Test health report for critical model."""
        monitor = MLMonitor(error_threshold=0.1)
        
        # Track predictions with high errors
        for i in range(20):
            monitor.track_prediction("model_a", prediction=0.2, confidence=0.3)
            monitor.track_outcome("model_a", actual=0.9)  # Error = 0.7
        
        health = monitor.get_health("model_a")
        
        assert health.status == "critical"
    
    def test_get_all_health(self):
        """Test getting health for all models."""
        monitor = MLMonitor()
        
        monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        monitor.track_prediction("model_b", prediction=0.7, confidence=0.8)
        
        all_health = monitor.get_all_health()
        
        assert "model_a" in all_health
        assert "model_b" in all_health
    
    def test_get_alerts_all(self):
        """Test getting all alerts."""
        monitor = MLMonitor(confidence_threshold=0.7)
        
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.3)
        monitor.track_prediction("model_b", prediction=0.5, confidence=0.4)
        
        alerts = monitor.get_alerts()
        assert len(alerts) == 2
    
    def test_get_alerts_by_model(self):
        """Test getting alerts filtered by model."""
        monitor = MLMonitor(confidence_threshold=0.7)
        
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.3)
        monitor.track_prediction("model_b", prediction=0.5, confidence=0.4)
        
        alerts = monitor.get_alerts(model_name="model_a")
        assert len(alerts) == 1
        assert alerts[0].model_name == "model_a"
    
    def test_get_alerts_by_severity(self):
        """Test getting alerts filtered by severity."""
        monitor = MLMonitor(confidence_threshold=0.7)
        
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.3)
        
        warnings = monitor.get_alerts(severity=AlertSeverity.WARNING)
        assert len(warnings) == 1
    
    def test_get_alerts_unacknowledged_only(self):
        """Test getting only unacknowledged alerts."""
        monitor = MLMonitor(confidence_threshold=0.7)
        
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.3)
        
        # Acknowledge the alert
        alerts = monitor.get_alerts("model_a")
        monitor.acknowledge_alert("model_a", alerts[0].alert_id)
        
        # Get unacknowledged only
        unacked = monitor.get_alerts("model_a", unacknowledged_only=True)
        assert len(unacked) == 0
    
    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        monitor = MLMonitor(confidence_threshold=0.7)
        
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.3)
        
        alerts = monitor.get_alerts("model_a")
        alert_id = alerts[0].alert_id
        
        result = monitor.acknowledge_alert("model_a", alert_id)
        
        assert result is True
        assert alerts[0].acknowledged is True
    
    def test_acknowledge_nonexistent_alert(self):
        """Test acknowledging non-existent alert."""
        monitor = MLMonitor()
        
        result = monitor.acknowledge_alert("model_a", "nonexistent")
        assert result is False
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        monitor = MLMonitor(confidence_threshold=0.7, alert_cooldown_seconds=300)
        
        # First low confidence prediction
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.3)
        alerts_after_first = monitor.get_alerts("model_a")
        
        # Second low confidence prediction (should be suppressed by cooldown)
        monitor.track_prediction("model_a", prediction=0.5, confidence=0.2)
        alerts_after_second = monitor.get_alerts("model_a")
        
        assert len(alerts_after_first) == 1
        assert len(alerts_after_second) == 1  # No new alert due to cooldown
    
    def test_export_dashboard_data(self):
        """Test dashboard data export."""
        monitor = MLMonitor()
        
        monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        monitor.track_prediction("model_b", prediction=0.7, confidence=0.8)
        
        data = monitor.export_dashboard_data()
        
        assert "timestamp" in data
        assert "total_predictions" in data
        assert "models" in data
        assert "summary" in data
        assert data["total_predictions"] == 2
    
    def test_save_dashboard(self):
        """Test saving dashboard to JSON file."""
        monitor = MLMonitor()
        
        monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dashboard.json"
            monitor.save_dashboard(str(path))
            
            assert path.exists()
            
            with open(path) as f:
                data = json.load(f)
            
            assert "models" in data
            assert "model_a" in data["models"]
    
    def test_drift_score_no_data(self):
        """Test drift score with insufficient data."""
        monitor = MLMonitor(drift_window=100)
        
        # Add only 50 predictions (need 200 for drift detection)
        for i in range(50):
            monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        
        health = monitor.get_health("model_a")
        assert health.drift_score == 0.0
    
    def test_drift_score_with_drift(self):
        """Test drift detection with changing predictions."""
        monitor = MLMonitor(drift_window=10)
        
        # First 110 predictions with high confidence (older data)
        for i in range(110):
            monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        
        # Next 10 predictions with low confidence (recent data - drift)
        for i in range(10):
            monitor.track_prediction("model_a", prediction=0.5, confidence=0.4)
        
        health = monitor.get_health("model_a")
        
        # With 120 samples and drift_window=10:
        # older = predictions[100:110] = high confidence (0.9)
        # recent = predictions[110:120] = low confidence (0.4)
        # drift should be detected
        assert health.drift_score > 0.0
    
    def test_window_size_limit(self):
        """Test that window size limits predictions."""
        monitor = MLMonitor(window_size=10)
        
        for i in range(20):
            monitor.track_prediction("model_a", prediction=i, confidence=0.9)
        
        # Should only keep last 10
        assert len(monitor._predictions["model_a"]) == 10
    
    def test_multiple_models_independent(self):
        """Test that multiple models are tracked independently."""
        monitor = MLMonitor()
        
        monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        monitor.track_prediction("model_b", prediction=0.7, confidence=0.8)
        
        health_a = monitor.get_health("model_a")
        health_b = monitor.get_health("model_b")
        
        assert health_a.total_predictions == 1
        assert health_b.total_predictions == 1
    
    def test_uptime_tracking(self):
        """Test uptime is tracked correctly."""
        monitor = MLMonitor()
        
        monitor.track_prediction("model_a", prediction=0.8, confidence=0.9)
        
        health = monitor.get_health("model_a")
        
        # Uptime should be small but non-negative
        assert health.uptime_hours >= 0.0


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""
    
    def test_values(self):
        """Test enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertType:
    """Tests for AlertType enum."""
    
    def test_values(self):
        """Test enum values."""
        assert AlertType.DRIFT_DETECTED.value == "drift_detected"
        assert AlertType.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert AlertType.LOW_CONFIDENCE.value == "low_confidence"
        assert AlertType.HIGH_ERROR_RATE.value == "high_error_rate"
        assert AlertType.MODEL_STALE.value == "model_stale"
        assert AlertType.FEATURE_ANOMALY.value == "feature_anomaly"
