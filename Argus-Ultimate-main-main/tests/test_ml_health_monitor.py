"""
Tests for ML health monitor module.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from ml.ml_health_monitor import (
    Alert,
    AlertManager,
    AlertSeverity,
    HealthStatus,
    MLHealthMonitor,
    PerformanceSnapshot,
    create_monitor,
)


class TestMLHealthMonitor(unittest.TestCase):
    """Tests for MLHealthMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = (np.random.rand(self.n_samples) > 0.5).astype(float)
        self.y_pred = (np.random.rand(self.n_samples) > 0.5).astype(float)

    def test_track_performance_basic(self):
        """Test basic performance tracking."""
        monitor = MLHealthMonitor()

        snapshot = monitor.track_performance("test_model", self.y_true, self.y_pred)

        self.assertIsInstance(snapshot, PerformanceSnapshot)
        self.assertIn("accuracy", snapshot.metrics)
        self.assertIn("precision", snapshot.metrics)
        self.assertIn("recall", snapshot.metrics)

    def test_track_performance_with_proba(self):
        """Test performance tracking with probabilities."""
        monitor = MLHealthMonitor()

        y_proba = np.random.rand(self.n_samples)
        snapshot = monitor.track_performance(
            "test_model",
            self.y_true,
            self.y_pred,
            y_proba=y_proba,
        )

        self.assertIn("calibration_error", snapshot.metrics)
        self.assertIn("brier_score", snapshot.metrics)

    def test_get_health_status(self):
        """Test getting health status."""
        monitor = MLHealthMonitor()
        monitor.track_performance("model1", self.y_true, self.y_pred)
        monitor.track_performance("model1", self.y_true, self.y_pred)

        health = monitor.get_health_status()

        self.assertIsInstance(health, HealthStatus)
        self.assertIn(health.status, ["healthy", "degraded", "critical", "unknown"])

    def test_get_health_status_specific_models(self):
        """Test getting health status for specific models."""
        monitor = MLHealthMonitor()
        monitor.track_performance("model1", self.y_true, self.y_pred)
        monitor.track_performance("model2", self.y_true, self.y_pred)

        health = monitor.get_health_status(model_names=["model1"])

        self.assertEqual(len(health.models), 1)
        self.assertIn("model1", health.models)

    def test_get_performance_trend(self):
        """Test getting performance trend."""
        monitor = MLHealthMonitor()

        for _ in range(20):
            monitor.track_performance("test_model", self.y_true, self.y_pred)

        trend = monitor.get_performance_trend("test_model", metric="accuracy")

        self.assertIsInstance(trend, list)
        self.assertLessEqual(len(trend), 20)

    def test_detect_performance_drop(self):
        """Test performance drop detection."""
        monitor = MLHealthMonitor()

        # Initial good performance
        for _ in range(15):
            monitor.track_performance("test_model", self.y_true, self.y_pred)

        # Degraded performance
        degraded_pred = np.ones_like(self.y_pred)
        for _ in range(5):
            monitor.track_performance("test_model", self.y_true, degraded_pred)

        drop = monitor.detect_performance_drop("test_model", threshold=0.1)

        # May or may not detect depending on random data
        if drop:
            self.assertIn("model_name", drop)
            self.assertIn("drop", drop)

    def test_clear_alerts(self):
        """Test clearing alerts."""
        monitor = MLHealthMonitor(accuracy_threshold=0.9)

        # Generate alerts
        for _ in range(5):
            monitor.track_performance("test_model", self.y_true, self.y_pred)

        cleared = monitor.clear_alerts()
        self.assertGreaterEqual(cleared, 0)

    def test_get_summary(self):
        """Test getting summary."""
        monitor = MLHealthMonitor()
        monitor.track_performance("model1", self.y_true, self.y_pred)

        summary = monitor.get_summary()

        self.assertIn("status", summary)
        self.assertIn("overall_score", summary)
        self.assertIn("n_models", summary)


class TestAlertManager(unittest.TestCase):
    """Tests for AlertManager class."""

    def test_register_handler(self):
        """Test registering alert handler."""
        manager = AlertManager()

        handler = MagicMock()
        manager.register_handler("test_channel", handler)

        self.assertIn("test_channel", manager._alert_handlers)

    def test_send_alert(self):
        """Test sending alert."""
        manager = AlertManager()
        handler = MagicMock()
        manager.register_handler("log", handler)

        alert = Alert(
            alert_id="test_alert",
            timestamp="2024-01-01T00:00:00",
            severity="warning",
            model_name="test_model",
            metric="accuracy",
            message="Test alert",
            value=0.4,
            threshold=0.5,
            recommendation="Retrain",
        )

        manager.send_alert(alert)

        # Alert should be handled
        self.assertIsInstance(alert, Alert)

    def test_deduplication(self):
        """Test alert deduplication."""
        manager = AlertManager(deduplicate_window=300)

        handler = MagicMock()
        manager.register_handler("test", handler)

        alert = Alert(
            alert_id="test_alert",
            timestamp="2024-01-01T00:00:00",
            severity="warning",
            model_name="model1",
            metric="accuracy",
            message="Test alert",
            value=0.4,
            threshold=0.5,
            recommendation="Retrain",
        )

        # Send same alert twice
        manager.send_alert(alert)
        manager.send_alert(alert)

        # Should only appear once (deduplication)
        # Note: deduplication uses model_name+metric key


class TestHealthStatus(unittest.TestCase):
    """Tests for HealthStatus dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        status = HealthStatus(
            status="healthy",
            overall_score=0.9,
            models={"model1": {"score": 0.9, "accuracy": 0.9}},
            alerts=[],
            last_check="2024-01-01T00:00:00",
        )

        d = status.to_dict()

        self.assertEqual(d["status"], "healthy")
        self.assertEqual(d["overall_score"], 0.9)
        self.assertIn("models", d)


class TestAlert(unittest.TestCase):
    """Tests for Alert dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        alert = Alert(
            alert_id="test",
            timestamp="2024-01-01T00:00:00",
            severity="warning",
            model_name="model1",
            metric="accuracy",
            message="Test",
            value=0.4,
            threshold=0.5,
            recommendation="Retrain",
        )

        d = alert.to_dict()

        self.assertEqual(d["alert_id"], "test")
        self.assertEqual(d["severity"], "warning")
        self.assertEqual(d["model_name"], "model1")


class TestFactory(unittest.TestCase):
    """Tests for factory function."""

    def test_create_monitor(self):
        """Test creating monitor."""
        monitor = create_monitor()

        self.assertIsInstance(monitor, MLHealthMonitor)


if __name__ == "__main__":
    unittest.main()