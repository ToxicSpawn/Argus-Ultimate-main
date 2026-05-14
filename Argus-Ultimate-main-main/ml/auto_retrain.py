"""
Automated Retraining Pipeline — triggers model retraining based on drift, schedule, or performance.

Features:
  - Scheduled retraining (daily, weekly, etc.)
  - Drift-triggered retraining
  - Performance degradation detection
  - A/B testing of new models vs current
  - Rollback capability
  - Retraining history and audit trail

Usage:
    retrain = AutoRetrainPipeline(
        model_name="regime_classifier",
        pipeline=pipeline_orchestrator,
        retrain_fn=train_fn,
        schedule="daily",
        drift_threshold=0.15,
    )
    
    # Check if retraining is needed
    if retrain.should_retrain():
        result = retrain.execute()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RetrainReason(Enum):
    """Reason for retraining."""
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL = "manual"
    NEW_DATA = "new_data"


class RetrainStatus(Enum):
    """Retraining job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RetrainJob:
    """Retraining job record."""
    job_id: str
    model_name: str
    reason: RetrainReason
    status: RetrainStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    old_model_id: Optional[str] = None
    new_model_id: Optional[str] = None
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    improvement: float = 0.0
    rolled_back: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "reason": self.reason.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "old_model_id": self.old_model_id,
            "new_model_id": self.new_model_id,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "improvement": round(self.improvement, 4),
            "rolled_back": self.rolled_back,
            "error_message": self.error_message,
        }


@dataclass
class RetrainConfig:
    """Retraining configuration."""
    # Schedule
    schedule: str = "daily"  # hourly, daily, weekly, monthly, none
    schedule_interval_hours: int = 24
    
    # Drift triggers
    drift_threshold: float = 0.15
    consecutive_drift_count: int = 3  # Number of consecutive drifts before retrain
    
    # Performance triggers
    performance_threshold: float = 0.05  # 5% degradation triggers retrain
    min_samples_for_check: int = 100
    
    # A/B testing
    ab_test_enabled: bool = True
    ab_test_duration_hours: int = 24
    ab_test_min_samples: int = 1000
    
    # Rollback
    auto_rollback: bool = True
    rollback_threshold: float = -0.1  # Rollback if >10% worse
    
    # History
    max_history: int = 100
    history_dir: str = "retrain_history"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule": self.schedule,
            "schedule_interval_hours": self.schedule_interval_hours,
            "drift_threshold": self.drift_threshold,
            "consecutive_drift_count": self.consecutive_drift_count,
            "performance_threshold": self.performance_threshold,
            "min_samples_for_check": self.min_samples_for_check,
            "ab_test_enabled": self.ab_test_enabled,
            "ab_test_duration_hours": self.ab_test_duration_hours,
            "ab_test_min_samples": self.ab_test_min_samples,
            "auto_rollback": self.auto_rollback,
            "rollback_threshold": self.rollback_threshold,
            "max_history": self.max_history,
            "history_dir": self.history_dir,
        }


class AutoRetrainPipeline:
    """
    Automated retraining pipeline.
    
    Monitors model health and triggers retraining when:
    1. Scheduled interval has passed
    2. Drift is detected above threshold
    3. Performance degrades significantly
    4. Manual trigger is requested
    
    Features A/B testing of new models and automatic rollback if worse.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[RetrainConfig] = None,
    ):
        self.model_name = model_name
        self.config = config or RetrainConfig()
        
        # State
        self._last_retrain_time: Optional[datetime] = None
        self._drift_count: int = 0
        self._consecutive_drifts: int = 0
        self._baseline_metrics: Dict[str, float] = {}
        self._history: List[RetrainJob] = []
        self._current_model_id: Optional[str] = None
        
        # Callbacks
        self._train_fn: Optional[Callable] = None
        self._evaluate_fn: Optional[Callable] = None
        self._get_data_fn: Optional[Callable] = None
        
        # Create history directory
        self._history_dir = Path(self.config.history_dir)
        self._history_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("AutoRetrainPipeline initialized: %s (schedule=%s)", 
                    model_name, self.config.schedule)
    
    def set_callbacks(
        self,
        train_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        get_data_fn: Optional[Callable] = None,
    ) -> None:
        """Set callback functions for training, evaluation, and data retrieval."""
        if train_fn is not None:
            self._train_fn = train_fn
        if evaluate_fn is not None:
            self._evaluate_fn = evaluate_fn
        if get_data_fn is not None:
            self._get_data_fn = get_data_fn
    
    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """Set baseline metrics for comparison."""
        self._baseline_metrics = metrics.copy()
        logger.info("Baseline metrics set: %s", metrics)
    
    def set_current_model(self, model_id: str) -> None:
        """Set current model ID."""
        self._current_model_id = model_id
    
    def should_retrain(self) -> Tuple[bool, Optional[RetrainReason]]:
        """
        Check if retraining is needed.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check schedule
        if self._check_schedule():
            return True, RetrainReason.SCHEDULED
        
        # Check drift
        if self._consecutive_drifts >= self.config.consecutive_drift_count:
            return True, RetrainReason.DRIFT_DETECTED
        
        return False, None
    
    def report_drift(self, drift_score: float, drift_detected: bool) -> None:
        """Report drift detection results."""
        self._drift_count += 1
        
        if drift_detected and drift_score > self.config.drift_threshold:
            self._consecutive_drifts += 1
            logger.warning(
                "Drift reported: score=%.4f, consecutive=%d/%d",
                drift_score, self._consecutive_drifts, self.config.consecutive_drift_count,
            )
        else:
            self._consecutive_drifts = max(0, self._consecutive_drifts - 1)
    
    def report_performance(self, metrics: Dict[str, float]) -> None:
        """Report current performance metrics."""
        if not self._baseline_metrics:
            return
        
        # Check for degradation
        for key, baseline in self._baseline_metrics.items():
            if key in metrics:
                current = metrics[key]
                if baseline != 0:
                    degradation = (baseline - current) / abs(baseline)
                    if degradation > self.config.performance_threshold:
                        logger.warning(
                            "Performance degradation detected: %s baseline=%.4f current=%.4f (%.1f%%)",
                            key, baseline, current, degradation * 100,
                        )
    
    def execute(
        self,
        reason: RetrainReason = RetrainReason.MANUAL,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> RetrainJob:
        """
        Execute retraining.
        
        Args:
            reason: Reason for retraining
            features: Optional training features (if None, will use get_data_fn)
            labels: Optional training labels
            
        Returns:
            RetrainJob with results
        """
        job_id = f"retrain_{self.model_name}_{int(time.time())}"
        job = RetrainJob(
            job_id=job_id,
            model_name=self.model_name,
            reason=reason,
            status=RetrainStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            old_model_id=self._current_model_id,
        )
        
        logger.info("Starting retraining: %s (reason=%s)", job_id, reason.value)
        
        try:
            # Get data if not provided
            if features is None or labels is None:
                if self._get_data_fn is None:
                    raise ValueError("No data provided and no get_data_fn set")
                features, labels = self._get_data_fn()
            
            # Evaluate current model
            if self._evaluate_fn and self._current_model_id:
                job.metrics_before = self._evaluate_fn(features, labels)
            
            # Train new model
            if self._train_fn is None:
                raise ValueError("No train_fn set")
            
            new_model_id, train_metrics = self._train_fn(features, labels)
            job.new_model_id = new_model_id
            
            # Evaluate new model
            if self._evaluate_fn:
                job.metrics_after = self._evaluate_fn(features, labels, model_id=new_model_id)
            
            # Calculate improvement
            job.improvement = self._calculate_improvement(
                job.metrics_before, job.metrics_after
            )
            
            # Check if should rollback
            if self.config.auto_rollback and job.improvement < self.config.rollback_threshold:
                logger.warning(
                    "New model is %.1f%% worse, rolling back",
                    abs(job.improvement) * 100,
                )
                job.rolled_back = True
                job.status = RetrainStatus.ROLLED_BACK
            else:
                job.status = RetrainStatus.COMPLETED
                self._current_model_id = new_model_id
                self._consecutive_drifts = 0
                self._last_retrain_time = datetime.now(timezone.utc)
                
                # Update baseline if improved
                if job.improvement > 0:
                    self._baseline_metrics = job.metrics_after.copy()
            
            job.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            job.status = RetrainStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            logger.error("Retraining failed: %s - %s", job_id, e)
        
        # Save job
        self._history.append(job)
        self._save_job(job)
        
        # Trim history
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]
        
        return job
    
    def get_history(self, limit: int = 20) -> List[RetrainJob]:
        """Get retraining history."""
        return self._history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retraining statistics."""
        completed = [j for j in self._history if j.status == RetrainStatus.COMPLETED]
        failed = [j for j in self._history if j.status == RetrainStatus.FAILED]
        rolled_back = [j for j in self._history if j.rolled_back]
        
        avg_improvement = 0.0
        if completed:
            avg_improvement = np.mean([j.improvement for j in completed])
        
        return {
            "model_name": self.model_name,
            "total_retrains": len(self._history),
            "successful": len(completed),
            "failed": len(failed),
            "rolled_back": len(rolled_back),
            "avg_improvement": round(avg_improvement, 4),
            "last_retrain": self._last_retrain_time.isoformat() if self._last_retrain_time else None,
            "consecutive_drifts": self._consecutive_drifts,
            "drift_count": self._drift_count,
        }
    
    def _check_schedule(self) -> bool:
        """Check if scheduled retraining is due."""
        if self.config.schedule == "none":
            return False
        
        if self._last_retrain_time is None:
            return True
        
        elapsed = datetime.now(timezone.utc) - self._last_retrain_time
        return elapsed.total_seconds() > self.config.schedule_interval_hours * 3600
    
    def _calculate_improvement(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> float:
        """Calculate improvement percentage."""
        if not before or not after:
            return 0.0
        
        # Use accuracy if available, otherwise use first common metric
        key = "accuracy" if "accuracy" in before else list(before.keys())[0]
        
        if key in before and key in after:
            baseline = before[key]
            current = after[key]
            
            if baseline == 0:
                return 0.0
            
            return (current - baseline) / abs(baseline)
        
        return 0.0
    
    def _save_job(self, job: RetrainJob) -> None:
        """Save job to history file."""
        path = self._history_dir / f"{job.job_id}.json"
        with open(path, "w") as f:
            json.dump(job.to_dict(), f, indent=2)
