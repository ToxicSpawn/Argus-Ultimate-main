"""
ml/model_rolling/__init__.py
=============================
Automated Model Rolling Pipeline — safe ML model deployment with drift detection,
shadow mode, canary rollout, and automatic rollback.

Exports:
    ModelRollingPipeline, RolloutStage, DriftAlert, ShadowResult,
    CanaryReport, RollbackEvent
"""

from __future__ import annotations

from ml.model_rolling.state_tracker import RolloutStage, ModelLifecycleState, RolloutEvent
from ml.model_rolling.drift_detector import DriftAlert, PredictionSample, RollingDriftDetector
from ml.model_rolling.deployment_orchestrator import (
    ShadowResult,
    CanaryReport,
    RollbackEvent,
    DeploymentPolicy,
    ShadowMode,
    ModelRollingPipeline,
)

__all__ = [
    # State
    "RolloutStage",
    "ModelLifecycleState",
    "RolloutEvent",
    # Drift
    "DriftAlert",
    "PredictionSample",
    "RollingDriftDetector",
    # Deployment
    "ShadowResult",
    "CanaryReport",
    "RollbackEvent",
    "DeploymentPolicy",
    "ShadowMode",
    "CanaryConfig",
    "ModelRollingPipeline",
]
