"""
Self-Evolution Pipeline v2.0
==============================
Continuous automatic improvement for Argus Ultimate.

Provides:
- Performance monitoring and degradation detection
- Automatic model retraining
- Strategy mutation and A/B testing
- Evolution tracking with rollback capability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EvolutionStatus(Enum):
    """Evolution status."""
    ACTIVE = "active"
    TESTING = "testing"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    RETIRED = "retired"


class ChangeType(Enum):
    """Type of evolution change."""
    PARAMETER_MUTATION = "parameter_mutation"
    MODEL_RETRAIN = "model_retrain"
    STRATEGY_UPDATE = "strategy_update"
    FEATURE_ADDITION = "feature_addition"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a component."""
    timestamp: datetime
    component_name: str
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    avg_trade_pnl: float
    total_trades: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvolutionRecord:
    """Record of an evolution change."""
    change_id: str
    timestamp: datetime
    component_name: str
    change_type: ChangeType
    description: str
    before_metrics: Optional[PerformanceMetrics]
    after_metrics: Optional[PerformanceMetrics]
    status: EvolutionStatus
    improvement_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Monitors component performance and detects degradation.
    """
    
    def __init__(
        self,
        degradation_threshold: float = 0.15,
        min_samples: int = 50
    ) -> None:
        """
        Initialize performance monitor.
        
        Args:
            degradation_threshold: Threshold for degradation detection (15%)
            min_samples: Minimum samples before declaring degradation
        """
        self.degradation_threshold = degradation_threshold
        self.min_samples = min_samples
        
        self._performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self._baselines: Dict[str, PerformanceMetrics] = {}
    
    def record_performance(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        name = metrics.component_name
        
        if name not in self._performance_history:
            self._performance_history[name] = []
        
        self._performance_history[name].append(metrics)
        
        # Keep last 1000 records
        if len(self._performance_history[name]) > 1000:
            self._performance_history[name] = self._performance_history[name][-1000:]
    
    def set_baseline(self, metrics: PerformanceMetrics) -> None:
        """Set baseline performance for a component."""
        self._baselines[metrics.component_name] = metrics
        logger.info("Baseline set for %s: sharpe=%.3f", metrics.component_name, metrics.sharpe_ratio)
    
    def check_degradation(self, component_name: str) -> Tuple[bool, float]:
        """
        Check if component has degraded.
        
        Returns:
            Tuple of (is_degraded, degradation_pct)
        """
        if component_name not in self._baselines:
            return False, 0.0
        
        if component_name not in self._performance_history:
            return False, 0.0
        
        history = self._performance_history[component_name]
        if len(history) < self.min_samples:
            return False, 0.0
        
        baseline = self._baselines[component_name]
        
        # Calculate recent performance (last N samples)
        recent = history[-self.min_samples:]
        recent_sharpe = np.mean([m.sharpe_ratio for m in recent])
        recent_win_rate = np.mean([m.win_rate for m in recent])
        
        # Compare to baseline
        sharpe_degradation = 0.0
        if baseline.sharpe_ratio != 0:
            sharpe_degradation = 1 - (recent_sharpe / baseline.sharpe_ratio)
        
        win_rate_degradation = 0.0
        if baseline.win_rate != 0:
            win_rate_degradation = 1 - (recent_win_rate / baseline.win_rate)
        
        # Use worst degradation
        max_degradation = max(sharpe_degradation, win_rate_degradation)
        
        is_degraded = max_degradation > self.degradation_threshold
        
        return is_degraded, max_degradation
    
    def get_performance_summary(self, component_name: str) -> Dict[str, Any]:
        """Get performance summary for a component."""
        if component_name not in self._performance_history:
            return {"status": "no_data"}
        
        history = self._performance_history[component_name]
        if not history:
            return {"status": "no_data"}
        
        recent = history[-100:] if len(history) > 100 else history
        
        return {
            "total_records": len(history),
            "recent_sharpe": np.mean([m.sharpe_ratio for m in recent]),
            "recent_win_rate": np.mean([m.win_rate for m in recent]),
            "recent_profit_factor": np.mean([m.profit_factor for m in recent]),
            "recent_max_drawdown": np.max([m.max_drawdown for m in recent]),
            "has_baseline": component_name in self._baselines,
        }


class AutoRetrainer:
    """
    Automatic model retraining when performance degrades.
    """
    
    def __init__(
        self,
        min_retrain_interval_hours: int = 24,
        validation_split: float = 0.2
    ) -> None:
        """
        Initialize auto-retrainer.
        
        Args:
            min_retrain_interval_hours: Minimum time between retraining
            validation_split: Fraction of data for validation
        """
        self.min_retrain_interval_hours = min_retrain_interval_hours
        self.validation_split = validation_split
        
        self._last_retrain: Dict[str, datetime] = {}
        self._retrain_count: Dict[str, int] = {}
    
    def should_retrain(self, component_name: str) -> bool:
        """Check if component should be retrained."""
        if component_name not in self._last_retrain:
            return True
        
        hours_since = (
            datetime.now() - self._last_retrain[component_name]
        ).total_seconds() / 3600
        
        return hours_since >= self.min_retrain_interval_hours
    
    def record_retrain(self, component_name: str) -> None:
        """Record that a retrain occurred."""
        self._last_retrain[component_name] = datetime.now()
        self._retrain_count[component_name] = self._retrain_count.get(component_name, 0) + 1
    
    def retrain_model(
        self,
        component_name: str,
        model: Any,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Retrain a model.
        
        Args:
            component_name: Name of component
            model: Model to retrain
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        if not self.should_retrain(component_name):
            logger.info("Skipping retrain for %s (too soon)", component_name)
            return {"status": "skipped"}
        
        logger.info("Retraining %s", component_name)
        
        # Split data if no validation set provided
        if val_data is None:
            split_idx = int(len(train_data) * (1 - self.validation_split))
            val_data = train_data[split_idx:]
            val_labels = train_labels[split_idx:]
            train_data = train_data[:split_idx]
            train_labels = train_labels[:split_idx]
        
        # Train model (simplified - in production, use actual training loop)
        if hasattr(model, 'fit'):
            model.fit(train_data, train_labels)
        
        # Evaluate
        train_metrics = self._evaluate(model, train_data, train_labels)
        val_metrics = self._evaluate(model, val_data, val_labels)
        
        self.record_retrain(component_name)
        
        return {
            "train_loss": train_metrics.get("loss", 0.0),
            "val_loss": val_metrics.get("loss", 0.0),
            "train_accuracy": train_metrics.get("accuracy", 0.0),
            "val_accuracy": val_metrics.get("accuracy", 0.0),
        }
    
    def _evaluate(
        self,
        model: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if hasattr(model, 'predict'):
            predictions = model.predict(data)
            accuracy = np.mean(predictions == labels) if len(labels) > 0 else 0.0
        else:
            accuracy = 0.0
        
        return {"accuracy": float(accuracy), "loss": 0.0}


class StrategyMutator:
    """
    Mutates strategy parameters for A/B testing.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2
    ) -> None:
        """
        Initialize strategy mutator.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Strength of mutations (std dev as fraction)
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
    
    def mutate_parameters(
        self,
        parameters: Dict[str, float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        Mutate strategy parameters.
        
        Args:
            parameters: Original parameters
            bounds: Optional bounds for each parameter
            
        Returns:
            Mutated parameters
        """
        mutated = {}
        
        for name, value in parameters.items():
            if np.random.random() < self.mutation_rate:
                # Apply mutation
                mutation = np.random.normal(0, abs(value) * self.mutation_strength)
                new_value = value + mutation
                
                # Apply bounds if provided
                if bounds and name in bounds:
                    min_val, max_val = bounds[name]
                    new_value = np.clip(new_value, min_val, max_val)
                
                mutated[name] = float(new_value)
            else:
                mutated[name] = value
        
        return mutated
    
    def crossover(
        self,
        params_a: Dict[str, float],
        params_b: Dict[str, float],
        crossover_rate: float = 0.5
    ) -> Dict[str, float]:
        """
        Crossover two parameter sets.
        
        Args:
            params_a: First parameter set
            params_b: Second parameter set
            crossover_rate: Probability of taking from params_b
            
        Returns:
            Crossover parameters
        """
        result = {}
        
        for key in params_a:
            if key in params_b:
                if np.random.random() < crossover_rate:
                    result[key] = params_b[key]
                else:
                    result[key] = params_a[key]
            else:
                result[key] = params_a[key]
        
        return result


class EvolutionTracker:
    """
    Tracks all evolution changes with rollback capability.
    """
    
    def __init__(self, max_history: int = 1000) -> None:
        """
        Initialize evolution tracker.
        
        Args:
            max_history: Maximum history records to keep
        """
        self.max_history = max_history
        self._records: List[EvolutionRecord] = []
        self._snapshots: Dict[str, Dict[str, Any]] = {}
    
    def record_change(
        self,
        component_name: str,
        change_type: ChangeType,
        description: str,
        before_metrics: Optional[PerformanceMetrics] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record an evolution change."""
        change_id = f"{component_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = EvolutionRecord(
            change_id=change_id,
            timestamp=datetime.now(),
            component_name=component_name,
            change_type=change_type,
            description=description,
            before_metrics=before_metrics,
            after_metrics=None,
            status=EvolutionStatus.TESTING,
            metadata=metadata or {}
        )
        
        self._records.append(record)
        
        # Keep history bounded
        if len(self._records) > self.max_history:
            self._records = self._records[-self.max_history:]
        
        return change_id
    
    def update_result(
        self,
        change_id: str,
        after_metrics: PerformanceMetrics,
        status: EvolutionStatus
    ) -> None:
        """Update change with results."""
        for record in self._records:
            if record.change_id == change_id:
                record.after_metrics = after_metrics
                record.status = status
                
                # Calculate improvement
                if record.before_metrics and after_metrics:
                    before_sharpe = record.before_metrics.sharpe_ratio
                    if before_sharpe != 0:
                        record.improvement_pct = (
                            (after_metrics.sharpe_ratio - before_sharpe) / abs(before_sharpe)
                        ) * 100
                
                break
    
    def save_snapshot(self, component_name: str, state: Dict[str, Any]) -> None:
        """Save snapshot for rollback."""
        self._snapshots[component_name] = {
            "state": state.copy(),
            "timestamp": datetime.now()
        }
    
    def rollback(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Rollback to last snapshot."""
        if component_name in self._snapshots:
            snapshot = self._snapshots[component_name]
            logger.info("Rolling back %s to snapshot from %s", component_name, snapshot["timestamp"])
            return snapshot["state"]
        
        logger.warning("No snapshot found for %s", component_name)
        return None
    
    def get_history(
        self,
        component_name: Optional[str] = None,
        n: int = 100
    ) -> List[EvolutionRecord]:
        """Get evolution history."""
        if component_name:
            filtered = [r for r in self._records if r.component_name == component_name]
            return filtered[-n:]
        
        return self._records[-n:]
    
    def get_success_rate(self, component_name: Optional[str] = None) -> float:
        """Get success rate of changes."""
        records = self.get_history(component_name, n=1000)
        
        if not records:
            return 0.0
        
        completed = [r for r in records if r.status in (EvolutionStatus.PROMOTED, EvolutionStatus.RETIRED)]
        promoted = [r for r in completed if r.status == EvolutionStatus.PROMOTED]
        
        return len(promoted) / len(completed) if completed else 0.0


class SelfEvolutionPipeline:
    """
    Main self-evolution pipeline for Argus.
    
    Coordinates performance monitoring, auto-retraining,
    strategy mutation, and evolution tracking.
    """
    
    def __init__(
        self,
        degradation_threshold: float = 0.15,
        min_retrain_interval_hours: int = 24
    ) -> None:
        """
        Initialize self-evolution pipeline.
        
        Args:
            degradation_threshold: Threshold for degradation detection
            min_retrain_interval_hours: Minimum time between retraining
        """
        self.performance_monitor = PerformanceMonitor(
            degradation_threshold=degradation_threshold
        )
        self.auto_retrainer = AutoRetrainer(
            min_retrain_interval_hours=min_retrain_interval_hours
        )
        self.mutator = StrategyMutator()
        self.tracker = EvolutionTracker()
        
        self._components: Dict[str, Any] = {}
        self._is_running = False
        
        logger.info("SelfEvolutionPipeline initialized")
    
    def register_component(
        self,
        name: str,
        component: Any,
        baseline_metrics: Optional[PerformanceMetrics] = None
    ) -> None:
        """
        Register a component for evolution.
        
        Args:
            name: Component name
            component: Component object
            baseline_metrics: Optional baseline metrics
        """
        self._components[name] = component
        
        if baseline_metrics:
            self.performance_monitor.set_baseline(baseline_metrics)
        
        logger.info("Registered component for evolution: %s", name)
    
    def monitor_performance(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Monitor component performance and trigger evolution if needed.
        
        Returns:
            Action taken (if any)
        """
        self.performance_monitor.record_performance(metrics)
        
        # Check for degradation
        is_degraded, degradation_pct = self.performance_monitor.check_degradation(
            metrics.component_name
        )
        
        if is_degraded:
            logger.warning(
                "Degradation detected for %s: %.1f%%",
                metrics.component_name,
                degradation_pct * 100
            )
            
            # Trigger auto-retrain
            if self.auto_retrainer.should_retrain(metrics.component_name):
                return {
                    "action": "retrain_triggered",
                    "component": metrics.component_name,
                    "degradation_pct": degradation_pct
                }
            
            return {
                "action": "degradation_detected",
                "component": metrics.component_name,
                "degradation_pct": degradation_pct,
                "message": "Too soon for retrain"
            }
        
        return {"action": "none"}
    
    def mutate_strategy(
        self,
        strategy_name: str,
        parameters: Dict[str, float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        Mutate strategy parameters for A/B testing.
        
        Args:
            strategy_name: Strategy name
            parameters: Current parameters
            bounds: Optional parameter bounds
            
        Returns:
            Mutated parameters
        """
        # Save current state for rollback
        self.tracker.save_snapshot(strategy_name, {"parameters": parameters})
        
        # Record change
        change_id = self.tracker.record_change(
            strategy_name,
            ChangeType.PARAMETER_MUTATION,
            "Parameter mutation for A/B testing",
            metadata={"original_params": parameters}
        )
        
        # Mutate
        mutated = self.mutator.mutate_parameters(parameters, bounds)
        
        logger.info("Mutated %s parameters: %s", strategy_name, change_id)
        
        return mutated
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution pipeline summary."""
        return {
            "registered_components": list(self._components.keys()),
            "total_changes": len(self.tracker._records),
            "success_rate": self.tracker.get_success_rate(),
            "recent_changes": [
                {
                    "id": r.change_id,
                    "component": r.component_name,
                    "type": r.change_type.value,
                    "status": r.status.value,
                    "improvement": r.improvement_pct
                }
                for r in self.tracker.get_history(n=10)
            ]
        }
