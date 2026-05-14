"""
Meta-Learning System v2.0
==========================
MAML-based meta-learning for rapid adaptation in Argus Ultimate.

Provides:
- Model-Agnostic Meta-Learning (MAML)
- Few-shot adaptation to new regimes/assets
- Task sampling for meta-training
- Online meta-learning with continual adaptation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A learning task for meta-learning."""
    task_id: str
    name: str
    support_data: np.ndarray  # Support set (few examples)
    support_labels: np.ndarray
    query_data: np.ndarray  # Query set (for evaluation)
    query_labels: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Result of few-shot adaptation."""
    task_id: str
    adapted_params: Dict[str, np.ndarray]
    support_loss: float
    query_loss: float
    adaptation_time_ms: float
    n_support_samples: int


@dataclass
class MetaTrainingStats:
    """Meta-training statistics."""
    epoch: int
    meta_train_loss: float
    meta_val_loss: float
    adaptation_speed: float  # Average adaptation time
    avg_query_accuracy: float


class TaskSampler:
    """
    Samples tasks for meta-learning.
    
    Creates diverse tasks from different:
    - Market regimes
    - Assets
    - Time periods
    """
    
    def __init__(
        self,
        n_support: int = 10,
        n_query: int = 20,
        task_types: Optional[List[str]] = None
    ) -> None:
        """
        Initialize task sampler.
        
        Args:
            n_support: Number of support samples per task
            n_query: Number of query samples per task
            task_types: Types of tasks to sample
        """
        self.n_support = n_support
        self.n_query = n_query
        self.task_types = task_types or ["regime_prediction", "price_direction", "volatility"]
    
    def sample_task(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        task_type: str = "price_direction"
    ) -> Task:
        """
        Sample a task from data.
        
        Args:
            data: Input features
            labels: Target labels
            task_type: Type of task
            
        Returns:
            Task with support and query sets
        """
        n_samples = len(data)
        n_total = self.n_support + self.n_query
        
        if n_samples < n_total:
            # Not enough data, use what we have
            indices = np.random.permutation(n_samples)
            support_idx = indices[:min(self.n_support, n_samples)]
            query_idx = indices[min(self.n_support, n_samples):]
        else:
            # Random split
            indices = np.random.permutation(n_samples)
            support_idx = indices[:self.n_support]
            query_idx = indices[self.n_support:self.n_support + self.n_query]
        
        task_id = f"{task_type}_{datetime.now().timestamp()}"
        
        return Task(
            task_id=task_id,
            name=task_type,
            support_data=data[support_idx],
            support_labels=labels[support_idx],
            query_data=data[query_idx],
            query_labels=labels[query_idx],
            metadata={"task_type": task_type}
        )
    
    def sample_task_batch(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 4
    ) -> List[Task]:
        """Sample a batch of tasks."""
        tasks = []
        for _ in range(batch_size):
            task_type = np.random.choice(self.task_types)
            task = self.sample_task(data, labels, task_type)
            tasks.append(task)
        return tasks


class MAML:
    """
    Model-Agnostic Meta-Learning implementation.
    
    MAML learns initial parameters that can be quickly adapted
    to new tasks with just a few gradient steps.
    
    Reference: Finn et al., "Model-Agnostic Meta-Learning" (2017)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = True  # First-order approximation for efficiency
    ) -> None:
        """
        Initialize MAML.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            inner_lr: Learning rate for inner loop (adaptation)
            meta_lr: Learning rate for outer loop (meta-learning)
            inner_steps: Number of gradient steps for adaptation
            first_order: Use first-order approximation (faster)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # Initialize meta-parameters (2-layer neural network)
        scale = 0.01
        self.meta_params = {
            "W1": np.random.randn(input_dim, hidden_dim) * scale,
            "b1": np.zeros(hidden_dim),
            "W2": np.random.randn(hidden_dim, output_dim) * scale,
            "b2": np.zeros(output_dim),
        }
        
        self._training_history: List[MetaTrainingStats] = []
    
    def forward(
        self,
        x: np.ndarray,
        params: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input data
            params: Parameters to use (defaults to meta_params)
            
        Returns:
            Output predictions
        """
        if params is None:
            params = self.meta_params
        
        # Layer 1
        z1 = x @ params["W1"] + params["b1"]
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = a1 @ params["W2"] + params["b2"]
        
        return z2
    
    def compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute MSE loss."""
        return float(np.mean((predictions - targets) ** 2))
    
    def compute_gradients(
        self,
        x: np.ndarray,
        y: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients using backpropagation.
        
        Returns gradients for all parameters.
        """
        # Forward pass with intermediate values
        z1 = x @ params["W1"] + params["b1"]
        a1 = np.maximum(0, z1)
        z2 = a1 @ params["W2"] + params["b2"]
        
        n = len(x)
        
        # Output layer gradients
        dz2 = 2 * (z2 - y) / n
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        # Hidden layer gradients
        da1 = dz2 @ params["W2"].T
        dz1 = da1 * (z1 > 0)  # ReLU derivative
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
    
    def adapt_to_task(
        self,
        task: Task,
        params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Adapt parameters to a specific task.
        
        Args:
            task: Task with support data
            params: Initial parameters (defaults to meta_params)
            
        Returns:
            Adapted parameters
        """
        if params is None:
            params = {k: v.copy() for k, v in self.meta_params.items()}
        
        # Inner loop: gradient descent on support set
        for _ in range(self.inner_steps):
            grads = self.compute_gradients(
                task.support_data,
                task.support_labels,
                params
            )
            
            # Update parameters
            for key in params:
                params[key] = params[key] - self.inner_lr * grads[key]
        
        return params
    
    def meta_train_step(
        self,
        tasks: List[Task]
    ) -> float:
        """
        Perform one meta-training step.
        
        Args:
            tasks: Batch of tasks
            
        Returns:
            Average meta-loss
        """
        meta_gradients = {k: np.zeros_like(v) for k, v in self.meta_params.items()}
        total_loss = 0.0
        
        for task in tasks:
            # Clone meta-parameters
            adapted_params = {k: v.copy() for k, v in self.meta_params.items()}
            
            # Inner loop adaptation
            for _ in range(self.inner_steps):
                grads = self.compute_gradients(
                    task.support_data,
                    task.support_labels,
                    adapted_params
                )
                for key in adapted_params:
                    adapted_params[key] = adapted_params[key] - self.inner_lr * grads[key]
            
            # Evaluate on query set
            query_preds = self.forward(task.query_data, adapted_params)
            query_loss = self.compute_loss(query_preds, task.query_labels)
            total_loss += query_loss
            
            # Compute gradients w.r.t. adapted parameters
            query_grads = self.compute_gradients(
                task.query_data,
                task.query_labels,
                adapted_params
            )
            
            # Accumulate meta-gradients
            for key in meta_gradients:
                meta_gradients[key] += query_grads[key]
        
        # Update meta-parameters
        for key in self.meta_params:
            self.meta_params[key] -= self.meta_lr * meta_gradients[key] / len(tasks)
        
        return total_loss / len(tasks)
    
    def meta_train(
        self,
        task_generator: Callable[[], List[Task]],
        n_epochs: int = 100,
        tasks_per_epoch: int = 4,
        val_generator: Optional[Callable[[], List[Task]]] = None
    ) -> List[MetaTrainingStats]:
        """
        Run meta-training loop.
        
        Args:
            task_generator: Function that generates tasks
            n_epochs: Number of training epochs
            tasks_per_epoch: Tasks per epoch
            val_generator: Optional validation task generator
            
        Returns:
            Training history
        """
        for epoch in range(n_epochs):
            # Generate tasks
            tasks = task_generator()
            
            # Meta-train step
            train_loss = self.meta_train_step(tasks)
            
            # Validation
            val_loss = 0.0
            if val_generator:
                val_tasks = val_generator()
                val_loss = self.meta_train_step(val_tasks)
            
            stats = MetaTrainingStats(
                epoch=epoch,
                meta_train_loss=train_loss,
                meta_val_loss=val_loss,
                adaptation_speed=0.0,
                avg_query_accuracy=0.0
            )
            self._training_history.append(stats)
            
            if epoch % 10 == 0:
                logger.info(
                    "Epoch %d: train_loss=%.4f, val_loss=%.4f",
                    epoch, train_loss, val_loss
                )
        
        return self._training_history


class RapidAdapter:
    """
    Rapid adaptation engine for new tasks/assets.
    
    Uses meta-learned initialization for fast few-shot learning.
    """
    
    def __init__(self, maml: MAML) -> None:
        """
        Initialize rapid adapter.
        
        Args:
            maml: Trained MAML model
        """
        self.maml = maml
        self._adaptation_cache: Dict[str, AdaptationResult] = {}
    
    def adapt(
        self,
        support_data: np.ndarray,
        support_labels: np.ndarray,
        task_id: str,
        n_steps: int = 3
    ) -> AdaptationResult:
        """
        Rapidly adapt to new task.
        
        Args:
            support_data: Few-shot examples
            support_labels: Labels for support examples
            task_id: Unique task identifier
            n_steps: Number of adaptation steps
            
        Returns:
            AdaptationResult with adapted parameters
        """
        import time
        start_time = time.time()
        
        # Create temporary task
        task = Task(
            task_id=task_id,
            name="rapid_adaptation",
            support_data=support_data,
            support_labels=support_labels,
            query_data=support_data[:1],  # Dummy query
            query_labels=support_labels[:1]
        )
        
        # Adapt
        adapted_params = self.maml.adapt_to_task(task)
        
        # Evaluate
        predictions = self.maml.forward(support_data, adapted_params)
        loss = self.maml.compute_loss(predictions, support_labels)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = AdaptationResult(
            task_id=task_id,
            adapted_params=adapted_params,
            support_loss=loss,
            query_loss=loss,  # Same for few-shot
            adaptation_time_ms=elapsed_ms,
            n_support_samples=len(support_data)
        )
        
        self._adaptation_cache[task_id] = result
        return result
    
    def predict(
        self,
        x: np.ndarray,
        task_id: str
    ) -> np.ndarray:
        """
        Make prediction using adapted parameters.
        
        Args:
            x: Input data
            task_id: Task ID to use adapted parameters
            
        Returns:
            Predictions
        """
        if task_id not in self._adaptation_cache:
            raise ValueError(f"No adaptation found for task {task_id}")
        
        adapted_params = self._adaptation_cache[task_id].adapted_params
        return self.maml.forward(x, adapted_params)


class MetaLearningSystem:
    """
    Main meta-learning system for Argus.
    
    Combines MAML, task sampling, and rapid adaptation
    for quick learning in new market conditions.
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        output_dim: int = 1
    ) -> None:
        """
        Initialize meta-learning system.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for regression)
        """
        self.maml = MAML(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            inner_lr=0.01,
            meta_lr=0.001,
            inner_steps=5,
            first_order=True
        )
        
        self.task_sampler = TaskSampler(
            n_support=10,
            n_query=20
        )
        
        self.adapter = RapidAdapter(self.maml)
        self._is_trained = False
        
        logger.info(
            "MetaLearningSystem initialized: input=%d, hidden=%d, output=%d",
            input_dim, hidden_dim, output_dim
        )
    
    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 50
    ) -> List[MetaTrainingStats]:
        """
        Train the meta-learning system.
        
        Args:
            data: Training data
            labels: Training labels
            n_epochs: Number of epochs
            
        Returns:
            Training history
        """
        def task_generator():
            return self.task_sampler.sample_task_batch(data, labels, batch_size=4)
        
        history = self.maml.meta_train(
            task_generator=task_generator,
            n_epochs=n_epochs,
            tasks_per_epoch=4
        )
        
        self._is_trained = True
        return history
    
    def adapt_to_regime(
        self,
        regime_data: np.ndarray,
        regime_labels: np.ndarray,
        regime_name: str
    ) -> AdaptationResult:
        """
        Rapidly adapt to a new market regime.
        
        Args:
            regime_data: Data from new regime
            regime_labels: Labels from new regime
            regime_name: Name of the regime
            
        Returns:
            AdaptationResult
        """
        return self.adapter.adapt(
            regime_data,
            regime_labels,
            task_id=f"regime_{regime_name}",
            n_steps=3
        )
    
    def predict(
        self,
        features: np.ndarray,
        regime_name: str
    ) -> np.ndarray:
        """
        Make prediction using regime-adapted model.
        
        Args:
            features: Input features
            regime_name: Regime to use for prediction
            
        Returns:
            Predictions
        """
        task_id = f"regime_{regime_name}"
        return self.adapter.predict(features, task_id)
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return {
            "is_trained": self._is_trained,
            "cached_adaptations": len(self.adapter._adaptation_cache),
            "training_epochs": len(self.maml._training_history),
        }
