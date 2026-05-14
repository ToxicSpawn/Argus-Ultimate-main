# pyright: reportMissingImports=false
"""
Classical-Quantum Interface for Reinforcement Learning.

This module provides:
- Classical pre-processing layer for state normalization and feature engineering
- Classical post-processing layer for action mapping
- Hybrid training loop coordinator
- Fallback mechanisms for quantum circuit failures
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ============================================================================
# Classical Pre-processing Layer
# ============================================================================

class NormalizationMethod(Enum):
    """Methods for state normalization."""
    MIN_MAX = auto()
    Z_SCORE = auto()
    ROBUST = auto()
    BATCH_NORM = auto()


@dataclass
class PreprocessingConfig:
    """Configuration for state pre-processing."""
    normalization: NormalizationMethod = NormalizationMethod.Z_SCORE
    clip_values: bool = True
    clip_min: float = -10.0
    clip_max: float = 10.0
    feature_scaling: bool = True
    use_moving_stats: bool = True
    moving_avg_window: int = 100


class ClassicalStatePreprocessor:
    """Pre-processes classical states before quantum encoding."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        
        # Running statistics for normalization
        self.mean: Optional[NDArray[np.float64]] = None
        self.std: Optional[NDArray[np.float64]] = None
        self.min_vals: Optional[NDArray[np.float64]] = None
        self.max_vals: Optional[NDArray[np.float64]] = None
        self.count = 0
        
        # Moving average buffer
        self.state_buffer: List[NDArray[np.float64]] = []
    
    def preprocess(self, state: NDArray[np.float64], training: bool = True) -> NDArray[np.float64]:
        """Pre-process a state for quantum encoding."""
        processed = state.copy()
        
        # Update statistics if training
        if training:
            self._update_statistics(state)
        
        # Apply normalization
        processed = self._normalize(processed)
        
        # Clip values
        if self.config.clip_values:
            processed = np.clip(processed, self.config.clip_min, self.config.clip_max)
        
        return processed
    
    def _update_statistics(self, state: NDArray[np.float64]) -> None:
        """Update running statistics."""
        self.count += 1
        
        if self.mean is None:
            self.mean = state.copy()
            self.std = np.ones_like(state)
            self.min_vals = state.copy()
            self.max_vals = state.copy()
        else:
            # Update moving average
            alpha = 1.0 / min(self.count, self.config.moving_avg_window)
            self.mean = (1 - alpha) * self.mean + alpha * state
            
            # Update std (Welford's algorithm approximation)
            variance = (state - self.mean) ** 2
            self.std = np.sqrt((1 - alpha) * self.std ** 2 + alpha * variance)
            self.std = np.maximum(self.std, 1e-8)  # Avoid division by zero
            
            # Update min/max
            self.min_vals = np.minimum(self.min_vals, state)
            self.max_vals = np.maximum(self.max_vals, state)
    
    def _normalize(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply normalization based on configuration."""
        if self.config.normalization == NormalizationMethod.Z_SCORE:
            if self.mean is not None and self.std is not None:
                return (state - self.mean) / self.std
        
        elif self.config.normalization == NormalizationMethod.MIN_MAX:
            if self.min_vals is not None and self.max_vals is not None:
                range_vals = self.max_vals - self.min_vals
                range_vals = np.maximum(range_vals, 1e-8)
                return (state - self.min_vals) / range_vals * 2 - 1  # Scale to [-1, 1]
        
        elif self.config.normalization == NormalizationMethod.ROBUST:
            if self.mean is not None and self.std is not None:
                # Robust scaling using median approximation (mean as proxy)
                median = self.mean
                mad = self.std * 0.6745  # Approximate MAD from std
                mad = np.maximum(mad, 1e-8)
                return (state - median) / mad
        
        return state
    
    def reset_statistics(self) -> None:
        """Reset running statistics."""
        self.mean = None
        self.std = None
        self.min_vals = None
        self.max_vals = None
        self.count = 0
        self.state_buffer.clear()


# ============================================================================
# Classical Post-processing Layer
# ============================================================================

class ActionMapping(Enum):
    """Methods for mapping quantum outputs to actions."""
    ARGMAX = auto()
    SOFTMAX_THRESHOLD = auto()
    STOCHASTIC = auto()
    DETERMINISTIC = auto()


@dataclass
class PostprocessingConfig:
    """Configuration for action post-processing."""
    mapping: ActionMapping = ActionMapping.ARGMAX
    temperature: float = 1.0
    min_probability: float = 0.01
    action_smoothing: bool = True
    smoothing_factor: float = 0.1


class ClassicalActionPostprocessor:
    """Post-processes quantum outputs into actionable decisions."""
    
    def __init__(self, config: Optional[PostprocessingConfig] = None):
        self.config = config or PostprocessingConfig()
        self.previous_action: Optional[int] = None
    
    def postprocess(
        self,
        quantum_output: NDArray[np.float64],
        action_dim: int,
        training: bool = True
    ) -> Tuple[int, NDArray[np.float64]]:
        """Post-process quantum output to get action."""
        # Ensure correct dimension
        if len(quantum_output) > action_dim:
            processed_output = quantum_output[:action_dim]
        elif len(quantum_output) < action_dim:
            processed_output = np.pad(quantum_output, (0, action_dim - len(quantum_output)))
        else:
            processed_output = quantum_output
        
        # Apply softmax with temperature
        if self.config.temperature > 0:
            logits = processed_output / self.config.temperature
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
        else:
            probabilities = processed_output / (np.sum(processed_output) + 1e-10)
        
        # Select action based on mapping
        if self.config.mapping == ActionMapping.ARGMAX:
            action = int(np.argmax(probabilities))
        
        elif self.config.mapping == ActionMapping.SOFTMAX_THRESHOLD:
            # Only select actions above threshold
            valid_actions = np.where(probabilities >= self.config.min_probability)[0]
            if len(valid_actions) > 0:
                action = int(np.random.choice(valid_actions, p=probabilities[valid_actions] / np.sum(probabilities[valid_actions])))
            else:
                action = int(np.argmax(probabilities))
        
        elif self.config.mapping == ActionMapping.STOCHASTIC:
            if training:
                action = np.random.choice(action_dim, p=probabilities)
            else:
                action = int(np.argmax(probabilities))
        
        else:  # DETERMINISTIC
            action = int(np.argmax(probabilities))
        
        # Apply action smoothing
        if self.config.action_smoothing and self.previous_action is not None:
            if random.random() < self.config.smoothing_factor:
                action = self.previous_action
        
        self.previous_action = action
        
        return action, probabilities


import random


# ============================================================================
# Hybrid Training Loop Coordinator
# ============================================================================

class TrainingPhase(Enum):
    """Phases of hybrid training."""
    CLASSICAL_WARMUP = auto()
    QUANTUM_TRAINING = auto()
    HYBRID_FINE_TUNING = auto()
    EVALUATION = auto()


@dataclass
class TrainingConfig:
    """Configuration for hybrid training loop."""
    classical_warmup_episodes: int = 100
    quantum_training_episodes: int = 500
    hybrid_finetuning_episodes: int = 200
    evaluation_interval: int = 50
    save_interval: int = 100
    max_grad_norm: float = 1.0
    gradient_clip: bool = True


class HybridTrainingLoop:
    """Coordinates hybrid quantum-classical training."""
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        quantum_step_fn: Optional[Callable] = None,
        classical_step_fn: Optional[Callable] = None,
        evaluation_fn: Optional[Callable] = None
    ):
        self.config = config or TrainingConfig()
        self.quantum_step_fn = quantum_step_fn
        self.classical_step_fn = classical_step_fn
        self.evaluation_fn = evaluation_fn
        
        self.current_phase = TrainingPhase.CLASSICAL_WARMUP
        self.episode = 0
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Phase trackers
        self.phase_start_episode = 0
        self.best_performance = float('-inf')
    
    def get_current_phase(self) -> TrainingPhase:
        """Get current training phase."""
        if self.episode < self.config.classical_warmup_episodes:
            return TrainingPhase.CLASSICAL_WARMUP
        elif self.episode < self.config.classical_warmup_episodes + self.config.quantum_training_episodes:
            return TrainingPhase.QUANTUM_TRAINING
        elif self.episode < (
            self.config.classical_warmup_episodes + 
            self.config.quantum_training_episodes + 
            self.config.hybrid_finetuning_episodes
        ):
            return TrainingPhase.HYBRID_FINE_TUNING
        else:
            return TrainingPhase.EVALUATION
    
    def execute_episode(
        self,
        env: Any,
        agent: Any,
        episode_idx: int
    ) -> Dict[str, Any]:
        """Execute one episode of training."""
        self.episode = episode_idx
        self.current_phase = self.get_current_phase()
        
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0.0
        episode_loss = 0.0
        step_count = 0
        done = False
        
        while not done:
            # Select action based on phase
            if self.current_phase == TrainingPhase.CLASSICAL_WARMUP:
                action, _ = agent.select_action(state, use_quantum=False, training=True)
            elif self.current_phase == TrainingPhase.QUANTUM_TRAINING:
                action, _ = agent.select_action(state, use_quantum=True, training=True)
            else:
                action, _ = agent.select_action(state, use_quantum=True, training=True)
            
            # Take step
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result
            
            # Train step
            if self.current_phase != TrainingPhase.EVALUATION:
                loss = agent.train_step(state, action, reward, next_state, done)
                episode_loss += loss
            
            state = next_state
            episode_reward += reward
            step_count += 1
        
        # Evaluation
        eval_metrics = {}
        if episode_idx % self.config.evaluation_interval == 0 and self.evaluation_fn:
            eval_metrics = self.evaluation_fn(agent, env)
        
        # Record metrics
        metrics = {
            "episode": episode_idx,
            "phase": self.current_phase.name,
            "reward": episode_reward,
            "loss": episode_loss / max(1, step_count),
            "steps": step_count,
            "eval_metrics": eval_metrics
        }
        self.metrics_history.append(metrics)
        
        # Update best performance
        if episode_reward > self.best_performance:
            self.best_performance = episode_reward
        
        return metrics
    
    def run_training(
        self,
        env: Any,
        agent: Any,
        total_episodes: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Run complete training loop."""
        total_episodes = total_episodes or (
            self.config.classical_warmup_episodes +
            self.config.quantum_training_episodes +
            self.config.hybrid_finetuning_episodes
        )
        
        logger.info(
            "Starting hybrid training for %d episodes (Classical: %d, Quantum: %d, Hybrid: %d)",
            total_episodes,
            self.config.classical_warmup_episodes,
            self.config.quantum_training_episodes,
            self.config.hybrid_finetuning_episodes
        )
        
        for episode in range(total_episodes):
            metrics = self.execute_episode(env, agent, episode)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(
                    "Episode %d/%d | Phase: %s | Reward: %.4f | Loss: %.4f",
                    episode + 1, total_episodes,
                    metrics["phase"], metrics["reward"], metrics["loss"]
                )
            
            # Save checkpoint
            if (episode + 1) % self.config.save_interval == 0:
                self._save_checkpoint(agent, episode)
        
        logger.info("Training completed. Best reward: %.4f", self.best_performance)
        
        return self.metrics_history
    
    def _save_checkpoint(self, agent: Any, episode: int) -> None:
        """Save training checkpoint."""
        # This would save model state in a real implementation
        logger.debug("Checkpoint saved at episode %d", episode)


# ============================================================================
# Fallback Mechanisms
# ============================================================================

class FallbackReason(Enum):
    """Reasons for fallback to classical."""
    QUANTUM_CIRCUIT_ERROR = auto()
    QUANTUM_TIMEOUT = auto()
    QUANTUM_ADVANTAGE_LOW = auto()
    RESOURCE_EXHAUSTED = auto()
    HARDWARE_UNAVAILABLE = auto()


class QuantumClassicalFallback:
    """Manages fallback from quantum to classical components."""
    
    def __init__(
        self,
        fallback_threshold: float = 0.1,
        max_failures: int = 3,
        cooldown_episodes: int = 10
    ):
        self.fallback_threshold = fallback_threshold
        self.max_failures = max_failures
        self.cooldown_episodes = cooldown_episodes
        
        self.failure_count = 0
        self.use_quantum = True
        self.cooldown_until = 0
        self.failure_history: List[Tuple[FallbackReason, float]] = []
    
    def check_fallback(
        self,
        quantum_performance: float,
        classical_performance: float,
        circuit_error: Optional[float] = None
    ) -> Tuple[bool, Optional[FallbackReason]]:
        """Check if fallback to classical is needed."""
        # Check cooldown
        if self.cooldown_until > 0:
            self.cooldown_until -= 1
            return False, None
        
        # Check quantum advantage
        if classical_performance > 0:
            advantage = (quantum_performance - classical_performance) / abs(classical_performance)
            if advantage < -self.fallback_threshold:
                self._record_failure(FallbackReason.QUANTUM_ADVANTAGE_LOW, advantage)
                return True, FallbackReason.QUANTUM_ADVANTAGE_LOW
        
        # Check circuit error
        if circuit_error is not None and circuit_error > 0.1:
            self._record_failure(FallbackReason.QUANTUM_CIRCUIT_ERROR, circuit_error)
            return True, FallbackReason.QUANTUM_CIRCUIT_ERROR
        
        # Check failure count
        if self.failure_count >= self.max_failures:
            self._record_failure(FallbackReason.RESOURCE_EXHAUSTED, self.failure_count)
            return True, FallbackReason.RESOURCE_EXHAUSTED
        
        return False, None
    
    def _record_failure(self, reason: FailureReason, value: float) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.failure_history.append((reason, value))
        
        # Set cooldown if too many failures
        if self.failure_count >= self.max_failures:
            self.cooldown_until = self.cooldown_episodes
            self.failure_count = 0
            logger.warning("Too many failures, entering cooldown for %d episodes", self.cooldown_episodes)
    
    def enable_quantum(self) -> None:
        """Enable quantum components."""
        self.use_quantum = True
        logger.info("Quantum components enabled")
    
    def disable_quantum(self, reason: FailureReason) -> None:
        """Disable quantum components and fall back to classical."""
        self.use_quantum = False
        logger.warning("Quantum components disabled, falling back to classical: %s", reason.name)
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback statistics."""
        return {
            "use_quantum": self.use_quantum,
            "failure_count": self.failure_count,
            "cooldown_until": self.cooldown_until,
            "total_failures": len(self.failure_history),
            "failure_reasons": [r.name for r, _ in self.failure_history]
        }


# ============================================================================
# Quantum Circuit Error Handler
# ============================================================================

class QuantumErrorHandler:
    """Handles errors in quantum circuit execution."""
    
    def __init__(self, max_retries: int = 3, timeout_ms: float = 1000):
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms
        self.error_count = 0
        self.last_error: Optional[Exception] = None
    
    def execute_with_retry(
        self,
        circuit_fn: Callable,
        fallback_fn: Callable,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Execute quantum circuit with retry and fallback."""
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                result = circuit_fn(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                
                if elapsed_ms > self.timeout_ms:
                    logger.warning("Quantum circuit timeout: %.2f ms", elapsed_ms)
                    if attempt == self.max_retries - 1:
                        return fallback_fn(*args, **kwargs)
                    continue
                
                self.error_count = 0
                return result
                
            except Exception as e:
                self.error_count += 1
                self.last_error = e
                logger.warning("Quantum circuit error (attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                
                if attempt == self.max_retries - 1:
                    logger.error("Max retries reached, using classical fallback")
                    return fallback_fn(*args, **kwargs)
        
        return fallback_fn(*args, **kwargs)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_count": self.error_count,
            "last_error": str(self.last_error) if self.last_error else None,
            "max_retries": self.max_retries,
            "timeout_ms": self.timeout_ms
        }


__all__ = [
    # Pre-processing
    "ClassicalStatePreprocessor",
    "PreprocessingConfig",
    "NormalizationMethod",
    
    # Post-processing
    "ClassicalActionPostprocessor",
    "PostprocessingConfig",
    "ActionMapping",
    
    # Training loop
    "HybridTrainingLoop",
    "TrainingConfig",
    "TrainingPhase",
    
    # Fallback
    "QuantumClassicalFallback",
    "FallbackReason",
    "QuantumErrorHandler"
]