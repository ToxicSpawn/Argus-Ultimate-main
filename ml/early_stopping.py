"""
Unified Early Stopping for ML Training.

Stops training when validation metric stops improving, restoring best weights.
Works with PyTorch, scikit-learn, and custom models.

Usage:
    from ml.early_stopping import EarlyStopping
    
    early_stop = EarlyStopping(patience=10, mode='max', min_delta=0.001)
    
    for epoch in range(100):
        train_loss = train_epoch(model)
        val_loss, val_metric = validate(model)
        
        early_stop(val_metric, model)
        
        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Best model is automatically restored
    best_model = early_stop.best_model
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    patience: int = 10           # Epochs to wait before stopping
    min_delta: float = 0.001     # Minimum improvement to reset patience
    mode: str = 'max'            # 'max' for accuracy/sharpe, 'min' for loss
    restore_best: bool = True    # Restore best weights on stop
    verbose: bool = True         # Log stopping decisions
    baseline: Optional[float] = None  # Baseline to improve upon


class EarlyStopping:
    """
    Unified early stopping for ML training.
    
    Tracks validation metric and stops training when improvement stalls.
    Automatically saves and restores best model weights.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'max' (higher is better) or 'min' (lower is better)
        restore_best: Whether to restore best weights on stop
        verbose: Whether to log stopping decisions
        baseline: Optional baseline metric to beat
    
    Example:
        >>> early_stop = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(100):
        ...     val_metric = train_and_validate()
        ...     early_stop(val_metric, model)
        ...     if early_stop.should_stop:
        ...         break
        >>> print(f"Best epoch: {early_stop.best_epoch}, Best score: {early_stop.best_score}")
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
        restore_best: bool = True,
        verbose: bool = True,
        baseline: Optional[float] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose
        self.baseline = baseline
        
        # State tracking
        self.best_score = baseline
        self.best_epoch = 0
        self.best_model = None
        self.counter = 0
        self.should_stop = False
        self.history = []
        
        # Internal
        self._initialized = False
        
        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}, min_delta={min_delta}")
    
    def __call__(
        self, 
        score: float, 
        model: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            model: Model to save if best (PyTorch, sklearn, etc.)
            epoch: Current epoch number
            metrics: Additional metrics to log
            
        Returns:
            True if training should stop
        """
        if epoch is None:
            epoch = len(self.history)
        
        self.history.append({
            'epoch': epoch,
            'score': score,
            'metrics': metrics or {},
        })
        
        # First call - initialize
        if not self._initialized:
            self.best_score = score
            self.best_epoch = epoch
            if model is not None and self.restore_best:
                self.best_model = self._save_model(model)
            self._initialized = True
            return False
        
        # Check for improvement
        improved = self._is_improvement(score)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            if model is not None and self.restore_best:
                self.best_model = self._save_model(model)
            
            if self.verbose:
                logger.info(f"Epoch {epoch}: Improved to {score:.6f}")
        else:
            self.counter += 1
            
            if self.verbose and self.counter % 5 == 0:
                logger.info(f"Epoch {epoch}: No improvement for {self.counter} epochs")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info(f"Early stopping at epoch {epoch} (best: {self.best_score:.6f} at epoch {self.best_epoch})")
        
        return self.should_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement over best."""
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        elif self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            raise ValueError(f"mode must be 'max' or 'min', got '{self.mode}'")
    
    def _save_model(self, model: Any) -> Any:
        """Save model weights/state."""
        # PyTorch model
        if hasattr(model, 'state_dict'):
            try:
                import torch
                return {k: v.clone() for k, v in model.state_dict().items()}
            except ImportError:
                pass
        
        # sklearn model
        if hasattr(model, 'get_params'):
            try:
                from sklearn.base import clone
                return clone(model)
            except ImportError:
                pass
        
        # Fallback: deep copy
        try:
            return deepcopy(model)
        except Exception:
            return None
    
    def restore(self, model: Any) -> None:
        """
        Restore best model weights.
        
        Args:
            model: Model to restore weights to
        """
        if self.best_model is None:
            logger.warning("No best model saved to restore")
            return
        
        # PyTorch
        if hasattr(model, 'load_state_dict') and isinstance(self.best_model, dict):
            model.load_state_dict(self.best_model)
            logger.info(f"Restored best model from epoch {self.best_epoch}")
        
        # sklearn
        elif hasattr(model, 'get_params') and hasattr(self.best_model, 'get_params'):
            model.set_params(**self.best_model.get_params())
            logger.info(f"Restored best model parameters from epoch {self.best_epoch}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current early stopping state."""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'should_stop': self.should_stop,
            'history': self.history,
        }
    
    def reset(self) -> None:
        """Reset early stopping state for new training."""
        self.best_score = self.baseline
        self.best_epoch = 0
        self.best_model = None
        self.counter = 0
        self.should_stop = False
        self.history = []
        self._initialized = False
        logger.info("EarlyStopping reset")


class MultiMetricEarlyStopping:
    """
    Early stopping with multiple metrics.
    
    Stops when ALL metrics fail to improve, or ANY metric fails to improve.
    
    Example:
        >>> early_stop = MultiMetricEarlyStopping(
        ...     metrics={'val_loss': 'min', 'val_sharpe': 'max'},
        ...     strategy='all'  # Stop when ALL metrics stagnate
        ... )
    """
    
    def __init__(
        self,
        metrics: Dict[str, str],  # {metric_name: mode}
        patience: int = 10,
        min_delta: float = 0.001,
        strategy: str = 'all',  # 'all' or 'any'
        restore_best: bool = True,
        verbose: bool = True,
    ):
        self.metrics = metrics
        self.patience = patience
        self.min_delta = min_delta
        self.strategy = strategy
        self.restore_best = restore_best
        self.verbose = verbose
        
        self.stoppers = {
            name: EarlyStopping(
                patience=patience,
                min_delta=min_delta,
                mode=mode,
                restore_best=False,  # Handle restoration separately
                verbose=False,
            )
            for name, mode in metrics.items()
        }
        
        self.best_model = None
        self.should_stop = False
        self.best_epoch = 0
        
        logger.info(f"MultiMetricEarlyStopping: {list(metrics.keys())}, strategy={strategy}")
    
    def __call__(
        self,
        scores: Dict[str, float],
        model: Optional[Any] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """
        Check all metrics for early stopping.
        
        Args:
            scores: Dict of {metric_name: score}
            model: Model to save if best
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        if epoch is None:
            epoch = len(list(self.stoppers.values())[0].history)
        
        # Update all stoppers
        results = []
        for name, stopper in self.stoppers.items():
            if name in scores:
                should_stop = stopper(scores[name], model=None, epoch=epoch)
                results.append(should_stop)
        
        # Check strategy
        if self.strategy == 'all':
            self.should_stop = all(results)
        elif self.strategy == 'any':
            self.should_stop = any(results)
        else:
            raise ValueError(f"strategy must be 'all' or 'any', got '{self.strategy}'")
        
        # Track best model based on primary metric (first in dict)
        primary_metric = list(self.metrics.keys())[0]
        primary_stopper = self.stoppers[primary_metric]
        
        if primary_stopper.best_epoch == epoch:
            self.best_epoch = epoch
            if model is not None and self.restore_best:
                self.best_model = primary_stopper._save_model(model)
            
            if self.verbose:
                logger.info(f"Epoch {epoch}: New best model (primary: {primary_metric}={primary_stopper.best_score:.6f})")
        
        if self.should_stop and self.verbose:
            logger.info(f"MultiMetricEarlyStopping triggered at epoch {epoch}")
            for name, stopper in self.stoppers.items():
                logger.info(f"  {name}: best={stopper.best_score:.6f} at epoch {stopper.best_epoch}")
        
        return self.should_stop
    
    def restore(self, model: Any) -> None:
        """Restore best model weights."""
        primary_metric = list(self.metrics.keys())[0]
        self.stoppers[primary_metric].restore(model)
