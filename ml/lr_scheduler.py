"""
Unified Learning Rate Scheduler for ML Training.

Supports all common LR schedules with warmup, tied to validation metrics.

Usage:
    from ml.lr_scheduler import UnifiedLRScheduler
    
    scheduler = UnifiedLRScheduler(
        optimizer,
        scheduler_type='cosine',
        warmup_epochs=5,
        max_epochs=100,
    )
    
    for epoch in range(100):
        train()
        val_loss = validate()
        scheduler.step(epoch, val_loss)
"""

import logging
import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LRSchedulerConfig:
    """Configuration for LR scheduler."""
    scheduler_type: str = 'cosine'  # cosine, plateau, step, exponential, warmup_cosine
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    max_epochs: int = 100
    
    # Plateau-specific
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4
    
    # Step-specific
    step_size: int = 30
    step_gamma: float = 0.1
    
    # Exponential-specific
    exp_gamma: float = 0.95
    
    # Cyclic-specific
    cyclic_base_lr: float = 1e-6
    cyclic_max_lr: float = 1e-3
    cyclic_step_size: int = 20


class UnifiedLRScheduler:
    """
    Unified learning rate scheduler with warmup.
    
    Supports: cosine annealing, ReduceLROnPlateau, step, exponential, cyclic.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Starting LR for warmup
        min_lr: Minimum learning rate
        max_epochs: Total epochs for cosine schedule
        verbose: Log LR changes
    
    Example:
        >>> scheduler = UnifiedLRScheduler(
        ...     optimizer,
        ...     scheduler_type='cosine',
        ...     warmup_epochs=5,
        ...     max_epochs=100,
        ... )
        >>> for epoch in range(100):
        ...     train()
        ...     val_loss = validate()
        ...     scheduler.step(epoch, val_loss)
        >>> print(scheduler.get_lr_history())
    """
    
    def __init__(
        self,
        optimizer,
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-6,
        min_lr: float = 1e-6,
        max_epochs: int = 100,
        verbose: bool = True,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        self.verbose = verbose
        
        # Store base LR from optimizer
        self.base_lr = optimizer.param_groups[0].get('lr', 1e-3)
        
        # Scheduler-specific params
        self.plateau_factor = kwargs.get('plateau_factor', 0.5)
        self.plateau_patience = kwargs.get('plateau_patience', 5)
        self.step_size = kwargs.get('step_size', 30)
        self.step_gamma = kwargs.get('step_gamma', 0.1)
        self.exp_gamma = kwargs.get('exp_gamma', 0.95)
        
        # Create underlying scheduler
        self._scheduler = self._create_scheduler()
        
        # Track LR history
        self.lr_history = []
        self._current_epoch = 0
        
        logger.info(f"UnifiedLRScheduler: type={scheduler_type}, warmup={warmup_epochs} epochs, base_lr={self.base_lr}")
    
    def _create_scheduler(self):
        """Create the underlying PyTorch scheduler."""
        try:
            import torch.optim.lr_scheduler as lr_scheduler
        except ImportError:
            logger.warning("PyTorch not available, using simulated scheduler")
            return None
        
        if self.scheduler_type == 'cosine':
            # Will be created on first step after warmup
            return None
        elif self.scheduler_type == 'plateau':
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.plateau_factor,
                patience=self.plateau_patience,
                min_lr=self.min_lr,
            )
        elif self.scheduler_type == 'step':
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.step_size,
                gamma=self.step_gamma,
            )
        elif self.scheduler_type == 'exponential':
            return lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.exp_gamma,
            )
        elif self.scheduler_type == 'none':
            return None
        else:
            logger.warning(f"Unknown scheduler type: {self.scheduler_type}, using cosine")
            return None
    
    def step(self, epoch: Optional[int] = None, metric: Optional[float] = None) -> float:
        """
        Step the scheduler.
        
        Args:
            epoch: Current epoch (if None, auto-increment)
            metric: Validation metric (for plateau scheduler)
            
        Returns:
            Current learning rate
        """
        if epoch is None:
            epoch = self._current_epoch
            self._current_epoch += 1
        
        # Warmup phase
        if epoch < self.warmup_epochs:
            lr = self._warmup_lr(epoch)
        else:
            lr = self._scheduled_lr(epoch, metric)
        
        # Apply LR to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Track history
        self.lr_history.append(lr)
        
        if self.verbose and epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: LR = {lr:.6f}")
        
        return lr
    
    def _warmup_lr(self, epoch: int) -> float:
        """Calculate warmup learning rate (linear)."""
        warmup_ratio = (epoch + 1) / self.warmup_epochs
        return self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * warmup_ratio
    
    def _scheduled_lr(self, epoch: int, metric: Optional[float] = None) -> float:
        """Calculate scheduled learning rate."""
        # Adjust epoch for post-warmup
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_max = self.max_epochs - self.warmup_epochs
        
        if self.scheduler_type == 'cosine' or self.scheduler_type == 'warmup_cosine':
            # Cosine annealing
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * adjusted_epoch / adjusted_max)
            )
        
        elif self.scheduler_type == 'plateau':
            # Use ReduceLROnPlateau
            if self._scheduler is not None and metric is not None:
                self._scheduler.step(metric)
            return self._get_current_lr()
        
        elif self.scheduler_type == 'step':
            if self._scheduler is not None:
                self._scheduler.step()
            return self._get_current_lr()
        
        elif self.scheduler_type == 'exponential':
            if self._scheduler is not None:
                self._scheduler.step()
            return self._get_current_lr()
        
        else:
            return self.base_lr
    
    def _get_current_lr(self) -> float:
        """Get current LR from optimizer."""
        return self.optimizer.param_groups[0].get('lr', self.base_lr)
    
    def get_lr_history(self) -> List[float]:
        """Get history of learning rates."""
        return self.lr_history
    
    def get_last_lr(self) -> float:
        """Get last learning rate."""
        if self.lr_history:
            return self.lr_history[-1]
        return self.base_lr


class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing scheduler (most popular for transformers).
    
    Linear warmup followed by cosine decay to min LR.
    
    Example:
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=100000,
        ...     min_lr=1e-6,
        ... )
        >>> for step in range(100000):
        ...     train_step()
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-6,
        verbose: bool = False,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.base_lr = optimizer.param_groups[0].get('lr', 1e-3)
        self._step_count = 0
        self.lr_history = []
    
    def step(self) -> float:
        """Step the scheduler (call after each training step)."""
        lr = self._compute_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.lr_history.append(lr)
        self._step_count += 1
        
        if self.verbose and self._step_count % 100 == 0:
            logger.info(f"Step {self._step_count}: LR = {lr:.8f}")
        
        return lr
    
    def _compute_lr(self) -> float:
        """Compute LR for current step."""
        if self._step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self._step_count + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (self._step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def get_lr_history(self) -> List[float]:
        return self.lr_history


class AdaptiveLRScheduler:
    """
    Adaptive LR based on training dynamics.
    
    Increases LR when loss is decreasing fast, decreases when plateauing.
    
    Example:
        >>> scheduler = AdaptiveLRScheduler(optimizer, target_loss_improvement=0.01)
        >>> for epoch in range(100):
        ...     loss = train_epoch()
        ...     scheduler.step(loss)
    """
    
    def __init__(
        self,
        optimizer,
        target_loss_improvement: float = 0.01,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.9,
        min_lr: float = 1e-6,
        max_lr: float = 1e-1,
        patience: int = 3,
    ):
        self.optimizer = optimizer
        self.target_improvement = target_loss_improvement
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        
        self.base_lr = optimizer.param_groups[0].get('lr', 1e-3)
        self.loss_history = []
        self.lr_history = []
        self.stagnation_count = 0
    
    def step(self, loss: float) -> float:
        """Step based on loss value."""
        current_lr = self.optimizer.param_groups[0].get('lr', self.base_lr)
        
        if self.loss_history:
            prev_loss = self.loss_history[-1]
            improvement = (prev_loss - loss) / (abs(prev_loss) + 1e-8)
            
            if improvement > self.target_improvement:
                # Good improvement - can increase LR
                new_lr = min(current_lr * self.increase_factor, self.max_lr)
                self.stagnation_count = 0
            elif improvement < -self.target_improvement * 0.5:
                # Getting worse - decrease LR
                new_lr = max(current_lr * self.decrease_factor, self.min_lr)
                self.stagnation_count += 1
            else:
                # Stagnating
                self.stagnation_count += 1
                if self.stagnation_count >= self.patience:
                    new_lr = max(current_lr * self.decrease_factor, self.min_lr)
                    self.stagnation_count = 0
                else:
                    new_lr = current_lr
        else:
            new_lr = current_lr
        
        self.loss_history.append(loss)
        self.lr_history.append(new_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return new_lr
