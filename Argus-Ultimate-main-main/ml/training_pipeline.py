"""
Unified ML Training Pipeline.

Combines all best practices into a single, reproducible training workflow:
- Data quality validation before training
- Early stopping with best weight restoration
- Learning rate scheduling with warmup
- Model registration with full metadata
- Experiment tracking and logging

Usage:
    from ml.training_pipeline import TrainingPipeline, TrainingConfig
    
    config = TrainingConfig(
        model_name="regime_classifier",
        epochs=100,
        patience=10,
        lr_scheduler="cosine",
        warmup_epochs=5,
    )
    
    pipeline = TrainingPipeline(config)
    result = pipeline.train(model, train_df, val_df)
    
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best metrics: {result.best_metrics}")
"""

import logging
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime

import numpy as np
import pandas as pd

from ml.early_stopping import EarlyStopping, EarlyStoppingConfig
from ml.lr_scheduler import UnifiedLRScheduler, LRSchedulerConfig
from ml.data_quality import DataQualityPipeline, DataQualityConfig, DataQualityReport
from ml.model_registry_enhanced import EnhancedModelRegistry, ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    
    # Model
    model_name: str = "unnamed_model"
    model_type: str = "sklearn"  # sklearn, pytorch, xgboost, lightgbm
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    early_stop_metric: str = "val_loss"
    early_stop_mode: str = "min"  # min for loss, max for accuracy/sharpe
    
    # Learning rate
    lr_scheduler: str = "cosine"  # cosine, plateau, step, none
    learning_rate: float = 0.001
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    
    # Data quality
    validate_data: bool = True
    min_samples: int = 100
    max_missing_pct: float = 0.1
    
    # Model registry
    register_model: bool = True
    auto_promote: bool = False
    promote_threshold: float = 1.0  # e.g., Sharpe > 1.0 to promote
    
    # Logging
    log_interval: int = 10  # Log every N epochs
    verbose: bool = True
    
    # Output
    save_dir: str = "data/training"
    experiment_name: Optional[str] = None


@dataclass
class TrainingResult:
    """Result from training pipeline."""
    
    success: bool
    best_epoch: int
    best_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    training_time_seconds: float
    early_stopped: bool
    model_metadata: Optional[ModelMetadata] = None
    quality_report: Optional[DataQualityReport] = None
    epoch_history: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "final_metrics": self.final_metrics,
            "training_time_seconds": self.training_time_seconds,
            "early_stopped": self.early_stopped,
            "n_epochs": len(self.epoch_history),
        }


class TrainingPipeline:
    """
    Unified ML Training Pipeline.
    
    Orchestrates the complete training workflow:
    1. Data quality validation
    2. Training loop with early stopping
    3. LR scheduling
    4. Model registration
    5. Result reporting
    
    Args:
        config: TrainingConfig with all training parameters
    
    Example:
        >>> config = TrainingConfig(
        ...     model_name="regime_classifier",
        ...     epochs=100,
        ...     patience=10,
        ... )
        >>> pipeline = TrainingPipeline(config)
        >>> 
        >>> # Train sklearn model
        >>> result = pipeline.train_sklearn(
        ...     model=xgb_model,
        ...     X_train=X_train, y_train=y_train,
        ...     X_val=X_val, y_val=y_val,
        ... )
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        
        # Initialize components
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode=self.config.early_stop_mode,
            restore_best=True,
            verbose=self.config.verbose,
        )
        
        self.data_quality = DataQualityPipeline(DataQualityConfig(
            min_samples=self.config.min_samples,
            max_missing_pct=self.config.max_missing_pct,
        ))
        
        self.registry = EnhancedModelRegistry()
        
        # Training state
        self.epoch_history: List[Dict[str, float]] = []
        self.start_time: float = 0
        
        # Create save directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TrainingPipeline initialized: model={self.config.model_name}, "
                    f"epochs={self.config.epochs}, patience={self.config.patience}")
    
    def train_sklearn(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fit_kwargs: Optional[Dict] = None,
    ) -> TrainingResult:
        """
        Train sklearn-style model with validation monitoring.
        
        Note: sklearn models don't have epochs, so we use incremental fitting
        or cross-validation for early stopping simulation.
        
        Args:
            model: sklearn-compatible model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            fit_kwargs: Additional kwargs for model.fit()
            
        Returns:
            TrainingResult
        """
        self.start_time = time.time()
        
        # 1. Data quality check
        quality_report = None
        if self.config.validate_data:
            passed, quality_report = self.data_quality.validate(X_train)
            if not passed and self.config.verbose:
                logger.warning(f"Data quality issues: {quality_report.issues}")
        
        # 2. Train model
        try:
            model.fit(X_train, y_train, **(fit_kwargs or {}))
            success = True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                best_epoch=0,
                best_metrics={},
                final_metrics={},
                training_time_seconds=time.time() - self.start_time,
                early_stopped=False,
                quality_report=quality_report,
            )
        
        # 3. Evaluate
        final_metrics = self._evaluate_sklearn(model, X_val, y_val)
        
        # 4. Register model
        model_metadata = None
        if self.config.register_model:
            model_metadata = self.registry.register(
                name=self.config.model_name,
                model=model,
                metrics=final_metrics,
                hyperparams=model.get_params() if hasattr(model, 'get_params') else {},
                train_samples=len(X_train),
                val_samples=len(X_val),
                n_features=X_train.shape[1],
                training_time_seconds=time.time() - self.start_time,
                total_epochs=1,
                tags=[self.config.model_type],
            )
        
        training_time = time.time() - self.start_time
        
        if self.config.verbose:
            logger.info(f"Training complete: metrics={final_metrics}, time={training_time:.1f}s")
        
        return TrainingResult(
            success=success,
            best_epoch=1,
            best_metrics=final_metrics,
            final_metrics=final_metrics,
            training_time_seconds=training_time,
            early_stopped=False,
            model_metadata=model_metadata,
            quality_report=quality_report,
        )
    
    def train_pytorch(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        optimizer: Any,
        loss_fn: Any,
        device: str = "cpu",
    ) -> TrainingResult:
        """
        Train PyTorch model with full early stopping and LR scheduling.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            device: "cpu" or "cuda"
            
        Returns:
            TrainingResult
        """
        import torch
        
        self.start_time = time.time()
        model = model.to(device)
        
        # Initialize LR scheduler
        lr_scheduler = None
        if self.config.lr_scheduler != "none":
            lr_scheduler = UnifiedLRScheduler(
                optimizer,
                scheduler_type=self.config.lr_scheduler,
                warmup_epochs=self.config.warmup_epochs,
                min_lr=self.config.min_lr,
                max_epochs=self.config.epochs,
                verbose=False,
            )
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Train
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_X)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            val_loss, val_metrics = self._validate_pytorch(
                model, val_loader, loss_fn, device
            )
            
            # Step LR scheduler
            if lr_scheduler:
                lr_scheduler.step(epoch, val_loss)
            
            # Log
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
            self.epoch_history.append(epoch_metrics)
            
            if self.config.verbose and epoch % self.config.log_interval == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            stop_metric = val_loss if self.config.early_stop_mode == "min" else val_metrics.get("accuracy", 0)
            self.early_stopping(stop_metric, model, epoch, epoch_metrics)
            
            if self.early_stopping.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Restore best model
        if self.config.early_stop_mode == "min" and self.early_stopping.best_model:
            self.early_stopping.restore(model)
        
        # Final evaluation
        final_val_loss, final_metrics = self._validate_pytorch(
            model, val_loader, loss_fn, device
        )
        
        # Register model
        model_metadata = None
        if self.config.register_model:
            model_metadata = self.registry.register(
                name=self.config.model_name,
                model=model,
                metrics=final_metrics,
                hyperparams={"lr": self.config.learning_rate, "epochs": len(self.epoch_history)},
                training_time_seconds=time.time() - self.start_time,
                early_stopped=self.early_stopping.should_stop,
                best_epoch=self.early_stopping.best_epoch,
                total_epochs=len(self.epoch_history),
            )
        
        training_time = time.time() - self.start_time
        
        return TrainingResult(
            success=True,
            best_epoch=self.early_stopping.best_epoch,
            best_metrics=self.early_stopping.best_score_metrics if hasattr(self.early_stopping, 'best_score_metrics') else final_metrics,
            final_metrics=final_metrics,
            training_time_seconds=training_time,
            early_stopped=self.early_stopping.should_stop,
            model_metadata=model_metadata,
            epoch_history=self.epoch_history,
        )
    
    def train_xgboost(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> TrainingResult:
        """
        Train XGBoost with early stopping via native API.
        
        Args:
            model: XGBoost model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            TrainingResult
        """
        self.start_time = time.time()
        
        # 1. Data quality check
        quality_report = None
        if self.config.validate_data:
            passed, quality_report = self.data_quality.validate(X_train)
        
        # 2. Train with early stopping
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=self.config.verbose and self.config.log_interval > 0,
            )
            success = True
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            success = False
        
        # 3. Get metrics
        final_metrics = self._evaluate_xgboost(model, X_val, y_val)
        
        # 4. Get early stopping info
        best_iteration = getattr(model, 'best_iteration', None) or getattr(model, 'best_ntree_limit', None)
        early_stopped = best_iteration is not None and best_iteration < self.config.epochs
        
        # 5. Register
        model_metadata = None
        if self.config.register_model:
            model_metadata = self.registry.register(
                name=self.config.model_name,
                model=model,
                metrics=final_metrics,
                hyperparams=model.get_params() if hasattr(model, 'get_params') else {},
                train_samples=len(X_train),
                val_samples=len(X_val),
                n_features=X_train.shape[1],
                training_time_seconds=time.time() - self.start_time,
                early_stopped=early_stopped,
                best_epoch=best_iteration or 0,
                total_epochs=model.get_num_boosting_rounds() if hasattr(model, 'get_num_boosting_rounds') else 0,
                tags=["xgboost"],
            )
        
        training_time = time.time() - self.start_time
        
        return TrainingResult(
            success=success,
            best_epoch=best_iteration or 0,
            best_metrics=final_metrics,
            final_metrics=final_metrics,
            training_time_seconds=training_time,
            early_stopped=early_stopped,
            model_metadata=model_metadata,
            quality_report=quality_report,
        )
    
    def _evaluate_sklearn(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate sklearn model."""
        metrics = {}
        
        try:
            score = model.score(X, y)
            metrics["accuracy"] = score
        except Exception:
            pass
        
        try:
            from sklearn.metrics import mean_squared_error, r2_score
            predictions = model.predict(X)
            metrics["mse"] = mean_squared_error(y, predictions)
            metrics["r2"] = r2_score(y, predictions)
        except Exception:
            pass
        
        return metrics
    
    def _validate_pytorch(
        self,
        model: Any,
        val_loader: Any,
        loss_fn: Any,
        device: str,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate PyTorch model."""
        import torch
        
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = loss_fn(output, batch_y)
                total_loss += loss.item()
                
                all_preds.append(output.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = {"val_loss": avg_loss}
        
        return avg_loss, metrics
    
    def _evaluate_xgboost(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate XGBoost model."""
        metrics = {}
        
        try:
            predictions = model.predict(X)
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            # Classification or regression
            if len(np.unique(y)) < 20:  # Likely classification
                metrics["accuracy"] = accuracy_score(y, predictions)
            else:
                metrics["mse"] = mean_squared_error(y, predictions)
                
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
        
        return metrics


class ExperimentTracker:
    """
    Track experiments for comparison and reproducibility.
    
    Saves metrics, hyperparameters, and artifacts for each run.
    
    Example:
        >>> tracker = ExperimentTracker("my_experiment")
        >>> tracker.start_run(params={"lr": 0.001})
        >>> tracker.log_metric("accuracy", 0.85, step=1)
        >>> tracker.log_artifact("model.pt")
        >>> tracker.end_run()
    """
    
    def __init__(self, experiment_name: str, tracking_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir) / experiment_name
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.metrics_history: List[Dict] = []
    
    def start_run(self, params: Optional[Dict] = None) -> str:
        """Start a new experiment run."""
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.tracking_dir / self.run_id
        self.run_dir.mkdir()
        self.metrics_history = []
        
        if params:
            with open(self.run_dir / "params.json", "w") as f:
                json.dump(params, f, indent=2, default=str)
        
        return self.run_id
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric."""
        entry = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": time.time(),
        }
        self.metrics_history.append(entry)
        
        if self.run_dir:
            with open(self.run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
    
    def log_artifact(self, source_path: str, name: Optional[str] = None) -> None:
        """Copy artifact to run directory."""
        import shutil
        
        if self.run_dir:
            dest = self.run_dir / "artifacts" / (name or Path(source_path).name)
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(source_path, dest)
    
    def end_run(self, summary: Optional[Dict] = None) -> None:
        """End current run with optional summary."""
        if self.run_dir and summary:
            with open(self.run_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
        
        self.run_id = None
        self.run_dir = None
    
    def compare_runs(self, metric: str = "val_loss", top_n: int = 10) -> List[Dict]:
        """Compare runs by metric."""
        results = []
        
        for run_dir in sorted(self.tracking_dir.iterdir(), reverse=True):
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                results.append({
                    "run": run_dir.name,
                    metric: summary.get(metric, None),
                    "summary": summary,
                })
        
        # Sort by metric (ascending for loss, descending for accuracy)
        results = [r for r in results if r.get(metric) is not None]
        results.sort(key=lambda x: x[metric], reverse="loss" not in metric.lower())
        
        return results[:top_n]
