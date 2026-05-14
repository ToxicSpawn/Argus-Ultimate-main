"""
Online Stacking and Model Versioning.

Provides:
- Online stacking: Combine base model predictions as meta-features
- Model versioning: Track model lineage, rollback, and A/B testing
- Model registry: Persist and load trained models
- Ensemble evolution: Adapt ensemble composition over time

Usage:
    # Stacking
    stacker = OnlineStacker(base_models=[model1, model2, model3])
    meta_features = stacker.stack(base_predictions)
    final_pred = meta_model.predict(meta_features)

    # Versioning
    registry = ModelRegistry()
    version_id = registry.save(model, metadata={"regime": "TREND_UP"})
    loaded = registry.load(version_id)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """A version of a trained model."""

    version_id: str
    model_name: str
    created_at: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    parent_version: Optional[str] = None
    status: str = "active"  # active, archived, deprecated
    hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StackingResult:
    """Result from stacking operation."""

    meta_features: np.ndarray
    base_predictions: Dict[str, np.ndarray]
    n_base_models: int
    meta_feature_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta_features_shape": self.meta_features.shape,
            "n_base_models": self.n_base_models,
            "base_model_names": list(self.base_predictions.keys()),
            "meta_feature_names": self.meta_feature_names,
        }


class OnlineStacker:
    """
    Online stacking: combine base model predictions as meta-features.

    Workflow:
    1. Train base models (diverse models: tree, linear, etc.)
    2. Generate out-of-fold predictions for training meta-model
    3. Stack meta-features into a meta-model
    4. Use meta-model for final predictions

    Supports:
    - Multiple base model types
    - Weighted combination (learned or fixed)
    - Online update of meta-model weights
    """

    def __init__(
        self,
        *,
        meta_learner: Optional[Any] = None,
        weights: Optional[List[float]] = None,
        use_proba: bool = True,
        add_original_features: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.meta_learner = meta_learner
        self.weights = weights
        self.use_proba = use_proba
        self.add_original_features = add_original_features
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._base_models: Dict[str, Any] = {}
        self._meta_model: Optional[Any] = None
        self._is_fitted = False

    def add_base_model(self, name: str, model: Any) -> "OnlineStacker":
        """Add a base model to the stack."""
        self._base_models[name] = model
        return self

    def remove_base_model(self, name: str) -> "OnlineStacker":
        """Remove a base model from the stack."""
        if name in self._base_models:
            del self._base_models[name]
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        base_predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> "OnlineStacker":
        """
        Fit the stacking ensemble.

        Args:
            X: Original features
            y: Labels
            base_predictions: Pre-computed base model predictions (optional)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if base_predictions is None:
            # Generate base predictions via cross-validation
            base_predictions = self._generate_oof_predictions(X, y)

        # Build meta-features
        meta_features, _ = self._build_meta_features(base_predictions, X if self.add_original_features else None)

        # Fit meta-learner
        if self.meta_learner is not None:
            self._meta_model = self.meta_learner
            self._meta_model.fit(meta_features, y)
        else:
            # Default: simple weighted average
            self._fit_weighted_average(meta_features, y)

        self._is_fitted = True
        return self

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Generate out-of-fold predictions for base models."""
        from sklearn.model_selection import KFold

        n_samples = len(X)
        base_predictions: Dict[str, np.ndarray] = {}

        n_folds = min(5, n_samples)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

        for name, model in self._base_models.items():
            oof_pred = np.zeros(n_samples)

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]

                # Clone model for training
                try:
                    model_clone = pickle.loads(pickle.dumps(model))
                except Exception:
                    model_clone = model

                # Fit and predict
                try:
                    model_clone.fit(X_train, y_train)
                    if self.use_proba and hasattr(model_clone, "predict_proba"):
                        proba = model_clone.predict_proba(X_val)
                        if proba.shape[1] == 2:
                            oof_pred[val_idx] = proba[:, 1]
                        else:
                            oof_pred[val_idx] = proba[:, 0]
                    else:
                        oof_pred[val_idx] = model_clone.predict(X_val)
                except Exception:
                    oof_pred[val_idx] = 0.5

            base_predictions[name] = oof_pred

        return base_predictions

    def _build_meta_features(
        self,
        base_predictions: Dict[str, np.ndarray],
        original_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build meta-features from base predictions."""
        # Stack predictions horizontally
        pred_list = list(base_predictions.values())
        n_samples = len(pred_list[0])

        meta_features = np.column_stack(pred_list)
        meta_names = list(base_predictions.keys())

        # Add rank features (percentile ranks)
        ranks = np.zeros_like(meta_features)
        for j, pred in enumerate(pred_list):
            order = np.argsort(np.argsort(pred))
            ranks[:, j] = order / max(len(pred) - 1, 1)
        meta_features = np.hstack([meta_features, ranks])
        meta_names.extend([f"{n}_rank" for n in base_predictions.keys()])

        # Add original features if requested
        if original_features is not None:
            # Use PCA to reduce dimensionality if too many features
            n_orig = original_features.shape[1]
            if n_orig > 10:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=10, random_state=self.seed)
                orig_reduced = pca.fit_transform(original_features)
            else:
                orig_reduced = original_features
            meta_features = np.hstack([meta_features, orig_reduced])
            meta_names.extend([f"orig_pc_{i}" for i in range(orig_reduced.shape[1])])

        return meta_features, meta_names

    def _fit_weighted_average(
        self,
        meta_features: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit weighted average as meta-learner."""
        n_models = meta_features.shape[1]
        if self.weights is None:
            # Initialize with equal weights
            self.weights = [1.0 / n_models] * n_models

        # Optimize weights via gradient descent
        best_weights = self.weights.copy()
        best_loss = float("inf")

        for _ in range(100):
            # Compute weighted prediction
            weights_arr = np.array(self.weights)
            pred = np.dot(meta_features, weights_arr)

            # Compute loss (MSE)
            loss = np.mean((pred - y) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_weights = self.weights.copy()

            # Gradient step
            gradient = 2 * np.mean((pred - y)[:, None] * meta_features, axis=0)
            lr = 0.01
            self.weights = np.maximum(self.weights - lr * gradient, 0.0)
            total = sum(self.weights)
            if total > 0:
                self.weights = [w / total for w in self.weights]

        self.weights = best_weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacked ensemble."""
        if not self._is_fitted:
            return np.zeros(len(X))

        # Generate base predictions
        base_predictions = {}
        for name, model in self._base_models.items():
            try:
                if self.use_proba and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    if proba.shape[1] == 2:
                        base_predictions[name] = proba[:, 1]
                    else:
                        base_predictions[name] = proba[:, 0]
                else:
                    base_predictions[name] = model.predict(X)
            except Exception:
                base_predictions[name] = np.zeros(len(X))

        # Build meta-features
        meta_features, _ = self._build_meta_features(base_predictions)

        # Predict with meta-learner
        if self._meta_model is not None:
            return self._meta_model.predict(meta_features)
        else:
            # Weighted average
            weights_arr = np.array(self.weights[:meta_features.shape[1]])
            return np.dot(meta_features, weights_arr)

    def stack(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack base predictions into meta-features."""
        meta_features, _ = self._build_meta_features(base_predictions)
        return meta_features

    def update_weights(
        self,
        base_predictions: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> None:
        """Update stacking weights online."""
        meta_features, _ = self._build_meta_features(base_predictions)
        self._fit_weighted_average(meta_features, y)


class ModelRegistry:
    """
    Model versioning and registry.

    Features:
    - Save/load models with metadata
    - Track model lineage (parent versions)
    - Rollback to previous versions
    - A/B testing support
    - Model comparison

    Storage: Local filesystem (pickle + JSON metadata)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_versions: int = 50,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path else Path("models")
        self.max_versions = max_versions

        # In-memory index
        self._versions: Dict[str, ModelVersion] = {}
        self._model_store: Dict[str, Any] = {}

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing versions
        self._load_index()

    def _generate_version_id(self, model_name: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        hash_input = f"{model_name}_{timestamp}_{self._rng.random() if hasattr(self, '_rng') else 0}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{model_name}_{timestamp}_{hash_suffix}"

    def save(
        self,
        model: Any,
        name: str,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
    ) -> str:
        """
        Save a model version.

        Returns:
            version_id: Unique identifier for the saved model
        """
        version_id = self._generate_version_id(name)

        # Create version metadata
        version = ModelVersion(
            version_id=version_id,
            model_name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            parameters=parameters or {},
            metrics=metrics or {},
            metadata=metadata or {},
            parent_version=parent_version,
            status="active",
        )

        # Compute model hash
        try:
            model_bytes = pickle.dumps(model)
            version.hash = hashlib.md5(model_bytes).hexdigest()
        except Exception:
            pass

        # Save model to disk
        model_path = self.storage_path / f"{version_id}.pkl"
        meta_path = self.storage_path / f"{version_id}.json"

        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(meta_path, "w") as f:
                json.dump(version.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model to disk: {e}")
            # Keep in memory
            self._model_store[version_id] = model

        # Update index
        self._versions[version_id] = version

        # Enforce max versions
        self._prune_old_versions(name)

        return version_id

    def load(self, version_id: str) -> Optional[Any]:
        """Load a model by version ID."""
        # Check memory first
        if version_id in self._model_store:
            return self._model_store[version_id]

        # Load from disk
        model_path = self.storage_path / f"{version_id}.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load model {version_id}: {e}")
                return None

        return None

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get version metadata."""
        return self._versions.get(version_id)

    def list_versions(
        self,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> List[ModelVersion]:
        """List model versions."""
        versions = list(self._versions.values())

        # Filter
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        if status:
            versions = [v for v in versions if v.status == status]

        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions[:limit]

    def rollback(self, version_id: str) -> Optional[str]:
        """
        Rollback to a previous version.

        Returns:
            new_version_id: ID of the restored version
        """
        version = self._versions.get(version_id)
        if version is None:
            return None

        # Load the model
        model = self.load(version_id)
        if model is None:
            return None

        # Save as new version (creating a restoration point)
        new_version_id = self.save(
            model=model,
            name=version.model_name,
            metrics=version.metrics.copy(),
            parameters=version.parameters.copy(),
            metadata={**version.metadata, "restored_from": version_id},
            parent_version=version_id,
        )

        return new_version_id

    def deprecate(self, version_id: str) -> bool:
        """Mark a version as deprecated."""
        if version_id in self._versions:
            self._versions[version_id].status = "deprecated"
            meta_path = self.storage_path / f"{version_id}.json"
            if meta_path.exists():
                with open(meta_path, "w") as f:
                    json.dump(self._versions[version_id].to_dict(), f, indent=2)
            return True
        return False

    def _load_index(self) -> None:
        """Load version index from disk."""
        for meta_path in self.storage_path.glob("*.json"):
            try:
                with open(meta_path, "r") as f:
                    version_dict = json.load(f)
                version = ModelVersion(**version_dict)
                self._versions[version.version_id] = version
            except Exception:
                pass

    def _prune_old_versions(self, model_name: str) -> None:
        """Remove old versions beyond max_versions."""
        versions = self.list_versions(model_name=model_name)
        if len(versions) <= self.max_versions:
            return

        # Archive oldest versions
        for version in versions[self.max_versions:]:
            version.status = "archived"
            meta_path = self.storage_path / f"{version.version_id}.json"
            if meta_path.exists():
                with open(meta_path, "w") as f:
                    json.dump(version.to_dict(), f, indent=2)


@dataclass
class EnsembleEvolution:
    """Track and evolve ensemble composition over time."""

    initial_weights: Dict[str, float]
    adaptation_rate: float = 0.1
    min_weight: float = 0.05
    performance_history: List[Dict[str, float]] = field(default_factory=list)

    def compute_adapted_weights(
        self,
        recent_performance: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute adapted ensemble weights based on recent performance.

        Args:
            recent_performance: Dict of model_name -> recent performance (e.g., Sharpe)

        Returns:
            Adapted weights (sum to 1.0)
        """
        if not recent_performance:
            return self.initial_weights.copy()

        # Compute target weights from performance
        # Use softmax-style normalization
        perf_values = np.array(list(recent_performance.values()))
        perf_values = np.maximum(perf_values, 0)  # Zero out negative

        if np.sum(perf_values) < 1e-10:
            return self.initial_weights.copy()

        # Softmax-style target
        target_weights = perf_values / np.sum(perf_values)

        # Start from initial or current weights
        current = np.array([self.initial_weights.get(k, 0) for k in recent_performance.keys()])

        # Blend toward target
        adapted = current * (1 - self.adaptation_rate) + target_weights * self.adaptation_rate

        # Enforce minimum weight
        adapted = np.maximum(adapted, self.min_weight)

        # Renormalize
        adapted = adapted / np.sum(adapted)

        return {k: float(v) for k, v in zip(recent_performance.keys(), adapted)}


def create_stacker(
    base_models: List[Any],
    meta_model: Optional[Any] = None,
) -> OnlineStacker:
    """Factory function to create stacker."""
    stacker = OnlineStacker(meta_learner=meta_model)
    for i, model in enumerate(base_models):
        stacker.add_base_model(f"model_{i}", model)
    return stacker


def create_registry(storage_path: Optional[str] = None) -> ModelRegistry:
    """Factory function to create model registry."""
    return ModelRegistry(storage_path=storage_path)


__all__ = [
    "OnlineStacker",
    "ModelRegistry",
    "ModelVersion",
    "StackingResult",
    "EnsembleEvolution",
    "create_stacker",
    "create_registry",
]