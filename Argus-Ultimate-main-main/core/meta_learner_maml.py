"""Model-Agnostic Meta-Learning (MAML) for cross-market adaptation.

A numpy-only, simplified MAML implementation that lets ARGUS share a
parameter initialisation across every traded market, then *quickly* adapt to a
specific symbol when presented with a handful of labelled samples.

Why this matters for ARGUS
--------------------------

Every crypto pair has its own microstructure. A separate model per symbol
forgets everything it has learned when a new pair is added; a single global
model washes out symbol-specific dynamics. MAML strikes a middle ground: the
shared weights are explicitly trained so that a few gradient steps on any
one symbol's data snap the parameters to a performant task-specific point.

Concretely, the outer loop does:

    theta <- theta - beta * sum_tasks grad_theta L_task( theta - alpha * grad_theta L_task( theta ) )

Each task's loss is a plain linear regression MSE over the samples registered
for that symbol. The inner update is a single step — this is enough to pick
up the "fast adaptation" flavour without the second-order overhead of full
MAML.

Classes
-------

* :class:`TaskContext` — dataclass holding feature/label buffers for one
  market (keyed by symbol).
* :class:`MetaLearner` — the main entry point. Register tasks with
  ``register_task(symbol)``, stream samples with ``add_samples``, run meta
  training with ``meta_train``, and fast-adapt with ``adapt_to_task`` before
  calling ``predict``.

All methods degrade gracefully when data is missing or shapes mismatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskContext:
    """Per-symbol task buffer used by :class:`MetaLearner`.

    ``features`` is appended to as observations arrive. ``labels`` is the
    matching target sequence. ``adapted_weights`` caches the last fast-adapted
    weight vector so repeated ``predict`` calls don't have to redo the inner
    loop.
    """

    symbol: str
    features: List[np.ndarray] = field(default_factory=list)
    labels: List[float] = field(default_factory=list)
    adapted_weights: Optional[np.ndarray] = None
    adapted_bias: float = 0.0
    last_adapted_step: int = -1
    sample_count: int = 0

    def stacked(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the buffer as ``(X, y)`` numpy arrays. Empty buffers return
        empty arrays."""

        if not self.features:
            return np.zeros((0, 0)), np.zeros(0)
        X = np.stack(self.features)
        y = np.asarray(self.labels, dtype=np.float64)
        return X, y


# ---------------------------------------------------------------------------
# Meta learner
# ---------------------------------------------------------------------------


class MetaLearner:
    """MAML-inspired meta-learner over a collection of trading tasks.

    Parameters
    ----------
    feature_dim
        Size of each input feature vector.
    inner_lr
        Learning rate for the inner (task-specific) gradient step.
    meta_lr
        Learning rate for the outer (shared) update.
    inner_steps
        Number of inner gradient steps taken during fast adaptation. Default
        is 1, which matches the "one-shot MAML" flavour used in the original
        paper for speed.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        inner_lr: float = 0.05,
        meta_lr: float = 0.01,
        inner_steps: int = 1,
        seed: int = 0,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.inner_lr = float(inner_lr)
        self.meta_lr = float(meta_lr)
        self.inner_steps = max(1, int(inner_steps))
        self._rng = np.random.default_rng(seed)
        self.shared_weights = self._rng.normal(scale=0.1, size=self.feature_dim)
        self.shared_bias = 0.0
        self.tasks: Dict[str, TaskContext] = {}
        self.meta_step = 0
        self.last_meta_loss = 0.0

    # -- task management -------------------------------------------------

    def register_task(self, symbol: str) -> TaskContext:
        """Create (or return) the :class:`TaskContext` for ``symbol``."""

        if symbol not in self.tasks:
            self.tasks[symbol] = TaskContext(symbol=symbol)
        return self.tasks[symbol]

    def add_samples(
        self, symbol: str, features: np.ndarray, labels: np.ndarray
    ) -> None:
        """Append samples to the task buffer for ``symbol``.

        ``features`` may be 1-D (a single sample) or 2-D (a batch). Mismatched
        feature lengths are silently clipped / zero-padded to ``feature_dim``
        so callers can be relaxed about shapes.
        """

        task = self.register_task(symbol)
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        labels = np.atleast_1d(np.asarray(labels, dtype=np.float64))
        if features.shape[0] != labels.shape[0]:
            logger.warning(
                "meta_learner: shape mismatch adding samples for %s: %s vs %s",
                symbol,
                features.shape,
                labels.shape,
            )
            return
        for row, y in zip(features, labels):
            # Pad/clip to feature_dim for robustness to upstream schema drift.
            padded = np.zeros(self.feature_dim, dtype=np.float64)
            n = min(self.feature_dim, row.size)
            padded[:n] = row[:n]
            task.features.append(padded)
            task.labels.append(float(y))
            task.sample_count += 1
        # Adapted weights are stale once new data arrives.
        task.adapted_weights = None

    # -- inner loop ------------------------------------------------------

    def _task_grad(
        self, w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Compute gradient + loss for a linear regression on one task."""

        if X.size == 0:
            return np.zeros_like(w), 0.0, 0.0
        preds = X @ w + b
        err = preds - y
        loss = float(0.5 * np.mean(err * err))
        grad_w = X.T @ err / max(1, X.shape[0])
        grad_b = float(err.mean())
        return grad_w, grad_b, loss

    def _inner_update(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Take ``inner_steps`` gradient steps starting from shared weights."""

        w = self.shared_weights.copy()
        b = self.shared_bias
        final_loss = 0.0
        for _ in range(self.inner_steps):
            grad_w, grad_b, loss = self._task_grad(w, b, X, y)
            w = w - self.inner_lr * grad_w
            b = b - self.inner_lr * grad_b
            final_loss = loss
        return w, b, final_loss

    # -- outer loop ------------------------------------------------------

    def meta_train(self, n_iterations: int = 1) -> float:
        """Run the outer meta-update for ``n_iterations`` iterations.

        For each iteration we compute the post-adaptation loss on every task
        (using a first-order approximation: re-evaluate the gradient at the
        task-adapted parameters and treat that as the meta gradient). The mean
        gradient across tasks is used to update ``shared_weights``.
        """

        if not self.tasks:
            return 0.0
        mean_loss = 0.0
        for _ in range(max(1, int(n_iterations))):
            grad_w_accum = np.zeros_like(self.shared_weights)
            grad_b_accum = 0.0
            count = 0
            iter_loss = 0.0
            for task in self.tasks.values():
                X, y = task.stacked()
                if X.size == 0:
                    continue
                # Inner adaptation from the current shared init.
                w_adapted, b_adapted, _ = self._inner_update(X, y)
                # First-order meta gradient: evaluate task loss gradient at the
                # adapted parameters.
                grad_w, grad_b, loss = self._task_grad(w_adapted, b_adapted, X, y)
                grad_w_accum += grad_w
                grad_b_accum += grad_b
                iter_loss += loss
                count += 1
            if count == 0:
                return 0.0
            grad_w_accum /= count
            grad_b_accum /= count
            self.shared_weights -= self.meta_lr * grad_w_accum
            self.shared_bias -= self.meta_lr * grad_b_accum
            self.meta_step += 1
            mean_loss = iter_loss / count
            self.last_meta_loss = mean_loss
        return float(mean_loss)

    # -- prediction ------------------------------------------------------

    def adapt_to_task(self, symbol: str, n_shots: Optional[int] = None) -> bool:
        """Fast-adapt shared weights to ``symbol`` using ``n_shots`` samples.

        If ``n_shots`` is ``None`` the full task buffer is used. Returns
        ``True`` if adaptation actually ran (i.e. there was data), else
        ``False``.
        """

        task = self.tasks.get(symbol)
        if task is None:
            logger.debug("meta_learner: no task registered for %s", symbol)
            return False
        X, y = task.stacked()
        if X.size == 0:
            return False
        if n_shots is not None and n_shots < X.shape[0]:
            X = X[-n_shots:]
            y = y[-n_shots:]
        w, b, _ = self._inner_update(X, y)
        task.adapted_weights = w
        task.adapted_bias = b
        task.last_adapted_step = self.meta_step
        return True

    def predict(self, symbol: str, features: np.ndarray) -> float:
        """Predict ``features`` for ``symbol`` using (possibly) adapted weights."""

        features = np.asarray(features, dtype=np.float64)
        if features.ndim > 1:
            features = features.flatten()
        padded = np.zeros(self.feature_dim, dtype=np.float64)
        n = min(self.feature_dim, features.size)
        padded[:n] = features[:n]
        task = self.tasks.get(symbol)
        if task is not None and task.adapted_weights is not None:
            return float(padded @ task.adapted_weights + task.adapted_bias)
        return float(padded @ self.shared_weights + self.shared_bias)

    # -- introspection ---------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary for monitoring."""

        return {
            "feature_dim": self.feature_dim,
            "inner_lr": self.inner_lr,
            "meta_lr": self.meta_lr,
            "inner_steps": self.inner_steps,
            "meta_step": self.meta_step,
            "last_meta_loss": float(self.last_meta_loss),
            "shared_weight_norm": float(np.linalg.norm(self.shared_weights)),
            "num_tasks": len(self.tasks),
            "tasks": {
                sym: {
                    "samples": ctx.sample_count,
                    "adapted": ctx.adapted_weights is not None,
                    "last_adapted_step": ctx.last_adapted_step,
                }
                for sym, ctx in self.tasks.items()
            },
        }
