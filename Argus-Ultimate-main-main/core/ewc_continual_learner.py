"""Elastic Weight Consolidation (EWC) for continual learning.

EWC is the technique of Kirkpatrick et al. (2017) that lets a single model
learn a *sequence* of tasks without catastrophically forgetting prior ones. It
does this by adding a quadratic penalty to the loss whenever a weight tries to
move away from the value it had when previous tasks were consolidated — with
the penalty strength per weight given by the diagonal of the Fisher
Information Matrix.

In ARGUS this lets us keep training a single shared model on a rolling
sequence of regimes or symbols without overwriting what the model learned in
earlier phases.

Classes
-------

* :class:`FisherInformation`
    Diagonal Fisher estimator. Accumulates squared gradients across a dataset.

* :class:`EWCContinualLearner`
    Linear-model continual learner. ``train_on_task`` does SGD with the EWC
    penalty, ``switch_task`` consolidates the current task's Fisher + optimal
    weights and starts a new task, and ``predict`` issues point forecasts.

All numpy. No external ML dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fisher information
# ---------------------------------------------------------------------------


@dataclass
class TaskAnchor:
    """Snapshot of one consolidated task.

    Stores the weights at which the task was "solved" and the diagonal Fisher
    importance for each parameter.
    """

    name: str
    weights: np.ndarray
    bias: float
    fisher_w: np.ndarray
    fisher_b: float
    sample_count: int = 0


class FisherInformation:
    """Diagonal Fisher Information estimator for a linear model.

    For linear regression with MSE the Hessian is ``X.T @ X / N`` — we use the
    squared gradient per sample as a cheap diagonal approximation that also
    generalises to non-Gaussian losses.
    """

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = int(feature_dim)
        self.fisher_w = np.zeros(self.feature_dim, dtype=np.float64)
        self.fisher_b = 0.0
        self.n_samples = 0

    def reset(self) -> None:
        self.fisher_w = np.zeros(self.feature_dim, dtype=np.float64)
        self.fisher_b = 0.0
        self.n_samples = 0

    def accumulate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        bias: float,
    ) -> None:
        """Accumulate squared per-sample gradients at ``(weights, bias)``."""

        X = np.atleast_2d(np.asarray(features, dtype=np.float64))
        y = np.atleast_1d(np.asarray(labels, dtype=np.float64))
        if X.size == 0 or X.shape[0] != y.shape[0]:
            return
        preds = X @ weights + bias
        err = preds - y
        # d/dw of 0.5 * (pred - y)^2 = err * x.
        grad_w_sq = (err[:, None] ** 2) * (X ** 2)
        grad_b_sq = err ** 2
        self.fisher_w += grad_w_sq.sum(axis=0)
        self.fisher_b += float(grad_b_sq.sum())
        self.n_samples += X.shape[0]

    def finalize(self) -> Tuple[np.ndarray, float]:
        """Return the mean squared-gradient (diagonal Fisher) and reset."""

        if self.n_samples == 0:
            return self.fisher_w.copy(), float(self.fisher_b)
        fw = self.fisher_w / self.n_samples
        fb = self.fisher_b / self.n_samples
        return fw, float(fb)


# ---------------------------------------------------------------------------
# Continual learner
# ---------------------------------------------------------------------------


class EWCContinualLearner:
    """Linear model continual learner guarded by EWC penalties.

    Parameters
    ----------
    feature_dim
        Input dimensionality.
    learning_rate
        SGD learning rate for the base loss.
    ewc_lambda
        Strength of the quadratic anchoring penalty. Higher values preserve
        previous tasks more aggressively at the cost of slower learning on
        the current task.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        learning_rate: float = 1e-2,
        ewc_lambda: float = 100.0,
        max_grad_norm: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.learning_rate = float(learning_rate)
        self.ewc_lambda = float(ewc_lambda)
        self.max_grad_norm = float(max_grad_norm)
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(scale=0.05, size=self.feature_dim)
        self.bias = 0.0
        self._fisher = FisherInformation(self.feature_dim)
        self.anchors: Dict[str, TaskAnchor] = {}
        self.current_task: Optional[str] = None
        self.task_history: List[str] = []
        self.step_count = 0
        self.last_base_loss = 0.0
        self.last_ewc_penalty = 0.0

    # -- task management -------------------------------------------------

    def register_task(self, name: str) -> None:
        """Declare a new task as the current task.

        If a task with that name already exists its anchor is left in place;
        this lets you revisit a task without discarding past consolidation.
        """

        self.current_task = str(name)
        if self.current_task not in self.task_history:
            self.task_history.append(self.current_task)

    def switch_task(self, new_task: str) -> Optional[TaskAnchor]:
        """Consolidate the current task and start ``new_task``.

        Returns the newly created :class:`TaskAnchor` so callers can persist or
        inspect it. Returns ``None`` if there is no current task.
        """

        if self.current_task is None:
            logger.debug("ewc: switch_task called with no current task")
            self.register_task(new_task)
            return None
        fisher_w, fisher_b = self._fisher.finalize()
        anchor = TaskAnchor(
            name=self.current_task,
            weights=self.weights.copy(),
            bias=float(self.bias),
            fisher_w=fisher_w.copy(),
            fisher_b=float(fisher_b),
            sample_count=self._fisher.n_samples,
        )
        self.anchors[self.current_task] = anchor
        self._fisher.reset()
        self.register_task(new_task)
        return anchor

    # -- training --------------------------------------------------------

    def _pad(self, X: np.ndarray) -> np.ndarray:
        """Clip / zero-pad ``X`` to ``self.feature_dim``."""

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] == self.feature_dim:
            return X
        out = np.zeros((X.shape[0], self.feature_dim), dtype=np.float64)
        n = min(self.feature_dim, X.shape[1])
        out[:, :n] = X[:, :n]
        return out

    def _ewc_penalty_grad(self) -> Tuple[np.ndarray, float, float]:
        """Compute the total EWC penalty value and its gradient."""

        if not self.anchors:
            return np.zeros_like(self.weights), 0.0, 0.0
        penalty_value = 0.0
        grad_w = np.zeros_like(self.weights)
        grad_b = 0.0
        for anchor in self.anchors.values():
            dw = self.weights - anchor.weights
            db = self.bias - anchor.bias
            penalty_value += float((anchor.fisher_w * dw * dw).sum())
            penalty_value += float(anchor.fisher_b * db * db)
            grad_w += 2.0 * anchor.fisher_w * dw
            grad_b += 2.0 * anchor.fisher_b * db
        return grad_w * self.ewc_lambda, float(grad_b * self.ewc_lambda), (
            penalty_value * self.ewc_lambda / 2.0
        )

    def train_on_task(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 1,
    ) -> Dict[str, float]:
        """SGD on ``(features, labels)`` for the *current* task.

        Each epoch sweeps the full batch once with a single gradient step. The
        gradient is the base MSE gradient plus the EWC penalty gradient from
        every previously consolidated task. Gradient norms are clipped by
        ``max_grad_norm`` to keep updates stable when the EWC penalty is
        large relative to the base loss.
        """

        if self.current_task is None:
            logger.debug("ewc: train_on_task called without register_task")
            self.register_task("default")
        X = self._pad(features)
        y = np.atleast_1d(np.asarray(labels, dtype=np.float64))
        if X.shape[0] != y.shape[0] or X.shape[0] == 0:
            return {"base_loss": 0.0, "ewc_penalty": 0.0}
        base_loss = 0.0
        penalty_val = 0.0
        for _ in range(max(1, int(epochs))):
            preds = X @ self.weights + self.bias
            err = preds - y
            base_loss = float(0.5 * np.mean(err * err))
            grad_w = X.T @ err / X.shape[0]
            grad_b = float(err.mean())
            ewc_grad_w, ewc_grad_b, penalty_val = self._ewc_penalty_grad()
            total_grad_w = grad_w + ewc_grad_w
            total_grad_b = grad_b + ewc_grad_b
            # Clip the combined gradient so a blow-up in the EWC penalty
            # cannot throw the weights into the floating-point wilderness.
            g_norm = float(np.linalg.norm(total_grad_w))
            if g_norm > self.max_grad_norm:
                scale = self.max_grad_norm / max(g_norm, 1e-9)
                total_grad_w = total_grad_w * scale
                total_grad_b = total_grad_b * scale
            self.weights -= self.learning_rate * total_grad_w
            self.bias -= self.learning_rate * total_grad_b
            self.step_count += 1
        # Accumulate Fisher once per train_on_task call using the *final*
        # trained weights. This matches the textbook definition — the Fisher
        # should reflect the importance of parameters at the task solution
        # — and avoids the inner-loop compounding that blew up the test.
        self._fisher.accumulate(X, y, self.weights, self.bias)
        self.last_base_loss = base_loss
        self.last_ewc_penalty = penalty_val
        return {"base_loss": base_loss, "ewc_penalty": penalty_val}

    # -- Phase W8: train from memory replay buffer -----------------------

    def train_from_replay_buffer(
        self,
        replay_buffer: Any,
        batch_size: int = 64,
        epochs: int = 1,
    ) -> Dict[str, Any]:
        """
        Train on a batch sampled from a ``MemoryReplayBuffer``.

        This wires the Phase W8 replay layer into EWC continual learning:
        offline replay reduces catastrophic forgetting on new regimes
        without requiring fresh live fills.

        Parameters
        ----------
        replay_buffer : Any
            Instance of ``core/memory/memory_replay.MemoryReplayBuffer``
            (or anything exposing ``sample(batch_size) -> List[ReplayEntry]``).
        batch_size : int
        epochs : int

        Returns
        -------
        Dict with keys ``{n_samples, base_loss, ewc_penalty}``.
        """
        try:
            samples = replay_buffer.sample(batch_size=batch_size)
        except Exception as exc:
            logger.debug("ewc: replay sample failed: %s", exc)
            return {"n_samples": 0, "base_loss": 0.0, "ewc_penalty": 0.0}

        if not samples:
            return {"n_samples": 0, "base_loss": 0.0, "ewc_penalty": 0.0}

        # Build (features, labels) from replay entries: use state as features
        # and reward as the regression target.
        features_list: List[np.ndarray] = []
        labels_list: List[float] = []
        for entry in samples:
            state = getattr(entry, "state", None)
            reward = getattr(entry, "reward", None)
            if state is None or reward is None:
                continue
            try:
                feat = np.asarray(state, dtype=np.float64).ravel()
                if feat.size != self.feature_dim:
                    # Pad / truncate
                    padded = np.zeros(self.feature_dim, dtype=np.float64)
                    n = min(feat.size, self.feature_dim)
                    padded[:n] = feat[:n]
                    feat = padded
                features_list.append(feat)
                labels_list.append(float(reward))
            except Exception:
                continue

        if not features_list:
            return {"n_samples": 0, "base_loss": 0.0, "ewc_penalty": 0.0}

        features = np.stack(features_list, axis=0)
        labels = np.asarray(labels_list, dtype=np.float64)

        result = self.train_on_task(features=features, labels=labels, epochs=epochs)
        return {
            "n_samples": len(features_list),
            "base_loss": result.get("base_loss", 0.0),
            "ewc_penalty": result.get("ewc_penalty", 0.0),
        }

    # -- prediction ------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return raw linear predictions for ``features``. Always returns 1D."""

        X = self._pad(features)
        preds = X @ self.weights + self.bias
        return preds

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of learner state."""

        return {
            "feature_dim": self.feature_dim,
            "learning_rate": self.learning_rate,
            "ewc_lambda": self.ewc_lambda,
            "current_task": self.current_task,
            "task_history": list(self.task_history),
            "consolidated_tasks": list(self.anchors.keys()),
            "step_count": self.step_count,
            "last_base_loss": float(self.last_base_loss),
            "last_ewc_penalty": float(self.last_ewc_penalty),
            "weight_norm": float(np.linalg.norm(self.weights)),
            "fisher_samples": self._fisher.n_samples,
        }
