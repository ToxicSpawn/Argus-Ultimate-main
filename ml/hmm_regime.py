"""
Hidden Markov Model Regime Detector — unsupervised market regime classification.

Uses a Gaussian HMM with 3-5 hidden states to classify market regimes from
returns and volatility data. Unlike rule-based classifiers, HMM discovers
regime structure from data.

States typically discovered:
  - Low volatility trending
  - High volatility ranging
  - Crash/tail risk
  - Recovery

Requires: hmmlearn (optional). Falls back to volatility-based classifier if unavailable.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import hmmlearn; availability checked at runtime.
try:
    from hmmlearn import hmm as _hmmlearn_hmm  # type: ignore

    _HMMLEARN_AVAILABLE = True
except ImportError:
    _hmmlearn_hmm = None
    _HMMLEARN_AVAILABLE = False
    logger.info("hmmlearn not available — HMM regime detector will use volatility-based fallback.")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HMMRegimeState:
    """Descriptor for a single HMM hidden state."""

    state_id: int
    label: str
    mean_return: float
    volatility: float
    sharpe_like: float
    probability: float


# ---------------------------------------------------------------------------
# Regime label constants
# ---------------------------------------------------------------------------

REGIME_TREND_UP = "TREND_UP"
REGIME_TREND_DOWN = "TREND_DOWN"
REGIME_RANGE = "RANGE"
REGIME_HIGH_VOL = "HIGH_VOL"
REGIME_CRISIS = "CRISIS"

_ARGUS_LABELS_4 = [REGIME_TREND_DOWN, REGIME_RANGE, REGIME_TREND_UP, REGIME_HIGH_VOL]
_ARGUS_LABELS_3 = [REGIME_TREND_DOWN, REGIME_RANGE, REGIME_TREND_UP]
_ARGUS_LABELS_5 = [REGIME_CRISIS, REGIME_TREND_DOWN, REGIME_RANGE, REGIME_TREND_UP, REGIME_HIGH_VOL]


def _default_labels(n_states: int) -> List[str]:
    mapping = {3: _ARGUS_LABELS_3, 4: _ARGUS_LABELS_4, 5: _ARGUS_LABELS_5}
    if n_states in mapping:
        return mapping[n_states]
    # Generic fallback for arbitrary counts
    base = [REGIME_CRISIS, REGIME_TREND_DOWN, REGIME_RANGE, REGIME_TREND_UP, REGIME_HIGH_VOL]
    return base[:n_states] if n_states <= 5 else base + [f"STATE_{i}" for i in range(5, n_states)]


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class HMMRegimeDetector:
    """
    Unsupervised market regime classifier using a Gaussian HMM.

    Parameters
    ----------
    n_states : int
        Number of hidden states (3-5 recommended).
    n_iter : int
        Maximum EM iterations for HMM training.
    min_history : int
        Minimum number of return observations required before fitting.
    state_labels : list[str] | None
        Custom labels for states (ordered from lowest to highest mean return).
        Defaults to ARGUS regime vocabulary.
    """

    N_STATES: int = 4

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 100,
        min_history: int = 60,
        state_labels: Optional[List[str]] = None,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2.")
        self.n_states = n_states
        self.n_iter = n_iter
        self.min_history = min_history
        self.state_labels: List[str] = state_labels if state_labels else _default_labels(n_states)
        if len(self.state_labels) != self.n_states:
            raise ValueError(
                f"state_labels length ({len(self.state_labels)}) must equal n_states ({self.n_states})."
            )

        self._model: Optional[object] = None  # GaussianHMM or None
        self._fitted: bool = False

        # State info populated after fitting
        self._state_info: List[HMMRegimeState] = []

        # Mapping: HMM raw state index → label index (sorted by mean_return)
        self._state_order: List[int] = []

        # Fallback percentile thresholds (set during fit for fallback path)
        self._fallback_vol_p25: float = 0.0
        self._fallback_vol_p75: float = 0.0
        self._fallback_vol_p90: float = 0.0
        self._fallback_mean_ret: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> bool:
        """
        Fit regime detector to a returns series.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of period returns (e.g. daily log-returns).

        Returns
        -------
        bool
            True if model converged / fitted successfully.
        """
        returns = np.asarray(returns, dtype=float).ravel()
        if len(returns) < self.min_history:
            logger.warning(
                "HMMRegimeDetector.fit: only %d observations, need %d. Not fitted.",
                len(returns),
                self.min_history,
            )
            return False

        if _HMMLEARN_AVAILABLE:
            return self._fit_hmm(returns)
        else:
            return self._fit_fallback(returns)

    def predict(self, returns: np.ndarray) -> str:
        """
        Predict the current regime label for the most recent return window.

        Parameters
        ----------
        returns : np.ndarray
            Recent returns used for prediction (last observation is most recent).

        Returns
        -------
        str
            One of the ARGUS regime label constants.
        """
        if not self._fitted:
            logger.warning("HMMRegimeDetector.predict called before fit — returning RANGE.")
            return REGIME_RANGE

        returns = np.asarray(returns, dtype=float).ravel()
        if _HMMLEARN_AVAILABLE and self._model is not None:
            return self._predict_hmm(returns)
        else:
            return self._predict_fallback(returns)

    def predict_proba(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Return probability distribution over regime labels.

        Parameters
        ----------
        returns : np.ndarray
            Recent returns.

        Returns
        -------
        dict
            Mapping from regime label → probability (sums to 1.0).
        """
        if not self._fitted:
            n = len(self.state_labels)
            return {label: 1.0 / n for label in self.state_labels}

        returns = np.asarray(returns, dtype=float).ravel()
        if _HMMLEARN_AVAILABLE and self._model is not None:
            return self._proba_hmm(returns)
        else:
            return self._proba_fallback(returns)

    def get_state_info(self) -> List[HMMRegimeState]:
        """Return descriptors for all fitted states, ordered by mean_return."""
        return list(self._state_info)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Forward transition probability forecast
    # ------------------------------------------------------------------

    def predict_transition_probs(
        self,
        returns: np.ndarray,
        horizon_steps: int = 12,
    ) -> Dict[str, float]:
        """
        Forecast the probability distribution over regimes ``horizon_steps``
        steps in the future, given the current most-likely state.

        Uses the HMM transition matrix: P(t+h) = P(t) @ transmat^h.
        Falls back to ``predict_proba`` when the HMM model is unavailable
        (i.e. equal to the current distribution — no forecasting information).

        Parameters
        ----------
        returns : np.ndarray
            Recent return observations (same as for ``predict_proba``).
        horizon_steps : int
            Number of steps to project forward (e.g. 12 bars = 1 hour
            on 5-min bars).

        Returns
        -------
        dict[str, float]
            Regime label → forward probability. Sums to 1.0.
        """
        if not self._fitted or not _HMMLEARN_AVAILABLE or self._model is None:
            # Fall back: current distribution = best estimate of future
            return self.predict_proba(returns)

        # Get current-state posterior distribution (over raw HMM states)
        returns = np.asarray(returns, dtype=float).ravel()
        obs = self._build_observation_matrix(returns)
        try:
            _, posteriors = self._model.score_samples(obs)
            current_proba = posteriors[-1]  # shape (n_states,)
        except Exception as exc:
            logger.warning("predict_transition_probs score_samples error: %s", exc)
            return self.predict_proba(returns)

        # Project forward: P(t+h) = P(t) @ transmat^h
        try:
            transmat = np.array(self._model.transmat_)  # shape (n_states, n_states)
            transmat_h = np.linalg.matrix_power(transmat, int(max(1, horizon_steps)))
            future_raw = current_proba @ transmat_h  # shape (n_states,)
        except Exception as exc:
            logger.warning("predict_transition_probs matrix_power error: %s", exc)
            return self.predict_proba(returns)

        # Map raw state indices → label indices via _state_order
        result: Dict[str, float] = {}
        for label_idx, raw_state in enumerate(self._state_order):
            label = self.state_labels[label_idx]
            result[label] = float(np.clip(future_raw[raw_state], 0.0, 1.0))

        # Normalise (matrix_power may introduce tiny float errors)
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    # ------------------------------------------------------------------
    # HMM path (hmmlearn available)
    # ------------------------------------------------------------------

    def _fit_hmm(self, returns: np.ndarray) -> bool:
        obs = self._build_observation_matrix(returns)
        model = _hmmlearn_hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(obs)
            except Exception as exc:
                logger.warning("HMM fit failed (%s) — using fallback.", exc)
                return self._fit_fallback(returns)

        converged: bool = model.monitor_.converged  # type: ignore[attr-defined]
        if not converged:
            logger.warning("HMM did not converge after %d iterations.", self.n_iter)

        self._model = model
        self._build_state_info_hmm(returns, model)
        self._fitted = True
        logger.info(
            "HMM fitted: %d states, converged=%s", self.n_states, converged
        )
        return converged

    def _predict_hmm(self, returns: np.ndarray) -> str:
        obs = self._build_observation_matrix(returns)
        try:
            raw_states = self._model.predict(obs)  # type: ignore[union-attr]
            last_raw = int(raw_states[-1])
            label_idx = self._state_order.index(last_raw)
            return self.state_labels[label_idx]
        except Exception as exc:
            logger.warning("HMM predict error (%s) — returning RANGE.", exc)
            return REGIME_RANGE

    def _proba_hmm(self, returns: np.ndarray) -> Dict[str, float]:
        obs = self._build_observation_matrix(returns)
        try:
            _, posteriors = self._model.score_samples(obs)  # type: ignore[union-attr]
            last_post = posteriors[-1]  # shape (n_states,)
            proba: Dict[str, float] = {}
            for label_idx, raw_state in enumerate(self._state_order):
                label = self.state_labels[label_idx]
                proba[label] = float(last_post[raw_state])
            return proba
        except Exception as exc:
            logger.warning("HMM score_samples error (%s) — uniform proba.", exc)
            n = len(self.state_labels)
            return {label: 1.0 / n for label in self.state_labels}

    def _build_state_info_hmm(self, returns: np.ndarray, model: object) -> None:
        """Populate _state_info and _state_order from fitted HMM."""
        means = model.means_[:, 0]  # type: ignore[union-attr]
        # Sort raw states by mean return ascending
        order = list(np.argsort(means))
        self._state_order = order

        # Decode observations to get stationary distribution
        obs = self._build_observation_matrix(returns)
        try:
            raw_states = model.predict(obs)  # type: ignore[union-attr]
            counts = np.bincount(raw_states, minlength=self.n_states)
            probs = counts / counts.sum()
        except Exception:
            probs = np.ones(self.n_states) / self.n_states

        covars = model.covars_  # type: ignore[union-attr]  # shape (n_states, n_features, n_features) or diag

        self._state_info = []
        for label_idx, raw in enumerate(order):
            mean_ret = float(means[raw])
            if covars.ndim == 3:
                vol = float(np.sqrt(covars[raw, 0, 0]))
            else:
                vol = float(np.sqrt(covars[raw, 0]))
            sharpe = mean_ret / vol if vol > 1e-12 else 0.0
            self._state_info.append(
                HMMRegimeState(
                    state_id=raw,
                    label=self.state_labels[label_idx],
                    mean_return=mean_ret,
                    volatility=vol,
                    sharpe_like=sharpe,
                    probability=float(probs[raw]),
                )
            )

    # ------------------------------------------------------------------
    # Fallback path (no hmmlearn)
    # ------------------------------------------------------------------

    def _fit_fallback(self, returns: np.ndarray) -> bool:
        """
        Simple volatility-percentile classifier.
        Computes rolling 20-period vol and stores quantile thresholds.
        """
        window = min(20, len(returns) // 3)
        vols = self._rolling_vol(returns, window)

        self._fallback_vol_p25 = float(np.percentile(vols, 25))
        self._fallback_vol_p75 = float(np.percentile(vols, 75))
        self._fallback_vol_p90 = float(np.percentile(vols, 90))
        self._fallback_mean_ret = float(np.mean(returns))

        # Build synthetic state info for API consistency
        self._state_info = [
            HMMRegimeState(0, REGIME_RANGE, 0.0, self._fallback_vol_p25, 0.0, 0.25),
            HMMRegimeState(1, REGIME_TREND_UP, max(self._fallback_mean_ret, 0.0001), self._fallback_vol_p25, 1.0, 0.35),
            HMMRegimeState(2, REGIME_HIGH_VOL, 0.0, self._fallback_vol_p75, 0.0, 0.30),
            HMMRegimeState(3, REGIME_CRISIS, -0.01, self._fallback_vol_p90 * 1.2, -1.0, 0.10),
        ][: self.n_states]
        self._state_order = list(range(self.n_states))
        self._fitted = True
        logger.info("Fallback vol-based regime detector fitted.")
        return True

    def _predict_fallback(self, returns: np.ndarray) -> str:
        window = min(20, len(returns))
        recent = returns[-window:]
        vol = float(np.std(recent)) if len(recent) > 1 else 0.0
        mean_ret = float(np.mean(recent))

        if vol >= self._fallback_vol_p90:
            return REGIME_CRISIS
        if vol >= self._fallback_vol_p75:
            return REGIME_HIGH_VOL
        if vol <= self._fallback_vol_p25:
            return REGIME_RANGE
        # Mid vol — use direction
        return REGIME_TREND_UP if mean_ret >= 0 else REGIME_TREND_DOWN

    def _proba_fallback(self, returns: np.ndarray) -> Dict[str, float]:
        regime = self._predict_fallback(returns)
        proba = {label: 0.05 for label in self.state_labels}
        if regime in proba:
            proba[regime] = 1.0 - 0.05 * (len(self.state_labels) - 1)
        total = sum(proba.values())
        return {k: v / total for k, v in proba.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_vol(returns: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation, padding early values with global std."""
        n = len(returns)
        vols = np.empty(n)
        global_std = float(np.std(returns)) if n > 1 else 1e-6
        for i in range(n):
            start = max(0, i - window + 1)
            segment = returns[start : i + 1]
            vols[i] = float(np.std(segment)) if len(segment) > 1 else global_std
        return vols

    @staticmethod
    def _build_observation_matrix(returns: np.ndarray) -> np.ndarray:
        """
        Build 2-column observation matrix: [return, rolling_vol].
        hmmlearn expects shape (T, n_features).
        """
        n = len(returns)
        window = min(20, max(2, n // 5))
        vols = np.empty(n)
        global_std = float(np.std(returns)) if n > 1 else 1e-6
        for i in range(n):
            start = max(0, i - window + 1)
            seg = returns[start : i + 1]
            vols[i] = float(np.std(seg)) if len(seg) > 1 else global_std
        obs = np.column_stack([returns, vols])
        return obs.astype(np.float64)
