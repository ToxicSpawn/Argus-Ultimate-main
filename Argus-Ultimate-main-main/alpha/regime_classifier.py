"""regime_classifier.py — Push 45.

Hidden Markov Model (HMM) regime classifier.

States
------
  0 : bull      — positive drift, low volatility   -> scalar 1.3
  1 : sideways  — near-zero drift, mid volatility  -> scalar 1.0
  2 : bear      — negative drift, high volatility  -> scalar 0.6

Features (per bar)
------------------
  f0 : log return  = log(close_t / close_{t-1})
  f1 : realised vol = rolling std of log returns (window=20)

Model
-----
  GaussianHMM from hmmlearn (n_components=3, covariance_type='full').
  Falls back to CrossAssetRegime scalar if hmmlearn is not installed.

Refit cadence
-------------
  Model is (re)fitted every `refit_every` bars (default 100).
  First fit requires at least `min_fit_bars` bars (default 120).

Usage
-----
  clf = RegimeClassifier(refit_every=100)
  scalar = clf.update(candles)    # call once per bar
  label  = clf.regime_label       # 'bull' | 'sideways' | 'bear'
  probs  = clf.regime_probs       # shape (3,) latest state probabilities
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SCALARS = {
    "bull":     1.3,
    "sideways": 1.0,
    "bear":     0.6,
}
_STATE_NAMES  = ["bull", "sideways", "bear"]
_N_COMPONENTS = 3
_DEFAULT_REFIT_EVERY  = 100
_DEFAULT_MIN_FIT_BARS = 120
_DEFAULT_VOL_WINDOW   = 20
_DEFAULT_N_ITER       = 50
_FALLBACK_SCALAR      = 1.0


class RegimeClassifier:
    """HMM-based 3-state regime classifier.

    Parameters
    ----------
    refit_every   : Refit HMM every N bars (default 100)
    min_fit_bars  : Minimum bars before first fit (default 120)
    vol_window    : Rolling window for realised volatility feature (default 20)
    n_iter        : HMM EM iterations (default 50)
    random_state  : Reproducibility seed (default 42)
    """

    def __init__(
        self,
        refit_every:  int = _DEFAULT_REFIT_EVERY,
        min_fit_bars: int = _DEFAULT_MIN_FIT_BARS,
        vol_window:   int = _DEFAULT_VOL_WINDOW,
        n_iter:       int = _DEFAULT_N_ITER,
        random_state: int = 42,
    ) -> None:
        self._refit_every  = refit_every
        self._min_fit_bars = min_fit_bars
        self._vol_window   = vol_window
        self._n_iter       = n_iter
        self._random_state = random_state

        self._model             = None
        self._bars_since_refit  = 0
        self._fit_count         = 0
        self._last_state        = 1          # default: sideways
        self._last_scalar       = _FALLBACK_SCALAR
        self._last_probs        = np.array([0.0, 1.0, 0.0])  # sideways
        self._state_map: dict   = {}         # HMM state idx -> canonical idx

        self._has_hmmlearn = self._check_hmmlearn()
        if not self._has_hmmlearn:
            logger.warning(
                "hmmlearn not installed. RegimeClassifier will use CrossAssetRegime fallback. "
                "Install with: pip install hmmlearn"
            )
            from alpha.cross_asset import CrossAssetRegime
            self._fallback = CrossAssetRegime()
        else:
            self._fallback = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def regime_label(self) -> str:
        """Current regime: 'bull', 'sideways', or 'bear'."""
        return _STATE_NAMES[self._last_state]

    @property
    def regime_scalar(self) -> float:
        """Position-sizing scalar for current regime."""
        return self._last_scalar

    @property
    def regime_probs(self) -> np.ndarray:
        """Softmax state probabilities, shape (3,): [bull, sideways, bear]."""
        return self._last_probs.copy()

    @property
    def fit_count(self) -> int:
        """Number of times the HMM has been fitted."""
        return self._fit_count

    def update(self, candles: np.ndarray) -> float:
        """Ingest latest candles and return regime scalar.

        Parameters
        ----------
        candles : np.ndarray shape (N, 6)  [ts, open, high, low, close, vol]

        Returns
        -------
        float : regime scalar (1.3 / 1.0 / 0.6)
        """
        if not self._has_hmmlearn:
            return self._fallback_update(candles)

        if len(candles) < self._min_fit_bars:
            return self._last_scalar

        features = self._build_features(candles)
        if features is None or len(features) < self._min_fit_bars - 1:
            return self._last_scalar

        # Refit on schedule
        self._bars_since_refit += 1
        needs_fit = (
            self._model is None or
            self._bars_since_refit >= self._refit_every
        )
        if needs_fit:
            self._fit(features)
            self._bars_since_refit = 0

        if self._model is None:
            return self._last_scalar

        # Decode current state
        try:
            state_seq = self._model.predict(features)
            current_raw = int(state_seq[-1])
            canonical   = self._state_map.get(current_raw, 1)  # default sideways
            self._last_state  = canonical
            self._last_scalar = _SCALARS[_STATE_NAMES[canonical]]

            # Posterior probabilities for last observation
            posteriors = self._model.predict_proba(features)
            raw_probs  = posteriors[-1]   # shape (n_components,)
            canon_probs = np.zeros(3)
            for raw_idx, can_idx in self._state_map.items():
                if raw_idx < len(raw_probs):
                    canon_probs[can_idx] += raw_probs[raw_idx]
            self._last_probs = canon_probs

        except Exception as exc:
            logger.debug("HMM decode failed: %s", exc)

        return self._last_scalar

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_features(self, candles: np.ndarray) -> Optional[np.ndarray]:
        """Build (log_return, realised_vol) feature matrix."""
        try:
            closes = candles[:, 4].astype(float)
            if len(closes) < 2:
                return None
            log_rets = np.log(closes[1:] / np.maximum(closes[:-1], 1e-10))
            # Rolling realised vol
            rv = np.array([
                log_rets[max(0, i - self._vol_window): i + 1].std()
                for i in range(len(log_rets))
            ])
            features = np.column_stack([log_rets, rv]).astype(np.float64)
            # Remove NaN / Inf rows
            mask = np.isfinite(features).all(axis=1)
            return features[mask]
        except Exception as exc:
            logger.debug("Feature build failed: %s", exc)
            return None

    def _fit(self, features: np.ndarray) -> None:
        """Fit GaussianHMM and build state -> canonical index map."""
        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(
                n_components   = _N_COMPONENTS,
                covariance_type= "full",
                n_iter         = self._n_iter,
                random_state   = self._random_state,
                verbose        = False,
            )
            model.fit(features)
            self._model = model
            self._fit_count += 1
            self._state_map = self._assign_states(model)
            logger.debug(
                "HMM refit #%d | state_map=%s means=%s",
                self._fit_count, self._state_map,
                model.means_.round(6).tolist(),
            )
        except Exception as exc:
            logger.warning("HMM fit failed: %s", exc)
            self._model = None

    def _assign_states(
        self, model
    ) -> dict:
        """Map HMM state indices to canonical [bull=0, sideways=1, bear=2].

        Strategy: rank by mean log-return of each state.
          highest return -> bull (0)
          middle return  -> sideways (1)
          lowest return  -> bear (2)
        """
        means = model.means_[:, 0]   # log-return dimension
        order = np.argsort(means)[::-1]   # descending
        state_map = {}
        for canonical_idx, raw_idx in enumerate(order):
            state_map[int(raw_idx)] = canonical_idx
        return state_map

    def _fallback_update(self, candles: np.ndarray) -> float:
        """Use CrossAssetRegime scalar when hmmlearn is unavailable."""
        try:
            scalar = self._fallback.get_scalar(candles)
            self._last_scalar = scalar
            if scalar > 1.1:
                self._last_state = 0   # bull
            elif scalar < 0.85:
                self._last_state = 2   # bear
            else:
                self._last_state = 1   # sideways
            return scalar
        except Exception:
            return _FALLBACK_SCALAR

    @staticmethod
    def _check_hmmlearn() -> bool:
        try:
            import hmmlearn  # noqa: F401
            return True
        except ImportError:
            return False
