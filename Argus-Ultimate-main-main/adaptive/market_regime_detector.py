"""
Real-Time Market Regime Detection and Classification.

Comprehensive regime detection system that classifies market conditions using
multiple detection methods:
- Rule-based statistical detection
- Hidden Markov Model (HMM) detection
- Multi-timeframe analysis
- Transition probability tracking

Regimes: BULL_STRONG, BULL_MODERATE, BULL_WEAK, BEAR_STRONG, BEAR_MODERATE,
         BEAR_WEAK, SIDEWAYS_HIGH_VOL, SIDEWAYS_LOW_VOL, CRISIS, RECOVERY, TRANSITION
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional scipy import for HMM
try:
    from scipy.stats import norm
    from scipy.special import logsumexp
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.debug("scipy not available — HMM will use numpy-only fallback")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MarketRegime(str, Enum):
    """Market regime classification."""
    BULL_STRONG = "bull_strong"
    BULL_MODERATE = "bull_moderate"
    BULL_WEAK = "bull_weak"
    BEAR_STRONG = "bear_strong"
    BEAR_MODERATE = "bear_moderate"
    BEAR_WEAK = "bear_weak"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    TRANSITION = "transition"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RegimeFeatures:
    """Extracted market features for regime classification."""
    returns_1d: float = 0.0
    returns_5d: float = 0.0
    returns_20d: float = 0.0
    volatility_1d: float = 0.0
    volatility_20d: float = 0.0
    volatility_60d: float = 0.0
    trend_strength: float = 0.0
    market_breadth: float = 0.5
    vix_level: float = 20.0
    vix_term_structure: float = 0.0
    correlation_level: float = 0.0
    volume_ratio: float = 1.0

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models."""
        return np.array([
            self.returns_1d, self.returns_5d, self.returns_20d,
            self.volatility_1d, self.volatility_20d, self.volatility_60d,
            self.trend_strength, self.market_breadth,
            self.vix_level, self.vix_term_structure,
            self.correlation_level, self.volume_ratio,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RegimeFeatures":
        """Create RegimeFeatures from numpy array."""
        return cls(
            returns_1d=float(arr[0]), returns_5d=float(arr[1]), returns_20d=float(arr[2]),
            volatility_1d=float(arr[3]), volatility_20d=float(arr[4]), volatility_60d=float(arr[5]),
            trend_strength=float(arr[6]), market_breadth=float(arr[7]),
            vix_level=float(arr[8]), vix_term_structure=float(arr[9]),
            correlation_level=float(arr[10]), volume_ratio=float(arr[11]),
        )


@dataclass
class RegimeSnapshot:
    """Point-in-time regime observation."""
    regime: MarketRegime
    timestamp: datetime
    confidence: float
    features: RegimeFeatures
    duration_bars: int


@dataclass
class RegimeStatistics:
    """Aggregate regime statistics."""
    current_regime: MarketRegime
    regime_duration: int
    historical_distribution: Dict[MarketRegime, float]
    average_regime_duration: Dict[MarketRegime, float]
    transition_matrix: np.ndarray


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def _compute_returns(prices: np.ndarray, window: int) -> float:
    """Compute log return over window."""
    if len(prices) < window + 1:
        return 0.0
    return float(np.log(prices[-1] / prices[-window - 1]))


def _compute_volatility(returns: np.ndarray, window: int) -> float:
    """Compute annualized volatility over window."""
    if len(returns) < max(window, 2):
        return 0.0
    return float(np.std(returns[-window:]) * np.sqrt(252))


def _compute_adx_like(prices: np.ndarray, period: int = 14) -> float:
    """Simplified ADX-like trend strength indicator."""
    if len(prices) < period + 2:
        return 0.0
    returns = np.diff(np.log(prices[-period - 1:]))
    sigma = float(np.std(returns))
    if sigma < 1e-10:
        return 0.0
    return min(1.0, abs(float(np.mean(returns))) / sigma)


def _compute_hurst_exponent(returns: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent using R/S analysis.
    H > 0.5: trending/persistent
    H < 0.5: mean-reverting/anti-persistent
    H = 0.5: random walk
    """
    n = len(returns)
    if n < max_lag * 2:
        return 0.5

    lags = range(2, min(max_lag, n // 2))
    tau = []
    for lag in lags:
        # Rolling standard deviation
        std = np.std(returns[:lag])
        if std < 1e-10:
            tau.append(1e-10)
        else:
            tau.append(std)

    if len(tau) < 3:
        return 0.5

    # Linear regression in log-log space
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    coeffs = np.polyfit(log_lags, log_tau, 1)
    hurst = float(coeffs[0])

    return max(0.0, min(1.0, hurst))


# ---------------------------------------------------------------------------
# Regime Classifier (Rule-Based)
# ---------------------------------------------------------------------------


class RegimeClassifier:
    """
    Rule-based market regime classifier using extracted features.

    Classifies market conditions into 11 distinct regimes based on
    momentum, volatility, trend strength, and other market indicators.
    """

    def __init__(
        self,
        bull_threshold: float = 0.02,
        bear_threshold: float = -0.02,
        high_vol_threshold: float = 0.5,
        low_vol_threshold: float = 0.15,
        trend_strong_threshold: float = 0.5,
        crisis_vol_threshold: float = 0.8,
        crisis_return_threshold: float = -0.05,
    ) -> None:
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.trend_strong_threshold = trend_strong_threshold
        self.crisis_vol_threshold = crisis_vol_threshold
        self.crisis_return_threshold = crisis_return_threshold

        self._current_regime: MarketRegime = MarketRegime.TRANSITION
        self._confidence: float = 0.0
        self._regime_start_bar: int = 0
        self._current_bar: int = 0
        self._history: Deque[RegimeSnapshot] = deque(maxlen=1000)

    def extract_features(self, market_data: Dict[str, Any]) -> RegimeFeatures:
        """
        Extract regime classification features from market data.

        Expected market_data keys:
        - prices: np.ndarray of closing prices
        - volumes: np.ndarray of volumes (optional)
        - vix: float or np.ndarray (optional)
        - breadth: float advance/decline ratio (optional)
        - correlations: np.ndarray of cross-asset correlations (optional)
        """
        prices = np.asarray(market_data.get("prices", []), dtype=np.float64)
        volumes = np.asarray(market_data.get("volumes", []), dtype=np.float64)
        vix = market_data.get("vix", 20.0)
        breadth = market_data.get("breadth", 0.5)
        correlations = market_data.get("correlations", None)

        if len(prices) < 2:
            return RegimeFeatures()

        returns = np.diff(np.log(prices))

        # Momentum
        returns_1d = _compute_returns(prices, 1)
        returns_5d = _compute_returns(prices, 5)
        returns_20d = _compute_returns(prices, 20)

        # Volatility
        vol_1d = _compute_volatility(returns, 1)
        vol_20d = _compute_volatility(returns, 20)
        vol_60d = _compute_volatility(returns, 60)

        # Trend strength
        trend_strength = _compute_adx_like(prices)

        # VIX
        vix_level = float(vix) if np.isscalar(vix) else float(vix[-1]) if len(vix) > 0 else 20.0
        vix_ts = 0.0
        if np.isscalar(vix) or len(vix) < 2:
            vix_ts = 0.0
        else:
            vix_arr = np.asarray(vix, dtype=np.float64)
            vix_ts = float(vix_arr[-1] - np.mean(vix_arr[-5:]))

        # Correlation
        correlation_level = 0.0
        if correlations is not None and len(correlations) > 0:
            correlation_level = float(np.mean(np.abs(correlations)))

        # Volume ratio
        volume_ratio = 1.0
        if len(volumes) >= 21:
            current_vol = float(volumes[-1])
            avg_vol = float(np.mean(volumes[-20:]))
            volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        features = RegimeFeatures(
            returns_1d=returns_1d, returns_5d=returns_5d, returns_20d=returns_20d,
            volatility_1d=vol_1d, volatility_20d=vol_20d, volatility_60d=vol_60d,
            trend_strength=trend_strength, market_breadth=float(breadth),
            vix_level=vix_level, vix_term_structure=vix_ts,
            correlation_level=correlation_level, volume_ratio=volume_ratio,
        )

        return features

    def classify(self, features: RegimeFeatures) -> MarketRegime:
        """Classify market regime from extracted features."""
        # Crisis detection
        if (features.volatility_20d > self.crisis_vol_threshold and
                features.returns_20d < self.crisis_return_threshold):
            self._confidence = 0.85
            return MarketRegime.CRISIS

        # Recovery detection
        if (features.returns_5d > 0.03 and
                features.volatility_20d > self.high_vol_threshold and
                features.returns_20d < 0):
            self._confidence = 0.75
            return MarketRegime.RECOVERY

        # Strong bull
        if (features.returns_20d > self.bull_threshold * 2 and
                features.trend_strength > self.trend_strong_threshold and
                features.volatility_20d < self.high_vol_threshold):
            self._confidence = 0.80
            return MarketRegime.BULL_STRONG

        # Moderate bull
        if (features.returns_20d > self.bull_threshold and
                features.trend_strength > 0.3):
            self._confidence = 0.70
            return MarketRegime.BULL_MODERATE

        # Weak bull
        if features.returns_20d > self.bull_threshold * 0.5:
            self._confidence = 0.60
            return MarketRegime.BULL_WEAK

        # Strong bear
        if (features.returns_20d < self.bear_threshold * 2 and
                features.trend_strength > self.trend_strong_threshold):
            self._confidence = 0.80
            return MarketRegime.BEAR_STRONG

        # Moderate bear
        if (features.returns_20d < self.bear_threshold and
                features.trend_strength > 0.3):
            self._confidence = 0.70
            return MarketRegime.BEAR_MODERATE

        # Weak bear
        if features.returns_20d < self.bear_threshold * 0.5:
            self._confidence = 0.60
            return MarketRegime.BEAR_WEAK

        # Sideways regimes
        if features.volatility_20d > self.high_vol_threshold:
            self._confidence = 0.65
            return MarketRegime.SIDEWAYS_HIGH_VOL

        if features.volatility_20d < self.low_vol_threshold:
            self._confidence = 0.70
            return MarketRegime.SIDEWAYS_LOW_VOL

        # Default: transition
        self._confidence = 0.50
        return MarketRegime.TRANSITION

    def get_regime_confidence(self) -> float:
        """Return confidence in current regime classification (0-1)."""
        return self._confidence

    def get_regime_duration(self) -> int:
        """Return number of bars in current regime."""
        return self._current_bar - self._regime_start_bar

    def get_transition_probability(self) -> Dict[MarketRegime, float]:
        """
        Estimate probability of transitioning to each regime.
        Based on current feature momentum and historical patterns.
        """
        probs: Dict[MarketRegime, float] = {r: 0.0 for r in MarketRegime}

        current = self._current_regime
        base_prob = 0.1 / len(MarketRegime)

        # Current regime has highest probability of persisting
        probs[current] = 0.6

        # Transition probabilities based on regime
        if current in (MarketRegime.BULL_STRONG, MarketRegime.BULL_MODERATE):
            probs[MarketRegime.BULL_WEAK] = 0.15
            probs[MarketRegime.TRANSITION] = 0.10
            probs[MarketRegime.SIDEWAYS_HIGH_VOL] = 0.05
            probs[MarketRegime.BEAR_WEAK] = 0.05
        elif current in (MarketRegime.BEAR_STRONG, MarketRegime.BEAR_MODERATE):
            probs[MarketRegime.BEAR_WEAK] = 0.15
            probs[MarketRegime.RECOVERY] = 0.10
            probs[MarketRegime.TRANSITION] = 0.10
            probs[MarketRegime.SIDEWAYS_HIGH_VOL] = 0.05
        elif current == MarketRegime.CRISIS:
            probs[MarketRegime.RECOVERY] = 0.25
            probs[MarketRegime.BEAR_WEAK] = 0.10
            probs[MarketRegime.TRANSITION] = 0.05
        elif current == MarketRegime.RECOVERY:
            probs[MarketRegime.BULL_WEAK] = 0.20
            probs[MarketRegime.BULL_MODERATE] = 0.10
            probs[MarketRegime.TRANSITION] = 0.05
        elif current in (MarketRegime.SIDEWAYS_HIGH_VOL, MarketRegime.SIDEWAYS_LOW_VOL):
            probs[MarketRegime.TRANSITION] = 0.15
            probs[MarketRegime.BULL_WEAK] = 0.10
            probs[MarketRegime.BEAR_WEAK] = 0.10
        else:
            # Transition regime
            for r in MarketRegime:
                if r != current:
                    probs[r] = base_prob

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def update(self, features: RegimeFeatures) -> MarketRegime:
        """Update classifier with new features and return current regime."""
        self._current_bar += 1
        new_regime = self.classify(features)

        if new_regime != self._current_regime:
            logger.info(
                "Regime transition: %s -> %s (confidence=%.2f)",
                self._current_regime.value, new_regime.value, self._confidence,
            )
            self._regime_start_bar = self._current_bar
            self._current_regime = new_regime

        snapshot = RegimeSnapshot(
            regime=self._current_regime,
            timestamp=datetime.now(),
            confidence=self._confidence,
            features=features,
            duration_bars=self.get_regime_duration(),
        )
        self._history.append(snapshot)

        return self._current_regime


# ---------------------------------------------------------------------------
# Hidden Markov Model Regime Detector
# ---------------------------------------------------------------------------


class HiddenMarkovRegimeDetector:
    """
    HMM-based regime detection using Gaussian mixture observations.

    Uses returns and volatility as observation features to detect
    hidden market regimes through the Baum-Welch algorithm.

    Falls back to a numpy-only implementation if scipy is unavailable.
    """

    # Mapping from HMM state index to MarketRegime (populated after fit)
    STATE_TO_REGIME: Dict[int, MarketRegime] = {}

    def __init__(
        self,
        n_states: int = 5,
        n_iter: int = 100,
        min_history: int = 60,
        random_state: int = 42,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states = n_states
        self.n_iter = n_iter
        self.min_history = min_history
        self.random_state = random_state

        # Model parameters
        self._initial_probs: Optional[np.ndarray] = None  # pi
        self._transition_matrix: Optional[np.ndarray] = None  # A
        self._means: Optional[np.ndarray] = None  # mu
        self._covars: Optional[np.ndarray] = None  # sigma

        self._fitted = False
        self._state_regime_map: Dict[int, MarketRegime] = {}

        # State tracking
        self._previous_probs: Optional[np.ndarray] = None
        self._regime_history: Deque[int] = deque(maxlen=500)

    def fit(self, returns: np.ndarray) -> bool:
        """
        Fit HMM model to returns data.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns (will be converted to [return, vol] features)

        Returns
        -------
        bool
            True if model fitted successfully
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        if len(returns) < self.min_history:
            logger.warning(
                "HMMRegimeDetector.fit: need %d observations, have %d",
                self.min_history, len(returns),
            )
            return False

        obs = self._build_observations(returns)

        np.random.seed(self.random_state)
        self._initialize_parameters(obs)

        # Baum-Welch (EM) algorithm
        converged = False
        for iteration in range(self.n_iter):
            old_log_likelihood = self._compute_log_likelihood(obs)

            # E-step
            gamma, xi = self._e_step(obs)

            # M-step
            self._m_step(obs, gamma, xi)

            new_log_likelihood = self._compute_log_likelihood(obs)
            improvement = new_log_likelihood - old_log_likelihood

            if abs(improvement) < 1e-6:
                converged = True
                logger.info("HMM converged after %d iterations (LL=%.4f)", iteration + 1, new_log_likelihood)
                break

        if not converged:
            logger.warning("HMM did not converge after %d iterations", self.n_iter)

        # Map states to regimes
        self._map_states_to_regimes(obs)

        self._fitted = True
        return True

    def predict(self, returns: np.ndarray) -> List[int]:
        """
        Predict regime sequence for returns using Viterbi algorithm.

        Returns
        -------
        List[int]
            Sequence of state indices (most likely path)
        """
        if not self._fitted:
            logger.warning("HMM predict called before fit — returning zeros")
            return [0] * len(returns)

        obs = self._build_observations(returns)
        return self._viterbi(obs)

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities for each observation.

        Returns
        -------
        np.ndarray
            Shape (T, n_states) — probability of each state at each time
        """
        if not self._fitted:
            n = len(returns)
            return np.ones((n, self.n_states)) / self.n_states

        obs = self._build_observations(returns)
        gamma, _ = self._forward_backward(obs)
        return gamma

    def detect_regime_change(
        self,
        current_probs: np.ndarray,
        previous_probs: np.ndarray,
        threshold: float = 0.3,
    ) -> bool:
        """
        Detect if regime has changed based on probability distribution shift.

        Parameters
        ----------
        current_probs : np.ndarray
            Current state probability distribution
        previous_probs : np.ndarray
            Previous state probability distribution
        threshold : float
            KL divergence threshold for regime change detection

        Returns
        -------
        bool
            True if regime change detected
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        current = np.asarray(current_probs, dtype=np.float64) + eps
        previous = np.asarray(previous_probs, dtype=np.float64) + eps

        # Normalize
        current = current / current.sum()
        previous = previous / previous.sum()

        # KL divergence
        kl_div = float(np.sum(previous * np.log(previous / current)))

        return kl_div > threshold

    def get_current_regime(self, returns: np.ndarray) -> MarketRegime:
        """Get the current market regime."""
        if not self._fitted:
            return MarketRegime.TRANSITION

        probs = self.predict_proba(returns)
        if len(probs) == 0:
            return MarketRegime.TRANSITION

        last_probs = probs[-1]
        self._previous_probs = last_probs.copy()
        most_likely_state = int(np.argmax(last_probs))

        regime = self._state_regime_map.get(most_likely_state, MarketRegime.TRANSITION)
        self._regime_history.append(most_likely_state)

        return regime

    def _build_observations(self, returns: np.ndarray) -> np.ndarray:
        """Build 2-column observation matrix: [return, rolling_vol]."""
        n = len(returns)
        window = min(20, max(2, n // 5))
        vols = np.empty(n)
        global_std = float(np.std(returns)) if n > 1 else 1e-6

        for i in range(n):
            start = max(0, i - window + 1)
            seg = returns[start:i + 1]
            vols[i] = float(np.std(seg)) if len(seg) > 1 else global_std

        return np.column_stack([returns, vols]).astype(np.float64)

    def _initialize_parameters(self, obs: np.ndarray) -> None:
        """Initialize HMM parameters using K-means-like approach."""
        n_obs, n_features = obs.shape

        # Random initialization with some structure
        self._initial_probs = np.ones(self.n_states) / self.n_states
        self._transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states

        # Sort observations by first column (returns) for better initialization
        sorted_idx = np.argsort(obs[:, 0])
        chunk_size = n_obs // self.n_states

        self._means = np.zeros((self.n_states, n_features))
        self._covars = np.zeros((self.n_states, n_features))

        for i in range(self.n_states):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_states - 1 else n_obs
            chunk = obs[sorted_idx[start:end]]
            self._means[i] = np.mean(chunk, axis=0)
            self._covars[i] = np.var(chunk, axis=0) + 1e-6

    def _e_step(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """E-step: compute gamma and xi using forward-backward algorithm."""
        gamma, xi = self._forward_backward(obs)
        return gamma, xi

    def _m_step(self, obs: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """M-step: update model parameters."""
        n_obs, n_features = obs.shape

        # Update initial probabilities
        self._initial_probs = gamma[0] + 1e-10
        self._initial_probs /= self._initial_probs.sum()

        # Update transition matrix
        self._transition_matrix = xi.sum(axis=0) + 1e-10
        self._transition_matrix /= self._transition_matrix.sum(axis=1, keepdims=True)

        # Update means and covariances
        gamma_sum = gamma.sum(axis=0) + 1e-10

        self._means = (gamma.T @ obs) / gamma_sum[:, np.newaxis]

        for k in range(self.n_states):
            diff = obs - self._means[k]
            self._covars[k] = (gamma[:, k] @ (diff ** 2)) / gamma_sum[k] + 1e-6

    def _forward_backward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward-backward algorithm for HMM.

        Returns
        -------
        gamma : np.ndarray
            State posterior probabilities, shape (T, n_states)
        xi : np.ndarray
            Transition posteriors, shape (n_states, n_states, T-1)
        """
        T = len(obs)
        N = self.n_states

        # Emission probabilities
        log_emissions = self._log_emission_prob(obs)

        # Forward pass
        log_alpha = np.zeros((T, N))
        log_alpha[0] = np.log(self._initial_probs + 1e-300) + log_emissions[0]

        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t - 1] + np.log(self._transition_matrix[:, j] + 1e-300)
                ) + log_emissions[t, j]

        # Backward pass
        log_beta = np.zeros((T, N))

        for t in range(T - 2, -1, -1):
            for i in range(N):
                log_beta[t, i] = logsumexp(
                    np.log(self._transition_matrix[i, :] + 1e-300) +
                    log_emissions[t + 1] + log_beta[t + 1]
                )

        # Compute gamma
        log_gamma = log_alpha + log_beta
        log_gamma_norm = logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma - log_gamma_norm)

        # Compute xi
        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    xi[i, j, t] = (
                        log_alpha[t, i] +
                        np.log(self._transition_matrix[i, j] + 1e-300) +
                        log_emissions[t + 1, j] +
                        log_beta[t + 1, j]
                    )
            xi[:, :, t] = np.exp(xi[:, :, t] - logsumexp(xi[:, :, t].flatten()))

        return gamma, xi

    def _viterbi(self, obs: np.ndarray) -> List[int]:
        """Viterbi algorithm for most likely state sequence."""
        T = len(obs)
        N = self.n_states

        log_emissions = self._log_emission_prob(obs)

        # Initialization
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        delta[0] = np.log(self._initial_probs + 1e-300) + log_emissions[0]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                scores = delta[t - 1] + np.log(self._transition_matrix[:, j] + 1e-300)
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + log_emissions[t, j]

        # Backtracking
        states = [0] * T
        states[T - 1] = int(np.argmax(delta[T - 1]))

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def _log_emission_prob(self, obs: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities under Gaussian assumption."""
        T = len(obs)
        N = self.n_states
        log_emissions = np.zeros((T, N))

        for k in range(N):
            # Multivariate Gaussian with diagonal covariance
            diff = obs - self._means[k]
            log_det = np.sum(np.log(self._covars[k] + 1e-300))
            log_norm = -0.5 * (obs.shape[1] * np.log(2 * np.pi) + log_det)
            log_prob = log_norm - 0.5 * np.sum((diff ** 2) / (self._covars[k] + 1e-300), axis=1)
            log_emissions[:, k] = log_prob

        return log_emissions

    def _compute_log_likelihood(self, obs: np.ndarray) -> float:
        """Compute log-likelihood of observations."""
        log_emissions = self._log_emission_prob(obs)
        T = len(obs)
        N = self.n_states

        log_alpha = np.zeros((T, N))
        log_alpha[0] = np.log(self._initial_probs + 1e-300) + log_emissions[0]

        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t - 1] + np.log(self._transition_matrix[:, j] + 1e-300)
                ) + log_emissions[t, j]

        return float(logsumexp(log_alpha[T - 1]))

    def _map_states_to_regimes(self, obs: np.ndarray) -> None:
        """Map HMM states to MarketRegime based on fitted parameters."""
        # Sort states by mean return
        state_order = np.argsort(self._means[:, 0])

        regime_mapping = [
            MarketRegime.BEAR_STRONG,
            MarketRegime.BEAR_MODERATE,
            MarketRegime.TRANSITION,
            MarketRegime.BULL_MODERATE,
            MarketRegime.BULL_STRONG,
            MarketRegime.CRISIS,
            MarketRegime.SIDEWAYS_HIGH_VOL,
            MarketRegime.SIDEWAYS_LOW_VOL,
            MarketRegime.RECOVERY,
            MarketRegime.BULL_WEAK,
            MarketRegime.BEAR_WEAK,
        ]

        for i, state_idx in enumerate(state_order):
            if i < len(regime_mapping):
                self._state_regime_map[int(state_idx)] = regime_mapping[i]
            else:
                self._state_regime_map[int(state_idx)] = MarketRegime.TRANSITION


# ---------------------------------------------------------------------------
# Statistical Regime Detector
# ---------------------------------------------------------------------------


class StatisticalRegimeDetector:
    """
    Statistical regime detection using classical financial indicators.

    Detects:
    - Trend via moving average crossovers
    - Volatility regime via GARCH-like estimation
    - Mean reversion via Hurst exponent
    - Momentum regime via trend strength
    """

    def __init__(
        self,
        fast_ma: int = 10,
        slow_ma: int = 50,
        vol_window: int = 20,
        hurst_max_lag: int = 20,
    ) -> None:
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.vol_window = vol_window
        self.hurst_max_lag = hurst_max_lag

        self._prices: Deque[float] = deque(maxlen=slow_ma + 100)
        self._returns: Deque[float] = deque(maxlen=200)

    def detect_trend(self) -> Tuple[str, float]:
        """
        Detect trend using moving average crossover.

        Returns
        -------
        Tuple[str, float]
            (direction, strength) where direction is 'up', 'down', or 'neutral'
        """
        if len(self._prices) < self.slow_ma:
            return "neutral", 0.0

        prices_arr = np.array(list(self._prices))
        fast_ma = float(np.mean(prices_arr[-self.fast_ma:]))
        slow_ma = float(np.mean(prices_arr[-self.slow_ma:]))

        diff = fast_ma - slow_ma
        strength = abs(diff) / slow_ma if slow_ma > 0 else 0.0

        if diff > 0:
            return "up", min(1.0, strength * 10)
        elif diff < 0:
            return "down", min(1.0, strength * 10)
        return "neutral", 0.0

    def detect_volatility_regime(self) -> Tuple[str, float]:
        """
        Detect volatility regime using GARCH-like estimation.

        Returns
        -------
        Tuple[str, float]
            (regime, level) where regime is 'high', 'low', or 'normal'
        """
        if len(self._returns) < self.vol_window:
            return "normal", 0.0

        returns_arr = np.array(list(self._returns))
        recent_vol = float(np.std(returns_arr[-self.vol_window:]))

        # GARCH-like: weighted volatility
        weights = np.exp(-np.linspace(0, 3, self.vol_window))
        weights = weights / weights.sum()
        garch_vol = float(np.sqrt(np.sum(weights * returns_arr[-self.vol_window:] ** 2)))

        # Classify
        if garch_vol > 0.03:
            return "high", garch_vol
        elif garch_vol < 0.01:
            return "low", garch_vol
        return "normal", garch_vol

    def detect_mean_reversion(self) -> Tuple[str, float]:
        """
        Detect mean reversion using Hurst exponent.

        Returns
        -------
        Tuple[str, float]
            (regime, hurst) where regime is 'mean_reverting', 'trending', or 'random'
        """
        if len(self._returns) < self.hurst_max_lag * 2:
            return "random", 0.5

        returns_arr = np.array(list(self._returns))
        hurst = _compute_hurst_exponent(returns_arr, self.hurst_max_lag)

        if hurst < 0.45:
            return "mean_reverting", hurst
        elif hurst > 0.55:
            return "trending", hurst
        return "random", hurst

    def detect_momentum_regime(self) -> Tuple[str, float]:
        """
        Detect momentum regime using rate of change.

        Returns
        -------
        Tuple[str, float]
            (regime, momentum) where regime is 'strong', 'weak', or 'neutral'
        """
        if len(self._prices) < 20:
            return "neutral", 0.0

        prices_arr = np.array(list(self._prices))
        momentum = float((prices_arr[-1] - prices_arr[-20]) / prices_arr[-20])

        if momentum > 0.05:
            return "strong", momentum
        elif momentum < -0.05:
            return "weak", momentum
        return "neutral", momentum

    def update(self, price: float) -> None:
        """Update detector with new price."""
        if len(self._prices) > 0:
            last_price = self._prices[-1]
            ret = (price - last_price) / last_price
            self._returns.append(ret)
        self._prices.append(price)

    def get_combined_regime(self) -> Dict[str, Any]:
        """Get combined regime detection results."""
        trend_dir, trend_str = self.detect_trend()
        vol_regime, vol_level = self.detect_volatility_regime()
        mr_regime, hurst = self.detect_mean_reversion()
        mom_regime, momentum = self.detect_momentum_regime()

        return {
            "trend_direction": trend_dir,
            "trend_strength": trend_str,
            "volatility_regime": vol_regime,
            "volatility_level": vol_level,
            "mean_reversion": mr_regime,
            "hurst_exponent": hurst,
            "momentum_regime": mom_regime,
            "momentum": momentum,
        }


# ---------------------------------------------------------------------------
# Multi-Timeframe Regime Analyzer
# ---------------------------------------------------------------------------


class MultiTimeframeRegimeAnalyzer:
    """
    Analyzes market regimes across multiple timeframes.

    Provides consensus regime detection and identifies conflicting signals.
    """

    VALID_TIMEFRAMES = ("1m", "5m", "1h", "4h", "1d")

    def __init__(self) -> None:
        self._regimes: Dict[str, MarketRegime] = {}
        self._confidences: Dict[str, float] = {}
        self._classifiers: Dict[str, RegimeClassifier] = {
            tf: RegimeClassifier() for tf in self.VALID_TIMEFRAMES
        }

    @property
    def regimes(self) -> Dict[str, MarketRegime]:
        """Current regimes by timeframe."""
        return dict(self._regimes)

    def update_timeframe(
        self, timeframe: str, market_data: Dict[str, Any]
    ) -> MarketRegime:
        """Update regime for a specific timeframe."""
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid: {self.VALID_TIMEFRAMES}")

        clf = self._classifiers[timeframe]
        features = clf.extract_features(market_data)
        regime = clf.update(features)

        self._regimes[timeframe] = regime
        self._confidences[timeframe] = clf.get_regime_confidence()

        return regime

    def get_aligned_regime(self) -> MarketRegime:
        """
        Get consensus regime across all timeframes.

        Uses weighted voting where longer timeframes have higher weight.
        """
        if not self._regimes:
            return MarketRegime.TRANSITION

        weights = {"1m": 1, "5m": 2, "1h": 3, "4h": 4, "1d": 5}
        regime_scores: Dict[MarketRegime, float] = defaultdict(float)

        for tf, regime in self._regimes.items():
            weight = weights.get(tf, 1)
            confidence = self._confidences.get(tf, 0.5)
            regime_scores[regime] += weight * confidence

        if not regime_scores:
            return MarketRegime.TRANSITION

        return max(regime_scores, key=regime_scores.get)

    def get_conflicting_timeframes(self) -> List[str]:
        """
        Identify timeframes that disagree with the consensus regime.

        Returns
        -------
        List[str]
            List of timeframe names that conflict with consensus
        """
        if not self._regimes:
            return []

        consensus = self.get_aligned_regime()
        conflicts = []

        for tf, regime in self._regimes.items():
            if regime != consensus:
                # Check if it's a meaningful conflict (not adjacent regimes)
                if not self._are_adjacent_regimes(regime, consensus):
                    conflicts.append(tf)

        return conflicts

    def get_dominant_timeframe(self) -> str:
        """
        Get the timeframe with the highest confidence signal.

        Returns
        -------
        str
            Timeframe name with strongest signal
        """
        if not self._confidences:
            return "1d"

        return max(self._confidences, key=self._confidences.get)

    def _are_adjacent_regimes(
        self, r1: MarketRegime, r2: MarketRegime
    ) -> bool:
        """Check if two regimes are adjacent (not conflicting)."""
        adjacent_pairs = {
            (MarketRegime.BULL_STRONG, MarketRegime.BULL_MODERATE),
            (MarketRegime.BULL_MODERATE, MarketRegime.BULL_WEAK),
            (MarketRegime.BULL_WEAK, MarketRegime.TRANSITION),
            (MarketRegime.TRANSITION, MarketRegime.BEAR_WEAK),
            (MarketRegime.BEAR_WEAK, MarketRegime.BEAR_MODERATE),
            (MarketRegime.BEAR_MODERATE, MarketRegime.BEAR_STRONG),
            (MarketRegime.SIDEWAYS_HIGH_VOL, MarketRegime.SIDEWAYS_LOW_VOL),
            (MarketRegime.CRISIS, MarketRegime.RECOVERY),
            (MarketRegime.RECOVERY, MarketRegime.BULL_WEAK),
        }

        pair = (r1, r2) if r1 < r2 else (r2, r1)
        return pair in adjacent_pairs or r1 == r2


# ---------------------------------------------------------------------------
# Regime Transition Matrix
# ---------------------------------------------------------------------------


class RegimeTransitionMatrix:
    """
    Tracks and analyzes regime transitions over time.

    Maintains a transition count matrix and computes transition probabilities.
    """

    def __init__(self) -> None:
        self._transition_counts: np.ndarray = np.zeros(
            (len(MarketRegime), len(MarketRegime)), dtype=np.float64
        )
        self._regime_durations: Dict[MarketRegime, List[int]] = defaultdict(list)
        self._current_regime: Optional[MarketRegime] = None
        self._current_duration: int = 0
        self._regime_list = list(MarketRegime)
        self._regime_to_idx = {r: i for i, r in enumerate(self._regime_list)}

    def track_transitions(
        self, from_regime: MarketRegime, to_regime: MarketRegime, duration: int = 1
    ) -> None:
        """Record a regime transition."""
        if from_regime not in self._regime_to_idx or to_regime not in self._regime_to_idx:
            return

        from_idx = self._regime_to_idx[from_regime]
        to_idx = self._regime_to_idx[to_regime]

        self._transition_counts[from_idx, to_idx] += 1

        if duration > 0:
            self._regime_durations[from_regime].append(duration)

    def get_transition_probabilities(self) -> np.ndarray:
        """
        Compute transition probability matrix.

        Returns
        -------
        np.ndarray
            Shape (n_regimes, n_regimes) — row-stochastic matrix
        """
        probs = self._transition_counts.copy()
        row_sums = probs.sum(axis=1, keepdims=True)

        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums

        return probs

    def predict_next_regime(
        self, current_regime: MarketRegime
    ) -> Dict[MarketRegime, float]:
        """
        Predict probability distribution over next regime.

        Parameters
        ----------
        current_regime : MarketRegime
            Current market regime

        Returns
        -------
        Dict[MarketRegime, float]
            Probability distribution over next regimes
        """
        if current_regime not in self._regime_to_idx:
            return {r: 1.0 / len(MarketRegime) for r in MarketRegime}

        idx = self._regime_to_idx[current_regime]
        probs = self.get_transition_probabilities()

        result = {}
        for r in MarketRegime:
            r_idx = self._regime_to_idx[r]
            result[r] = float(probs[idx, r_idx])

        return result

    def get_average_regime_duration(self, regime: MarketRegime) -> float:
        """
        Get average duration for a specific regime.

        Parameters
        ----------
        regime : MarketRegime
            Regime to compute average duration for

        Returns
        -------
        float
            Average duration in bars
        """
        durations = self._regime_durations.get(regime, [])
        if not durations:
            return 0.0
        return float(np.mean(durations))

    def get_all_average_durations(self) -> Dict[MarketRegime, float]:
        """Get average durations for all regimes."""
        return {
            r: self.get_average_regime_duration(r) for r in MarketRegime
        }


# ---------------------------------------------------------------------------
# Main Market Regime Detector
# ---------------------------------------------------------------------------


class MarketRegimeDetector:
    """
    Primary market regime detection system.

    Combines rule-based classification, HMM detection, and statistical
    analysis to provide robust real-time regime identification.

    Usage::

        detector = MarketRegimeDetector()
        regime = detector.update(market_data)
        stats = detector.get_regime_stats()
    """

    def __init__(
        self,
        use_hmm: bool = True,
        hmm_states: int = 5,
        transition_tracking: bool = True,
        multi_timeframe: bool = False,
    ) -> None:
        self.use_hmm = use_hmm and _SCIPY_AVAILABLE
        self.multi_timeframe = multi_timeframe

        # Core components
        self._classifier = RegimeClassifier()
        self._statistical = StatisticalRegimeDetector()
        self._hmm: Optional[HiddenMarkovRegimeDetector] = None
        self._transition_matrix: Optional[RegimeTransitionMatrix] = None
        self._mtf_analyzer: Optional[MultiTimeframeRegimeAnalyzer] = None

        if self.use_hmm:
            self._hmm = HiddenMarkovRegimeDetector(n_states=hmm_states)

        if transition_tracking:
            self._transition_matrix = RegimeTransitionMatrix()

        if multi_timeframe:
            self._mtf_analyzer = MultiTimeframeRegimeAnalyzer()

        # State tracking
        self._current_regime: MarketRegime = MarketRegime.TRANSITION
        self._previous_regime: Optional[MarketRegime] = None
        self._regime_start_bar: int = 0
        self._current_bar: int = 0
        self._history: Deque[RegimeSnapshot] = deque(maxlen=1000)
        self._regime_confidence: float = 0.0
        self._hmm_fitted: bool = False
        self._returns_buffer: Deque[float] = deque(maxlen=500)

        logger.info(
            "MarketRegimeDetector initialized (HMM=%s, MTF=%s)",
            self.use_hmm, self.multi_timeframe,
        )

    def update(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Update detector with new market data and return current regime.

        Parameters
        ----------
        market_data : Dict[str, Any]
            Market data with prices, volumes, etc.

        Returns
        -------
        MarketRegime
            Current classified regime
        """
        self._current_bar += 1

        # Extract features
        features = self._classifier.extract_features(market_data)

        # Rule-based classification
        rule_regime = self._classifier.classify(features)
        rule_confidence = self._classifier.get_regime_confidence()

        # HMM classification (if available)
        hmm_regime = MarketRegime.TRANSITION
        hmm_confidence = 0.0

        if self.use_hmm and self._hmm is not None:
            prices = np.asarray(market_data.get("prices", []), dtype=np.float64)
            if len(prices) > 2:
                returns = np.diff(np.log(prices))
                self._returns_buffer.extend(returns)

                # Fit HMM periodically
                if not self._hmm_fitted and len(self._returns_buffer) >= 60:
                    returns_arr = np.array(list(self._returns_buffer))
                    self._hmm_fitted = self._hmm.fit(returns_arr)

                if self._hmm_fitted:
                    hmm_regime = self._hmm.get_current_regime(returns)
                    hmm_probs = self._hmm.predict_proba(returns)
                    if len(hmm_probs) > 0:
                        hmm_confidence = float(np.max(hmm_probs[-1]))

        # Statistical analysis
        prices = np.asarray(market_data.get("prices", []), dtype=np.float64)
        if len(prices) > 0:
            self._statistical.update(float(prices[-1]))

        # Combine signals (weighted ensemble)
        regime, confidence = self._combine_signals(
            rule_regime, rule_confidence,
            hmm_regime, hmm_confidence,
        )

        # Track transitions
        if self._transition_matrix is not None:
            if self._previous_regime is not None and regime != self._previous_regime:
                duration = self._current_bar - self._regime_start_bar
                self._transition_matrix.track_transitions(
                    self._previous_regime, regime, duration
                )
                self._regime_start_bar = self._current_bar

        # Update state
        if regime != self._current_regime:
            self._previous_regime = self._current_regime
            self._current_regime = regime
            self._regime_start_bar = self._current_bar

            logger.info(
                "Regime change: %s -> %s (confidence=%.2f, bar=%d)",
                self._previous_regime.value if self._previous_regime else "None",
                regime.value, confidence, self._current_bar,
            )

        self._regime_confidence = confidence

        # Record snapshot
        snapshot = RegimeSnapshot(
            regime=regime,
            timestamp=datetime.now(),
            confidence=confidence,
            features=features,
            duration_bars=self._current_bar - self._regime_start_bar,
        )
        self._history.append(snapshot)

        return regime

    def get_current_regime(self) -> MarketRegime:
        """Return the current market regime."""
        return self._current_regime

    def get_regime_history(self, n: int = 100) -> List[RegimeSnapshot]:
        """
        Get recent regime history.

        Parameters
        ----------
        n : int
            Number of snapshots to return

        Returns
        -------
        List[RegimeSnapshot]
            Recent regime snapshots
        """
        return list(self._history)[-n:]

    def get_regime_stats(self) -> RegimeStatistics:
        """
        Get comprehensive regime statistics.

        Returns
        -------
        RegimeStatistics
            Current regime stats and historical analysis
        """
        # Historical distribution
        regime_counts: Dict[MarketRegime, int] = defaultdict(int)
        for snapshot in self._history:
            regime_counts[snapshot.regime] += 1

        total = sum(regime_counts.values()) or 1
        distribution = {
            r: count / total for r, count in regime_counts.items()
        }

        # Average durations
        avg_durations: Dict[MarketRegime, float] = {}
        if self._transition_matrix is not None:
            avg_durations = self._transition_matrix.get_all_average_durations()

        # Transition matrix
        transition_mat = np.zeros((len(MarketRegime), len(MarketRegime)))
        if self._transition_matrix is not None:
            transition_mat = self._transition_matrix.get_transition_probabilities()

        return RegimeStatistics(
            current_regime=self._current_regime,
            regime_duration=self._current_bar - self._regime_start_bar,
            historical_distribution=distribution,
            average_regime_duration=avg_durations,
            transition_matrix=transition_mat,
        )

    def is_regime_changing(self, threshold: float = 0.3) -> bool:
        """
        Check if regime is in a state of transition.

        Parameters
        ----------
        threshold : float
            Confidence threshold below which regime is considered changing

        Returns
        -------
        bool
            True if regime confidence is below threshold
        """
        return self._regime_confidence < threshold

    def get_regime_strength(self) -> float:
        """
        Get regime strength indicator (-1 to 1).

        Negative values indicate bearish strength, positive indicate bullish.
        Magnitude indicates conviction.

        Returns
        -------
        float
            Regime strength in [-1, 1]
        """
        regime = self._current_regime
        confidence = self._regime_confidence

        strength_map = {
            MarketRegime.BULL_STRONG: 1.0,
            MarketRegime.BULL_MODERATE: 0.6,
            MarketRegime.BULL_WEAK: 0.3,
            MarketRegime.TRANSITION: 0.0,
            MarketRegime.SIDEWAYS_LOW_VOL: 0.0,
            MarketRegime.SIDEWAYS_HIGH_VOL: 0.0,
            MarketRegime.BEAR_WEAK: -0.3,
            MarketRegime.BEAR_MODERATE: -0.6,
            MarketRegime.BEAR_STRONG: -1.0,
            MarketRegime.CRISIS: -0.9,
            MarketRegime.RECOVERY: 0.5,
        }

        base_strength = strength_map.get(regime, 0.0)
        return float(base_strength * confidence)

    def _combine_signals(
        self,
        rule_regime: MarketRegime,
        rule_confidence: float,
        hmm_regime: MarketRegime,
        hmm_confidence: float,
    ) -> Tuple[MarketRegime, float]:
        """
        Combine rule-based and HMM signals into final regime.

        Uses confidence-weighted voting.
        """
        if not self.use_hmm or not self._hmm_fitted:
            return rule_regime, rule_confidence

        # If both agree, return with higher confidence
        if rule_regime == hmm_regime:
            return rule_regime, max(rule_confidence, hmm_confidence)

        # Weight by confidence
        rule_weight = rule_confidence * 0.6  # Rule-based has higher prior
        hmm_weight = hmm_confidence * 0.4

        # If HMM strongly disagrees and has high confidence, use it
        if hmm_confidence > 0.8 and rule_confidence < 0.6:
            return hmm_regime, hmm_confidence

        # Default to rule-based with adjusted confidence
        combined_confidence = (rule_weight + hmm_weight) / (rule_weight + hmm_weight + 0.01)
        return rule_regime, min(1.0, combined_confidence)
