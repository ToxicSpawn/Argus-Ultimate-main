"""Bayesian probabilistic programming for market models.

This module provides a lightweight probabilistic programming framework
tailored for modelling uncertainty in market dynamics. It supports:

* Composable probability distributions (Normal, Beta, Gamma).
* Bayesian models built from named random variables and evidence.
* A market-specific model that tracks trend, volatility, and momentum
  as latent variables with conjugate or MCMC-based updates.

The design is deliberately self-contained: only `numpy` and the Python
standard library are required. Where closed-form conjugate posteriors
exist we use them (e.g. Normal-Normal for the trend variable); for the
remaining variables we fall back to a simple Metropolis-Hastings sampler
to provide posterior samples and uncertainty bands.

Example:
    >>> model = BayesianMarketModel()
    >>> for r in returns:
    ...     model.update(r)
    >>> forecast = model.predict(n_ahead=5)
    >>> unc = model.get_uncertainty()
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base distribution class and concrete implementations
# ---------------------------------------------------------------------------


class Distribution:
    """Abstract base class for probability distributions.

    Subclasses must override :meth:`sample` and :meth:`log_prob`.
    """

    name: str = "distribution"

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw ``n`` samples from the distribution."""
        raise NotImplementedError

    def log_prob(self, x: float) -> float:
        """Return the log-probability density at ``x``."""
        raise NotImplementedError

    def mean(self) -> float:
        """Return the distribution mean (default via sampling)."""
        return float(np.mean(self.sample(2048)))

    def std(self) -> float:
        """Return the distribution standard deviation (default via sampling)."""
        return float(np.std(self.sample(2048)))


@dataclass
class Normal(Distribution):
    """Univariate normal distribution ``N(mu, sigma^2)``."""

    mu: float = 0.0
    sigma: float = 1.0
    name: str = "normal"

    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.normal(self.mu, max(self.sigma, 1e-9), size=n)

    def log_prob(self, x: float) -> float:
        sigma = max(self.sigma, 1e-9)
        return -0.5 * math.log(2 * math.pi * sigma * sigma) - 0.5 * ((x - self.mu) / sigma) ** 2

    def mean(self) -> float:
        return float(self.mu)

    def std(self) -> float:
        return float(max(self.sigma, 0.0))


@dataclass
class Beta(Distribution):
    """Beta distribution ``Beta(alpha, beta)`` on ``(0, 1)``."""

    alpha: float = 1.0
    beta: float = 1.0
    name: str = "beta"

    def sample(self, n: int = 1) -> np.ndarray:
        a = max(self.alpha, 1e-6)
        b = max(self.beta, 1e-6)
        return np.random.beta(a, b, size=n)

    def log_prob(self, x: float) -> float:
        if not 0.0 < x < 1.0:
            return -math.inf
        a = max(self.alpha, 1e-6)
        b = max(self.beta, 1e-6)
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        return (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - log_beta

    def mean(self) -> float:
        return float(self.alpha / max(self.alpha + self.beta, 1e-9))


@dataclass
class Gamma(Distribution):
    """Gamma distribution parameterised by shape ``k`` and scale ``theta``."""

    shape: float = 1.0
    scale: float = 1.0
    name: str = "gamma"

    def sample(self, n: int = 1) -> np.ndarray:
        k = max(self.shape, 1e-6)
        t = max(self.scale, 1e-6)
        return np.random.gamma(k, t, size=n)

    def log_prob(self, x: float) -> float:
        if x <= 0.0:
            return -math.inf
        k = max(self.shape, 1e-6)
        t = max(self.scale, 1e-6)
        return (k - 1.0) * math.log(x) - x / t - k * math.log(t) - math.lgamma(k)

    def mean(self) -> float:
        return float(self.shape * self.scale)


# ---------------------------------------------------------------------------
# Bayesian model composition and inference
# ---------------------------------------------------------------------------


@dataclass
class _Variable:
    name: str
    prior: Distribution
    evidence: Optional[float] = None
    posterior_samples: np.ndarray = field(default_factory=lambda: np.array([]))


class BayesianModel:
    """Composable probabilistic model over named random variables.

    Variables are added with :meth:`add_variable`, observations are
    clamped via :meth:`condition`, and posterior samples for an
    individual variable can be drawn with :meth:`posterior`. For
    variables without conjugate updates we fall back to a local
    Metropolis-Hastings sweep which is adequate for one-dimensional
    market latents.
    """

    def __init__(self) -> None:
        self._variables: Dict[str, _Variable] = {}
        self._joint_log_prob: Optional[Callable[[Dict[str, float]], float]] = None
        self._mcmc_steps: int = 500
        self._mcmc_burn: int = 100

    def add_variable(self, name: str, dist: Distribution) -> None:
        """Register ``name`` with prior distribution ``dist``."""
        if name in self._variables:
            logger.debug("bayesian_model: overwriting variable %s", name)
        self._variables[name] = _Variable(name=name, prior=dist)

    def condition(self, name: str, value: float) -> None:
        """Set evidence for a named variable (clamps the posterior)."""
        if name not in self._variables:
            raise KeyError(f"variable {name!r} not registered")
        self._variables[name].evidence = float(value)

    def set_joint_log_prob(self, fn: Callable[[Dict[str, float]], float]) -> None:
        """Provide an optional joint log-probability for MCMC inference."""
        self._joint_log_prob = fn

    def _current_values(self, proposal: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name, var in self._variables.items():
            if proposal is not None and name in proposal:
                values[name] = proposal[name]
            elif var.evidence is not None:
                values[name] = var.evidence
            else:
                values[name] = var.prior.mean()
        return values

    def posterior(self, name: str, n_samples: int = 500) -> np.ndarray:
        """Return ``n_samples`` draws from the posterior of ``name``."""
        if name not in self._variables:
            raise KeyError(f"variable {name!r} not registered")
        var = self._variables[name]
        if var.evidence is not None:
            return np.full(n_samples, var.evidence, dtype=float)
        if self._joint_log_prob is None:
            samples = var.prior.sample(n_samples)
            var.posterior_samples = samples
            return samples
        return self._mcmc_posterior(name, n_samples)

    def _mcmc_posterior(self, name: str, n_samples: int) -> np.ndarray:
        var = self._variables[name]
        current = var.prior.mean()
        scale = max(var.prior.std(), 1e-3)
        samples = np.empty(n_samples, dtype=float)
        accepted = 0
        state = self._current_values()
        state[name] = current
        assert self._joint_log_prob is not None
        current_lp = self._joint_log_prob(state) + var.prior.log_prob(current)
        total_iters = self._mcmc_burn + n_samples
        for i in range(total_iters):
            proposal = current + random.gauss(0.0, scale)
            prop_state = dict(state)
            prop_state[name] = proposal
            try:
                proposal_lp = self._joint_log_prob(prop_state) + var.prior.log_prob(proposal)
            except Exception:  # noqa: BLE001 - defensive MCMC guard
                proposal_lp = -math.inf
            if proposal_lp - current_lp > math.log(random.random() + 1e-12):
                current, current_lp = proposal, proposal_lp
                state[name] = current
                accepted += 1
            if i >= self._mcmc_burn:
                samples[i - self._mcmc_burn] = current
        var.posterior_samples = samples
        logger.debug("mcmc_posterior %s acceptance=%.3f", name, accepted / max(total_iters, 1))
        return samples


# ---------------------------------------------------------------------------
# Market-specific model
# ---------------------------------------------------------------------------


class BayesianMarketModel:
    """Bayesian model of short-term market dynamics.

    The latent state consists of:

    * ``trend``      — expected return (Normal prior).
    * ``volatility`` — scale of returns (Gamma prior).
    * ``momentum``   — persistence factor in ``[0, 1]`` (Beta prior).

    The model uses a conjugate Normal update for the trend term and a
    running method-of-moments update for volatility. Momentum is
    updated through a simple posterior reweighting of its Beta prior.
    """

    def __init__(
        self,
        prior_trend_mu: float = 0.0,
        prior_trend_sigma: float = 0.01,
        prior_vol_shape: float = 2.0,
        prior_vol_scale: float = 0.01,
        prior_mom_alpha: float = 2.0,
        prior_mom_beta: float = 2.0,
    ) -> None:
        self.trend_prior = Normal(prior_trend_mu, prior_trend_sigma)
        self.vol_prior = Gamma(prior_vol_shape, prior_vol_scale)
        self.mom_prior = Beta(prior_mom_alpha, prior_mom_beta)

        self.model = BayesianModel()
        self.model.add_variable("trend", self.trend_prior)
        self.model.add_variable("volatility", self.vol_prior)
        self.model.add_variable("momentum", self.mom_prior)

        self._observations: List[float] = []
        self._trend_mu = prior_trend_mu
        self._trend_sigma = prior_trend_sigma
        self._vol_shape = prior_vol_shape
        self._vol_scale = prior_vol_scale
        self._mom_alpha = prior_mom_alpha
        self._mom_beta = prior_mom_beta
        self._n_updates = 0

    def update(self, observation: float) -> None:
        """Incorporate a single return observation."""
        obs = float(observation)
        self._observations.append(obs)
        if len(self._observations) > 2048:
            self._observations = self._observations[-2048:]
        self._n_updates += 1

        # Conjugate Normal-Normal update for the trend.
        var_prior = max(self._trend_sigma ** 2, 1e-12)
        lik_var = max(self._vol_shape * self._vol_scale ** 2, 1e-6)
        post_var = 1.0 / (1.0 / var_prior + 1.0 / lik_var)
        post_mu = post_var * (self._trend_mu / var_prior + obs / lik_var)
        self._trend_mu = float(post_mu)
        self._trend_sigma = float(math.sqrt(post_var))
        self.trend_prior.mu = self._trend_mu
        self.trend_prior.sigma = self._trend_sigma

        # Method of moments for volatility -> Gamma update.
        if len(self._observations) >= 4:
            sample_var = float(np.var(self._observations[-64:]))
            sample_var = max(sample_var, 1e-10)
            self._vol_shape = 2.0 + 0.5 * min(len(self._observations), 64)
            self._vol_scale = sample_var / self._vol_shape
            self.vol_prior.shape = self._vol_shape
            self.vol_prior.scale = self._vol_scale

        # Beta update for momentum using sign continuation indicator.
        if len(self._observations) >= 2:
            prev = self._observations[-2]
            same_sign = 1.0 if (prev * obs) > 0 else 0.0
            self._mom_alpha += same_sign
            self._mom_beta += 1.0 - same_sign
            self.mom_prior.alpha = self._mom_alpha
            self.mom_prior.beta = self._mom_beta

    def predict(self, n_ahead: int = 1, n_paths: int = 256) -> Dict[str, Any]:
        """Return a forecast summary over ``n_ahead`` future steps."""
        trend_samples = self.trend_prior.sample(n_paths)
        vol_samples = np.sqrt(np.abs(self.vol_prior.sample(n_paths)) + 1e-12)
        mom_samples = self.mom_prior.sample(n_paths)

        last = self._observations[-1] if self._observations else 0.0
        paths = np.zeros((n_paths, n_ahead), dtype=float)
        prev = np.full(n_paths, last, dtype=float)
        for t in range(n_ahead):
            noise = np.random.normal(0.0, 1.0, size=n_paths)
            step = trend_samples + mom_samples * prev + vol_samples * noise
            paths[:, t] = step
            prev = step

        means = paths.mean(axis=0).tolist()
        lower = np.percentile(paths, 5, axis=0).tolist()
        upper = np.percentile(paths, 95, axis=0).tolist()
        return {
            "mean": means,
            "lower_5": lower,
            "upper_95": upper,
            "paths": paths.tolist(),
        }

    def get_uncertainty(self) -> Dict[str, float]:
        """Return a compact uncertainty summary of latent variables."""
        return {
            "trend_mu": float(self._trend_mu),
            "trend_sigma": float(self._trend_sigma),
            "vol_mean": float(self.vol_prior.mean()),
            "vol_std": float(self.vol_prior.std()),
            "momentum_mean": float(self.mom_prior.mean()),
            "total_observations": float(len(self._observations)),
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the model's state."""
        return {
            "n_updates": self._n_updates,
            "n_observations": len(self._observations),
            "trend": {"mu": self._trend_mu, "sigma": self._trend_sigma},
            "volatility": {"shape": self._vol_shape, "scale": self._vol_scale},
            "momentum": {"alpha": self._mom_alpha, "beta": self._mom_beta},
            "uncertainty": self.get_uncertainty(),
        }


__all__ = [
    "Distribution",
    "Normal",
    "Beta",
    "Gamma",
    "BayesianModel",
    "BayesianMarketModel",
]
