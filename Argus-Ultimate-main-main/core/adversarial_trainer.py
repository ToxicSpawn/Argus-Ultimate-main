"""Adversarial robustness training for ARGUS strategies.

This module stress-tests a strategy function by generating worst-case market
states in its immediate neighbourhood. It is inspired by the Fast Gradient
Sign Method (FGSM) from adversarial ML: instead of perturbing an image to
fool a classifier, we perturb a market state to fool a trading strategy.

Two flavours of perturbation are supported:

* **Untargeted** — push the state in the direction that *degrades* the
  strategy's expected reward (we estimate the gradient numerically because
  the strategy function is a black box).
* **Targeted** — push the state in a direction that makes the strategy
  output a specified *wrong* action (e.g. forces it to buy when it should
  sell).

Classes
-------

* :class:`AdversarialGenerator` — produces perturbations for a given state.
* :class:`RobustStrategyTrainer` — trains / evaluates a strategy against the
  adversarial generator and reports a 0..1 robustness score.

The strategy function signature expected throughout is::

    def strategy_fn(state: np.ndarray) -> float

where the return value is interpreted as the "action" (typically a signed
confidence in ``[-1, 1]``: positive = buy, negative = sell). The reward
function is a pluggable callable ``reward_fn(state, action) -> float`` which
defaults to a simple ``state[0] * action`` — i.e. the strategy is rewarded
for aligning its action with the sign of the first state dimension.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


StrategyFn = Callable[[np.ndarray], float]
RewardFn = Callable[[np.ndarray, float], float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_reward(state: np.ndarray, action: float) -> float:
    """Default reward: dot-product between action and state[0].

    This keeps things standalone — users will typically override with a
    real P&L model.
    """

    if state.size == 0:
        return 0.0
    return float(state[0] * action)


@dataclass
class Vulnerability:
    """A specific adversarial example that broke the strategy."""

    original_state: np.ndarray
    adversarial_state: np.ndarray
    original_action: float
    adversarial_action: float
    original_reward: float
    adversarial_reward: float
    perturbation_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_norm": float(self.perturbation_norm),
            "original_action": float(self.original_action),
            "adversarial_action": float(self.adversarial_action),
            "original_reward": float(self.original_reward),
            "adversarial_reward": float(self.adversarial_reward),
            "reward_drop": float(self.original_reward - self.adversarial_reward),
        }


# ---------------------------------------------------------------------------
# Adversarial generator
# ---------------------------------------------------------------------------


class AdversarialGenerator:
    """Generate worst-case perturbations of a state for a black-box strategy.

    Because the strategy is not assumed to be differentiable, the generator
    estimates the gradient of ``reward_fn(state, strategy_fn(state))`` with
    respect to ``state`` via finite differences. It then takes the *sign*
    of the gradient and scales it by ``epsilon`` — standard FGSM.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        reward_fn: Optional[RewardFn] = None,
        finite_diff_step: float = 1e-3,
    ) -> None:
        self.epsilon = float(epsilon)
        self.reward_fn = reward_fn or _default_reward
        self.finite_diff_step = float(finite_diff_step)

    def _eval(self, strategy_fn: StrategyFn, state: np.ndarray) -> Tuple[float, float]:
        action = float(strategy_fn(state))
        reward = float(self.reward_fn(state, action))
        return action, reward

    def _grad(self, strategy_fn: StrategyFn, state: np.ndarray) -> np.ndarray:
        """Finite difference estimate of reward gradient w.r.t. ``state``."""

        base_action, base_reward = self._eval(strategy_fn, state)
        del base_action  # unused
        grad = np.zeros_like(state, dtype=np.float64)
        step = self.finite_diff_step
        for i in range(state.size):
            bumped = state.copy()
            bumped[i] += step
            _, r_plus = self._eval(strategy_fn, bumped)
            grad[i] = (r_plus - base_reward) / step
        return grad

    def untargeted(self, strategy_fn: StrategyFn, state: np.ndarray) -> np.ndarray:
        """Return ``state + perturbation`` that minimises reward (FGSM)."""

        state = np.asarray(state, dtype=np.float64)
        grad = self._grad(strategy_fn, state)
        perturbation = -self.epsilon * np.sign(grad)
        return state + perturbation

    def targeted(
        self,
        strategy_fn: StrategyFn,
        state: np.ndarray,
        target_action: float,
    ) -> np.ndarray:
        """Return ``state + perturbation`` that drives the strategy's output
        towards ``target_action`` (e.g. +1 to force a buy)."""

        state = np.asarray(state, dtype=np.float64)
        # Numeric gradient of the *action* w.r.t. state.
        base_action = float(strategy_fn(state))
        grad = np.zeros_like(state)
        step = self.finite_diff_step
        for i in range(state.size):
            bumped = state.copy()
            bumped[i] += step
            a_plus = float(strategy_fn(bumped))
            grad[i] = (a_plus - base_action) / step
        # Move in the direction that increases action if target > current,
        # decrease otherwise.
        direction = np.sign(target_action - base_action) * np.sign(grad)
        return state + self.epsilon * direction

    def snapshot(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "finite_diff_step": self.finite_diff_step,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class RobustStrategyTrainer:
    """Harden a strategy against adversarial perturbations.

    Because many ARGUS strategies are *fixed* (rule-based or pre-trained
    externally) this class does not attempt to modify the strategy in place.
    Instead it provides:

    * ``train_strategy`` — returns a *new* strategy function that wraps the
      original and filters its raw output based on adversarial consistency
      (if small perturbations flip the sign of the action, the wrapped
      strategy emits a neutralised action).
    * ``evaluate_robustness`` — returns a 0..1 robustness score summarising
      how well the strategy holds up under adversarial states.
    * ``get_vulnerabilities`` — returns concrete adversarial examples that
      the strategy failed on, for audit/diagnostics.
    """

    def __init__(
        self,
        generator: Optional[AdversarialGenerator] = None,
        reward_fn: Optional[RewardFn] = None,
        consistency_threshold: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.generator = generator or AdversarialGenerator()
        self.reward_fn = reward_fn or _default_reward
        self.consistency_threshold = float(consistency_threshold)
        self._rng = np.random.default_rng(seed)
        self.vulnerabilities: List[Vulnerability] = []
        self.last_robustness: float = 0.0
        self.trainings: int = 0
        self.evaluations: int = 0

    # -- helper ----------------------------------------------------------

    def _assess_state(
        self, strategy_fn: StrategyFn, state: np.ndarray
    ) -> Tuple[float, float, Vulnerability]:
        """Evaluate ``state`` + its adversarial counterpart.

        Returns ``(original_reward, adversarial_reward, vulnerability)``.
        """

        orig_action = float(strategy_fn(state))
        orig_reward = float(self.reward_fn(state, orig_action))
        adv_state = self.generator.untargeted(strategy_fn, state)
        adv_action = float(strategy_fn(adv_state))
        adv_reward = float(self.reward_fn(adv_state, adv_action))
        vuln = Vulnerability(
            original_state=state.copy(),
            adversarial_state=adv_state,
            original_action=orig_action,
            adversarial_action=adv_action,
            original_reward=orig_reward,
            adversarial_reward=adv_reward,
            perturbation_norm=float(np.linalg.norm(adv_state - state)),
        )
        return orig_reward, adv_reward, vuln

    # -- API -------------------------------------------------------------

    def train_strategy(
        self,
        strategy_fn: StrategyFn,
        real_states: List[np.ndarray],
    ) -> StrategyFn:
        """Return a robustness-filtered wrapper around ``strategy_fn``.

        The wrapper checks adversarial consistency at call time: if a small
        perturbation would substantially change the raw action, the wrapper
        halves the magnitude of the output. States from ``real_states`` are
        used to calibrate the consistency threshold via their empirical
        action variance.
        """

        self.trainings += 1
        if not real_states:
            calibration = self.consistency_threshold
        else:
            diffs = []
            for s in real_states:
                s = np.asarray(s, dtype=np.float64)
                orig_action = float(strategy_fn(s))
                adv_state = self.generator.untargeted(strategy_fn, s)
                adv_action = float(strategy_fn(adv_state))
                diffs.append(abs(orig_action - adv_action))
            calibration = max(
                self.consistency_threshold,
                float(np.percentile(diffs, 75)) if diffs else self.consistency_threshold,
            )

        gen = self.generator

        def wrapped(state: np.ndarray) -> float:
            state = np.asarray(state, dtype=np.float64)
            raw = float(strategy_fn(state))
            adv = gen.untargeted(strategy_fn, state)
            adv_action = float(strategy_fn(adv))
            if abs(raw - adv_action) > calibration:
                return 0.5 * raw
            return raw

        return wrapped

    def evaluate_robustness(
        self,
        strategy_fn: StrategyFn,
        states: Optional[List[np.ndarray]] = None,
        n_random: int = 32,
        state_dim: int = 8,
    ) -> float:
        """Return a robustness score in ``[0, 1]``.

        The score is defined as the fraction of sampled states for which the
        adversarial reward is at least 50% of the clean reward (or where the
        clean reward was non-positive, a floor of 0.5 is applied so purely
        bad states don't punish robustness).
        """

        self.evaluations += 1
        if states is None:
            states = [
                self._rng.normal(scale=1.0, size=state_dim)
                for _ in range(n_random)
            ]
        if not states:
            self.last_robustness = 1.0
            return 1.0
        survived = 0
        self.vulnerabilities.clear()
        for s in states:
            orig_r, adv_r, vuln = self._assess_state(strategy_fn, np.asarray(s, dtype=np.float64))
            if orig_r <= 0:
                # Clean reward already non-positive — can't meaningfully say
                # the strategy was "robust" here. Give half-credit.
                survived += 1 if adv_r >= orig_r else 0
            else:
                if adv_r >= 0.5 * orig_r:
                    survived += 1
                else:
                    self.vulnerabilities.append(vuln)
        score = survived / max(1, len(states))
        self.last_robustness = float(score)
        return self.last_robustness

    def get_vulnerabilities(
        self,
        strategy_fn: StrategyFn,
        states: Optional[List[np.ndarray]] = None,
        n_random: int = 32,
        state_dim: int = 8,
    ) -> List[Vulnerability]:
        """Find adversarial examples that the strategy fails on.

        This is a thin convenience wrapper over :meth:`evaluate_robustness`
        that returns the collected vulnerabilities.
        """

        self.evaluate_robustness(
            strategy_fn,
            states=states,
            n_random=n_random,
            state_dim=state_dim,
        )
        return list(self.vulnerabilities)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "trainings": self.trainings,
            "evaluations": self.evaluations,
            "last_robustness": float(self.last_robustness),
            "vulnerabilities_found": len(self.vulnerabilities),
            "consistency_threshold": self.consistency_threshold,
            "generator": self.generator.snapshot(),
        }
