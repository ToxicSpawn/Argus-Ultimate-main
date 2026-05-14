"""
Market Speed Learning Bridge
==============================
Connects MarketSpeedOrchestrator to the full continuous adaptation stack.

WHAT THIS FILE DOES:
  - Instantiates all 4 ContinuousAdaptationEngine dependencies from existing
    Argus modules (ObservationRecorder, ParameterDriftOptimizer,
    AdaptiveGateManager, AdaptationHealthMonitor)
  - Monkey-patches MarketSpeedOrchestrator so every make_decision() call
    feeds the ObservationRecorder, and every record_trade_outcome() call
    completes the observation AND ticks the adaptation engine
  - Implements the _run_full_sweep() stub using BayesianOptimizer
  - Wires JaxPPOTrainer / FlaxPPOUpdater as a FAST-PATH policy updater that
    fires every JAX_UPDATE_EVERY_N_DECISIONS (default 10) outcomes, giving
    ~5 second weight-update cadence at 0.5 s tick speed
  - Exposes get_learning_bridge() singleton for use anywhere

DUAL-LOOP ARCHITECTURE:
  ┌─ 0.5 s tick ──────────────────────────────────────────────────────────┐
  │  make_decision() → [read JAX policy logits] → execute                 │
  └───────────────────────────────────────────────────────────────────────┘
         ↑ params updated in background
  ┌─ ~5 s cadence (10 outcomes) ──────────────────────────────────────────┐
  │  FlaxPPOUpdater.update(mini_rollout) → inference.update_params()      │
  └───────────────────────────────────────────────────────────────────────┘
  ┌─ ~25 s cadence (50 outcomes) ─────────────────────────────────────────┐
  │  ContinuousAdaptationEngine.tick() → param drift + gate adapt         │
  └───────────────────────────────────────────────────────────────────────┘
  ┌─ ~8 min cadence (1000 cycles) ────────────────────────────────────────┐
  │  BayesianOptimizer full sweep                                         │
  └───────────────────────────────────────────────────────────────────────┘

USAGE in main.py / launch scripts:
    from core.market_speed_learning_bridge import install_learning_bridge
    install_learning_bridge(portfolio_value=my_capital)   # call once

That single call wires everything. No other changes required.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import helpers — never crash if a module is absent
# ---------------------------------------------------------------------------

def _try_import(module_path: str, class_name: str) -> Optional[type]:
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name, None)
    except Exception as exc:
        logger.debug("LearningBridge: could not import %s.%s — %s", module_path, class_name, exc)
        return None


# ---------------------------------------------------------------------------
# ParameterDriftOptimizer shim
# ---------------------------------------------------------------------------

class _ParameterDriftOptimizer:
    """
    Minimal parameter drift optimizer.
    Tracks registered parameters and drifts them toward recent performance
    gradient using a simple +/- epsilon hill-climber.
    """

    def __init__(self, max_drift_pct: float = 0.01) -> None:
        self._params: Dict[str, Dict[str, Any]] = {}
        self._max_drift = max_drift_pct
        self._pnl_history: list = []

    def register(
        self,
        name: str,
        initial: float,
        min_val: float,
        max_val: float,
        description: str = "",
    ) -> None:
        if name not in self._params:
            self._params[name] = {
                "current_value": initial,
                "initial_value": initial,
                "min": min_val,
                "max": max_val,
                "description": description,
                "direction": 1,
                "last_pnl": 0.0,
            }

    def feed_pnl(self, pnl: float) -> None:
        self._pnl_history.append(pnl)
        if len(self._pnl_history) > 200:
            self._pnl_history.pop(0)

    def compute_drifts(self, regime: str = "") -> Dict[str, float]:
        if not self._pnl_history:
            return {}
        recent_pnl = sum(self._pnl_history[-20:]) if len(self._pnl_history) >= 20 else 0.0
        drifts: Dict[str, float] = {}
        for name, state in self._params.items():
            cur = state["current_value"]
            rng = state["max"] - state["min"]
            epsilon = rng * self._max_drift
            direction = state["direction"] if recent_pnl >= 0 else -state["direction"]
            new_val = cur + direction * epsilon
            new_val = max(state["min"], min(state["max"], new_val))
            drifts[name] = new_val
        return drifts

    def apply_drifts(self, drifts: Dict[str, float]) -> int:
        applied = 0
        for name, new_val in drifts.items():
            if name in self._params:
                self._params[name]["current_value"] = new_val
                applied += 1
        return applied

    def get_parameter_state(self, name: str) -> Optional[Dict[str, Any]]:
        return self._params.get(name)

    def check_for_reverts(self, signals: Dict[str, float]) -> None:
        for name, signal in signals.items():
            if name in self._params and signal < -5.0:
                self._params[name]["current_value"] = self._params[name]["initial_value"]
                logger.info("ParameterDriftOptimizer: reverted %s to initial", name)

    def get_current_params(self) -> Dict[str, float]:
        return {k: v["current_value"] for k, v in self._params.items()}


# ---------------------------------------------------------------------------
# JAX Fast-Path Manager
# ---------------------------------------------------------------------------

class _JaxFastPath:
    """
    Wraps JaxPPOTrainer + FlaxPPOUpdater + PolicyInference into a lightweight
    fast-path updater that:

      1. Accumulates (obs, action, reward, log_prob, value) tuples from live
         trading into a mini-rollout buffer.
      2. Every JAX_UPDATE_EVERY_N_OUTCOMES outcomes, runs one FlaxPPOUpdater
         update step (JIT-compiled, ~2-5 ms on GPU / ~15 ms on CPU).
      3. Writes updated params back to PolicyInference so the next
         make_decision() forward-pass uses fresh weights.
      4. Exposes action_and_value() for optional use by the orchestrator.

    Falls back gracefully to a no-op if JAX/Flax are not installed.
    """

    UPDATE_EVERY: int = 10          # PPO update every N outcomes (~5 s at 0.5 s ticks)
    MIN_ROLLOUT: int = 10           # never update with fewer than this many samples
    MAX_ROLLOUT: int = 512          # cap rollout buffer size

    def __init__(self) -> None:
        self._available = False
        self._inference = None
        self._updater = None
        self._rollout_buffer: List[Dict[str, Any]] = []
        self._outcome_counter: int = 0
        self._total_updates: int = 0
        self._last_loss: float = 0.0
        self._lock = threading.Lock()

        self._init_jax()

    def _init_jax(self) -> None:
        try:
            from core.jax_policy_network import NetworkFactory, PolicyConfig
            from core.jax_rl_trainer import JaxRLConfig

            # Build network + updater via factory
            inference, updater = NetworkFactory.build(
                config=PolicyConfig(
                    obs_dim=83,        # matches JaxRLEnvironment Batch-3 obs space
                    n_actions=5,
                    hidden_dims=(256, 256, 128),
                    learning_rate=3e-4,
                ),
                seed=42,
            )
            self._inference = inference
            self._updater = updater
            self._available = True
            logger.info(
                "JaxFastPath: PolicyInference + FlaxPPOUpdater ready "
                "(update every %d outcomes)",
                self.UPDATE_EVERY,
            )
        except Exception as exc:
            logger.warning(
                "JaxFastPath: JAX/Flax not available — fast-path disabled (%s). "
                "Install with: pip install jax[cuda12] flax optax",
                exc,
            )

    # ------------------------------------------------------------------
    # Called at every make_decision() — run a forward pass and stash result
    # ------------------------------------------------------------------

    def forward(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Run JIT forward pass. Returns (action, log_prob, value).
        Falls back to (0, 0.0, 0.0) if JAX unavailable.
        """
        if not self._available or self._inference is None:
            return 0, 0.0, 0.0
        try:
            return self._inference.action_and_value(obs)
        except Exception as exc:
            logger.debug("JaxFastPath.forward error: %s", exc)
            return 0, 0.0, 0.0

    # ------------------------------------------------------------------
    # Called at every record_trade_outcome() — buffer + maybe update
    # ------------------------------------------------------------------

    def record(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
    ) -> None:
        if not self._available:
            return

        with self._lock:
            self._rollout_buffer.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "log_prob": log_prob,
                "value": value,
            })
            # Trim buffer to cap
            if len(self._rollout_buffer) > self.MAX_ROLLOUT:
                self._rollout_buffer = self._rollout_buffer[-self.MAX_ROLLOUT:]

            self._outcome_counter += 1

            if (
                self._outcome_counter % self.UPDATE_EVERY == 0
                and len(self._rollout_buffer) >= self.MIN_ROLLOUT
            ):
                self._run_update()

    # ------------------------------------------------------------------
    # PPO mini-update — runs inside the lock
    # ------------------------------------------------------------------

    def _run_update(self) -> None:
        buf = self._rollout_buffer[-self.MAX_ROLLOUT:]
        if not buf:
            return

        try:
            import numpy as _np

            obs_arr   = _np.array([s["obs"]      for s in buf], dtype=_np.float32)
            act_arr   = _np.array([s["action"]   for s in buf], dtype=_np.int32)
            rew_arr   = _np.array([s["reward"]   for s in buf], dtype=_np.float32)
            lp_arr    = _np.array([s["log_prob"] for s in buf], dtype=_np.float32)
            val_arr   = _np.array([s["value"]    for s in buf], dtype=_np.float32)

            # Simple Monte-Carlo returns (no bootstrapping needed for short rollout)
            returns = _np.zeros_like(rew_arr)
            running = 0.0
            for i in reversed(range(len(rew_arr))):
                running = rew_arr[i] + 0.99 * running
                returns[i] = running

            rollout = {
                "obs":       obs_arr,
                "actions":   act_arr,
                "log_probs": lp_arr,
                "values":    val_arr,
                "returns":   returns,
            }

            t0 = time.monotonic()
            loss = self._updater.update(rollout)
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Push new params into inference
            self._inference.update_params(self._updater.params)

            self._last_loss = loss
            self._total_updates += 1

            logger.info(
                "JaxFastPath: PPO update #%d | loss=%.4f | samples=%d | %.1f ms",
                self._total_updates, loss, len(buf), elapsed_ms,
            )

            # Clear buffer after successful update
            self._rollout_buffer.clear()

        except Exception as exc:
            logger.warning("JaxFastPath: update failed — %s", exc)

    # ------------------------------------------------------------------
    # Build a minimal obs vector from live market data
    # ------------------------------------------------------------------

    @staticmethod
    def obs_from_decision(decision: Any, price: float) -> np.ndarray:
        """
        Construct an 83-dim obs vector from a live decision object.
        Uses whatever fields are available; fills missing dims with zeros.
        """
        obs = np.zeros(83, dtype=np.float32)
        try:
            # Positions 0-79: LOB depth features (unavailable live → zeros)
            # Position 80: mid_price (normalised)
            obs[80] = float(price) / 100_000.0
            # Position 81: position size
            obs[81] = float(getattr(decision, "position_size", 0.0))
            # Position 82: confidence
            obs[82] = float(getattr(decision, "confidence", 0.0))
        except Exception:
            pass
        return obs

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "total_ppo_updates": self._total_updates,
            "last_loss": self._last_loss,
            "buffer_size": len(self._rollout_buffer),
            "outcome_counter": self._outcome_counter,
            "update_every_n": self.UPDATE_EVERY,
        }


# ---------------------------------------------------------------------------
# Main bridge class
# ---------------------------------------------------------------------------

class MarketSpeedLearningBridge:
    """
    Wires the full ContinuousAdaptationEngine stack + JAX fast-path into
    MarketSpeedOrchestrator.

    Components assembled:
        ObservationRecorder          — experience replay buffer
        _ParameterDriftOptimizer     — hill-climber param drift (or Argus own)
        AdaptiveGateManager          — confidence / edge gate adaptation
        AdaptationHealthMonitor      — throttle/revert safety guard
        ContinuousAdaptationEngine   — master orchestrator for all of the above
        BayesianOptimizer            — wired into full_sweep (every 1000 cycles)
        _JaxFastPath                 — JIT PPO update every 10 outcomes (~5 s)
    """

    ADAPT_TICK_EVERY_N_DECISIONS = 50
    BAYESIAN_SWEEP_EVERY_N_CYCLES = 1000

    def __init__(self) -> None:
        self._decision_counter = 0
        self._portfolio_value: float = 10_000.0
        self._current_regime: str = "NORMAL"
        self._cumulative_pnl: float = 0.0
        self._lock = threading.Lock()

        self._pending_obs: Dict[str, str] = {}

        # ── 1. ObservationRecorder ──────────────────────────────────────────
        from core.observation_recorder import ObservationRecorder
        self.recorder = ObservationRecorder(
            max_size=100_000,
            persist_path="data/observations/observations.jsonl",
            persist_interval_obs=500,
        )
        logger.info("LearningBridge: ObservationRecorder ready")

        # ── 2. ParameterDriftOptimizer ──────────────────────────────────────
        PDO = _try_import("core.parameter_drift_optimizer", "ParameterDriftOptimizer")
        self.param_optimizer = PDO() if PDO else _ParameterDriftOptimizer()
        logger.info("LearningBridge: ParameterDriftOptimizer ready (%s)", type(self.param_optimizer).__name__)

        # ── 3. AdaptiveGateManager ──────────────────────────────────────────
        AGM = _try_import("core.adaptive_gate_manager", "AdaptiveGateManager")
        self.gate_manager = AGM() if AGM else None
        if self.gate_manager:
            logger.info("LearningBridge: AdaptiveGateManager ready")
        else:
            logger.warning("LearningBridge: AdaptiveGateManager not available — gate adaptation disabled")

        # ── 4. AdaptationHealthMonitor ──────────────────────────────────────
        AHM = _try_import("core.adaptation_health_monitor", "AdaptationHealthMonitor")
        self.health_monitor = AHM() if AHM else None
        if self.health_monitor:
            logger.info("LearningBridge: AdaptationHealthMonitor ready")
        else:
            logger.warning("LearningBridge: AdaptationHealthMonitor not available — auto-revert disabled")

        # ── 5. ContinuousAdaptationEngine ───────────────────────────────────
        from core.continuous_adaptation_engine import (
            ContinuousAdaptationEngine,
            AdaptationConfig,
            AdaptationMode,
        )
        cfg = AdaptationConfig(
            enabled=True,
            mode=AdaptationMode.NORMAL,
            parameter_drift_cycles=self.ADAPT_TICK_EVERY_N_DECISIONS,
            gate_adapt_cycles=self.ADAPT_TICK_EVERY_N_DECISIONS * 2,
            health_check_cycles=500,
            full_sweep_cycles=self.BAYESIAN_SWEEP_EVERY_N_CYCLES,
            max_simultaneous_adaptations=10,
            auto_throttle_enabled=True,
            auto_revert_enabled=True,
        )
        self.adaptation_engine = ContinuousAdaptationEngine(
            observation_recorder=self.recorder,
            parameter_optimizer=self.param_optimizer,
            gate_manager=self.gate_manager,
            health_monitor=self.health_monitor,
            config=cfg,
        )
        self.adaptation_engine.register_default_parameters()
        logger.info("LearningBridge: ContinuousAdaptationEngine ready")

        # ── 6. BayesianOptimizer (for full_sweep) ──────────────────────────
        BO = _try_import("core.bayesian_optimizer", "BayesianOptimizer")
        self.bayesian_optimizer = BO() if BO else None
        if self.bayesian_optimizer:
            self.adaptation_engine._run_full_sweep = self._run_full_sweep_impl
            logger.info("LearningBridge: BayesianOptimizer wired into full_sweep")
        else:
            logger.warning("LearningBridge: BayesianOptimizer not available — full_sweep remains a no-op")

        # ── 7. EWC Continual Learner (optional) ────────────────────────────
        EWC = _try_import("core.ewc_continual_learner", "EWCContinualLearner")
        self.ewc_learner = EWC() if EWC else None
        if self.ewc_learner:
            logger.info("LearningBridge: EWCContinualLearner ready — catastrophic forgetting protection active")

        # ── 8. JAX Fast-Path (NEW) ──────────────────────────────────────────
        self.jax_fast_path = _JaxFastPath()
        logger.info(
            "LearningBridge: JAX fast-path %s",
            "ACTIVE (PPO update every %d outcomes)" % self.jax_fast_path.UPDATE_EVERY
            if self.jax_fast_path._available
            else "INACTIVE (JAX/Flax not installed)",
        )

        logger.info("LearningBridge: all components assembled ✓")

    # -----------------------------------------------------------------------
    # Decision hook — called BEFORE the orchestrator returns a decision
    # -----------------------------------------------------------------------

    def on_decision(self, decision: Any, price: float, portfolio_value: float) -> Tuple[str, int, float, float]:
        """
        Record the decision and run a JAX forward pass.

        Returns (obs_id, jax_action, jax_log_prob, jax_value).
        The jax_action is advisory only — the orchestrator's own logic
        is already baked into `decision` before this is called.
        """
        self._portfolio_value = portfolio_value
        self._current_regime = getattr(decision, "regime", "NORMAL")

        obs = self.recorder.record_decision(
            symbol=getattr(decision, "symbol", "BTC/USD"),
            regime=self._current_regime,
            price=price,
            strategy="argus_unified",
            action=getattr(decision, "action", "hold").upper(),
            confidence=getattr(decision, "confidence", 0.0),
            final_size_pct=getattr(decision, "position_size", 0.0),
            cycle_number=self._decision_counter,
            portfolio_value_aud=portfolio_value,
        )

        # JAX forward pass
        obs_vec = _JaxFastPath.obs_from_decision(decision, price)
        jax_action, jax_log_prob, jax_value = self.jax_fast_path.forward(obs_vec)

        # Attach JAX advisory fields to the decision object (non-breaking)
        decision._obs_id = obs.obs_id
        decision._jax_obs_vec = obs_vec
        decision._jax_action = jax_action
        decision._jax_log_prob = jax_log_prob
        decision._jax_value = jax_value

        return obs.obs_id, jax_action, jax_log_prob, jax_value

    # -----------------------------------------------------------------------
    # Outcome hook — called when a trade result is known
    # -----------------------------------------------------------------------

    def on_outcome(self, obs_id: str, decision: Any, pnl: float) -> None:
        """
        Complete the observation, feed the JAX fast-path buffer, and tick
        the adaptation engine.
        """
        # Complete observation record
        self.recorder.complete_observation(
            obs_id=obs_id,
            executed=True,
            pnl_aud=pnl,
            exit_reason="market",
        )

        # Feed PnL to param optimizer
        if hasattr(self.param_optimizer, "feed_pnl"):
            self.param_optimizer.feed_pnl(pnl)

        # Feed EWC
        if self.ewc_learner and hasattr(self.ewc_learner, "record_outcome"):
            try:
                self.ewc_learner.record_outcome(pnl=pnl, regime=self._current_regime)
            except Exception:
                pass

        # ── JAX fast-path: buffer this outcome and maybe run PPO update ──
        obs_vec = getattr(decision, "_jax_obs_vec", None)
        if obs_vec is not None:
            self.jax_fast_path.record(
                obs=obs_vec,
                action=getattr(decision, "_jax_action", 0),
                reward=float(pnl),
                log_prob=getattr(decision, "_jax_log_prob", 0.0),
                value=getattr(decision, "_jax_value", 0.0),
            )

        # Update running totals
        with self._lock:
            self._cumulative_pnl += pnl
            self._decision_counter += 1

        # Slow-path adaptation engine every 50 decisions
        if self._decision_counter % self.ADAPT_TICK_EVERY_N_DECISIONS == 0:
            self._tick_adaptation()

    # -----------------------------------------------------------------------
    # Adaptation engine tick
    # -----------------------------------------------------------------------

    def _tick_adaptation(self) -> None:
        try:
            result = self.adaptation_engine.tick(
                cycle_number=self._decision_counter,
                portfolio_value_aud=self._portfolio_value,
                current_regime=self._current_regime,
                cumulative_pnl=self._cumulative_pnl,
            )
            actions = result.get("actions", [])
            if actions:
                logger.info(
                    "LearningBridge [cycle %d | regime=%s | pnl=%.2f]: %s",
                    self._decision_counter,
                    self._current_regime,
                    self._cumulative_pnl,
                    ", ".join(actions),
                )
        except Exception as exc:
            logger.warning("LearningBridge: adaptation tick error — %s", exc)

    # -----------------------------------------------------------------------
    # Full sweep
    # -----------------------------------------------------------------------

    def _run_full_sweep_impl(self) -> None:
        logger.info("LearningBridge: starting full Bayesian sweep (cycle=%d)", self._decision_counter)
        try:
            recent_obs = self.recorder.recent(hours=48)
            completed = [o for o in recent_obs if o.executed and o.pnl_aud != 0.0]

            if len(completed) < 30:
                logger.info("LearningBridge: skipping Bayesian sweep — only %d completed obs", len(completed))
                return

            if hasattr(self.bayesian_optimizer, "suggest") and hasattr(self.bayesian_optimizer, "observe"):
                for _ in range(5):
                    candidate = self.bayesian_optimizer.suggest()
                    if candidate is None:
                        break
                    ct = candidate.get("confidence_threshold", 0.55)
                    relevant = [o for o in completed if abs(o.confidence - ct) < 0.15]
                    score = sum(o.pnl_aud for o in relevant) / max(len(relevant), 1)
                    self.bayesian_optimizer.observe(candidate, score)
                    logger.debug("LearningBridge Bayesian step: candidate=%s score=%.4f", candidate, score)
            else:
                logger.debug("LearningBridge: BayesianOptimizer has no suggest/observe — skipping")

            logger.info("LearningBridge: full Bayesian sweep complete")
        except Exception as exc:
            logger.warning("LearningBridge: full_sweep error — %s", exc)

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "decisions": self._decision_counter,
            "cumulative_pnl": self._cumulative_pnl,
            "portfolio_value": self._portfolio_value,
            "current_regime": self._current_regime,
            "adaptation_engine": self.adaptation_engine.snapshot(),
            "observation_recorder": self.recorder.snapshot(),
            "current_params": self.param_optimizer.get_current_params()
            if hasattr(self.param_optimizer, "get_current_params") else {},
            "jax_fast_path": self.jax_fast_path.snapshot(),
        }


# ---------------------------------------------------------------------------
# Patch installer
# ---------------------------------------------------------------------------

_bridge_instance: Optional[MarketSpeedLearningBridge] = None
_bridge_lock = threading.Lock()


def get_learning_bridge() -> MarketSpeedLearningBridge:
    global _bridge_instance
    if _bridge_instance is None:
        with _bridge_lock:
            if _bridge_instance is None:
                _bridge_instance = MarketSpeedLearningBridge()
    return _bridge_instance


def install_learning_bridge(
    portfolio_value: float = 10_000.0,
    adapt_every_n_decisions: int = 50,
    jax_update_every_n_outcomes: int = 10,
) -> MarketSpeedLearningBridge:
    """
    Wire the learning bridge into MarketSpeedOrchestrator.

    Patches make_decision() and record_trade_outcome() so the full stack —
    including the JAX fast-path PPO updater — runs automatically.

    Parameters
    ----------
    portfolio_value : float
        Starting capital; used by the health monitor for drawdown detection.
    adapt_every_n_decisions : int
        How often the slow-path ContinuousAdaptationEngine ticks (default 50).
    jax_update_every_n_outcomes : int
        How often the JAX PPO fast-path updates network weights (default 10,
        ~5 seconds at 0.5 s tick speed). Set higher to reduce CPU/GPU load.

    Returns the bridge instance for optional direct access.
    """
    from core.market_speed_orchestrator import MarketSpeedOrchestrator

    bridge = get_learning_bridge()
    bridge._portfolio_value = portfolio_value
    bridge.ADAPT_TICK_EVERY_N_DECISIONS = adapt_every_n_decisions
    bridge.jax_fast_path.UPDATE_EVERY = jax_update_every_n_outcomes

    # ── Patch make_decision ──────────────────────────────────────────────────
    _original_make_decision = MarketSpeedOrchestrator.make_decision

    def _patched_make_decision(self_orch, price, volume, symbol="BTC/USD", metadata=None):
        decision = _original_make_decision(self_orch, price, volume, symbol, metadata)

        # Record + JAX forward pass
        bridge.on_decision(
            decision=decision,
            price=price,
            portfolio_value=bridge._portfolio_value,
        )
        # _obs_id, _jax_action, _jax_log_prob, _jax_value are now on decision

        return decision

    MarketSpeedOrchestrator.make_decision = _patched_make_decision

    # ── Patch record_trade_outcome ───────────────────────────────────────────
    _original_record_outcome = MarketSpeedOrchestrator.record_trade_outcome

    def _patched_record_outcome(self_orch, decision, actual_pnl):
        _original_record_outcome(self_orch, decision, actual_pnl)

        obs_id = getattr(decision, "_obs_id", None)
        if obs_id:
            bridge.on_outcome(obs_id=obs_id, decision=decision, pnl=actual_pnl)

    MarketSpeedOrchestrator.record_trade_outcome = _patched_record_outcome

    logger.info(
        "LearningBridge: installed into MarketSpeedOrchestrator "
        "(adapt_every=%d decisions | jax_update_every=%d outcomes | portfolio=%.2f)",
        adapt_every_n_decisions,
        jax_update_every_n_outcomes,
        portfolio_value,
    )

    return bridge


__all__ = [
    "MarketSpeedLearningBridge",
    "install_learning_bridge",
    "get_learning_bridge",
]
