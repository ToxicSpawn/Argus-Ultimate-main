"""
ml/model_rolling/deployment_orchestrator.py
============================================
Orchestrates safe model deployments through shadow → canary → production,
with automatic rollback on drift or performance degradation.

Supports:
  - Shadow mode: new model runs in parallel, predictions compared
  - Canary: gradual traffic increase (10 → 25 → 50 → 100 %)
  - A/B testing: statistically significant win/loss detection
  - Automatic rollback: triggers when drift alerts fire
  - Manual hold / promote / rollback via API
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

from ml.model_rolling.state_tracker import (
    ModelLifecycleState,
    RolloutEvent,
    RolloutStage,
)
from ml.model_rolling.drift_detector import (
    DriftAlert,
    DriftStatus,
    RollingDriftDetector,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ShadowMode(Enum):
    OFF        = "off"
    SHADOW_ONLY = "shadow_only"   # only shadow runs, champion serves
    COMPARISON  = "comparison"   # both run, results compared


@dataclass
class DeploymentPolicy:
    """
    Tunable deployment policy for a model.
    """
    # Drift thresholds
    psi_rollback_threshold    : float = 0.5
    warning_threshold         : float = 0.2

    # Performance degradation
    min_sharpe_threshold      : float = 0.5
    min_win_rate              : float = 0.45
    min_trades_for_significance: int  = 100

    # Canary stages (traffic fractions)
    canary_stages             : List[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 1.0]
    )
    canary_min_duration_seconds: float = 300.0   # 5 min per stage minimum
    canary_promote_threshold   : float = 0.05   # challenger must beat champion by ≥ 5 %

    # Auto-recovery
    auto_rollback_on_drift    : bool  = True
    rollback_on_degradation   : bool  = True
    check_interval_seconds    : float = 30.0

    # A/B test
    ab_confidence_level       : float = 0.95


@dataclass
class ShadowResult:
    """Outcome from one shadow evaluation cycle."""
    shadow_version : str
    champion_version: str
    n_samples      : int
    shadow_sharpe  : float
    champion_sharpe: float
    shadow_error   : float
    champion_error : float
    winner         : str            # "shadow", "champion", "inconclusive"
    mean_abs_error_shadow  : float
    mean_abs_error_champion: float
    timestamp      : float


@dataclass
class CanaryReport:
    """Performance report at current canary stage."""
    challenger_version : str
    champion_version   : str
    stage              : RolloutStage
    n_samples          : int
    challenger_sharpe  : float
    champion_sharpe    : float
    challenger_win_rate: float
    p_value            : float        # Welch's t-test p-value
    is_significant     : bool
    recommendation     : str          # "promote", "demote", "hold"
    confidence         : float


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    event_id           : str
    model_name         : str
    from_version       : str
    to_version         : str
    reason             : str
    trigger_alert      : Optional[DriftAlert]
    timestamp          : float


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _welch_ttest(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t-test p-value for two independent samples."""
    n0, n1 = len(a), len(b)
    if n0 < 2 or n1 < 2:
        return 1.0
    m0, m1 = np.mean(a), np.mean(b)
    v0, v1 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(v0 / n0 + v1 / n1)
    if se < 1e-12:
        return 1.0
    t_stat = abs(m0 - m1) / se
    # Welch-Satterthwaite degrees of freedom
    num = (v0 / n0 + v1 / n1) ** 2
    denom = (v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1)
    df = num / denom if denom > 0 else 1
    # Approximate p-value using normal CDF
    p = 2.0 * (1.0 - _norm_cdf(t_stat))
    return max(0.0, min(1.0, p))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - p * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _sharpe_from_returns(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from return series."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(252 * 24))  # per-hour assumption


# ---------------------------------------------------------------------------
# Model factory protocol
# ---------------------------------------------------------------------------

class ModelFactory(Protocol):
    """Callable that returns a loaded model object."""
    def __call__(self, model_name: str, version: str) -> Any: ...


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class ModelRollingPipeline:
    """
    Orchestrates the full model rolling lifecycle.

    Parameters
    ----------
    model_name        : str              — name of the model being managed
    champion_loader   : ModelFactory      — loads champion (production) model
    challenger_loader : ModelFactory      — loads challenger (candidate) model
    state_path        : str              — path for persisting lifecycle state
    policy            : DeploymentPolicy — deployment thresholds
    """

    def __init__(
        self,
        model_name       : str,
        champion_loader  : ModelFactory,
        challenger_loader: ModelFactory,
        state_path       : str = "data/model_rolling",
        policy           : Optional[DeploymentPolicy] = None,
    ) -> None:
        self.model_name = model_name
        self.champion_loader  = champion_loader
        self.challenger_loader = challenger_loader
        self.policy = policy or DeploymentPolicy()
        self.state_path = Path(state_path)
        self.state_path.mkdir(parents=True, exist_ok=True)

        self._state_file = self.state_path / f"{model_name}_lifecycle.json"

        # Load or create lifecycle state
        self._lifecycle = self._load_lifecycle()

        # Per-signal drift detectors
        self._drift = RollingDriftDetector(model_name)
        # Register default signals
        for sig in ["sharpe", "mse", "win_rate"]:
            self._drift.register_signal(
                sig,
                rollback_threshold=self.policy.psi_rollback_threshold,
            )

        # Shadow and canary evaluation buffers
        self._champion_preds : deque = deque(maxlen=2000)
        self._champion_actuals: deque = deque(maxlen=2000)
        self._champion_rets   : deque = deque(maxlen=2000)
        self._shadow_preds    : deque = deque(maxlen=2000)
        self._shadow_actuals  : deque = deque(maxlen=2000)
        self._shadow_rets    : deque = deque(maxlen=2000)

        # Shadow mode
        self._shadow_mode = ShadowMode.OFF

        # Rollback event log
        self._rollback_events: List[RollbackEvent] = []

        # Current canary stage metadata
        self._canary_entered_at : float = 0.0
        self._canary_version    : str   = ""

        # Control flags
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()

        # Callbacks: (alert: DriftAlert) -> None
        self._on_rollback_callbacks : List[Callable[[RollbackEvent], None]] = []
        self._on_promote_callbacks  : List[Callable[[str], None]] = []

    # ------------------------------------------------------------------ API

    def champion_version(self) -> Optional[str]:
        return self._lifecycle.current_champion()

    def challenger_version(self) -> Optional[str]:
        return self._lifecycle.challenger()

    def rollout_stage(self) -> Optional[RolloutStage]:
        return self._lifecycle.current_champion_stage()

    def shadow_mode(self) -> ShadowMode:
        return self._shadow_mode

    def lifecycle_history(self) -> List[RolloutEvent]:
        return self._lifecycle.history()

    def rollback_history(self) -> List[RollbackEvent]:
        return list(self._rollback_events)

    def drift_summary(self) -> Dict[str, int]:
        return self._drift.summary()

    # ------------------------------------------------------------------ Lifecycle transitions

    def register_challenger(self, version: str) -> RolloutEvent:
        """
        Register a new challenger version (trained externally).
        Immediately enters SHADOW mode.
        """
        ev = self._lifecycle.promote_to_shadow(version)
        self._shadow_mode = ShadowMode.COMPARISON
        self._save_lifecycle()
        logger.info("ModelRolling [%s]: challenger '%s' registered in shadow mode", self.model_name, version)
        return ev

    def promote_challenger(self) -> Optional[RolloutEvent]:
        """
        Promote the current challenger to canary at 10 % traffic.
        """
        challenger = self._lifecycle.challenger()
        if not challenger:
            logger.warning("ModelRolling [%s]: no challenger to promote", self.model_name)
            return None

        ev = self._lifecycle.promote_to_canary(challenger, 0.10)
        self._canary_version    = challenger
        self._canary_entered_at = time.time()
        self._save_lifecycle()
        logger.info("ModelRolling [%s]: challenger '%s' promoted to canary", self.model_name, challenger)
        return ev

    def advance_canary(self) -> Optional[RolloutEvent]:
        """
        Advance canary to next traffic stage if performance is acceptable.
        """
        challenger = self._canary_version or self._lifecycle.challenger()
        if not challenger:
            return None

        current_stage = self._lifecycle.stage_of(challenger)
        if current_stage is None:
            return None

        # Find next stage
        remaining = [s for s in self.policy.canary_stages if s >
                     self._current_canary_fraction(current_stage)]
        if not remaining:
            # Fully promoted — move to production
            return self._promote_to_production(challenger)

        next_frac = remaining[0]
        ev = self._lifecycle.promote_to_canary(challenger, next_frac)
        self._canary_entered_at = time.time()
        self._save_lifecycle()
        return ev

    def _current_canary_fraction(self, stage: RolloutStage) -> float:
        map_ = {
            RolloutStage.CANARY_10Pct: 0.10,
            RolloutStage.CANARY_25Pct: 0.25,
            RolloutStage.CANARY_50Pct: 0.50,
        }
        return map_.get(stage, 0.0)

    def _promote_to_production(self, version: str) -> RolloutEvent:
        ev = self._lifecycle.promote_to_production(version)
        self._shadow_mode = ShadowMode.OFF
        self._canary_version = ""
        self._save_lifecycle()
        for cb in self._on_promote_callbacks:
            try: cb(version)
            except Exception as e: logger.warning("on_promote callback error: %s", e)
        return ev

    def manual_rollback(self, to_version: Optional[str] = None) -> Optional[RolloutEvent]:
        """
        Manually rollback to a specific version or the previous champion.
        """
        target = to_version or self._lifecycle.previous()
        if not target:
            logger.warning("ModelRolling [%s]: no version to rollback to", self.model_name)
            return None
        ev = self._lifecycle.rollback_to(target)
        self._shadow_mode = ShadowMode.OFF
        self._canary_version = ""
        self._save_lifecycle()
        self._record_rollback(
            from_ver=self._lifecycle.current_champion() or "unknown",
            to_ver=target,
            reason="manual_rollback",
            alert=None,
        )
        return ev

    def hold_canary(self) -> None:
        """Pause canary advancement. Resets the stage timer."""
        self._canary_entered_at = time.time()

    def disable_shadow(self) -> None:
        self._shadow_mode = ShadowMode.OFF

    # ------------------------------------------------------------------ Record predictions

    def record_champion(
        self,
        prediction: float,
        actual    : float,
        ret       : float,       # single-period return
        timestamp : Optional[float] = None,
    ) -> None:
        ts = timestamp or time.time()
        self._champion_preds.append(prediction)
        self._champion_actuals.append(actual)
        self._champion_rets.append(ret)
        self._drift.record("sharpe", ret, ret)  # drift on returns
        self._drift.record("mse", prediction, actual)

    def record_shadow(
        self,
        prediction: float,
        actual    : float,
        ret       : float,
        timestamp : Optional[float] = None,
    ) -> None:
        if self._shadow_mode == ShadowMode.OFF:
            return
        ts = timestamp or time.time()
        self._shadow_preds.append(prediction)
        self._shadow_actuals.append(actual)
        self._shadow_rets.append(ret)
        self._drift.record("sharpe", ret, ret)
        self._drift.record("mse", prediction, actual)

    # ------------------------------------------------------------------ Evaluation

    def get_shadow_result(self) -> Optional[ShadowResult]:
        """Compare shadow vs champion over recorded samples."""
        if len(self._shadow_preds) < 10:
            return None

        champ_rets = np.array(self._champion_rets)
        shad_rets = np.array(self._shadow_rets)

        champ_sharpe  = _sharpe_from_returns(champ_rets)
        shadow_sharpe = _sharpe_from_returns(shad_rets)

        champ_errs = np.abs(np.array(self._champion_preds) - np.array(self._champion_actuals))
        shad_errs  = np.abs(np.array(self._shadow_preds)  - np.array(self._shadow_actuals))

        winner = "inconclusive"
        if shadow_sharpe > champ_sharpe * (1 + self.policy.canary_promote_threshold):
            winner = "shadow"
        elif champ_sharpe > shadow_sharpe * (1 + self.policy.canary_promote_threshold):
            winner = "champion"

        return ShadowResult(
            shadow_version  = self._lifecycle.challenger() or "unknown",
            champion_version= self._lifecycle.current_champion() or "unknown",
            n_samples       = len(self._shadow_rets),
            shadow_sharpe   = shadow_sharpe,
            champion_sharpe = champ_sharpe,
            shadow_error    = float(np.mean(shad_errs)),
            champion_error  = float(np.mean(champ_errs)),
            winner          = winner,
            mean_abs_error_shadow   = float(np.mean(shad_errs)),
            mean_abs_error_champion = float(np.mean(champ_errs)),
            timestamp       = time.time(),
        )

    def get_canary_report(self) -> Optional[CanaryReport]:
        """Performance report for the current canary stage."""
        challenger = self._canary_version or self._lifecycle.challenger()
        champion   = self._lifecycle.current_champion()

        if not challenger or not champion:
            return None

        if len(self._champion_rets) < self.policy.min_trades_for_significance:
            return CanaryReport(
                challenger_version = challenger,
                champion_version   = champion,
                stage              = RolloutStage.SHADOW,
                n_samples          = len(self._champion_rets),
                challenger_sharpe  = 0.0,
                champion_sharpe    = _sharpe_from_returns(np.array(self._champion_rets)),
                challenger_win_rate= 0.0,
                p_value            = 1.0,
                is_significant     = False,
                recommendation     = "hold",
                confidence         = 0.0,
            )

        champ_rets = np.array(self._champion_rets)
        shad_rets  = np.array(self._shadow_rets)

        champ_sharpe  = _sharpe_from_returns(champ_rets)
        shadow_sharpe = _sharpe_from_returns(shad_rets)

        # Win rate
        champ_wins = int(np.sum(champ_rets > 0))
        shad_wins  = int(np.sum(shad_rets  > 0))
        champ_win_rate = champ_wins / len(champ_rets) if len(champ_rets) > 0 else 0.0
        shad_win_rate  = shad_wins  / len(shad_rets)  if len(shad_rets)  > 0 else 0.0

        # Statistical test
        p_val = _welch_ttest(shad_rets, champ_rets)
        conf  = 1.0 - p_val
        is_sig = p_val < (1.0 - self.policy.ab_confidence_level)

        # Recommendation
        improvement = (shadow_sharpe - champ_sharpe) / abs(champ_sharpe) if champ_sharpe != 0 else 0.0

        if is_sig and improvement > self.policy.canary_promote_threshold:
            rec = "promote"
        elif is_sig and improvement < -self.policy.canary_promote_threshold:
            rec = "demote"
        else:
            rec = "hold"

        current_stage = self._lifecycle.stage_of(challenger) or RolloutStage.SHADOW

        return CanaryReport(
            challenger_version = challenger,
            champion_version   = champion,
            stage             = current_stage,
            n_samples         = len(champ_rets),
            challenger_sharpe = shadow_sharpe,
            champion_sharpe   = champ_sharpe,
            challenger_win_rate= shad_win_rate,
            p_value           = p_val,
            is_significant    = is_sig,
            recommendation    = rec,
            confidence        = conf,
        )

    # ------------------------------------------------------------------ Drift check & auto-rollback

    def check_drift_and_step(self) -> List[DriftAlert]:
        """
        Called periodically by the background monitor.
        Checks drift and takes automatic action based on policy.
        Returns list of fired alerts.
        """
        alerts = self._drift.check()
        rollback_fired = False

        for alert in alerts:
            if alert.should_rollback and self.policy.auto_rollback_on_drift:
                logger.warning(
                    "ModelRolling [%s]: auto-rollback triggered by drift alert on '%s'",
                    self.model_name, alert.signal,
                )
                self._do_auto_rollback(alert)
                rollback_fired = True

        # Canary stage advancement check
        if not rollback_fired and self._canary_version:
            report = self.get_canary_report()
            if report and report.recommendation == "promote":
                stage = self._lifecycle.stage_of(self._canary_version)
                # Only auto-advance if minimum duration has passed
                elapsed = time.time() - self._canary_entered_at
                if elapsed >= self.policy.canary_min_duration_seconds:
                    logger.info(
                        "ModelRolling [%s]: auto-advancing canary (recommendation=promote)",
                        self.model_name,
                    )
                    self.advance_canary()

        return alerts

    def _do_auto_rollback(self, alert: DriftAlert) -> None:
        prev = self._lifecycle.previous()
        if not prev:
            prev = self._lifecycle.current_champion()
        self._record_rollback(
            from_ver=self._lifecycle.current_champion() or "",
            to_ver=prev or "",
            reason=f"drift_on_{alert.signal}",
            alert=alert,
        )
        if prev:
            self._lifecycle.rollback_to(prev)
        else:
            self._lifecycle.mark_rolled_back(self._lifecycle.current_champion() or "")
        self._shadow_mode  = ShadowMode.OFF
        self._canary_version = ""
        self._save_lifecycle()

    def _record_rollback(
        self,
        from_ver: str,
        to_ver  : str,
        reason  : str,
        alert   : Optional[DriftAlert],
    ) -> None:
        ev = RollbackEvent(
            event_id        = uuid.uuid4().hex[:12],
            model_name      = self.model_name,
            from_version    = from_ver,
            to_version      = to_ver,
            reason          = reason,
            trigger_alert   = alert,
            timestamp       = time.time(),
        )
        self._rollback_events.append(ev)
        for cb in self._on_rollback_callbacks:
            try: cb(ev)
            except Exception as e: logger.warning("on_rollback callback error: %s", e)
        logger.warning(
            "ModelRolling [%s]: ROLLBACK %s → %s | reason=%s",
            self.model_name, from_ver, to_ver, reason,
        )

    # ------------------------------------------------------------------ Callbacks

    def on_rollback(self, cb: Callable[[RollbackEvent], None]) -> None:
        self._on_rollback_callbacks.append(cb)

    def on_promote(self, cb: Callable[[str], None]) -> None:
        self._on_promote_callbacks.append(cb)

    # ------------------------------------------------------------------ Persistence

    def _load_lifecycle(self) -> ModelLifecycleState:
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                return ModelLifecycleState.from_dict(data)
            except Exception as e:
                logger.warning("Failed to load lifecycle state: %s", e)
        return ModelLifecycleState(self.model_name)

    def _save_lifecycle(self) -> None:
        try:
            with open(self._state_file, "w") as f:
                json.dump(self._lifecycle.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning("Failed to save lifecycle state: %s", e)

    # ------------------------------------------------------------------ Status

    def status(self) -> Dict[str, Any]:
        return {
            "model_name"          : self.model_name,
            "champion_version"   : self.champion_version(),
            "challenger_version" : self.challenger_version(),
            "rollout_stage"      : self.rollout_stage().value if self.rollout_stage() else None,
            "shadow_mode"        : self._shadow_mode.value,
            "canary_elapsed_s"   : (
                time.time() - self._canary_entered_at
                if self._canary_version else None
            ),
            "champion_samples"   : len(self._champion_rets),
            "shadow_samples"     : len(self._shadow_rets),
            "drift_signals"      : self._drift.signals(),
            "drift_sample_counts": self._drift.summary(),
            "rollback_count"     : len(self._rollback_events),
        }
