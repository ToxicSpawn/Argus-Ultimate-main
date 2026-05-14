"""Canary deploy strategy — Push 100 StatePersist.

Implements a traffic-splitting canary rollout for Argus strategy
versions.  Traffic is routed to the canary cohort based on a
configurable weight that ramps automatically (or manually) once
health gates pass.

Integrates with:
  • Prometheus metrics  — canary vs stable order/PnL counters
  • Redis               — persists canary state across restarts
  • Regime detector     — pauses ramp during CRISIS regime

Typical lifecycle::

    PENDING → RAMPING → FULL → (ROLLED_BACK)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class CanaryPhase(str, Enum):
    PENDING      = "PENDING"      # not yet started
    RAMPING      = "RAMPING"      # weight increasing
    FULL         = "FULL"         # 100% traffic on canary
    ROLLED_BACK  = "ROLLED_BACK"  # reverted to stable
    PAUSED       = "PAUSED"       # temporarily halted


@dataclass
class CanaryConfig:
    strategy_name:     str
    canary_version:    str
    stable_version:    str
    initial_weight:    float = 0.05    # fraction routed to canary at start
    ramp_step:         float = 0.10    # weight increment per ramp tick
    ramp_interval_s:   float = 300.0  # seconds between auto-ramp ticks
    max_weight:        float = 1.0    # ceiling (1.0 = full canary)
    # Health gates — rollback if breached
    max_error_rate:    float = 0.02   # 2%
    max_latency_p99_ms: float = 500.0
    min_pnl_ratio:     float = 0.90   # canary PnL / stable PnL must be >= this
    auto_ramp:         bool  = True
    pause_on_crisis:   bool  = True


@dataclass
class CanaryState:
    config:         CanaryConfig
    phase:          CanaryPhase  = CanaryPhase.PENDING
    current_weight: float        = 0.0
    started_at:     float        = field(default_factory=time.time)
    last_ramp_at:   float        = 0.0
    rollback_reason: str         = ""
    ticks:          int          = 0


class CanaryController:
    """Controls traffic weight for a single canary deployment."""

    def __init__(
        self,
        config: CanaryConfig,
        state_persist: Optional[Any] = None,   # core.state_persist.StatePersist
    ) -> None:
        self._cfg   = config
        self._sp    = state_persist
        self._state = CanaryState(config=config)
        self._restore()

    # ─── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin canary ramp from initial_weight."""
        if self._state.phase not in (CanaryPhase.PENDING, CanaryPhase.PAUSED):
            log.warning("Canary already in phase %s — ignoring start()", self._state.phase)
            return
        self._state.phase          = CanaryPhase.RAMPING
        self._state.current_weight = self._cfg.initial_weight
        self._state.last_ramp_at   = time.time()
        self._save()
        log.info("Canary started: %s → %.0f%% weight",
                 self._cfg.canary_version, self._state.current_weight * 100)

    def pause(self, reason: str = "") -> None:
        if self._state.phase == CanaryPhase.RAMPING:
            self._state.phase = CanaryPhase.PAUSED
            self._save()
            log.info("Canary paused: %s", reason)

    def resume(self) -> None:
        if self._state.phase == CanaryPhase.PAUSED:
            self._state.phase = CanaryPhase.RAMPING
            self._save()
            log.info("Canary resumed")

    def rollback(self, reason: str = "manual") -> None:
        self._state.phase           = CanaryPhase.ROLLED_BACK
        self._state.current_weight  = 0.0
        self._state.rollback_reason = reason
        self._save()
        log.warning("Canary ROLLED BACK (%s): weight → 0", reason)

    def tick(
        self,
        *,
        error_rate:     float = 0.0,
        latency_p99_ms: float = 0.0,
        pnl_ratio:      float = 1.0,
        current_regime: str   = "",
    ) -> CanaryPhase:
        """Evaluate health gates and auto-ramp if enabled.  Call every ~interval_s."""
        if self._state.phase not in (CanaryPhase.RAMPING,):
            return self._state.phase

        # Crisis pause gate
        if self._cfg.pause_on_crisis and current_regime == "CRISIS":
            self.pause("CRISIS regime")
            return self._state.phase

        # Health gate checks → rollback
        if error_rate > self._cfg.max_error_rate:
            self.rollback(f"error_rate={error_rate:.3f} > {self._cfg.max_error_rate}")
            return self._state.phase
        if latency_p99_ms > self._cfg.max_latency_p99_ms:
            self.rollback(f"p99={latency_p99_ms:.1f}ms > {self._cfg.max_latency_p99_ms}ms")
            return self._state.phase
        if pnl_ratio < self._cfg.min_pnl_ratio:
            self.rollback(f"pnl_ratio={pnl_ratio:.3f} < {self._cfg.min_pnl_ratio}")
            return self._state.phase

        # Auto-ramp
        if self._cfg.auto_ramp:
            now = time.time()
            if now - self._state.last_ramp_at >= self._cfg.ramp_interval_s:
                self._ramp()

        return self._state.phase

    @property
    def weight(self) -> float:
        """Current canary traffic weight [0, 1]."""
        return self._state.current_weight

    @property
    def phase(self) -> CanaryPhase:
        return self._state.phase

    def route_to_canary(self, token: float) -> bool:
        """Return True if this request (token ∈ [0,1)) should hit the canary."""
        return (
            self._state.phase in (CanaryPhase.RAMPING, CanaryPhase.FULL)
            and token < self._state.current_weight
        )

    # ─── Internal ────────────────────────────────────────────────────────────

    def _ramp(self) -> None:
        new_weight = min(
            self._state.current_weight + self._cfg.ramp_step,
            self._cfg.max_weight,
        )
        self._state.current_weight = new_weight
        self._state.last_ramp_at   = time.time()
        self._state.ticks         += 1
        if new_weight >= self._cfg.max_weight:
            self._state.phase = CanaryPhase.FULL
            log.info("Canary FULL: %s at 100%%", self._cfg.canary_version)
        else:
            log.info("Canary ramp tick %d: %.0f%%", self._state.ticks, new_weight * 100)
        self._save()

    def _save(self) -> None:
        if self._sp is None:
            return
        try:
            key_data = {
                "phase":          self._state.phase.value,
                "current_weight": self._state.current_weight,
                "started_at":     self._state.started_at,
                "last_ramp_at":   self._state.last_ramp_at,
                "rollback_reason": self._state.rollback_reason,
                "ticks":          self._state.ticks,
                "config":         asdict(self._cfg),
            }
            self._sp._set(  # type: ignore[attr-defined]
                self._sp._key("canary", self._cfg.strategy_name),  # type: ignore[attr-defined]
                key_data,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Canary _save failed: %s", exc)

    def _restore(self) -> None:
        if self._sp is None:
            return
        try:
            data = self._sp._get(  # type: ignore[attr-defined]
                self._sp._key("canary", self._cfg.strategy_name)  # type: ignore[attr-defined]
            )
            if not data:
                return
            self._state.phase           = CanaryPhase(data["phase"])
            self._state.current_weight  = data["current_weight"]
            self._state.started_at      = data["started_at"]
            self._state.last_ramp_at    = data["last_ramp_at"]
            self._state.rollback_reason = data.get("rollback_reason", "")
            self._state.ticks           = data.get("ticks", 0)
            log.info("Canary restored: %s phase=%s weight=%.0f%%",
                     self._cfg.strategy_name, self._state.phase.value,
                     self._state.current_weight * 100)
        except Exception as exc:  # noqa: BLE001
            log.warning("Canary _restore failed: %s — starting fresh", exc)
