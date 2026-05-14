"""
Capital Migration — staged paper → live capital deployment workflow.

Stages:
  1. PAPER      — virtual capital only, no real trades
  2. MICRO      — $100 AUD real capital, all limits at 10%
  3. SEED       — $500 AUD, limits at 25%
  4. LIVE       — full configured capital, all limits active

Each stage requires:
  - Minimum days of operation
  - Sharpe ratio above threshold
  - Maximum drawdown below threshold
  - No circuit breaker events in last N days
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    PAPER = "paper"
    MICRO = "micro"
    SEED = "seed"
    LIVE = "live"


STAGE_ORDER = [Stage.PAPER, Stage.MICRO, Stage.SEED, Stage.LIVE]


@dataclass
class StageRequirements:
    stage: Stage
    min_days: int              # Minimum days at this stage before advancing
    min_sharpe: float          # Annualised Sharpe ratio
    max_drawdown_pct: float    # Maximum drawdown (%)
    max_circuit_breaks: int    # Circuit breaker events allowed in last 7d
    capital_aud: float         # Capital allocated at this stage
    position_limit_pct: float  # % of full configured limits


# Stage advancement requirements
STAGE_REQUIREMENTS: Dict[Stage, StageRequirements] = {
    Stage.PAPER: StageRequirements(
        stage=Stage.PAPER,
        min_days=3,
        min_sharpe=0.0,   # No Sharpe requirement for paper
        max_drawdown_pct=50.0,
        max_circuit_breaks=999,
        capital_aud=0.0,
        position_limit_pct=100.0,
    ),
    Stage.MICRO: StageRequirements(
        stage=Stage.MICRO,
        min_days=7,
        min_sharpe=0.3,
        max_drawdown_pct=20.0,
        max_circuit_breaks=2,
        capital_aud=100.0,
        position_limit_pct=10.0,
    ),
    Stage.SEED: StageRequirements(
        stage=Stage.SEED,
        min_days=14,
        min_sharpe=0.5,
        max_drawdown_pct=15.0,
        max_circuit_breaks=1,
        capital_aud=500.0,
        position_limit_pct=25.0,
    ),
    Stage.LIVE: StageRequirements(
        stage=Stage.LIVE,
        min_days=30,
        min_sharpe=0.8,
        max_drawdown_pct=12.0,
        max_circuit_breaks=0,
        capital_aud=1000.0,
        position_limit_pct=100.0,
    ),
}


@dataclass
class MigrationCheck:
    requirement: str
    required: str
    actual: str
    passed: bool


@dataclass
class MigrationAssessment:
    current_stage: Stage
    next_stage: Optional[Stage]
    can_advance: bool
    checks: List[MigrationCheck]
    recommendation: str
    assessed_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def summary(self) -> str:
        lines = [
            f"Stage: {self.current_stage.value} → {self.next_stage.value if self.next_stage else 'COMPLETE'}",
            f"Can advance: {'YES' if self.can_advance else 'NO'}",
        ]
        for c in self.checks:
            status = "✓" if c.passed else "✗"
            lines.append(f"  {status} {c.requirement}: need {c.required}, got {c.actual}")
        lines.append(f"Recommendation: {self.recommendation}")
        return "\n".join(lines)


@dataclass
class PerformanceSnapshot:
    """Current system performance metrics for migration assessment."""
    days_at_stage: int
    sharpe_annualised: float
    max_drawdown_pct: float
    circuit_breaks_7d: int
    total_trades: int
    current_stage: Stage


class CapitalMigration:
    """
    Staged capital deployment manager.

    Usage::

        migration = CapitalMigration()
        perf = PerformanceSnapshot(
            days_at_stage=5,
            sharpe_annualised=0.6,
            max_drawdown_pct=8.0,
            circuit_breaks_7d=0,
            total_trades=42,
            current_stage=Stage.MICRO,
        )
        assessment = migration.assess(perf)
        if assessment.can_advance:
            # Prompt user to confirm stage advancement
    """

    def __init__(self) -> None:
        self._stage_start: Dict[Stage, datetime] = {
            Stage.PAPER: datetime.now(tz=timezone.utc)
        }
        self._current_stage = Stage.PAPER
        self._history: List[Dict] = []

    @property
    def current_stage(self) -> Stage:
        return self._current_stage

    @property
    def current_capital_aud(self) -> float:
        return STAGE_REQUIREMENTS[self._current_stage].capital_aud

    @property
    def current_position_limit_pct(self) -> float:
        return STAGE_REQUIREMENTS[self._current_stage].position_limit_pct

    def assess(self, perf: PerformanceSnapshot) -> MigrationAssessment:
        """Evaluate whether conditions are met to advance to next stage."""
        current = self._current_stage
        idx = STAGE_ORDER.index(current)
        next_stage = STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None

        if next_stage is None:
            return MigrationAssessment(
                current_stage=current,
                next_stage=None,
                can_advance=False,
                checks=[],
                recommendation="Already at maximum stage (LIVE).",
            )

        req = STAGE_REQUIREMENTS[next_stage]
        checks: List[MigrationCheck] = []

        # Check 1: Days at current stage
        checks.append(MigrationCheck(
            requirement="Days at current stage",
            required=f"≥ {req.min_days}d",
            actual=f"{perf.days_at_stage}d",
            passed=perf.days_at_stage >= req.min_days,
        ))

        # Check 2: Sharpe ratio
        checks.append(MigrationCheck(
            requirement="Annualised Sharpe",
            required=f"≥ {req.min_sharpe:.1f}",
            actual=f"{perf.sharpe_annualised:.2f}",
            passed=perf.sharpe_annualised >= req.min_sharpe,
        ))

        # Check 3: Max drawdown
        checks.append(MigrationCheck(
            requirement="Max drawdown",
            required=f"≤ {req.max_drawdown_pct:.0f}%",
            actual=f"{perf.max_drawdown_pct:.1f}%",
            passed=perf.max_drawdown_pct <= req.max_drawdown_pct,
        ))

        # Check 4: Circuit breaker events
        checks.append(MigrationCheck(
            requirement="Circuit breaks (7d)",
            required=f"≤ {req.max_circuit_breaks}",
            actual=str(perf.circuit_breaks_7d),
            passed=perf.circuit_breaks_7d <= req.max_circuit_breaks,
        ))

        # Check 5: Minimum trade count
        min_trades = 10
        checks.append(MigrationCheck(
            requirement="Total trades",
            required=f"≥ {min_trades}",
            actual=str(perf.total_trades),
            passed=perf.total_trades >= min_trades,
        ))

        can_advance = all(c.passed for c in checks)
        failed = [c.requirement for c in checks if not c.passed]

        if can_advance:
            rec = (
                f"All conditions met. Ready to advance to {next_stage.value.upper()} "
                f"with AUD {req.capital_aud:.0f} capital "
                f"({req.position_limit_pct:.0f}% position limits). "
                "Confirm with: migration.advance()"
            )
        else:
            rec = f"Cannot advance. Failing: {', '.join(failed)}"

        return MigrationAssessment(
            current_stage=current,
            next_stage=next_stage,
            can_advance=can_advance,
            checks=checks,
            recommendation=rec,
        )

    def advance(self, confirmed: bool = False) -> bool:
        """
        Advance to the next stage. Requires explicit confirmation.
        Returns True if stage was advanced.
        """
        if not confirmed:
            logger.warning("Stage advancement requires confirmed=True")
            return False
        idx = STAGE_ORDER.index(self._current_stage)
        if idx + 1 >= len(STAGE_ORDER):
            logger.info("Already at maximum stage")
            return False
        old_stage = self._current_stage
        self._current_stage = STAGE_ORDER[idx + 1]
        self._stage_start[self._current_stage] = datetime.now(tz=timezone.utc)
        self._history.append({
            "from": old_stage.value,
            "to": self._current_stage.value,
            "at": datetime.now(tz=timezone.utc).isoformat(),
        })
        logger.info(
            "CAPITAL MIGRATION: %s → %s (AUD %.0f, limits %.0f%%)",
            old_stage.value, self._current_stage.value,
            self.current_capital_aud, self.current_position_limit_pct,
        )
        return True

    def rollback(self, reason: str, confirmed: bool = False) -> bool:
        """Roll back to previous stage on poor performance."""
        if not confirmed:
            logger.warning("Rollback requires confirmed=True")
            return False
        idx = STAGE_ORDER.index(self._current_stage)
        if idx == 0:
            logger.info("Already at minimum stage (PAPER)")
            return False
        old_stage = self._current_stage
        self._current_stage = STAGE_ORDER[idx - 1]
        self._history.append({
            "from": old_stage.value,
            "to": self._current_stage.value,
            "at": datetime.now(tz=timezone.utc).isoformat(),
            "reason": reason,
        })
        logger.warning(
            "CAPITAL ROLLBACK: %s → %s. Reason: %s",
            old_stage.value, self._current_stage.value, reason,
        )
        return True

    def history(self) -> List[Dict]:
        return list(self._history)

    def stage_requirements(self, stage: Optional[Stage] = None) -> StageRequirements:
        s = stage or self._current_stage
        return STAGE_REQUIREMENTS[s]

    # ------------------------------------------------------------------
    # Stage transition validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_stage_transition(current: Stage, target: Stage) -> bool:
        """Validate that *target* is the immediate next stage after *current*.

        Only sequential transitions are allowed:
        PAPER -> MICRO -> SEED -> LIVE.  Skipping stages raises ValueError.

        Returns True when the transition is valid.
        """
        current_idx = STAGE_ORDER.index(current)
        target_idx = STAGE_ORDER.index(target)

        if target_idx == current_idx + 1:
            return True

        if target_idx <= current_idx:
            raise ValueError(
                f"Cannot transition from {current.value} to {target.value}: "
                f"target must be a later stage"
            )

        # Skipping stages
        skipped = [s.value for s in STAGE_ORDER[current_idx + 1: target_idx]]
        raise ValueError(
            f"Cannot skip stages: {current.value} -> {target.value}. "
            f"Must pass through: {', '.join(skipped)}"
        )

    def _check_stage_prerequisites(self, stage: Stage) -> tuple:
        """Check whether prerequisites are met for entering *stage*.

        Prerequisites:
          - MICRO: >= 7 days in PAPER + positive cumulative P&L
          - SEED / MEDIUM: >= 14 days in previous stage (MICRO)
          - LIVE: >= 30 days in previous stage (SEED)

        Returns
        -------
        (ok, reason) : (bool, str)
        """
        if stage == Stage.PAPER:
            return True, "PAPER has no prerequisites"

        prev_idx = STAGE_ORDER.index(stage) - 1
        prev_stage = STAGE_ORDER[prev_idx]

        start_time = self._stage_start.get(prev_stage)
        if start_time is None:
            return False, f"No recorded start time for {prev_stage.value}"

        days_in_prev = (datetime.now(tz=timezone.utc) - start_time).total_seconds() / 86400.0

        if stage == Stage.MICRO:
            if days_in_prev < 7:
                return False, (
                    f"MICRO requires >= 7 days in PAPER, "
                    f"only {days_in_prev:.1f} days elapsed"
                )
            # Positive P&L check — caller must set _paper_pnl
            paper_pnl = getattr(self, "_paper_pnl", None)
            if paper_pnl is not None and paper_pnl <= 0:
                return False, (
                    f"MICRO requires positive P&L in PAPER, got {paper_pnl:.2f}"
                )
            return True, "MICRO prerequisites met"

        if stage == Stage.SEED:
            required_days = 14
            if days_in_prev < required_days:
                return False, (
                    f"SEED requires >= {required_days} days in MICRO, "
                    f"only {days_in_prev:.1f} days elapsed"
                )
            return True, "SEED prerequisites met"

        if stage == Stage.LIVE:
            required_days = 30
            if days_in_prev < required_days:
                return False, (
                    f"LIVE requires >= {required_days} days in SEED, "
                    f"only {days_in_prev:.1f} days elapsed"
                )
            return True, "LIVE prerequisites met"

        return False, f"Unknown stage: {stage.value}"
