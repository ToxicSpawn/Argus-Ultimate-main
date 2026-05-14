"""
adaptive/auto_risk_adjuster.py --- Automatic Risk Level Adjustment.

Continuously assesses system risk posture and outputs a RiskAssessment that
drives global position-sizing multipliers.  Factors include drawdown depth,
volatility regime, win/loss streaks, time of day, day of week, and proximity
to macro events.

Usage::

    adjuster = AutoRiskAdjuster(config=cfg_section)
    assessment = adjuster.assess_risk_level(risk_state)
    adjuster.apply_risk_level(assessment, system=my_system)

Standalone --- no hard imports on the rest of the ARGUS tree at module load.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    """Output of a risk-level evaluation."""

    level: str                    # "conservative" | "normal" | "aggressive"
    position_multiplier: float    # 0.5 -- 1.5
    max_exposure_pct: float       # overall max exposure as fraction
    reason: str                   # human-readable explanation
    factors: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Low-liquidity hours (UTC) -- crypto is 24/7 but volume dips in these windows
# ---------------------------------------------------------------------------

_LOW_LIQUIDITY_HOURS_UTC = frozenset({0, 1, 2, 3, 4, 5})  # 00:00-05:59 UTC (Asia off, US asleep)
_WEEKEND_DAYS = frozenset({5, 6})  # Saturday, Sunday (lower volume)


# ---------------------------------------------------------------------------
# AutoRiskAdjuster
# ---------------------------------------------------------------------------

class AutoRiskAdjuster:
    """Automatic risk-level adjustment engine.

    Parameters
    ----------
    config : dict, optional
        ``auto_risk_adjuster`` section from unified config.
    """

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._enabled: bool = bool(cfg.get("enabled", True))

        # Drawdown thresholds (convex reduction)
        self._dd_start_reduce: float = float(cfg.get("dd_start_reduce_pct", 5.0))
        self._dd_max_reduce: float = float(cfg.get("dd_max_reduce_pct", 25.0))
        self._dd_floor_mult: float = float(cfg.get("dd_floor_mult", 0.3))

        # Volatility thresholds
        self._vol_high: float = float(cfg.get("vol_high", 0.8))
        self._vol_low: float = float(cfg.get("vol_low", 0.2))
        self._vol_high_mult: float = float(cfg.get("vol_high_mult", 0.65))
        self._vol_low_mult: float = float(cfg.get("vol_low_mult", 1.15))

        # Streak parameters
        self._loss_streak_threshold: int = int(cfg.get("loss_streak_threshold", 3))
        self._loss_streak_decay: float = float(cfg.get("loss_streak_decay", 0.10))
        self._win_streak_threshold: int = int(cfg.get("win_streak_threshold", 5))
        self._win_streak_boost: float = float(cfg.get("win_streak_boost", 0.03))

        # Time-based
        self._low_liquidity_mult: float = float(cfg.get("low_liquidity_mult", 0.85))
        self._weekend_mult: float = float(cfg.get("weekend_mult", 0.90))

        # Macro event proximity
        self._macro_event_hours: float = float(cfg.get("macro_event_hours", 2.0))
        self._macro_event_mult: float = float(cfg.get("macro_event_mult", 0.60))

        # Exposure caps by level
        self._exposure_conservative: float = float(cfg.get("exposure_conservative", 0.50))
        self._exposure_normal: float = float(cfg.get("exposure_normal", 0.75))
        self._exposure_aggressive: float = float(cfg.get("exposure_aggressive", 0.90))

        self._history: List[RiskAssessment] = []

        logger.info(
            "AutoRiskAdjuster initialised (dd_start=%.1f%%, vol_hi=%.2f, macro_hrs=%.1f)",
            self._dd_start_reduce, self._vol_high, self._macro_event_hours,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_risk_level(
        self,
        risk_state: Dict[str, Any],
    ) -> RiskAssessment:
        """Assess the current risk level and return position-sizing guidance.

        Parameters
        ----------
        risk_state : dict
            Expected keys (all optional):
            - ``drawdown_pct`` (float): current drawdown from equity peak
            - ``volatility`` (float): annualised realised vol
            - ``win_streak`` (int): current consecutive wins
            - ``loss_streak`` (int): current consecutive losses
            - ``utc_hour`` (int): current UTC hour (0-23), auto-detected if absent
            - ``day_of_week`` (int): 0=Mon ... 6=Sun, auto-detected if absent
            - ``upcoming_events`` (list[dict]): events with ``hours_until`` key
        """
        if not self._enabled:
            return RiskAssessment(
                level="normal",
                position_multiplier=1.0,
                max_exposure_pct=self._exposure_normal,
                reason="AutoRiskAdjuster disabled; using default risk level.",
            )

        mult = 1.0
        factors: Dict[str, float] = {}
        reasons: List[str] = []

        # --- Factor 1: Drawdown (convex reduction) ---
        dd = float(risk_state.get("drawdown_pct", 0.0))
        if dd > self._dd_start_reduce:
            # Convex: accelerating reduction as DD deepens
            dd_frac = min(1.0, (dd - self._dd_start_reduce) / max(self._dd_max_reduce - self._dd_start_reduce, 1.0))
            dd_mult = max(self._dd_floor_mult, 1.0 - dd_frac ** 1.5 * (1.0 - self._dd_floor_mult))
            mult *= dd_mult
            factors["drawdown"] = round(dd_mult, 3)
            reasons.append(f"drawdown {dd:.1f}% -> {dd_mult:.2f}x")

        # --- Factor 2: Volatility ---
        vol = float(risk_state.get("volatility", 0.0))
        if vol > self._vol_high:
            mult *= self._vol_high_mult
            factors["volatility"] = self._vol_high_mult
            reasons.append(f"high vol {vol:.2f} -> {self._vol_high_mult:.2f}x")
        elif vol > 0 and vol < self._vol_low:
            mult *= self._vol_low_mult
            factors["volatility"] = self._vol_low_mult
            reasons.append(f"low vol {vol:.2f} -> {self._vol_low_mult:.2f}x")

        # --- Factor 3: Loss streak ---
        loss_streak = int(risk_state.get("loss_streak", 0))
        if loss_streak >= self._loss_streak_threshold:
            excess = loss_streak - self._loss_streak_threshold + 1
            streak_mult = max(0.4, 1.0 - excess * self._loss_streak_decay)
            mult *= streak_mult
            factors["loss_streak"] = round(streak_mult, 3)
            reasons.append(f"loss streak {loss_streak} -> {streak_mult:.2f}x")

        # --- Factor 4: Win streak ---
        win_streak = int(risk_state.get("win_streak", 0))
        if win_streak >= self._win_streak_threshold:
            excess = win_streak - self._win_streak_threshold + 1
            streak_mult = min(1.15, 1.0 + excess * self._win_streak_boost)
            mult *= streak_mult
            factors["win_streak"] = round(streak_mult, 3)
            reasons.append(f"win streak {win_streak} -> {streak_mult:.2f}x")

        # --- Factor 5: Time of day ---
        now_utc = datetime.now(timezone.utc)
        utc_hour = int(risk_state.get("utc_hour", now_utc.hour))
        if utc_hour in _LOW_LIQUIDITY_HOURS_UTC:
            mult *= self._low_liquidity_mult
            factors["low_liquidity_hour"] = self._low_liquidity_mult
            reasons.append(f"low-liquidity hour (UTC {utc_hour:02d}) -> {self._low_liquidity_mult:.2f}x")

        # --- Factor 6: Day of week ---
        dow = int(risk_state.get("day_of_week", now_utc.weekday()))
        if dow in _WEEKEND_DAYS:
            mult *= self._weekend_mult
            factors["weekend"] = self._weekend_mult
            reasons.append(f"weekend (day={dow}) -> {self._weekend_mult:.2f}x")

        # --- Factor 7: Upcoming macro events ---
        upcoming = risk_state.get("upcoming_events") or []
        for evt in upcoming:
            hours_until = float(evt.get("hours_until", 999))
            if hours_until <= self._macro_event_hours:
                mult *= self._macro_event_mult
                factors["macro_event"] = self._macro_event_mult
                event_name = str(evt.get("name", "macro event"))
                reasons.append(
                    f"macro event '{event_name}' in {hours_until:.1f}h -> {self._macro_event_mult:.2f}x"
                )
                break  # one event reduction is enough

        # Clamp
        mult = max(0.5, min(1.5, mult))

        # Classify level
        if mult < 0.75:
            level = "conservative"
            max_exp = self._exposure_conservative
        elif mult > 1.1:
            level = "aggressive"
            max_exp = self._exposure_aggressive
        else:
            level = "normal"
            max_exp = self._exposure_normal

        reason_str = "; ".join(reasons) if reasons else "No risk adjustments needed."

        assessment = RiskAssessment(
            level=level,
            position_multiplier=round(mult, 3),
            max_exposure_pct=max_exp,
            reason=reason_str,
            factors=factors,
        )

        self._history.append(assessment)
        if len(self._history) > 500:
            self._history = self._history[-250:]

        logger.info(
            "RiskAssessment: level=%s, mult=%.3f, exposure=%.0f%% | %s",
            level, mult, max_exp * 100, reason_str,
        )
        return assessment

    def apply_risk_level(
        self,
        assessment: RiskAssessment,
        *,
        system: Any = None,
    ) -> bool:
        """Apply a risk assessment to the trading system.

        Parameters
        ----------
        assessment : RiskAssessment
            The assessment to apply.
        system : object, optional
            Trading system with ``set_position_multiplier(float)`` and/or
            ``set_max_exposure(float)`` methods.

        Returns
        -------
        bool
            True if applied successfully.
        """
        if system is None:
            logger.warning("apply_risk_level: no system provided; logging only.")
            return False

        try:
            if hasattr(system, "set_position_multiplier"):
                system.set_position_multiplier(assessment.position_multiplier)
            if hasattr(system, "set_max_exposure"):
                system.set_max_exposure(assessment.max_exposure_pct)

            logger.info(
                "Applied risk level: %s (mult=%.3f, exp=%.0f%%)",
                assessment.level, assessment.position_multiplier, assessment.max_exposure_pct * 100,
            )
            return True
        except Exception:
            logger.exception("Failed to apply risk level")
            return False

    @property
    def history(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._history]
