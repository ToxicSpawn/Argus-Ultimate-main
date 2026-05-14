"""
Adaptive Risk Calibrator for ARGUS.

Automatically adjusts risk parameters based on market regime, volatility,
drawdown state, and recent trade performance. Implements a continuous
risk-score (0.0 = maximum risk-off, 1.0 = full risk-on) that drives
position sizing, stop placement, and trade frequency limits.

Key behaviours:
  - CRISIS regime: halve positions, tighten stops, widen spreads
  - LOW_VOL regime: allow larger positions, tighter stops
  - Consecutive losses: reduce size by 20% per loss (max 60% reduction)
  - Drawdown recovery: gradually ramp back to normal over 48 hours
  - Fast regime transitions: more conservative sizing
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_POSITION_PCT = 0.05   # 5% of portfolio per position
_DEFAULT_STOP_LOSS_PCT = 0.02     # 2% stop loss
_DEFAULT_TAKE_PROFIT_PCT = 0.04   # 4% take profit
_DEFAULT_MAX_DAILY_TRADES = 20
_DEFAULT_SPREAD_MULTIPLIER = 1.0

# Regime multipliers for position sizing
_REGIME_POSITION_MULTIPLIER = {
    "CRISIS": 0.50,
    "HIGH_VOL": 0.65,
    "MEAN_REVERTING": 0.90,
    "TRENDING_UP": 1.10,
    "TRENDING_DOWN": 0.75,
    "LOW_VOL": 1.20,
    "BREAKOUT": 0.85,
    "UNKNOWN": 0.80,
}

# Regime-specific stop adjustments (multiplicative)
_REGIME_STOP_MULTIPLIER = {
    "CRISIS": 1.50,       # wider stops in crisis
    "HIGH_VOL": 1.30,
    "MEAN_REVERTING": 0.85,
    "TRENDING_UP": 0.90,
    "TRENDING_DOWN": 1.10,
    "LOW_VOL": 0.75,      # tighter stops in low vol
    "BREAKOUT": 1.20,
    "UNKNOWN": 1.00,
}

# Recovery ramp duration in seconds (48 hours)
_RECOVERY_RAMP_SECONDS = 48 * 3600

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RegimeSnapshot:
    """Single observation of regime state."""
    timestamp: float
    regime: str
    volatility: float
    drawdown: float
    portfolio_value: float


@dataclass
class AdjustedLimits:
    """Risk parameter overrides returned by the calibrator."""
    max_position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_daily_trades: int
    spread_multiplier: float
    risk_score: float
    reason: str


# ---------------------------------------------------------------------------
# AdaptiveRiskCalibrator
# ---------------------------------------------------------------------------


class AdaptiveRiskCalibrator:
    """
    Dynamically adjusts risk parameters based on market conditions.

    Called each cycle with current regime, volatility, drawdown and
    portfolio value. Returns adjusted risk limits that override the
    static configuration.

    Parameters
    ----------
    max_history : int
        Maximum regime/drawdown history entries to retain.
    consecutive_loss_reduction : float
        Position reduction per consecutive loss (0.20 = 20%).
    max_loss_reduction : float
        Maximum cumulative reduction from consecutive losses (0.60 = 60%).
    """

    def __init__(
        self,
        max_history: int = 1000,
        consecutive_loss_reduction: float = 0.20,
        max_loss_reduction: float = 0.60,
    ) -> None:
        self.max_history = max_history
        self.consecutive_loss_reduction = consecutive_loss_reduction
        self.max_loss_reduction = max_loss_reduction

        self._regime_history: Deque[RegimeSnapshot] = deque(maxlen=max_history)
        self._drawdown_history: Deque[Tuple[float, float]] = deque(maxlen=max_history)
        self._consecutive_losses: int = 0
        self._peak_drawdown: float = 0.0
        self._drawdown_recovery_start: Optional[float] = None
        self._last_regime: Optional[str] = None
        self._last_regime_change: float = 0.0
        self._regime_change_count_1h: int = 0

    # ── Update ────────────────────────────────────────────────────────────

    def update(
        self,
        regime: str,
        volatility: float,
        current_drawdown: float,
        portfolio_value: float,
    ) -> None:
        """Called each cycle with current market state."""
        now = time.time()
        snap = RegimeSnapshot(
            timestamp=now,
            regime=regime,
            volatility=float(volatility),
            drawdown=float(current_drawdown),
            portfolio_value=float(portfolio_value),
        )
        self._regime_history.append(snap)
        self._drawdown_history.append((now, float(current_drawdown)))

        # Track regime transitions
        if self._last_regime is not None and regime != self._last_regime:
            self._last_regime_change = now
            self._regime_change_count_1h += 1
            logger.info(
                "AdaptiveRiskCalibrator: regime transition %s -> %s",
                self._last_regime, regime,
            )
        self._last_regime = regime

        # Purge old regime changes from counter (sliding 1h window)
        cutoff = now - 3600
        # Approximate: reduce count if the first regime change was more than 1h ago
        while (
            len(self._regime_history) > 2
            and self._regime_history[0].timestamp < cutoff
        ):
            # Only decrement if there was a regime change at the removed entry
            if len(self._regime_history) > 1:
                old = self._regime_history[0]
                nxt = self._regime_history[1]
                if old.regime != nxt.regime and self._regime_change_count_1h > 0:
                    self._regime_change_count_1h -= 1
            self._regime_history.popleft()

        # Drawdown tracking for recovery ramp
        if current_drawdown > self._peak_drawdown:
            self._peak_drawdown = current_drawdown
            self._drawdown_recovery_start = None  # still deepening
        elif (
            current_drawdown < self._peak_drawdown * 0.5
            and self._drawdown_recovery_start is None
            and self._peak_drawdown > 0.01
        ):
            # Drawdown recovered more than 50% — start recovery ramp
            self._drawdown_recovery_start = now
            logger.info(
                "AdaptiveRiskCalibrator: drawdown recovery started "
                "(peak=%.2f%%, current=%.2f%%)",
                self._peak_drawdown * 100, current_drawdown * 100,
            )

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade PnL for consecutive loss tracking."""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    # ── Adjusted limits ───────────────────────────────────────────────────

    def get_adjusted_limits(self) -> Dict[str, float]:
        """
        Return risk parameter overrides based on current conditions.

        Returns dict with keys:
          max_position_pct, stop_loss_pct, take_profit_pct,
          max_daily_trades, spread_multiplier
        """
        regime = self._last_regime or "UNKNOWN"
        now = time.time()

        # Base values
        max_pos = _DEFAULT_MAX_POSITION_PCT
        stop_loss = _DEFAULT_STOP_LOSS_PCT
        take_profit = _DEFAULT_TAKE_PROFIT_PCT
        max_trades = _DEFAULT_MAX_DAILY_TRADES
        spread_mult = _DEFAULT_SPREAD_MULTIPLIER

        reasons: List[str] = []

        # 1. Regime-based adjustment
        pos_mult = _REGIME_POSITION_MULTIPLIER.get(regime, 0.80)
        stop_mult = _REGIME_STOP_MULTIPLIER.get(regime, 1.0)

        max_pos *= pos_mult
        stop_loss *= stop_mult
        take_profit *= (2.0 - stop_mult)  # inverse: wider stops -> tighter TP

        if regime == "CRISIS":
            spread_mult *= 1.5
            max_trades = max(5, max_trades // 2)
            reasons.append(f"CRISIS regime: pos*{pos_mult:.2f}")
        elif regime == "LOW_VOL":
            reasons.append(f"LOW_VOL: pos*{pos_mult:.2f}")

        # 2. Consecutive loss reduction
        if self._consecutive_losses > 0:
            loss_factor = max(
                1.0 - self.consecutive_loss_reduction * self._consecutive_losses,
                1.0 - self.max_loss_reduction,
            )
            max_pos *= loss_factor
            reasons.append(
                f"consecutive_losses={self._consecutive_losses}: "
                f"pos*{loss_factor:.2f}"
            )

        # 3. Drawdown recovery ramp (gradually return to normal over 48h)
        if self._drawdown_recovery_start is not None:
            elapsed = now - self._drawdown_recovery_start
            ramp = min(elapsed / _RECOVERY_RAMP_SECONDS, 1.0)
            # During recovery, scale between 0.5 and 1.0
            recovery_factor = 0.5 + 0.5 * ramp
            max_pos *= recovery_factor
            if ramp < 1.0:
                reasons.append(f"recovery_ramp={ramp:.2f}: pos*{recovery_factor:.2f}")
            else:
                # Full recovery — reset
                self._drawdown_recovery_start = None
                self._peak_drawdown = 0.0

        # 4. Fast regime transitions -> more conservative
        if self._regime_change_count_1h >= 3:
            transition_penalty = max(0.6, 1.0 - 0.1 * self._regime_change_count_1h)
            max_pos *= transition_penalty
            reasons.append(
                f"fast_transitions={self._regime_change_count_1h}: "
                f"pos*{transition_penalty:.2f}"
            )

        return {
            "max_position_pct": round(max_pos, 6),
            "stop_loss_pct": round(stop_loss, 6),
            "take_profit_pct": round(take_profit, 6),
            "max_daily_trades": max_trades,
            "spread_multiplier": round(spread_mult, 4),
            "reason": "; ".join(reasons) if reasons else "nominal",
        }

    # ── Regime transition detection ───────────────────────────────────────

    def detect_regime_transition(self) -> Optional[str]:
        """Detect if regime just changed. Return description or None."""
        if len(self._regime_history) < 2:
            return None
        prev = self._regime_history[-2]
        curr = self._regime_history[-1]
        if prev.regime != curr.regime:
            return (
                f"Regime transition: {prev.regime} -> {curr.regime} "
                f"(vol: {prev.volatility:.4f} -> {curr.volatility:.4f})"
            )
        return None

    # ── Risk score ────────────────────────────────────────────────────────

    def get_risk_score(self) -> float:
        """
        Return composite risk score: 0.0 (maximum risk-off) to 1.0 (full risk-on).

        Components:
          - Regime contribution (40%)
          - Volatility percentile (20%)
          - Consecutive loss penalty (20%)
          - Drawdown state (20%)
        """
        regime = self._last_regime or "UNKNOWN"

        # Regime score (0-1)
        regime_scores = {
            "CRISIS": 0.1,
            "HIGH_VOL": 0.3,
            "TRENDING_DOWN": 0.4,
            "UNKNOWN": 0.5,
            "MEAN_REVERTING": 0.6,
            "BREAKOUT": 0.65,
            "TRENDING_UP": 0.8,
            "LOW_VOL": 0.9,
        }
        regime_score = regime_scores.get(regime, 0.5)

        # Volatility score: lower vol = higher risk-on
        vol_score = 0.5
        if self._regime_history:
            recent_vol = self._regime_history[-1].volatility
            # Normalize: assume 0.01 (1%) is low, 0.10 (10%) is high
            vol_score = max(0.0, min(1.0, 1.0 - (recent_vol - 0.01) / 0.09))

        # Loss penalty
        loss_penalty = max(
            0.0,
            1.0 - self.consecutive_loss_reduction * self._consecutive_losses,
        )

        # Drawdown score
        dd_score = 1.0
        if self._regime_history:
            dd = self._regime_history[-1].drawdown
            # 0% drawdown = 1.0, 10% drawdown = 0.0
            dd_score = max(0.0, min(1.0, 1.0 - dd / 0.10))

        composite = (
            0.40 * regime_score
            + 0.20 * vol_score
            + 0.20 * loss_penalty
            + 0.20 * dd_score
        )
        return round(max(0.0, min(1.0, composite)), 4)

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, object]:
        """Return current state for dashboard/logging."""
        return {
            "risk_score": self.get_risk_score(),
            "regime": self._last_regime,
            "consecutive_losses": self._consecutive_losses,
            "peak_drawdown": self._peak_drawdown,
            "regime_changes_1h": self._regime_change_count_1h,
            "history_length": len(self._regime_history),
            "limits": self.get_adjusted_limits(),
        }
