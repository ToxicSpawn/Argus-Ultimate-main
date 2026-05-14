"""
Risk Limits Manager — per-symbol and portfolio-level hard limits.

Enforces:
  - Maximum position size per symbol (USD)
  - Maximum total open exposure (USD)
  - Maximum daily loss (USD)
  - Maximum drawdown from peak equity (%)
  - Maximum consecutive losses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LimitCheck:
    name: str
    passed: bool
    reason: str
    current: float
    limit: float


@dataclass
class LimitsResult:
    allow: bool
    checks: List[LimitCheck]
    size_factor: float = 1.0  # Scaling factor when limits suggest reduction

    @property
    def failing(self) -> List[str]:
        return [c.name for c in self.checks if not c.passed]

    def summary(self) -> str:
        status = "ALLOW" if self.allow else "BLOCK"
        lines = [f"{status} (size_factor={self.size_factor:.2f})"]
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  {mark} {c.name}: {c.current:.2f} vs limit {c.limit:.2f} — {c.reason}")
        return "\n".join(lines)


class RiskLimitsManager:
    """
    Hard limit enforcer for position sizing and portfolio risk.

    Usage::

        mgr = RiskLimitsManager(initial_capital=10_000.0, config={})
        result = mgr.check_order("BTC/USD", "BUY", size_usd=500.0)
        if not result.allow:
            logger.warning("Order blocked: %s", result.failing)
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        config: Optional[Dict] = None,
    ) -> None:
        cfg = config or {}
        self._initial_capital = float(initial_capital)
        self._equity = float(initial_capital)
        self._peak_equity = float(initial_capital)

        # Limits (USD unless noted)
        self.max_position_usd: float = float(
            cfg.get("max_position_usd", initial_capital * 0.20)
        )
        self.max_total_exposure_usd: float = float(
            cfg.get("max_total_exposure_usd", initial_capital * 0.80)
        )
        self.max_daily_loss_usd: float = float(
            cfg.get("max_daily_loss_usd", initial_capital * 0.03)
        )
        self.max_drawdown_pct: float = float(
            cfg.get("max_drawdown_pct", 15.0)
        )
        self.max_consecutive_losses: int = int(
            cfg.get("max_consecutive_losses", 5)
        )

        # State
        self._positions: Dict[str, float] = {}  # symbol → open USD notional
        self._daily_pnl: float = 0.0
        self._reset_date: date = datetime.now(tz=timezone.utc).date()
        self._consecutive_losses: int = 0

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_equity(self, equity_usd: float) -> None:
        """Called after each cycle with current portfolio value."""
        self._equity = max(0.0, float(equity_usd))
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

    def record_fill(self, symbol: str, side: str, size_usd: float, pnl_usd: float = 0.0) -> None:
        """Update internal state after a fill."""
        self._maybe_reset_daily()
        side = str(side).upper()
        size_usd = abs(float(size_usd))
        if side == "BUY":
            self._positions[symbol] = self._positions.get(symbol, 0.0) + size_usd
        elif side == "SELL":
            self._positions[symbol] = max(0.0, self._positions.get(symbol, 0.0) - size_usd)
        self._daily_pnl += float(pnl_usd)
        if pnl_usd < 0:
            self._consecutive_losses += 1
        elif pnl_usd > 0:
            self._consecutive_losses = 0

    def _maybe_reset_daily(self) -> None:
        today = datetime.now(tz=timezone.utc).date()
        if today != self._reset_date:
            self._daily_pnl = 0.0
            self._reset_date = today

    # ------------------------------------------------------------------
    # Order gate
    # ------------------------------------------------------------------

    def check_order(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        *,
        allow_reduced: bool = True,
    ) -> LimitsResult:
        """
        Check whether an order is within all configured limits.

        Args:
            symbol: Trading pair (e.g. "BTC/USD")
            side: "BUY" or "SELL"
            size_usd: Proposed order notional in USD
            allow_reduced: If True, suggest a scaled-down size rather than hard-block

        Returns:
            LimitsResult with allow flag and per-check details.
        """
        self._maybe_reset_daily()
        checks: List[LimitCheck] = []
        size_usd = abs(float(size_usd))

        # 1. Position size
        current_pos = self._positions.get(symbol, 0.0)
        new_pos = current_pos + size_usd if str(side).upper() == "BUY" else current_pos
        checks.append(LimitCheck(
            name="max_position_usd",
            passed=new_pos <= self.max_position_usd,
            reason=f"position would be {new_pos:.0f}",
            current=new_pos,
            limit=self.max_position_usd,
        ))

        # 2. Total exposure
        total_exposure = sum(self._positions.values()) + size_usd
        checks.append(LimitCheck(
            name="max_total_exposure_usd",
            passed=total_exposure <= self.max_total_exposure_usd,
            reason=f"total exposure would be {total_exposure:.0f}",
            current=total_exposure,
            limit=self.max_total_exposure_usd,
        ))

        # 3. Daily loss
        checks.append(LimitCheck(
            name="max_daily_loss_usd",
            passed=self._daily_pnl >= -self.max_daily_loss_usd,
            reason=f"daily P&L {self._daily_pnl:.0f}",
            current=abs(min(self._daily_pnl, 0.0)),
            limit=self.max_daily_loss_usd,
        ))

        # 4. Drawdown
        dd_pct = (
            (self._peak_equity - self._equity) / max(self._peak_equity, 1e-9) * 100.0
        )
        checks.append(LimitCheck(
            name="max_drawdown_pct",
            passed=dd_pct <= self.max_drawdown_pct,
            reason=f"drawdown {dd_pct:.1f}%",
            current=dd_pct,
            limit=self.max_drawdown_pct,
        ))

        # 5. Consecutive losses
        checks.append(LimitCheck(
            name="max_consecutive_losses",
            passed=self._consecutive_losses < self.max_consecutive_losses,
            reason=f"consecutive losses {self._consecutive_losses}",
            current=float(self._consecutive_losses),
            limit=float(self.max_consecutive_losses),
        ))

        all_passed = all(c.passed for c in checks)

        if all_passed:
            return LimitsResult(allow=True, checks=checks, size_factor=1.0)

        if not allow_reduced:
            return LimitsResult(allow=False, checks=checks, size_factor=0.0)

        # Compute a size factor that would make all failing size-based checks pass
        size_factor = 1.0
        if not checks[0].passed:  # position size
            headroom = max(0.0, self.max_position_usd - current_pos)
            size_factor = min(size_factor, headroom / max(size_usd, 1e-9))
        if not checks[1].passed:  # total exposure
            current_total = sum(self._positions.values())
            headroom = max(0.0, self.max_total_exposure_usd - current_total)
            size_factor = min(size_factor, headroom / max(size_usd, 1e-9))

        # Hard blocks (non-size) override reduced sizing
        hard_fail = not checks[2].passed or not checks[3].passed or not checks[4].passed
        if hard_fail:
            logger.warning(
                "RiskLimitsManager hard block for %s %s: %s",
                side, symbol, [c.name for c in checks if not c.passed],
            )
            return LimitsResult(allow=False, checks=checks, size_factor=0.0)

        size_factor = max(0.0, min(1.0, size_factor))
        allow = size_factor > 0.01
        if allow and size_factor < 1.0:
            logger.info(
                "RiskLimitsManager reduced order %s %s by factor %.2f",
                side, symbol, size_factor,
            )
        return LimitsResult(allow=allow, checks=checks, size_factor=size_factor)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict:
        self._maybe_reset_daily()
        dd_pct = (self._peak_equity - self._equity) / max(self._peak_equity, 1e-9) * 100.0
        return {
            "equity_usd": self._equity,
            "peak_equity_usd": self._peak_equity,
            "drawdown_pct": dd_pct,
            "daily_pnl_usd": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "positions_usd": dict(self._positions),
            "total_exposure_usd": sum(self._positions.values()),
        }
