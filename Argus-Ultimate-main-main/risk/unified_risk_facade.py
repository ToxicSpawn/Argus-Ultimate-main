"""unified_risk_facade.py — single entry point for all Argus risk checks.

Was 0 bytes. Now provides:
    RiskFacade.evaluate(symbol, qty, price, portfolio_equity) -> RiskDecision

This aggregates:
  1. Drawdown halt check  (is drawdown > max_drawdown_pct?)
  2. Daily loss limit     (is today's loss > daily_loss_limit?)
  3. Max position size    (would this trade exceed max_position_pct?)
  4. Kelly fraction sizing (what's the Kelly-optimal qty?)

Usage:
    from risk.unified_risk_facade import RiskFacade, RiskDecision

    facade = RiskFacade(config)
    decision = facade.evaluate("BTC/USDT", qty=0.1, price=65000.0, portfolio_equity=10000.0)
    if not decision.approved:
        logger.warning("Trade blocked: %s", decision.reason)
        return
    execute_order(symbol, decision.approved_qty, price)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    """Result of a risk evaluation."""
    approved: bool
    approved_qty: float
    requested_qty: float
    symbol: str
    reason: str = ""
    checks: Dict[str, bool] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RiskFacade:
    """
    Unified risk facade — runs all risk checks in sequence.

    Config keys read (with sensible defaults):
        max_drawdown_pct:   float  = 15.0   (halt trading above this %)
        daily_loss_limit:   float  = 0.05   (5% of equity per day)
        max_position_pct:   float  = 0.10   (10% of equity per position)
        kelly_fraction:     float  = 0.25   (fractional Kelly multiplier)
        kelly_win_rate:     float  = 0.55   (default win rate for sizing)
        kelly_win_loss:     float  = 1.5    (avg win / avg loss ratio)
    """

    def __init__(self, config: Any = None) -> None:
        self._cfg = config or {}
        self._peak_equity: float = 0.0
        self._day_start_equity: float = 0.0
        self._day_start_ts: float = time.time()
        self._halted: bool = False
        self._halt_reason: str = ""

    def _cfg_get(self, key: str, default: float) -> float:
        if isinstance(self._cfg, dict):
            return float(self._cfg.get(key, default))
        return float(getattr(self._cfg, key, default))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        symbol: str,
        qty: float,
        price: float,
        portfolio_equity: float,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> RiskDecision:
        """Run all risk checks and return a RiskDecision."""
        checks: Dict[str, bool] = {}

        # Update peak equity
        if portfolio_equity > self._peak_equity:
            self._peak_equity = portfolio_equity

        # Reset daily tracking on new calendar day
        self._maybe_reset_day(portfolio_equity)

        # 1. Global halt check
        if self._halted:
            return RiskDecision(
                approved=False,
                approved_qty=0.0,
                requested_qty=qty,
                symbol=symbol,
                reason=f"HALTED: {self._halt_reason}",
                checks={"halt": False},
            )

        # 2. Drawdown check
        dd_pct = self._drawdown_pct(portfolio_equity)
        max_dd = self._cfg_get("max_drawdown_pct", 15.0)
        dd_ok = dd_pct < max_dd
        checks["drawdown"] = dd_ok
        if not dd_ok:
            self._halted = True
            self._halt_reason = f"drawdown {dd_pct:.1f}% >= {max_dd:.1f}%"
            logger.critical(
                "RiskFacade: TRADING HALTED — %s", self._halt_reason
            )
            return RiskDecision(
                approved=False, approved_qty=0.0, requested_qty=qty,
                symbol=symbol, reason=self._halt_reason, checks=checks,
            )

        # 3. Daily loss limit
        daily_loss = self._cfg_get("daily_loss_limit", 0.05)
        daily_loss_ok = self._daily_loss_pct(portfolio_equity) < daily_loss
        checks["daily_loss"] = daily_loss_ok
        if not daily_loss_ok:
            return RiskDecision(
                approved=False, approved_qty=0.0, requested_qty=qty,
                symbol=symbol,
                reason=f"daily loss limit {daily_loss*100:.1f}% reached",
                checks=checks,
            )

        # 4. Position size cap
        max_pos_pct = self._cfg_get("max_position_pct", 0.10)
        max_trade_value = portfolio_equity * max_pos_pct
        max_qty_by_cap = max_trade_value / max(price, 1e-9)
        capped_qty = min(float(qty), max_qty_by_cap)
        checks["position_cap"] = capped_qty > 0

        # 5. Kelly sizing
        wr = win_rate if win_rate is not None else self._cfg_get("kelly_win_rate", 0.55)
        wl = win_loss_ratio if win_loss_ratio is not None else self._cfg_get("kelly_win_loss", 1.5)
        kf = self._cfg_get("kelly_fraction", 0.25)
        kelly_qty = self._kelly_qty(portfolio_equity, price, wr, wl, kf)
        final_qty = min(capped_qty, kelly_qty) if kelly_qty > 0 else capped_qty
        checks["kelly"] = final_qty > 0

        approved = all(checks.values()) and final_qty > 0
        reason = "approved" if approved else "blocked by risk checks"

        return RiskDecision(
            approved=approved,
            approved_qty=round(final_qty, 8),
            requested_qty=qty,
            symbol=symbol,
            reason=reason,
            checks=checks,
        )

    def resume(self) -> None:
        """Manually resume trading after a halt (e.g. after manual review)."""
        logger.warning("RiskFacade: trading RESUMED by manual override")
        self._halted = False
        self._halt_reason = ""

    def reset_day(self, current_equity: float) -> None:
        """Force a daily reset (for testing or end-of-day processes)."""
        self._day_start_equity = current_equity
        self._day_start_ts = time.time()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _drawdown_pct(self, equity: float) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - equity) / self._peak_equity * 100.0)

    def _daily_loss_pct(self, equity: float) -> float:
        if self._day_start_equity <= 0:
            return 0.0
        loss = (self._day_start_equity - equity) / self._day_start_equity
        return max(0.0, loss)

    def _maybe_reset_day(self, equity: float) -> None:
        now = time.time()
        if self._day_start_equity == 0.0:
            self._day_start_equity = equity
            self._day_start_ts = now
            return
        # Reset if >24h since last reset
        if now - self._day_start_ts >= 86400:
            self._day_start_equity = equity
            self._day_start_ts = now

    @staticmethod
    def _kelly_qty(
        equity: float,
        price: float,
        win_rate: float,
        win_loss_ratio: float,
        fraction: float,
    ) -> float:
        """Fractional Kelly position size in units."""
        b = float(win_loss_ratio)
        p = float(win_rate)
        q = 1.0 - p
        kelly_f = (b * p - q) / max(b, 1e-9)
        kelly_f = max(0.0, min(1.0, kelly_f))  # clamp 0-1
        trade_equity = equity * kelly_f * float(fraction)
        return trade_equity / max(price, 1e-9)
