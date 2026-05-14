#!/usr/bin/env python3
"""
Advanced Risk Manager - S+ Tier
Comprehensive risk management with circuit breakers and dynamic position sizing.

Batch-3 additions
-----------------
* Per-symbol drawdown tracking (peak/trough per asset, not just portfolio-wide).
* Regime-gated maximum exposure: exposure cap scales with the current regime
  confidence passed in via update_regime().
* KellySizer confidence-interval integration: calculate_position_size() now
  accepts an optional kelly_ci tuple (lower, upper) to widen/narrow the
  Kelly fraction based on estimation uncertainty.
* check() convenience method so AdvancedRiskManager can be used directly as
  an ExecutionEngine risk_facade without an adapter.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration"""

    max_drawdown: float = 0.10          # 10% max portfolio drawdown
    max_daily_loss: float = 0.05        # 5% max daily loss
    max_position_size: float = 0.20     # 20% max position size (per symbol)
    max_correlation: float = 0.80       # Max correlation between positions
    circuit_breaker_threshold: float = 0.15  # 15% portfolio DD triggers CB

    # Per-symbol drawdown: position is reduced when asset-level DD exceeds this
    max_symbol_drawdown: float = 0.12   # 12% per-symbol drawdown limit

    # Regime-gate: base exposure multiplier bounds
    regime_min_exposure: float = 0.30   # floor multiplier in low-confidence regime
    regime_max_exposure: float = 1.00   # ceiling (default full exposure)


class AdvancedRiskManager:
    """
    Advanced Risk Manager - S+ Tier
    Comprehensive risk management with circuit breakers and dynamic position sizing.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.portfolio_value = 100_000.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.daily_pnl = 0.0
        self.peak_value = self.portfolio_value
        self.circuit_breaker_triggered = False

        # Risk metrics
        self.value_at_risk = 0.0
        self.expected_shortfall = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0

        # Historical data
        self.pnl_history: List[float] = []
        self.drawdown_history: List[float] = []

        # --- Batch-3: per-symbol drawdown tracking ---
        # symbol -> {peak_value, current_value, max_dd}
        self._symbol_peaks: Dict[str, float] = {}
        self._symbol_drawdowns: Dict[str, float] = {}

        # --- Batch-3: regime-gate state ---
        # Regime confidence in [0, 1]; 1.0 = full conviction, 0.0 = no conviction
        self._regime_confidence: float = 1.0
        self._current_regime: str = "unknown"
        self._regime_exposure_mult: float = 1.0

        logger.info("Advanced Risk Manager initialised (batch-3)")

    # ------------------------------------------------------------------
    # Regime gate (Batch-3)
    # ------------------------------------------------------------------

    def update_regime(self, regime: str, confidence: float) -> None:
        """
        Inform the risk manager of the current market regime.

        Parameters
        ----------
        regime : str
            Regime label, e.g. "trending", "ranging", "volatile".
        confidence : float
            Detector confidence in [0, 1].  Exposure cap scales linearly
            between regime_min_exposure and regime_max_exposure.
        """
        self._current_regime = regime
        self._regime_confidence = max(0.0, min(1.0, confidence))
        lo = self.limits.regime_min_exposure
        hi = self.limits.regime_max_exposure
        self._regime_exposure_mult = lo + (hi - lo) * self._regime_confidence
        logger.info(
            "Regime updated: %s conf=%.2f -> exposure_mult=%.2f",
            regime, self._regime_confidence, self._regime_exposure_mult,
        )

    # ------------------------------------------------------------------
    # check() — risk_facade interface (Batch-3)
    # ------------------------------------------------------------------

    def check(self, request: Any) -> bool:
        """
        Pre-execution risk gate compatible with ExecutionEngine.risk_facade.

        Blocks if:
        - Circuit breaker is triggered.
        - Global drawdown exceeds limit.
        - Daily loss exceeds limit.
        - Prospective position would breach per-symbol drawdown limit.
        - Regime-gated exposure cap would be exceeded.

        Parameters
        ----------
        request : ExecutionRequest (or any object with .symbol, .quantity, .price)
        """
        if self.circuit_breaker_triggered:
            logger.warning("check(): blocked — circuit breaker active")
            return False

        if self.max_drawdown > self.limits.max_drawdown:
            logger.warning(
                "check(): blocked — portfolio drawdown %.2f%% > limit %.2f%%",
                self.max_drawdown * 100, self.limits.max_drawdown * 100,
            )
            return False

        pv = self.portfolio_value or 1.0
        if -self.daily_pnl / pv > self.limits.max_daily_loss:
            logger.warning(
                "check(): blocked — daily loss %.2f%% > limit %.2f%%",
                (-self.daily_pnl / pv) * 100,
                self.limits.max_daily_loss * 100,
            )
            return False

        # Per-symbol drawdown gate
        sym = getattr(request, "symbol", None)
        if sym and sym in self._symbol_drawdowns:
            sym_dd = self._symbol_drawdowns[sym]
            if sym_dd > self.limits.max_symbol_drawdown:
                logger.warning(
                    "check(): blocked — symbol %s drawdown %.2f%% > limit %.2f%%",
                    sym, sym_dd * 100, self.limits.max_symbol_drawdown * 100,
                )
                return False

        # Regime-gate: check prospective exposure
        price = getattr(request, "price", None) or 0.0
        qty = getattr(request, "quantity", 0.0)
        prospective_value = qty * price
        current_exposure = sum(
            abs(p.get("value", 0)) for p in self.positions.values()
        )
        max_allowed_exposure = pv * self._regime_exposure_mult
        if current_exposure + prospective_value > max_allowed_exposure:
            logger.warning(
                "check(): blocked — exposure %.2f + %.2f > regime-gated cap %.2f (mult=%.2f regime=%s)",
                current_exposure, prospective_value,
                max_allowed_exposure, self._regime_exposure_mult, self._current_regime,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Portfolio value & per-symbol tracking (Batch-3 enhanced)
    # ------------------------------------------------------------------

    def update_portfolio_value(self, new_value: float) -> None:
        """
        Update portfolio value and recalculate risk metrics.

        Args:
            new_value: New portfolio value
        """
        old_value = self.portfolio_value
        self.portfolio_value = new_value

        if new_value > self.peak_value:
            self.peak_value = new_value

        self.daily_pnl = new_value - old_value
        current_drawdown = (
            (self.peak_value - new_value) / self.peak_value
            if self.peak_value > 0 else 0.0
        )
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        self.pnl_history.append(self.daily_pnl)
        self.drawdown_history.append(current_drawdown)

        max_history = 252
        self.pnl_history = self.pnl_history[-max_history:]
        self.drawdown_history = self.drawdown_history[-max_history:]

        self._update_risk_metrics()

    def update_symbol_price(self, symbol: str, current_price: float) -> None:
        """
        Track per-symbol peak and drawdown.

        Call this every time a mark-to-market price update arrives for a
        symbol.  The drawdown is stored and consulted by check().
        """
        if symbol not in self._symbol_peaks:
            self._symbol_peaks[symbol] = current_price
        else:
            if current_price > self._symbol_peaks[symbol]:
                self._symbol_peaks[symbol] = current_price

        peak = self._symbol_peaks[symbol]
        dd = (peak - current_price) / peak if peak > 0 else 0.0
        self._symbol_drawdowns[symbol] = dd

        if dd > self.limits.max_symbol_drawdown:
            logger.warning(
                "Symbol drawdown alert: %s DD=%.2f%% (limit %.2f%%)",
                symbol, dd * 100, self.limits.max_symbol_drawdown * 100,
            )

    def get_symbol_drawdown(self, symbol: str) -> float:
        """Return the current per-symbol drawdown fraction (0.0 if unknown)."""
        return self._symbol_drawdowns.get(symbol, 0.0)

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def _update_risk_metrics(self) -> None:
        if len(self.pnl_history) < 30:
            return
        returns = np.array(self.pnl_history)
        self.value_at_risk = -np.percentile(returns, 5)
        tail_returns = returns[returns <= -self.value_at_risk]
        self.expected_shortfall = (
            -np.mean(tail_returns) if len(tail_returns) > 0 else 0
        )
        if returns.std() > 0:
            self.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    def check_risk_limits(self) -> Dict[str, Any]:
        """Check if any risk limits are violated."""
        results = {
            "drawdown_limit_breached": self.max_drawdown > self.limits.max_drawdown,
            "daily_loss_limit_breached": (
                -self.daily_pnl / max(self.portfolio_value, 1)
                > self.limits.max_daily_loss
            ),
            "circuit_breaker_triggered": (
                self.max_drawdown > self.limits.circuit_breaker_threshold
            ),
            "position_limits_ok": self._check_position_limits(),
            "correlation_limits_ok": self._check_correlation_limits(),
            "regime_exposure_mult": self._regime_exposure_mult,
            "current_regime": self._current_regime,
        }
        self.circuit_breaker_triggered = results["circuit_breaker_triggered"]
        return results

    def _check_position_limits(self) -> bool:
        for position in self.positions.values():
            pv = position.get("value", 0)
            if pv / max(self.portfolio_value, 1) > self.limits.max_position_size:
                return False
        return True

    def _check_correlation_limits(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Position sizing — Kelly CI integration (Batch-3)
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        asset: str,
        entry_price: float,
        stop_loss: float,
        volatility: float,
        confidence: float = 0.5,
        kelly_ci: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Calculate optimal position size based on risk management rules.

        Parameters
        ----------
        asset : str
        entry_price : float
        stop_loss : float
        volatility : float
        confidence : float
            Point estimate of strategy win-rate (0-1).
        kelly_ci : tuple (lower, upper), optional
            95 % confidence interval for the Kelly fraction as produced by
            KellySizer.  When supplied, the fraction is shrunk towards the
            lower bound to penalise estimation uncertainty::

                effective_kelly = lower + (fraction - lower) * shrinkage

            where shrinkage = (upper - lower) is clipped to [0, 1] so wider
            CIs produce more conservative sizing.
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0.0

        win_rate = confidence
        avg_win_loss_ratio = 2.0

        if win_rate <= 0:
            return 0.0

        kelly_fraction = (
            (win_rate * (avg_win_loss_ratio + 1) - 1) / avg_win_loss_ratio
        )
        kelly_fraction = max(0.0, min(kelly_fraction * 0.5, 0.10))

        # --- Kelly CI shrinkage (Batch-3) ---
        if kelly_ci is not None:
            ci_lo, ci_hi = kelly_ci
            ci_width = max(0.0, ci_hi - ci_lo)
            # Shrinkage factor: narrow CI -> stay near point estimate;
            # wide CI -> shrink aggressively towards lower bound.
            shrinkage = max(0.0, 1.0 - ci_width)   # [0, 1]
            kelly_fraction = ci_lo + (kelly_fraction - ci_lo) * shrinkage
            kelly_fraction = max(0.0, kelly_fraction)
            logger.debug(
                "Kelly CI shrinkage: ci=(%.4f, %.4f) width=%.4f "
                "shrinkage=%.3f -> kelly_fraction=%.4f",
                ci_lo, ci_hi, ci_width, shrinkage, kelly_fraction,
            )

        base_risk = self.portfolio_value * 0.01
        base_size = base_risk / risk_per_share
        position_size = base_size * kelly_fraction * 10

        vol_adjustment = 1.0 / (1.0 + volatility)
        position_size *= vol_adjustment

        # Regime-gate: cap by regime exposure mult
        regime_max_position = self.limits.max_position_size * self._regime_exposure_mult
        max_size = (self.portfolio_value * regime_max_position) / entry_price
        position_size = min(position_size, max_size)
        position_size = max(0.0, position_size)

        return position_size

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add_position(
        self,
        asset: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
    ) -> bool:
        if self.circuit_breaker_triggered:
            logger.warning("Circuit breaker triggered — not adding new positions")
            return False

        position_value = size * entry_price
        if position_value / max(self.portfolio_value, 1) > self.limits.max_position_size:
            logger.warning(
                "Position size exceeds limit: %.2f%%",
                position_value / self.portfolio_value * 100,
            )
            return False

        # Seed per-symbol peak tracking
        if asset not in self._symbol_peaks:
            self._symbol_peaks[asset] = entry_price

        self.positions[asset] = {
            "size": size,
            "entry_price": entry_price,
            "current_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "value": position_value,
            "entry_time": datetime.now(),
            "unrealized_pnl": 0.0,
        }
        logger.info("Added position: %s x %.6f @ %.4f", asset, size, entry_price)
        return True

    def update_position(self, asset: str, current_price: float) -> Dict[str, Any]:
        if asset not in self.positions:
            return {"error": "Position not found"}

        # Update per-symbol peak/drawdown
        self.update_symbol_price(asset, current_price)

        position = self.positions[asset]
        position["current_price"] = current_price
        position["value"] = position["size"] * current_price

        entry_value = position["size"] * position["entry_price"]
        position["unrealized_pnl"] = position["value"] - entry_value

        if position["stop_loss"] is not None:
            if position["size"] > 0 and current_price <= position["stop_loss"]:
                return {"exit_signal": "stop_loss", "exit_price": position["stop_loss"], "pnl": position["unrealized_pnl"]}
            elif position["size"] < 0 and current_price >= position["stop_loss"]:
                return {"exit_signal": "stop_loss", "exit_price": position["stop_loss"], "pnl": position["unrealized_pnl"]}

        if position["take_profit"] is not None:
            if position["size"] > 0 and current_price >= position["take_profit"]:
                return {"exit_signal": "take_profit", "exit_price": position["take_profit"], "pnl": position["unrealized_pnl"]}
            elif position["size"] < 0 and current_price <= position["take_profit"]:
                return {"exit_signal": "take_profit", "exit_price": position["take_profit"], "pnl": position["unrealized_pnl"]}

        return {"status": "holding", "unrealized_pnl": position["unrealized_pnl"]}

    def close_position(self, asset: str, exit_price: Optional[float] = None) -> Dict[str, Any]:
        if asset not in self.positions:
            return {"error": "Position not found"}

        position = self.positions[asset]
        if exit_price is None:
            exit_price = position["current_price"]

        entry_value = position["size"] * position["entry_price"]
        exit_value = position["size"] * exit_price
        realized_pnl = exit_value - entry_value
        self.portfolio_value += realized_pnl

        result = {
            "asset": asset,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "size": position["size"],
            "realized_pnl": realized_pnl,
            "holding_time": (
                datetime.now() - position["entry_time"]
            ).total_seconds() / 3600,
            "symbol_drawdown_at_close": self._symbol_drawdowns.get(asset, 0.0),
        }
        del self.positions[asset]
        logger.info("Closed position: %s  P&L: %.2f", asset, realized_pnl)
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_portfolio_risk_report(self) -> Dict[str, Any]:
        total_exposure = sum(abs(p["value"]) for p in self.positions.values())
        return {
            "portfolio_value": self.portfolio_value,
            "total_exposure": total_exposure,
            "exposure_ratio": total_exposure / max(self.portfolio_value, 1),
            "regime_exposure_mult": self._regime_exposure_mult,
            "current_regime": self._current_regime,
            "regime_confidence": self._regime_confidence,
            "current_drawdown": self.max_drawdown,
            "daily_pnl": self.daily_pnl,
            "value_at_risk": self.value_at_risk,
            "expected_shortfall": self.expected_shortfall,
            "sharpe_ratio": self.sharpe_ratio,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "active_positions": len(self.positions),
            "symbol_drawdowns": dict(self._symbol_drawdowns),
            "risk_limits": {
                "max_drawdown": self.limits.max_drawdown,
                "max_daily_loss": self.limits.max_daily_loss,
                "max_position_size": self.limits.max_position_size,
                "circuit_breaker_threshold": self.limits.circuit_breaker_threshold,
                "max_symbol_drawdown": self.limits.max_symbol_drawdown,
            },
        }

    def emergency_stop(self) -> Dict[str, Any]:
        logger.warning("EMERGENCY STOP ACTIVATED")
        closed_positions, total_pnl = [], 0
        for asset in list(self.positions.keys()):
            result = self.close_position(asset)
            if "realized_pnl" in result:
                closed_positions.append(result)
                total_pnl += result["realized_pnl"]
        self.circuit_breaker_triggered = True
        return {
            "emergency_stop": True,
            "positions_closed": len(closed_positions),
            "total_pnl": total_pnl,
            "final_portfolio_value": self.portfolio_value,
        }
