#!/usr/bin/env python3
"""
Argus Trading Bot - Unified Risk Manager
=======================================

This module is a **working, importable** unified risk manager intended to be the
single source of truth for core runtime risk controls:
- Daily loss limit (percentage)
- Max consecutive losses circuit-breaker
- Max exposure / leverage checks (lightweight)
- Circuit breaker cooldown with safe reset conditions

Note: This file previously contained corrupted syntax. It has been rebuilt to a
minimal, reliable implementation so other parts of the system can depend on it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Deque, Dict, List, Optional
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RiskMetrics:
    # VaR metrics (simple historical proxy; can be replaced with more advanced methods)
    var_95: float
    var_99: float

    # Drawdown
    drawdown: float
    max_drawdown: float

    # Exposure
    current_capital: float
    total_exposure: float
    leverage: float

    # Performance
    daily_pnl: float
    daily_return_pct: float

    # Flags
    risk_level: RiskLevel
    circuit_breaker_active: bool
    consecutive_losses: int


@dataclass
class StrategyRiskLimits:
    """Per-strategy risk limits."""
    max_daily_loss_pct: float = 2.0         # Max daily loss as % of capital
    max_consecutive_losses: int = 5         # Max consecutive losses before cooldown
    max_position_pct: float = 10.0          # Max single position as % of capital
    cooldown_after_loss_streak_minutes: int = 60  # Cooldown period after loss streak


class UnifiedRiskManager:
    def __init__(
        self,
        initial_capital: float,
        max_daily_loss: float = 0.02,
        max_position_loss: float = 0.01,
        max_total_exposure: float = 0.8,
        max_leverage: float = 3.0,
        max_consecutive_losses: int = 5,
        circuit_breaker_cooldown_minutes: int = 60,
        var_lookback_days: int = 30,
        var_confidence: float = 0.95,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.peak_capital = float(initial_capital)

        self.max_daily_loss = float(max_daily_loss)
        self.max_position_loss = float(max_position_loss)
        self.max_total_exposure = float(max_total_exposure)
        self.max_leverage = float(max_leverage)
        self.max_consecutive_losses = int(max_consecutive_losses)

        self.circuit_breaker_cooldown = timedelta(minutes=int(circuit_breaker_cooldown_minutes))
        self.circuit_breaker_active = False
        self.circuit_breaker_activated_at: Optional[datetime] = None
        self.circuit_breaker_reason: str = ""

        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self._last_reset_date = datetime.now().date()

        # Light-weight history for VaR proxy
        self.var_lookback_days = int(var_lookback_days)
        self.var_confidence = float(var_confidence)
        self.returns_history: Deque[float] = deque(maxlen=self.var_lookback_days * 24)

        # Exposure bookkeeping (caller can keep this updated)
        self.total_exposure_usd = 0.0

        # Margin requirement tracking for perpetual futures
        self._margin_requirements: Dict[str, float] = {}

        # Correlation monitor (injected via set_correlation_monitor)
        self._correlation_monitor: Optional[object] = None

        # State store for capital persistence across restarts
        self._state_store: Optional[object] = None

        logger.info("UnifiedRiskManager ready")
        logger.info("   Initial capital: %.2f", self.initial_capital)
        logger.info("   Max daily loss: %.2f%%", self.max_daily_loss * 100.0)
        logger.info("   Max total exposure: %.2f%%", self.max_total_exposure * 100.0)
        logger.info("   Max leverage: %.2fx", self.max_leverage)

    def attach_state_store(self, state_store: object) -> None:
        """Attach an ExecutionStateStore to persist/restore capital across restarts."""
        self._state_store = state_store
        # Attempt to restore persisted capital
        try:
            get_val = getattr(state_store, "get_account_value", None)
            if callable(get_val):
                saved_capital = get_val("current_capital")
                saved_peak = get_val("peak_capital")
                saved_daily = get_val("daily_pnl")
                saved_consec = get_val("consecutive_losses")
                if saved_capital is not None and float(saved_capital) > 0:
                    self.current_capital = float(saved_capital)
                    logger.info("Restored capital from state store: %.2f (was %.2f)", self.current_capital, self.initial_capital)
                if saved_peak is not None and float(saved_peak) > 0:
                    self.peak_capital = float(saved_peak)
                if saved_daily is not None:
                    self.daily_pnl = float(saved_daily)
                if saved_consec is not None:
                    self.consecutive_losses = int(saved_consec)
                saved_date = get_val("last_reset_date")
                if saved_date is not None:
                    try:
                        from datetime import date as _date
                        self._last_reset_date = _date.fromisoformat(str(saved_date))
                    except (ValueError, TypeError):
                        pass  # keep default (today)
        except Exception as e:
            logger.debug("Could not restore capital from state store: %s", e)

    def _maybe_reset_daily(self) -> None:
        today = datetime.now().date()
        if today != self._last_reset_date:
            # FIX #13: Persist BEFORE updating memory state so restart restores correctly
            if self._state_store is not None:
                try:
                    self._state_store.set_account_value("last_reset_date", today.isoformat())
                    self._state_store.set_account_value("daily_pnl", 0.0)
                    self._state_store.set_account_value("consecutive_losses", 0)
                except Exception as _e:
                    logger.warning("Daily reset persist failed — skipping reset: %s", _e)
                    return  # Don't reset in-memory if we can't persist
            self._last_reset_date = today
            self.daily_pnl = 0.0
            self.consecutive_losses = 0

    def update_capital(self, new_capital: float, pnl: float = 0.0) -> None:
        self._maybe_reset_daily()

        old_capital = float(self.current_capital)
        new_capital_f = float(new_capital)
        pnl_f = float(pnl)

        self.current_capital = new_capital_f
        self.daily_pnl += pnl_f

        if new_capital_f > self.peak_capital:
            self.peak_capital = new_capital_f

        if old_capital > 0:
            self.returns_history.append((new_capital_f - old_capital) / old_capital)

        # Persist capital to state store so it survives restarts
        if self._state_store is not None:
            try:
                self._state_store.set_account_value("current_capital", new_capital_f)
                self._state_store.set_account_value("peak_capital", self.peak_capital)
                self._state_store.set_account_value("daily_pnl", self.daily_pnl)
                self._state_store.set_account_value("consecutive_losses", self.consecutive_losses)
            except Exception as _e:
                logger.debug("Capital persist: %s", _e)

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade outcome (updates consecutive loss tracking and daily P&L)."""
        self._maybe_reset_daily()
        pnl_f = float(pnl)
        # Update daily P&L in real-time (was previously only via update_capital)
        self.daily_pnl += pnl_f
        if pnl_f >= 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        # Immediately check circuit breaker after each trade
        if self.check_circuit_breaker():
            logger.warning("CIRCUIT BREAKER triggered after trade (PnL=%.2f, daily=%.2f, consec_losses=%d)",
                           pnl_f, self.daily_pnl, self.consecutive_losses)
        # Persist state
        if self._state_store is not None:
            try:
                self._state_store.set_account_value("daily_pnl", self.daily_pnl)
                self._state_store.set_account_value("consecutive_losses", self.consecutive_losses)
            except Exception as _e:
                logger.debug("unified_risk_manager error: %s", _e)

    def set_total_exposure(self, exposure_usd: float) -> None:
        self.total_exposure_usd = float(exposure_usd)

    def is_daily_loss_limit_exceeded(self) -> bool:
        self._maybe_reset_daily()
        if self.max_daily_loss <= 0:
            return False
        # Calculate daily loss as percentage of initial capital
        daily_loss_pct = abs(self.daily_pnl) / self.initial_capital if self.initial_capital > 0 else 0
        return self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss

    def is_circuit_breaker_active(self) -> bool:
        self._maybe_reset_daily()
        # If active, allow safe reset only if cooldown elapsed and conditions are OK
        if self.circuit_breaker_active and self.circuit_breaker_activated_at:
            elapsed = datetime.now() - self.circuit_breaker_activated_at
            if elapsed >= self.circuit_breaker_cooldown:
                # Safe reset conditions
                if not self.is_daily_loss_limit_exceeded() and self.consecutive_losses < self.max_consecutive_losses:
                    logger.info("Circuit breaker cooldown complete and conditions improved; resuming trading")
                    self.circuit_breaker_active = False
                    self.circuit_breaker_activated_at = None
                    self.circuit_breaker_reason = ""
        return self.circuit_breaker_active

    def check_circuit_breaker(self) -> bool:
        """Returns True if trading should be halted."""
        self._maybe_reset_daily()

        # Check daily loss limit
        if self.is_daily_loss_limit_exceeded():
            self._activate_circuit_breaker(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
            return True

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._activate_circuit_breaker(f"Max consecutive losses: {self.consecutive_losses}")
            return True

        # Check circuit breaker cooldown and reset conditions
        if self.circuit_breaker_active and self.circuit_breaker_activated_at:
            elapsed = datetime.now() - self.circuit_breaker_activated_at
            if elapsed >= self.circuit_breaker_cooldown:
                # Safe reset conditions
                if not self.is_daily_loss_limit_exceeded() and self.consecutive_losses < self.max_consecutive_losses:
                    logger.info("Circuit breaker cooldown complete and conditions improved; resuming trading")
                    self.circuit_breaker_active = False
                    self.circuit_breaker_activated_at = None
                    self.circuit_breaker_reason = ""

        return self.circuit_breaker_active

    def _activate_circuit_breaker(self, reason: str) -> None:
        if not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            self.circuit_breaker_activated_at = datetime.now()
            self.circuit_breaker_reason = reason

    def trip_circuit_breaker(self, reason: str) -> None:
        """Public API to trip circuit breaker (e.g. VaR breach, external alert)."""
        self._activate_circuit_breaker(str(reason))
        logger.warning("CIRCUIT BREAKER ACTIVE: %s", reason)

    # ------------------------------------------------------------------
    # Regime-aware position sizing
    # ------------------------------------------------------------------

    @staticmethod
    def get_regime_adjusted_position_limit(base_limit: float, regime: str) -> float:
        """
        Scale *base_limit* down (or keep it flat) based on the current market regime.

        Multipliers:
          CRISIS / EXTREME          → 0.25 (quarter size — preserve capital)
          HIGH_VOL / ELEVATED       → 0.50 (half size)
          TRENDING_DOWN / BEAR      → 0.65 (moderately reduced)
          BREAKOUT                  → 0.80 (slightly reduced — fast moves, higher slippage)
          RANGE / NORMAL / BULL /
          TRENDING_UP / UNKNOWN     → 1.00 (full size)

        Args:
            base_limit: The base maximum position size (e.g. max_position_size_aud).
            regime:     Regime string as produced by strategy_engine / HMM classifier.

        Returns:
            Adjusted limit (always >= 0).
        """
        _regime = str(regime).upper().strip()
        _MULTIPLIERS: Dict[str, float] = {
            "CRISIS": 0.25,
            "EXTREME": 0.25,
            "HIGH_VOL": 0.50,
            "HIGH_VOLATILITY": 0.50,
            "ELEVATED": 0.50,
            "TRENDING_DOWN": 0.65,
            "BEAR": 0.65,
            "BEARISH": 0.65,
            "BREAKOUT": 0.80,
        }
        multiplier = _MULTIPLIERS.get(_regime, 1.0)
        adjusted = float(base_limit) * multiplier
        if multiplier < 1.0:
            logger.debug(
                "Regime-adjusted position limit: %.2f → %.2f (regime=%s, multiplier=%.2f)",
                base_limit, adjusted, regime, multiplier,
            )
        return adjusted

    def set_correlation_monitor(self, monitor: object) -> None:
        """Inject a CorrelationMonitor for cross-asset position limit scaling."""
        self._correlation_monitor = monitor
        logger.info("UnifiedRiskManager: correlation monitor attached")

    def get_corr_adjusted_position_limit(
        self,
        base_limit: float,
        regime: str,
        symbols: Optional[List[str]] = None,
    ) -> float:
        """
        Like get_regime_adjusted_position_limit but also multiplies by the
        correlation penalty when a CorrelationMonitor is available.

        The correlation penalty ranges from 0.5 (avg pairwise corr >= 0.8)
        to 1.0 (avg pairwise corr <= 0.3), linearly interpolated between.

        Args:
            base_limit: The base maximum position size.
            regime:     Current market regime string.
            symbols:    Optional list of symbols to compute correlation for.
                        If None, all monitored symbols are used.

        Returns:
            Adjusted limit (always >= 0).
        """
        adjusted = self.get_regime_adjusted_position_limit(base_limit, regime)

        if self._correlation_monitor is not None:
            try:
                penalty = self._correlation_monitor.get_correlation_penalty(symbols)
                if penalty < 1.0:
                    logger.debug(
                        "Correlation penalty applied: %.2f → %.2f (penalty=%.2f)",
                        adjusted, adjusted * penalty, penalty,
                    )
                adjusted *= penalty
            except Exception as exc:
                logger.debug("Correlation penalty lookup failed: %s", exc)

        return adjusted

    def check_flash_crash(self, symbol: str, current_price: float, previous_price: float,
                          flash_crash_pct: float = 0.15) -> bool:
        """
        Detect flash crash: if price moved more than flash_crash_pct in one cycle.
        Returns True if flash crash detected (and trips circuit breaker).
        """
        if previous_price <= 0 or current_price <= 0:
            return False
        move_pct = abs(current_price - previous_price) / previous_price
        if move_pct >= flash_crash_pct:
            reason = f"FLASH CRASH detected on {symbol}: {move_pct*100:.1f}% move ({previous_price:.2f} -> {current_price:.2f})"
            self.trip_circuit_breaker(reason)
            return True
        return False

    def check_cycle_latency(self, cycle_duration_ms: float, max_latency_ms: float = 30000.0) -> bool:
        """
        Check if trading cycle took too long (possible network issues).
        Returns True if latency exceeded and trips circuit breaker.
        """
        if max_latency_ms <= 0:
            return False
        if cycle_duration_ms > max_latency_ms:
            reason = f"LATENCY SPIKE: cycle took {cycle_duration_ms:.0f}ms (max {max_latency_ms:.0f}ms)"
            self.trip_circuit_breaker(reason)
            return True
        return False

    # ------------------------------------------------------------------
    # Margin requirement tracking (perpetual futures)
    # ------------------------------------------------------------------

    def update_margin_requirement(self, symbol: str, required_margin_usd: float) -> None:
        """Update the margin requirement for a symbol. Set to 0 to clear on position close."""
        margin = float(required_margin_usd)
        if margin <= 0.0:
            self._margin_requirements.pop(symbol, None)
            logger.debug("Margin cleared for %s", symbol)
        else:
            self._margin_requirements[symbol] = margin
            logger.debug("Margin updated for %s: %.2f USD", symbol, margin)

    def get_total_margin(self) -> float:
        """Sum of all current margin requirements across symbols."""
        return sum(self._margin_requirements.values())

    def get_free_margin(self, total_capital_usd: float) -> float:
        """Available margin = capital minus total margin in use."""
        return float(total_capital_usd) - self.get_total_margin()

    def check_margin_available(self, additional_margin_usd: float, total_capital_usd: float) -> bool:
        """Return True if there is enough free margin to accommodate additional_margin_usd."""
        free = self.get_free_margin(float(total_capital_usd))
        return free >= float(additional_margin_usd)

    # ------------------------------------------------------------------
    # Pre-trade risk check
    # ------------------------------------------------------------------

    def pre_trade_risk_check(
        self,
        symbol: str,
        position_size_usd: float,
        required_margin_usd: float = 0.0,
        total_capital_usd: Optional[float] = None,
    ) -> tuple:
        """
        Comprehensive pre-trade risk gate.

        Returns:
            (approved: bool, reason: str)
        """
        capital = float(total_capital_usd) if total_capital_usd is not None else self.current_capital

        # 1. Circuit breaker
        if self.check_circuit_breaker():
            return False, "circuit_breaker_active"

        # 2. Daily loss limit
        if self.is_daily_loss_limit_exceeded():
            return False, "daily_loss_limit_exceeded"

        # 3. Exposure / leverage check
        projected_exposure = self.total_exposure_usd + abs(float(position_size_usd))
        if capital > 0 and projected_exposure / capital > self.max_leverage:
            return False, f"max_leverage_exceeded ({projected_exposure / capital:.2f}x > {self.max_leverage:.2f}x)"

        # 4. Margin check (for perpetual futures)
        if required_margin_usd > 0.0:
            if not self.check_margin_available(float(required_margin_usd), capital):
                free = self.get_free_margin(capital)
                return False, f"insufficient_margin (need={required_margin_usd:.2f}, free={free:.2f})"

        return True, "approved"

    # ------------------------------------------------------------------
    # Liquidation price calculator
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_liquidation_price(
        entry_price: float,
        leverage: float,
        side: str,
        maintenance_margin_pct: float = 0.5,
    ) -> float:
        """
        Calculate the price at which a leveraged position gets liquidated.

        Args:
            entry_price: Position entry price.
            leverage: Leverage multiplier (e.g. 10 for 10x).
            side: "long" or "short" (case-insensitive).
            maintenance_margin_pct: Maintenance margin as a percentage (default 0.5%).

        Returns:
            Liquidation price. Always >= 0.
        """
        entry = float(entry_price)
        lev = float(leverage)
        mm = float(maintenance_margin_pct)
        if lev <= 0:
            raise ValueError("leverage must be positive")
        if entry <= 0:
            raise ValueError("entry_price must be positive")

        _side = str(side).strip().upper()
        if _side in ("LONG", "BUY"):
            liq = entry * (1.0 - 1.0 / lev + mm / 100.0)
        elif _side in ("SHORT", "SELL"):
            liq = entry * (1.0 + 1.0 / lev - mm / 100.0)
        else:
            raise ValueError(f"side must be 'long' or 'short', got '{side}'")
        return max(0.0, liq)

    # ------------------------------------------------------------------
    # Margin call auto-reduce
    # ------------------------------------------------------------------

    def check_margin_call(
        self,
        total_capital_usd: float,
        margin_call_pct: float = 80.0,
    ) -> List[dict]:
        """
        Check if total margin usage exceeds margin_call_pct of capital.

        If so, returns a list of positions to reduce (largest margin first),
        each with ``symbol`` and ``reduce_by_pct`` (enough to get back to 70% usage).

        Returns:
            List of dicts with keys ``symbol`` and ``reduce_by_pct``, or empty list.
        """
        capital = float(total_capital_usd)
        if capital <= 0:
            return []

        total_margin = self.get_total_margin()
        usage_pct = (total_margin / capital) * 100.0

        if usage_pct <= margin_call_pct:
            return []

        # Target: reduce to 70% usage
        target_margin = capital * 0.70
        excess = total_margin - target_margin
        if excess <= 0:
            return []

        # Sort by margin descending (largest first)
        sorted_positions = sorted(
            self._margin_requirements.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        reductions: List[dict] = []
        remaining_excess = excess
        for symbol, margin in sorted_positions:
            if remaining_excess <= 0:
                break
            reduce_amount = min(margin, remaining_excess)
            reduce_pct = (reduce_amount / margin) * 100.0
            reductions.append({
                "symbol": symbol,
                "reduce_by_pct": round(reduce_pct, 2),
            })
            remaining_excess -= reduce_amount

        logger.warning(
            "MARGIN CALL: usage %.1f%% > %.1f%%, reducing %d positions",
            usage_pct, margin_call_pct, len(reductions),
        )
        return reductions

    def auto_reduce_positions(self, total_capital_usd: float) -> List[dict]:
        """
        Return a reduction plan if margin usage exceeds 80% of capital.

        Convenience wrapper around check_margin_call with default thresholds.
        """
        return self.check_margin_call(total_capital_usd, margin_call_pct=80.0)

    # ------------------------------------------------------------------
    # Microstructure anomaly detection
    # ------------------------------------------------------------------

    def detect_microstructure_anomaly(
        self,
        prices: List[float],
        timestamps: List[float],
        normal_spread_bps: float = 5.0,
        rolling_volume_window: int = 20,
    ) -> dict:
        """
        Detect microstructure anomalies indicative of flash crashes.

        Checks:
          1. Bid-ask spread > 5x normal (inferred from price volatility)
          2. Volume spike > 10x rolling average (approximated via price changes)
          3. Price reversal > 2% within 60 seconds

        Parameters
        ----------
        prices:
            Ordered sequence of trade/mid prices (most recent last).
        timestamps:
            Corresponding UNIX timestamps (same length as prices).
        normal_spread_bps:
            Baseline spread in basis points for comparison.
        rolling_volume_window:
            Window size for computing rolling average of volume proxy.

        Returns
        -------
        dict with keys:
            anomaly  — bool, True if any anomaly detected
            type     — str describing the anomaly kind(s), empty if none
            severity — float in [0, 1], higher = more severe
        """
        result = {"anomaly": False, "type": "", "severity": 0.0}

        if len(prices) < 3 or len(timestamps) < 3 or len(prices) != len(timestamps):
            return result

        anomaly_types: List[str] = []
        max_severity = 0.0

        prices_arr = np.array(prices, dtype=float)
        ts_arr = np.array(timestamps, dtype=float)

        # 1. Spread proxy: use max consecutive price jump as spread indicator
        abs_returns_bps = np.abs(np.diff(prices_arr) / prices_arr[:-1]) * 10000.0
        if len(abs_returns_bps) > 0:
            max_jump_bps = float(np.max(abs_returns_bps))
            if max_jump_bps > normal_spread_bps * 5:
                anomaly_types.append("spread_blow_out")
                severity = min(1.0, max_jump_bps / (normal_spread_bps * 10))
                max_severity = max(max_severity, severity)

        # 2. Volume spike proxy: magnitude of price change vs rolling average
        if len(abs_returns_bps) >= rolling_volume_window:
            rolling_avg = float(np.mean(abs_returns_bps[-rolling_volume_window:]))
            if rolling_avg > 0:
                recent_move = float(abs_returns_bps[-1])
                spike_ratio = recent_move / rolling_avg
                if spike_ratio > 10.0:
                    anomaly_types.append("volume_spike")
                    severity = min(1.0, spike_ratio / 20.0)
                    max_severity = max(max_severity, severity)

        # 3. Price reversal > 2% within 60 seconds
        n = len(prices_arr)
        for i in range(n - 1, 0, -1):
            dt = ts_arr[i] - ts_arr[i - 1]
            if dt > 60.0:
                break
            # Check from i back to all points within 60s window
            window_start = i
            while window_start > 0 and (ts_arr[i] - ts_arr[window_start - 1]) <= 60.0:
                window_start -= 1
            window_prices = prices_arr[window_start:i + 1]
            if len(window_prices) >= 3:
                p_min = float(np.min(window_prices))
                p_max = float(np.max(window_prices))
                if p_max > 0:
                    swing_pct = (p_max - p_min) / p_max
                    if swing_pct >= 0.02:
                        # Check for reversal: price went down then back up, or vice versa
                        mid_idx = len(window_prices) // 2
                        if mid_idx > 0 and mid_idx < len(window_prices) - 1:
                            first_half_trend = window_prices[mid_idx] - window_prices[0]
                            second_half_trend = window_prices[-1] - window_prices[mid_idx]
                            if first_half_trend * second_half_trend < 0:  # sign change = reversal
                                anomaly_types.append("price_reversal")
                                severity = min(1.0, swing_pct / 0.05)
                                max_severity = max(max_severity, severity)
                                break
            break  # only check the most recent 60s window

        if anomaly_types:
            result["anomaly"] = True
            result["type"] = ",".join(anomaly_types)
            result["severity"] = round(max_severity, 4)

        return result

    # ------------------------------------------------------------------
    # Stop-loss auto-execution
    # ------------------------------------------------------------------

    def __init_trailing_highs(self) -> None:
        """Lazy-init trailing highs dict (avoids changing __init__ signature)."""
        if not hasattr(self, "_trailing_highs"):
            self._trailing_highs: Dict[str, float] = {}
        if not hasattr(self, "_position_entry_times"):
            self._position_entry_times: Dict[str, datetime] = {}
        if not hasattr(self, "_strategy_daily_pnl"):
            self._strategy_daily_pnl: Dict[str, float] = {}
        if not hasattr(self, "_strategy_consecutive_losses"):
            self._strategy_consecutive_losses: Dict[str, int] = {}
        if not hasattr(self, "_strategy_cooldown_until"):
            self._strategy_cooldown_until: Dict[str, datetime] = {}
        if not hasattr(self, "_strategy_limits"):
            self._strategy_limits: Optional['StrategyRiskLimits'] = None

    def update_trailing_stops(self, symbol: str, current_price: float) -> None:
        """Update trailing stop high-water mark for a position."""
        self.__init_trailing_highs()
        if current_price <= 0:
            return
        prev = self._trailing_highs.get(symbol, 0.0)
        if current_price > prev:
            self._trailing_highs[symbol] = current_price

    def register_entry_time(self, symbol: str, entry_time: Optional[datetime] = None) -> None:
        """Register when a position was entered (for time stops)."""
        self.__init_trailing_highs()
        self._position_entry_times[symbol] = entry_time or datetime.now()

    def clear_position_tracking(self, symbol: str) -> None:
        """Remove tracking data for a closed position."""
        self.__init_trailing_highs()
        self._trailing_highs.pop(symbol, None)
        self._position_entry_times.pop(symbol, None)

    def check_stops(
        self,
        positions: Dict[str, dict],
        current_prices: Dict[str, float],
        stop_loss_pct: float = 0.02,
        trail_pct: float = 0.015,
        max_hold_hours: float = 72.0,
    ) -> List[dict]:
        """Check all open positions against their stop levels.

        Returns list of dicts for positions that should be closed:
            {symbol, side, quantity, reason, stop_price, current_price}

        Stop types checked:
        - Fixed stop: entry_price * (1 - stop_loss_pct) for longs
        - Trailing stop: track highest price since entry, stop at high * (1 - trail_pct)
        - Time stop: close if held > max_hold_hours
        """
        self.__init_trailing_highs()
        triggered: List[dict] = []

        for symbol, pos in (positions or {}).items():
            if pos is None:
                continue
            quantity = float((pos or {}).get("quantity") or 0.0)
            if quantity <= 0:
                continue

            entry_price = float((pos or {}).get("entry_price") or 0.0)
            current_price = float(current_prices.get(symbol) or (pos or {}).get("current_price") or 0.0)
            side = str((pos or {}).get("side", "long")).lower()

            if entry_price <= 0 or current_price <= 0:
                continue

            # Update trailing high water mark
            self.update_trailing_stops(symbol, current_price)

            # 1. Fixed stop
            if side in ("long", "buy"):
                fixed_stop = entry_price * (1.0 - stop_loss_pct)
                if current_price <= fixed_stop:
                    triggered.append({
                        "symbol": symbol,
                        "side": "SELL",
                        "quantity": quantity,
                        "reason": f"fixed_stop_loss: price {current_price:.2f} <= stop {fixed_stop:.2f} ({stop_loss_pct:.1%} below entry {entry_price:.2f})",
                        "stop_price": fixed_stop,
                        "current_price": current_price,
                    })
                    continue
            else:  # short
                fixed_stop = entry_price * (1.0 + stop_loss_pct)
                if current_price >= fixed_stop:
                    triggered.append({
                        "symbol": symbol,
                        "side": "BUY",
                        "quantity": quantity,
                        "reason": f"fixed_stop_loss: price {current_price:.2f} >= stop {fixed_stop:.2f} ({stop_loss_pct:.1%} above entry {entry_price:.2f})",
                        "stop_price": fixed_stop,
                        "current_price": current_price,
                    })
                    continue

            # 2. Trailing stop
            high_water = self._trailing_highs.get(symbol, entry_price)
            if side in ("long", "buy"):
                trailing_stop = high_water * (1.0 - trail_pct)
                if current_price <= trailing_stop and high_water > entry_price:
                    triggered.append({
                        "symbol": symbol,
                        "side": "SELL",
                        "quantity": quantity,
                        "reason": f"trailing_stop: price {current_price:.2f} <= trail {trailing_stop:.2f} (high={high_water:.2f}, trail={trail_pct:.1%})",
                        "stop_price": trailing_stop,
                        "current_price": current_price,
                    })
                    continue
            else:
                low_water = self._trailing_highs.get(symbol, entry_price)  # re-use dict for short low-water
                trailing_stop = low_water * (1.0 + trail_pct)
                if current_price >= trailing_stop and low_water < entry_price:
                    triggered.append({
                        "symbol": symbol,
                        "side": "BUY",
                        "quantity": quantity,
                        "reason": f"trailing_stop: price {current_price:.2f} >= trail {trailing_stop:.2f} (low={low_water:.2f}, trail={trail_pct:.1%})",
                        "stop_price": trailing_stop,
                        "current_price": current_price,
                    })
                    continue

            # 3. Time stop
            entry_time = self._position_entry_times.get(symbol)
            if entry_time is not None and max_hold_hours > 0:
                held_hours = (datetime.now() - entry_time).total_seconds() / 3600.0
                if held_hours >= max_hold_hours:
                    close_side = "SELL" if side in ("long", "buy") else "BUY"
                    triggered.append({
                        "symbol": symbol,
                        "side": close_side,
                        "quantity": quantity,
                        "reason": f"time_stop: held {held_hours:.1f}h >= max {max_hold_hours:.1f}h",
                        "stop_price": current_price,
                        "current_price": current_price,
                    })

        return triggered

    # ------------------------------------------------------------------
    # Margin enforcement / deleveraging
    # ------------------------------------------------------------------

    def enforce_margin(
        self,
        positions: Dict[str, dict],
        current_prices: Dict[str, float],
        total_capital: float,
    ) -> List[dict]:
        """Check if portfolio exceeds margin / leverage limits.

        Returns list of positions to force-close (largest notional first)
        until leverage is back under max_leverage.
        """
        if total_capital <= 0:
            return []

        # Calculate total notional
        position_notionals: List[tuple] = []  # (symbol, notional, quantity, side)
        total_notional = 0.0
        for symbol, pos in (positions or {}).items():
            if pos is None:
                continue
            quantity = float((pos or {}).get("quantity") or 0.0)
            if quantity <= 0:
                continue
            current_price = float(current_prices.get(symbol) or (pos or {}).get("current_price") or 0.0)
            if current_price <= 0:
                continue
            notional = abs(quantity * current_price)
            total_notional += notional
            side = str((pos or {}).get("side", "long")).lower()
            entry_price = float((pos or {}).get("entry_price") or current_price)
            # Calculate unrealized PnL for sorting (cut losers first)
            if side in ("long", "buy"):
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            position_notionals.append((symbol, notional, quantity, side, pnl))

        current_leverage = total_notional / total_capital
        if current_leverage <= self.max_leverage:
            return []

        logger.critical(
            "MARGIN ENFORCEMENT: leverage %.2fx > max %.2fx (notional=%.2f, capital=%.2f)",
            current_leverage, self.max_leverage, total_notional, total_capital,
        )

        return self.deleverage(positions, self.max_leverage, current_prices, total_capital)

    def deleverage(
        self,
        positions: Dict[str, dict],
        target_leverage: float,
        current_prices: Dict[str, float],
        total_capital: float,
    ) -> List[dict]:
        """Force-close positions until portfolio leverage <= target.

        Closes highest-loss positions first (cut losers).
        Returns list of {symbol, quantity_to_close, reason, side}.
        """
        if total_capital <= 0:
            return []

        # Build position list with PnL
        pos_list: List[tuple] = []
        total_notional = 0.0
        for symbol, pos in (positions or {}).items():
            if pos is None:
                continue
            quantity = float((pos or {}).get("quantity") or 0.0)
            if quantity <= 0:
                continue
            current_price = float(current_prices.get(symbol) or (pos or {}).get("current_price") or 0.0)
            if current_price <= 0:
                continue
            notional = abs(quantity * current_price)
            total_notional += notional
            side = str((pos or {}).get("side", "long")).lower()
            entry_price = float((pos or {}).get("entry_price") or current_price)
            if side in ("long", "buy"):
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            pos_list.append((symbol, notional, quantity, side, pnl, current_price))

        target_notional = target_leverage * total_capital
        excess = total_notional - target_notional
        if excess <= 0:
            return []

        # Sort by PnL ascending (worst losers first)
        pos_list.sort(key=lambda x: x[4])

        closures: List[dict] = []
        remaining_excess = excess
        for symbol, notional, quantity, side, pnl, current_price in pos_list:
            if remaining_excess <= 0:
                break
            # Close enough of this position to cover the excess
            if notional <= remaining_excess:
                qty_to_close = quantity
            else:
                fraction = remaining_excess / notional
                qty_to_close = quantity * fraction
            close_side = "SELL" if side in ("long", "buy") else "BUY"
            closures.append({
                "symbol": symbol,
                "quantity_to_close": qty_to_close,
                "reason": f"deleverage: reducing to {target_leverage:.1f}x (pnl={pnl:.2f})",
                "side": close_side,
                "current_price": current_price,
            })
            remaining_excess -= min(notional, remaining_excess)

        return closures

    # ------------------------------------------------------------------
    # Per-strategy risk limits
    # ------------------------------------------------------------------

    def set_strategy_limits(self, limits: 'StrategyRiskLimits') -> None:
        """Set per-strategy risk limits."""
        self.__init_trailing_highs()
        self._strategy_limits = limits

    def record_strategy_trade(self, strategy_name: str, pnl: float) -> None:
        """Record a trade outcome for a specific strategy."""
        self.__init_trailing_highs()
        self._strategy_daily_pnl.setdefault(strategy_name, 0.0)
        self._strategy_daily_pnl[strategy_name] += float(pnl)
        if pnl < 0:
            self._strategy_consecutive_losses[strategy_name] = (
                self._strategy_consecutive_losses.get(strategy_name, 0) + 1
            )
        else:
            self._strategy_consecutive_losses[strategy_name] = 0

    def check_strategy_limits(
        self,
        strategy_name: str,
        daily_pnl: Optional[float] = None,
        consecutive_losses: Optional[int] = None,
        total_capital: Optional[float] = None,
    ) -> tuple:
        """Check whether a strategy is allowed to trade.

        Returns (allowed: bool, reason: str).
        """
        self.__init_trailing_highs()
        limits = self._strategy_limits
        if limits is None:
            return True, "no_limits_configured"

        capital = float(total_capital or self.current_capital)
        if capital <= 0:
            capital = 1.0

        # Use tracked values if not explicitly provided
        if daily_pnl is None:
            daily_pnl = self._strategy_daily_pnl.get(strategy_name, 0.0)
        if consecutive_losses is None:
            consecutive_losses = self._strategy_consecutive_losses.get(strategy_name, 0)

        # 1. Daily loss check
        daily_loss_pct = abs(daily_pnl) / capital * 100.0  # as percentage
        if daily_pnl < 0 and daily_loss_pct >= limits.max_daily_loss_pct:
            return False, f"strategy_daily_loss: {daily_loss_pct:.2f}% >= {limits.max_daily_loss_pct:.2f}%"

        # 2. Consecutive losses check
        if consecutive_losses >= limits.max_consecutive_losses:
            # Set cooldown
            cooldown_end = self._strategy_cooldown_until.get(strategy_name)
            now = datetime.now()
            if cooldown_end is None or now >= cooldown_end:
                # Start new cooldown
                self._strategy_cooldown_until[strategy_name] = now + timedelta(
                    minutes=limits.cooldown_after_loss_streak_minutes
                )
            cooldown_end = self._strategy_cooldown_until[strategy_name]
            if now < cooldown_end:
                remaining = (cooldown_end - now).total_seconds() / 60.0
                return False, f"strategy_loss_streak_cooldown: {consecutive_losses} consecutive losses, {remaining:.0f}min remaining"

        # 3. Cooldown still active from previous streak
        cooldown_end = self._strategy_cooldown_until.get(strategy_name)
        if cooldown_end is not None and datetime.now() < cooldown_end:
            remaining = (cooldown_end - datetime.now()).total_seconds() / 60.0
            return False, f"strategy_cooldown_active: {remaining:.0f}min remaining"

        return True, "approved"

    # ------------------------------------------------------------------
    # Portfolio-level risk controls (advanced)
    # ------------------------------------------------------------------

    def check_portfolio_stop(self, daily_pnl_pct: float) -> bool:
        """Return True (halt trading) if daily loss exceeds the portfolio stop threshold.

        Default threshold: 5% daily loss.  Configurable via constructor's
        *max_daily_loss* or by overriding *portfolio_stop_pct* at runtime.

        Args:
            daily_pnl_pct: Daily P&L as a **signed** decimal (e.g. -0.05 for -5%).

        Returns:
            True if trading should be halted, False otherwise.
        """
        threshold = float(getattr(self, "_portfolio_stop_pct", 0.05) or 0.05)
        if daily_pnl_pct < 0 and abs(daily_pnl_pct) >= threshold:
            reason = (
                f"Portfolio daily stop: loss {daily_pnl_pct*100:.2f}% "
                f">= threshold {threshold*100:.1f}%"
            )
            self._activate_circuit_breaker(reason)
            logger.warning("PORTFOLIO STOP triggered: %s", reason)
            return True
        return False

    def check_correlation_risk(
        self,
        positions: Dict[str, dict],
        correlation_matrix: Dict[str, float],
    ) -> float:
        """Compute portfolio-level correlation risk score.

        If all active positions are highly correlated (>0.7 pairwise) the
        risk score approaches 1.0, indicating concentrated directional
        exposure.

        Args:
            positions: Dict of {symbol: position_dict} for open positions.
            correlation_matrix: Dict keyed as ``"SYM_A:SYM_B"`` or
                ``("SYM_A", "SYM_B")`` with correlation values in [-1, 1].

        Returns:
            Risk score in [0, 1].  >0.8 means dangerously correlated.
        """
        syms = [s for s, p in (positions or {}).items()
                if p and float((p or {}).get("quantity", 0) or 0) > 0]

        if len(syms) < 2:
            return 0.0

        corr_values: list = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a, b = syms[i], syms[j]
                val = (
                    correlation_matrix.get(f"{a}:{b}")
                    or correlation_matrix.get(f"{b}:{a}")
                    or correlation_matrix.get((a, b))
                    or correlation_matrix.get((b, a))
                )
                if val is not None:
                    corr_values.append(abs(float(val)))

        if not corr_values:
            return 0.0

        avg_corr = float(np.mean(corr_values))
        # Map average absolute correlation to a risk score:
        # 0.0 corr -> 0.0 risk, 0.3 -> ~0.25, 0.7 -> ~0.80, 1.0 -> 1.0
        # Use a power curve for non-linear scaling
        risk_score = min(1.0, avg_corr ** 1.2 * 1.15)

        if risk_score > 0.8:
            logger.warning(
                "HIGH correlation risk: avg_corr=%.3f across %d positions, risk_score=%.3f",
                avg_corr, len(syms), risk_score,
            )

        return round(risk_score, 4)

    def check_overnight_risk(
        self,
        utc_hour: int,
        positions: Dict[str, dict],
    ) -> List[dict]:
        """Identify positions that should be reduced during low-liquidity overnight hours.

        During UTC 02:00–05:00 (after US close, before Asia open) liquidity
        thins and flash-crash risk increases.  Returns a list of positions
        that should be reduced by 50%.

        Args:
            utc_hour: Current hour in UTC (0-23).
            positions: Dict of {symbol: position_dict}.

        Returns:
            List of dicts with ``symbol`` and ``reduce_by_pct`` (50) for
            positions that should be cut, or empty list outside the window.
        """
        overnight_start = int(getattr(self, "_overnight_start_utc", 2) or 2)
        overnight_end = int(getattr(self, "_overnight_end_utc", 5) or 5)
        reduce_pct = float(getattr(self, "_overnight_reduce_pct", 50.0) or 50.0)

        if not (overnight_start <= utc_hour < overnight_end):
            return []

        reductions: List[dict] = []
        for symbol, pos in (positions or {}).items():
            if pos is None:
                continue
            quantity = float((pos or {}).get("quantity", 0) or 0)
            if quantity <= 0:
                continue
            reductions.append({
                "symbol": symbol,
                "reduce_by_pct": reduce_pct,
                "reason": f"overnight_risk: UTC hour {utc_hour} in [{overnight_start}-{overnight_end})",
            })

        if reductions:
            logger.info(
                "Overnight risk: recommending %.0f%% reduction for %d positions (UTC %02d:00)",
                reduce_pct, len(reductions), utc_hour,
            )
        return reductions

    def get_regime_risk_multiplier(self, regime: str) -> float:
        """Return position size multiplier based on current market regime.

        Regime multipliers:
            trending  -> 1.0  (full size, clear directional opportunity)
            ranging   -> 0.8  (reduced, lower edge in chop)
            volatile  -> 0.5  (half size, protect capital)
            crisis    -> 0.2  (minimal exposure, survival mode)
            unknown   -> 0.7  (conservative default)

        Args:
            regime: Market regime string (case-insensitive).

        Returns:
            Multiplier in (0, 1].
        """
        _REGIME_MAP: Dict[str, float] = {
            "trending": 1.0,
            "trending_up": 1.0,
            "bull": 1.0,
            "bullish": 1.0,
            "ranging": 0.8,
            "range": 0.8,
            "normal": 0.8,
            "mean_reversion": 0.8,
            "volatile": 0.5,
            "high_vol": 0.5,
            "high_volatility": 0.5,
            "elevated": 0.5,
            "crisis": 0.2,
            "extreme": 0.2,
            "crash": 0.2,
            "trending_down": 0.6,
            "bear": 0.6,
            "bearish": 0.6,
            "breakout": 0.7,
        }
        key = str(regime).strip().lower()
        multiplier = _REGIME_MAP.get(key, 0.7)
        if multiplier < 1.0:
            logger.debug("Regime risk multiplier: regime=%s -> %.2f", regime, multiplier)
        return multiplier

    def check_flash_crash_1m(
        self,
        symbol: str,
        price_change_1m_pct: float,
        threshold_pct: float = 3.0,
        pause_minutes: float = 5.0,
    ) -> bool:
        """Detect rapid price moves and pause trading for a cooldown period.

        Unlike the existing ``check_flash_crash`` (which compares two prices
        and uses a 15% default), this method is tuned for **1-minute** granularity
        with a 3% default threshold and a 5-minute trading pause.

        Args:
            symbol: The trading pair (e.g. ``"BTC/USD"``).
            price_change_1m_pct: Absolute percentage change in the last minute
                (e.g. 3.5 means 3.5%).
            threshold_pct: Trigger threshold in percent (default 3%).
            pause_minutes: How long to pause trading (default 5 min).

        Returns:
            True if flash move detected (circuit breaker tripped for
            *pause_minutes*), False otherwise.
        """
        if abs(price_change_1m_pct) >= threshold_pct:
            # Override cooldown to the shorter flash-crash pause
            old_cooldown = self.circuit_breaker_cooldown
            self.circuit_breaker_cooldown = timedelta(minutes=pause_minutes)
            reason = (
                f"FLASH MOVE on {symbol}: {price_change_1m_pct:+.2f}% in 1 min "
                f"(threshold: {threshold_pct:.1f}%). Pausing for {pause_minutes:.0f} min."
            )
            self.trip_circuit_breaker(reason)
            # Restore original cooldown so normal circuit breaker behaviour is preserved
            self.circuit_breaker_cooldown = old_cooldown
            return True
        return False

    def get_risk_metrics(self) -> RiskMetrics:
        self._maybe_reset_daily()

        drawdown = 0.0
        if self.peak_capital > 0:
            drawdown = max(0.0, (self.peak_capital - self.current_capital) / self.peak_capital)

        # Simple historical VaR proxy from returns history
        returns = np.array(list(self.returns_history), dtype=float)
        if returns.size >= 10:
            var_95 = float(np.quantile(returns, 1 - 0.95)) * self.current_capital
            var_99 = float(np.quantile(returns, 1 - 0.99)) * self.current_capital
        else:
            var_95 = 0.0
            var_99 = 0.0

        base = max(self.current_capital, 1e-9)
        daily_return_pct = float(self.daily_pnl / base)

        leverage = 0.0
        if base > 0:
            leverage = float(self.total_exposure_usd / base)

        # Risk level heuristic
        risk_level = RiskLevel.LOW
        if self.is_daily_loss_limit_exceeded() or self.is_circuit_breaker_active():
            risk_level = RiskLevel.CRITICAL
        elif drawdown > 0.10 or leverage > self.max_leverage * 0.8:
            risk_level = RiskLevel.HIGH
        elif drawdown > 0.05 or leverage > self.max_leverage * 0.5:
            risk_level = RiskLevel.MEDIUM

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            drawdown=drawdown,
            max_drawdown=drawdown,  # placeholder; can be expanded with equity history
            current_capital=float(self.current_capital),
            total_exposure=float(self.total_exposure_usd),
            leverage=leverage,
            daily_pnl=float(self.daily_pnl),
            daily_return_pct=daily_return_pct,
            risk_level=risk_level,
            circuit_breaker_active=bool(self.is_circuit_breaker_active()),
            consecutive_losses=int(self.consecutive_losses),
        )

