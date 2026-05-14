"""
monitoring/enterprise_risk_integration.py
==========================================
Integrates existing risk infrastructure into the operator dashboard.

Wires together:
  - risk/advanced_risk_engine.py (3486 lines, institutional-grade VaR/CVaR)
  - risk/realtime_var_aggregator.py (streaming VaR)
  - risk/stress_tester_enhanced.py (stress testing)
  - risk/risk_limits_manager.py (limit enforcement)
  - monitoring/operator_dashboard.py (dashboard state)

This is the glue that makes the existing risk modules visible and actionable
through the operator dashboard and WebSocket feed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EnterpriseRiskSnapshot:
    """Full enterprise risk snapshot for dashboard consumption."""
    timestamp       : float
    # VaR metrics
    var_95          : float
    var_99          : float
    cvar_95         : float
    cvar_99         : float
    # Portfolio metrics
    portfolio_value : float
    total_exposure  : float
    net_exposure    : float
    leverage        : float
    # Drawdown metrics
    current_drawdown_pct: float
    max_drawdown_pct    : float
    # Risk utilisation
    risk_budget_used_pct: float
    var_limit_pct   : float
    var_utilisation : float
    # Per-asset breakdown
    asset_risks     : Dict[str, Dict[str, float]]
    # Stress test summary
    worst_stress_loss: float
    worst_stress_scenario: str
    # Alerts
    alerts          : List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp"             : self.timestamp,
            "var_95"               : self.var_95,
            "var_99"               : self.var_99,
            "cvar_95"              : self.cvar_95,
            "cvar_99"              : self.cvar_99,
            "portfolio_value"      : self.portfolio_value,
            "total_exposure"       : self.total_exposure,
            "net_exposure"         : self.net_exposure,
            "leverage"             : self.leverage,
            "current_drawdown_pct" : self.current_drawdown_pct,
            "max_drawdown_pct"     : self.max_drawdown_pct,
            "risk_budget_used_pct" : self.risk_budget_used_pct,
            "var_limit_pct"        : self.var_limit_pct,
            "var_utilisation"      : self.var_utilisation,
            "asset_risks"          : self.asset_risks,
            "worst_stress_loss"    : self.worst_stress_loss,
            "worst_stress_scenario": self.worst_stress_scenario,
            "alerts"               : self.alerts,
        }


@dataclass
class StressTestSummary:
    """Summary of latest stress test run."""
    scenario_name   : str
    initial_value   : float
    stressed_value  : float
    loss_pct        : float
    recovery_days   : int
    timestamp       : float


# ---------------------------------------------------------------------------
# Enterprise Risk Integrator
# ---------------------------------------------------------------------------

class EnterpriseRiskIntegrator:
    """
    Wires existing risk modules into a unified interface for the dashboard.

    Parameters
    ----------
    initial_capital     : float — starting capital for risk calculations
    var_limit_pct       : float — max VaR as % of capital (default 2%)
    stress_test_interval: float — seconds between stress test runs (default 3600)
    """

    def __init__(
        self,
        initial_capital     : float = 1_000_000.0,
        var_limit_pct       : float = 0.02,
        stress_test_interval: float = 3600.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.var_limit_pct   = var_limit_pct
        self.stress_test_interval = stress_test_interval

        # Internal state
        self._portfolio_value : float = initial_capital
        self._positions       : Dict[str, float] = {}  # symbol -> notional
        self._returns_history : Deque[float] = deque(maxlen=2000)
        self._price_history   : Dict[str, Deque[float]] = {}
        self._peak_equity     : float = initial_capital
        self._current_equity  : float = initial_capital
        self._total_exposure  : float = 0.0
        self._alerts          : List[Dict[str, Any]] = []
        self._last_stress_test: float = 0.0
        self._stress_results  : List[StressTestSummary] = []
        self._lock            = threading.Lock()

        # Lazy-loaded risk engines
        self._var_engine      : Optional[Any] = None
        self._stress_engine   : Optional[Any] = None
        self._limits_manager  : Optional[Any] = None

        logger.info(
            "EnterpriseRiskIntegrator: initialised | capital=%.0f | var_limit=%.1f%%",
            initial_capital, var_limit_pct * 100,
        )

    # ------------------------------------------------------------------ Lazy init

    def _init_var_engine(self) -> None:
        """Lazily initialise the real-time VaR aggregator."""
        if self._var_engine is not None:
            return
        try:
            from risk.realtime_var_aggregator import StreamingVaRCalculator
            self._var_engine = StreamingVaRCalculator(
                confidence_levels=[0.95, 0.99, 0.999],
                lookback_window=1000,
                decay_factor=0.94,
            )
            logger.info("EnterpriseRiskIntegrator: VaR engine initialised")
        except Exception as e:
            logger.warning("Failed to init VaR engine: %s", e)

    def _init_stress_engine(self) -> None:
        """Lazily initialise the stress test engine."""
        if self._stress_engine is not None:
            return
        try:
            from risk.stress_tester_enhanced import StressTestEngine
            self._stress_engine = StressTestEngine(
                portfolio=self._positions,
                random_seed=42,
            )
            logger.info("EnterpriseRiskIntegrator: stress engine initialised")
        except Exception as e:
            logger.warning("Failed to init stress engine: %s", e)

    def _init_limits_manager(self) -> None:
        """Lazily initialise the risk limits manager."""
        if self._limits_manager is not None:
            return
        try:
            from risk.risk_limits_manager import RiskLimitsManager
            self._limits_manager = RiskLimitsManager(
                initial_capital=self.initial_capital,
                config={
                    "max_position_usd": self.initial_capital * 0.20,
                    "max_total_exposure_usd": self.initial_capital * 0.80,
                    "max_daily_loss_usd": self.initial_capital * 0.03,
                    "max_drawdown_pct": 0.15,
                    "max_consecutive_losses": 10,
                },
            )
            logger.info("EnterpriseRiskIntegrator: limits manager initialised")
        except Exception as e:
            logger.warning("Failed to init limits manager: %s", e)

    # ------------------------------------------------------------------ Data ingestion

    def update_price(self, symbol: str, price: float) -> None:
        """Update price history for a symbol."""
        with self._lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=1000)
            self._price_history[symbol].append(price)

    def update_position(self, symbol: str, notional: float) -> None:
        """Update position notional for a symbol."""
        with self._lock:
            self._positions[symbol] = notional
            self._total_exposure = sum(abs(v) for v in self._positions.values())

    def update_equity(self, equity: float) -> None:
        """Update current equity and compute returns."""
        with self._lock:
            prev_equity = self._current_equity
            self._current_equity = equity
            self._portfolio_value = equity

            if equity > self._peak_equity:
                self._peak_equity = equity

            # Record return
            if prev_equity > 0:
                ret = (equity - prev_equity) / prev_equity
                self._returns_history.append(ret)

            # Feed to VaR engine
            if self._var_engine is not None:
                try:
                    self._var_engine.update_return(ret)
                except Exception:
                    pass

    def record_trade(self, symbol: str, pnl: float, side: str) -> None:
        """Record a trade for limits tracking."""
        self._init_limits_manager()
        if self._limits_manager is not None:
            try:
                self._limits_manager.update_equity(self._current_equity)
            except Exception:
                pass

    # ------------------------------------------------------------------ Risk queries

    def compute_snapshot(self) -> EnterpriseRiskSnapshot:
        """Compute full enterprise risk snapshot."""
        self._init_var_engine()

        var_95, var_99, cvar_95, cvar_99 = self._compute_var_metrics()
        dd_pct, max_dd_pct = self._compute_drawdown()
        leverage = self._compute_leverage()
        utilisation = var_95 / (self.initial_capital * self.var_limit_pct) \
            if self.var_limit_pct > 0 else 0.0

        # Asset-level risk breakdown
        asset_risks = self._compute_asset_risks()

        # Stress test (run periodically)
        worst_loss = 0.0
        worst_scenario = "none"
        now = time.time()
        if now - self._last_stress_test > self.stress_test_interval:
            self._run_stress_test()
            self._last_stress_test = now

        if self._stress_results:
            worst = min(self._stress_results, key=lambda s: s.loss_pct)
            worst_loss = worst.loss_pct
            worst_scenario = worst.scenario_name

        # Check for alerts
        alerts = self._check_alerts(var_95, utilisation, dd_pct)

        return EnterpriseRiskSnapshot(
            timestamp            = now,
            var_95               = var_95,
            var_99               = var_99,
            cvar_95              = cvar_95,
            cvar_99              = cvar_99,
            portfolio_value      = self._portfolio_value,
            total_exposure       = self._total_exposure,
            net_exposure         = sum(self._positions.values()),
            leverage             = leverage,
            current_drawdown_pct = dd_pct,
            max_drawdown_pct     = max_dd_pct,
            risk_budget_used_pct = min(100.0, utilisation * 100),
            var_limit_pct        = self.var_limit_pct * 100,
            var_utilisation      = utilisation,
            asset_risks          = asset_risks,
            worst_stress_loss    = worst_loss,
            worst_stress_scenario= worst_scenario,
            alerts               = alerts,
        )

    def _compute_var_metrics(self) -> Tuple[float, float, float, float]:
        """Compute VaR and CVaR metrics."""
        returns = list(self._returns_history)
        if len(returns) < 50:
            return 0.0, 0.0, 0.0, 0.0

        arr = np.array(returns)
        portfolio_value = max(1.0, self._portfolio_value)

        # Historical VaR
        var_95 = float(-np.percentile(arr, 5)) * portfolio_value
        var_99 = float(-np.percentile(arr, 1)) * portfolio_value

        # CVaR (Expected Shortfall)
        cvar_95_arr = arr[arr <= np.percentile(arr, 5)]
        cvar_99_arr = arr[arr <= np.percentile(arr, 1)]
        cvar_95 = float(-np.mean(cvar_95_arr)) * portfolio_value if len(cvar_95_arr) > 0 else var_95
        cvar_99 = float(-np.mean(cvar_99_arr)) * portfolio_value if len(cvar_99_arr) > 0 else var_99

        return var_95, var_99, cvar_95, cvar_99

    def _compute_drawdown(self) -> Tuple[float, float]:
        """Compute current and max drawdown percentages."""
        if self._peak_equity <= 0:
            return 0.0, 0.0
        current_dd = (self._current_equity - self._peak_equity) / self._peak_equity * 100
        return current_dd, current_dd  # max_dd tracked separately in full system

    def _compute_leverage(self) -> float:
        """Compute gross leverage."""
        if self._portfolio_value <= 0:
            return 0.0
        return self._total_exposure / self._portfolio_value

    def _compute_asset_risks(self) -> Dict[str, Dict[str, float]]:
        """Compute per-asset risk contribution."""
        result: Dict[str, Dict[str, float]] = {}
        if self._portfolio_value <= 0:
            return result

        for symbol, notional in self._positions.items():
            weight = abs(notional) / self._total_exposure if self._total_exposure > 0 else 0.0

            # Estimate vol from price history
            prices = self._price_history.get(symbol)
            vol = 0.0
            if prices and len(prices) > 10:
                price_arr = np.array(prices)
                log_rets = np.diff(np.log(price_arr))
                vol = float(np.std(log_rets) * np.sqrt(365 * 24))  # annualised

            # Contribution to portfolio VaR (simplified)
            contribution = weight * vol * self._portfolio_value

            result[symbol] = {
                "notional"    : abs(notional),
                "weight"      : weight,
                "annual_vol"  : vol,
                "var_contribution": contribution,
                "direction"   : "long" if notional > 0 else "short",
            }

        return result

    def _run_stress_test(self) -> None:
        """Run stress tests using the enhanced stress engine."""
        self._init_stress_engine()
        if self._stress_engine is None:
            return

        try:
            # Run each historical scenario
            scenarios = getattr(self._stress_engine, '_historical_scenarios', [])
            if not scenarios:
                scenarios = self._stress_engine.load_historical_scenarios()

            for scenario in scenarios:
                try:
                    result = self._stress_engine.run_stress_test(
                        portfolio=dict(self._positions),
                        scenario=scenario,
                    )
                    self._stress_results.append(StressTestSummary(
                        scenario_name  = result.scenario_name,
                        initial_value  = result.initial_portfolio_value,
                        stressed_value = result.stressed_portfolio_value,
                        loss_pct       = result.max_drawdown * 100,
                        recovery_days  = result.recovery_days,
                        timestamp      = time.time(),
                    ))
                except Exception as e:
                    logger.debug("Scenario %s failed: %s", scenario.name, e)

            # Keep only last 100 results
            self._stress_results = self._stress_results[-100:]

        except Exception as e:
            logger.warning("Stress test failed: %s", e)

    def _check_alerts(
        self,
        var_95: float,
        utilisation: float,
        dd_pct: float,
    ) -> List[Dict[str, Any]]:
        """Check risk thresholds and generate alerts."""
        alerts: List[Dict[str, Any]] = []
        now = time.time()

        # VaR limit breach
        if utilisation > 1.0:
            alerts.append({
                "type"    : "var_limit_breach",
                "severity": "critical",
                "message" : f"VaR utilisation {utilisation:.1%} exceeds limit",
                "value"   : utilisation,
                "threshold": 1.0,
                "ts"      : now,
            })
        elif utilisation > 0.8:
            alerts.append({
                "type"    : "var_limit_warning",
                "severity": "warning",
                "message" : f"VaR utilisation {utilisation:.1%} approaching limit",
                "value"   : utilisation,
                "threshold": 0.8,
                "ts"      : now,
            })

        # Drawdown alert
        if dd_pct < -10.0:
            alerts.append({
                "type"    : "drawdown_critical",
                "severity": "critical",
                "message" : f"Drawdown {dd_pct:.1f}% exceeds 10% threshold",
                "value"   : dd_pct,
                "threshold": -10.0,
                "ts"      : now,
            })
        elif dd_pct < -5.0:
            alerts.append({
                "type"    : "drawdown_warning",
                "severity": "warning",
                "message" : f"Drawdown {dd_pct:.1f}% exceeds 5% threshold",
                "value"   : dd_pct,
                "threshold": -5.0,
                "ts"      : now,
            })

        # Leverage alert
        leverage = self._compute_leverage()
        if leverage > 5.0:
            alerts.append({
                "type"    : "leverage_warning",
                "severity": "warning",
                "message" : f"Leverage {leverage:.1f}x exceeds 5x threshold",
                "value"   : leverage,
                "threshold": 5.0,
                "ts"      : now,
            })

        return alerts

    # ------------------------------------------------------------------ Pre-trade check

    def check_order(
        self,
        symbol    : str,
        side      : str,
        size_usd  : float,
    ) -> Tuple[bool, str]:
        """
        Pre-trade risk check. Returns (allowed, reason).
        """
        self._init_limits_manager()

        if self._limits_manager is not None:
            try:
                result = self._limits_manager.check_order(symbol, side, size_usd)
                if not result.allow:
                    return False, f"Risk limits: {', '.join(result.failing)}"
            except Exception as e:
                logger.warning("Limits check failed: %s", e)

        # Additional VaR-based check
        var_95, _, _, _ = self._compute_var_metrics()
        utilisation = var_95 / (self.initial_capital * self.var_limit_pct) \
            if self.var_limit_pct > 0 else 0.0

        if utilisation > 0.95:
            return False, f"VaR utilisation {utilisation:.1%} too high to add risk"

        return True, "OK"

    # ------------------------------------------------------------------ Dashboard integration

    def feed_to_dashboard(self, dashboard_state: Any) -> None:
        """
        Push current risk snapshot to the operator dashboard state.
        Call this periodically or after significant events.
        """
        snapshot = self.compute_snapshot()

        # Update dashboard state
        dashboard_state.var_95    = snapshot.var_95
        dashboard_state.cvar_95   = snapshot.cvar_95
        dashboard_state.max_dd_pct= snapshot.max_drawdown_pct
        dashboard_state.cur_dd_pct= snapshot.current_drawdown_pct
        dashboard_state.position_exposure = snapshot.total_exposure
        dashboard_state.leverage  = snapshot.leverage

        # Store alerts for dashboard
        if hasattr(dashboard_state, '_risk_alerts'):
            dashboard_state._risk_alerts = snapshot.alerts

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary dict for quick dashboard updates."""
        snap = self.compute_snapshot()
        return {
            "var_95_usd"           : snap.var_95,
            "var_99_usd"           : snap.var_99,
            "cvar_95_usd"          : snap.cvar_95,
            "cvar_99_usd"          : snap.cvar_99,
            "portfolio_value"      : snap.portfolio_value,
            "total_exposure"       : snap.total_exposure,
            "leverage"             : snap.leverage,
            "drawdown_pct"         : snap.current_drawdown_pct,
            "max_drawdown_pct"     : snap.max_drawdown_pct,
            "var_utilisation_pct"  : snap.var_utilisation * 100,
            "risk_budget_used_pct" : snap.risk_budget_used_pct,
            "n_alerts"             : len(snap.alerts),
            "worst_stress_loss_pct": snap.worst_stress_loss,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_integrator: Optional[EnterpriseRiskIntegrator] = None


def get_enterprise_risk(
    initial_capital: float = 1_000_000.0,
    var_limit_pct  : float = 0.02,
) -> EnterpriseRiskIntegrator:
    """Get or create the enterprise risk integrator singleton."""
    global _integrator
    if _integrator is None:
        _integrator = EnterpriseRiskIntegrator(
            initial_capital=initial_capital,
            var_limit_pct=var_limit_pct,
        )
    return _integrator
