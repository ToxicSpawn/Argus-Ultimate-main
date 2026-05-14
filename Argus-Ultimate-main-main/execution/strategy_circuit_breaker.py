"""
Per-Strategy Circuit Breaker — independent kill switches for trading strategies.

State machine
-------------
    CLOSED  →  normal operation, all checks passing
    OPEN    →  tripped; strategy is halted; sends kill signal
    HALF_OPEN → recovery test window (recovery_period_s has elapsed)

Trip conditions (any one suffices to trip CLOSED → OPEN):
    1. Session drawdown > max_drawdown_pct
    2. Consecutive losses > max_consecutive_losses
    3. Order rate in last 60 s > max_order_rate_per_min
    4. Position size > max_position_usd (checked via caller)

Recovery:
    After recovery_period_s seconds, OPEN → HALF_OPEN.
    A manual `reset()` or one profitable trade in HALF_OPEN → CLOSED.

Usage
-----
    panel = StrategyBreakerPanel()
    panel.register("my_strat", BreakerConfig(max_drawdown_pct=3.0))
    breaker = panel.get("my_strat")
    breaker.record_trade(pnl=-200.0, timestamp_ns=time.time_ns())
    state = breaker.check()
    if state == BreakerState.OPEN:
        ... halt strategy ...
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BreakerState(Enum):
    CLOSED    = "CLOSED"     # running normally
    OPEN      = "OPEN"       # tripped — strategy halted
    HALF_OPEN = "HALF_OPEN"  # testing recovery


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BreakerConfig:
    """Configuration parameters for a single circuit breaker."""
    max_drawdown_pct: float        = 5.0      # session drawdown %
    max_consecutive_losses: int    = 10       # consecutive losing trades
    max_order_rate_per_min: int    = 300      # orders per minute
    max_position_usd: float        = 10000.0  # absolute position in USD
    recovery_period_s: float       = 300.0    # seconds before HALF_OPEN
    loss_window_s: float           = 300.0    # rolling window for loss counting
    peak_pnl_seed: float           = 0.0      # starting peak for drawdown calc


# ---------------------------------------------------------------------------
# Per-strategy breaker
# ---------------------------------------------------------------------------

class StrategyCircuitBreaker:
    """
    Circuit breaker for a single strategy.

    Thread-safe: all methods are synchronous (no async I/O).

    Parameters
    ----------
    strategy_name : str
    config : BreakerConfig
    """

    def __init__(self, strategy_name: str, config: BreakerConfig) -> None:
        self.strategy_name = strategy_name
        self.config = config

        self._state = BreakerState.CLOSED
        self._trip_reason: str = ""
        self._trip_time_ns: int = 0

        # PnL tracking
        self._session_pnl: float = 0.0
        self._peak_pnl: float = config.peak_pnl_seed
        self._max_drawdown_seen: float = 0.0
        self._last_5_trades: Deque[float] = deque(maxlen=5)

        # Consecutive losses
        self._consecutive_losses: int = 0

        # Trade log: (timestamp_ns, pnl)
        self._trade_log: List[Tuple[int, float]] = []

        # Order rate tracking: ring buffer of order timestamps (ns)
        self._order_ts: Deque[int] = deque()

        # Error / rejection counters
        self._total_orders: int = 0
        self._rejected_orders: int = 0
        self._error_orders: int = 0

        logger.info(
            "StrategyCircuitBreaker initialised: strategy=%s config=%s",
            strategy_name, config
        )

    # ------------------------------------------------------------------
    # Record events
    # ------------------------------------------------------------------

    def record_trade(self, pnl: float, timestamp_ns: int) -> None:
        """
        Register a completed trade.

        Updates session PnL, peak PnL, drawdown, and consecutive-loss counter.
        """
        self._session_pnl += pnl
        self._trade_log.append((timestamp_ns, pnl))
        self._last_5_trades.append(pnl)

        if self._session_pnl > self._peak_pnl:
            self._peak_pnl = self._session_pnl

        if pnl < 0.0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        drawdown = self._compute_drawdown()
        if drawdown > self._max_drawdown_seen:
            self._max_drawdown_seen = drawdown

        # If HALF_OPEN: a profitable trade restores to CLOSED
        if self._state == BreakerState.HALF_OPEN and pnl > 0:
            logger.info(
                "Strategy %s: profitable trade in HALF_OPEN; restoring to CLOSED",
                self.strategy_name,
            )
            self._state = BreakerState.CLOSED
            self._trip_reason = ""

    def record_order(
        self,
        is_rejected: bool = False,
        is_error: bool = False,
    ) -> None:
        """Register an order submission event."""
        now_ns = time.time_ns()
        self._order_ts.append(now_ns)
        self._total_orders += 1
        if is_rejected:
            self._rejected_orders += 1
        if is_error:
            self._error_orders += 1

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def check(self) -> BreakerState:
        """
        Evaluate all trip conditions.

        If any condition is violated, transition CLOSED → OPEN.
        If OPEN and recovery_period_s has elapsed, transition → HALF_OPEN.
        Returns current state after evaluation.
        """
        if self._state == BreakerState.OPEN:
            elapsed_s = (time.time_ns() - self._trip_time_ns) / 1e9
            if elapsed_s >= self.config.recovery_period_s:
                self._state = BreakerState.HALF_OPEN
                logger.info(
                    "Strategy %s: OPEN → HALF_OPEN after %.1f s",
                    self.strategy_name, elapsed_s
                )
            return self._state

        if self._state == BreakerState.HALF_OPEN:
            return self._state

        # CLOSED: evaluate all conditions
        reason = self._evaluate_conditions()
        if reason:
            self._trip(reason)

        return self._state

    # ------------------------------------------------------------------
    # Trip / reset
    # ------------------------------------------------------------------

    def _trip(self, reason: str) -> None:
        """Internal trip: CLOSED → OPEN."""
        self._state = BreakerState.OPEN
        self._trip_reason = reason
        self._trip_time_ns = time.time_ns()
        last5 = list(self._last_5_trades)
        logger.warning(
            "CIRCUIT BREAKER TRIPPED: strategy=%s reason='%s' "
            "session_pnl=%.2f consecutive_losses=%d last_5_trades=%s",
            self.strategy_name,
            reason,
            self._session_pnl,
            self._consecutive_losses,
            last5,
        )

    def reset(self) -> None:
        """Manual reset: any state → CLOSED."""
        previous = self._state
        self._state = BreakerState.CLOSED
        self._trip_reason = ""
        self._trip_time_ns = 0
        self._consecutive_losses = 0
        logger.info(
            "Strategy %s: manual reset %s → CLOSED", self.strategy_name, previous.value
        )

    # ------------------------------------------------------------------
    # Condition evaluation
    # ------------------------------------------------------------------

    def _compute_drawdown(self) -> float:
        """Session drawdown as percentage of peak PnL (clamped ≥ 0)."""
        if self._peak_pnl <= 0:
            return 0.0
        dd = (self._peak_pnl - self._session_pnl) / abs(self._peak_pnl) * 100.0
        return max(dd, 0.0)

    def _order_rate_per_min(self) -> float:
        """Orders per minute over the last 60 seconds."""
        now_ns = time.time_ns()
        cutoff_ns = now_ns - 60 * 1_000_000_000
        while self._order_ts and self._order_ts[0] < cutoff_ns:
            self._order_ts.popleft()
        return len(self._order_ts)  # already a per-minute count (60 s window)

    def _evaluate_conditions(self) -> str:
        """
        Evaluate all trip conditions.  Returns reason string if any triggered,
        empty string otherwise.

        Evaluation order:
            1. Consecutive losses  (discrete, clear signal)
            2. Order rate          (rate limiter)
            3. Drawdown            (requires meaningful session PnL history)
        """
        # Condition 1: consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            return (
                f"consecutive losses {self._consecutive_losses} "
                f">= limit {self.config.max_consecutive_losses}"
            )

        # Condition 2: order rate
        rate = self._order_rate_per_min()
        if rate > self.config.max_order_rate_per_min:
            return (
                f"order rate {rate:.0f}/min exceeds limit "
                f"{self.config.max_order_rate_per_min}/min"
            )

        # Condition 3: drawdown (only meaningful once some trades have occurred)
        if len(self._trade_log) >= 1:
            dd = self._compute_drawdown()
            if dd > self.config.max_drawdown_pct:
                return (
                    f"drawdown {dd:.2f}% exceeds limit {self.config.max_drawdown_pct}%"
                )

        return ""

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return full status snapshot."""
        drawdown = self._compute_drawdown()
        order_rate = self._order_rate_per_min()
        trip_time_s = (
            self._trip_time_ns / 1e9
            if self._trip_time_ns
            else None
        )
        time_since_trip_s: Optional[float] = None
        if self._trip_time_ns:
            time_since_trip_s = (time.time_ns() - self._trip_time_ns) / 1e9

        return {
            "strategy": self.strategy_name,
            "state": self._state.value,
            "trip_reason": self._trip_reason,
            "trip_time_epoch_s": trip_time_s,
            "time_since_trip_s": time_since_trip_s,
            "session_pnl": self._session_pnl,
            "peak_pnl": self._peak_pnl,
            "drawdown_pct": drawdown,
            "max_drawdown_pct": self._max_drawdown_seen,
            "consecutive_losses": self._consecutive_losses,
            "order_rate_per_min": order_rate,
            "total_orders": self._total_orders,
            "rejected_orders": self._rejected_orders,
            "error_orders": self._error_orders,
            "last_5_trades": list(self._last_5_trades),
        }


# ---------------------------------------------------------------------------
# Breaker panel
# ---------------------------------------------------------------------------

_DEFAULT_STRATEGIES: List[Tuple[str, BreakerConfig]] = [
    ("market_maker",     BreakerConfig(max_drawdown_pct=5.0,  max_consecutive_losses=10, max_order_rate_per_min=300)),
    ("cross_venue_arb",  BreakerConfig(max_drawdown_pct=3.0,  max_consecutive_losses=8,  max_order_rate_per_min=200)),
    ("hft_scalping",     BreakerConfig(max_drawdown_pct=2.0,  max_consecutive_losses=15, max_order_rate_per_min=500)),
    ("void_breaker",     BreakerConfig(max_drawdown_pct=10.0, max_consecutive_losses=20, max_order_rate_per_min=150)),
]


class StrategyBreakerPanel:
    """
    Panel managing circuit breakers for all registered strategies.

    Pre-registers the default Argus strategies:
        market_maker, cross_venue_arb, hft_scalping, void_breaker
    """

    def __init__(self) -> None:
        self._breakers: Dict[str, StrategyCircuitBreaker] = {}

        # Auto-register defaults
        for name, cfg in _DEFAULT_STRATEGIES:
            self.register(name, cfg)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        strategy_name: str,
        config: Optional[BreakerConfig] = None,
    ) -> StrategyCircuitBreaker:
        """
        Register a new strategy circuit breaker.

        If config is None, default BreakerConfig is used.
        Replaces existing breaker if strategy_name already registered.
        """
        cfg = config or BreakerConfig()
        breaker = StrategyCircuitBreaker(strategy_name, cfg)
        self._breakers[strategy_name] = breaker
        logger.info("BreakerPanel: registered strategy '%s'", strategy_name)
        return breaker

    def get(self, strategy_name: str) -> StrategyCircuitBreaker:
        """Return breaker for strategy_name; raises KeyError if not registered."""
        if strategy_name not in self._breakers:
            raise KeyError(f"Strategy '{strategy_name}' not registered in BreakerPanel")
        return self._breakers[strategy_name]

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def check_all(self) -> Dict[str, BreakerState]:
        """Evaluate all registered breakers and return a name → state map."""
        return {name: b.check() for name, b in self._breakers.items()}

    def trip_all(self, reason: str) -> None:
        """
        Emergency: forcibly trip ALL registered breakers to OPEN.

        Use for market-wide risk events (e.g. flash crash, connectivity loss).
        """
        logger.critical("BreakerPanel.trip_all: reason='%s'", reason)
        for name, breaker in self._breakers.items():
            if breaker._state != BreakerState.OPEN:
                breaker._trip(f"[PANEL TRIP_ALL] {reason}")

    def any_open(self) -> bool:
        """Return True if at least one registered breaker is OPEN."""
        return any(b._state == BreakerState.OPEN for b in self._breakers.values())

    def reset_all(self) -> None:
        """Reset all breakers to CLOSED."""
        for b in self._breakers.values():
            b.reset()

    # ------------------------------------------------------------------
    # Panel status
    # ------------------------------------------------------------------

    def get_panel_status(self) -> dict:
        """Return status snapshot for all registered strategies."""
        strategies = {name: b.get_status() for name, b in self._breakers.items()}
        open_count = sum(
            1 for b in self._breakers.values() if b._state == BreakerState.OPEN
        )
        half_open_count = sum(
            1 for b in self._breakers.values() if b._state == BreakerState.HALF_OPEN
        )
        return {
            "total_strategies": len(self._breakers),
            "open_count": open_count,
            "half_open_count": half_open_count,
            "all_clear": open_count == 0 and half_open_count == 0,
            "strategies": strategies,
        }

    def __len__(self) -> int:
        return len(self._breakers)

    def __contains__(self, name: str) -> bool:
        return name in self._breakers
