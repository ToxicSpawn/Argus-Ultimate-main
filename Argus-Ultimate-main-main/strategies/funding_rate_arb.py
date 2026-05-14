"""
Funding Rate Arbitrage Strategy — Delta-Neutral Carry Trade
============================================================

Captures perpetual futures funding payments by simultaneously holding:
  - Long spot (spot leg)
  - Short perpetual (perp leg)

Net delta = 0, so price moves cancel out. When funding rate is positive
(longs pay shorts), the short perp leg collects funding every 8 hours.
When the rate is negative, we either reverse legs or skip the symbol.

Architecture:
  - FundingRate: Data model for funding rate snapshots
  - FundingRateTracker: Historical tracking and prediction
  - BasisCalculator: Basis and yield computations
  - FundingRateArbitrage: Opportunity discovery and sizing
  - ArbPosition: Position tracking dataclass
  - FundingArbExecutor: Position lifecycle management
  - RiskManager: Risk limit enforcement
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FUNDING_INTERVAL_HOURS = 8
_DEFAULT_FUNDING_PERIODS_PER_DAY = 3.0
_DEFAULT_DAYS_PER_YEAR = 365.0

# Risk defaults
_DEFAULT_MAX_POSITION_SIZE = 100_000.0
_DEFAULT_MAX_FUNDING_EXPOSURE = 50_000.0
_DEFAULT_STOP_LOSS_RATE = -0.0005

# Opportunity thresholds
_DEFAULT_MIN_ANNUALIZED_YIELD = 0.10
_DEFAULT_RISK_PER_TRADE = 0.02

# Rate history defaults
_DEFAULT_HISTORY_MAXLEN = 1080
_DEFAULT_PREDICTION_PERIODS = 3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FundingRate:
    """Snapshot of a funding rate for a specific exchange and symbol."""

    exchange: str
    symbol: str
    rate: float
    timestamp: datetime
    next_funding_time: datetime
    predicted_rate: float = 0.0

    @property
    def annualized_rate(self) -> float:
        return self.rate * _DEFAULT_FUNDING_PERIODS_PER_DAY * _DEFAULT_DAYS_PER_YEAR

    def __repr__(self) -> str:
        return (
            f"FundingRate(exchange={self.exchange!r}, symbol={self.symbol!r}, "
            f"rate={self.rate:.6f}, annualized={self.annualized_rate:.4f})"
        )


@dataclass
class ArbOpportunity:
    """A funding rate arbitrage opportunity."""

    exchange: str
    symbol: str
    current_rate: float
    predicted_rate: float
    annualized_yield: float
    basis: float
    spot_price: float
    perp_price: float
    next_funding_time: datetime
    confidence: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ArbOpportunity(symbol={self.symbol!r}, exchange={self.exchange!r}, "
            f"annualized_yield={self.annualized_yield:.4f}, rate={self.current_rate:.6f})"
        )


@dataclass
class ArbPosition:
    """Tracks a single funding arbitrage position."""

    symbol: str
    exchange: str
    spot_position: float
    perp_position: float
    entry_funding_rate: float
    entry_time: datetime
    expected_exit_time: datetime

    realized_pnl: float = 0.0
    funding_collected: float = 0.0
    funding_payments_received: int = 0
    status: str = "open"

    def age_hours(self, now: Optional[datetime] = None) -> float:
        ref = now if now is not None else datetime.now(timezone.utc)
        return (ref - self.entry_time).total_seconds() / 3600.0

    def periods_elapsed(self, now: Optional[datetime] = None) -> int:
        hours = self.age_hours(now)
        return int(hours / _DEFAULT_FUNDING_INTERVAL_HOURS)

    def __repr__(self) -> str:
        return (
            f"ArbPosition(symbol={self.symbol!r}, exchange={self.exchange!r}, "
            f"spot={self.spot_position:.6f}, perp={self.perp_position:.6f}, "
            f"entry_rate={self.entry_funding_rate:.6f}, status={self.status!r})"
        )


@dataclass
class PnLResult:
    """Result of closing a position."""

    symbol: str
    exchange: str
    realized_pnl: float
    funding_collected: float
    holding_periods: int
    holding_hours: float
    entry_rate: float
    exit_rate: float
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PositionUpdate:
    """Periodic update for an open position."""

    symbol: str
    exchange: str
    current_rate: float
    funding_collected: float
    unrealized_pnl: float
    delta_drift: float
    periods_remaining: int
    status: str


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_size: float = _DEFAULT_MAX_POSITION_SIZE
    max_funding_exposure: float = _DEFAULT_MAX_FUNDING_EXPOSURE
    stop_loss_rate: float = _DEFAULT_STOP_LOSS_RATE
    max_concurrent_positions: int = 5
    max_drawdown_pct: float = 0.05
    max_single_loss_pct: float = 0.02


# ---------------------------------------------------------------------------
# FundingRateTracker
# ---------------------------------------------------------------------------

class FundingRateTracker:
    """
    Tracks funding rate history and provides prediction capabilities.

    Maintains per-(exchange, symbol) rate histories with configurable
    lookback windows and computes rolling averages and predictions.
    """

    def __init__(self, maxlen: int = _DEFAULT_HISTORY_MAXLEN) -> None:
        self._maxlen = maxlen
        self._history: Dict[str, Deque[FundingRate]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._current_rates: Dict[str, FundingRate] = {}

    @staticmethod
    def _key(exchange: str, symbol: str) -> str:
        return f"{exchange}:{symbol.upper()}"

    def track_rate(
        self,
        exchange: str,
        symbol: str,
        rate: float,
        timestamp: Optional[datetime] = None,
        next_funding_time: Optional[datetime] = None,
        predicted_rate: float = 0.0,
    ) -> FundingRate:
        """Record a new funding rate observation."""
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
        if next_funding_time is None:
            next_funding_time = ts + timedelta(hours=_DEFAULT_FUNDING_INTERVAL_HOURS)

        fr = FundingRate(
            exchange=exchange,
            symbol=symbol.upper(),
            rate=rate,
            timestamp=ts,
            next_funding_time=next_funding_time,
            predicted_rate=predicted_rate,
        )

        key = self._key(exchange, symbol)
        self._history[key].append(fr)
        self._current_rates[key] = fr

        logger.debug(
            "Tracked rate: %s %s rate=%.6f predicted=%.6f",
            exchange, symbol.upper(), rate, predicted_rate,
        )
        return fr

    def get_current_rate(self, exchange: str, symbol: str) -> Optional[FundingRate]:
        """Return the most recent funding rate for the given exchange/symbol."""
        key = self._key(exchange, symbol)
        return self._current_rates.get(key)

    def get_rate_history(
        self,
        exchange: str,
        symbol: str,
        hours: float = 24.0,
    ) -> List[FundingRate]:
        """Return funding rates within the last N hours."""
        key = self._key(exchange, symbol)
        history = self._history.get(key, deque())
        if not history:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [fr for fr in history if fr.timestamp >= cutoff]

    def compute_average_rate(
        self,
        exchange: str,
        symbol: str,
        periods: int = _DEFAULT_PREDICTION_PERIODS,
    ) -> float:
        """Compute the average funding rate over the last N periods."""
        key = self._key(exchange, symbol)
        history = list(self._history.get(key, deque()))
        if not history:
            return 0.0

        recent = history[-periods:] if len(history) >= periods else history
        return sum(fr.rate for fr in recent) / len(recent)

    def predict_next_rate(self, exchange: str, symbol: str) -> float:
        """
        Predict the next funding rate using weighted moving average.

        Uses exponential weighting: recent observations get higher weight.
        Falls back to the last known predicted_rate if available.
        """
        key = self._key(exchange, symbol)
        history = list(self._history.get(key, deque()))

        if not history:
            current = self._current_rates.get(key)
            return current.predicted_rate if current else 0.0

        current = self._current_rates.get(key)
        if current and current.predicted_rate != 0.0:
            model_pred = current.predicted_rate
        else:
            model_pred = None

        ema = self._compute_ema([fr.rate for fr in history], span=min(12, len(history)))

        if model_pred is not None:
            return 0.6 * ema + 0.4 * model_pred
        return ema

    @staticmethod
    def _compute_ema(values: List[float], span: int = 12) -> float:
        """Compute exponential moving average."""
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]

        multiplier = 2.0 / (span + 1)
        ema = values[0]
        for val in values[1:]:
            ema = val * multiplier + ema * (1 - multiplier)
        return ema

    def get_all_symbols(self) -> List[Tuple[str, str]]:
        """Return list of (exchange, symbol) pairs being tracked."""
        return [key.split(":", 1) for key in self._current_rates]


# ---------------------------------------------------------------------------
# BasisCalculator
# ---------------------------------------------------------------------------

class BasisCalculator:
    """
    Computes basis, annualized basis, and implied yield between
    perpetual futures and spot prices.
    """

    @staticmethod
    def compute_basis(perp_price: float, spot_price: float) -> float:
        """
        Compute the basis (premium/discount) of perp relative to spot.

        basis = (perp_price - spot_price) / spot_price

        Positive basis → perp trades at premium (contango)
        Negative basis → perp trades at discount (backwardation)
        """
        if spot_price <= 0:
            logger.warning("compute_basis: spot_price=%.6f <= 0, returning 0", spot_price)
            return 0.0
        return (perp_price - spot_price) / spot_price

    @staticmethod
    def compute_annualized_basis(
        basis: float,
        funding_interval_hours: float = _DEFAULT_FUNDING_INTERVAL_HOURS,
    ) -> float:
        """
        Annualize the basis to compare with funding rates.

        annualized_basis = basis * (24 / interval) * 365
        """
        periods_per_day = 24.0 / max(funding_interval_hours, 1.0)
        return basis * periods_per_day * _DEFAULT_DAYS_PER_YEAR

    @staticmethod
    def compute_implied_yield(basis: float, funding_rate: float) -> float:
        """
        Compute the implied yield from basis and funding rate.

        implied_yield = annualized_basis + annualized_funding_rate

        This represents the total expected return from a delta-neutral
        position that captures both basis convergence and funding payments.
        """
        annualized_basis = BasisCalculator.compute_annualized_basis(basis)
        annualized_funding = (
            funding_rate * _DEFAULT_FUNDING_PERIODS_PER_DAY * _DEFAULT_DAYS_PER_YEAR
        )
        return annualized_basis + annualized_funding


# ---------------------------------------------------------------------------
# FundingRateArbitrage
# ---------------------------------------------------------------------------

class FundingRateArbitrage:
    """
    Discovers and evaluates funding rate arbitrage opportunities.

    Scans tracked funding rates to find symbols where the annualized
    yield exceeds a minimum threshold, then computes optimal position
    sizes and expected returns.
    """

    def __init__(
        self,
        tracker: FundingRateTracker,
        basis_calculator: Optional[BasisCalculator] = None,
        min_annualized_yield: float = _DEFAULT_MIN_ANNUALIZED_YIELD,
    ) -> None:
        self.tracker = tracker
        self.basis_calculator = basis_calculator or BasisCalculator()
        self.min_annualized_yield = min_annualized_yield

        self._spot_prices: Dict[str, float] = {}
        self._perp_prices: Dict[str, float] = {}

    def update_prices(
        self,
        exchange: str,
        symbol: str,
        spot_price: float,
        perp_price: float,
    ) -> None:
        """Cache current spot and perp prices for a symbol."""
        key = f"{exchange}:{symbol.upper()}"
        self._spot_prices[key] = spot_price
        self._perp_prices[key] = perp_price

    def find_opportunities(
        self,
        min_annualized_yield: Optional[float] = None,
    ) -> List[ArbOpportunity]:
        """
        Scan all tracked symbols for funding arbitrage opportunities.

        Returns opportunities sorted by annualized yield (descending).
        """
        threshold = (
            min_annualized_yield
            if min_annualized_yield is not None
            else self.min_annualized_yield
        )

        opportunities: List[ArbOpportunity] = []

        for exchange, symbol in self.tracker.get_all_symbols():
            current = self.tracker.get_current_rate(exchange, symbol)
            if current is None:
                continue

            predicted = self.tracker.predict_next_rate(exchange, symbol)
            annualized = current.annualized_rate

            if annualized < threshold:
                continue

            key = f"{exchange}:{symbol}"
            spot_price = self._spot_prices.get(key, 0.0)
            perp_price = self._perp_prices.get(key, 0.0)

            basis = 0.0
            if spot_price > 0 and perp_price > 0:
                basis = self.basis_calculator.compute_basis(perp_price, spot_price)

            confidence = self._compute_confidence(current, predicted, annualized, threshold)

            opp = ArbOpportunity(
                exchange=exchange,
                symbol=symbol,
                current_rate=current.rate,
                predicted_rate=predicted,
                annualized_yield=annualized,
                basis=basis,
                spot_price=spot_price,
                perp_price=perp_price,
                next_funding_time=current.next_funding_time,
                confidence=confidence,
            )
            opportunities.append(opp)

        opportunities.sort(key=lambda o: o.annualized_yield, reverse=True)

        logger.info(
            "Found %d funding arb opportunities (threshold=%.4f)",
            len(opportunities), threshold,
        )
        return opportunities

    def compute_position_size(
        self,
        capital: float,
        risk_per_trade: float = _DEFAULT_RISK_PER_TRADE,
        funding_rate: float = 0.0,
    ) -> float:
        """
        Compute optimal position size based on capital and risk parameters.

        Uses Kelly-inspired sizing: position = capital * risk_per_trade * rate_factor
        where rate_factor scales with the funding rate magnitude.
        """
        if capital <= 0:
            return 0.0
        if funding_rate == 0.0:
            return capital * risk_per_trade

        rate_factor = min(abs(funding_rate) * 1000.0, 2.0)
        position = capital * risk_per_trade * rate_factor

        logger.debug(
            "Position size: capital=%.2f risk=%.4f rate=%.6f size=%.2f",
            capital, risk_per_trade, funding_rate, position,
        )
        return position

    def calculate_expected_return(
        self,
        entry_funding: float,
        holding_periods: int,
    ) -> float:
        """
        Calculate expected return from holding a position for N funding periods.

        expected_return = entry_funding * holding_periods
        """
        return entry_funding * holding_periods

    def calculate_funding_pnl(
        self,
        position_size: float,
        funding_rate: float,
        periods: int,
    ) -> float:
        """
        Calculate the P&L from funding payments.

        funding_pnl = position_size * funding_rate * periods
        """
        return position_size * funding_rate * periods

    @staticmethod
    def _compute_confidence(
        current: FundingRate,
        predicted: float,
        annualized: float,
        threshold: float,
    ) -> float:
        """Compute opportunity confidence score [0, 1]."""
        if annualized <= threshold:
            return 0.0

        rate_confidence = min((annualized - threshold) / threshold, 1.0) * 0.5

        prediction_agreement = 0.0
        if predicted != 0.0:
            agreement = 1.0 - abs(current.rate - predicted) / max(abs(current.rate), 1e-10)
            prediction_agreement = max(0.0, agreement) * 0.3

        history_confidence = 0.2

        return min(1.0, rate_confidence + prediction_agreement + history_confidence)


# ---------------------------------------------------------------------------
# FundingArbExecutor
# ---------------------------------------------------------------------------

class FundingArbExecutor:
    """
    Manages the lifecycle of funding arbitrage positions.

    Executes new arb trades, monitors open positions, and handles
    position closing when conditions warrant.
    """

    def __init__(
        self,
        tracker: FundingRateTracker,
        risk_manager: Optional["RiskManager"] = None,
    ) -> None:
        self.tracker = tracker
        self.risk_manager = risk_manager or RiskManager()
        self._positions: Dict[str, ArbPosition] = {}
        self._closed_positions: List[PnLResult] = []

    @staticmethod
    def _position_key(symbol: str, exchange: str) -> str:
        return f"{exchange}:{symbol.upper()}"

    def execute_arb(
        self,
        opportunity: ArbOpportunity,
        capital: float,
    ) -> Optional[ArbPosition]:
        """
        Execute a funding rate arbitrage trade.

        Opens a delta-neutral position:
          - Long spot (spot_position > 0)
          - Short perp (perp_position < 0)

        Returns the created ArbPosition or None if execution fails.
        """
        key = self._position_key(opportunity.symbol, opportunity.exchange)

        if key in self._positions:
            existing = self._positions[key]
            if existing.status == "open":
                logger.warning(
                    "Position already open: %s %s", opportunity.exchange, opportunity.symbol
                )
                return None

        if capital <= 0:
            logger.warning("execute_arb: invalid capital=%.2f", capital)
            return None

        spot_size = capital / max(opportunity.spot_price, 1e-10)
        perp_size = -capital / max(opportunity.perp_price, 1e-10)

        now = datetime.now(timezone.utc)
        expected_exit = now + timedelta(
            hours=_DEFAULT_FUNDING_INTERVAL_HOURS * 3
        )

        position = ArbPosition(
            symbol=opportunity.symbol.upper(),
            exchange=opportunity.exchange,
            spot_position=spot_size,
            perp_position=perp_size,
            entry_funding_rate=opportunity.current_rate,
            entry_time=now,
            expected_exit_time=expected_exit,
            status="open",
        )

        if not self.risk_manager.check_risk_limits(position):
            logger.warning(
                "Risk limits violated for %s %s — position rejected",
                opportunity.exchange, opportunity.symbol,
            )
            return None

        self._positions[key] = position

        logger.info(
            "Executed arb: %s %s spot=%.6f perp=%.6f rate=%.6f",
            opportunity.exchange, opportunity.symbol,
            spot_size, perp_size, opportunity.current_rate,
        )
        return position

    def close_position(
        self,
        position: ArbPosition,
        exit_rate: Optional[float] = None,
    ) -> PnLResult:
        """
        Close an open arbitrage position and compute final P&L.
        """
        key = self._position_key(position.symbol, position.exchange)
        stored = self._positions.get(key)

        if stored is None or stored.status != "open":
            logger.warning("close_position: position not found or not open: %s", key)
            return PnLResult(
                symbol=position.symbol,
                exchange=position.exchange,
                realized_pnl=0.0,
                funding_collected=0.0,
                holding_periods=0,
                holding_hours=0.0,
                entry_rate=position.entry_funding_rate,
                exit_rate=exit_rate or 0.0,
            )

        current_rate = exit_rate if exit_rate is not None else self._get_current_rate(
            position.exchange, position.symbol
        )

        periods = stored.periods_elapsed()
        hours = stored.age_hours()

        final_funding_pnl = self._calculate_position_pnl(stored, current_rate or 0.0)
        stored.realized_pnl = final_funding_pnl
        stored.status = "closed"

        result = PnLResult(
            symbol=position.symbol,
            exchange=position.exchange,
            realized_pnl=final_funding_pnl,
            funding_collected=stored.funding_collected,
            holding_periods=periods,
            holding_hours=hours,
            entry_rate=stored.entry_funding_rate,
            exit_rate=current_rate or 0.0,
        )

        self._closed_positions.append(result)

        logger.info(
            "Closed position: %s %s pnl=%.4f funding=%.4f periods=%d",
            position.exchange, position.symbol,
            final_funding_pnl, stored.funding_collected, periods,
        )
        return result

    def monitor_positions(
        self,
        now: Optional[datetime] = None,
    ) -> List[PositionUpdate]:
        """
        Monitor all open positions and return status updates.
        """
        updates: List[PositionUpdate] = []

        for key, pos in list(self._positions.items()):
            if pos.status != "open":
                continue

            current_rate = self._get_current_rate(pos.exchange, pos.symbol) or 0.0
            periods_remaining = max(
                0,
                int(
                    (pos.expected_exit_time - (now or datetime.now(timezone.utc))).total_seconds()
                    / (_DEFAULT_FUNDING_INTERVAL_HOURS * 3600)
                ),
            )

            unrealized = self._calculate_unrealized_pnl(pos, current_rate)
            delta_drift = self._compute_delta_drift(pos)

            update = PositionUpdate(
                symbol=pos.symbol,
                exchange=pos.exchange,
                current_rate=current_rate,
                funding_collected=pos.funding_collected,
                unrealized_pnl=unrealized,
                delta_drift=delta_drift,
                periods_remaining=periods_remaining,
                status=pos.status,
            )
            updates.append(update)

        return updates

    def should_close(
        self,
        position: ArbPosition,
        current_rate: Optional[float] = None,
    ) -> bool:
        """
        Determine if a position should be closed.

        Closes if:
        - Current rate has crossed below stop_loss_rate
        - Position has reached expected exit time
        - Rate has reversed sign (funding turned against position)
        """
        if position.status != "open":
            return False

        rate = (
            current_rate
            if current_rate is not None
            else self._get_current_rate(position.exchange, position.symbol)
        )
        if rate is None:
            return False

        if rate <= self.risk_manager.stop_loss_rate:
            logger.warning(
                "Stop loss triggered: %s %s rate=%.6f <= stop=%.6f",
                position.exchange, position.symbol, rate,
                self.risk_manager.stop_loss_rate,
            )
            return True

        if datetime.now(timezone.utc) >= position.expected_exit_time:
            logger.info(
                "Expected exit reached: %s %s", position.exchange, position.symbol
            )
            return True

        if position.entry_funding_rate > 0 and rate < 0:
            logger.info(
                "Rate reversal: %s %s entered at %.6f now %.6f",
                position.exchange, position.symbol,
                position.entry_funding_rate, rate,
            )
            return True

        return False

    def get_open_positions(self) -> List[ArbPosition]:
        """Return all open positions."""
        return [p for p in self._positions.values() if p.status == "open"]

    def get_closed_positions(self) -> List[PnLResult]:
        """Return all closed position results."""
        return list(self._closed_positions)

    def _get_current_rate(self, exchange: str, symbol: str) -> Optional[float]:
        """Get current funding rate from tracker."""
        fr = self.tracker.get_current_rate(exchange, symbol)
        return fr.rate if fr else None

    def _calculate_position_pnl(
        self,
        position: ArbPosition,
        current_rate: float,
    ) -> float:
        """Calculate total realized P&L for a position."""
        periods = position.periods_elapsed()
        avg_rate = (position.entry_funding_rate + current_rate) / 2.0

        notional = abs(position.perp_position) * 1.0
        funding_pnl = notional * avg_rate * periods

        return funding_pnl + position.funding_collected

    def _calculate_unrealized_pnl(
        self,
        position: ArbPosition,
        current_rate: float,
    ) -> float:
        """Calculate unrealized P&L based on current funding rate."""
        rate_change = current_rate - position.entry_funding_rate
        notional = abs(position.perp_position)
        return notional * rate_change

    @staticmethod
    def _compute_delta_drift(position: ArbPosition) -> float:
        """Compute delta drift as percentage deviation from neutral."""
        spot_value = position.spot_position
        perp_value = position.perp_position
        total = abs(spot_value) + abs(perp_value)
        if total == 0:
            return 0.0
        return abs(spot_value + perp_value) / total * 100.0


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Enforces risk limits for funding arbitrage positions.

    Checks:
    - Maximum position size
    - Maximum total funding exposure
    - Stop loss on adverse funding rate moves
    - Maximum concurrent positions
    - Drawdown limits
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self._active_positions: Dict[str, ArbPosition] = {}
        self._total_exposure: float = 0.0
        self._realized_pnl: float = 0.0
        self._peak_pnl: float = 0.0

    @property
    def max_position_size(self) -> float:
        return self.limits.max_position_size

    @property
    def max_funding_exposure(self) -> float:
        return self.limits.max_funding_exposure

    @property
    def stop_loss_rate(self) -> float:
        return self.limits.stop_loss_rate

    def check_risk_limits(self, position: ArbPosition) -> bool:
        """
        Validate a new position against all risk limits.

        Returns True if the position passes all checks.
        """
        position_notional = self._compute_notional(position)

        if position_notional > self.limits.max_position_size:
            logger.warning(
                "Risk check failed: position size %.2f > max %.2f",
                position_notional, self.limits.max_position_size,
            )
            return False

        if self._total_exposure + position_notional > self.limits.max_funding_exposure:
            logger.warning(
                "Risk check failed: total exposure %.2f would exceed max %.2f",
                self._total_exposure + position_notional,
                self.limits.max_funding_exposure,
            )
            return False

        if len(self._active_positions) >= self.limits.max_concurrent_positions:
            logger.warning(
                "Risk check failed: max concurrent positions (%d) reached",
                self.limits.max_concurrent_positions,
            )
            return False

        if position.entry_funding_rate <= self.limits.stop_loss_rate:
            logger.warning(
                "Risk check failed: entry rate %.6f <= stop_loss %.6f",
                position.entry_funding_rate, self.limits.stop_loss_rate,
            )
            return False

        return True

    def register_position(self, position: ArbPosition) -> None:
        """Register a new position for tracking."""
        key = f"{position.exchange}:{position.symbol}"
        self._active_positions[key] = position
        self._total_exposure += self._compute_notional(position)
        logger.info(
            "Registered position: %s exposure=%.2f total=%.2f",
            key, self._compute_notional(position), self._total_exposure,
        )

    def unregister_position(self, position: ArbPosition) -> None:
        """Remove a position from tracking."""
        key = f"{position.exchange}:{position.symbol}"
        if key in self._active_positions:
            self._active_positions.pop(key)
            self._total_exposure -= self._compute_notional(position)
            self._total_exposure = max(0.0, self._total_exposure)
            logger.info(
                "Unregistered position: %s remaining_exposure=%.2f",
                key, self._total_exposure,
            )

    def update_pnl(self, pnl: float) -> None:
        """Update running P&L and check drawdown limits."""
        self._realized_pnl += pnl
        self._peak_pnl = max(self._peak_pnl, self._realized_pnl)

        if self._peak_pnl > 0:
            drawdown = (self._peak_pnl - self._realized_pnl) / self._peak_pnl
            if drawdown > self.limits.max_drawdown_pct:
                logger.warning(
                    "Drawdown limit breached: %.2f%% > %.2f%%",
                    drawdown * 100, self.limits.max_drawdown_pct * 100,
                )
                return False
        return True

    def get_exposure_summary(self) -> Dict[str, Any]:
        """Return current risk exposure summary."""
        return {
            "total_exposure": self._total_exposure,
            "max_exposure": self.limits.max_funding_exposure,
            "exposure_pct": (
                self._total_exposure / self.limits.max_funding_exposure * 100
                if self.limits.max_funding_exposure > 0
                else 0.0
            ),
            "active_positions": len(self._active_positions),
            "max_concurrent": self.limits.max_concurrent_positions,
            "realized_pnl": self._realized_pnl,
            "peak_pnl": self._peak_pnl,
            "stop_loss_rate": self.limits.stop_loss_rate,
        }

    @staticmethod
    def _compute_notional(position: ArbPosition) -> float:
        """Compute the notional value of a position."""
        return abs(position.spot_position) + abs(position.perp_position)
