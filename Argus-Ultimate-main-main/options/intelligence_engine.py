"""
Options Intelligence Engine - Ultimate Edge Module

Provides options-aware trading intelligence:
- Greeks calculation (delta, gamma, theta, vega, rho)
- Delta hedging automation
- Implied volatility surface
- Options strategy signals
- Risk-adjusted position sizing

This module enables hedging strategies and income generation through options.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionsStrategy(str, Enum):
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    COLLAR = "collar"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"


@dataclass
class Greeks:
    """Option Greeks values."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    iv: float = 0.0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptionContract:
    """Option contract specification."""
    underlying_price: float
    strike_price: float
    time_to_expiry_days: float
    risk_free_rate: float = 0.05
    iv: float = 0.3
    option_type: OptionType = OptionType.CALL
    quantity: float = 1.0


@dataclass
class HedgingSignal:
    """Delta hedging signal."""
    action: str
    hedge_quantity: float
    current_delta: float
    target_delta: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptionsSignal:
    """Trading signal from options analysis."""
    action: str
    confidence: float
    strategy: Optional[OptionsStrategy]
    greeks: Greeks
    reasons: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


# Standard normal CDF and PDF
def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Volatility
        option_type: CALL or PUT

    Returns:
        Option price
    """
    if T <= 0:
        if option_type == OptionType.CALL:
            return max(0, S - K)
        else:
            return max(0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == OptionType.CALL:
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    return price


def calculate_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> Greeks:
    """
    Calculate all option Greeks using Black-Scholes model.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Volatility
        option_type: CALL or PUT

    Returns:
        Greeks dataclass with all values
    """
    if T <= 0:
        if option_type == OptionType.CALL:
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0, iv=sigma, price=black_scholes_price(S, K, max(0.001, T), r, sigma, option_type))

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    delta_d = norm_cdf(d1) if option_type == OptionType.CALL else norm_cdf(d1) - 1

    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))

    if option_type == OptionType.CALL:
        theta = (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365
        rho = K * T * math.exp(-r * T) * norm_cdf(d2) / 100
    else:
        theta = (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100

    vega = S * norm_pdf(d1) * math.sqrt(T) / 100

    price = black_scholes_price(S, K, T, r, sigma, option_type)

    return Greeks(
        delta=delta_d,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        iv=sigma,
        price=price,
    )


class OptionsIntelligence:
    """
    Options intelligence engine for hedging and income strategies.

    Provides:
    - Real-time Greeks calculation
    - Delta hedging automation
    - IV surface analysis
    - Options strategy signals
    """

    def __init__(
        self,
        target_delta: float = 0.0,
        rebalance_threshold: float = 0.05,
        risk_free_rate: float = 0.05,
    ):
        self.target_delta = target_delta
        self.rebalance_threshold = rebalance_threshold
        self.risk_free_rate = risk_free_rate

        self._position_delta = 0.0
        self._options_held: List[Tuple[OptionContract, Greeks]] = []
        self._iv_history: Dict[str, List[float]] = {}

    def calculate_contract_greeks(
        self,
        underlying_price: float,
        strike_price: float,
        days_to_expiry: float,
        iv: float,
        option_type: OptionType,
        quantity: float = 1.0,
    ) -> Greeks:
        """Calculate Greeks for an option contract."""
        T = days_to_expiry / 365.0
        greeks = calculate_greeks(
            S=underlying_price,
            K=strike_price,
            T=T,
            r=self.risk_free_rate,
            sigma=iv,
            option_type=option_type,
        )
        greeks.delta *= quantity
        greeks.gamma *= quantity
        greeks.theta *= quantity
        greeks.vega *= quantity
        greeks.rho *= quantity
        return greeks

    def add_option(
        self,
        underlying_price: float,
        strike_price: float,
        days_to_expiry: float,
        iv: float,
        option_type: OptionType,
        quantity: float = 1.0,
    ) -> Greeks:
        """Add an option to the portfolio and update Greeks."""
        contract = OptionContract(
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry_days=days_to_expiry,
            risk_free_rate=self.risk_free_rate,
            iv=iv,
            option_type=option_type,
            quantity=quantity,
        )
        greeks = self.calculate_contract_greeks(
            underlying_price=underlying_price,
            strike_price=strike_price,
            days_to_expiry=days_to_expiry,
            iv=iv,
            option_type=option_type,
            quantity=quantity,
        )
        self._options_held.append((contract, greeks))
        self._position_delta += greeks.delta
        return greeks

    def remove_option(self, index: int) -> bool:
        """Remove an option from the portfolio."""
        if 0 <= index < len(self._options_held):
            _, greeks = self._options_held.pop(index)
            self._position_delta -= greeks.delta
            return True
        return False

    def get_delta_hedge_signal(
        self,
        current_underlying_price: float,
        current_position_size: float,
    ) -> HedgingSignal:
        """
        Calculate delta hedging needs.

        Args:
            current_underlying_price: Current price of underlying
            current_position_size: Size of current position (in base currency)

        Returns:
            HedgingSignal with action to take
        """
        underlying_delta = current_position_size / current_underlying_price

        total_delta = underlying_delta + self._position_delta

        target_delta = self.target_delta

        delta_to_hedge = target_delta - total_delta

        if abs(delta_to_hedge) < self.rebalance_threshold:
            return HedgingSignal(
                action='hold',
                hedge_quantity=0.0,
                current_delta=total_delta,
                target_delta=target_delta,
                reason="Delta within threshold",
            )

        hedge_qty = delta_to_hedge
        action = 'buy' if hedge_qty > 0 else 'sell'

        return HedgingSignal(
            action=action,
            hedge_quantity=abs(hedge_qty),
            current_delta=total_delta,
            target_delta=target_delta,
            reason=f"Delta off by {delta_to_hedge:.4f} - {'buy' if action == 'buy' else 'sell'} {abs(delta_to_hedge):.4f} units",
        )

    def calculate_portfolio_delta(
        self,
        underlying_price: float,
        position_size: float,
    ) -> float:
        """Calculate total portfolio delta."""
        underlying_delta = position_size / underlying_price
        return underlying_delta + self._position_delta

    def get_iv_rank(self, symbol: str, current_iv: float, lookback: int = 30) -> float:
        """
        Get IV rank (0-100) based on historical IV.

        Args:
            symbol: Symbol identifier
            current_iv: Current implied volatility
            lookback: Number of historical IV points to consider

        Returns:
            IV rank (0 = lowest, 100 = highest)
        """
        if symbol not in self._iv_history:
            self._iv_history[symbol] = []

        self._iv_history[symbol].append(current_iv)
        if len(self._iv_history[symbol]) > lookback * 2:
            self._iv_history[symbol].pop(0)

        if len(self._iv_history[symbol]) < lookback:
            return 50.0

        historical_ivs = self._iv_history[symbol][-lookback:]
        rank = sum(1 for iv in historical_ivs if current_iv > iv) / len(historical_ivs) * 100
        return rank

    def get_signal(
        self,
        underlying_price: float,
        days_to_expiry: float,
        iv: float,
        current_position_size: float,
    ) -> OptionsSignal:
        """Generate options-based trading signal."""
        reasons = []

        iv_rank = self.get_iv_rank("default", iv)

        if iv_rank > 80:
            reasons.append(f"IV Rank very high ({iv_rank:.1f}) - consider selling options")
        elif iv_rank < 20:
            reasons.append(f"IV Rank very low ({iv_rank:.1f}) - consider buying options")

        delta = self.get_delta_hedge_signal(underlying_price, current_position_size)
        if delta.action != 'hold':
            reasons.append(delta.reason)

        greeks = self.calculate_contract_greeks(
            underlying_price=underlying_price,
            strike_price=underlying_price,
            days_to_expiry=days_to_expiry,
            iv=iv,
            option_type=OptionType.CALL,
        )

        action = 'hold'
        confidence = 0.5

        if iv_rank > 80 and delta.action == 'sell':
            action = 'sell'
            confidence = 0.75
            reasons.append("High IV + negative delta = good time to hedge/sell")
        elif iv_rank < 20 and delta.action == 'buy':
            action = 'buy'
            confidence = 0.75
            reasons.append("Low IV + positive delta = good time to buy hedges")

        return OptionsSignal(
            action=action,
            confidence=min(0.95, confidence),
            strategy=None,
            greeks=greeks,
            reasons=reasons,
        )

    def reset(self) -> None:
        """Reset all state."""
        self._position_delta = 0.0
        self._options_held.clear()
        self._iv_history.clear()
        logger.info("OptionsIntelligence reset")
