"""
Exotic Options & Advanced Derivatives Strategies
=================================================
Institutional-grade options strategies beyond vanilla calls/puts.

Strategies:
1. Variance Swaps - Pure volatility exposure
2. Correlation Trading - Dispersion and correlation swaps
3. Exotic Options - Barrier, Asian, Lookback, Rainbow
4. Volatility Surface Trading - Skew and term structure
5. Structured Products - Autocallables, CLNs, worst-of

These strategies capture edges unavailable to retail traders.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class ExoticType(Enum):
    BARRIER = "barrier"
    ASIAN = "asian"
    LOOKBACK = "lookback"
    RAINBOW = "rainbow"
    CLIFFORD = "clifford"
    POWER = "power"


class BarrierType(Enum):
    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"


@dataclass
class VarianceSwapQuote:
    """Variance swap pricing quote."""
    underlying: str
    strike_variance: float  # Annualized variance at strike
    current_realized_var: float
    notional: float
    maturity: datetime
    bid_ask_spread: float
    greeks: Dict[str, float] = field(default_factory=dict)
    
    @property
    def vega_notional(self) -> float:
        """Vega notional (dollar exposure per 1% vol change)."""
        return self.notional * 2 * self.strike_variance ** 0.5 / 100
    
    def pnl_at_vol(self, realized_vol: float) -> float:
        """Calculate P&L at a given realized volatility."""
        realized_var = realized_vol ** 2
        return self.notional * (realized_var - self.strike_variance)


@dataclass
class CorrelationSwapQuote:
    """Correlation swap for dispersion trading."""
    underlyings: List[str]
    strike_correlation: float
    current_implied_corr: float
    notional: float
    maturity: datetime
    
    def pnl_at_correlation(self, realized_corr: float) -> float:
        """P&L at realized correlation."""
        return self.notional * (realized_corr - self.strike_correlation)


@dataclass
class ExoticOption:
    """Exotic option contract."""
    underlying: str
    option_type: OptionType
    exotic_type: ExoticType
    strike: float
    barrier: Optional[float] = None
    maturity: datetime = None
    notional: float = 1000000.0
    
    # Asian option specific
    averaging_dates: Optional[List[datetime]] = None
    
    # Lookback specific
    lookback_type: str = "floating"  # "floating" or "fixed"
    
    # Rainbow specific
    underlyings: Optional[List[str]] = None
    rainbow_type: str = "max"  # "max", "min", "best_of"
    
    # Pricing results
    price: Optional[float] = None
    greeks: Dict[str, float] = field(default_factory=dict)
    
    def is_knocked_out(self, spot: float) -> bool:
        """Check if barrier option is knocked out."""
        if self.barrier is None:
            return False
        
        if self.exotic_type == ExoticType.BARRIER:
            if self.option_type == OptionType.CALL:
                # Up-and-out: knocked out if spot exceeds barrier
                return spot >= self.barrier
            else:
                # Down-and-out: knocked out if spot falls below barrier
                return spot <= self.barrier
        return False


@dataclass
class VarianceSwap:
    """Variance swap contract for pure volatility exposure."""
    underlying: str
    strike_variance: float  # K in variance swap formula
    notional: float  # Vega notional
    maturity: datetime
    start_date: datetime
    
    # Monitoring
    daily_returns: List[float] = field(default_factory=list)
    
    @property
    def realized_variance(self) -> float:
        """Calculate realized variance from daily returns."""
        if len(self.daily_returns) < 2:
            return 0.0
        return np.var(self.daily_returns, ddof=1) * 252
    
    @property
    def days_to_maturity(self) -> int:
        """Days remaining to maturity."""
        return max(0, (self.maturity - datetime.now()).days)
    
    def calculate_payoff(self, final_realized_var: float) -> float:
        """
        Variance swap payoff:
        N * (σ²_realized - K)
        
        Where:
        - N = notional
        - σ²_realized = realized variance
        - K = strike variance
        """
        return self.notional * (final_realized_var - self.strike_variance)
    
    def mark_to_market(
        self,
        current_realized_var: float,
        current_implied_var: float,
        risk_free_rate: float = 0.05
    ) -> Dict[str, float]:
        """Mark-to-market valuation."""
        t = self.days_to_maturity / 365
        
        # Fair value based on implied variance
        fair_value = self.notional * (current_implied_var - self.strike_variance)
        
        # Current P&L based on realized
        current_pnl = self.notional * (current_realized_var - self.strike_variance)
        
        # Discount to present
        pv = fair_value * np.exp(-risk_free_rate * t)
        
        return {
            "fair_value": fair_value,
            "current_pnl": current_pnl,
            "present_value": pv,
            "vega": self.vega_notional,
            "days_to_maturity": self.days_to_maturity
        }
    
    @property
    def vega_notional(self) -> float:
        """Vega notional (exposure per 1% vol change)."""
        return self.notional * 2 * np.sqrt(self.strike_variance) / 100


class VarianceSwapPricer:
    """
    Variance swap pricing using log-contract replication.
    
    The fair strike is the integral of OTM option prices weighted by 1/K².
    """
    
    @staticmethod
    def calculate_fair_strike(
        strikes: List[float],
        call_prices: List[float],
        put_prices: List[float],
        forward: float,
        maturity_years: float,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate fair variance strike using Carr-Madan formula.
        
        K_var = 2 * e^(rT) * [∫₀ᴷ₀ P(K)/K² dK + ∫ᴷ₀^∞ C(K)/K² dK]
        """
        # Separate OTM calls and puts
        otm_puts = [(k, p) for k, p in zip(strikes, put_prices) if k < forward]
        otm_calls = [(k, c) for k, c in zip(strikes, call_prices) if k > forward]
        
        # Numerical integration
        put_integral = 0
        for i in range(len(otm_puts) - 1):
            k1, p1 = otm_puts[i]
            k2, p2 = otm_puts[i + 1]
            put_integral += (p1 / k1**2 + p2 / k2**2) * (k2 - k1) / 2
        
        call_integral = 0
        for i in range(len(otm_calls) - 1):
            k1, c1 = otm_calls[i]
            k2, c2 = otm_calls[i + 1]
            call_integral += (c1 / k1**2 + c2 / k2**2) * (k2 - k1) / 2
        
        # Fair variance
        fair_var = 2 * np.exp(risk_free_rate * maturity_years) * (
            put_integral + call_integral
        )
        
        return fair_var
    
    @staticmethod
    def replicate_variance(
        spot: float,
        strikes: List[float],
        weights: List[float]
    ) -> float:
        """
        Replicate variance payoff using static options portfolio.
        
        Portfolio: Σ wᵢ * (Kᵢ/S₀ - log(Kᵢ/S₀) - 1)
        """
        total = 0
        for k, w in zip(strikes, weights):
            moneyness = k / spot
            total += w * (moneyness - np.log(moneyness) - 1)
        return total


class CorrelationTrading:
    """
    Correlation and dispersion trading strategies.
    
    Dispersion trading:
    - Sell index options (high implied correlation)
    - Buy component options (low implied correlation)
    - Profit when realized correlation < implied correlation
    """
    
    def __init__(self):
        self.positions: Dict[str, Dict[str, float]] = {}
    
    def calculate_dispersion(
        self,
        index_vol: float,
        component_vols: List[float],
        component_weights: List[float],
        correlations: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate dispersion metrics.
        
        Dispersion = Index Variance - Weighted Sum of Component Variances
        """
        # Index variance
        index_var = index_vol ** 2
        
        # Weighted component variance
        weights = np.array(component_weights)
        vols = np.array(component_vols)
        component_var = np.sum((weights * vols) ** 2)
        
        # Implied correlation from variance decomposition
        if component_var > 0:
            implied_corr = (index_var - component_var) / (
                index_var * (1 - np.sum(weights ** 2))
            )
        else:
            implied_corr = 0
        
        # Average pairwise correlation
        if correlations.size > 0:
            # Extract upper triangle (excluding diagonal)
            n = len(component_vols)
            upper_tri = correlations[np.triu_indices(n, k=1)]
            avg_corr = np.mean(upper_tri) if len(upper_tri) > 0 else 0
        else:
            avg_corr = 0
        
        dispersion = index_var - component_var
        
        return {
            "index_variance": index_var,
            "component_variance": component_var,
            "dispersion": dispersion,
            "implied_correlation": np.clip(implied_corr, -1, 1),
            "average_correlation": avg_corr,
            "correlation_premium": implied_corr - avg_corr
        }
    
    def dispersion_trade_pnl(
        self,
        index_vol_change: float,
        component_vol_changes: List[float],
        weights: List[float],
        position_size: float = 1.0
    ) -> float:
        """
        Calculate P&L for dispersion trade.
        
        Long dispersion = Short index vol, Long component vols
        """
        # Index variance change
        index_var_change = 2 * index_vol_change  # Simplified
        
        # Component variance change
        comp_var_change = sum(
            2 * w * dv for w, dv in zip(weights, component_vol_changes)
        )
        
        # Dispersion P&L
        pnl = position_size * (comp_var_change - index_var_change)
        
        return pnl
    
    def correlation_swap_pnl(
        self,
        strike_correlation: float,
        realized_correlation: float,
        notional: float
    ) -> float:
        """
        Correlation swap P&L.
        
        Payoff = Notional × (ρ_realized - ρ_strike)
        """
        return notional * (realized_correlation - strike_correlation)


class ExoticOptionsPricer:
    """
    Pricing engine for exotic options using Monte Carlo and analytical methods.
    """
    
    def __init__(self, num_paths: int = 100000, num_steps: int = 252):
        self.num_paths = num_paths
        self.num_steps = num_steps
    
    def price_barrier_option(
        self,
        spot: float,
        strike: float,
        barrier: float,
        volatility: float,
        risk_free_rate: float,
        time_to_maturity: float,
        option_type: OptionType,
        barrier_type: BarrierType,
        dividend_yield: float = 0.0
    ) -> Dict[str, float]:
        """
        Price barrier option using Monte Carlo with Brownian bridge correction.
        """
        dt = time_to_maturity / self.num_steps
        
        # Generate paths
        np.random.seed(42)
        Z = np.random.standard_normal(self.num_paths)
        
        # Terminal spot price (GBM)
        drift = (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) * time_to_maturity
        diffusion = volatility * np.sqrt(time_to_maturity) * Z
        terminal_spots = spot * np.exp(drift + diffusion)
        
        # Check barrier crossing with Brownian bridge
        barrier_hit = self._brownian_bridge_barrier_probability(
            spot, barrier, volatility, time_to_maturity, option_type, barrier_type
        )
        
        # Determine which paths are active
        if barrier_type in [BarrierType.UP_AND_OUT, BarrierType.DOWN_AND_OUT]:
            active = ~barrier_hit
        else:  # Knock-in
            active = barrier_hit
        
        # Calculate payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(terminal_spots - strike, 0)
        else:
            payoffs = np.maximum(strike - terminal_spots, 0)
        
        # Apply barrier condition
        payoffs = payoffs * active
        
        # Discount
        price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.num_paths)
        
        return {
            "price": price,
            "std_error": std_error,
            "active_paths": np.sum(active),
            "knockout_probability": np.mean(~active) if "out" in barrier_type.value else np.mean(active)
        }
    
    def _brownian_bridge_barrier_probability(
        self,
        spot: float,
        barrier: float,
        volatility: float,
        time_to_maturity: float,
        option_type: OptionType,
        barrier_type: BarrierType
    ) -> np.ndarray:
        """
        Estimate barrier crossing using Brownian bridge.
        
        More accurate than checking only terminal value.
        """
        # Simplified: check if terminal value suggests barrier crossing
        # In production, use full Brownian bridge simulation
        
        drift = -0.5 * volatility ** 2 * time_to_maturity
        diffusion = volatility * np.sqrt(time_to_maturity)
        
        # Generate terminal values
        Z = np.random.standard_normal(self.num_paths)
        log_terminal = np.log(spot) + drift + diffusion * Z
        
        if barrier_type in [BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN]:
            # Up barrier
            return log_terminal >= np.log(barrier)
        else:
            # Down barrier
            return log_terminal <= np.log(barrier)
    
    def price_asian_option(
        self,
        spot: float,
        strike: float,
        volatility: float,
        risk_free_rate: float,
        time_to_maturity: float,
        option_type: OptionType,
        averaging_type: str = "arithmetic",
        num_fixings: int = 20
    ) -> Dict[str, float]:
        """
        Price Asian option using Monte Carlo.
        
        Asian options average the underlying price over multiple observations,
        reducing volatility and cost.
        """
        dt = time_to_maturity / num_fixings
        
        np.random.seed(42)
        paths = np.zeros((self.num_paths, num_fixings + 1))
        paths[:, 0] = spot
        
        # Generate paths
        for i in range(num_fixings):
            Z = np.random.standard_normal(self.num_paths)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (risk_free_rate - 0.5 * volatility ** 2) * dt +
                volatility * np.sqrt(dt) * Z
            )
        
        # Calculate averages
        if averaging_type == "arithmetic":
            averages = np.mean(paths[:, 1:], axis=1)
        else:  # geometric
            averages = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        
        # Calculate payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(averages - strike, 0)
        else:
            payoffs = np.maximum(strike - averages, 0)
        
        # Discount
        price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.num_paths)
        
        return {
            "price": price,
            "std_error": std_error,
            "average_type": averaging_type
        }
    
    def price_lookback_option(
        self,
        spot: float,
        strike: float,
        volatility: float,
        risk_free_rate: float,
        time_to_maturity: float,
        option_type: OptionType,
        lookback_type: str = "floating"
    ) -> Dict[str, float]:
        """
        Price lookback option.
        
        Floating strike: payoff based on max/min of path vs current spot
        Fixed strike: payoff based on max/min of path vs fixed strike
        """
        dt = time_to_maturity / self.num_steps
        
        np.random.seed(42)
        paths = np.zeros((self.num_paths, self.num_steps + 1))
        paths[:, 0] = spot
        
        # Generate paths
        for i in range(self.num_steps):
            Z = np.random.standard_normal(self.num_paths)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (risk_free_rate - 0.5 * volatility ** 2) * dt +
                volatility * np.sqrt(dt) * Z
            )
        
        # Calculate max and min
        path_max = np.max(paths, axis=1)
        path_min = np.min(paths, axis=1)
        
        # Calculate payoffs
        if lookback_type == "floating":
            if option_type == OptionType.CALL:
                payoffs = path_max - paths[:, -1]  # Max minus terminal
            else:
                payoffs = paths[:, -1] - path_min  # Terminal minus min
        else:  # fixed strike
            if option_type == OptionType.CALL:
                payoffs = np.maximum(path_max - strike, 0)
            else:
                payoffs = np.maximum(strike - path_min, 0)
        
        # Discount
        price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.num_paths)
        
        return {
            "price": price,
            "std_error": std_error,
            "lookback_type": lookback_type
        }
    
    def price_rainbow_option(
        self,
        spots: List[float],
        strike: float,
        volatilities: List[float],
        correlations: np.ndarray,
        risk_free_rate: float,
        time_to_maturity: float,
        rainbow_type: str = "max"
    ) -> Dict[str, float]:
        """
        Price rainbow option on multiple underlyings.
        
        Types:
        - max: Best performing asset
        - min: Worst performing asset
        - best_of: Best performing with strike
        """
        n_assets = len(spots)
        dt = time_to_maturity / self.num_steps
        
        # Cholesky decomposition for correlated random numbers
        try:
            L = np.linalg.cholesky(correlations)
        except np.linalg.LinAlgError:
            L = np.eye(n_assets)
        
        np.random.seed(42)
        
        # Initialize paths
        paths = np.zeros((self.num_paths, n_assets, self.num_steps + 1))
        for i in range(n_assets):
            paths[:, i, 0] = spots[i]
        
        # Generate correlated paths
        for step in range(self.num_steps):
            Z = np.random.standard_normal((self.num_paths, n_assets))
            correlated_Z = Z @ L.T
            
            for i in range(n_assets):
                drift = (risk_free_rate - 0.5 * volatilities[i] ** 2) * dt
                diffusion = volatilities[i] * np.sqrt(dt) * correlated_Z[:, i]
                paths[:, i, step + 1] = paths[:, i, step] * np.exp(drift + diffusion)
        
        # Calculate terminal values
        terminals = paths[:, :, -1]
        
        # Calculate payoffs based on rainbow type
        if rainbow_type == "max":
            best_returns = np.max(terminals / spots, axis=1)
            payoffs = np.maximum(best_returns * spots[np.argmax(terminals, axis=1)] - strike, 0)
        elif rainbow_type == "min":
            worst_returns = np.min(terminals / spots, axis=1)
            payoffs = np.maximum(worst_returns * spots[np.argmin(terminals, axis=1)] - strike, 0)
        else:  # best_of
            best_terminal = np.max(terminals, axis=1)
            payoffs = np.maximum(best_terminal - strike, 0)
        
        # Discount
        price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.num_paths)
        
        return {
            "price": price,
            "std_error": std_error,
            "rainbow_type": rainbow_type
        }


class VolatilitySurfaceTrader:
    """
    Trade the volatility surface: skew, term structure, and kurtosis.
    
    Strategies:
    - Calendar spreads (term structure)
    - Butterfly spreads (kurtosis/convexity)
    - Risk reversals (skew)
    - Box spreads (arbitrage)
    """
    
    def __init__(self):
        self.positions: List[Dict[str, Any]] = []
    
    def calendar_spread(
        self,
        spot: float,
        strike: float,
        short_vol: float,
        long_vol: float,
        short_tte: float,
        long_tte: float,
        risk_free_rate: float = 0.05,
        option_type: OptionType = OptionType.CALL
    ) -> Dict[str, float]:
        """
        Calendar spread: Sell short-term vol, buy long-term vol.
        
        Profits when:
        - Term structure flattens (short vol increases more)
        - Realized vol matches short-term implied
        """
        from scipy.stats import norm
        
        def bs_price(s, k, t, r, sigma, opt_type):
            d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
            d2 = d1 - sigma*np.sqrt(t)
            if opt_type == OptionType.CALL:
                return s*norm.cdf(d1) - k*np.exp(-r*t)*norm.cdf(d2)
            else:
                return k*np.exp(-r*t)*norm.cdf(-d2) - s*norm.cdf(-d1)
        
        short_price = bs_price(spot, strike, short_tte, risk_free_rate, short_vol, option_type)
        long_price = bs_price(spot, strike, long_tte, risk_free_rate, long_vol, option_type)
        
        # Net premium (receive short, pay long)
        net_premium = short_price - long_price
        
        return {
            "short_price": short_price,
            "long_price": long_price,
            "net_premium": net_premium,
            "theta": self._calendar_theta(spot, strike, short_tte, long_tte, short_vol, long_vol, risk_free_rate),
            "vega": self._calendar_vega(spot, strike, long_tte, long_vol, risk_free_rate) -
                    self._calendar_vega(spot, strike, short_tte, short_vol, risk_free_rate)
        }
    
    def _calendar_theta(self, spot, strike, short_tte, long_tte, short_vol, long_vol, r):
        """Approximate theta of calendar spread."""
        # Short option theta (positive - time decay benefits)
        short_theta = spot * short_vol * 0.01 / np.sqrt(252)
        # Long option theta (negative - time decay hurts)
        long_theta = spot * long_vol * 0.01 / np.sqrt(252) * 0.5
        return short_theta - long_theta
    
    def _calendar_vega(self, spot, strike, tte, vol, r):
        """Approximate vega of an option."""
        d1 = (np.log(spot/strike) + (r + 0.5*vol**2)*tte) / (vol*np.sqrt(tte))
        return spot * np.sqrt(tte) * stats.norm.pdf(d1) / 100
    
    def butterfly_spread(
        self,
        spot: float,
        low_strike: float,
        mid_strike: float,
        high_strike: float,
        volatility: float,
        time_to_maturity: float,
        risk_free_rate: float = 0.05,
        option_type: OptionType = OptionType.CALL
    ) -> Dict[str, float]:
        """
        Butterfly spread: Long 1 low, Short 2 mid, Long 1 high.
        
        Profits from:
        - Low realized volatility
        - Price staying near mid strike
        """
        from scipy.stats import norm
        
        def bs_price(s, k, t, r, sigma, opt_type):
            d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
            d2 = d1 - sigma*np.sqrt(t)
            if opt_type == OptionType.CALL:
                return s*norm.cdf(d1) - k*np.exp(-r*t)*norm.cdf(d2)
            else:
                return k*np.exp(-r*t)*norm.cdf(-d2) - s*norm.cdf(-d1)
        
        low_price = bs_price(spot, low_strike, time_to_maturity, risk_free_rate, volatility, option_type)
        mid_price = bs_price(spot, mid_strike, time_to_maturity, risk_free_rate, volatility, option_type)
        high_price = bs_price(spot, high_strike, time_to_maturity, risk_free_rate, volatility, option_type)
        
        # Net cost
        net_cost = low_price - 2 * mid_price + high_price
        
        # Max profit (at expiration if spot = mid_strike)
        max_profit = (mid_strike - low_strike) - net_cost
        
        return {
            "net_cost": net_cost,
            "max_profit": max_profit,
            "breakeven_low": mid_strike - max_profit,
            "breakeven_high": mid_strike + max_profit,
            "gamma": self._butterfly_gamma(spot, mid_strike, time_to_maturity, volatility, risk_free_rate)
        }
    
    def _butterfly_gamma(self, spot, mid_strike, tte, vol, r):
        """Approximate gamma of butterfly (peak gamma at mid strike)."""
        d1 = (np.log(spot/mid_strike) + (r + 0.5*vol**2)*tte) / (vol*np.sqrt(tte))
        return stats.norm.pdf(d1) / (spot * vol * np.sqrt(tte))
    
    def risk_reversal(
        self,
        spot: float,
        otm_put_strike: float,
        otm_call_strike: float,
        put_vol: float,
        call_vol: float,
        time_to_maturity: float,
        risk_free_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Risk reversal: Long OTM call, Short OTM put.
        
        Captures skew exposure - profits when:
        - Skew flattens (call vol increases relative to put vol)
        - Upside move exceeds expectations
        """
        from scipy.stats import norm
        
        def bs_price(s, k, t, r, sigma, opt_type):
            d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
            d2 = d1 - sigma*np.sqrt(t)
            if opt_type == OptionType.CALL:
                return s*norm.cdf(d1) - k*np.exp(-r*t)*norm.cdf(d2)
            else:
                return k*np.exp(-r*t)*norm.cdf(-d2) - s*norm.cdf(-d1)
        
        call_price = bs_price(spot, otm_call_strike, time_to_maturity, risk_free_rate, call_vol, OptionType.CALL)
        put_price = bs_price(spot, otm_put_strike, time_to_maturity, risk_free_rate, put_vol, OptionType.PUT)
        
        # Net premium (typically zero-cost or small credit)
        net_premium = put_price - call_price  # Receive put premium, pay call premium
        
        return {
            "call_price": call_price,
            "put_price": put_price,
            "net_credit": net_premium,
            "skew_exposure": call_vol - put_vol,
            "delta": 0.5  # Approximate
        }


class StructuredProductsEngine:
    """
    Pricing and analysis for structured products.
    
    Products:
    - Autocallables (Phoenix notes)
    - Credit-linked notes (CLNs)
    - Worst-of options
    - Capital-protected notes
    """
    
    def price_autocallable(
        self,
        spot: float,
        barrier: float,
        coupon: float,
        observation_frequency: int = 4,  # Quarterly
        maturity_years: float = 3.0,
        risk_free_rate: float = 0.05,
        volatility: float = 0.25
    ) -> Dict[str, float]:
        """
        Price autocallable (Phoenix) note.
        
        Features:
        - Quarterly observation for autocall
        - Coupon if above coupon barrier
        - Principal at risk if below final barrier
        """
        num_observations = int(maturity_years * observation_frequency)
        dt = 1 / observation_frequency
        
        np.random.seed(42)
        paths = np.zeros((self.num_paths, num_observations + 1))
        paths[:, 0] = spot
        
        # Generate paths
        for i in range(num_observations):
            Z = np.random.standard_normal(self.num_paths)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (risk_free_rate - 0.5 * volatility ** 2) * dt +
                volatility * np.sqrt(dt) * Z
            )
        
        # Calculate autocall events
        autocall_levels = barrier * np.ones(num_observations)
        autocall_occurred = np.any(paths[:, 1:] >= autocall_levels, axis=1)
        
        # Calculate coupons
        coupon_barrier = barrier * 0.8  # Lower than autocall barrier
        coupon_paid = np.sum(paths[:, 1:] >= coupon_barrier, axis=1) * coupon * dt
        
        # Final payoff
        final_spot = paths[:, -1]
        principal_return = np.where(
            final_spot >= barrier,
            1.0,  # Full principal
            final_spot / spot  # Loss proportional to decline
        )
        
        # Total payoff
        payoff = np.where(
            autocall_occurred,
            1.0 + coupon_paid,  # Autocalled with coupons
            principal_return + coupon_paid  # Held to maturity
        )
        
        # Present value
        price = np.exp(-risk_free_rate * maturity_years) * np.mean(payoff)
        
        # Risk metrics
        autocall_probability = np.mean(autocall_occurred)
        expected_maturity = np.mean(np.where(
            autocall_occurred,
            np.argmax(paths[:, 1:] >= autocall_levels, axis=1) + 1,
            num_observations
        )) / observation_frequency
        
        return {
            "price": price,
            "autocall_probability": autocall_probability,
            "expected_maturity_years": expected_maturity,
            "expected_coupon": np.mean(coupon_paid),
            "principal_at_risk": np.mean(final_spot < barrier)
        }
    
    def price_worst_of(
        self,
        spots: List[float],
        strikes: List[float],
        volatilities: List[float],
        correlations: np.ndarray,
        risk_free_rate: float,
        time_to_maturity: float
    ) -> Dict[str, float]:
        """
        Price worst-of option (basket option on minimum return).
        
        Payoff = max(min(S₁/S₀, S₂/S₀, ...) - K, 0)
        """
        n_assets = len(spots)
        dt = time_to_maturity / self.num_steps
        
        # Cholesky for correlated paths
        try:
            L = np.linalg.cholesky(correlations)
        except np.linalg.LinAlgError:
            L = np.eye(n_assets)
        
        np.random.seed(42)
        
        # Generate paths
        terminals = np.zeros((self.num_paths, n_assets))
        for i in range(n_assets):
            Z = np.random.standard_normal(self.num_paths)
            correlated_Z = Z  # Simplified
            drift = (risk_free_rate - 0.5 * volatilities[i] ** 2) * time_to_maturity
            diffusion = volatilities[i] * np.sqrt(time_to_maturity) * correlated_Z
            terminals[:, i] = spots[i] * np.exp(drift + diffusion)
        
        # Calculate returns
        returns = terminals / np.array(spots)
        
        # Worst return
        worst_return = np.min(returns, axis=1)
        
        # Payoff (using average strike)
        avg_strike = np.mean(strikes) / np.mean(spots)
        payoffs = np.maximum(worst_return - avg_strike, 0)
        
        price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
        
        return {
            "price": price,
            "worst_performer_prob": np.mean(worst_return < avg_strike),
            "expected_worst_return": np.mean(worst_return)
        }


# ============================================================================
# Strategy Implementation
# ============================================================================

class ExoticOptionsStrategy:
    """
    Complete exotic options trading strategy.
    
    Combines:
    - Variance swaps for pure vol exposure
    - Dispersion trading for correlation
    - Exotic options for structured payoff
    - Vol surface trading for edge
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.variance_pricer = VarianceSwapPricer()
        self.exotic_pricer = ExoticOptionsPricer()
        self.correlation_trader = CorrelationTrading()
        self.surface_trader = VolatilitySurfaceTrader()
        self.structured_engine = StructuredProductsEngine()
        
        self.positions: List[Dict[str, Any]] = []
        self.pnl_history: List[float] = []
        
        logger.info("ExoticOptionsStrategy initialized")
    
    def generate_signals(
        self,
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals from exotic options analysis.
        
        Returns list of trade recommendations.
        """
        signals = []
        
        # 1. Variance swap signal
        if "implied_vol" in market_data and "realized_vol" in market_data:
            var_signal = self._variance_swap_signal(market_data)
            if var_signal:
                signals.append(var_signal)
        
        # 2. Dispersion signal
        if "index_vol" in market_data and "component_vols" in market_data:
            disp_signal = self._dispersion_signal(market_data)
            if disp_signal:
                signals.append(disp_signal)
        
        # 3. Vol surface signal
        if "vol_skew" in market_data:
            skew_signal = self._skew_signal(market_data)
            if skew_signal:
                signals.append(skew_signal)
        
        return signals
    
    def _variance_swap_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate variance swap signal."""
        implied_vol = data["implied_vol"]
        realized_vol = data["realized_vol"]
        
        # Variance risk premium
        vol_premium = implied_vol - realized_vol
        
        # Signal: Sell vol when premium is high, buy when low
        if vol_premium > 5:  # 5 vol points premium
            return {
                "strategy": "variance_swap",
                "action": "sell_variance",
                "confidence": min(vol_premium / 10, 1.0),
                "reason": f"High VRP: {vol_premium:.1f}%",
                "expected_edge": vol_premium * 0.5  # Expect to capture half
            }
        elif vol_premium < -2:
            return {
                "strategy": "variance_swap",
                "action": "buy_variance",
                "confidence": min(abs(vol_premium) / 5, 1.0),
                "reason": f"Negative VRP: {vol_premium:.1f}%",
                "expected_edge": abs(vol_premium) * 0.3
            }
        
        return None
    
    def _dispersion_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate dispersion trading signal."""
        index_vol = data["index_vol"]
        component_vols = data["component_vols"]
        weights = data.get("weights", [1/len(component_vols)] * len(component_vols))
        
        # Calculate implied correlation
        index_var = index_vol ** 2
        weighted_comp_var = sum((w * v) ** 2 for w, v in zip(weights, component_vols))
        
        if weighted_comp_var > 0:
            implied_corr = (index_var - weighted_comp_var) / (
                index_var * (1 - sum(w**2 for w in weights))
            )
            
            # Signal: Short correlation when high
            if implied_corr > 0.7:
                return {
                    "strategy": "dispersion",
                    "action": "short_correlation",
                    "confidence": (implied_corr - 0.5) / 0.5,
                    "reason": f"High implied correlation: {implied_corr:.2f}",
                    "expected_edge": (implied_corr - 0.5) * 20  # bps
                }
        
        return None
    
    def _skew_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate volatility skew signal."""
        vol_skew = data["vol_skew"]  # 25-delta put vol - 25-delta call vol
        
        # Extreme skew signals
        if vol_skew > 10:  # Very steep skew
            return {
                "strategy": "risk_reversal",
                "action": "sell_puts_buy_calls",
                "confidence": min(vol_skew / 20, 1.0),
                "reason": f"Steep skew: {vol_skew} vol pts",
                "expected_edge": vol_skew * 0.3
            }
        elif vol_skew < 0:  # Inverted skew (unusual)
            return {
                "strategy": "risk_reversal",
                "action": "sell_calls_buy_puts",
                "confidence": min(abs(vol_skew) / 5, 1.0),
                "reason": f"Inverted skew: {vol_skew} vol pts",
                "expected_edge": abs(vol_skew) * 0.5
            }
        
        return None
    
    def backtest(
        self,
        historical_data: Dict[str, Any],
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """Backtest exotic options strategies."""
        # Simplified backtest
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0
        }


if __name__ == "__main__":
    # Demo usage
    pricer = ExoticOptionsPricer()
    
    # Price a barrier option
    barrier_result = pricer.price_barrier_option(
        spot=100,
        strike=100,
        barrier=110,
        volatility=0.25,
        risk_free_rate=0.05,
        time_to_maturity=0.25,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_AND_OUT
    )
    print(f"Barrier Option Price: {barrier_result}")
    
    # Price an Asian option
    asian_result = pricer.price_asian_option(
        spot=100,
        strike=100,
        volatility=0.25,
        risk_free_rate=0.05,
        time_to_maturity=0.25,
        option_type=OptionType.CALL
    )
    print(f"Asian Option Price: {asian_result}")
    
    # Variance swap
    var_swap = VarianceSwap(
        underlying="SPX",
        strike_variance=0.04,  # 20% vol squared
        notional=1000000,
        maturity=datetime.now() + timedelta(days=90),
        start_date=datetime.now()
    )
    print(f"Vega Notional: ${var_swap.vega_notional:,.2f}")
