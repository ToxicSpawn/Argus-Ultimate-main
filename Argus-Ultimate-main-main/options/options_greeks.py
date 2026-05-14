"""
Options Greeks Calculator
==========================
Calculates option pricing and Greeks:
- Delta: Price sensitivity to underlying
- Gamma: Delta sensitivity to underlying
- Theta: Time decay
- Vega: Volatility sensitivity
- Rho: Interest rate sensitivity

Uses Black-Scholes model with crypto adaptations.
"""

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


class OptionStyle(Enum):
    """Option exercise style."""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class OptionContract:
    """Option contract parameters."""
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry_timestamp: float
    premium: float = 0.0
    implied_volatility: float = 0.5
    style: OptionStyle = OptionStyle.EUROPEAN
    contract_size: float = 1.0  # Contracts per unit


@dataclass
class Greeks:
    """Option Greeks."""
    delta: float  # ∂V/∂S
    gamma: float  # ∂²V/∂S²
    theta: float  # ∂V/∂t (per day)
    vega: float   # ∂V/∂σ (per 1% change)
    rho: float    # ∂V/∂r (per 1% change)
    
    # Additional metrics
    implied_volatility: float = 0.0
    time_to_expiry_days: float = 0.0
    moneyness: float = 0.0  # S/K ratio
    intrinsic_value: float = 0.0
    time_value: float = 0.0


@dataclass
class OptionPricing:
    """Complete option pricing result."""
    contract: OptionContract
    theoretical_price: float
    market_price: float
    greeks: Greeks
    breakeven_price: float
    max_profit: float
    max_loss: float
    probability_itm: float
    expected_value: float


class BlackScholesCalculator:
    """
    Black-Scholes Option Pricing
    =============================
    Standard Black-Scholes model for European options.
    """
    
    @staticmethod
    def normal_cdf(x: float) -> float:
        """Cumulative distribution function for standard normal."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def normal_pdf(x: float) -> float:
        """Probability density function for standard normal."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    
    @classmethod
    def calculate_d1_d2(
        cls,
        S: float,  # Spot price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        r: float,  # Risk-free rate
        sigma: float  # Volatility
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters."""
        if T <= 0 or sigma <= 0:
            return 0, 0
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
    
    @classmethod
    def call_price(
        cls,
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate call option price."""
        if T <= 0:
            return max(0, S - K)
        
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma)
        price = S * cls.normal_cdf(d1) - K * math.exp(-r * T) * cls.normal_cdf(d2)
        return max(0, price)
    
    @classmethod
    def put_price(
        cls,
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate put option price."""
        if T <= 0:
            return max(0, K - S)
        
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma)
        price = K * math.exp(-r * T) * cls.normal_cdf(-d2) - S * cls.normal_cdf(-d1)
        return max(0, price)
    
    @classmethod
    def calculate_greeks(
        cls,
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: OptionType
    ) -> Greeks:
        """Calculate all Greeks."""
        if T <= 0 or sigma <= 0:
            # At expiry or invalid
            if option_type == OptionType.CALL:
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            
            return Greeks(
                delta=delta,
                gamma=0,
                theta=0,
                vega=0,
                rho=0,
                implied_volatility=sigma,
                time_to_expiry_days=0,
                moneyness=S / K,
                intrinsic_value=max(0, (S - K) if option_type == OptionType.CALL else (K - S)),
                time_value=0
            )
        
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = cls.normal_cdf(d1)
        else:
            delta = cls.normal_cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = cls.normal_pdf(d1) / (S * sigma * math.sqrt(T))
        
        # Theta (per day)
        term1 = -(S * cls.normal_pdf(d1) * sigma) / (2 * math.sqrt(T))
        if option_type == OptionType.CALL:
            term2 = r * K * math.exp(-r * T) * cls.normal_cdf(d2)
            theta = (term1 - term2) / 365
        else:
            term2 = r * K * math.exp(-r * T) * cls.normal_cdf(-d2)
            theta = (term1 + term2) / 365
        
        # Vega (per 1% change in vol)
        vega = S * cls.normal_pdf(d1) * math.sqrt(T) / 100
        
        # Rho (per 1% change in rate)
        if option_type == OptionType.CALL:
            rho = K * T * math.exp(-r * T) * cls.normal_cdf(d2) / 100
        else:
            rho = -K * T * math.exp(-r * T) * cls.normal_cdf(-d2) / 100
        
        # Additional metrics
        intrinsic = max(0, (S - K) if option_type == OptionType.CALL else (K - S))
        theoretical = cls.call_price(S, K, T, r, sigma) if option_type == OptionType.CALL else cls.put_price(S, K, T, r, sigma)
        time_value = max(0, theoretical - intrinsic)
        
        # Probability ITM
        prob_itm = cls.normal_cdf(d2) if option_type == OptionType.CALL else cls.normal_cdf(-d2)
        
        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            implied_volatility=sigma,
            time_to_expiry_days=T * 365,
            moneyness=S / K,
            intrinsic_value=intrinsic,
            time_value=time_value
        )


class ImpliedVolatilityCalculator:
    """
    Implied Volatility Calculator
    ==============================
    Finds IV using Newton-Raphson method.
    """
    
    @classmethod
    def calculate(
        cls,
        market_price: float,
        S: float, K: float, T: float, r: float,
        option_type: OptionType,
        max_iterations: int = 100,
        tolerance: float = 0.0001
    ) -> float:
        """Calculate implied volatility."""
        # Initial guess
        sigma = 0.5
        
        for _ in range(max_iterations):
            if option_type == OptionType.CALL:
                price = BlackScholesCalculator.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholesCalculator.put_price(S, K, T, r, sigma)
            
            # Calculate vega for Newton-Raphson
            d1, _ = BlackScholesCalculator.calculate_d1_d2(S, K, T, r, sigma)
            vega = S * BlackScholesCalculator.normal_pdf(d1) * math.sqrt(T)
            
            if vega == 0:
                break
            
            # Update sigma
            diff = price - market_price
            sigma = sigma - diff / vega
            
            # Ensure sigma is positive
            sigma = max(0.001, sigma)
            
            if abs(diff) < tolerance:
                break
        
        return sigma


class OptionsChainAnalyzer:
    """
    Options Chain Analyzer
    ======================
    Analyzes entire options chain for opportunities.
    """
    
    def __init__(self):
        self.chain: Dict[str, List[OptionContract]] = {}
    
    def add_contract(self, contract: OptionContract) -> None:
        """Add option contract to chain."""
        key = f"{contract.underlying}_{contract.expiry_timestamp}"
        if key not in self.chain:
            self.chain[key] = []
        self.chain[key].append(contract)
    
    def find_volatility_smile(self, underlying: str, expiry: float) -> Dict[float, float]:
        """Analyze volatility smile/skew."""
        key = f"{underlying}_{expiry}"
        if key not in self.chain:
            return {}
        
        smile = {}
        for contract in self.chain[key]:
            smile[contract.strike] = contract.implied_volatility
        
        return dict(sorted(smile.items()))
    
    def find_arbitrage_opportunities(self, underlying: str) -> List[Dict[str, Any]]:
        """Find put-call parity violations."""
        opportunities = []
        
        for key, contracts in self.chain.items():
            if not key.startswith(underlying):
                continue
            
            # Group by strike
            by_strike: Dict[float, Dict[str, OptionContract]] = {}
            for contract in contracts:
                if contract.strike not in by_strike:
                    by_strike[contract.strike] = {}
                by_strike[contract.strike][contract.option_type.value] = contract
            
            for strike, pair in by_strike.items():
                if "call" in pair and "put" in pair:
                    call = pair["call"]
                    put = pair["put"]
                    
                    # Put-call parity: C - P = S - K*e^(-rT)
                    parity_diff = (call.premium - put.premium) - (call.strike - put.strike)
                    
                    if abs(parity_diff) > 10:  # Significant deviation
                        opportunities.append({
                            "type": "put_call_parity",
                            "strike": strike,
                            "call_price": call.premium,
                            "put_price": put.premium,
                            "parity_diff": parity_diff,
                            "action": "sell_call_buy_put" if parity_diff > 0 else "sell_put_buy_call"
                        })
        
        return opportunities
    
    def calculate_greeks_portfolio(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate Greeks for a portfolio."""
        total_greeks = {
            "delta": 0,
            "gamma": 0,
            "theta": 0,
            "vega": 0,
            "rho": 0
        }
        
        for position in positions:
            contract = position["contract"]
            quantity = position["quantity"]  # Positive = long, negative = short
            
            greeks = BlackScholesCalculator.calculate_greeks(
                position["underlying_price"],
                contract.strike,
                position["time_to_expiry"],
                position["risk_free_rate"],
                contract.implied_volatility,
                contract.option_type
            )
            
            total_greeks["delta"] += greeks.delta * quantity * contract.contract_size
            total_greeks["gamma"] += greeks.gamma * quantity * contract.contract_size
            total_greeks["theta"] += greeks.theta * quantity * contract.contract_size
            total_greeks["vega"] += greeks.vega * quantity * contract.contract_size
            total_greeks["rho"] += greeks.rho * quantity * contract.contract_size
        
        return total_greeks


class OptionsStrategyBuilder:
    """
    Options Strategy Builder
    ========================
    Builds and analyzes common options strategies.
    """
    
    @staticmethod
    def long_call(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, Any]:
        """Build long call strategy."""
        premium = BlackScholesCalculator.call_price(S, K, T, r, sigma)
        greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, OptionType.CALL)
        
        return {
            "strategy": "Long Call",
            "legs": [{"type": "call", "action": "buy", "strike": K, "premium": premium}],
            "max_loss": premium,
            "max_profit": float('inf'),
            "breakeven": K + premium,
            "greeks": greeks
        }
    
    @staticmethod
    def long_put(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, Any]:
        """Build long put strategy."""
        premium = BlackScholesCalculator.put_price(S, K, T, r, sigma)
        greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, OptionType.PUT)
        
        return {
            "strategy": "Long Put",
            "legs": [{"type": "put", "action": "buy", "strike": K, "premium": premium}],
            "max_loss": premium,
            "max_profit": K * 10,  # Theoretical max
            "breakeven": K - premium,
            "greeks": greeks
        }
    
    @staticmethod
    def straddle(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, Any]:
        """Build long straddle strategy."""
        call_prem = BlackScholesCalculator.call_price(S, K, T, r, sigma)
        put_prem = BlackScholesCalculator.put_price(S, K, T, r, sigma)
        total_prem = call_prem + put_prem
        
        call_greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, OptionType.CALL)
        put_greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, OptionType.PUT)
        
        return {
            "strategy": "Long Straddle",
            "legs": [
                {"type": "call", "action": "buy", "strike": K, "premium": call_prem},
                {"type": "put", "action": "buy", "strike": K, "premium": put_prem}
            ],
            "max_loss": total_prem,
            "max_profit": float('inf'),
            "breakeven_upper": K + total_prem,
            "breakeven_lower": K - total_prem,
            "greeks": {
                "delta": call_greeks.delta + put_greeks.delta,
                "gamma": call_greeks.gamma + put_greeks.gamma,
                "theta": call_greeks.theta + put_greeks.theta,
                "vega": call_greeks.vega + put_greeks.vega
            }
        }
    
    @staticmethod
    def iron_condor(
        S: float, K1: float, K2: float, K3: float, K4: float,
        T: float, r: float, sigma: float
    ) -> Dict[str, Any]:
        """Build iron condor strategy (K1 < K2 < K3 < K4)."""
        # Sell put at K2, buy put at K1
        # Sell call at K3, buy call at K4
        put_sell_prem = BlackScholesCalculator.put_price(S, K2, T, r, sigma)
        put_buy_prem = BlackScholesCalculator.put_price(S, K1, T, r, sigma)
        call_sell_prem = BlackScholesCalculator.call_price(S, K3, T, r, sigma)
        call_buy_prem = BlackScholesCalculator.call_price(S, K4, T, r, sigma)
        
        net_credit = put_sell_prem - put_buy_prem + call_sell_prem - call_buy_prem
        
        return {
            "strategy": "Iron Condor",
            "legs": [
                {"type": "put", "action": "buy", "strike": K1, "premium": put_buy_prem},
                {"type": "put", "action": "sell", "strike": K2, "premium": put_sell_prem},
                {"type": "call", "action": "sell", "strike": K3, "premium": call_sell_prem},
                {"type": "call", "action": "buy", "strike": K4, "premium": call_buy_prem}
            ],
            "max_loss": (K2 - K1) - net_credit,
            "max_profit": net_credit,
            "breakeven_lower": K2 - net_credit,
            "breakeven_upper": K3 + net_credit,
            "profit_range": (K2, K3)
        }


# Export
__all__ = [
    "OptionType",
    "OptionStyle",
    "OptionContract",
    "Greeks",
    "OptionPricing",
    "BlackScholesCalculator",
    "ImpliedVolatilityCalculator",
    "OptionsChainAnalyzer",
    "OptionsStrategyBuilder"
]
