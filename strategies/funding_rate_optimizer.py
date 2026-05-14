"""
Funding Rate Optimizer
======================
Optimizes funding rate payments for perpetual futures:
- Monitors funding rates across exchanges
- Auto-switches between spot and futures
- Calculates optimal hedge ratios
- Identifies funding rate arbitrage opportunities
- Predicts funding rate changes

Funding rates are paid every 8 hours on most exchanges.
Positive rate = longs pay shorts (bullish sentiment)
Negative rate = shorts pay longs (bearish sentiment)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    DERIBIT = "deribit"
    DYDX = "dydx"
    GMX = "gmx"


class PositionType(Enum):
    """Position types."""
    SPOT = "spot"
    PERPETUAL_LONG = "perp_long"
    PERPETUAL_SHORT = "perp_short"
    HEDGED = "hedged"


@dataclass
class FundingRate:
    """Funding rate data."""
    exchange: Exchange
    symbol: str
    rate: float  # Per 8 hours
    predicted_rate: float = 0.0
    next_funding_time: float = 0.0
    annualized_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        # Calculate annualized rate (3 funding periods per day * 365 days)
        self.annualized_rate = self.rate * 3 * 365 * 100


@dataclass
class FundingPosition:
    """Position with funding implications."""
    symbol: str
    exchange: Exchange
    position_type: PositionType
    size: float
    entry_price: float
    current_price: float
    funding_rate: float = 0.0
    funding_paid: float = 0.0
    funding_received: float = 0.0
    net_funding: float = 0.0


@dataclass
class FundingOpportunity:
    """Funding rate arbitrage opportunity."""
    symbol: str
    long_exchange: Exchange
    short_exchange: Exchange
    long_rate: float
    short_rate: float
    rate_spread: float
    annualized_return: float
    estimated_profit_usd: float
    confidence: float


class FundingRateTracker:
    """
    Funding Rate Tracker
    ====================
    Tracks funding rates across exchanges.
    """
    
    def __init__(self):
        self.rates: Dict[str, Dict[Exchange, FundingRate]] = {}
        self.rate_history: Dict[str, List[FundingRate]] = {}
        self.funding_schedule = 8 * 3600  # 8 hours in seconds
    
    def update_rate(self, funding_rate: FundingRate) -> None:
        """Update funding rate for a symbol."""
        symbol = funding_rate.symbol
        
        if symbol not in self.rates:
            self.rates[symbol] = {}
        if symbol not in self.rate_history:
            self.rate_history[symbol] = []
        
        self.rates[symbol][funding_rate.exchange] = funding_rate
        self.rate_history[symbol].append(funding_rate)
        
        # Keep only recent history
        if len(self.rate_history[symbol]) > 1000:
            self.rate_history[symbol] = self.rate_history[symbol][-1000:]
    
    def get_best_funding_rate(self, symbol: str, side: str) -> Tuple[Exchange, float]:
        """Get best funding rate for a side.
        
        Args:
            symbol: Trading pair
            side: "long" or "short"
        
        Returns:
            Tuple of (exchange, rate)
        """
        if symbol not in self.rates:
            return Exchange.BINANCE, 0.0
        
        rates = self.rates[symbol]
        
        if side == "long":
            # Longs PAY positive rates, RECEIVE negative rates
            # Best = most negative (receive payment)
            best_exchange = min(rates.items(), key=lambda x: x[1].rate)
        else:
            # Shorts RECEIVE positive rates, PAY negative rates
            # Best = most positive (receive payment)
            best_exchange = max(rates.items(), key=lambda x: x[1].rate)
        
        return best_exchange[0], best_exchange[1].rate
    
    def predict_funding_rate(self, symbol: str, exchange: Exchange) -> float:
        """Predict next funding rate based on history."""
        if symbol not in self.rate_history:
            return 0.0
        
        history = self.rate_history[symbol]
        exchange_history = [r.rate for r in history if r.exchange == exchange]
        
        if len(exchange_history) < 3:
            return exchange_history[-1] if exchange_history else 0.0
        
        # Simple prediction: weighted average of recent rates
        weights = np.exp(np.linspace(-1, 0, min(10, len(exchange_history))))
        recent = exchange_history[-len(weights):]
        
        predicted = np.average(recent, weights=weights)
        return float(predicted)
    
    def get_rate_volatility(self, symbol: str, exchange: Exchange) -> float:
        """Calculate funding rate volatility."""
        if symbol not in self.rate_history:
            return 0.0
        
        history = self.rate_history[symbol]
        exchange_history = [r.rate for r in history if r.exchange == exchange]
        
        if len(exchange_history) < 2:
            return 0.0
        
        return float(np.std(exchange_history))


class FundingOptimizer:
    """
    Funding Rate Optimizer
    ======================
    Optimizes positions based on funding rates.
    """
    
    def __init__(self, min_funding_rate: float = 0.0001):
        self.tracker = FundingRateTracker()
        self.min_funding_rate = min_funding_rate  # 0.01% minimum
        self.positions: Dict[str, FundingPosition] = {}
        self.total_funding_earned: float = 0.0
        
    def should_hedge(self, symbol: str) -> Tuple[bool, str, float]:
        """Determine if spot position should be hedged with perps.
        
        Returns:
            Tuple of (should_hedge, reason, expected_return)
        """
        if symbol not in self.tracker.rates:
            return False, "No funding data", 0.0
        
        rates = self.tracker.rates[symbol]
        
        # Find extreme funding rates
        max_rate = max(r.rate for r in rates.values())
        min_rate = min(r.rate for r in rates.values())
        
        # Check if funding rate is extreme enough to hedge
        if max_rate > self.min_funding_rate:
            # Positive funding = longs pay shorts
            # Strategy: Hold spot + short perp to earn funding
            annualized = max_rate * 3 * 365 * 100
            return True, f"High positive funding ({max_rate:.4%})", annualized
        
        elif min_rate < -self.min_funding_rate:
            # Negative funding = shorts pay longs
            # Strategy: Short spot (or hold stable) + long perp to earn funding
            annualized = abs(min_rate) * 3 * 365 * 100
            return True, f"High negative funding ({min_rate:.4%})", annualized
        
        return False, "Funding rate neutral", 0.0
    
    def calculate_optimal_hedge(
        self,
        symbol: str,
        spot_value: float,
        current_price: float
    ) -> Dict[str, Any]:
        """Calculate optimal hedge ratio."""
        should_hedge, reason, expected_return = self.should_hedge(symbol)
        
        if not should_hedge:
            return {
                "hedge_needed": False,
                "reason": reason,
                "perp_position": 0,
                "perp_side": None
            }
        
        # Get best exchange for funding
        rates = self.tracker.rates.get(symbol, {})
        
        if not rates:
            return {"hedge_needed": False, "reason": "No rates", "perp_position": 0}
        
        # Determine hedge direction based on funding
        avg_rate = np.mean([r.rate for r in rates.values()])
        
        if avg_rate > 0:
            # Positive funding: short perp to earn
            perp_side = "short"
            best_exchange, best_rate = self.tracker.get_best_funding_rate(symbol, "short")
        else:
            # Negative funding: long perp to earn
            perp_side = "long"
            best_exchange, best_rate = self.tracker.get_best_funding_rate(symbol, "long")
        
        # Calculate position size
        perp_size = spot_value / current_price
        
        # Calculate expected funding income
        funding_per_period = perp_size * current_price * abs(best_rate)
        daily_income = funding_per_period * 3  # 3 periods per day
        monthly_income = daily_income * 30
        
        return {
            "hedge_needed": True,
            "reason": reason,
            "perp_side": perp_side,
            "perp_size": perp_size,
            "best_exchange": best_exchange.value,
            "funding_rate": best_rate,
            "expected_daily_income": daily_income,
            "expected_monthly_income": monthly_income,
            "annualized_return_pct": expected_return
        }
    
    def find_funding_arbitrage(self, symbol: str) -> List[FundingOpportunity]:
        """Find funding rate arbitrage across exchanges."""
        if symbol not in self.tracker.rates:
            return []
        
        rates = self.tracker.rates[symbol]
        opportunities = []
        
        exchanges = list(rates.keys())
        
        for i, ex1 in enumerate(exchanges):
            for ex2 in exchanges[i+1:]:
                rate1 = rates[ex1].rate
                rate2 = rates[ex2].rate
                
                # Long on exchange with lower rate, short on higher rate
                if rate2 > rate1 + 0.0001:  # 0.01% spread minimum
                    spread = rate2 - rate1
                    annualized = spread * 3 * 365 * 100
                    
                    opp = FundingOpportunity(
                        symbol=symbol,
                        long_exchange=ex1,
                        short_exchange=ex2,
                        long_rate=rate1,
                        short_rate=rate2,
                        rate_spread=spread,
                        annualized_return=annualized,
                        estimated_profit_usd=0,  # Would need position size
                        confidence=0.7
                    )
                    opportunities.append(opp)
        
        # Sort by annualized return
        opportunities.sort(key=lambda x: x.annualized_return, reverse=True)
        
        return opportunities
    
    def calculate_funding_pnl(self, position: FundingPosition) -> Dict[str, Any]:
        """Calculate funding P&L for a position."""
        if position.position_type == PositionType.SPOT:
            return {"funding_paid": 0, "funding_received": 0, "net_funding": 0}
        
        # Funding is paid/received based on position direction and rate sign
        funding_amount = position.size * position.current_price * position.funding_rate
        
        if position.position_type == PositionType.PERPETUAL_LONG:
            if position.funding_rate > 0:
                # Long pays when rate is positive
                funding_paid = funding_amount
                funding_received = 0
            else:
                # Long receives when rate is negative
                funding_paid = 0
                funding_received = abs(funding_amount)
        else:  # PERPETUAL_SHORT
            if position.funding_rate > 0:
                # Short receives when rate is positive
                funding_paid = 0
                funding_received = funding_amount
            else:
                # Short pays when rate is negative
                funding_paid = abs(funding_amount)
                funding_received = 0
        
        net_funding = funding_received - funding_paid
        
        return {
            "funding_paid": funding_paid,
            "funding_received": funding_received,
            "net_funding": net_funding
        }
    
    def get_funding_report(self) -> Dict[str, Any]:
        """Get funding rate report."""
        report = {
            "total_positions": len(self.positions),
            "total_funding_earned": self.total_funding_earned,
            "symbols_tracked": len(self.tracker.rates),
            "current_rates": {},
            "opportunities": []
        }
        
        # Current rates for major symbols
        for symbol, rates in self.tracker.rates.items():
            report["current_rates"][symbol] = {
                ex.value: {
                    "rate": r.rate,
                    "annualized": r.annualized_rate
                }
                for ex, r in rates.items()
            }
        
        # Find opportunities for major symbols
        major_symbols = ["BTC", "ETH", "SOL"]
        for symbol in major_symbols:
            opps = self.find_funding_arbitrage(symbol)
            report["opportunities"].extend([
                {
                    "symbol": o.symbol,
                    "long_exchange": o.long_exchange.value,
                    "short_exchange": o.short_exchange.value,
                    "spread": o.rate_spread,
                    "annualized": o.annualized_return
                }
                for o in opps[:3]  # Top 3 per symbol
            ])
        
        return report


class FundingRatePredictor:
    """
    Funding Rate Predictor
    ======================
    Predicts future funding rates using ML.
    """
    
    def __init__(self):
        self.model_weights = {}  # Simplified model
        self.features_history: List[Dict[str, float]] = []
    
    def extract_features(
        self,
        symbol: str,
        price_history: List[float],
        volume_history: List[float],
        open_interest: float,
        long_short_ratio: float
    ) -> Dict[str, float]:
        """Extract features for prediction."""
        if len(price_history) < 20:
            return {}
        
        prices = np.array(price_history[-20:])
        volumes = np.array(volume_history[-20:])
        
        returns = np.diff(np.log(prices))
        
        features = {
            "price_momentum_5d": (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0,
            "price_momentum_20d": (prices[-1] - prices[0]) / prices[0],
            "volatility": float(np.std(returns)),
            "volume_trend": float(np.mean(volumes[-5:]) / np.mean(volumes[-20:]) - 1),
            "open_interest": open_interest,
            "long_short_ratio": long_short_ratio,
            "price_acceleration": float(returns[-1] - returns[-2]) if len(returns) >= 2 else 0
        }
        
        return features
    
    def predict(
        self,
        symbol: str,
        features: Dict[str, float],
        current_rate: float
    ) -> float:
        """Predict next funding rate."""
        if not features:
            return current_rate
        
        # Simplified linear model
        # In production: use trained ML model
        
        prediction = current_rate
        
        # Momentum effect
        momentum = features.get("price_momentum_5d", 0)
        prediction += momentum * 0.0001  # Positive momentum -> higher funding
        
        # Volume effect
        volume_trend = features.get("volume_trend", 0)
        prediction += volume_trend * 0.00005
        
        # Long/short ratio effect
        ls_ratio = features.get("long_short_ratio", 1.0)
        if ls_ratio > 1.5:
            prediction += 0.0001  # Crowded longs -> higher funding
        elif ls_ratio < 0.7:
            prediction -= 0.0001  # Crowded shorts -> lower funding
        
        # Mean reversion
        prediction = prediction * 0.8 + current_rate * 0.2
        
        return float(np.clip(prediction, -0.01, 0.01))


# Export
__all__ = [
    "Exchange",
    "PositionType",
    "FundingRate",
    "FundingPosition",
    "FundingOpportunity",
    "FundingRateTracker",
    "FundingOptimizer",
    "FundingRatePredictor"
]
