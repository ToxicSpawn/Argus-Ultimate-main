"""
MARKET ADAPTATION SYSTEM
========================
Argus ADAPTS to market conditions - never controls them.

Core Philosophy:
- Follow the market, don't fight it
- Detect conditions early, adjust parameters
- Reduce risk in bad conditions, increase in good
- Use ALL available data to understand market state
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from enum import Enum
import time

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market conditions Argus adapts to."""
    BULL_STRONG = "bull_strong"       # Strong uptrend, low volatility
    BULL_WEAK = "bull_weak"           # Weak uptrend, moderate volatility
    SIDEWAYS = "sideways"             # Range-bound, mean-reversion works
    BEAR_WEAK = "bear_weak"           # Weak downtrend
    BEAR_STRONG = "bear_strong"       # Strong downtrend, high volatility
    HIGH_VOLATILITY = "high_vol"      # Volatile, news-driven
    LOW_LIQUIDITY = "low_liq"         # Thin orderbooks, wide spreads
    CRASH = "crash"                   # Flash crash, panic selling
    PUMP = "pump"                     # FOMO buying, euphoria


@dataclass
class MarketState:
    """Current market state snapshot."""
    condition: MarketCondition = MarketCondition.SIDEWAYS
    confidence: float = 0.5
    trend_strength: float = 0.0       # -1 to 1
    volatility: float = 0.02          # Daily volatility
    volume_ratio: float = 1.0         # vs average
    spread_bps: float = 5.0           # Bid-ask spread
    orderbook_imbalance: float = 0.0  # -1 to 1
    momentum: float = 0.0             # Short-term momentum
    mean_reversion_score: float = 0.0 # How mean-reverting
    liquidity_score: float = 1.0      # 0-1, higher = more liquid
    fear_greed: float = 0.5           # 0 = fear, 1 = greed
    timestamp: float = field(default_factory=time.time)


class MarketAdaptationSystem:
    """
    Main adaptation system - adjusts everything based on market conditions.
    
    What it adapts:
    1. Position sizes (smaller in bad conditions)
    2. Strategy selection (trend-following vs mean-reversion)
    3. Entry/exit thresholds (tighter in volatile markets)
    4. Risk limits (stricter in crashes)
    5. Trading frequency (less in choppy markets)
    """
    
    def __init__(self):
        self.current_state = MarketState()
        self.state_history: deque = deque(maxlen=10000)
        self.condition_durations: Dict[str, float] = {}
        self.last_condition_change = time.time()
        
        # Adaptation parameters
        self.position_multiplier = 1.0
        self.aggressiveness = 0.5
        self.risk_multiplier = 1.0
        self.strategy_weights: Dict[str, float] = {}
        
        # Detection thresholds
        self.trend_threshold = 0.3
        self.volatility_threshold = 0.03
        self.volume_spike_threshold = 2.0
        
        logger.info("MarketAdaptationSystem initialized - following market")
    
    async def analyze_market(
        self,
        price_data: List[float],
        volume_data: List[float],
        orderbook: Dict[str, List],
        trades: List[Dict] = None,
    ) -> MarketState:
        """
        Analyze current market conditions.
        
        Returns MarketState with all adaptation parameters.
        """
        if len(price_data) < 20:
            return self.current_state
        
        prices = np.array(price_data[-100:])
        volumes = np.array(volume_data[-100:]) if volume_data else np.ones(100)
        
        # Calculate all metrics
        returns = np.diff(np.log(prices))
        
        # Trend detection
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        trend_strength = (sma_20 - sma_50) / (sma_50 + 1e-10)
        
        # Volatility
        volatility = float(np.std(returns) * np.sqrt(252))
        
        # Volume analysis
        avg_volume = np.mean(volumes)
        current_volume = volumes[-1]
        volume_ratio = current_volume / (avg_volume + 1e-10)
        
        # Orderbook analysis
        spread_bps = self._calculate_spread(orderbook)
        imbalance = self._calculate_imbalance(orderbook)
        
        # Momentum (short-term)
        momentum = (prices[-1] - prices[-5]) / (prices[-5] + 1e-10)
        
        # Mean reversion score
        mean_reversion = self._calculate_mean_reversion(prices)
        
        # Liquidity score
        liquidity = self._calculate_liquidity(orderbook, volume_ratio)
        
        # Fear/Greed (simplified)
        fear_greed = self._calculate_fear_greed(
            trend_strength, volatility, volume_ratio, momentum
        )
        
        # Detect market condition
        condition, confidence = self._detect_condition(
            trend_strength=trend_strength,
            volatility=volatility,
            volume_ratio=volume_ratio,
            momentum=momentum,
            mean_reversion=mean_reversion,
            liquidity=liquidity,
        )
        
        # Update state
        self.current_state = MarketState(
            condition=condition,
            confidence=confidence,
            trend_strength=float(trend_strength),
            volatility=volatility,
            volume_ratio=float(volume_ratio),
            spread_bps=spread_bps,
            orderbook_imbalance=imbalance,
            momentum=float(momentum),
            mean_reversion_score=mean_reversion,
            liquidity_score=liquidity,
            fear_greed=fear_greed,
        )
        
        # Track condition duration
        self._update_condition_tracking(condition)
        
        # Store history
        self.state_history.append(self.current_state)
        
        # Adapt parameters based on condition
        self._adapt_parameters()
        
        return self.current_state
    
    def _detect_condition(
        self,
        trend_strength: float,
        volatility: float,
        volume_ratio: float,
        momentum: float,
        mean_reversion: float,
        liquidity: float,
    ) -> Tuple[MarketCondition, float]:
        """Detect current market condition."""
        scores = {}
        
        # Bull Strong: strong uptrend, low vol, high volume
        scores[MarketCondition.BULL_STRONG] = (
            max(0, trend_strength) * 0.4 +
            max(0, 1 - volatility / 0.5) * 0.3 +
            min(volume_ratio / 2, 1) * 0.3
        )
        
        # Bull Weak: weak uptrend
        scores[MarketCondition.BULL_WEAK] = (
            max(0, min(trend_strength, 0.3)) * 0.5 +
            (1 - abs(momentum)) * 0.3 +
            max(0, mean_reversion) * 0.2
        )
        
        # Bear Strong: strong downtrend, high vol
        scores[MarketCondition.BEAR_STRONG] = (
            max(0, -trend_strength) * 0.4 +
            min(volatility / 0.5, 1) * 0.3 +
            max(0, -momentum) * 0.3
        )
        
        # Bear Weak: weak downtrend
        scores[MarketCondition.BEAR_WEAK] = (
            max(0, min(-trend_strength, 0.3)) * 0.5 +
            (1 - abs(momentum)) * 0.3 +
            max(0, mean_reversion) * 0.2
        )
        
        # Sideways: mean-reverting, low trend
        scores[MarketCondition.SIDEWAYS] = (
            (1 - abs(trend_strength)) * 0.4 +
            max(0, mean_reversion) * 0.4 +
            (1 - abs(momentum)) * 0.2
        )
        
        # High Volatility: high vol, unpredictable
        scores[MarketCondition.HIGH_VOLATILITY] = (
            min(volatility / 0.8, 1) * 0.6 +
            (1 - abs(trend_strength)) * 0.2 +
            min(volume_ratio / 3, 1) * 0.2
        )
        
        # Low Liquidity: wide spreads, low volume
        scores[MarketCondition.LOW_LIQUIDITY] = (
            (1 - liquidity) * 0.6 +
            max(0, (1 - volume_ratio)) * 0.2 +
            min(volatility / 0.3, 1) * 0.2
        )
        
        # Crash: extreme negative momentum, high vol
        scores[MarketCondition.CRASH] = (
            max(0, -momentum * 10) * 0.5 +
            min(volatility / 1.0, 1) * 0.3 +
            min(volume_ratio / 5, 1) * 0.2
        )
        
        # Pump: extreme positive momentum, high volume
        scores[MarketCondition.PUMP] = (
            max(0, momentum * 10) * 0.5 +
            min(volume_ratio / 5, 1) * 0.3 +
            max(0, trend_strength) * 0.2
        )
        
        # Find best match
        best_condition = max(scores, key=scores.get)
        best_score = scores[best_condition]
        
        return best_condition, min(best_score, 1.0)
    
    def _adapt_parameters(self):
        """Adapt trading parameters based on current condition."""
        condition = self.current_state.condition
        
        # Position size multipliers by condition
        position_multipliers = {
            MarketCondition.BULL_STRONG: 1.2,      # Increase in strong bull
            MarketCondition.BULL_WEAK: 0.8,        # Reduce in weak bull
            MarketCondition.SIDEWAYS: 0.6,         # Reduce in choppy
            MarketCondition.BEAR_WEAK: 0.5,        # Reduce more in weak bear
            MarketCondition.BEAR_STRONG: 0.3,      # Minimal in strong bear
            MarketCondition.HIGH_VOLATILITY: 0.4,  # Reduce in volatile
            MarketCondition.LOW_LIQUIDITY: 0.3,    # Minimal in illiquid
            MarketCondition.CRASH: 0.1,            # Almost nothing in crash
            MarketCondition.PUMP: 0.7,             # Moderate in pump (FOMO risk)
        }
        
        self.position_multiplier = position_multipliers.get(condition, 0.5)
        
        # Aggressiveness by condition
        aggressiveness = {
            MarketCondition.BULL_STRONG: 0.8,
            MarketCondition.BULL_WEAK: 0.5,
            MarketCondition.SIDEWAYS: 0.4,
            MarketCondition.BEAR_WEAK: 0.3,
            MarketCondition.BEAR_STRONG: 0.2,
            MarketCondition.HIGH_VOLATILITY: 0.3,
            MarketCondition.LOW_LIQUIDITY: 0.2,
            MarketCondition.CRASH: 0.1,
            MarketCondition.PUMP: 0.5,
        }
        
        self.aggressiveness = aggressiveness.get(condition, 0.5)
        
        # Risk multiplier (lower = stricter risk limits)
        risk_multipliers = {
            MarketCondition.BULL_STRONG: 1.0,
            MarketCondition.BULL_WEAK: 0.8,
            MarketCondition.SIDEWAYS: 0.7,
            MarketCondition.BEAR_WEAK: 0.6,
            MarketCondition.BEAR_STRONG: 0.4,
            MarketCondition.HIGH_VOLATILITY: 0.5,
            MarketCondition.LOW_LIQUIDITY: 0.4,
            MarketCondition.CRASH: 0.2,
            MarketCondition.PUMP: 0.6,
        }
        
        self.risk_multiplier = risk_multipliers.get(condition, 0.5)
        
        # Strategy weights by condition
        self.strategy_weights = self._get_strategy_weights(condition)
        
        logger.debug(
            f"Adapted to {condition.value}: "
            f"position={self.position_multiplier:.2f}, "
            f"aggro={self.aggressiveness:.2f}, "
            f"risk={self.risk_multiplier:.2f}"
        )
    
    def _get_strategy_weights(self, condition: MarketCondition) -> Dict[str, float]:
        """Get strategy weights for current condition."""
        # Different strategies work better in different conditions
        weights = {
            MarketCondition.BULL_STRONG: {
                "trend_following": 0.6,
                "momentum": 0.3,
                "breakout": 0.1,
                "mean_reversion": 0.0,
            },
            MarketCondition.BULL_WEAK: {
                "trend_following": 0.3,
                "momentum": 0.2,
                "mean_reversion": 0.3,
                "swing": 0.2,
            },
            MarketCondition.SIDEWAYS: {
                "mean_reversion": 0.6,
                "swing": 0.3,
                "grid": 0.1,
                "trend_following": 0.0,
            },
            MarketCondition.BEAR_WEAK: {
                "mean_reversion": 0.4,
                "swing": 0.3,
                "trend_following": 0.2,
                "momentum": 0.1,
            },
            MarketCondition.BEAR_STRONG: {
                "trend_following": 0.5,  # Short-side trend following
                "momentum": 0.3,
                "mean_reversion": 0.2,
            },
            MarketCondition.HIGH_VOLATILITY: {
                "volatility": 0.4,
                "breakout": 0.3,
                "mean_reversion": 0.2,
                "trend_following": 0.1,
            },
            MarketCondition.LOW_LIQUIDITY: {
                "mean_reversion": 0.5,
                "grid": 0.3,
                "swing": 0.2,
            },
            MarketCondition.CRASH: {
                "mean_reversion": 0.4,  # Buy the dip (carefully)
                "volatility": 0.3,
                "trend_following": 0.3,  # Short-side
            },
            MarketCondition.PUMP: {
                "momentum": 0.4,
                "trend_following": 0.3,
                "breakout": 0.2,
                "mean_reversion": 0.1,  # Don't fight the pump
            },
        }
        
        return weights.get(condition, {"mean_reversion": 0.5, "trend_following": 0.5})
    
    def _calculate_spread(self, orderbook: Dict[str, List]) -> float:
        """Calculate bid-ask spread in basis points."""
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return 10.0  # Default
        
        best_bid = bids[0][0] if isinstance(bids[0], list) else bids[0]
        best_ask = asks[0][0] if isinstance(asks[0], list) else asks[0]
        
        mid = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid) / mid * 10000  # bps
        
        return float(spread)
    
    def _calculate_imbalance(self, orderbook: Dict[str, List]) -> float:
        """Calculate orderbook imbalance (-1 to 1)."""
        bids = orderbook.get("bids", [])[:10]
        asks = orderbook.get("asks", [])[:10]
        
        bid_volume = sum(b[1] for b in bids) if bids else 0
        ask_volume = sum(a[1] for a in asks) if asks else 0
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    def _calculate_mean_reversion(self, prices: np.ndarray) -> float:
        """Calculate mean reversion score."""
        if len(prices) < 20:
            return 0.0
        
        # Hurst exponent (simplified)
        # H < 0.5 = mean reverting, H > 0.5 = trending
        returns = np.diff(np.log(prices))
        
        # Autocorrelation at lag 1
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0.0
        
        # Negative autocorrelation = mean reverting
        return float(-autocorr)  # Flip sign so positive = mean reverting
    
    def _calculate_liquidity(self, orderbook: Dict[str, List], volume_ratio: float) -> float:
        """Calculate liquidity score (0-1)."""
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # Orderbook depth
        bid_depth = len(bids)
        ask_depth = len(asks)
        depth_score = min((bid_depth + ask_depth) / 40, 1.0)
        
        # Volume
        volume_score = min(volume_ratio / 2, 1.0)
        
        # Combined
        return (depth_score + volume_score) / 2
    
    def _calculate_fear_greed(
        self,
        trend_strength: float,
        volatility: float,
        volume_ratio: float,
        momentum: float,
    ) -> float:
        """Calculate fear/greed index (0-1)."""
        # Greed indicators
        greed = 0.0
        greed += max(0, trend_strength) * 0.3
        greed += max(0, momentum * 5) * 0.3
        greed += min(volume_ratio / 3, 1) * 0.2
        greed += max(0, (1 - volatility / 0.5)) * 0.2
        
        # Fear indicators
        fear = 0.0
        fear += max(0, -trend_strength) * 0.3
        fear += max(0, -momentum * 5) * 0.3
        fear += min(volatility / 0.5, 1) * 0.2
        fear += max(0, (volume_ratio - 2) / 3) * 0.2  # Panic selling
        
        # Normalize
        total = greed + fear
        if total == 0:
            return 0.5
        
        return greed / total
    
    def _update_condition_tracking(self, condition: MarketCondition):
        """Track how long we've been in each condition."""
        current_time = time.time()
        
        if self.current_state.condition != condition:
            # Condition changed
            duration = current_time - self.last_condition_change
            cond_name = self.current_state.condition.value
            self.condition_durations[cond_name] = duration
            self.last_condition_change = current_time
    
    def get_adaptation_params(self) -> Dict[str, Any]:
        """Get current adaptation parameters."""
        return {
            "condition": self.current_state.condition.value,
            "confidence": self.current_state.confidence,
            "position_multiplier": self.position_multiplier,
            "aggressiveness": self.aggressiveness,
            "risk_multiplier": self.risk_multiplier,
            "strategy_weights": self.strategy_weights,
            "state": {
                "trend_strength": self.current_state.trend_strength,
                "volatility": self.current_state.volatility,
                "volume_ratio": self.current_state.volume_ratio,
                "spread_bps": self.current_state.spread_bps,
                "liquidity": self.current_state.liquidity_score,
                "fear_greed": self.current_state.fear_greed,
            }
        }
    
    def should_trade(self) -> Tuple[bool, str]:
        """Determine if we should trade in current conditions."""
        condition = self.current_state.condition
        
        # Never trade in extreme conditions
        if condition == MarketCondition.CRASH:
            return False, "Crash detected - waiting for stability"
        
        if condition == MarketCondition.LOW_LIQUIDITY and self.current_state.spread_bps > 50:
            return False, "Spread too wide (>50 bps)"
        
        # Reduce trading in bad conditions
        if condition == MarketCondition.BEAR_STRONG and self.current_state.confidence < 0.7:
            return False, "Strong bear market - preserving capital"
        
        return True, f"Trading allowed in {condition.value}"
    
    def get_position_size(self, base_size: float) -> float:
        """Get adapted position size."""
        return base_size * self.position_multiplier
    
    def get_stop_loss(self, base_stop_pct: float) -> float:
        """Get adapted stop loss (tighter in volatile markets)."""
        volatility_factor = 1.0 + self.current_state.volatility * 10
        return base_stop_pct * volatility_factor * self.risk_multiplier
    
    def get_take_profit(self, base_tp_pct: float) -> float:
        """Get adapted take profit."""
        # In trending markets, let winners run
        if self.current_state.condition in [MarketCondition.BULL_STRONG, MarketCondition.BEAR_STRONG]:
            return base_tp_pct * 2.0
        # In sideways, take profits quicker
        elif self.current_state.condition == MarketCondition.SIDEWAYS:
            return base_tp_pct * 0.7
        return base_tp_pct


# =============================================================================
# QUICK ADAPTATION FUNCTIONS
# =============================================================================

def get_market_adaptation() -> MarketAdaptationSystem:
    """Get market adaptation system instance."""
    return MarketAdaptationSystem()


async def quick_adapt(
    price_data: List[float],
    volume_data: List[float] = None,
    orderbook: Dict = None,
) -> Dict[str, Any]:
    """Quick market adaptation check."""
    adapter = get_market_adaptation()
    
    if volume_data is None:
        volume_data = [1.0] * len(price_data)
    
    if orderbook is None:
        orderbook = {"bids": [[50000, 1]], "asks": [[50010, 1]]}
    
    state = await adapter.analyze_market(price_data, volume_data, orderbook)
    
    return {
        "condition": state.condition.value,
        "confidence": state.confidence,
        "position_multiplier": adapter.position_multiplier,
        "should_trade": adapter.should_trade()[0],
        "reason": adapter.should_trade()[1],
    }
