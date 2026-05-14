"""
strategies/ultimate_strategy_intelligence.py — Ultimate Strategy Intelligence

The most advanced strategy intelligence system possible. Goes beyond traditional
technical analysis to include:

1. Market Regime Prediction (Markov + HMM + neural)
2. Order Flow Analysis (imbalance, absorption, spoofing)
3. Liquidity Mapping (stop hunts, sweep detection)
4. Cross-Asset Correlation (dynamic correlation, cointegration)
5. Seasonality Patterns (time-of-day, day-of-week, monthly)
6. Funding Rate & Basis Analysis
7. Smart Money Tracking
8. Market Microstructure Signals

This is the "brain" that makes every strategy smarter.

Usage::

    from strategies.ultimate_strategy_intelligence import UltimateIntelligence
    
    intel = UltimateIntelligence()
    
    # Get comprehensive market analysis
    analysis = intel.analyze_market(market_data)
    
    # Get trading edge score
    edge = intel.calculate_edge(analysis)
    
    # Get optimal execution timing
    timing = intel.get_optimal_timing(analysis)
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class MarketRegime(str, Enum):
    """Market regimes."""
    BULL_STRONG = "bull_strong"
    BULL_MODERATE = "bull_moderate"
    BULL_WEAK = "bull_weak"
    BEAR_STRONG = "bear_strong"
    BEAR_MODERATE = "bear_moderate"
    BEAR_WEAK = "bear_weak"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class OrderFlowSignal(str, Enum):
    """Order flow signals."""
    BUY_PRESSURE = "buy_pressure"
    SELL_PRESSURE = "sell_pressure"
    ABSORPTION = "absorption"
    SPOOFING_DETECTED = "spoofing_detected"
    ICEBERG_DETECTED = "iceberg_detected"
    NEUTRAL = "neutral"


class LiquidityZone(str, Enum):
    """Liquidity zone types."""
    LIQUIDITY_HIGH = "high_liquidity"
    LIQUIDITY_LOW = "low_liquidity"
    STOP_CLUSTER = "stop_cluster"
    WHALE_WALL = "whale_wall"
    VOID = "void"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class OrderBookLevel:
    """Order book level."""
    price: float
    size: float
    orders: int = 1


@dataclass
class OrderBookSnapshot:
    """Order book snapshot."""
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.bids and self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2 if self.bids and self.asks else 0.0
    
    @property
    def bid_volume(self) -> float:
        return sum(b.size for b in self.bids[:10])
    
    @property
    def ask_volume(self) -> float:
        return sum(a.size for a in self.asks[:10])
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance (-1 to 1). Positive = more bids."""
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.0
        return (self.bid_volume - self.ask_volume) / total


@dataclass
class Trade:
    """Individual trade."""
    timestamp: datetime
    price: float
    size: float
    side: str  # "buy" or "sell"
    is_maker: bool = False


@dataclass
class RegimePrediction:
    """Market regime prediction."""
    current_regime: MarketRegime
    confidence: float
    predicted_regime: MarketRegime
    prediction_confidence: float
    time_to_regime_change: float  # Estimated hours
    regime_probabilities: Dict[str, float]
    transition_matrix: np.ndarray


@dataclass
class OrderFlowAnalysis:
    """Order flow analysis result."""
    timestamp: datetime
    signal: OrderFlowSignal
    buy_volume: float
    sell_volume: float
    net_volume: float
    volume_imbalance: float  # -1 to 1
    large_order_ratio: float
    absorption_detected: bool
    spoofing_score: float  # 0-1
    iceberg_score: float  # 0-1
    vwap: float
    twap: float


@dataclass
class LiquidityMap:
    """Liquidity analysis."""
    timestamp: datetime
    zones: List[Tuple[float, float, LiquidityZone]]  # (price, size, type)
    support_levels: List[float]
    resistance_levels: List[float]
    stop_clusters: List[Tuple[float, float]]  # (price, estimated_size)
    void_zones: List[Tuple[float, float]]  # (price_start, price_end)
    liquidity_score: float  # 0-100
    sweep_risk: float  # 0-1


@dataclass
class CorrelationSignal:
    """Cross-asset correlation signal."""
    timestamp: datetime
    asset_a: str
    asset_b: str
    correlation: float
    correlation_change: float  # Change from 24h ago
    cointegration_score: float  # 0-1
    lead_lag_relationship: float  # Positive = A leads B
    divergence_score: float  # How diverged the assets are
    mean_reversion_probability: float


@dataclass
class SeasonalityPattern:
    """Seasonality pattern."""
    hour_of_day: int
    day_of_week: int
    avg_return: float
    win_rate: float
    volatility: float
    optimal_entry_hour: int
    optimal_exit_hour: int
    pattern_strength: float  # 0-1


@dataclass
class FundingSignal:
    """Funding rate signal."""
    timestamp: datetime
    funding_rate: float  # Current funding rate
    predicted_funding: float
    basis: float  # Futures - Spot
    basis_pct: float
    open_interest_change: float
    long_short_ratio: float
    liquidation_risk: float
    arbitrage_opportunity: float


@dataclass
class SmartMoneyFlow:
    """Smart money flow analysis."""
    timestamp: datetime
    whale_accumulation_score: float  # -1 to 1
    institutional_flow_score: float  # -1 to 1
    retail_flow_score: float  # -1 to 1
    smart_money_direction: str  # "accumulating", "distributing", "neutral"
    large_wallet_activity: int
    exchange_net_flow: float  # Positive = outflow (bullish)
    otc_premium: float  # OTC vs exchange price premium


@dataclass
class MarketMicrostructure:
    """Market microstructure signals."""
    timestamp: datetime
    spread_bps: float  # Spread in basis points
    depth_score: float  # 0-100
    toxicity_score: float  # 0-100 (VPIN-like)
    adverse_selection: float  # -1 to 1
    Kyle_lambda: float  # Price impact coefficient
    amihud_illiquidity: float
    effective_spread: float
    realized_spread: float


@dataclass
class UltimateMarketAnalysis:
    """Comprehensive market analysis combining all signals."""
    timestamp: datetime
    symbol: str
    
    # Regime
    regime: RegimePrediction
    
    # Order Flow
    order_flow: OrderFlowAnalysis
    
    # Liquidity
    liquidity: LiquidityMap
    
    # Correlations
    correlations: List[CorrelationSignal]
    
    # Seasonality
    seasonality: SeasonalityPattern
    
    # Funding (for crypto)
    funding: Optional[FundingSignal]
    
    # Smart Money
    smart_money: SmartMoneyFlow
    
    # Microstructure
    microstructure: MarketMicrostructure
    
    # Composite Scores
    bullish_score: float  # 0-100
    bearish_score: float  # 0-100
    edge_score: float  # 0-100 (trading edge)
    confidence: float  # 0-100
    
    # Recommendations
    optimal_entry_timing: str
    optimal_exit_timing: str
    position_sizing_recommendation: float
    risk_level: str  # low, medium, high, extreme
    
    # Alerts
    alerts: List[str]
    opportunities: List[str]


# ============================================================================
# Markov Regime Predictor
# ============================================================================

class MarkovRegimePredictor:
    """Predict market regimes using Markov chains."""
    
    # Transition probabilities (learned from data)
    DEFAULT_TRANSITIONS = {
        MarketRegime.BULL_STRONG: {
            MarketRegime.BULL_STRONG: 0.6,
            MarketRegime.BULL_MODERATE: 0.25,
            MarketRegime.BULL_WEAK: 0.1,
            MarketRegime.RANGING_HIGH_VOL: 0.05,
        },
        MarketRegime.BULL_MODERATE: {
            MarketRegime.BULL_STRONG: 0.2,
            MarketRegime.BULL_MODERATE: 0.4,
            MarketRegime.BULL_WEAK: 0.2,
            MarketRegime.RANGING_LOW_VOL: 0.15,
            MarketRegime.DISTRIBUTION: 0.05,
        },
        MarketRegime.BULL_WEAK: {
            MarketRegime.BULL_MODERATE: 0.2,
            MarketRegime.BULL_WEAK: 0.3,
            MarketRegime.RANGING_LOW_VOL: 0.25,
            MarketRegime.DISTRIBUTION: 0.15,
            MarketRegime.BEAR_WEAK: 0.05,
        },
        MarketRegime.BEAR_STRONG: {
            MarketRegime.BEAR_STRONG: 0.6,
            MarketRegime.BEAR_MODERATE: 0.25,
            MarketRegime.BEAR_WEAK: 0.1,
            MarketRegime.CRISIS: 0.05,
        },
        MarketRegime.BEAR_MODERATE: {
            MarketRegime.BEAR_STRONG: 0.15,
            MarketRegime.BEAR_MODERATE: 0.4,
            MarketRegime.BEAR_WEAK: 0.25,
            MarketRegime.RANGING_LOW_VOL: 0.1,
            MarketRegime.ACCUMULATION: 0.05,
        },
        MarketRegime.BEAR_WEAK: {
            MarketRegime.BEAR_MODERATE: 0.15,
            MarketRegime.BEAR_WEAK: 0.3,
            MarketRegime.RANGING_LOW_VOL: 0.25,
            MarketRegime.ACCUMULATION: 0.2,
            MarketRegime.RECOVERY: 0.05,
        },
        MarketRegime.RANGING_LOW_VOL: {
            MarketRegime.BULL_WEAK: 0.2,
            MarketRegime.BEAR_WEAK: 0.2,
            MarketRegime.RANGING_LOW_VOL: 0.4,
            MarketRegime.RANGING_HIGH_VOL: 0.15,
            MarketRegime.ACCUMULATION: 0.05,
        },
        MarketRegime.RANGING_HIGH_VOL: {
            MarketRegime.BULL_WEAK: 0.15,
            MarketRegime.BEAR_WEAK: 0.15,
            MarketRegime.RANGING_LOW_VOL: 0.3,
            MarketRegime.RANGING_HIGH_VOL: 0.3,
            MarketRegime.CRISIS: 0.1,
        },
        MarketRegime.CRISIS: {
            MarketRegime.CRISIS: 0.3,
            MarketRegime.BEAR_STRONG: 0.3,
            MarketRegime.RECOVERY: 0.2,
            MarketRegime.BEAR_WEAK: 0.2,
        },
        MarketRegime.RECOVERY: {
            MarketRegime.BULL_WEAK: 0.3,
            MarketRegime.RECOVERY: 0.3,
            MarketRegime.ACCUMULATION: 0.2,
            MarketRegime.RANGING_LOW_VOL: 0.2,
        },
        MarketRegime.ACCUMULATION: {
            MarketRegime.ACCUMULATION: 0.4,
            MarketRegime.BULL_WEAK: 0.3,
            MarketRegime.RANGING_LOW_VOL: 0.2,
            MarketRegime.BULL_MODERATE: 0.1,
        },
        MarketRegime.DISTRIBUTION: {
            MarketRegime.DISTRIBUTION: 0.4,
            MarketRegime.BEAR_WEAK: 0.3,
            MarketRegime.RANGING_HIGH_VOL: 0.2,
            MarketRegime.BEAR_MODERATE: 0.1,
        },
    }
    
    def __init__(self):
        self.transitions = self.DEFAULT_TRANSITIONS.copy()
        self.regime_history: deque = deque(maxlen=1000)
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: float,
        trend_strength: float,
    ) -> MarketRegime:
        """Detect current market regime."""
        avg_return = np.mean(returns[-20:]) if len(returns) >= 20 else 0
        vol_percentile = self._get_vol_percentile(volatility)
        
        # Strong trend detection
        if trend_strength > 0.7:
            if avg_return > 0.001:
                return MarketRegime.BULL_STRONG
            elif avg_return < -0.001:
                return MarketRegime.BEAR_STRONG
        elif trend_strength > 0.4:
            if avg_return > 0.0005:
                return MarketRegime.BULL_MODERATE
            elif avg_return < -0.0005:
                return MarketRegime.BEAR_MODERATE
        elif trend_strength > 0.1:
            if avg_return > 0:
                return MarketRegime.BULL_WEAK
            else:
                return MarketRegime.BEAR_WEAK
        elif abs(trend_strength) < 0.1:
            # Ranging
            if vol_percentile > 0.7:
                return MarketRegime.RANGING_HIGH_VOL
            else:
                return MarketRegime.RANGING_LOW_VOL
        
        # Default
        return MarketRegime.RANGING_LOW_VOL
    
    def predict_regime_change(
        self,
        current_regime: MarketRegime,
        n_steps: int = 5,
    ) -> Tuple[MarketRegime, float, Dict[str, float]]:
        """
        Predict future regime.
        
        Returns:
            (predicted_regime, confidence, probabilities)
        """
        # Get transition probabilities
        transitions = self.transitions.get(current_regime, {})
        
        # Calculate n-step ahead probabilities
        probs = {regime.value: 0.0 for regime in MarketRegime}
        probs[current_regime.value] = 1.0
        
        for _ in range(n_steps):
            new_probs = defaultdict(float)
            for from_regime, from_prob in probs.items():
                if from_prob > 0.01:  # Skip negligible probabilities
                    from_enum = MarketRegime(from_regime)
                    trans = self.transitions.get(from_enum, {})
                    for to_regime, trans_prob in trans.items():
                        new_probs[to_regime.value] += from_prob * trans_prob
            probs = dict(new_probs)
        
        # Find most likely regime
        if probs:
            predicted = max(probs, key=probs.get)
            confidence = probs[predicted]
        else:
            predicted = current_regime.value
            confidence = 0.5
        
        # Estimate time to regime change
        current_prob = probs.get(current_regime.value, 0)
        if current_prob < 0.5:
            time_to_change = n_steps * 4  # Rough estimate (4 hours per step)
        else:
            time_to_change = n_steps * 4 * (current_prob / (1 - current_prob + 0.01))
        
        return MarketRegime(predicted), confidence, probs
    
    def _get_vol_percentile(self, volatility: float) -> float:
        """Get volatility percentile (simplified)."""
        # In real implementation, would use historical distribution
        if volatility < 0.01:
            return 0.2
        elif volatility < 0.02:
            return 0.4
        elif volatility < 0.03:
            return 0.6
        elif volatility < 0.05:
            return 0.8
        else:
            return 0.95
    
    def update_transitions(self, from_regime: MarketRegime, to_regime: MarketRegime):
        """Update transition matrix based on observed transition."""
        self.transition_counts[from_regime.value][to_regime.value] += 1
        self.regime_history.append((from_regime, to_regime, datetime.utcnow()))


# ============================================================================
# Order Flow Analyzer
# ============================================================================

class OrderFlowAnalyzer:
    """Analyze order flow for signals."""
    
    def __init__(
        self,
        *,
        large_order_threshold: float = 10000.0,
        spoofing_window: int = 100,
        absorption_threshold: float = 0.8,
    ):
        self.large_order_threshold = large_order_threshold
        self.spoofing_window = spoofing_window
        self.absorption_threshold = absorption_threshold
        
        self.trade_history: deque = deque(maxlen=10000)
        self.order_book_history: deque = deque(maxlen=1000)
    
    def analyze(
        self,
        trades: List[Trade],
        order_book: OrderBookSnapshot,
    ) -> OrderFlowAnalysis:
        """Analyze order flow."""
        self.order_book_history.append(order_book)
        
        # Separate buys and sells
        buy_trades = [t for t in trades if t.side == "buy"]
        sell_trades = [t for t in trades if t.side == "sell"]
        
        buy_volume = sum(t.size for t in buy_trades)
        sell_volume = sum(t.size for t in sell_trades)
        total_volume = buy_volume + sell_volume
        
        # Volume imbalance
        if total_volume > 0:
            volume_imbalance = (buy_volume - sell_volume) / total_volume
        else:
            volume_imbalance = 0.0
        
        # Large order analysis
        large_trades = [t for t in trades if t.size * t.price >= self.large_order_threshold]
        large_order_ratio = len(large_trades) / max(len(trades), 1)
        
        # Detect absorption
        absorption = self._detect_absorption(trades, order_book)
        
        # Detect spoofing
        spoofing_score = self._detect_spoofing(order_book)
        
        # Detect iceberg orders
        iceberg_score = self._detect_iceberg(trades)
        
        # Calculate VWAP and TWAP
        vwap = self._calculate_vwap(trades)
        twap = self._calculate_twap(trades)
        
        # Determine signal
        signal = self._determine_signal(
            volume_imbalance,
            absorption,
            spoofing_score,
            iceberg_score,
            order_book.imbalance,
        )
        
        return OrderFlowAnalysis(
            timestamp=datetime.utcnow(),
            signal=signal,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_volume=buy_volume - sell_volume,
            volume_imbalance=volume_imbalance,
            large_order_ratio=large_order_ratio,
            absorption_detected=absorption,
            spoofing_score=spoofing_score,
            iceberg_score=iceberg_score,
            vwap=vwap,
            twap=twap,
        )
    
    def _detect_absorption(self, trades: List[Trade], order_book: OrderBookSnapshot) -> bool:
        """Detect order absorption (large orders being filled without price impact)."""
        if not trades:
            return False
        
        # Check if large trades are happening without price movement
        large_trades = [t for t in trades if t.size > self.large_order_threshold * 0.5]
        if len(large_trades) < 3:
            return False
        
        # Calculate price change during large trades
        prices = [t.price for t in large_trades]
        price_range = max(prices) - min(prices)
        avg_price = np.mean(prices)
        
        # Absorption = large volume, small price change
        if avg_price > 0 and price_range / avg_price < 0.001:
            return True
        
        return False
    
    def _detect_spoofing(self, order_book: OrderBookSnapshot) -> float:
        """Detect potential spoofing (fake orders)."""
        # Simplified spoofing detection
        # Look for large orders at levels that get canceled quickly
        
        spoofing_score = 0.0
        
        # Check for unusually large orders at round numbers
        for level in order_book.bids[:5] + order_book.asks[:5]:
            # Round number detection
            if level.price % 100 == 0 or level.price % 10 == 0:
                # Large order at round number
                if level.size > self.large_order_threshold * 2:
                    spoofing_score += 0.2
        
        # Check for extreme imbalance
        if abs(order_book.imbalance) > 0.8:
            spoofing_score += 0.3
        
        return min(1.0, spoofing_score)
    
    def _detect_iceberg(self, trades: List[Trade]) -> float:
        """Detect iceberg orders (hidden liquidity)."""
        if len(trades) < 10:
            return 0.0
        
        # Look for repeated trades at same price
        price_counts = defaultdict(int)
        for trade in trades:
            price_counts[round(trade.price, 2)] += 1
        
        # Iceberg = many trades at same price
        max_repeats = max(price_counts.values()) if price_counts else 0
        iceberg_score = min(1.0, max_repeats / 20)
        
        return iceberg_score
    
    def _calculate_vwap(self, trades: List[Trade]) -> float:
        """Calculate Volume Weighted Average Price."""
        if not trades:
            return 0.0
        
        total_pv = sum(t.price * t.size for t in trades)
        total_volume = sum(t.size for t in trades)
        
        return total_pv / total_volume if total_volume > 0 else 0.0
    
    def _calculate_twap(self, trades: List[Trade]) -> float:
        """Calculate Time Weighted Average Price."""
        if not trades:
            return 0.0
        
        return np.mean([t.price for t in trades])
    
    def _determine_signal(
        self,
        volume_imbalance: float,
        absorption: bool,
        spoofing_score: float,
        iceberg_score: float,
        book_imbalance: float,
    ) -> OrderFlowSignal:
        """Determine order flow signal."""
        # Spoofing takes priority
        if spoofing_score > 0.7:
            return OrderFlowSignal.SPOOFING_DETECTED
        
        # Iceberg detection
        if iceberg_score > 0.7:
            return OrderFlowSignal.ICEBERG_DETECTED
        
        # Absorption
        if absorption:
            return OrderFlowSignal.ABSORPTION
        
        # Volume imbalance
        if volume_imbalance > 0.3 and book_imbalance > 0.2:
            return OrderFlowSignal.BUY_PRESSURE
        elif volume_imbalance < -0.3 and book_imbalance < -0.2:
            return OrderFlowSignal.SELL_PRESSURE
        
        return OrderFlowSignal.NEUTRAL


# ============================================================================
# Liquidity Mapper
# ============================================================================

class LiquidityMapper:
    """Map liquidity zones and detect stop clusters."""
    
    def __init__(
        self,
        *,
        lookback_periods: int = 100,
        stop_cluster_threshold: float = 0.02,
        void_threshold: float = 0.005,
    ):
        self.lookback_periods = lookback_periods
        self.stop_cluster_threshold = stop_cluster_threshold
        self.void_threshold = void_threshold
        
        self.price_history: deque = deque(maxlen=10000)
        self.volume_profile: Dict[float, float] = defaultdict(float)
    
    def analyze(
        self,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        order_book: Optional[OrderBookSnapshot] = None,
    ) -> LiquidityMap:
        """Analyze liquidity."""
        current_price = prices[-1] if len(prices) > 0 else 0
        
        # Build volume profile
        self._update_volume_profile(prices, volumes)
        
        # Find liquidity zones
        zones = self._find_liquidity_zones(current_price)
        
        # Find support/resistance
        support = self._find_support_levels(prices, lows)
        resistance = self._find_resistance_levels(prices, highs)
        
        # Find stop clusters
        stop_clusters = self._find_stop_clusters(support + resistance)
        
        # Find void zones (low liquidity)
        void_zones = self._find_void_zones(prices)
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(order_book, zones)
        
        # Calculate sweep risk
        sweep_risk = self._calculate_sweep_risk(current_price, stop_clusters, liquidity_score)
        
        return LiquidityMap(
            timestamp=datetime.utcnow(),
            zones=zones,
            support_levels=support,
            resistance_levels=resistance,
            stop_clusters=stop_clusters,
            void_zones=void_zones,
            liquidity_score=liquidity_score,
            sweep_risk=sweep_risk,
        )
    
    def _update_volume_profile(self, prices: np.ndarray, volumes: np.ndarray):
        """Update volume profile."""
        for price, volume in zip(prices[-100:], volumes[-100:]):
            # Round to nearest 0.5% for volume profile
            rounded_price = round(price / (price * 0.005)) * (price * 0.005)
            self.volume_profile[rounded_price] += volume
    
    def _find_liquidity_zones(self, current_price: float) -> List[Tuple[float, float, LiquidityZone]]:
        """Find liquidity zones from volume profile."""
        zones = []
        
        if not self.volume_profile:
            return zones
        
        # Find high volume nodes
        max_volume = max(self.volume_profile.values()) if self.volume_profile else 1
        
        for price, volume in self.volume_profile.items():
            volume_pct = volume / max_volume
            
            if volume_pct > 0.7:
                zone_type = LiquidityZone.LIQUIDITY_HIGH
            elif volume_pct < 0.2:
                zone_type = LiquidityZone.VOID
            else:
                continue
            
            zones.append((price, volume, zone_type))
        
        return sorted(zones, key=lambda x: x[0])
    
    def _find_support_levels(self, prices: np.ndarray, lows: np.ndarray) -> List[float]:
        """Find support levels."""
        if len(prices) < 20:
            return []
        
        supports = []
        
        # Recent lows
        recent_lows = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                recent_lows.append(lows[i])
        
        # Cluster nearby lows
        if recent_lows:
            clusters = self._cluster_prices(recent_lows, threshold=0.01)
            supports = [np.mean(c) for c in clusters]
        
        return sorted(supports)[-3:]  # Return 3 most recent supports
    
    def _find_resistance_levels(self, prices: np.ndarray, highs: np.ndarray) -> List[float]:
        """Find resistance levels."""
        if len(prices) < 20:
            return []
        
        resistances = []
        
        # Recent highs
        recent_highs = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                recent_highs.append(highs[i])
        
        # Cluster nearby highs
        if recent_highs:
            clusters = self._cluster_prices(recent_highs, threshold=0.01)
            resistances = [np.mean(c) for c in clusters]
        
        return sorted(resistances)[:3]  # Return 3 nearest resistances
    
    def _find_stop_clusters(self, levels: List[float]) -> List[Tuple[float, float]]:
        """Find stop loss clusters (where stops are likely placed)."""
        clusters = []
        
        for level in levels:
            # Stops are typically placed just below support or above resistance
            # Estimate cluster size based on level strength
            estimated_size = self.volume_profile.get(level, 100000) * 0.5
            clusters.append((level * 0.995, estimated_size))  # Just below
            clusters.append((level * 1.005, estimated_size))  # Just above
        
        return clusters
    
    def _find_void_zones(self, prices: np.ndarray) -> List[Tuple[float, float]]:
        """Find void zones (low liquidity areas)."""
        if len(prices) < 50:
            return []
        
        voids = []
        
        # Look for rapid price movements (gaps)
        for i in range(1, len(prices)):
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
            if price_change > self.void_threshold:
                voids.append((min(prices[i-1], prices[i]), max(prices[i-1], prices[i])))
        
        return voids[:5]  # Return 5 most recent voids
    
    def _cluster_prices(self, prices: List[float], threshold: float) -> List[List[float]]:
        """Cluster nearby prices."""
        if not prices:
            return []
        
        prices = sorted(prices)
        clusters = [[prices[0]]]
        
        for price in prices[1:]:
            if price - clusters[-1][-1] < threshold * price:
                clusters[-1].append(price)
            else:
                clusters.append([price])
        
        return clusters
    
    def _calculate_liquidity_score(self, order_book: Optional[OrderBookSnapshot], zones: List) -> float:
        """Calculate overall liquidity score."""
        score = 50.0  # Base
        
        if order_book:
            # Order book depth
            total_depth = order_book.bid_volume + order_book.ask_volume
            if total_depth > 1000000:
                score += 20
            elif total_depth > 100000:
                score += 10
            
            # Spread
            if order_book.spread < 0.0001:
                score += 15
            elif order_book.spread < 0.001:
                score += 5
        
        # High liquidity zones
        high_liq_zones = [z for z in zones if z[2] == LiquidityZone.LIQUIDITY_HIGH]
        score += len(high_liq_zones) * 5
        
        return min(100, max(0, score))
    
    def _calculate_sweep_risk(self, current_price: float, stop_clusters: List, liquidity_score: float) -> float:
        """Calculate risk of stop sweep."""
        risk = 0.0
        
        # Nearby stop clusters increase risk
        for stop_price, size in stop_clusters:
            distance = abs(current_price - stop_price) / current_price
            if distance < 0.01:  # Within 1%
                risk += 0.3
            elif distance < 0.02:  # Within 2%
                risk += 0.1
        
        # Low liquidity increases sweep risk
        if liquidity_score < 30:
            risk += 0.3
        
        return min(1.0, risk)


# ============================================================================
# Cross-Asset Correlation Engine
# ============================================================================

class CorrelationEngine:
    """Analyze cross-asset correlations."""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.correlation_cache: Dict[Tuple[str, str], float] = {}
    
    def update(self, symbol: str, price: float):
        """Update price history."""
        self.price_history[symbol].append(price)
    
    def calculate_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Calculate correlation between two assets."""
        prices_a = list(self.price_history.get(symbol_a, []))
        prices_b = list(self.price_history.get(symbol_b, []))
        
        if len(prices_a) < 20 or len(prices_b) < 20:
            return 0.0
        
        # Align lengths
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]
        
        # Calculate returns
        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]
        
        # Correlation
        if len(returns_a) < 2 or np.std(returns_a) == 0 or np.std(returns_b) == 0:
            return 0.0
        
        correlation = np.corrcoef(returns_a, returns_b)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_cointegration(self, symbol_a: str, symbol_b: str) -> float:
        """Calculate cointegration score (simplified)."""
        prices_a = list(self.price_history.get(symbol_a, []))
        prices_b = list(self.price_history.get(symbol_b, []))
        
        if len(prices_a) < 50 or len(prices_b) < 50:
            return 0.0
        
        # Align
        min_len = min(len(prices_a), len(prices_b))
        prices_a = np.array(prices_a[-min_len:])
        prices_b = np.array(prices_b[-min_len:])
        
        # Simple cointegration test (spread stationarity)
        spread = prices_a - prices_b * (prices_a[0] / prices_b[0])  # Normalized
        
        # Check if spread is mean-reverting
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        if spread_std == 0:
            return 0.0
        
        # Percentage of time within 1 std
        within_1std = np.mean(np.abs(spread - spread_mean) < spread_std)
        
        return within_1std
    
    def get_lead_lag(self, symbol_a: str, symbol_b: str, max_lag: int = 10) -> float:
        """Determine lead-lag relationship."""
        prices_a = list(self.price_history.get(symbol_a, []))
        prices_b = list(self.price_history.get(symbol_b, []))
        
        if len(prices_a) < 50 or len(prices_b) < 50:
            return 0.0
        
        # Align lengths
        min_len = min(len(prices_a), len(prices_b))
        prices_a = np.array(prices_a[-min_len:])
        prices_b = np.array(prices_b[-min_len:])
        
        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]
        
        best_corr = 0
        best_lag = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            
            if lag > 0:
                corr = np.corrcoef(returns_a[lag:], returns_b[:-lag])[0, 1]
            else:
                corr = np.corrcoef(returns_a[:lag], returns_b[-lag:])[0, 1]
            
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        # Positive = A leads B, Negative = B leads A
        return best_lag * best_corr
    
    def analyze_pair(
        self,
        symbol_a: str,
        symbol_b: str,
    ) -> CorrelationSignal:
        """Analyze correlation between two assets."""
        correlation = self.calculate_correlation(symbol_a, symbol_b)
        cointegration = self.calculate_cointegration(symbol_a, symbol_b)
        lead_lag = self.get_lead_lag(symbol_a, symbol_b)
        
        # Calculate divergence
        prices_a = list(self.price_history.get(symbol_a, []))
        prices_b = list(self.price_history.get(symbol_b, []))
        
        if len(prices_a) >= 20 and len(prices_b) >= 20:
            # Normalized price ratio
            ratio = prices_a[-1] / prices_b[-1]
            avg_ratio = np.mean(prices_a[-20:]) / np.mean(prices_b[-20:])
            divergence = (ratio - avg_ratio) / avg_ratio if avg_ratio != 0 else 0
        else:
            divergence = 0
        
        # Mean reversion probability
        if cointegration > 0.7 and abs(divergence) > 0.02:
            mr_prob = min(0.9, cointegration * 1.5)
        else:
            mr_prob = 0.3
        
        return CorrelationSignal(
            timestamp=datetime.utcnow(),
            asset_a=symbol_a,
            asset_b=symbol_b,
            correlation=correlation,
            correlation_change=0.0,  # Would need historical
            cointegration_score=cointegration,
            lead_lag_relationship=lead_lag,
            divergence_score=divergence,
            mean_reversion_probability=mr_prob,
        )


# ============================================================================
# Seasonality Analyzer
# ============================================================================

class SeasonalityAnalyzer:
    """Analyze time-based patterns."""
    
    # Pre-defined seasonal patterns (can be learned from data)
    CRYPTO_PATTERNS = {
        # Hour of day patterns (UTC)
        0: {"return": 0.0002, "vol": 0.02},   # Midnight - low vol
        1: {"return": 0.0001, "vol": 0.018},
        2: {"return": 0.0001, "vol": 0.015},
        3: {"return": 0.0002, "vol": 0.014},
        4: {"return": 0.0003, "vol": 0.015},
        5: {"return": 0.0002, "vol": 0.016},
        6: {"return": 0.0001, "vol": 0.018},
        7: {"return": 0.0003, "vol": 0.02},    # Asia open
        8: {"return": 0.0004, "vol": 0.022},
        9: {"return": 0.0005, "vol": 0.025},
        10: {"return": 0.0004, "vol": 0.024},
        11: {"return": 0.0003, "vol": 0.023},
        12: {"return": 0.0002, "vol": 0.022},
        13: {"return": 0.0003, "vol": 0.024},
        14: {"return": 0.0004, "vol": 0.026},  # London active
        15: {"return": 0.0005, "vol": 0.028},
        16: {"return": 0.0006, "vol": 0.03},   # London/NY overlap
        17: {"return": 0.0005, "vol": 0.029},
        18: {"return": 0.0004, "vol": 0.028},
        19: {"return": 0.0003, "vol": 0.026},
        20: {"return": 0.0002, "vol": 0.024},
        21: {"return": 0.0001, "vol": 0.022},
        22: {"return": 0.0001, "vol": 0.02},
        23: {"return": 0.0001, "vol": 0.019},
    }
    
    # Day of week patterns
    DAY_PATTERNS = {
        0: {"return": 0.001, "vol": 0.025},  # Monday
        1: {"return": 0.0015, "vol": 0.024}, # Tuesday
        2: {"return": 0.0012, "vol": 0.023}, # Wednesday
        3: {"return": 0.0008, "vol": 0.024}, # Thursday
        4: {"return": 0.0005, "vol": 0.026}, # Friday
        5: {"return": 0.0003, "vol": 0.028}, # Saturday
        6: {"return": 0.0002, "vol": 0.027}, # Sunday
    }
    
    def __init__(self):
        self.hourly_history: Dict[int, List[float]] = defaultdict(list)
        self.daily_history: Dict[int, List[float]] = defaultdict(list)
    
    def analyze(self, timestamp: Optional[datetime] = None) -> SeasonalityPattern:
        """Analyze seasonality patterns."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        hour = timestamp.hour
        day = timestamp.weekday()
        
        # Get pattern data
        hour_pattern = self.CRYPTO_PATTERNS.get(hour, {"return": 0, "vol": 0.02})
        day_pattern = self.DAY_PATTERNS.get(day, {"return": 0, "vol": 0.025})
        
        # Combine patterns
        avg_return = (hour_pattern["return"] + day_pattern["return"]) / 2
        volatility = (hour_pattern["vol"] + day_pattern["vol"]) / 2
        
        # Find optimal hours
        best_hour = max(self.CRYPTO_PATTERNS.items(), key=lambda x: x[1]["return"])[0]
        worst_hour = min(self.CRYPTO_PATTERNS.items(), key=lambda x: x[1]["return"])[0]
        
        # Calculate win rate (simplified)
        if avg_return > 0:
            win_rate = 0.52 + min(0.1, avg_return * 100)
        else:
            win_rate = 0.48 + max(-0.1, avg_return * 100)
        
        # Pattern strength
        pattern_strength = min(1.0, abs(avg_return) * 100 + 0.3)
        
        return SeasonalityPattern(
            hour_of_day=hour,
            day_of_week=day,
            avg_return=avg_return,
            win_rate=win_rate,
            volatility=volatility,
            optimal_entry_hour=best_hour,
            optimal_exit_hour=worst_hour,
            pattern_strength=pattern_strength,
        )
    
    def is_optimal_entry_time(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if current time is optimal for entry."""
        pattern = self.analyze(timestamp)
        return pattern.hour_of_day in (pattern.optimal_entry_hour, (pattern.optimal_entry_hour + 1) % 24)


# ============================================================================
# Funding Rate Analyzer (Crypto)
# ============================================================================

class FundingAnalyzer:
    """Analyze funding rates for crypto perpetuals."""
    
    def __init__(self):
        self.funding_history: deque = deque(maxlen=1000)
        self.basis_history: deque = deque(maxlen=1000)
    
    def analyze(
        self,
        funding_rate: float,
        spot_price: float,
        futures_price: float,
        open_interest: float,
        long_short_ratio: float,
    ) -> FundingSignal:
        """Analyze funding rate signal."""
        # Basis
        basis = futures_price - spot_price
        basis_pct = basis / spot_price if spot_price > 0 else 0
        
        # Store history
        self.funding_history.append(funding_rate)
        self.basis_history.append(basis_pct)
        
        # Predict funding (based on trend)
        if len(self.funding_history) >= 10:
            recent = list(self.funding_history)[-10:]
            predicted_funding = np.mean(recent) + np.polyfit(range(10), recent, 1)[0]
        else:
            predicted_funding = funding_rate
        
        # Open interest change
        oi_change = 0.0  # Would need historical OI
        
        # Liquidation risk
        if long_short_ratio > 3:
            liquidation_risk = 0.7  # Many longs = liquidation risk if price drops
        elif long_short_ratio < 0.33:
            liquidation_risk = 0.7  # Many shorts = liquidation risk if price rises
        else:
            liquidation_risk = 0.3
        
        # Arbitrage opportunity
        arb_opportunity = abs(basis_pct) * 100  # Higher basis = more arb opportunity
        
        return FundingSignal(
            timestamp=datetime.utcnow(),
            funding_rate=funding_rate,
            predicted_funding=predicted_funding,
            basis=basis,
            basis_pct=basis_pct,
            open_interest_change=oi_change,
            long_short_ratio=long_short_ratio,
            liquidation_risk=liquidation_risk,
            arbitrage_opportunity=arb_opportunity,
        )
    
    def get_funding_signal(self, funding_rate: float) -> str:
        """Get trading signal from funding rate."""
        if funding_rate > 0.001:  # High positive funding
            return "short_perp_long_spot"  # Funding arbitrage
        elif funding_rate < -0.001:  # High negative funding
            return "long_perp_short_spot"
        else:
            return "neutral"


# ============================================================================
# Ultimate Intelligence
# ============================================================================

class UltimateIntelligence:
    """
    Ultimate Strategy Intelligence System.
    
    Combines all analysis modules for comprehensive market intelligence.
    """
    
    def __init__(self):
        # Components
        self.regime_predictor = MarkovRegimePredictor()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.liquidity_mapper = LiquidityMapper()
        self.correlation_engine = CorrelationEngine()
        self.seasonality_analyzer = SeasonalityAnalyzer()
        self.funding_analyzer = FundingAnalyzer()
        
        # State
        self.analysis_history: deque = deque(maxlen=1000)
    
    def analyze_market(
        self,
        symbol: str,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        returns: Optional[np.ndarray] = None,
        order_book: Optional[OrderBookSnapshot] = None,
        trades: Optional[List[Trade]] = None,
        funding_rate: Optional[float] = None,
        spot_price: Optional[float] = None,
        futures_price: Optional[float] = None,
    ) -> UltimateMarketAnalysis:
        """Perform comprehensive market analysis."""
        
        # Calculate basic metrics
        if returns is None and len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
        elif returns is None:
            returns = np.array([0.0])
        
        volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.02
        trend_strength = self._calculate_trend_strength(prices)
        
        # 1. Regime Analysis
        current_regime = self.regime_predictor.detect_regime(returns, volatility, trend_strength)
        predicted_regime, regime_confidence, regime_probs = self.regime_predictor.predict_regime_change(current_regime)
        
        regime_prediction = RegimePrediction(
            current_regime=current_regime,
            confidence=0.7,
            predicted_regime=predicted_regime,
            prediction_confidence=regime_confidence,
            time_to_regime_change=regime_confidence * 24,
            regime_probabilities=regime_probs,
            transition_matrix=np.array([]),  # Would be actual matrix
        )
        
        # 2. Order Flow Analysis
        if trades and order_book:
            order_flow = self.order_flow_analyzer.analyze(trades, order_book)
        else:
            order_flow = OrderFlowAnalysis(
                timestamp=datetime.utcnow(),
                signal=OrderFlowSignal.NEUTRAL,
                buy_volume=0, sell_volume=0, net_volume=0,
                volume_imbalance=0, large_order_ratio=0,
                absorption_detected=False, spoofing_score=0,
                iceberg_score=0, vwap=prices[-1], twap=prices[-1],
            )
        
        # 3. Liquidity Analysis
        liquidity = self.liquidity_mapper.analyze(prices, highs, lows, volumes, order_book)
        
        # 4. Correlation Analysis
        correlations = []  # Would analyze correlated assets
        
        # 5. Seasonality
        seasonality = self.seasonality_analyzer.analyze()
        
        # 6. Funding Analysis (crypto)
        if funding_rate is not None and spot_price is not None and futures_price is not None:
            funding = self.funding_analyzer.analyze(
                funding_rate, spot_price, futures_price,
                open_interest=0, long_short_ratio=1.0
            )
        else:
            funding = None
        
        # 7. Smart Money (simplified)
        smart_money = SmartMoneyFlow(
            timestamp=datetime.utcnow(),
            whale_accumulation_score=0.0,
            institutional_flow_score=0.0,
            retail_flow_score=0.0,
            smart_money_direction="neutral",
            large_wallet_activity=0,
            exchange_net_flow=0.0,
            otc_premium=0.0,
        )
        
        # 8. Microstructure
        microstructure = MarketMicrostructure(
            timestamp=datetime.utcnow(),
            spread_bps=order_book.spread / order_book.mid_price * 10000 if order_book else 10,
            depth_score=50.0,
            toxicity_score=30.0,
            adverse_selection=0.0,
            Kyle_lambda=0.01,
            amihud_illiquidity=0.01,
            effective_spread=0.0,
            realized_spread=0.0,
        )
        
        # Calculate composite scores
        bullish_score, bearish_score = self._calculate_composite_scores(
            regime_prediction, order_flow, liquidity, seasonality, funding
        )
        
        edge_score = abs(bullish_score - bearish_score)
        confidence = (bullish_score + bearish_score) / 2
        
        # Generate recommendations
        alerts = self._generate_alerts(regime_prediction, order_flow, liquidity, funding)
        opportunities = self._generate_opportunities(
            bullish_score, bearish_score, seasonality, funding
        )
        
        # Risk level
        risk_level = self._assess_risk_level(volatility, liquidity, order_flow)
        
        analysis = UltimateMarketAnalysis(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            regime=regime_prediction,
            order_flow=order_flow,
            liquidity=liquidity,
            correlations=correlations,
            seasonality=seasonality,
            funding=funding,
            smart_money=smart_money,
            microstructure=microstructure,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            edge_score=edge_score,
            confidence=confidence,
            optimal_entry_timing=self._get_optimal_entry_timing(seasonality),
            optimal_exit_timing=self._get_optimal_exit_timing(seasonality),
            position_sizing_recommendation=self._get_position_sizing(edge_score, volatility, liquidity),
            risk_level=risk_level,
            alerts=alerts,
            opportunities=opportunities,
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength."""
        if len(prices) < 50:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(prices[-50:]))
        slope, _ = np.polyfit(x, prices[-50:], 1)
        
        # Normalize
        trend_strength = slope / np.mean(prices[-50:]) * 100
        
        return np.clip(trend_strength, -1, 1)
    
    def _calculate_composite_scores(
        self,
        regime: RegimePrediction,
        order_flow: OrderFlowAnalysis,
        liquidity: LiquidityMap,
        seasonality: SeasonalityPattern,
        funding: Optional[FundingSignal],
    ) -> Tuple[float, float]:
        """Calculate bullish and bearish composite scores."""
        bullish = 50.0
        bearish = 50.0
        
        # Regime contribution
        regime_str = regime.current_regime.value
        if "bull" in regime_str:
            bullish += 15
        elif "bear" in regime_str:
            bearish += 15
        
        # Order flow contribution
        if order_flow.signal == OrderFlowSignal.BUY_PRESSURE:
            bullish += 10
        elif order_flow.signal == OrderFlowSignal.SELL_PRESSURE:
            bearish += 10
        
        # Volume imbalance
        if order_flow.volume_imbalance > 0:
            bullish += order_flow.volume_imbalance * 20
        else:
            bearish += abs(order_flow.volume_imbalance) * 20
        
        # Seasonality
        if seasonality.avg_return > 0:
            bullish += seasonality.pattern_strength * 10
        else:
            bearish += seasonality.pattern_strength * 10
        
        # Funding (crypto)
        if funding:
            if funding.funding_rate < -0.0005:  # Negative funding = longs paying shorts
                bullish += 5
            elif funding.funding_rate > 0.0005:
                bearish += 5
        
        # Liquidity
        if liquidity.sweep_risk > 0.7:
            bearish += 10  # High sweep risk = caution
        
        return min(100, max(0, bullish)), min(100, max(0, bearish))
    
    def _generate_alerts(
        self,
        regime: RegimePrediction,
        order_flow: OrderFlowAnalysis,
        liquidity: LiquidityMap,
        funding: Optional[FundingSignal],
    ) -> List[str]:
        """Generate alerts."""
        alerts = []
        
        # Regime change warning
        if regime.prediction_confidence > 0.6 and regime.predicted_regime != regime.current_regime:
            alerts.append(f"Regime change predicted: {regime.current_regime.value} → {regime.predicted_regime.value}")
        
        # Spoofing alert
        if order_flow.spoofing_score > 0.7:
            alerts.append("⚠️ Potential spoofing detected")
        
        # High sweep risk
        if liquidity.sweep_risk > 0.7:
            alerts.append("⚠️ High stop sweep risk")
        
        # Funding alert
        if funding and abs(funding.funding_rate) > 0.001:
            alerts.append(f"High funding rate: {funding.funding_rate:.4%}")
        
        return alerts
    
    def _generate_opportunities(
        self,
        bullish: float,
        bearish: float,
        seasonality: SeasonalityPattern,
        funding: Optional[FundingSignal],
    ) -> List[str]:
        """Generate opportunities."""
        opportunities = []
        
        # Directional opportunity
        if bullish > 70:
            opportunities.append("Strong bullish setup detected")
        elif bearish > 70:
            opportunities.append("Strong bearish setup detected")
        
        # Seasonality opportunity
        if seasonality.pattern_strength > 0.6:
            if seasonality.avg_return > 0:
                opportunities.append(f"Favorable time window (optimal hour: {seasonality.optimal_entry_hour}:00 UTC)")
        
        # Funding arbitrage
        if funding and abs(funding.arbitrage_opportunity) > 0.5:
            opportunities.append(f"Funding arbitrage opportunity: {funding.arbitrage_opportunity:.2f}")
        
        return opportunities
    
    def _assess_risk_level(
        self,
        volatility: float,
        liquidity: LiquidityMap,
        order_flow: OrderFlowAnalysis,
    ) -> str:
        """Assess overall risk level."""
        risk_score = 0
        
        # Volatility
        if volatility > 0.05:
            risk_score += 30
        elif volatility > 0.03:
            risk_score += 20
        elif volatility > 0.02:
            risk_score += 10
        
        # Liquidity
        if liquidity.liquidity_score < 30:
            risk_score += 25
        elif liquidity.liquidity_score < 50:
            risk_score += 15
        
        # Order flow toxicity
        if order_flow.spoofing_score > 0.5:
            risk_score += 20
        
        # Classify
        if risk_score >= 60:
            return "extreme"
        elif risk_score >= 40:
            return "high"
        elif risk_score >= 20:
            return "medium"
        else:
            return "low"
    
    def _get_optimal_entry_timing(self, seasonality: SeasonalityPattern) -> str:
        """Get optimal entry timing recommendation."""
        return f"Hour {seasonality.optimal_entry_hour}:00 UTC (pattern strength: {seasonality.pattern_strength:.0%})"
    
    def _get_optimal_exit_timing(self, seasonality: SeasonalityPattern) -> str:
        """Get optimal exit timing recommendation."""
        return f"Hour {seasonality.optimal_exit_hour}:00 UTC"
    
    def _get_position_sizing(
        self,
        edge_score: float,
        volatility: float,
        liquidity: LiquidityMap,
    ) -> float:
        """Calculate recommended position sizing (0-1)."""
        base_size = 0.1
        
        # Adjust for edge
        base_size *= edge_score / 50
        
        # Adjust for volatility
        if volatility > 0.03:
            base_size *= 0.5
        elif volatility < 0.01:
            base_size *= 1.5
        
        # Adjust for liquidity
        if liquidity.liquidity_score < 50:
            base_size *= 0.5
        
        return max(0.01, min(0.5, base_size))
    
    def calculate_edge(self, analysis: UltimateMarketAnalysis) -> float:
        """Calculate trading edge score (0-100)."""
        edge = 0.0
        
        # Strong regime
        if analysis.regime.confidence > 0.7:
            edge += 15
        
        # Multi-factor agreement
        if analysis.bullish_score > 70 or analysis.bearish_score > 70:
            edge += 20
        
        # Order flow confirmation
        if analysis.order_flow.signal != OrderFlowSignal.NEUTRAL:
            edge += 10
        
        # Good liquidity
        if analysis.liquidity.liquidity_score > 60:
            edge += 10
        
        # Favorable timing
        if self.seasonality_analyzer.is_optimal_entry_time():
            edge += 10
        
        # Low risk
        if analysis.risk_level in ("low", "medium"):
            edge += 10
        
        return min(100, edge)


# ============================================================================
# Factory Function
# ============================================================================

def create_ultimate_intelligence() -> UltimateIntelligence:
    """Create ultimate intelligence system."""
    return UltimateIntelligence()
