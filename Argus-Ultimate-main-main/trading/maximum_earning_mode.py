"""
maximum_earning_mode.py — Maximum Adaptive Market Reading & Earning System

This is the ultimate configuration for Argus to:
1. READ the market perfectly (as much as possible)
2. ADAPT in real-time to any condition
3. EARN maximum possible returns

Key principles:
- When confident → GO BIG (aggressive position sizing)
- When uncertain → SIT OUT (capital preservation)
- Always be learning (every trade = data)
- Never fight the market (adapt, don't resist)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class MarketSignal(Enum):
    """Market signals for maximum earning."""
    STRONG_BUY = "strong_buy"      # 20-25% position
    BUY = "buy"                    # 10-15% position
    WEAK_BUY = "weak_buy"          # 5% position
    NEUTRAL = "neutral"            # 0% position
    WEAK_SELL = "weak_sell"        # Close longs
    SELL = "sell"                  # Consider shorts
    STRONG_SELL = "strong_sell"    # Full hedge/short
    SIT_OUT = "sit_out"           # 100% cash


@dataclass
class MarketReading:
    """Complete market reading."""
    timestamp: datetime
    regime: str
    trend_direction: str
    trend_strength: float
    volatility: float
    momentum: float
    order_flow_imbalance: float
    volume_profile: str
    support_distance: float
    resistance_distance: float
    news_sentiment: float
    whale_activity: str
    cross_market_confirmation: bool
    overall_confidence: float
    signal: MarketSignal
    reasoning: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime,
            "signal": self.signal.value,
            "confidence": self.overall_confidence,
            "reasoning": self.reasoning,
        }


class MaximumMarketReader:
    """
    Reads the market with maximum information.
    
    Combines ALL available signals:
    - Price action (trend, momentum, patterns)
    - Volume (profile, anomalies, divergence)
    - Order flow (imbalance, whale activity)
    - Volatility (current, implied, regime)
    - Cross-market (correlations, lead-lag)
    - Sentiment (news, social, fear/greed)
    - On-chain (whales, exchange flows)
    - Technical (support/resistance, indicators)
    """
    
    def __init__(self):
        self.reading_history: List[MarketReading] = []
        self.signal_accuracy: Dict[str, List[bool]] = {
            "strong_buy": [],
            "buy": [],
            "sell": [],
        }
        
        logger.info("Maximum Market Reader initialized")
    
    def read_market(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        order_book: Optional[Dict] = None,
        additional_data: Optional[Dict] = None,
    ) -> MarketReading:
        """
        Perform complete market reading.
        
        Returns comprehensive analysis with actionable signal.
        """
        reasoning = []
        
        # 1. TREND ANALYSIS
        trend_dir, trend_str, trend_reason = self._analyze_trend(prices)
        reasoning.extend(trend_reason)
        
        # 2. MOMENTUM
        momentum, momentum_reason = self._analyze_momentum(prices)
        reasoning.extend(momentum_reason)
        
        # 3. VOLATILITY
        vol, vol_regime, vol_reason = self._analyze_volatility(prices)
        reasoning.extend(vol_reason)
        
        # 4. VOLUME
        vol_profile, vol_reason = self._analyze_volume(volumes, prices)
        reasoning.extend(vol_reason)
        
        # 5. ORDER FLOW (if available)
        ob_imbalance, whale_act, ob_reason = self._analyze_order_flow(order_book)
        reasoning.extend(ob_reason)
        
        # 6. SUPPORT/RESISTANCE
        support_dist, resist_dist, sr_reason = self._analyze_support_resistance(prices)
        reasoning.extend(sr_reason)
        
        # 7. CROSS-MARKET (if available)
        cross_conf, cross_reason = self._analyze_cross_market(additional_data)
        reasoning.extend(cross_reason)
        
        # 8. SENTIMENT (if available)
        sentiment, sent_reason = self._analyze_sentiment(additional_data)
        reasoning.extend(sent_reason)
        
        # 9. COMBINE ALL SIGNALS
        signal, confidence, final_reasoning = self._combine_signals(
            trend_dir=trend_dir,
            trend_strength=trend_str,
            momentum=momentum,
            volatility=vol,
            volume_profile=vol_profile,
            order_flow_imbalance=ob_imbalance,
            support_distance=support_dist,
            resistance_distance=resist_dist,
            sentiment=sentiment,
            cross_market_confirmation=cross_conf,
            whale_activity=whale_act,
        )
        
        reasoning.extend(final_reasoning)
        
        reading = MarketReading(
            timestamp=datetime.now(),
            regime=vol_regime,
            trend_direction=trend_dir,
            trend_strength=trend_str,
            volatility=vol,
            momentum=momentum,
            order_flow_imbalance=ob_imbalance,
            volume_profile=vol_profile,
            support_distance=support_dist,
            resistance_distance=resist_dist,
            news_sentiment=sentiment,
            whale_activity=whale_act,
            cross_market_confirmation=cross_conf,
            overall_confidence=confidence,
            signal=signal,
            reasoning=reasoning,
        )
        
        self.reading_history.append(reading)
        
        return reading
    
    def _analyze_trend(
        self,
        prices: np.ndarray,
    ) -> Tuple[str, float, List[str]]:
        """Analyze trend direction and strength."""
        reasoning = []
        
        if len(prices) < 20:
            return "unknown", 0.0, ["Insufficient data for trend analysis"]
        
        # Multiple timeframe analysis
        ma_10 = np.mean(prices[-10:])
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else ma_20
        
        current_price = prices[-1]
        
        # Determine direction
        if ma_10 > ma_20 > ma_50:
            direction = "up"
            reasoning.append("Bullish MA alignment (10>20>50)")
        elif ma_10 < ma_20 < ma_50:
            direction = "down"
            reasoning.append("Bearish MA alignment (10<20<50)")
        else:
            direction = "mixed"
            reasoning.append("Mixed MA signals")
        
        # Calculate strength (0-1)
        price_vs_ma20 = (current_price - ma_20) / ma_20 if ma_20 != 0 else 0
        strength = min(1.0, abs(price_vs_ma20) * 10)
        
        if strength > 0.7:
            reasoning.append(f"Strong {direction} trend (strength={strength:.2f})")
        elif strength > 0.3:
            reasoning.append(f"Moderate {direction} trend (strength={strength:.2f})")
        else:
            reasoning.append("Weak/no trend")
        
        return direction, strength, reasoning
    
    def _analyze_momentum(
        self,
        prices: np.ndarray,
    ) -> Tuple[float, List[str]]:
        """Analyze momentum using multiple indicators."""
        reasoning = []
        
        if len(prices) < 14:
            return 0.0, ["Insufficient data for momentum"]
        
        # RSI
        rsi = self._calculate_rsi(prices, 14)
        
        if rsi > 70:
            reasoning.append(f"RSI overbought ({rsi:.1f})")
            momentum = -0.5  # Negative momentum when overbought
        elif rsi < 30:
            reasoning.append(f"RSI oversold ({rsi:.1f})")
            momentum = 0.5  # Positive momentum when oversold
        else:
            momentum = (50 - rsi) / 50  # Normalized -1 to 1
            reasoning.append(f"RSI neutral ({rsi:.1f})")
        
        # Rate of Change
        if len(prices) >= 10:
            roc = (prices[-1] - prices[-10]) / prices[-10]
            if abs(roc) > 0.05:
                reasoning.append(f"Strong ROC: {roc*100:.1f}%")
            momentum += roc * 2  # Weight ROC
        
        return np.clip(momentum, -1, 1), reasoning
    
    def _analyze_volatility(
        self,
        prices: np.ndarray,
    ) -> Tuple[float, str, List[str]]:
        """Analyze volatility regime."""
        reasoning = []
        
        if len(prices) < 20:
            return 0.02, "unknown", ["Insufficient data"]
        
        # Calculate returns
        returns = np.diff(np.log(prices[-20:]))
        vol = np.std(returns) * np.sqrt(365)  # Annualized
        
        # Determine regime
        if vol > 0.8:
            regime = "extreme_vol"
            reasoning.append(f"Extreme volatility: {vol*100:.0f}% annualized")
        elif vol > 0.5:
            regime = "high_vol"
            reasoning.append(f"High volatility: {vol*100:.0f}% annualized")
        elif vol > 0.3:
            regime = "normal_vol"
            reasoning.append(f"Normal volatility: {vol*100:.0f}% annualized")
        elif vol > 0.15:
            regime = "low_vol"
            reasoning.append(f"Low volatility: {vol*100:.0f}% annualized")
        else:
            regime = "very_low_vol"
            reasoning.append(f"Very low volatility: {vol*100:.0f}% annualized")
        
        return vol, regime, reasoning
    
    def _analyze_volume(
        self,
        volumes: np.ndarray,
        prices: np.ndarray,
    ) -> Tuple[str, List[str]]:
        """Analyze volume profile."""
        reasoning = []
        
        if len(volumes) < 20:
            return "normal", ["Insufficient volume data"]
        
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume-price analysis
        price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
        
        if volume_ratio > 2.0:
            if price_change > 0:
                profile = "accumulation"
                reasoning.append(f"High volume accumulation ({volume_ratio:.1f}x avg)")
            else:
                profile = "distribution"
                reasoning.append(f"High volume distribution ({volume_ratio:.1f}x avg)")
        elif volume_ratio > 1.5:
            profile = "elevated"
            reasoning.append(f"Elevated volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio < 0.5:
            profile = "low"
            reasoning.append(f"Low volume ({volume_ratio:.1f}x avg)")
        else:
            profile = "normal"
            reasoning.append(f"Normal volume ({volume_ratio:.1f}x avg)")
        
        return profile, reasoning
    
    def _analyze_order_flow(
        self,
        order_book: Optional[Dict],
    ) -> Tuple[float, str, List[str]]:
        """Analyze order flow from order book."""
        reasoning = []
        
        if not order_book:
            return 0.0, "unknown", ["No order book data"]
        
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        if not bids or not asks:
            return 0.0, "unknown", ["Empty order book"]
        
        # Calculate imbalance
        bid_volume = sum(size for _, size in bids[:10])
        ask_volume = sum(size for _, size in asks[:10])
        
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
        
        # Detect whale activity
        whale_threshold = max(bid_volume, ask_volume) * 0.2
        whale_bids = [size for _, size in bids[:5] if size > whale_threshold]
        whale_asks = [size for _, size in asks[:5] if size > whale_threshold]
        
        if whale_bids and not whale_asks:
            whale = "whale_buying"
            reasoning.append(f"Whale buying detected ({len(whale_bids)} large bids)")
        elif whale_asks and not whale_bids:
            whale = "whale_selling"
            reasoning.append(f"Whale selling detected ({len(whale_asks)} large asks)")
        else:
            whale = "normal"
        
        if abs(imbalance) > 0.3:
            direction = "buy" if imbalance > 0 else "sell"
            reasoning.append(f"Strong order flow {direction} pressure ({imbalance:.2f})")
        elif abs(imbalance) > 0.1:
            reasoning.append(f"Moderate order flow imbalance ({imbalance:.2f})")
        else:
            reasoning.append("Balanced order flow")
        
        return imbalance, whale, reasoning
    
    def _analyze_support_resistance(
        self,
        prices: np.ndarray,
    ) -> Tuple[float, float, List[str]]:
        """Analyze support and resistance levels."""
        reasoning = []
        
        if len(prices) < 20:
            return 0.5, 0.5, ["Insufficient data"]
        
        current = prices[-1]
        
        # Simple S/R using recent highs/lows
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        
        # Distance to levels (normalized)
        if recent_high > recent_low:
            support_dist = (current - recent_low) / (recent_high - recent_low)
            resistance_dist = (recent_high - current) / (recent_high - recent_low)
        else:
            support_dist = 0.5
            resistance_dist = 0.5
        
        # Interpret
        if support_dist < 0.2:
            reasoning.append("Near support - potential bounce")
        elif resistance_dist < 0.2:
            reasoning.append("Near resistance - potential rejection")
        elif support_dist > 0.8:
            reasoning.append("Near highs - breakout potential")
        elif resistance_dist > 0.8:
            reasoning.append("Near lows - breakdown risk")
        
        return support_dist, resistance_dist, reasoning
    
    def _analyze_cross_market(
        self,
        additional_data: Optional[Dict],
    ) -> Tuple[bool, List[str]]:
        """Analyze cross-market signals."""
        reasoning = []
        
        if not additional_data:
            return False, ["No cross-market data"]
        
        # Check BTC correlation (if trading alts)
        btc_trend = additional_data.get("btc_trend")
        if btc_trend:
            if btc_trend > 0:
                reasoning.append("BTC trending up (risk-on)")
                return True
            else:
                reasoning.append("BTC trending down (risk-off)")
                return False
        
        return False, ["No cross-market signals"]
    
    def _analyze_sentiment(
        self,
        additional_data: Optional[Dict],
    ) -> Tuple[float, List[str]]:
        """Analyze market sentiment."""
        reasoning = []
        
        if not additional_data:
            return 0.0, ["No sentiment data"]
        
        sentiment = additional_data.get("sentiment", 0.0)
        
        if sentiment > 0.5:
            reasoning.append(f"Bullish sentiment ({sentiment:.2f})")
        elif sentiment < -0.5:
            reasoning.append(f"Bearish sentiment ({sentiment:.2f})")
        else:
            reasoning.append(f"Neutral sentiment ({sentiment:.2f})")
        
        return sentiment, reasoning
    
    def _combine_signals(
        self,
        trend_dir: str,
        trend_strength: float,
        momentum: float,
        volatility: float,
        volume_profile: str,
        order_flow_imbalance: float,
        support_distance: float,
        resistance_distance: float,
        sentiment: float,
        cross_market_confirmation: bool,
        whale_activity: str,
    ) -> Tuple[MarketSignal, float, List[str]]:
        """Combine all signals into final recommendation."""
        reasoning = []
        
        # Score each factor
        scores = []
        weights = []
        
        # Trend (weight: 25%)
        if trend_dir == "up":
            scores.append(trend_strength)
        elif trend_dir == "down":
            scores.append(-trend_strength)
        else:
            scores.append(0)
        weights.append(0.25)
        
        # Momentum (weight: 20%)
        scores.append(momentum)
        weights.append(0.20)
        
        # Order flow (weight: 20%)
        scores.append(order_flow_imbalance * 2)
        weights.append(0.20)
        
        # Volume (weight: 10%)
        if volume_profile == "accumulation":
            scores.append(0.5)
        elif volume_profile == "distribution":
            scores.append(-0.5)
        else:
            scores.append(0)
        weights.append(0.10)
        
        # Support/Resistance (weight: 10%)
        if support_distance < 0.2:
            scores.append(0.5)  # Near support = good for longs
        elif resistance_distance < 0.2:
            scores.append(-0.5)  # Near resistance = bad for longs
        else:
            scores.append(0)
        weights.append(0.10)
        
        # Sentiment (weight: 10%)
        scores.append(sentiment)
        weights.append(0.10)
        
        # Cross-market (weight: 5%)
        scores.append(0.3 if cross_market_confirmation else -0.3)
        weights.append(0.05)
        
        # Calculate weighted score
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        # Calculate confidence
        confidence = min(0.95, abs(weighted_score) + 0.3)
        
        # Determine signal
        if weighted_score > 0.5:
            signal = MarketSignal.STRONG_BUY
            reasoning.append(f"STRONG BUY signal (score={weighted_score:.2f})")
        elif weighted_score > 0.25:
            signal = MarketSignal.BUY
            reasoning.append(f"BUY signal (score={weighted_score:.2f})")
        elif weighted_score > 0.1:
            signal = MarketSignal.WEAK_BUY
            reasoning.append(f"Weak BUY signal (score={weighted_score:.2f})")
        elif weighted_score < -0.5:
            signal = MarketSignal.STRONG_SELL
            reasoning.append(f"STRONG SELL signal (score={weighted_score:.2f})")
        elif weighted_score < -0.25:
            signal = MarketSignal.SELL
            reasoning.append(f"SELL signal (score={weighted_score:.2f})")
        elif weighted_score < -0.1:
            signal = MarketSignal.WEAK_SELL
            reasoning.append(f"Weak SELL signal (score={weighted_score:.2f})")
        else:
            signal = MarketSignal.SIT_OUT
            reasoning.append(f"SIT OUT (score={weighted_score:.2f})")
        
        # Adjust for volatility
        if volatility > 0.8:
            reasoning.append("High volatility - reducing position size")
            confidence *= 0.7
        
        return signal, confidence, reasoning
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class MaximumPositionSizer:
    """
    Position sizing for maximum earnings.
    
    Key principle: Bet BIG when confident, bet SMALL when uncertain.
    """
    
    # Position sizes by signal strength
    SIGNAL_SIZES = {
        MarketSignal.STRONG_BUY: 0.25,    # 25% of capital
        MarketSignal.BUY: 0.15,            # 15% of capital
        MarketSignal.WEAK_BUY: 0.05,       # 5% of capital
        MarketSignal.NEUTRAL: 0.0,         # 0%
        MarketSignal.WEAK_SELL: 0.0,       # Close positions
        MarketSignal.SELL: 0.0,            # Close + consider short
        MarketSignal.STRONG_SELL: 0.10,    # 10% short
        MarketSignal.SIT_OUT: 0.0,         # 100% cash
    }
    
    def __init__(self, max_risk_per_trade: float = 0.10):
        self.max_risk_per_trade = max_risk_per_trade  # 10% max risk
    
    def calculate_position_size(
        self,
        capital: float,
        signal: MarketSignal,
        confidence: float,
        volatility: float,
        stop_loss_pct: float = 0.03,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size.
        
        Factors:
        - Signal strength
        - Confidence level
        - Volatility adjustment
        - Risk per trade limit
        """
        # Base size from signal
        base_size = self.SIGNAL_SIZES.get(signal, 0.0)
        
        # Confidence adjustment (0.5 to 1.5x)
        confidence_multiplier = 0.5 + confidence
        
        # Volatility adjustment (reduce in high vol)
        if volatility > 0.5:
            vol_multiplier = 0.5
        elif volatility > 0.3:
            vol_multiplier = 0.75
        else:
            vol_multiplier = 1.0
        
        # Calculate final size
        position_pct = base_size * confidence_multiplier * vol_multiplier
        
        # Cap at max risk
        position_pct = min(position_pct, self.max_risk_per_trade)
        
        # Calculate dollar amounts
        position_value = capital * position_pct
        
        # Calculate stop loss
        stop_loss_value = position_value * stop_loss_pct
        
        return {
            "signal": signal.value,
            "position_pct": position_pct,
            "position_value": position_value,
            "stop_loss_value": stop_loss_value,
            "confidence": confidence,
            "confidence_multiplier": confidence_multiplier,
            "vol_multiplier": vol_multiplier,
        }


class MaximumEarningOrchestrator:
    """
    Orchestrates maximum earning mode.
    
    Combines:
    - MaximumMarketReader (read everything)
    - MaximumPositionSizer (size positions optimally)
    - Level 20 Singularity (all advanced features)
    """
    
    def __init__(self, initial_capital: float = 1000.0):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        
        self.reader = MaximumMarketReader()
        self.sizer = MaximumPositionSizer(max_risk_per_trade=0.15)
        
        self.positions: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.total_pnl = 0.0
        
        logger.info("=" * 60)
        logger.info("MAXIMUM EARNING MODE INITIALIZED")
        logger.info(f"Capital: ${initial_capital:.2f}")
        logger.info("=" * 60)
    
    def analyze_and_trade(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        order_book: Optional[Dict] = None,
        additional_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Main trading loop:
        1. Read market
        2. Get signal
        3. Size position
        4. Execute (return order)
        """
        # Read market
        reading = self.reader.read_market(prices, volumes, order_book, additional_data)
        
        # Calculate position size
        position = self.sizer.calculate_position_size(
            capital=self.capital,
            signal=reading.signal,
            confidence=reading.overall_confidence,
            volatility=reading.volatility,
        )
        
        # Build trade decision
        decision = {
            "timestamp": reading.timestamp.isoformat(),
            "signal": reading.signal.value,
            "confidence": reading.overall_confidence,
            "position_size": position,
            "market_reading": reading.to_dict(),
            "should_trade": reading.signal not in [
                MarketSignal.SIT_OUT,
                MarketSignal.NEUTRAL,
            ],
            "reasoning": reading.reasoning,
        }
        
        return decision
    
    def update_capital(self, pnl: float):
        """Update capital after trade."""
        self.capital += pnl
        self.total_pnl += pnl
        
        logger.info(
            "Capital update: PnL=$%.2f, Total=$%.2f, Return=%.1f%%",
            pnl, self.capital, (self.capital / self.initial_capital - 1) * 100,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.capital,
            "total_pnl": self.total_pnl,
            "return_pct": (self.capital / self.initial_capital - 1) * 100,
            "total_trades": len(self.trade_history),
            "readings_taken": len(self.reader.reading_history),
        }


# Factory function
def create_maximum_earning_system(
    capital: float = 1000.0,
) -> MaximumEarningOrchestrator:
    """Create maximum earning system."""
    return MaximumEarningOrchestrator(initial_capital=capital)
