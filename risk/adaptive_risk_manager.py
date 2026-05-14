#!/usr/bin/env python3
"""
Adaptive Risk Manager - Portfolio and Market Adaptive Risk Control
Dynamically adjusts risk based on portfolio composition, market conditions, and sentiment

NEW: Integrated with NeuromodulatedRiskManager for dynamic risk limits
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_HIGH_VOL = "bull_high_vol"
    BULL_LOW_VOL = "bull_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    EXTREME_VOLATILITY = "extreme_volatility"
    MARKET_CRASH = "market_crash"


class RiskLevel(Enum):
    """Adaptive risk levels"""
    ULTRA_CONSERVATIVE = "ultra_conservative"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    HIGH_RISK = "high_risk"


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio metrics"""
    total_value: float
    total_exposure: float
    exposure_pct: float
    diversification_score: float
    correlation_matrix: np.ndarray
    risk_contribution: Dict[str, float]
    concentration_risk: float
    sector_exposure: Dict[str, float]
    asset_class_exposure: Dict[str, float]
    liquidity_score: float


@dataclass
class MarketConditions:
    """Real-time market conditions"""
    volatility_index: float  # VIX or equivalent
    trend_strength: float  # ADX or similar
    momentum: float  # RSI or momentum indicator
    fear_greed_index: float
    put_call_ratio: float
    market_breadth: float  # Advance-decline ratio
    intermarket_correlation: float
    regime: MarketRegime
    regime_confidence: float
    sentiment_score: float  # -1 to 1 (bearish to bullish)


@dataclass
class AdaptiveLimits:
    """Dynamically adjusted risk limits"""
    max_position_size_pct: float
    max_portfolio_exposure_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_leverage: float
    max_concentration_pct: float
    max_correlation_threshold: float
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float


class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple indicators
    """

    def __init__(self, lookback_periods: int = 252):
        self.lookback_periods = lookback_periods
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.indicators_history: List[Dict] = []

    def update_market_data(self, price: float, volume: float, indicators: Dict) -> None:
        """Update market data for regime analysis"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.indicators_history.append(indicators)

        # Maintain lookback window
        if len(self.price_history) > self.lookback_periods:
            self.price_history.pop(0)
            self.volume_history.pop(0)
            self.indicators_history.pop(0)

    def detect_regime(self) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime using machine learning approach

        Returns:
            Tuple of (regime, confidence_score)
        """
        if len(self.price_history) < 50:
            return MarketRegime.SIDEWAYS_LOW_VOL, 0.5

        # Calculate trend metrics
        returns = np.diff(np.log(self.price_history))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        trend = np.polyfit(range(len(self.price_history)), self.price_history, 1)[0]
        trend_strength = abs(trend) / np.std(self.price_history)

        # Calculate volume metrics
        volume_sma = np.mean(self.volume_history[-20:]) if len(self.volume_history) >= 20 else np.mean(self.volume_history)
        volume_ratio = self.volume_history[-1] / volume_sma if volume_sma > 0 else 1.0

        # Get latest indicators
        latest_indicators = self.indicators_history[-1]
        rsi = latest_indicators.get('rsi', 50)
        macd_signal = latest_indicators.get('macd_signal', 0)
        bb_width = latest_indicators.get('bollinger_width', 0.02)

        # Regime classification logic
        trend_direction = "bull" if trend > 0 else "bear"
        vol_level = "high_vol" if volatility > 0.25 else "low_vol"

        # Special cases
        if volatility > 0.50:  # Extreme volatility
            if trend < -0.001:  # Sharp decline
                return MarketRegime.MARKET_CRASH, 0.9
            else:
                return MarketRegime.EXTREME_VOLATILITY, 0.85
        elif abs(trend) < 0.0005:  # Sideways
            return getattr(MarketRegime, f"SIDEWAYS_{vol_level.upper()}"), 0.7
        else:  # Trending market
            return getattr(MarketRegime, f"{trend_direction.upper()}_{vol_level.upper()}"), 0.8

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """Get risk adjustments for current regime"""
        adjustments = {
            MarketRegime.BULL_LOW_VOL: {
                "position_size_multiplier": 1.2,
                "stop_loss_multiplier": 1.5,
                "max_exposure_multiplier": 1.1,
                "confidence_boost": 0.1
            },
            MarketRegime.BULL_HIGH_VOL: {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.2,
                "max_exposure_multiplier": 0.9,
                "confidence_boost": 0.0
            },
            MarketRegime.BEAR_LOW_VOL: {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 1.0,
                "max_exposure_multiplier": 0.8,
                "confidence_boost": -0.1
            },
            MarketRegime.BEAR_HIGH_VOL: {
                "position_size_multiplier": 0.6,
                "stop_loss_multiplier": 0.8,
                "max_exposure_multiplier": 0.7,
                "confidence_boost": -0.2
            },
            MarketRegime.SIDEWAYS_LOW_VOL: {
                "position_size_multiplier": 0.9,
                "stop_loss_multiplier": 1.1,
                "max_exposure_multiplier": 0.85,
                "confidence_boost": 0.0
            },
            MarketRegime.SIDEWAYS_HIGH_VOL: {
                "position_size_multiplier": 0.7,
                "stop_loss_multiplier": 0.9,
                "max_exposure_multiplier": 0.75,
                "confidence_boost": -0.1
            },
            MarketRegime.EXTREME_VOLATILITY: {
                "position_size_multiplier": 0.4,
                "stop_loss_multiplier": 0.6,
                "max_exposure_multiplier": 0.5,
                "confidence_boost": -0.3
            },
            MarketRegime.MARKET_CRASH: {
                "position_size_multiplier": 0.2,
                "stop_loss_multiplier": 0.4,
                "max_exposure_multiplier": 0.3,
                "confidence_boost": -0.5
            }
        }

        return adjustments.get(regime, {
            "position_size_multiplier": 0.8,
            "stop_loss_multiplier": 1.0,
            "max_exposure_multiplier": 0.8,
            "confidence_boost": 0.0
        })


class PortfolioRiskAnalyzer:
    """
    Advanced portfolio risk analysis with correlation and concentration analysis
    """

    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.logger = logging.getLogger("PortfolioRiskAnalyzer")

    def add_position(self, symbol: str, position_data: Dict) -> None:
        """Add position to portfolio tracking"""
        self.positions[symbol] = position_data
        if symbol not in self.price_history:
            self.price_history[symbol] = []

    def update_price(self, symbol: str, price: float) -> None:
        """Update price history for correlation analysis"""
        if symbol in self.price_history:
            self.price_history[symbol].append(price)
            # Maintain 252-day history
            if len(self.price_history[symbol]) > 252:
                self.price_history[symbol].pop(0)

    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        if not self.positions:
            return PortfolioMetrics(0, 0, 0, 0, np.array([]), {}, 0, {}, {}, 0)

        # Calculate exposures
        total_value = sum(pos.get('current_value', 0) for pos in self.positions.values())
        total_exposure = sum(abs(pos.get('exposure', 0)) for pos in self.positions.values())
        exposure_pct = total_exposure / total_value if total_value > 0 else 0

        # Calculate diversification score (inverse of concentration)
        weights = np.array([pos.get('weight', 0) for pos in self.positions.values()])
        diversification_score = 1.0 - np.max(weights) if len(weights) > 0 else 0

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix()

        # Risk contribution analysis
        risk_contribution = self._calculate_risk_contribution(weights, correlation_matrix)

        # Concentration risk (Herfindahl-Hirschman Index)
        concentration_risk = np.sum(weights ** 2)

        # Sector and asset class exposure (simplified)
        sector_exposure = self._calculate_sector_exposure()
        asset_class_exposure = self._calculate_asset_class_exposure()

        # Liquidity score
        liquidity_score = self._calculate_liquidity_score()

        return PortfolioMetrics(
            total_value=total_value,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            diversification_score=diversification_score,
            correlation_matrix=correlation_matrix,
            risk_contribution=risk_contribution,
            concentration_risk=concentration_risk,
            sector_exposure=sector_exposure,
            asset_class_exposure=asset_class_exposure,
            liquidity_score=liquidity_score
        )

    def _calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix from price histories"""
        symbols = list(self.price_history.keys())
        if len(symbols) < 2:
            return np.array([[1.0]])

        # Get common time period
        min_length = min(len(self.price_history[s]) for s in symbols)
        if min_length < 30:  # Need at least 30 data points
            return np.eye(len(symbols))

        # Calculate returns
        returns_data = []
        for symbol in symbols:
            prices = self.price_history[symbol][-min_length:]
            returns = np.diff(np.log(prices))
            returns_data.append(returns)

        returns_matrix = np.column_stack(returns_data)
        return np.corrcoef(returns_matrix.T)

    def _calculate_risk_contribution(self, weights: np.ndarray, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate risk contribution of each position"""
        symbols = list(self.positions.keys())
        contributions = {}

        if len(weights) != len(symbols) or correlation_matrix.size == 0:
            return {symbol: 0 for symbol in symbols}

        try:
            # Simplified risk contribution (marginal VaR contribution)
            portfolio_vol = np.sqrt(weights.T @ correlation_matrix @ weights)
            for i, symbol in enumerate(symbols):
                marginal_contribution = weights[i] * correlation_matrix[i] @ weights
                contributions[symbol] = marginal_contribution / portfolio_vol if portfolio_vol > 0 else 0
        except (FloatingPointError, ValueError, TypeError, IndexError, ZeroDivisionError):
            contributions = {symbol: 1.0 / len(symbols) for symbol in symbols}

        return contributions

    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure (simplified)"""
        sector_exposure = {}
        for symbol, pos in self.positions.items():
            sector = pos.get('sector', 'unknown')
            exposure = pos.get('exposure', 0)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + exposure
        return sector_exposure

    def _calculate_asset_class_exposure(self) -> Dict[str, float]:
        """Calculate asset class exposure"""
        asset_exposure = {}
        for symbol, pos in self.positions.items():
            asset_class = pos.get('asset_class', 'unknown')
            exposure = pos.get('exposure', 0)
            asset_exposure[asset_class] = asset_exposure.get(asset_class, 0) + exposure
        return asset_exposure

    def _calculate_liquidity_score(self) -> float:
        """Calculate portfolio liquidity score"""
        if not self.positions:
            return 0.0

        # Simplified liquidity based on position sizes and market cap
        liquidity_scores = []
        for pos in self.positions.values():
            size_pct = pos.get('weight', 0)
            market_cap = pos.get('market_cap', 1e9)  # Default to large cap
            # Larger positions in smaller caps = less liquid
            score = min(1.0, market_cap / (size_pct * 1e12)) if size_pct > 0 else 1.0
            liquidity_scores.append(score)

        return np.mean(liquidity_scores) if liquidity_scores else 0.0


class SentimentRiskAdjuster:
    """
    Integrate sentiment analysis into risk management
    """

    def __init__(self):
        self.sentiment_history: List[Dict] = []
        self.sentiment_weights = {
            'twitter': 0.3,
            'reddit': 0.2,
            'news': 0.25,
            'fear_greed': 0.15,
            'technical': 0.1
        }

    def update_sentiment(self, sentiment_data: Dict) -> None:
        """Update sentiment data"""
        self.sentiment_history.append(sentiment_data)
        if len(self.sentiment_history) > 100:  # Keep last 100 readings
            self.sentiment_history.pop(0)

    def calculate_sentiment_score(self) -> float:
        """Calculate composite sentiment score"""
        if not self.sentiment_history:
            return 0.0

        latest = self.sentiment_history[-1]
        composite_score = 0.0
        total_weight = 0.0

        for source, weight in self.sentiment_weights.items():
            if source in latest:
                # Normalize to -1 to 1 scale
                score = latest[source]
                if source == 'fear_greed':  # 0-100 scale, convert to -1 to 1
                    score = (score - 50) / 50
                composite_score += score * weight
                total_weight += weight

        return composite_score / total_weight if total_weight > 0 else 0.0

    def get_sentiment_risk_adjustment(self) -> Dict[str, float]:
        """Get risk adjustments based on sentiment"""
        sentiment_score = self.calculate_sentiment_score()

        # Extreme negative sentiment increases caution
        if sentiment_score < -0.7:
            return {
                "position_size_multiplier": 0.5,
                "max_exposure_multiplier": 0.6,
                "stop_loss_tightness": 1.5,  # Tighter stops
                "confidence_penalty": -0.3
            }
        elif sentiment_score < -0.3:
            return {
                "position_size_multiplier": 0.8,
                "max_exposure_multiplier": 0.8,
                "stop_loss_tightness": 1.2,
                "confidence_penalty": -0.1
            }
        elif sentiment_score > 0.7:
            return {
                "position_size_multiplier": 1.1,
                "max_exposure_multiplier": 1.0,
                "stop_loss_tightness": 0.9,  # Wider stops allowed
                "confidence_penalty": 0.1
            }
        elif sentiment_score > 0.3:
            return {
                "position_size_multiplier": 1.0,
                "max_exposure_multiplier": 0.95,
                "stop_loss_tightness": 1.0,
                "confidence_penalty": 0.0
            }
        else:
            return {
                "position_size_multiplier": 0.9,
                "max_exposure_multiplier": 0.9,
                "stop_loss_tightness": 1.1,
                "confidence_penalty": 0.0
            }


class AdaptiveRiskManager:
    """
    Master adaptive risk management system
    Dynamically adjusts all risk parameters based on portfolio and market conditions
    
    NEW: Integrated with NeuromodulatedRiskManager for dynamic risk limits
    """

    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Core components
        self.market_detector = MarketRegimeDetector()
        self.portfolio_analyzer = PortfolioRiskAnalyzer()
        self.sentiment_adjuster = SentimentRiskAdjuster()

        # Neuromodulated Risk Manager (NEW)
        try:
            from risk.neuromodulated_risk import NeuromodulatedRiskManager
            self.neuromodulated_risk = NeuromodulatedRiskManager(initial_capital=initial_capital)
            self.use_neuromodulated_risk = True
            logger.info("NeuromodulatedRiskManager initialized")
        except ImportError:
            self.neuromodulated_risk = None
            self.use_neuromodulated_risk = False
            logger.warning("NeuromodulatedRiskManager not available")

        # Risk state
        self.current_limits = self._get_base_limits()
        self.risk_level = RiskLevel.MODERATE
        self.last_adjustment = datetime.now()

        # Historical tracking
        self.adjustment_history: List[Dict] = []
        self.violation_history: List[Dict] = []

        self.logger = logging.getLogger("AdaptiveRiskManager")

    def _get_base_limits(self) -> AdaptiveLimits:
        """Get base risk limits"""
        return AdaptiveLimits(
            max_position_size_pct=0.05,  # 5% of portfolio
            max_portfolio_exposure_pct=0.80,  # 80% max exposure
            max_daily_loss_pct=0.025,  # 2.5% daily loss
            max_drawdown_pct=0.15,  # 15% max drawdown
            max_leverage=3.0,
            max_concentration_pct=0.20,  # 20% max in single position
            max_correlation_threshold=0.7,  # 0.7 correlation limit
            position_size_multiplier=1.0,
            stop_loss_multiplier=1.0,
            take_profit_multiplier=1.0
        )

    def update_market_conditions(self, market_data: Dict) -> None:
        """Update market conditions for adaptive adjustments"""
        # Update regime detector
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        indicators = market_data.get('indicators', {})

        self.market_detector.update_market_data(price, volume, indicators)

        # Update sentiment
        sentiment_data = market_data.get('sentiment', {})
        self.sentiment_adjuster.update_sentiment(sentiment_data)

        # Update Neuromodulated Risk Manager (NEW)
        if self.use_neuromodulated_risk:
            volatility = indicators.get('volatility', 0.0)
            is_shock = volatility > 0.5  # Define shock as extreme volatility
            self.neuromodulated_risk.update(
                pnl=0,  # No PnL change from market data alone
                volatility=volatility,
                is_shock=is_shock,
                win=False,
            )

    def update_portfolio(self, positions: Dict[str, Dict]) -> None:
        """Update portfolio positions"""
        # Update portfolio analyzer
        for symbol, pos_data in positions.items():
            self.portfolio_analyzer.add_position(symbol, pos_data)
            if 'current_price' in pos_data:
                self.portfolio_analyzer.update_price(symbol, pos_data['current_price'])

    def calculate_adaptive_limits(self) -> AdaptiveLimits:
        """
        Calculate adaptive risk limits based on current conditions
        
        This is the core adaptive logic that adjusts all risk parameters
        """
        # Get current market regime
        regime, regime_confidence = self.market_detector.detect_regime()
        regime_adjustments = self.market_detector.get_regime_adjustments(regime)

        # Get portfolio metrics
        portfolio_metrics = self.portfolio_analyzer.calculate_portfolio_metrics()

        # Get sentiment adjustments
        sentiment_adjustments = self.sentiment_adjuster.get_sentiment_risk_adjustment()

        # Calculate portfolio size factor (larger portfolios can take more risk)
        portfolio_size_factor = min(2.0, max(0.5, self.current_capital / 10000))

        # Calculate diversification factor (better diversification = more risk allowed)
        diversification_factor = min(1.5, max(0.5, portfolio_metrics.diversification_score + 0.5))

        # Calculate concentration risk factor
        concentration_factor = max(0.5, 1.0 - portfolio_metrics.concentration_risk)

        # Calculate correlation risk factor
        avg_correlation = np.mean(portfolio_metrics.correlation_matrix) if portfolio_metrics.correlation_matrix.size > 0 else 0.5
        correlation_factor = max(0.5, 1.0 - avg_correlation)

        # Combine all factors
        master_risk_multiplier = (
            portfolio_size_factor *
            diversification_factor *
            concentration_factor *
            correlation_factor *
            regime_adjustments['max_exposure_multiplier'] *
            sentiment_adjustments['max_exposure_multiplier']
        )

        # Get neuromodulated risk limits (NEW)
        if self.use_neuromodulated_risk:
            neuromodulated_limits = self.neuromodulated_risk.get_limits()
            neuromodulated_multiplier = self.neuromodulated_risk.get_risk_multiplier()
            master_risk_multiplier *= neuromodulated_multiplier
        else:
            neuromodulated_limits = None

        # Determine risk level
        if master_risk_multiplier < 0.6:
            self.risk_level = RiskLevel.ULTRA_CONSERVATIVE
        elif master_risk_multiplier < 0.8:
            self.risk_level = RiskLevel.CONSERVATIVE
        elif master_risk_multiplier < 1.2:
            self.risk_level = RiskLevel.MODERATE
        elif master_risk_multiplier < 1.5:
            self.risk_level = RiskLevel.AGGRESSIVE
        else:
            self.risk_level = RiskLevel.HIGH_RISK

        # Calculate final limits
        base_limits = self._get_base_limits()

        adaptive_limits = AdaptiveLimits(
            max_position_size_pct=min(0.10, base_limits.max_position_size_pct * master_risk_multiplier),
            max_portfolio_exposure_pct=min(0.90, base_limits.max_portfolio_exposure_pct * master_risk_multiplier),
            max_daily_loss_pct=base_limits.max_daily_loss_pct,
            max_drawdown_pct=base_limits.max_drawdown_pct,
            max_leverage=min(5.0, base_limits.max_leverage * master_risk_multiplier),
            max_concentration_pct=min(0.25, base_limits.max_concentration_pct * (2 - master_risk_multiplier)),
            max_correlation_threshold=base_limits.max_correlation_threshold,
            position_size_multiplier=(
                regime_adjustments['position_size_multiplier'] *
                sentiment_adjustments['position_size_multiplier'] *
                master_risk_multiplier
            ),
            stop_loss_multiplier=(
                regime_adjustments['stop_loss_multiplier'] *
                sentiment_adjustments['stop_loss_tightness']
            ),
            take_profit_multiplier=max(0.8, master_risk_multiplier)
        )

        # Override with neuromodulated limits if available (NEW)
        if neuromodulated_limits:
            adaptive_limits.max_position_size_pct = min(
                adaptive_limits.max_position_size_pct,
                neuromodulated_limits.max_position_size
            )
            adaptive_limits.max_drawdown_pct = min(
                adaptive_limits.max_drawdown_pct,
                neuromodulated_limits.max_drawdown
            )
            adaptive_limits.stop_loss_multiplier = min(
                adaptive_limits.stop_loss_multiplier,
                neuromodulated_limits.stop_loss / 0.02  # Convert to multiplier
            )
            adaptive_limits.take_profit_multiplier = min(
                adaptive_limits.take_profit_multiplier,
                neuromodulated_limits.take_profit / 0.04  # Convert to multiplier
            )

        # Store adjustment history
        self.adjustment_history.append({
            'timestamp': datetime.now(),
            'regime': regime.value,
            'regime_confidence': regime_confidence,
            'portfolio_size_factor': portfolio_size_factor,
            'diversification_factor': diversification_factor,
            'concentration_factor': concentration_factor,
            'correlation_factor': correlation_factor,
            'sentiment_score': self.sentiment_adjuster.calculate_sentiment_score(),
            'master_risk_multiplier': master_risk_multiplier,
            'neuromodulated_multiplier': neuromodulated_multiplier if self.use_neuromodulated_risk else 1.0,
            'risk_level': self.risk_level.value,
            'limits': adaptive_limits
        })

        self.current_limits = adaptive_limits
        self.last_adjustment = datetime.now()

        return adaptive_limits

    def check_trade_risk(self, symbol: str, position_size: float, entry_price: float,
                        stop_price: float, portfolio_value: float) -> Tuple[bool, str, Dict]:
        """
        Comprehensive adaptive risk check for a potential trade

        Returns:
            (approved, reason, risk_metrics)
        """
        # Get current adaptive limits
        limits = self.calculate_adaptive_limits()

        # Calculate trade metrics
        position_value = position_size * entry_price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        stop_distance = abs(entry_price - stop_price)
        risk_pct = stop_distance / entry_price

        # Check position size limit
        max_allowed_size_pct = limits.max_position_size_pct * limits.position_size_multiplier
        if position_pct > max_allowed_size_pct:
            return False, f"Position size {position_pct:.1%} exceeds limit {max_allowed_size_pct:.1%}", {}

        # Check concentration limit
        current_exposure = sum(pos.get('exposure', 0) for pos in self.portfolio_analyzer.positions.values())
        if symbol in self.portfolio_analyzer.positions:
            current_exposure += position_value
        else:
            current_exposure += position_value

        concentration_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        if concentration_pct > limits.max_concentration_pct:
            return False, f"Concentration {concentration_pct:.1%} exceeds limit {limits.max_concentration_pct:.1%}", {}

        # Check portfolio exposure limit
        exposure_pct = current_exposure / portfolio_value if portfolio_value > 0 else 0
        if exposure_pct > limits.max_portfolio_exposure_pct:
            return False, f"Portfolio exposure {exposure_pct:.1%} exceeds limit {limits.max_portfolio_exposure_pct:.1%}", {}

        # Check correlation risk
        correlation_risk = self._check_correlation_risk(symbol, position_pct)
        if correlation_risk > limits.max_correlation_threshold:
            return False, f"Correlation risk {correlation_risk:.2f} exceeds threshold {limits.max_correlation_threshold:.2f}", {}

        # All checks passed
        risk_metrics = {
            'position_pct': position_pct,
            'risk_pct': risk_pct,
            'exposure_pct': exposure_pct,
            'correlation_risk': correlation_risk,
            'risk_level': self.risk_level.value,
            'adjusted_stop_distance': stop_distance * limits.stop_loss_multiplier,
            'adjusted_position_size': position_size * limits.position_size_multiplier
        }

        return True, "Approved", risk_metrics

    def _check_correlation_risk(self, symbol: str, position_pct: float) -> float:
        """Check correlation risk for new position"""
        if symbol not in self.portfolio_analyzer.price_history:
            return 0.0

        # Simplified correlation check
        portfolio_correlations = []
        for existing_symbol in self.portfolio_analyzer.positions.keys():
            if existing_symbol in self.portfolio_analyzer.price_history and symbol in self.portfolio_analyzer.price_history:
                try:
                    symbol_returns = np.diff(np.log(self.portfolio_analyzer.price_history[symbol][-60:]))
                    existing_returns = np.diff(np.log(self.portfolio_analyzer.price_history[existing_symbol][-60:]))
                    corr = np.corrcoef(symbol_returns, existing_returns)[0, 1]
                    portfolio_correlations.append(corr)
                except (FloatingPointError, ValueError, TypeError, IndexError, ZeroDivisionError):
                    continue

        avg_correlation = np.mean(portfolio_correlations) if portfolio_correlations else 0.0
        return avg_correlation

    def get_comprehensive_risk_report(self) -> Dict:
        """Get comprehensive risk report"""
        limits = self.calculate_adaptive_limits()
        portfolio_metrics = self.portfolio_analyzer.calculate_portfolio_metrics()
        regime, confidence = self.market_detector.detect_regime()
        sentiment_score = self.sentiment_adjuster.calculate_sentiment_score()

        report = {
            'timestamp': datetime.now(),
            'risk_level': self.risk_level.value,
            'market_regime': {
                'regime': regime.value,
                'confidence': confidence
            },
            'sentiment_score': sentiment_score,
            'portfolio_metrics': {
                'total_value': portfolio_metrics.total_value,
                'exposure_pct': portfolio_metrics.exposure_pct,
                'diversification_score': portfolio_metrics.diversification_score,
                'concentration_risk': portfolio_metrics.concentration_risk,
                'liquidity_score': portfolio_metrics.liquidity_score
            },
            'adaptive_limits': {
                'max_position_size_pct': limits.max_position_size_pct,
                'max_portfolio_exposure_pct': limits.max_portfolio_exposure_pct,
                'max_daily_loss_pct': limits.max_daily_loss_pct,
                'max_drawdown_pct': limits.max_drawdown_pct,
                'max_leverage': limits.max_leverage,
                'position_size_multiplier': limits.position_size_multiplier,
                'stop_loss_multiplier': limits.stop_loss_multiplier
            },
            'recent_adjustments': self.adjustment_history[-5:] if self.adjustment_history else [],
            'recommendations': self._generate_recommendations()
        }

        # Add neuromodulated risk stats (NEW)
        if self.use_neuromodulated_risk:
            neuromodulated_stats = self.neuromodulated_risk.get_neuromodulator_levels()
            report['neuromodulated_risk'] = {
                'dopamine': neuromodulated_stats['dopamine'],
                'serotonin': neuromodulated_stats['serotonin'],
                'norepinephrine': neuromodulated_stats['norepinephrine'],
                'risk_multiplier': self.neuromodulated_risk.get_risk_multiplier(),
                'shock_detected': self.neuromodulated_risk.is_shock_detected(),
            }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        limits = self.current_limits
        portfolio_metrics = self.portfolio_analyzer.calculate_portfolio_metrics()

        if self.risk_level in [RiskLevel.HIGH_RISK, RiskLevel.AGGRESSIVE]:
            recommendations.append("⚠️  High risk level detected - consider reducing position sizes")

        if portfolio_metrics.diversification_score < 0.6:
            recommendations.append("📊 Low diversification - consider adding uncorrelated assets")

        if portfolio_metrics.concentration_risk > 0.15:
            recommendations.append("🎯 High concentration risk - reduce largest positions")

        if portfolio_metrics.liquidity_score < 0.7:
            recommendations.append("💧 Low liquidity - avoid large positions in illiquid assets")

        if limits.position_size_multiplier < 0.8:
            recommendations.append("📉 Conservative sizing active - market conditions warrant caution")

        # Neuromodulated risk recommendations (NEW)
        if self.use_neuromodulated_risk:
            if self.neuromodulated_risk.is_shock_detected():
                recommendations.append("🚨 SHOCK DETECTED - Trading halted or reduced")
            neuromodulated_stats = self.neuromodulated_risk.get_neuromodulator_levels()
            if neuromodulated_stats['dopamine'] > 0.8:
                recommendations.append("🎯 High dopamine - Market conditions favorable for risk-taking")
            if neuromodulated_stats['serotonin'] < 0.3:
                recommendations.append("😟 Low serotonin - High volatility, proceed with caution")

        if not recommendations:
            recommendations.append("✅ Risk parameters within acceptable ranges")

        return recommendations

    def emergency_stop_check(self, current_drawdown: float, daily_loss: float) -> Tuple[bool, str]:
        """Check for emergency stop conditions"""
        limits = self.current_limits

        if current_drawdown > limits.max_drawdown_pct:
            return True, f"Drawdown {current_drawdown:.1%} exceeds limit {limits.max_drawdown_pct:.1%}"

        if daily_loss > limits.max_daily_loss_pct:
            return True, f"Daily loss {daily_loss:.1%} exceeds limit {limits.max_daily_loss_pct:.1%}"

        return False, "OK"


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create adaptive risk manager
    arm = AdaptiveRiskManager(initial_capital=10000)

    # Simulate market data updates
    market_data = {
        'price': 50000,
        'volume': 1000000,
        'indicators': {
            'rsi': 65,
            'macd_signal': 0.5,
            'bollinger_width': 0.03,
            'volatility': 0.2,
        },
        'sentiment': {
            'twitter': 0.2,
            'reddit': 0.1,
            'news': -0.1,
            'fear_greed': 60
        }
    }

    arm.update_market_conditions(market_data)

    # Simulate portfolio
    positions = {
        'BTC': {'current_value': 5000, 'exposure': 5000, 'weight': 0.5, 'sector': 'crypto', 'asset_class': 'digital'},
        'ETH': {'current_value': 3000, 'exposure': 3000, 'weight': 0.3, 'sector': 'crypto', 'asset_class': 'digital'},
        'SPY': {'current_value': 2000, 'exposure': 2000, 'weight': 0.2, 'sector': 'equity', 'asset_class': 'stocks'}
    }

    arm.update_portfolio(positions)

    # Get adaptive limits
    limits = arm.calculate_adaptive_limits()
    logger.info(f"Current Risk Level: {arm.risk_level.value}")
    logger.info(f"Max Position Size: {limits.max_position_size_pct:.1%}")
    logger.info(f"Position Size Multiplier: {limits.position_size_multiplier:.2f}")
    
    # Test trade approval
    approved, reason, metrics = arm.check_trade_risk('SOL', 100, 100, 95, 10000)
    logger.info(f"Trade Approved: {approved}")
    logger.info(f"Reason: {reason}")
    
    # Get comprehensive report
    report = arm.get_comprehensive_risk_report()
    logger.info(f"Risk Report: {report['risk_level']}")
    if 'neuromodulated_risk' in report:
        logger.info(f"Neuromodulated Risk: {report['neuromodulated_risk']}")
