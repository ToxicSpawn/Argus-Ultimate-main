# pyright: reportMissingImports=false
"""
Trading-Specific Components for Quantum Reinforcement Learning.

This module provides:
- Market state encoder for converting market data to quantum states
- Action decoder for mapping quantum outputs to trading actions
- Reward function for trading performance evaluation
- Risk management integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ============================================================================
# Market State Encoder
# ============================================================================

class FeatureType(Enum):
    """Types of market features to encode."""
    PRICE = auto()
    VOLUME = auto()
    TECHNICAL = auto()
    ORDER_BOOK = auto()
    SENTIMENT = auto()
    MACRO = auto()


@dataclass
class MarketStateConfig:
    """Configuration for market state encoding."""
    price_features: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "adj_close"])
    volume_features: List[str] = field(default_factory=lambda: ["volume", "quote_volume"])
    technical_features: List[str] = field(default_factory=lambda: ["rsi", "macd", "bb_upper", "bb_lower", "atr"])
    lookback_window: int = 10
    normalization_window: int = 100
    include_position_info: bool = True
    include_portfolio_info: bool = True


class MarketStateEncoder:
    """Encodes market data into quantum-ready state representations."""
    
    def __init__(self, config: Optional[MarketStateConfig] = None):
        self.config = config or MarketStateConfig()
        
        # Running statistics for normalization
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        self.feature_mins: Dict[str, float] = {}
        self.feature_maxs: Dict[str, float] = {}
        self.sample_count = 0
        
        # Feature buffer for computing technical indicators
        self.price_buffer: List[float] = []
        self.volume_buffer: List[float] = []
    
    def encode(
        self,
        market_data: Dict[str, Any],
        position_info: Optional[Dict[str, Any]] = None,
        portfolio_info: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Encode market data into state vector."""
        features: List[float] = []
        
        # Encode price features
        price_features = self._encode_price_features(market_data)
        features.extend(price_features)
        
        # Encode volume features
        volume_features = self._encode_volume_features(market_data)
        features.extend(volume_features)
        
        # Compute and encode technical features
        technical_features = self._compute_technical_features(market_data)
        features.extend(technical_features)
        
        # Encode order book features if available
        if "order_book" in market_data:
            order_book_features = self._encode_order_book(market_data["order_book"])
            features.extend(order_book_features)
        
        # Encode position info if requested
        if self.config.include_position_info and position_info:
            position_features = self._encode_position_info(position_info)
            features.extend(position_features)
        
        # Encode portfolio info if requested
        if self.config.include_portfolio_info and portfolio_info:
            portfolio_features = self._encode_portfolio_info(portfolio_info)
            features.extend(portfolio_features)
        
        # Convert to numpy array
        state = np.array(features, dtype=np.float64)
        
        # Update buffers
        if "close" in market_data:
            self.price_buffer.append(float(market_data["close"]))
            if len(self.price_buffer) > self.config.lookback_window * 2:
                self.price_buffer = self.price_buffer[-self.config.lookback_window * 2:]
        
        if "volume" in market_data:
            self.volume_buffer.append(float(market_data["volume"]))
            if len(self.volume_buffer) > self.config.lookback_window * 2:
                self.volume_buffer = self.volume_buffer[-self.config.lookback_window * 2:]
        
        # Update statistics
        self._update_statistics(state)
        
        # Normalize
        state = self._normalize_state(state)
        
        return state
    
    def _encode_price_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Encode price-related features."""
        features = []
        
        for feature in self.config.price_features:
            if feature in market_data:
                value = float(market_data[feature])
                features.append(value)
            else:
                features.append(0.0)
        
        # Add price change features
        if "close" in market_data and len(self.price_buffer) > 1:
            current_price = float(market_data["close"])
            prev_price = self.price_buffer[-1] if self.price_buffer else current_price
            
            # Price change
            price_change = (current_price - prev_price) / (prev_price + 1e-8)
            features.append(price_change)
            
            # Price momentum
            if len(self.price_buffer) >= 5:
                momentum = (current_price - self.price_buffer[-5]) / (self.price_buffer[-5] + 1e-8)
                features.append(momentum)
            else:
                features.append(0.0)
        
        return features
    
    def _encode_volume_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Encode volume-related features."""
        features = []
        
        for feature in self.config.volume_features:
            if feature in market_data:
                value = float(market_data[feature])
                features.append(value)
            else:
                features.append(0.0)
        
        # Add volume change features
        if "volume" in market_data and len(self.volume_buffer) > 1:
            current_volume = float(market_data["volume"])
            avg_volume = np.mean(self.volume_buffer[-10:]) if len(self.volume_buffer) >= 10 else current_volume
            
            # Relative volume
            relative_volume = current_volume / (avg_volume + 1e-8)
            features.append(relative_volume)
        
        return features
    
    def _compute_technical_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Compute and encode technical indicators."""
        features = []
        
        if len(self.price_buffer) < self.config.lookback_window:
            # Not enough data for technical indicators
            return [0.0] * len(self.config.technical_features)
        
        prices = np.array(self.price_buffer[-self.config.lookback_window:])
        
        # RSI
        rsi = self._compute_rsi(prices)
        features.append(rsi)
        
        # MACD
        macd, signal = self._compute_macd(prices)
        features.append(macd)
        features.append(signal)
        
        # Bollinger Bands
        bb_upper, bb_lower = self._compute_bollinger_bands(prices)
        features.append(bb_upper)
        features.append(bb_lower)
        
        # ATR
        atr = self._compute_atr(prices)
        features.append(atr)
        
        return features
    
    def _compute_rsi(self, prices: NDArray[np.float64], period: int = 14) -> float:
        """Compute RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _compute_macd(self, prices: NDArray[np.float64]) -> Tuple[float, float]:
        """Compute MACD indicator."""
        if len(prices) < 26:
            return 0.0, 0.0
        
        # EMA-12
        ema_12 = self._compute_ema(prices, 12)
        # EMA-26
        ema_26 = self._compute_ema(prices, 26)
        
        macd = ema_12 - ema_26
        
        # Signal line (EMA-9 of MACD)
        signal = macd  # Simplified; would need MACD history for proper signal
        
        return float(macd), float(signal)
    
    def _compute_ema(self, prices: NDArray[np.float64], period: int) -> float:
        """Compute Exponential Moving Average."""
        if len(prices) < period:
            return float(np.mean(prices))
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return float(ema)
    
    def _compute_bollinger_bands(self, prices: NDArray[np.float64], period: int = 20) -> Tuple[float, float]:
        """Compute Bollinger Bands."""
        if len(prices) < period:
            return 0.0, 0.0
        
        recent = prices[-period:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        bb_upper = (mean + 2 * std - prices[-1]) / (prices[-1] + 1e-8)
        bb_lower = (mean - 2 * std - prices[-1]) / (prices[-1] + 1e-8)
        
        return float(bb_upper), float(bb_lower)
    
    def _compute_atr(self, prices: NDArray[np.float64], period: int = 14) -> float:
        """Compute Average True Range."""
        if len(prices) < period + 1:
            return 0.0
        
        # Simplified ATR using price ranges
        ranges = np.abs(np.diff(prices))
        atr = np.mean(ranges[-period:])
        
        return float(atr / (prices[-1] + 1e-8))  # Normalize by current price
    
    def _encode_order_book(self, order_book: Dict[str, Any]) -> List[float]:
        """Encode order book features."""
        features = []
        
        # Bid-ask spread
        if "best_bid" in order_book and "best_ask" in order_book:
            spread = (order_book["best_ask"] - order_book["best_bid"]) / (order_book["best_bid"] + 1e-8)
            features.append(spread)
        
        # Order book imbalance
        if "bid_volume" in order_book and "ask_volume" in order_book:
            total_volume = order_book["bid_volume"] + order_book["ask_volume"]
            if total_volume > 0:
                imbalance = (order_book["bid_volume"] - order_book["ask_volume"]) / total_volume
            else:
                imbalance = 0.0
            features.append(imbalance)
        
        return features
    
    def _encode_position_info(self, position_info: Dict[str, Any]) -> List[float]:
        """Encode current position information."""
        features = []
        
        # Position size (normalized)
        position_size = position_info.get("size", 0.0)
        features.append(float(position_size))
        
        # Unrealized PnL
        unrealized_pnl = position_info.get("unrealized_pnl", 0.0)
        features.append(float(unrealized_pnl))
        
        # Position duration
        duration = position_info.get("duration", 0.0)
        features.append(float(duration))
        
        return features
    
    def _encode_portfolio_info(self, portfolio_info: Dict[str, Any]) -> List[float]:
        """Encode portfolio information."""
        features = []
        
        # Portfolio value
        portfolio_value = portfolio_info.get("value", 0.0)
        features.append(float(portfolio_value))
        
        # Cash ratio
        cash = portfolio_info.get("cash", 0.0)
        if portfolio_value > 0:
            cash_ratio = cash / portfolio_value
        else:
            cash_ratio = 1.0
        features.append(float(cash_ratio))
        
        # Daily PnL
        daily_pnl = portfolio_info.get("daily_pnl", 0.0)
        features.append(float(daily_pnl))
        
        return features
    
    def _update_statistics(self, state: NDArray[np.float64]) -> None:
        """Update feature statistics."""
        self.sample_count += 1
        
        if self.sample_count == 1:
            self.feature_means = {f"feature_{i}": float(v) for i, v in enumerate(state)}
            self.feature_stds = {f"feature_{i}": 1.0 for i, _ in enumerate(state)}
            self.feature_mins = {f"feature_{i}": float(v) for i, v in enumerate(state)}
            self.feature_maxs = {f"feature_{i}": float(v) for i, v in enumerate(state)}
        else:
            alpha = 1.0 / min(self.sample_count, self.config.normalization_window)
            
            for i, v in enumerate(state):
                key = f"feature_{i}"
                old_mean = self.feature_means[key]
                self.feature_means[key] = (1 - alpha) * old_mean + alpha * float(v)
                
                variance = (float(v) - old_mean) ** 2
                self.feature_stds[key] = np.sqrt((1 - alpha) * self.feature_stds[key] ** 2 + alpha * variance)
                
                self.feature_mins[key] = min(self.feature_mins[key], float(v))
                self.feature_maxs[key] = max(self.feature_maxs[key], float(v))
    
    def _normalize_state(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize state using running statistics."""
        if self.sample_count < 10:
            return state
        
        normalized = np.zeros_like(state)
        
        for i, v in enumerate(state):
            key = f"feature_{i}"
            mean = self.feature_means[key]
            std = max(self.feature_stds[key], 1e-8)
            
            # Z-score normalization
            normalized[i] = (float(v) - mean) / std
        
        # Clip extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized


# ============================================================================
# Action Decoder
# ============================================================================

class TradingAction(Enum):
    """Trading actions."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    CLOSE = auto()


@dataclass
class ActionConfig:
    """Configuration for action decoding."""
    action_dim: int = 4  # buy, sell, hold, close
    position_size_multiplier: float = 1.0
    min_position_size: float = 0.01
    max_position_size: float = 1.0
    use_continuous_actions: bool = False
    continuous_action_dim: int = 2  # direction, size


class ActionDecoder:
    """Decodes quantum outputs into trading actions."""
    
    def __init__(self, config: Optional[ActionConfig] = None):
        self.config = config or ActionConfig()
    
    def decode(
        self,
        quantum_output: NDArray[np.float64],
        current_position: float = 0.0,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Decode quantum output into trading action."""
        if self.config.use_continuous_actions:
            return self._decode_continuous(quantum_output, current_position, confidence)
        else:
            return self._decode_discrete(quantum_output, current_position, confidence)
    
    def _decode_discrete(
        self,
        quantum_output: NDArray[np.float64],
        current_position: float,
        confidence: float
    ) -> Dict[str, Any]:
        """Decode discrete actions."""
        # Get action probabilities
        if len(quantum_output) >= self.config.action_dim:
            probs = quantum_output[:self.config.action_dim]
        else:
            probs = np.pad(quantum_output, (0, self.config.action_dim - len(quantum_output)))
        
        probs = np.exp(probs) / (np.sum(np.exp(probs)) + 1e-10)
        
        # Select action
        action_idx = int(np.argmax(probs))
        
        # Map to trading action
        action_map = {
            0: TradingAction.BUY,
            1: TradingAction.SELL,
            2: TradingAction.HOLD,
            3: TradingAction.CLOSE
        }
        
        action = action_map.get(action_idx, TradingAction.HOLD)
        
        # Determine position size based on action
        position_size = self._determine_position_size(action, current_position, confidence)
        
        return {
            "action": action.name,
            "action_idx": action_idx,
            "position_size": position_size,
            "confidence": confidence,
            "probabilities": probs.tolist()
        }
    
    def _decode_continuous(
        self,
        quantum_output: NDArray[np.float64],
        current_position: float,
        confidence: float
    ) -> Dict[str, Any]:
        """Decode continuous actions (direction and size)."""
        if len(quantum_output) >= self.config.continuous_action_dim:
            direction = np.tanh(quantum_output[0])  # -1 to 1 (sell to buy)
            size = (np.tanh(quantum_output[1]) + 1) / 2  # 0 to 1
        else:
            direction = 0.0
            size = 0.5
        
        # Apply confidence scaling
        size = size * confidence
        
        # Scale position size
        position_change = direction * size * self.config.position_size_multiplier
        position_change = np.clip(
            position_change,
            -self.config.max_position_size,
            self.config.max_position_size
        )
        
        # Determine action type
        if abs(position_change) < 0.01:
            action = TradingAction.HOLD
        elif position_change > 0:
            action = TradingAction.BUY
        else:
            action = TradingAction.SELL
        
        return {
            "action": action.name,
            "direction": float(direction),
            "size": float(size),
            "position_change": float(position_change),
            "new_position": float(current_position + position_change),
            "confidence": confidence
        }
    
    def _determine_position_size(
        self,
        action: TradingAction,
        current_position: float,
        confidence: float
    ) -> float:
        """Determine position size based on action and confidence."""
        base_size = self.config.position_size_multiplier * confidence
        base_size = np.clip(base_size, self.config.min_position_size, self.config.max_position_size)
        
        if action == TradingAction.BUY:
            return base_size
        elif action == TradingAction.SELL:
            return -base_size
        elif action == TradingAction.CLOSE:
            return -current_position  # Close existing position
        else:  # HOLD
            return 0.0


# ============================================================================
# Reward Function
# ============================================================================

class RewardType(Enum):
    """Types of reward functions."""
    SIMPLE_RETURN = auto()
    RISK_ADJUSTED = auto()
    SHARPE_RATIO = auto()
    DRAWDOWN_PENALIZED = auto()
    MULTI_OBJECTIVE = auto()


@dataclass
class RewardConfig:
    """Configuration for reward function."""
    reward_type: RewardType = RewardType.RISK_ADJUSTED
    return_weight: float = 1.0
    risk_penalty: float = 0.1
    drawdown_penalty: float = 0.5
    sharpe_window: int = 20
    transaction_cost: float = 0.001
    holding_penalty: float = 0.0001


class TradingRewardFunction:
    """Computes rewards for trading actions."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # History for computing metrics
        self.returns_history: List[float] = []
        self.portfolio_history: List[float] = []
        self.max_portfolio_value: float = 0.0
    
    def compute(
        self,
        action: Dict[str, Any],
        market_data: Dict[str, Any],
        portfolio_info: Dict[str, Any],
        previous_portfolio_value: float
    ) -> float:
        """Compute reward for trading action."""
        current_portfolio_value = portfolio_info.get("value", 0.0)
        
        # Compute return
        if previous_portfolio_value > 0:
            period_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        else:
            period_return = 0.0
        
        # Subtract transaction costs
        position_change = abs(action.get("position_change", 0.0))
        transaction_cost = position_change * self.config.transaction_cost
        adjusted_return = period_return - transaction_cost
        
        # Update history
        self.returns_history.append(adjusted_return)
        self.portfolio_history.append(current_portfolio_value)
        self.max_portfolio_value = max(self.max_portfolio_value, current_portfolio_value)
        
        # Compute reward based on type
        if self.config.reward_type == RewardType.SIMPLE_RETURN:
            reward = adjusted_return
        
        elif self.config.reward_type == RewardType.RISK_ADJUSTED:
            reward = self._risk_adjusted_reward(adjusted_return)
        
        elif self.config.reward_type == RewardType.SHARPE_RATIO:
            reward = self._sharpe_reward(adjusted_return)
        
        elif self.config.reward_type == RewardType.DRAWDOWN_PENALIZED:
            reward = self._drawdown_penalized_reward(adjusted_return, current_portfolio_value)
        
        else:  # MULTI_OBJECTIVE
            reward = self._multi_objective_reward(adjusted_return, current_portfolio_value, action)
        
        return float(reward)
    
    def _risk_adjusted_reward(self, period_return: float) -> float:
        """Compute risk-adjusted reward."""
        if len(self.returns_history) < 2:
            return period_return
        
        # Compute recent volatility
        recent_returns = np.array(self.returns_history[-self.config.sharpe_window:])
        volatility = np.std(recent_returns) + 1e-8
        
        # Risk-adjusted return
        risk_adjusted = period_return / volatility
        
        return float(risk_adjusted)
    
    def _sharpe_reward(self, period_return: float) -> float:
        """Compute Sharpe ratio based reward."""
        if len(self.returns_history) < self.config.sharpe_window:
            return period_return
        
        # Compute rolling Sharpe ratio
        recent_returns = np.array(self.returns_history[-self.config.sharpe_window:])
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-8
        
        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        return float(sharpe)
    
    def _drawdown_penalized_reward(self, period_return: float, portfolio_value: float) -> float:
        """Compute drawdown-penalized reward."""
        # Compute current drawdown
        drawdown = (self.max_portfolio_value - portfolio_value) / (self.max_portfolio_value + 1e-8)
        
        # Penalize for drawdown
        drawdown_penalty = drawdown * self.config.drawdown_penalty
        
        return float(period_return - drawdown_penalty)
    
    def _multi_objective_reward(
        self,
        period_return: float,
        portfolio_value: float,
        action: Dict[str, Any]
    ) -> float:
        """Compute multi-objective reward combining return and risk."""
        # Return component
        return_component = period_return * self.config.return_weight
        
        # Risk component
        risk_component = -self._compute_risk_penalty() * self.config.risk_penalty
        
        # Drawdown component
        drawdown = (self.max_portfolio_value - portfolio_value) / (self.max_portfolio_value + 1e-8)
        drawdown_component = -drawdown * self.config.drawdown_penalty
        
        # Holding penalty (encourage activity)
        position_change = abs(action.get("position_change", 0.0))
        if position_change < 0.01:
            holding_penalty = -self.config.holding_penalty
        else:
            holding_penalty = 0.0
        
        return float(return_component + risk_component + drawdown_component + holding_penalty)
    
    def _compute_risk_penalty(self) -> float:
        """Compute risk penalty based on recent volatility."""
        if len(self.returns_history) < 10:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-self.config.sharpe_window:])
        volatility = np.std(recent_returns)
        
        return float(volatility)
    
    def reset(self) -> None:
        """Reset reward function state."""
        self.returns_history.clear()
        self.portfolio_history.clear()
        self.max_portfolio_value = 0.0


# ============================================================================
# Risk Management Integration
# ============================================================================

@dataclass
class RiskLimits:
    """Risk limits for trading."""
    max_position_size: float = 1.0
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    max_drawdown: float = 0.10  # 10% max drawdown
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_concentration: float = 0.3  # 30% max in single position


class RiskManager:
    """Integrates risk management with quantum RL."""
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        
        # Tracking
        self.daily_pnl: float = 0.0
        self.max_portfolio_value: float = 0.0
        self.current_portfolio_value: float = 0.0
        self.positions: Dict[str, float] = {}
    
    def check_action(
        self,
        action: Dict[str, Any],
        portfolio_info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if action is within risk limits."""
        self._update_portfolio_info(portfolio_info)
        
        violations = []
        adjusted_action = action.copy()
        
        # Check position size
        position_change = abs(action.get("position_change", 0.0))
        if position_change > self.limits.max_position_size:
            violations.append("position_size_exceeded")
            adjusted_action["position_change"] = np.sign(action.get("position_change", 0.0)) * self.limits.max_position_size
        
        # Check drawdown
        if self.current_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.current_portfolio_value) / self.max_portfolio_value
            if drawdown > self.limits.max_drawdown:
                violations.append("max_drawdown_exceeded")
                adjusted_action["position_change"] = 0.0  # Force hold
        
        # Check daily loss
        if self.daily_pnl < -self.limits.max_daily_loss * self.max_portfolio_value:
            violations.append("max_daily_loss_exceeded")
            adjusted_action["position_change"] = 0.0  # Force hold
        
        # Check concentration
        symbol = portfolio_info.get("symbol", "default")
        current_concentration = abs(self.positions.get(symbol, 0.0)) / (self.current_portfolio_value + 1e-8)
        new_concentration = current_concentration + abs(position_change)
        
        if new_concentration > self.limits.max_concentration:
            violations.append("concentration_exceeded")
            max_additional = self.limits.max_concentration - current_concentration
            adjusted_action["position_change"] = np.sign(action.get("position_change", 0.0)) * max_additional
        
        approved = len(violations) == 0
        
        return approved, {
            "approved": approved,
            "violations": violations,
            "original_action": action,
            "adjusted_action": adjusted_action,
            "risk_metrics": {
                "daily_pnl": self.daily_pnl,
                "drawdown": (self.max_portfolio_value - self.current_portfolio_value) / (self.max_portfolio_value + 1e-8),
                "concentration": new_concentration
            }
        }
    
    def _update_portfolio_info(self, portfolio_info: Dict[str, Any]) -> None:
        """Update portfolio information."""
        self.current_portfolio_value = portfolio_info.get("value", 0.0)
        self.max_portfolio_value = max(self.max_portfolio_value, self.current_portfolio_value)
        self.daily_pnl = portfolio_info.get("daily_pnl", 0.0)
        
        # Update positions
        if "positions" in portfolio_info:
            self.positions = portfolio_info["positions"]
    
    def reset_daily(self) -> None:
        """Reset daily tracking."""
        self.daily_pnl = 0.0
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        drawdown = 0.0
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.current_portfolio_value) / self.max_portfolio_value
        
        return {
            "daily_pnl": self.daily_pnl,
            "drawdown": drawdown,
            "current_value": self.current_portfolio_value,
            "max_value": self.max_portfolio_value
        }


__all__ = [
    # Market state encoder
    "MarketStateEncoder",
    "MarketStateConfig",
    "FeatureType",
    
    # Action decoder
    "ActionDecoder",
    "ActionConfig",
    "TradingAction",
    
    # Reward function
    "TradingRewardFunction",
    "RewardConfig",
    "RewardType",
    
    # Risk management
    "RiskManager",
    "RiskLimits"
]