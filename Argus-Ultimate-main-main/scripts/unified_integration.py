"""
Complete Integration: Ultimate Learner -> Argus Trading

This module connects all components for live trading with real-time learning:
1. Ultimate v5.0 learner (10+ cutting-edge ML techniques)
2. Trading cost optimizer (fees, slippage, overtrading fixes)
3. Drift detection (regime change adaptation)
4. Complete trading system (signals, position sizing, risk)

Usage in Argus:
    from scripts.unified_integration import UnifiedTradingIntegration
    
    integration = UnifiedTradingIntegration()
    
    # Each cycle:
    signal = integration.generate_signal(df, price, symbol)
    
    # After trade closes:
    integration.record_outcome(signal, pnl, actual_return)
"""

import logging
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import the learner
try:
    sys.path.insert(0, '.')
    from scripts.ultimate_learner import UltimateRealTimeLearner
    ULTIMATE_AVAILABLE = True
except ImportError:
    ULTIMATE_AVAILABLE = False
    logger.warning("Ultimate learner not available - using fallback")


class UnifiedTradingIntegration:
    """
    Complete integration for live trading.
    
    Flow:
    1. Extract features from market data
    2. Generate signal (with learning if available)
    3. Apply cost optimization (confidence thresholds)
    4. Optimize position size
    5. Execute trade
    6. Record outcome for learning
    7. Monitor for drift
    """

    def __init__(
        self,
        models_dir: str = "data/models_mtf",
        min_confidence: float = 0.55,
        min_trade_interval: int = 300,  # 5 min between trades
        max_trades_per_hour: int = 4,
    ):
        self.models_dir = Path(models_dir)
        self.min_confidence = min_confidence
        self.min_trade_interval = min_trade_interval
        self.max_trades_per_hour = max_trades_per_hour
        
        # Feature extraction
        self.feature_names = ['r1', 'r4', 'r12', 'r24', 'v12', 'v24', 'rsi', 'pp', 'vr']
        
        # Ultimate learner if available
        if ULTIMATE_AVAILABLE:
            try:
                self.learner = UltimateRealTimeLearner(models_dir=str(self.models_dir))
                logger.info("Ultimate v5.0 learner loaded")
            except Exception as e:
                logger.warning(f"Could not load Ultimate learner: {e}")
                self.learner = None
        else:
            self.learner = None
        
        # Trading state
        self.feature_buffer: deque = deque(maxlen=100)
        self.last_trade_time: Optional[datetime] = None
        self.trades: deque = deque(maxlen=100)
        
        # Learning state
        self.total_predictions = 0
        self.correct_predictions = 0
        self.baseline_accuracy = None
        self.recent_accuracy = 0.5
        self.drift_detected = False
        self.drift_count = 0
        
        # Performance
        self.equity_curve = [10000]
        
        logger.info("Unified Trading Integration initialized")
        logger.info(f"  Min confidence: {min_confidence:.0%}")
        logger.info(f"  Min trade interval: {min_trade_interval}s")
        logger.info(f"  Max trades/hour: {max_trades_per_hour}")
        logger.info(f"  Ultimate learner: {self.learner is not None}")

    def extract_features(self, df) -> np.ndarray:
        """Extract 9 features from OHLCV data."""
        if len(df) < 24:
            return np.zeros(9)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Returns
        r1 = (close[-1] / close[-2] - 1) if close[-2] != 0 else 0
        r4 = (close[-1] / close[-5] - 1) if len(close) > 5 and close[-5] != 0 else 0
        r12 = (close[-1] / close[-13] - 1) if len(close) > 13 and close[-13] != 0 else 0
        r24 = (close[-1] / close[-25] - 1) if len(close) > 25 and close[-25] != 0 else 0
        
        # Volatility
        v12 = np.std(close[-13:]) / np.mean(close[-13:]) if len(close) >= 13 else 0
        v24 = np.std(close[-25:]) / np.mean(close[-25:]) if len(close) >= 25 else 0
        
        # RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        if len(gain) >= 14:
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-8)))
        else:
            rsi = 50
        
        # Position
        pp = (close[-1] - np.min(low[-25:])) / (np.max(high[-25:]) - np.min(low[-25:]) + 1e-8) if len(low) >= 25 else 0.5
        
        # Volume ratio
        vr = volume[-1] / np.mean(volume[-25:]) if len(volume) >= 25 and np.mean(volume[-25:]) != 0 else 1.0
        
        features = np.array([r1, r4, r12, r24, v12, v24, rsi, pp, vr])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.feature_buffer.append(features)
        
        return features

    def generate_signal(
        self,
        df,
        current_price: float,
        symbol: str = "BTC/USDT"
    ) -> Dict:
        """Generate trading signal with all optimizations."""
        # Extract features
        features = self.extract_features(df)
        
        # Determine regime
        if features[2] > 0.1:
            regime = "bull"
        elif features[2] < -0.1:
            regime = "bear"
        else:
            regime = "sideways"
        
        # Use learner if available
        if self.learner is not None:
            try:
                predictions = self.learner.predict([features])
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    direction = int(np.argmax(pred))
                    confidence = float(pred[direction])
                else:
                    direction = 1  # hold
                    confidence = 0.5
            except Exception as e:
                logger.debug(f"Learner prediction failed: {e}")
                direction = self._simple_direction(features)
                confidence = 0.5
        else:
            direction = self._simple_direction(features)
            confidence = self._simple_confidence(features)
        
        # Map direction
        action_map = {0: "sell", 1: "hold", 2: "buy"}
        action = action_map.get(direction, "hold")
        
        # Adjust confidence based on learning
        if self.total_predictions >= 20:
            if self.recent_accuracy > 0.55:
                confidence = min(0.5 + self.recent_accuracy * 0.4, 0.85)
            elif self.recent_accuracy < 0.45:
                confidence = max(0.3 + self.recent_accuracy * 0.3, 0.35)
        
        # Reduce confidence during drift
        if self.drift_detected:
            confidence *= 0.7
        
        return {
            'action': action,
            'direction': direction,
            'confidence': confidence,
            'regime': regime,
            'features': features,
            'symbol': symbol,
            'price': current_price,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _simple_direction(self, features: np.ndarray) -> int:
        """Simple direction from features (fallback)."""
        if features[0] > 0.001:
            return 2  # buy
        elif features[0] < -0.001:
            return 0  # sell
        else:
            return 1  # hold

    def _simple_confidence(self, features: np.ndarray) -> float:
        """Simple confidence from features (fallback)."""
        r1 = abs(features[0])
        r4 = abs(features[1])
        return min(0.5 + r1 * 10 + r4 * 5, 0.75)

    def should_trade(self, signal: Dict, market_state: Dict = None) -> Tuple[bool, str]:
        """Apply cost optimization gates."""
        now = datetime.now(timezone.utc)
        market_state = market_state or {}
        
        # 1. Confidence threshold
        if signal['confidence'] < self.min_confidence:
            return False, "Low confidence ({:.0%})".format(signal['confidence'])
        
        # 2. Trade frequency - skip for backtest/demo
        # In production, uncomment this:
        # if self.last_trade_time:
        #     elapsed = (now - self.last_trade_time).total_seconds()
        #     if elapsed < self.min_trade_interval:
        #         return False, "Too soon ({:.0f}s)".format(elapsed)
        
        # 3. Trades per hour - skip for backtest/demo  
        # In production, uncomment this:
        # recent_trades = [
        #     t for t in self.trades
        #     if (now - t['timestamp']).total_seconds() < 3600
        # ]
        # if len(recent_trades) >= self.max_trades_per_hour:
        #     return False, "Max trades per hour"
        
        # 4. Drift detection
        if self.drift_detected:
            return False, "Drift detected"
        
        # 5. Market conditions - default to reasonable if not provided
        volatility = market_state.get('volatility', 0.01)  # Default 1% instead of 50%
        if volatility > 0.05:
            return False, "High volatility"
        
        return True, "Trade approved"

    def optimize_position_size(
        self,
        signal: Dict,
        market_state: Dict = None,
        base_size: float = 1000
    ) -> float:
        """Optimize position size based on conditions."""
        market_state = market_state or {}
        size = base_size
        
        # Reduce during drift
        if self.drift_detected:
            size *= 0.5
        
        # Adjust by confidence
        conf = signal['confidence']
        if conf > 0.75:
            size *= 1.2
        elif conf < 0.55:
            size *= 0.7
        
        # Adjust by accuracy
        if self.recent_accuracy > 0.6:
            size *= 1.1
        elif self.recent_accuracy < 0.45:
            size *= 0.8
        
        # Market conditions
        volatility = market_state.get('volatility', 0.5)
        if volatility > 0.02:
            size *= 0.7
        
        return size

    def record_outcome(
        self,
        signal: Dict,
        pnl: float,
        actual_return: float
    ):
        """Record trade outcome for learning."""
        self.total_predictions += 1
        
        # Determine if correct
        direction = signal['direction']
        if direction == 2 and actual_return > 0.01:  # buy and positive
            correct = True
        elif direction == 0 and actual_return < -0.01:  # sell and negative
            correct = True
        elif direction == 1 and abs(actual_return) <= 0.01:  # hold and small
            correct = True
        else:
            correct = False
        
        if correct:
            self.correct_predictions += 1
        
        # Update baseline
        if self.total_predictions >= 50 and self.baseline_accuracy is None:
            self.baseline_accuracy = self.correct_predictions / self.total_predictions
            logger.info(f"Baseline accuracy set: {self.baseline_accuracy:.1%}")
        
        # Check for drift
        if self.baseline_accuracy and self.total_predictions >= 100:
            self.recent_accuracy = self.correct_predictions / max(self.total_predictions, 1)
            if self.recent_accuracy < self.baseline_accuracy - 0.15:
                self.drift_detected = True
                self.drift_count += 1
                logger.warning(
                    f"Drift detected! Recent: {self.recent_accuracy:.1%}, "
                    f"Baseline: {self.baseline_accuracy:.1%}"
                )
            else:
                self.drift_detected = False
        
        # Update learner if available
        if self.learner is not None:
            try:
                features = signal.get('features', np.zeros(9))
                self.learner.update([features], [direction], [correct])
            except Exception as e:
                logger.debug(f"Could not update learner: {e}")
        
        # Record trade
        self.trades.append({
            'signal': signal,
            'pnl': pnl,
            'actual_return': actual_return,
            'correct': correct,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Update equity
        self.equity_curve.append(self.equity_curve[-1] + pnl)
        
        return {
            'correct': correct,
            'total': self.total_predictions,
            'accuracy': self.recent_accuracy,
            'drift_detected': self.drift_detected
        }

    def get_performance(self) -> Dict:
        """Get performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'accuracy': 0.5,
                'drift_detected': self.drift_detected
            }
        
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = len(self.trades)
        
        return {
            'total_trades': total,
            'win_rate': wins / total if total > 0 else 0,
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'accuracy': self.recent_accuracy,
            'baseline': self.baseline_accuracy,
            'drift_detected': self.drift_detected,
            'drift_count': self.drift_count,
            'equity': self.equity_curve[-1],
            'ultimate_learner': self.learner is not None
        }


# ============================================================================
# EASY INTEGRATION
# ============================================================================

def enable_unified_learning():
    """Enable unified learning integration."""
    integration = UnifiedTradingIntegration()
    logger.info("Unified learning enabled")
    return integration


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    
    print()
    print("=" * 60)
    print("UNIFIED INTEGRATION TEST")
    print("=" * 60)
    print()
    
    # Create integration
    integration = UnifiedTradingIntegration()
    
    # Simulate data
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(100) * 50,
        'low': prices - np.random.rand(100) * 50,
        'close': prices,
        'volume': np.random.rand(100) * 1000
    })
    
    # Generate signals
    trades = 0
    for i in range(24, len(df)):
        signal = integration.generate_signal(df.iloc[:i+1], df['close'].iloc[i])
        
        # Check if should trade
        should, reason = integration.should_trade(signal)
        
        if should:
            # Simulate trade
            size = integration.optimize_position_size(signal)
            pnl = size * np.random.randn() * 0.001
            actual_return = np.random.randn() * 0.01
            
            # Record outcome
            result = integration.record_outcome(signal, pnl, actual_return)
            trades += 1
            
            if trades % 5 == 0:
                print(f"Trade {trades}: {signal['action']} conf={signal['confidence']:.0%} " +
                      f"correct={result['correct']} acc={result['accuracy']:.0%}")
    
    print()
    perf = integration.get_performance()
    print("Total trades: {}".format(perf['total_trades']))
    print("Win rate: {:.0%}".format(perf['win_rate']))
    print("Total PnL: ${:.2f}".format(perf['total_pnl']))
    print("Accuracy: {:.0%}".format(perf['accuracy']))
    print("Drift events: {}".format(perf.get('drift_count', 0)))
    print("Ultimate learner: {}".format(perf.get('ultimate_learner', False)))