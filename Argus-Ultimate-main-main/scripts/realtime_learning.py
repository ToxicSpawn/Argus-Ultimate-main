"""
Real-Time Learning Integration for Argus

This connects the learning system to live Argus trading:
1. Real-time data input (from Argus data feed)
2. Real-time learning (updates after every bar)
3. Real-time signals (to execution)
4. Real-time performance tracking

Usage:
    from scripts.realtime_learning import RealTimeLearning
    
    # Initialize once
    rt = RealTimeLearning()
    
    # In on_bar() callback:
    rt.on_bar(df, price, symbol)
    
    # Get signal
    signal = rt.get_signal()
    
    # After trade:
    rt.on_trade(pnl, actual_return)
    
    # Get stats
    rt.get_performance()
"""

import logging
import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Callable
import threading

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False


class RealTimeLearning:
    """
    Real-time learning that integrates with live Argus.
    
    Flow:
    1. on_bar() → called every new bar
    2. Updates features 
    3. Runs learning
    4. Generates signal
    5. on_trade() → called after trade closes
    6. Updates with outcome
    
    Works with:
    - Paper trading (simulation)
    - Live trading
    - Backtest
    """

    def __init__(
        self,
        # Learning rates
        fast_lr: float = 0.1,
        slow_lr: float = 0.01,
        # Thresholds
        min_confidence: float = 0.50,
        min_trade_interval: int = 300,
        max_trades_per_hour: int = 4,
        # Feature config
        feature_dim: int = 9,
        # Offline mode (for Sydney)
        offline: bool = True,
    ):
        # Config
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.min_confidence = min_confidence
        self.min_trade_interval = min_trade_interval
        self.max_trades_per_hour = max_trades_per_hour
        self.feature_dim = feature_dim
        self.offline = offline
        
        # Feature storage (sliding window)
        self.price_history: deque = deque(maxlen=100)
        self.high_history: deque = deque(maxlen=100)
        self.low_history: deque = deque(maxlen=100)
        self.volume_history: deque = deque(maxlen=100)
        
        # Learning state
        self.features: deque = deque(maxlen=500)
        self.returns: deque = deque(maxlen=500)
        self.trades: deque = deque(maxlen=200)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Model weights (simple linear)
        self.weights = np.zeros(feature_dim)
        self.bias = 0.0
        
        # Feature importance
        self.feature_importance = np.ones(feature_dim)
        
        # Accuracy tracking
        self.recent_correct = deque(maxlen=30)
        self.baseline = None
        self.drift_detected = False
        self.drift_count = 0
        
        # Timing
        self.last_trade_time = None
        self.last_bar_time = None
        
        # Stats
        self.total_bars = 0
        self.total_trades = 0
        self.total_pnl = 0
        
        # Performance
        self.equity = 10000
        self.equity_curve = [10000]
        
        # Feature names
        self.feature_names = ['r1', 'r4', 'r12', 'r24', 'v12', 'v24', 'rsi', 'pp', 'vr']
        
        # Callbacks
        self.on_signal_callback: Optional[Callable] = None
        self.on_trade_callback: Optional[Callable] = None
        
        logger.info("=" * 60)
        logger.info("REAL-TIME LEARNING INITIALIZED")
        logger.info("=" * 60)
        logger.info("Offline: {}".format(offline))
        logger.info("Min confidence: {:.0%}".format(min_confidence))
        logger.info("Min trade interval: {}s".format(min_trade_interval))
        logger.info("Max trades/hour: {}".format(max_trades_per_hour))
        logger.info("=" * 60)

    def on_bar(self, df, current_price: float, symbol: str = "BTC/USDT"):
        """
        Called on every new bar.
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current price
            symbol: Trading symbol
        """
        with self._lock:
            # Update history
            self._update_history(df)
            
            # Need enough data
            if len(self.price_history) < 25:
                return None
            
            # Extract features
            features = self._extract_features()
            
            # Calculate return
            if len(self.price_history) >= 2:
                ret = (current_price / list(self.price_history)[-2]) - 1
            else:
                ret = 0
            
            # Learn from bar
            self._learn_from_bar(features, ret)
            
            # Update timing
            self.last_bar_time = datetime.now(timezone.utc)
            
            self.total_bars += 1
            
            return features

    def _update_history(self, df):
        """Update price history."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        for i in range(len(close)):
            self.price_history.append(close[i])
            self.high_history.append(high[i])
            self.low_history.append(low[i])
            self.volume_history.append(volume[i])

    def _extract_features(self) -> np.ndarray:
        """Extract 9 features from history."""
        close = np.array(list(self.price_history))
        high = np.array(list(self.high_history))
        low = np.array(list(self.low_history))
        volume = np.array(list(self.volume_history))
        
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
        if len(low) >= 25:
            num = close[-1] - np.min(low[-25:])
            den = np.max(high[-25:]) - np.min(low[-25:]) + 1e-8
            pp = num / den
        else:
            pp = 0.5
        
        # Volume ratio
        vr = volume[-1] / np.mean(volume[-25:]) if len(volume) >= 25 else 1.0
        
        features = np.array([r1, r4, r12, r24, v12, v24, rsi, pp, vr])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.features.append(features.copy())
        
        return features

    def _learn_from_bar(self, features: np.ndarray, ret: float):
        """Learn from every bar."""
        self.returns.append(ret)
        
        # Update weights if we have enough data
        if len(self.features) >= 20:
            self._update_weights(features, ret)
        
        # Update feature importance
        if len(self.features) >= 30:
            self._update_importance()

    def _update_weights(self, features: np.ndarray, ret: float):
        """Update model weights."""
        # Simple correlation-based update
        if len(self.features) >= 20:
            recent_features = np.array(list(self.features)[-20:])
            recent_returns = np.array(list(self.returns)[-20:])
            
            for i in range(self.feature_dim):
                if abs(recent_features[:, i]).sum() > 0.001:
                    try:
                        corr = np.corrcoef(recent_features[:, i], recent_returns)[0, 1]
                        if not np.isnan(corr):
                            self.weights[i] += self.slow_lr * corr
                    except:
                        pass
            
            # Bias update
            if abs(recent_returns).sum() > 0:
                self.bias += self.slow_lr * np.mean(recent_returns)

    def _update_importance(self):
        """Update feature importance."""
        if len(self.features) < 30:
            return
        
        recent_features = np.array(list(self.features)[-30:])
        recent_returns = np.array(list(self.returns)[-30:])
        
        for i in range(self.feature_dim):
            if abs(recent_features[:, i]).sum() > 0.01:
                try:
                    corr = np.corrcoef(recent_features[:, i], recent_returns)[0, 1]
                    if not np.isnan(corr):
                        self.feature_importance[i] = 0.9 * self.feature_importance[i] + 0.1 * abs(corr)
                except:
                    pass

    def detect_regime(self) -> str:
        """Detect current regime."""
        if len(self.features) < 25:
            return "sideways"
        
        f = self.features[-1]
        
        if f[3] > 0.02 and f[5] < 0.02:
            return "bull"
        elif f[3] < -0.02 and f[5] < 0.02:
            return "bear"
        return "sideways"

    def get_signal(self) -> Dict:
        """Get current trading signal."""
        with self._lock:
            if len(self.features) < 25:
                return {
                    'action': 'hold',
                    'direction': 1,
                    'confidence': 0.5,
                    'regime': 'sideways',
                    'score': 0
                }
            
            features = self.features[-1]
            regime = self.detect_regime()
            
            # Calculate signal
            weighted = features * self.weights * self.feature_importance
            score = np.sum(weighted) + self.bias
            
            # Map to action
            if score > 0.005:
                action = 'buy'
                direction = 2
            elif score < -0.005:
                action = 'sell'
                direction = 0
            else:
                action = 'hold'
                direction = 1
            
            # Confidence
            weight_mag = np.mean(np.abs(self.weights))
            accuracy = self.get_accuracy()
            
            confidence = 0.5 + weight_mag + (accuracy - 0.5) * 0.2
            confidence = np.clip(confidence, 0.35, 0.85)
            
            return {
                'action': action,
                'direction': direction,
                'confidence': confidence,
                'regime': regime,
                'score': score,
                'weight_magnitude': weight_mag,
                'features': features
            }

    def should_trade(self, signal: Dict) -> Tuple[bool, str]:
        """Check if should trade."""
        now = datetime.now(timezone.utc)
        
        # Confidence
        if signal['confidence'] < self.min_confidence:
            return False, "Low confidence ({:.0%})".format(signal['confidence'])
        
        # Hold signal
        if signal['action'] == 'hold':
            return False, "Hold signal"
        
        # Trade interval
        if self.last_trade_time:
            elapsed = (now - self.last_trade_time).total_seconds()
            if elapsed < self.min_trade_interval:
                return False, "Too soon ({:.0f}s)".format(elapsed)
        
        # Max trades per hour
        if self.total_trades > 0 and self.total_trades % 100 == 0:
            # Check recent trades
            recent_trades = [
                t for t in self.trades
                if (now - t.get('timestamp', now)).total_seconds() < 3600
            ]
            if len(recent_trades) >= self.max_trades_per_hour:
                return False, "Max trades/hour"
        
        # Drift
        if self.drift_detected:
            return False, "Drift detected"
        
        return True, "Trade approved"

    def on_trade(self, pnl: float, actual_return: float):
        """
        Called after trade closes.
        
        Args:
            pnl: Profit/loss in $
            actual_return: Actual return as decimal (e.g., 0.02 for 2%)
        """
        with self._lock:
            # Get last signal
            signal = self.get_signal()
            
            # Record trade
            trade = {
                'signal': signal,
                'pnl': pnl,
                'actual_return': actual_return,
                'timestamp': datetime.now(timezone.utc),
                'equity': self.equity + pnl
            }
            self.trades.append(trade)
            
            # Update equity
            self.equity += pnl
            self.equity_curve.append(self.equity)
            
            # Correct?
            direction = signal['direction']
            if direction == 2 and actual_return > 0.01:
                correct = True
            elif direction == 0 and actual_return < -0.01:
                correct = True
            elif direction == 1 and abs(actual_return) <= 0.01:
                correct = True
            else:
                correct = False
            
            self.recent_correct.append(1 if correct else 0)
            
            # Fast learning from outcome
            if len(self.recent_correct) >= 3:
                self._fast_learn(signal, correct, actual_return)
            
            # Drift detection
            if len(self.recent_correct) >= 20:
                self._check_drift()
            
            # Update timing
            self.last_trade_time = datetime.now(timezone.utc)
            self.total_trades += 1
            self.total_pnl += pnl
            
            # Callback
            if self.on_trade_callback:
                self.on_trade_callback(trade)

    def _fast_learn(self, signal: Dict, correct: bool, ret: float):
        """Fast learning from trade outcome."""
        if len(self.features) < 2:
            return
        
        features = self.features[-1]
        
        # Correct = reinforce
        delta = self.fast_lr if correct else -self.fast_lr
        direction_mult = 1 if signal['direction'] == 2 else -1 if signal['direction'] == 0 else 0
        
        # Update weights
        self.weights += delta * direction_mult * features * 0.5
        self.bias += delta * ret * 0.1

    def _check_drift(self):
        """Check for drift."""
        if len(self.recent_correct) < 20:
            return
        
        recent = list(self.recent_correct)[-20:]
        older = list(self.recent_correct)[:20]
        
        recent_acc = sum(recent) / 20
        older_acc = sum(older) / 20
        
        if recent_acc < older_acc - 0.15:
            self.drift_detected = True
            self.drift_count += 1
            self.weights *= 0.5  # Halve weights
            logger.warning("Drift! Recent: {:.0%}, Older: {:.0%}".format(recent_acc, older_acc))
        else:
            self.drift_detected = False

    def get_accuracy(self) -> float:
        """Get accuracy."""
        if len(self.recent_correct) == 0:
            return 0.5
        return sum(self.recent_correct) / len(self.recent_correct)

    def get_performance(self) -> Dict:
        """Get performance."""
        top_features = [
            (self.feature_names[i], self.feature_importance[i])
            for i in range(self.feature_dim)
        ]
        top_features.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_bars': self.total_bars,
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'accuracy': self.get_accuracy(),
            'equity': self.equity,
            'weights': self.weights.tolist(),
            'feature_importance': top_features[:3],
            'drift_detected': self.drift_detected,
            'drift_count': self.drift_count,
            'regime': self.detect_regime()
        }


# ============================================================================
# INTEGRATION WITH ARGUS
# ============================================================================

def create_realtime_learning() -> RealTimeLearning:
    """Create realtime learning for Argus."""
    return RealTimeLearning(
        fast_lr=0.1,
        slow_lr=0.01,
        min_confidence=0.50,
        min_trade_interval=300,
        max_trades_per_hour=4,
        offline=True  # Set False for live exchange
    )


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    
    print()
    print("=" * 60)
    print("REAL-TIME LEARNING TEST")
    print("=" * 60)
    print()
    
    # Create
    rt = create_realtime_learning()
    
    # Simulate data
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(200) * 100)
    
    df = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.rand(200) * 1000 + 500
    })
    
    # Run
    trades = 0
    for i in range(25, len(df)):
        price = df['close'].iloc[i]
        
        # On bar
        rt.on_bar(df.iloc[:i+1], price)
        
        # Get signal
        signal = rt.get_signal()
        
        # Should trade?
        if rt.should_trade(signal)[0] and i > 50:
            # Simulate trade
            ret = np.random.randn() * 0.01
            pnl = 1000 * ret
            
            trades += 1
            rt.on_trade(pnl, ret)
        
        if i % 30 == 0:
            perf = rt.get_performance()
            print("Bar {:3d}: Trades={:3d} PnL=${:.0f} Acc={:.0%} Regime={}".format(
                i, trades, perf['total_pnl'], perf['accuracy'], perf['regime']))
    
    print()
    perf = rt.get_performance()
    print("=" * 60)
    print("FINAL")
    print("=" * 60)
    print("Total bars: {}".format(perf['total_bars']))
    print("Total trades: {}".format(perf['total_trades']))
    print("Total PnL: ${:.2f}".format(perf['total_pnl']))
    print("Accuracy: {:.0%}".format(perf['accuracy']))
    print("Equity: ${:.2f}".format(perf['equity']))
    print("Drift events: {}".format(perf['drift_count']))
    print("Regime: {}".format(perf['regime']))