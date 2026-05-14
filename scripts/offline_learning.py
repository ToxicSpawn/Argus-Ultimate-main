"""
Offline Learning System v4.0

Complete offline learning system:
- Local data generation (no exchange needed)
- All v4.0 ML features
- Works completely offline

Run: py scripts/offline_learning.py
"""

import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import threading

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# LOCAL DATA SYSTEM
# ============================================================================

class LocalMarketGenerator:
    """Generate realistic market data."""
    
    def __init__(self, initial_price: float = 50000):
        self.price = initial_price
        self.prices = [initial_price]
        self.regime = "sideways"
    
    def step(self) -> float:
        """Generate next return."""
        # Volatility
        vol = 0.02
        if len(self.prices) > 20:
            vol = np.std(self.prices[-20:]) / np.mean(self.prices[-20:])
        
        # Regime drift
        drift = {'bull': 0.001, 'bear': -0.001, 'sideways': 0}[self.regime]
        
        r = drift + np.random.randn() * vol
        self.price *= (1 + r)
        self.price = max(self.price, 100)
        self.prices.append(self.price)
        
        return r
    
    def generate_df(self, n: int) -> 'pd.DataFrame':
        """Generate DataFrame."""
        import pandas as pd
        
        returns = []
        for _ in range(n):
            returns.append(self.step())
        
        prices = [self.price]
        for r in reversed(returns):
            prices.append(prices[-1] / (1 + r))
        prices.reverse()
        
        df = pd.DataFrame({
            'close': prices[1:],
            'high': [p * 1.01 for p in prices[1:]],
            'low': [p * 0.99 for p in prices[1:]],
            'open': prices[:-1],
            'volume': [abs(np.random.randn()) * 1000 + 500 for _ in range(n)]
        })
        
        return df


# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

class GeneticOptimizer:
    """GA for parameter optimization."""
    
    def __init__(self, pop_size: int = 10):
        self.pop_size = pop_size
        self.population = [np.random.randn(12) * 0.1 for _ in range(pop_size)]
        self.best = None
        self.best_fitness = -float('inf')
        self.gen = 0
    
    def evolve(self, fitness: float):
        """Evolve one generation."""
        # Track best
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best = self.population[0].copy()
        
        # Simple evolution
        if np.random.rand() < 0.3:
            idx = np.random.randint(len(self.population))
            self.population[idx] += np.random.randn(12) * 0.05
        
        self.gen += 1
        
        return {
            'gen': self.gen,
            'best': self.best_fitness
        }


# ============================================================================
# RL AGENT
# ============================================================================

class RLAgent:
    """Q-learning agent."""
    
    def __init__(self):
        self.q = np.zeros((9, 3))
        self.epsilon = 1.0
        self.history = deque(maxlen=50)
    
    def act(self, features: np.ndarray, regime: str) -> int:
        """Choose action."""
        state = hash(tuple(features[:3])) % 9
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        
        return np.argmax(self.q[state])
    
    def update(self, s, a, r, ns):
        """Update Q."""
        self.q[s, a] += 0.1 * (r + 0.95 * max(self.q[ns]) - self.q[s, a])
        self.history.append(r)
        self.epsilon = max(0.1, self.epsilon * 0.99)


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class SydneyLearningSystem:
    """
    Offline learning system for Sydney.
    
    Works without any exchange API.
    Uses local generated data.
    """
    
    def __init__(self):
        # Data
        self.generator = LocalMarketGenerator(50000)
        
        # GA
        self.ga = GeneticOptimizer(10)
        
        # RL
        self.rl = RLAgent()
        
        # History
        self.features: deque = deque(maxlen=200)
        self.returns: deque = deque(maxlen=200)
        self.trades: deque = deque(maxlen=100)
        
        self._lock = threading.Lock()
        
        # Stats
        self.correct = deque(maxlen=30)
        self.total_learned = 0
        self.total_trades = 0
        self.equity = 10000
        
        logger.info("=" * 50)
        logger.info("OFFLINE LEARNING SYSTEM v4.0")
        logger.info("=" * 50)
        logger.info("Data: Local (offline)")
        logger.info("GA: Enabled")
        logger.info("RL: Enabled")
        logger.info("=" * 50)
    
    def generate_data(self, n: int = 100) -> 'pd.DataFrame':
        """Generate training data."""
        return self.generator.generate_df(n)
    
    def extract_features(self, close, high, low, volume) -> np.ndarray:
        """Extract 9 features."""
        if len(close) < 25:
            return np.zeros(9)
        
        r1 = close[-1] / close[-2] - 1 if close[-2] != 0 else 0
        r4 = close[-1] / close[-5] - 1 if len(close) > 5 else 0
        r12 = close[-1] / close[-13] - 1 if len(close) > 13 else 0
        r24 = close[-1] / close[-25] - 1 if len(close) > 25 else 0
        
        v12 = np.std(close[-13:]) / np.mean(close[-13:]) if len(close) >= 13 else 0
        v24 = np.std(close[-25:]) / np.mean(close[-25:]) if len(close) >= 25 else 0
        
        # RSI
        d = np.diff(close)
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        if len(g) >= 14:
            avg_g = np.mean(g[-14:])
            avg_l = np.mean(l[-14:])
            rsi = 100 - (100 / (1 + avg_g / max(avg_l, 1e-8)))
        else:
            rsi = 50
        
        # Position
        if len(low) >= 25:
            num = close[-1] - np.min(low[-25:])
            den = np.max(high[-25:]) - np.min(low[-25:]) + 1e-8
            pp = num / den
        else:
            pp = 0.5
        
        vr = volume[-1] / np.mean(volume[-25:]) if len(volume) >= 25 else 1.0
        
        return np.array([r1, r4, r12, r24, v12, v24, rsi, pp, vr])
    
    def detect_regime(self, features: np.ndarray) -> str:
        """Detect regime."""
        if features[3] > 0.02 and features[5] < 0.02:
            return "bull"
        elif features[3] < -0.02 and features[5] < 0.02:
            return "bear"
        return "sideways"
    
    def learn_bar(self, features: np.ndarray, ret: float):
        """Learn from bar."""
        with self._lock:
            self.features.append(features.copy())
            self.returns.append(ret)
            self.total_learned += 1
            
            # GA evolve every 30
            if self.total_learned % 30 == 0:
                acc = self.get_accuracy()
                self.ga.evolve(acc)
    
    def generate_signal(self, features: np.ndarray) -> Dict:
        """Generate signal."""
        regime = self.detect_regime(features)
        
        # RL action
        rl_a = self.rl.act(features, regime)
        
        # Heuristic
        if features[0] > 0.005:
            direction = 2
        elif features[0] < -0.005:
            direction = 0
        else:
            direction = 1
        
        # Blend
        direction = int(0.3 * rl_a + 0.7 * direction)
        
        confidence = 0.50
        if self.total_learned > 30:
            confidence += 0.05
        
        action = {0: "sell", 1: "hold", 2: "buy"}[direction]
        
        return {
            'action': action,
            'direction': direction,
            'confidence': confidence,
            'regime': regime,
            'features': features
        }
    
    def should_trade(self, signal: Dict) -> Tuple[bool, str]:
        if signal['confidence'] < 0.45:
            return False, "Low conf"
        if signal['action'] == 'hold':
            return False, "Hold"
        if self.total_trades < 10:
            return True, "OK"
        return True, "OK"
    
    def learn_outcome(self, signal: Dict, pnl: float, ret: float):
        """Learn from trade."""
        with self._lock:
            self.trades.append({'signal': signal, 'pnl': pnl})
            
            correct = (signal['direction'] == 2 and ret > 0.01) or \
                    (signal['direction'] == 0 and ret < -0.01) or \
                    (signal['direction'] == 1 and abs(ret) < 0.01)
            
            self.correct.append(1 if correct else 0)
            self.total_trades += 1
            self.equity += pnl
            
            # RL update
            if len(self.features) >= 2:
                s = hash(tuple(self.features[-2][:3])) % 9
                ns = hash(tuple(self.features[-1][:3])) % 9
                self.rl.update(s, signal['direction'], ret, ns)
    
    def get_accuracy(self) -> float:
        if len(self.correct) == 0:
            return 0.5
        return sum(self.correct) / len(self.correct)
    
    def get_stats(self) -> Dict:
        return {
            'learned': self.total_learned,
            'trades': self.total_trades,
            'accuracy': self.get_accuracy(),
            'ga_gen': self.ga.gen,
            'ga_best': self.ga.best_fitness,
            'rl_epsilon': self.rl.epsilon,
            'equity': self.equity
        }


# ============================================================================
# TEST
# ============================================================================

def run_test(cycles: int = 200):
    """Run test."""
    import pandas as pd
    
    print()
    print("=" * 50)
    print("SYDNEY LEARNING SYSTEM TEST")
    print("=" * 50)
    print()
    
    system = SydneyLearningSystem()
    
    # Generate data
    print("Generating data...")
    df = system.generate_data(cycles + 50)
    
    trades = 0
    for i in range(50, len(df)):
        close = df['close'].values[:i+1]
        high = df['high'].values[:i+1]
        low = df['low'].values[:i+1]
        volume = df['volume'].values[:i+1]
        
        features = system.extract_features(close, high, low, volume)
        ret = close[-1] / close[-2] - 1
        
        # Learn
        system.learn_bar(features, ret)
        
        # Signal
        signal = system.generate_signal(features)
        
        # Trade?
        if system.should_trade(signal)[0]:
            pnl = 1000 * ret * (1 if signal['direction'] == 2 else -1)
            pnl -= 2
            
            trades += 1
            system.learn_outcome(signal, pnl, ret)
        
        if i % 30 == 0:
            s = system.get_stats()
            print("Cycle {:3d}: Trades={:3d} Acc={:.0%} GA={} RL={:.2f}".format(
                i - 50, trades, s['accuracy'], s['ga_gen'], s['rl_epsilon']))
    
    print()
    s = system.get_stats()
    print("=" * 50)
    print("FINAL")
    print("=" * 50)
    print("Total learned: {}".format(s['learned']))
    print("Total trades: {}".format(s['trades']))
    print("Accuracy: {:.0%}".format(s['accuracy']))
    print("GA generation: {}".format(s['ga_gen']))
    print("GA best: {:.0%}".format(s['ga_best']))
    print("RL epsilon: {:.2f}".format(s['rl_epsilon']))
    print("Equity: ${:.2f}".format(s['equity']))
    
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_test(200)