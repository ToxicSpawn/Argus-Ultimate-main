"""
Live Continuous Learning

Learns every 0.5 seconds in real-time.

Run: py scripts/live_learn.py
"""

import logging
import asyncio
import time
from collections import deque
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


class LiveLearner:
    """Learns continuously every 0.5s."""

    def __init__(self):
        self.prices = deque(maxlen=500)
        self.returns = deque(maxlen=500)
        self.features = deque(maxlen=500)
        
        self.weights = np.zeros(9)
        self.bias = 0.0
        self.feature_importance = np.ones(9)
        self.feature_names = ['r1', 'r4', 'r12', 'r24', 'v12', 'v24', 'rsi', 'pp', 'vr']
        
        self.total_updates = 0
        self.correct = deque(maxlen=30)
        
        self.signal = 'hold'
        self.confidence = 0.5
        self.price = 0.0
        self.is_running = False
        self.start_time = None
        
        print("LiveLearner initialized - updates every 0.5s")

    def update(self, price: float):
        """Update every 0.5 seconds."""
        self.price = price
        self.prices.append(price)
        
        if len(self.prices) >= 2:
            ret = (price / list(self.prices)[-2]) - 1
            self.returns.append(ret)
        
        if len(self.prices) >= 25:
            f = self._extract()
            self.features.append(f.copy())
            self._learn(f, ret)
        
        self.total_updates += 1
        self._make_signal()

    def _extract(self):
        p = np.array(list(self.prices))
        if len(p) < 25:
            return np.zeros(9)
        
        r1 = p[-1] / p[-2] - 1
        r4 = p[-1] / p[-5] - 1
        r12 = p[-1] / p[-13] - 1
        r24 = p[-1] / p[-25] - 1
        v12 = np.std(p[-13:]) / np.mean(p[-13:])
        v24 = np.std(p[-25:]) / np.mean(p[-25:])
        
        d = np.diff(p)
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        rsi = 50
        if len(g) >= 14:
            rsi = 100 - (100 / (1 + np.mean(g[-14:]) / max(np.mean(l[-14:]), 1e-8)))
        
        return np.array([r1, r4, r12, r24, v12, v24, rsi, 0.5, 1.0])

    def _learn(self, f, r):
        if len(self.features) < 10:
            return
        
        rf = np.array(list(self.features)[-20:])
        rr = np.array(list(self.returns)[-20:])
        
        for i in range(9):
            if abs(rf[:, i]).sum() > 0.001:
                try:
                    c = np.corrcoef(rf[:, i], rr)[0, 1]
                    if not np.isnan(c):
                        self.feature_importance[i] = 0.95 * self.feature_importance[i] + 0.05 * abs(c)
                except:
                    pass
        
        for i in range(9):
            if self.feature_importance[i] > 0.3:
                self.weights[i] += 0.001 * r * self.feature_importance[i]
        
        self.bias += 0.0001 * r

    def _make_signal(self):
        if len(self.features) < 10:
            self.signal = 'hold'
            self.confidence = 0.5
            return
        
        f = self.features[-1]
        score = np.dot(f, self.weights) * self.feature_importance.sum()
        
        if score > 0.001:
            self.signal = 'buy'
        elif score < -0.001:
            self.signal = 'sell'
        else:
            self.signal = 'hold'
        
        self.confidence = min(0.5 + np.sum(np.abs(self.weights)) * 2, 0.85)

    def should_trade(self):
        if self.signal == 'hold':
            return False, "Hold"
        if self.confidence < 0.45:
            return False, "Low conf"
        if len(self.features) < 20:
            return False, "Warming up"
        return True, "OK"

    def get_state(self):
        return {
            'signal': self.signal,
            'confidence': self.confidence,
            'price': self.price,
            'updates': self.total_updates,
            'weights': float(np.sum(np.abs(self.weights))),
            'top': sorted(zip(self.feature_names, self.feature_importance), key=lambda x: -x[1])[:3]
        }


class Sim:
    def __init__(self, price=50000):
        self.price = price
    
    def next(self):
        self.price *= 1 + np.random.randn() * 0.002
        return max(self.price, 100)


async def main():
    logging.basicConfig(level=logging.WARNING)
    
    print("=" * 50)
    print("LIVE CONTINUOUS LEARNING")
    print("Runs every 0.5 seconds")
    print("Ctrl+C to stop")
    print("=" * 50)
    
    learner = LiveLearner()
    sim = Sim(50000)
    
    learner.is_running = True
    learner.start_time = time.time()
    start = time.time()
    
    try:
        while True:
            learner.update(sim.next())
            
            if learner.total_updates % 20 == 0:
                s = learner.get_state()
                print("{:4.0f}s | {:4d} | ${:.0f} | {} ({:.0%})".format(
                    time.time() - start, s['updates'], s['price'], s['signal'], s['confidence']))
                
                if learner.should_trade()[0]:
                    print("  >>> {}".format(s['signal'].upper()))
            
            await asyncio.sleep(0.5)
    
    except KeyboardInterrupt:
        pass
    
    print("=" * 50)
    s = learner.get_state()
    print("Updates: {} | Signal: {} | Weights: {:.3f}".format(
        s['updates'], s['signal'], s['weights']))
    print("Top features: {}".format(s['top']))


if __name__ == "__main__":
    asyncio.run(main())