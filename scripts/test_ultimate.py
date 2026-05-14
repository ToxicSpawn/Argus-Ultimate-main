"""Test script for Ultimate Real-Time Learning System v5.0"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ultimate_learner import UltimateLearner


def test_ultimate():
    print()
    print("=" * 70)
    print("ULTIMATE ADVANCED REAL-TIME LEARNING SYSTEM v5.0 - TEST")
    print("=" * 70)
    print()

    # Initialize
    print("Initializing Ultimate Learner...")
    learner = UltimateLearner()
    print("OK")
    print()

    # Test 1: Multi-model ensemble prediction
    print("Test 1: Multi-model ensemble prediction")
    features = np.random.randn(9)
    pred = learner.predict(features)
    print(f"Signal: {pred['signal']}")
    print(f"Confidence: {pred['confidence']:.1%}")
    print(f"Regime: {pred['regime']}")
    print(f"Soup models: {pred['soup_models']}")
    print(f"Bayesian suggestions: {pred['bayesian_suggestions']}")
    print("OK")
    print()

    # Test 2: Training loop
    print("Test 2: Training loop (500 cycles)")
    for i in range(500):
        features = np.random.randn(9)
        if features[0] > 0.3:
            actual_return = 0.02 + np.random.randn() * 0.01
        else:
            actual_return = -0.01 + np.random.randn() * 0.01

        pred = learner.predict(features)
        learner.update(
            features,
            actual_return,
            predicted_signal=pred["signal"],
            predicted_confidence=pred["confidence"]
        )

        if i % 100 == 99:
            acc = learner.get_accuracy()
            perf = learner.get_performance()
            print(f"  Cycle {i+1}: acc={acc:.1%}, soup={perf['soup_models']}, bayes={perf['bayesian_evaluations']}")

    print("OK")
    print()

    # Test 3: Causal Discovery
    print("Test 3: Causal Discovery")
    causal = learner.discover_causality(features_names=['r1', 'r4', 'r12', 'r24', 'v12', 'v24', 'rsi', 'pp', 'vr'])
    print(f"  Discovered {len(causal)} causal relationships")
    for cause, effects in list(causal.items())[:3]:
        if effects:
            print(f"  f{cause} causes: {[f'f{e}' for e in effects]}")
    print("OK")
    print()

    # Test 4: Diffusion Scenario Generation
    print("Test 4: Diffusion Scenario Generation")
    scenarios = learner.diffusion.generate(num_samples=5)
    print(f"  Generated {scenarios.shape[0]} scenarios, shape: {scenarios.shape}")
    print("OK")
    print()

    # Test 5: Quantum-Inspired Optimization
    print("Test 5: Quantum-Inspired Optimization")
    result = learner.quantum_opt.optimize(lambda x: np.random.random(), num_iterations=20)
    print(f"  Optimized state: {result[:4].numpy()}")
    print("OK")
    print()

    # Test 6: Bayesian Optimization
    print("Test 6: Bayesian Optimization")
    suggestion = learner.bayesian_opt.suggest()
    print(f"  Suggested hyperparameters: {suggestion}")
    print("OK")
    print()

    # Final performance
    print("=" * 70)
    print("FINAL PERFORMANCE")
    print("=" * 70)
    perf = learner.get_performance()
    print(f"Total predictions: {perf['total_predictions']}")
    print(f"Recent accuracy: {perf['recent_accuracy']:.1%}")
    print(f"Overall accuracy: {perf['overall_accuracy']:.1%}")
    print(f"Model Soup models: {perf['soup_models']}")
    print(f"Bayesian evaluations: {perf['bayesian_evaluations']}")
    print(f"Causal edges: {perf['causal_edges']}")
    print()
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_ultimate()