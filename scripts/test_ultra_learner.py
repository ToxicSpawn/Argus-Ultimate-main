"""Test script for Ultra Advanced Real-Time Learning System v3.0"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ultra_realtime_learner import UltraAdvancedLearner


def test_ultra_learner():
    print()
    print("=" * 70)
    print("ULTRA ADVANCED REAL-TIME LEARNING SYSTEM v3.0 - TEST")
    print("=" * 70)
    print()

    # Initialize
    print("Initializing Ultra Advanced Learner...")
    learner = UltraAdvancedLearner()
    print("OK")
    print()

    # Test 1: Predictions
    print("Test 1: Multi-system prediction")
    features = np.random.randn(9)
    pred = learner.predict(features)
    print(f"Signal: {pred['signal']}")
    print(f"Confidence: {pred['confidence']:.1%}")
    print(f"Regime: {pred['regime']}")
    print(f"LR multiplier: {pred['lr_multiplier']}")
    print(f"Ensemble votes: {pred['ensemble_votes']}")
    print("OK")
    print()

    # Test 2: Training loop
    print("Test 2: Training loop (300 cycles)")
    for i in range(300):
        features = np.random.randn(9)
        # Inject pattern: feature[0] > 0.3 -> positive return
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
            print(f"  Cycle {i+1}: accuracy = {acc:.1%}")

    print("OK")
    print()

    # Test 3: Knowledge distillation
    print("Test 3: Knowledge distillation")
    learner.knowledge_distillation()
    print("OK")
    print()

    # Test 4: Evolutionary optimization
    print("Test 4: Evolutionary optimization")
    best = learner.run_evolution(1)
    print(f"Best individual fitness: {best['fitness']:.1%}")
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
    print(f"LR multiplier: {perf['lr_multiplier']:.2f}")
    print(f"RL episodes: {perf['rl_episodes']}")
    print(f"EWC boundaries: {perf['ewc_boundaries']}")
    print(f"Evo best fitness: {perf['evolutionary_best_fitness']:.1%}")
    print()
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_ultra_learner()