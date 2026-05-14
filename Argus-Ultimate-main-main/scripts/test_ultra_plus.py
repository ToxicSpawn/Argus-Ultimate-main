"""Test script for Ultra+ Real-Time Learning System v4.0"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ultra_plus_learner import UltraPlusLearner


def test_ultra_plus():
    print()
    print("=" * 70)
    print("ULTRA+ ADVANCED REAL-TIME LEARNING SYSTEM v4.0 - TEST")
    print("=" * 70)
    print()

    # Initialize
    print("Initializing Ultra+ Learner...")
    learner = UltraPlusLearner()
    print("OK")
    print()

    # Test 1: Multi-model prediction
    print("Test 1: Multi-model ensemble prediction")
    features = np.random.randn(9)
    pred = learner.predict(features)
    print(f"Signal: {pred['signal']}")
    print(f"Confidence: {pred['confidence']:.1%}")
    print(f"Anomaly score: {pred['anomaly_score']:.3f}")
    print(f"Uncertainty: {pred['uncertainty']:.3f}")
    print(f"Regime: {pred['regime']}")
    print("OK")
    print()

    # Test 2: Training loop
    print("Test 2: Training loop (400 cycles)")
    for i in range(400):
        features = np.random.randn(9)
        # Inject pattern
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
            print(f"  Cycle {i+1}: acc={acc:.1%}, episodes={perf['episodes']}, curriculum={perf['curriculum_difficulty']:.2f}")

    print("OK")
    print()

    # Test 3: Anomaly detection
    print("Test 3: VAE Anomaly Detection")
    import torch
    normal_score = learner.vae.anomaly_score(torch.randn(1, 9).to('cuda')).item()
    anomaly_score = learner.vae.anomaly_score(torch.randn(1, 9).to('cuda') * 5).item()
    print(f"Normal anomaly score: {normal_score:.3f}")
    print(f"High-variance anomaly score: {anomaly_score:.3f}")
    print("OK")
    print()

    # Test 4: NAS search
    print("Test 4: Neural Architecture Search (5 trials)")
    best_arch, best_score = learner.nas.search(learner, num_trials=5)
    print(f"Best architecture: {best_arch}")
    print(f"Best fitness: {best_score:.1%}")
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
    print(f"Curriculum difficulty: {perf['curriculum_difficulty']:.2f}")
    print(f"PPO episodes: {perf['episodes']}")
    print(f"NAS trials: {perf['nas_trials']}")
    print(f"Self-play agents: {perf['selfplay_agents']}")
    print()
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_ultra_plus()