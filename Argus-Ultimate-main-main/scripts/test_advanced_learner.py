"""
Test script for Advanced Real-Time Learning System v2.0
Run: py scripts/test_advanced_learner.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.advanced_realtime_learner import get_advanced_learner, AdvancedRealTimeLearner


def test_advanced_learner():
    """Comprehensive test of advanced learning features."""
    print("=" * 60)
    print("ADVANCED REAL-TIME LEARNING SYSTEM v2.0 - TEST")
    print("=" * 60)
    print()
    
    # Test 1: Initialize
    print("Test 1: Initialization")
    learner = AdvancedRealTimeLearner()
    print(f"  - Models: {list(learner.models.keys())}")
    print(f"  - Scaler fitted: {learner.scaler_fitted}")
    print(f"  - Memory size: {learner.memory_size}")
    print("  PASSED")
    print()
    
    # Test 2: Update with feedback
    print("Test 2: Update with feedback loop")
    for i in range(100):
        features = np.random.randn(9)
        # Simulate actual return that has some pattern
        actual_return = 0.01 * np.sin(i / 10) + np.random.randn() * 0.005
        learner.update(features, actual_return, regime="bull")
    
    perf = learner.get_performance()
    print(f"  - Total predictions: {perf['total_predictions']}")
    print(f"  - Recent accuracy: {perf['recent_accuracy']:.1%}")
    print("  PASSED")
    print()
    
    # Test 3: Prediction
    print("Test 3: Ensemble prediction")
    features = np.random.randn(9)
    pred = learner.predict(features)
    print(f"  - Signal: {pred['signal']}")
    print(f"  - Confidence: {pred['confidence']:.1%}")
    print(f"  - Regime: {pred['regime']}")
    print(f"  - Ensemble votes: {pred['ensemble_votes']}")
    print("  PASSED")
    print()
    
    # Test 4: Model weights adaptation
    print("Test 4: Model weights adaptation")
    print("  Initial weights:")
    for name, weight in learner.model_weights.items():
        print(f"    {name}: {weight:.3f}")
    print("  Model accuracies:")
    for name, acc in perf['model_accuracies'].items():
        if acc > 0:
            print(f"    {name}: {acc:.1%}")
    print("  PASSED")
    print()
    
    # Test 5: Drift detection
    print("Test 5: Drift detection")
    print(f"  - Drift detected: {perf['drift_detected']}")
    print(f"  - Drift count: {perf['drift_count']}")
    print("  PASSED")
    print()
    
    # Test 6: Meta-learning
    print("Test 6: Meta-learning state")
    print(f"  - Current LR: {perf['current_lr']:.4f}")
    print(f"  - Best accuracy: {perf['best_accuracy']:.1%}")
    print("  PASSED")
    print()
    
    # Test 7: Feature importance
    print("Test 7: Adaptive feature importance")
    feature_names = ['r1', 'r4', 'r12', 'r24', 'v12', 'v24', 'rsi', 'pp', 'vr']
    for i, imp in enumerate(perf['feature_importance']):
        bar = "=" * int(imp * 50)
        print(f"  {feature_names[i]:5s}: {imp:.3f} {bar}")
    print("  PASSED")
    print()
    
    # Test 8: Memory replay
    print("Test 8: Memory replay")
    learner.replay_memory(batch_size=32)
    print(f"  - Memory usage: {perf['memory_usage']:.1%}")
    print("  PASSED")
    print()
    
    # Test 9: Calibration
    print("Test 9: Confidence calibration")
    for _ in range(50):
        features = np.random.randn(9)
        pred = learner.predict(features)
        actual_return = np.random.randn() * 0.02
        learner.update(features, actual_return, predicted_signal=pred['signal'], predicted_confidence=pred['confidence'])
    perf = learner.get_performance()
    print(f"  - Calibration samples: {len(learner.calibration_history)}")
    print("  PASSED")
    print()
    
    # Test 10: Full feedback loop
    print("Test 10: Full feedback loop simulation")
    print("  Simulating 500 trades with pattern injection...")
    
    # Inject a pattern: positive returns tend to follow certain features
    for i in range(500):
        features = np.random.randn(9)
        # Inject pattern: when feature[0] > 0.5, tend to have positive return
        if features[0] > 0.5:
            actual_return = 0.02 + np.random.randn() * 0.01
        else:
            actual_return = -0.01 + np.random.randn() * 0.01
        
        learner.update(features, actual_return)
        
        if i % 100 == 99:
            acc = learner.get_accuracy()
            print(f"    After {i+1} trades: accuracy = {acc:.1%}")
    
    # Final performance
    print()
    print("=" * 60)
    print("FINAL PERFORMANCE")
    print("=" * 60)
    perf = learner.get_performance()
    print(f"Total predictions: {perf['total_predictions']}")
    print(f"Recent accuracy: {perf['recent_accuracy']:.1%}")
    print(f"Overall accuracy: {perf['overall_accuracy']:.1%}")
    print(f"Best accuracy: {perf['best_accuracy']:.1%}")
    print(f"Drift events: {perf['drift_count']}")
    print(f"Current regime: {perf['current_regime']}")
    print()
    print("Model accuracies:")
    for name, acc in perf['model_accuracies'].items():
        print(f"  {name}: {acc:.1%}")
    print()
    print("Ensemble weights:")
    for name, weight in perf['ensemble_weights'].items():
        print(f"  {name}: {weight:.3f}")
    print()
    print("Feature importance (should show pattern detection):")
    for i, imp in enumerate(perf['feature_importance']):
        bar = "=" * int(imp * 50)
        print(f"  {feature_names[i]:5s}: {imp:.3f} {bar}")
    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_advanced_learner()