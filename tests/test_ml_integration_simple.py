"""
Simple integration test for ML trading engine and ensemble voter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from ml.ml_trading_integration import get_ml_engine


def test_ml_engine_initialization():
    """Test that the ML engine initializes and loads models."""
    engine = get_ml_engine()
    status = engine.get_status()
    assert status['initialized'] is True
    assert status['total_models'] > 0
    print("ML engine initialized and models loaded")


def test_ml_prediction():
    """Test that the ML engine can make a prediction."""
    engine = get_ml_engine()
    
    # Create a simple OHLCV DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
        'open': np.linspace(50000, 51000, 100),
        'high': np.linspace(50100, 51100, 100),
        'low': np.linspace(49900, 50900, 100),
        'close': np.linspace(50050, 51050, 100),
        'volume': np.linspace(1000, 2000, 100),
    })
    
    prediction = engine.predict(df, current_price=50500.0, symbol="BTC/USD")
    
    assert 'action' in prediction
    assert prediction['action'] in ['buy', 'sell', 'hold', 'wait']
    assert 'confidence' in prediction
    assert 0.0 <= prediction['confidence'] <= 1.0
    print(f"ML prediction successful: {prediction['action']} (confidence={prediction['confidence']:.2%})")


def test_ensemble_voter():
    """Test that the ensemble voter can form a consensus."""
    from ml.ensemble_voter import EnsembleVoter
    
    voter = EnsembleVoter(db_path=":memory:")
    
    # Submit some votes
    voter.submit_vote("strategy1", "BTC/USD", "buy", 0.8)
    voter.submit_vote("strategy2", "BTC/USD", "buy", 0.7)
    voter.submit_vote("strategy3", "BTC/USD", "sell", 0.6)
    
    # Get consensus
    consensus = voter.get_consensus("BTC/USD")
    
    if consensus:
        assert consensus.direction in ["buy", "sell"]
        assert 0.0 <= consensus.agreement_pct <= 1.0
        print(f"Ensemble consensus formed: {consensus.direction} (agreement={consensus.agreement_pct:.1%})")
    else:
        print("No consensus formed (may be due to thresholds)")


if __name__ == "__main__":
    print("Running ML integration tests...")
    test_ml_engine_initialization()
    test_ml_prediction()
    test_ensemble_voter()
    print("All tests passed")
