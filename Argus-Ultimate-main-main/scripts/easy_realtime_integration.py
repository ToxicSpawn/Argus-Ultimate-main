"""
Easy Integration for Real-Time Learning

Add this to any trading strategy to enable real-time learning:

Usage:
    from scripts.easy_realtime_integration import enable_realtime_learning
    
    # In your trading strategy:
    enable_realtime_learning()
    
    # The system will now automatically:
    # - Extract features from market data
    # - Learn from trade outcomes
    # - Adapt to new patterns
    # - Detect concept drift
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Global learning state
_learning_enabled = False
_learner = None
_pending_trades = {}


def enable_realtime_learning():
    """Enable real-time learning for all trading."""
    global _learning_enabled, _learner
    
    if _learning_enabled:
        return _learner
    
    try:
        # Add parent directory to path
        import sys
        from pathlib import Path
        parent = Path(__file__).resolve().parent.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        
        from scripts.realtime_learning_integration import get_bridge
        _learner = get_bridge()
        _learning_enabled = True
        logger.info("Real-time learning enabled")
        return _learner
    except Exception as e:
        logger.warning(f"Could not enable real-time learning: {e}")
        return None


def get_predictions() -> Optional[Dict]:
    """Get the bridge for predictions."""
    global _learner, _learning_enabled
    if not _learning_enabled:
        enable_realtime_learning()
    return _learner


def learn_from_trade(symbol: str, signal: Dict, pnl: float, actual_return: float):
    """
    Call this after each trade to enable learning.
    
    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        signal: Signal dict from predict() call
        pnl: Profit/loss from trade
        actual_return: Actual return percentage
    """
    global _learner
    if _learner is None:
        enable_realtime_learning()
    
    if _learner is not None:
        _learner.update_from_trade(symbol, signal, pnl, actual_return)


def get_learning_status() -> Dict:
    """Get current learning status."""
    global _learner
    if _learner is None:
        return {'enabled': False}
    
    return {
        'enabled': True,
        **(_learner.get_status())
    }


def adapt_position_size(base_size: float) -> float:
    """
    Adapt position size based on learning state.
    
    Args:
        base_size: Base position size
    
    Returns:
        Adjusted position size
    """
    global _learner
    if _learner is None:
        return base_size
    
    multiplier = _learner.suggest_position_multiplier()
    return base_size * multiplier


# ============================================================================
# DECORATOR FOR AUTOMATIC LEARNING
# ============================================================================

def with_realtime_learning(func: Callable) -> Callable:
    """
    Decorator that automatically enables learning for a trading function.
    
    Usage:
        @with_realtime_learning
        def on_trade_complete(self, trade):
            # Your trade logic
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Enable learning if not already
        enable_realtime_learning()
        
        # Call original function
        result = func(*args, **kwargs)
        
        return result
    
    return wrapper


def learning_signal(func: Callable) -> Callable:
    """
    Decorator for signal generation functions.
    Automatically extracts features and enables learning.
    
    Usage:
        @learning_signal
        def generate_signal(self, df, price):
            # Your signal logic
            return {'action': 'buy', 'confidence': 0.8}
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _learner
        
        # Enable learning if not already
        enable_realtime_learning()
        
        # Call original function
        result = func(*args, **kwargs)
        
        # If result contains features, record them
        if _learner is not None and isinstance(result, dict):
            if 'features' not in result and len(args) > 1:
                # Try to extract features from dataframe
                df = kwargs.get('df') or (args[1] if len(args) > 1 else None)
                if df is not None:
                    result['features'] = _learner.extract_features(df)
        
        return result
    
    return wrapper


# ============================================================================
# SIMPLE CLI TO TEST
# ============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("REAL-TIME LEARNING - EASY INTEGRATION")
    print("=" * 60)
    print()
    print("To enable real-time learning in your strategy:")
    print()
    print("1. Import the module:")
    print("   from scripts.easy_realtime_integration import enable_realtime_learning")
    print()
    print("2. Enable learning at startup:")
    print("   enable_realtime_learning()")
    print()
    print("3. After each trade, record the outcome:")
    print("   from scripts.easy_realtime_integration import learn_from_trade")
    print("   learn_from_trade('BTC/USDT', signal, pnl, actual_return)")
    print()
    print("4. Get adapted position size:")
    print("   from scripts.easy_realtime_integration import adapt_position_size")
    print("   size = adapt_position_size(1000)  # $1000 base")
    print()
    print("=" * 60)
    print()
    
    # Quick test
    enable_realtime_learning()
    status = get_learning_status()
    print(f"Status: {'Enabled' if status.get('enabled') else 'Disabled'}")
    
    if status.get('enabled'):
        print(f"Total trades: {status.get('total_trades', 0)}")
        print(f"Accuracy: {status.get('recent_accuracy', 0):.1%}")
        print(f"Position multiplier: {adapt_position_size(1000):.2f}")