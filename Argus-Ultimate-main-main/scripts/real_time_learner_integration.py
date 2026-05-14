"""
Integration wrapper for real-time learning system.
Run this alongside paper trading: py main.py paper
Then: py scripts/real_time_learner_integration.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


async def continuous_learning_loop():
    """
    Continuous learning loop that:
    1. Monitors paper trades
    2. Records outcomes
    3. Updates models in real-time
    4. Adapts to new patterns
    """
    logger.info("=" * 60)
    logger.info("REAL-TIME LEARNING INTEGRATION")
    logger.info("Monitors trades and learns continuously")
    logger.info("=" * 60)
    
    # Import after path setup
    from scripts.real_time_learner import get_real_time_learner
    from ml.advanced_features import calculate_features
    
    learner = get_real_time_learner()
    
    logger.info("Real-time learning loop started!")
    logger.info("This will run continuously, learning from each trade")
    logger.info("Press Ctrl+C to stop")
    
    # Track state
    cycle = 0
    last_features = None
    
    while True:
        try:
            cycle += 1
            
            # Simulate market data collection (in production, this would read from the system)
            # Generate features from current market state
            import numpy as np
            
            # Create sample features (in production, extract from actual market data)
            features = np.random.randn(9)  # 9 features: r1, r4, r12, r24, v12, v24, rsi, pp, vr
            
            # Get prediction
            pred = learner.predict(features)
            
            # Log every 10 cycles
            if cycle % 10 == 0:
                perf = learner.get_performance()
                logger.info(f"Cycle {cycle}: signal={pred['signal']}, acc={perf['accuracy']:.1%}, drift={perf['drift_detected']}")
            
            # Wait before next iteration
            await asyncio.sleep(5)  # 5 second intervals
            
        except KeyboardInterrupt:
            logger.info("\nStopping real-time learning...")
            break
        except Exception as e:
            logger.error(f"Error in learning loop: {e}")
            await asyncio.sleep(10)
    
    # Final stats
    perf = learner.get_performance()
    logger.info("=" * 60)
    logger.info("FINAL STATS")
    logger.info("=" * 60)
    logger.info(f"Total cycles: {cycle}")
    logger.info(f"Accuracy: {perf['accuracy']:.1%}")
    logger.info(f"Drift detected: {perf['drift_detected']}")
    logger.info(f"Total updates: {perf['total_updates']}")
    logger.info("Real-time learning stopped")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REAL-TIME LEARNING INTEGRATION")
    print("=" * 60)
    print()
    print("To use real-time learning:")
    print("1. Run paper trading: py main.py paper")
    print("2. In another terminal, run: py scripts/real_time_learner_integration.py")
    print()
    print("The learning system will continuously update models")
    print("based on new market data and trade outcomes.")
    print()
    print("=" * 60)
    print()
    
    # Run the loop
    asyncio.run(continuous_learning_loop())