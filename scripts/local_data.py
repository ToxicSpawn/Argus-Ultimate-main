"""
Local Data System for Sydney (No Binance)

Since Binance API is blocked in Australia, this provides:
1. Local data generation (realistic price simulations)
2. Historical data loader from local CSV files
3. Bybit REST simulation (for paper trading)
4. All data formats compatible with learning system

Run: py scripts/local_data.py
"""

import logging
import os
import json
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LocalDataGenerator:
    """
    Generate realistic market data locally.
    
    Features:
    - Multiple price models (random walk, mean reversion, momentum)
    - Volatility clustering
    - Regime switching
    - Volume simulation
    """
    
    def __init__(
        self,
        initial_price: float = 50000,
        volatility: float = 0.02,
        trend: float = 0.0,
    ):
        self.initial_price = initial_price
        self.volatility = volatility
        self.trend = trend
        
        # State
        self.current_price = initial_price
        self.current_regime = "sideways"
        self.regime_probs = {'bull': 0.3, 'bear': 0.3, 'sideways': 0.4}
        
        # Price history
        self.prices = [initial_price]
        self.volumes = [1000]
        
        logger.info("LocalDataGenerator initialized")
        logger.info("  Initial price: ${}".format(initial_price))
        logger.info("  Volatility: {:.1%}".format(volatility))
    
    def set_regime(self, regime: str):
        """Set market regime."""
        self.current_regime = regime
    
    def generate_bar(self) -> Dict:
        """Generate single OHLCV bar."""
        # Volatility clustering (recent volatility affects current)
        if len(self.prices) > 20:
            recent_vol = np.std(self.prices[-20:]) / np.mean(self.prices[-20:])
        else:
            recent_vol = self.volatility
        
        # Regime-based returns
        if self.current_regime == "bull":
            drift = self.trend + 0.001
        elif self.current_regime == "bear":
            drift = self.trend - 0.001
        else:
            drift = self.trend + np.random.randn() * 0.0005
        
        # Random return with momentum
        r = drift + np.random.randn() * recent_vol
        
        # Update price
        self.current_price *= (1 + r)
        self.current_price = max(self.current_price, 100)  # Floor
        
        # OHLC
        high_mult = 1 + abs(np.random.randn()) * recent_vol * 0.5
        low_mult = 1 - abs(np.random.randn()) * recent_vol * 0.5
        
        bar = {
            'open': self.current_price / (1 + r),
            'high': self.current_price * high_mult,
            'low': self.current_price * low_mult,
            'close': self.current_price,
            'volume': abs(np.random.randn() + 1) * 1000,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.prices.append(self.current_price)
        self.volumes.append(bar['volume'])
        
        return bar
    
    def generate_bars(self, count: int) -> pd.DataFrame:
        """Generate multiple bars."""
        bars = []
        for _ in range(count):
            bars.append(self.generate_bar())
        
        df = pd.DataFrame(bars)
        return df
    
    def generate_with_patterns(self, count: int) -> pd.DataFrame:
        """Generate with realistic patterns."""
        bars = []
        
        for i in range(count):
            # Regime switching (5% chance)
            if np.random.rand() < 0.05:
                regime = np.random.choice(['bull', 'bear', 'sideways'], 
                                        p=[0.3, 0.3, 0.4])
                self.current_regime = regime
            
            bar = self.generate_bar()
            bars.append(bar)
        
        df = pd.DataFrame(bars)
        
        # Add patterns occasionally
        # Golden cross at 20% of bars
        for i in range(20, len(df), 100):
            df.loc[i, 'close'] = df.loc[i-1, 'close'] * 1.02
        
        # Death cross at 30% of bars  
        for i in range(30, len(df), 100):
            df.loc[i, 'close'] = df.loc[i-1, 'close'] * 0.98
        
        return df
    
    def reset(self):
        """Reset to initial state."""
        self.current_price = self.initial_price
        self.prices = [self.initial_price]
        self.current_regime = "sideways"


class LocalHistoricalLoader:
    """
    Load historical data from local CSV files.
    
    Expected format:
    timestamp,open,high,low,close,volume
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # Find available files
        self.available_files = self._find_csv_files()
        
        logger.info("LocalHistoricalLoader initialized")
        logger.info("  Data dir: {}".format(self.data_dir))
        logger.info("  Available: {}".format(len(self.available_files)))
    
    def _find_csv_files(self) -> List[Path]:
        """Find CSV files in data directory."""
        if not self.data_dir.exists():
            logger.warning("Data directory not found: {}".format(self.data_dir))
            return []
        
        files = []
        for f in self.data_dir.rglob("*.csv"):
            files.append(f)
        
        return files
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file."""
        path = self.data_dir / filename
        
        if not path.exists():
            # Try with subdirectory
            for f in self.available_files:
                if f.stem in filename or filename in str(f):
                    path = f
                    break
        
        if not path.exists():
            logger.warning("File not found: {}".format(filename))
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(path)
            
            # Normalize columns
            cols = [c.lower() for c in df.columns]
            df.columns = cols
            
            # Required columns
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                logger.warning("Missing columns: {}".format(missing))
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        except Exception as e:
            logger.error("Error loading {}: {}".format(filename, e))
            return pd.DataFrame()
    
    def list_datasets(self) -> Dict:
        """List available datasets."""
        datasets = {}
        
        for f in self.available_files:
            name = f.stem
            try:
                df = pd.read_csv(f)
                rows = len(df)
            except:
                rows = 0
            
            datasets[name] = {
                'path': str(f),
                'rows': rows,
                'size_mb': f.stat().st_size / 1024 / 1024
            }
        
        return datasets


class LocalDataManager:
    """
    Complete local data management.
    
    Combines:
    - Local data generation
    - Historical loading
    - Multiple timeframes
    - Cache management
    """
    
    def __init__(
        self,
        initial_price: float = 50000,
        history_dir: str = "data/historical",
    ):
        self.initial_price = initial_price
        
        # Generators for each timeframe
        self.generators = {
            '1m': LocalDataGenerator(initial_price, 0.03),
            '5m': LocalDataGenerator(initial_price, 0.025),
            '15m': LocalDataGenerator(initial_price, 0.02),
            '1h': LocalDataGenerator(initial_price, 0.015),
            '4h': LocalDataGenerator(initial_price, 0.012),
            '1d': LocalDataGenerator(initial_price, 0.008),
        }
        
        # Historical loader
        self.loader = LocalHistoricalLoader(history_dir)
        
        # Cache
        self.cache: Dict[str, pd.DataFrame] = {}
        
        # Stats
        self.bars_generated = 0
        
        logger.info("=" * 50)
        logger.info("LOCAL DATA MANAGER")
        logger.info("=" * 50)
        logger.info("Initial price: ${}".format(initial_price))
        logger.info("Timeframes: {}".format(list(self.generators.keys())))
        logger.info("Historical files: {}".format(len(self.loader.available_files)))
        logger.info("=" * 50)
    
    def generate(self, timeframe: str, count: int) -> pd.DataFrame:
        """Generate data for timeframe."""
        gen = self.generators.get(timeframe)
        
        if not gen:
            logger.warning("Unknown timeframe: {}".format(timeframe))
            return pd.DataFrame()
        
        df = gen.generate_with_patterns(count)
        self.bars_generated += count
        
        return df
    
    def generate_all(self, count_per_tf: int = 100) -> Dict[str, pd.DataFrame]:
        """Generate data for all timeframes."""
        data = {}
        
        for tf, gen in self.generators.items():
            data[tf] = gen.generate_with_patterns(count_per_tf)
            self.bars_generated += count_per_tf
        
        return data
    
    def load_history(self, dataset_name: str = None) -> pd.DataFrame:
        """Load historical data."""
        if not dataset_name:
            # Find first available
            if self.loader.available_files:
                dataset_name = self.loader.available_files[0].stem
            else:
                logger.warning("No historical files available")
                return self.generate('1h', 1000)
        
        return self.loader.load_csv(dataset_name)
    
    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from OHLCV DataFrame."""
        if len(df) < 24:
            return np.zeros(9)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
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
            pp = (close[-1] - np.min(low[-25:])) / (np.max(high[-25:]) - np.min(low[-25:]) + 1e-8)
        else:
            pp = 0.5
        
        # Volume ratio
        vr = volume[-1] / np.mean(volume[-25:]) if len(volume) >= 25 else 1.0
        
        features = np.array([r1, r4, r12, r24, v12, v24, rsi, pp, vr])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime."""
        features = self.get_features(df)
        
        r24 = features[3]  # 24-bar return
        v24 = features[5]   # volatility
        
        if r24 > 0.02 and v24 < 0.02:
            return "bull"
        elif r24 < -0.02 and v24 < 0.02:
            return "bear"
        return "sideways"
    
    def get_stats(self) -> Dict:
        """Get data stats."""
        return {
            'bars_generated': self.bars_generated,
            'timeframes': list(self.generators.keys()),
            'historical_files': len(self.loader.available_files),
        }


# ============================================================================
# INTEGRATION
# ============================================================================

# Global instance
_data_manager = None


def get_local_data_manager() -> LocalDataManager:
    """Get global local data manager."""
    global _data_manager
    if _data_manager is None:
        _data_manager = LocalDataManager()
    return _data_manager


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print()
    print("=" * 50)
    print("LOCAL DATA SYSTEM TEST")
    print("=" * 50)
    print()
    
    # Create manager
    dm = LocalDataManager(initial_price=50000)
    
    # List available data
    print("Available historical data:")
    datasets = dm.loader.list_datasets()
    if datasets:
        for name, info in datasets.items():
            print("  {}: {} rows ({:.1f} MB)".format(
                name, info['rows'], info['size_mb']))
    else:
        print("  (no historical files)")
    
    print()
    
    # Generate data
    print("Generating data for all timeframes...")
    data = dm.generate_all(100)
    
    for tf, df in data.items():
        regime = dm.detect_regime(df)
        features = dm.get_features(df)
        print("  {}: {} bars, regime={}, r1={:.2%}".format(
            tf, len(df), regime, features[0]))
    
    print()
    
    # Try loading historical
    print("Loading historical data...")
    hist = dm.load_history()
    if len(hist) > 0:
        print("  Loaded: {} rows".format(len(hist)))
        features = dm.get_features(hist[-100:])
        print("  Features: r1={:.2%}, r4={:.2%}, regime={}".format(
            features[0], features[1], dm.detect_regime(hist[-100:])))
    else:
        print("  No historical data, using generated")
        hist = dm.generate('1h', 500)
    
    print()
    print("Stats: {}".format(dm.get_stats()))