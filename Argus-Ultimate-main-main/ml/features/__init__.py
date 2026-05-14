'''
Argus Trading Bot - Feature Engineering
S+ Tier Feature Engineering Pipeline
'''

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    include_technical: bool = True
    include_volume: bool = True
    include_momentum: bool = True
    include_volatility: bool = True


class FeatureEngineer:
    """Feature engineering for ML models."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def compute_features(self, df):
        """Compute features from OHLCV data."""
        from .feature_library import FeatureLibrary
        lib = FeatureLibrary()
        return lib.compute_all_features(df)


__all__ = ["FeatureEngineer", "FeatureConfig"]
