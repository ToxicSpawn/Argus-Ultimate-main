"""
Feature Store for ARGUS.

Computes, caches, and serves ML features derived from OHLCV data.
Supports feature drift detection via Kolmogorov-Smirnov tests.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional scipy for KS test
try:
    from scipy.stats import ks_2samp  # type: ignore
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.debug("scipy not available — drift detection will use fallback")


@dataclass
class DriftReport:
    """Result of a feature drift check."""
    drifted_features: List[str] = field(default_factory=list)
    ks_statistics: Dict[str, float] = field(default_factory=dict)
    is_significant: bool = False


class FeatureStore:
    """
    Compute, cache, and retrieve ML features from OHLCV data.

    Features are cached to Parquet files keyed by (symbol, feature_set).
    Staleness threshold: 1 hour.
    """

    STALE_SECONDS = 3600  # 1 hour

    def __init__(self, cache_dir: str = "data/features") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _cache_path(self, symbol: str, feature_set: str) -> Path:
        safe = symbol.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe}_{feature_set}.parquet"

    # ------------------------------------------------------------------ #
    # Feature computation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_default_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a standard set of features from an OHLCV DataFrame.

        Features: returns, volatility (20-period rolling std of returns),
        RSI (14-period), MACD (12/26/9), Bollinger band width, volume_ratio,
        spread (high-low normalised).
        """
        out = pd.DataFrame(index=df.index)

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # Returns
        out["returns"] = close.pct_change()

        # Volatility (20-period rolling std of returns)
        out["volatility"] = out["returns"].rolling(20).std()

        # RSI (14-period)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        out["macd"] = ema12 - ema26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_histogram"] = out["macd"] - out["macd_signal"]

        # Bollinger Bands width
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        out["bollinger_width"] = (bb_upper - bb_lower) / sma20.replace(0, np.nan)

        # Volume ratio (current / 20-period SMA)
        vol_sma = volume.rolling(20).mean()
        out["volume_ratio"] = volume / vol_sma.replace(0, np.nan)

        # Spread (normalised)
        out["spread"] = (high - low) / close.replace(0, np.nan)

        return out

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute_and_cache(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        feature_set: str = "default",
    ) -> pd.DataFrame:
        """
        Compute features from *ohlcv_df*, cache to disk, and return them.
        """
        if feature_set == "default":
            features = self._compute_default_features(ohlcv_df)
        else:
            raise ValueError(f"Unknown feature_set: {feature_set!r}")

        path = self._cache_path(symbol, feature_set)
        features.to_parquet(path, index=False, engine="pyarrow")
        logger.info(
            "FeatureStore: cached %d rows of '%s' features for %s → %s",
            len(features), feature_set, symbol, path,
        )
        return features

    def get_features(
        self,
        symbol: str,
        feature_set: str = "default",
    ) -> Optional[pd.DataFrame]:
        """
        Load cached features. Returns None if cache is missing or stale (>1 h).
        """
        path = self._cache_path(symbol, feature_set)
        if not path.exists():
            return None

        age = time.time() - path.stat().st_mtime
        if age > self.STALE_SECONDS:
            logger.info(
                "FeatureStore: cache stale (%.0fs old) for %s/%s",
                age, symbol, feature_set,
            )
            return None

        df = pd.read_parquet(path, engine="pyarrow")
        logger.debug(
            "FeatureStore: loaded %d cached features for %s/%s", len(df), symbol, feature_set,
        )
        return df

    # ------------------------------------------------------------------ #
    # Drift detection
    # ------------------------------------------------------------------ #

    def check_drift(
        self,
        current_features: pd.DataFrame,
        reference_features: pd.DataFrame,
        p_threshold: float = 0.05,
    ) -> DriftReport:
        """
        Compare distributions of *current_features* vs *reference_features*
        using a two-sample Kolmogorov-Smirnov test per feature column.

        Flags features whose p-value < *p_threshold*.
        """
        report = DriftReport()

        common_cols = [
            c for c in current_features.columns
            if c in reference_features.columns
        ]

        for col in common_cols:
            cur = current_features[col].dropna().astype(float)
            ref = reference_features[col].dropna().astype(float)

            if len(cur) < 5 or len(ref) < 5:
                continue

            if _SCIPY_AVAILABLE:
                stat, pval = ks_2samp(cur, ref)
            else:
                # Simple fallback: compare means/stds
                stat = abs(cur.mean() - ref.mean()) / max(ref.std(), 1e-9)
                pval = 1.0 if stat < 2.0 else 0.01  # rough heuristic

            report.ks_statistics[col] = stat

            if pval < p_threshold:
                report.drifted_features.append(col)

        report.is_significant = len(report.drifted_features) > 0
        return report
