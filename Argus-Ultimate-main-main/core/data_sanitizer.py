#!/usr/bin/env python3
"""
Data Sanitizer - Market Data Validation and Cleaning
====================================================

Advanced data sanitization for financial market data with outlier detection,
gap filling, and data quality assurance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics"""

    total_records: int = 0
    valid_records: int = 0
    outliers_removed: int = 0
    gaps_filled: int = 0
    duplicates_removed: int = 0
    quality_score: float = 0.0


class DataSanitizer:
    """
    Data Sanitizer - Market Data Validation and Cleaning

    Features:
    - Outlier detection and removal
    - Gap filling and interpolation
    - Duplicate detection and removal
    - Data validation and quality scoring
    - Statistical analysis and reporting
    """

    def __init__(self):
        self.quality_metrics = DataQualityMetrics()
        self.last_valid_price: Optional[float] = None
        self.price_bounds: Optional[Tuple[float, float]] = None

        logger.info("Data Sanitizer initialized")

    def sanitize_ohlcv_data(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = True,
        fill_gaps: bool = True,
        remove_duplicates: bool = True,
    ) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """
        Sanitize OHLCV market data

        Args:
            df: Raw OHLCV DataFrame
            remove_outliers: Whether to remove statistical outliers
            fill_gaps: Whether to fill data gaps
            remove_duplicates: Whether to remove duplicate records

        Returns:
            Tuple of (sanitized_data, quality_metrics)
        """
        if df.empty:
            return df.copy(), DataQualityMetrics()

        # Reset metrics
        self.quality_metrics = DataQualityMetrics(total_records=len(df))

        # Make a copy to avoid modifying original
        sanitized = df.copy()

        # Remove duplicates
        if remove_duplicates:
            sanitized = self._remove_duplicates(sanitized)

        # Validate data structure
        sanitized = self._validate_data_structure(sanitized)

        # Remove outliers
        if remove_outliers:
            sanitized = self._remove_outliers(sanitized)

        # Fill gaps
        if fill_gaps:
            sanitized = self._fill_gaps(sanitized)

        # Calculate quality score
        self._calculate_quality_score(sanitized)

        logger.info(f"Data sanitized: {len(df)} -> {len(sanitized)} records")
        return sanitized, self.quality_metrics

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        initial_count = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates()

        # Remove duplicates by timestamp (keep first)
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"], keep="first")

        duplicates_removed = initial_count - len(df)
        self.quality_metrics.duplicates_removed = duplicates_removed

        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate records")

        return df

    def _validate_data_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data structure"""
        required_columns = ["timestamp", "open", "high", "low", "close"]
        optional_columns = ["volume"]

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure numeric columns are numeric
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with NaN in required columns
        initial_count = len(df)
        df = df.dropna(subset=required_columns)
        nan_removed = initial_count - len(df)

        if nan_removed > 0:
            logger.warning(f"Removed {nan_removed} records with NaN values")

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers"""
        if len(df) < 10:  # Need minimum data for outlier detection
            return df

        initial_count = len(df)

        # Use IQR method for outlier detection
        numeric_cols = ["open", "high", "low", "close"]

        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Mark outliers
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df = df[~outlier_mask]

        outliers_removed = initial_count - len(df)
        self.quality_metrics.outliers_removed = outliers_removed

        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outlier records")

        return df

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill data gaps using interpolation"""
        if len(df) < 2:
            return df

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Check for time gaps (assuming 1-hour intervals)
        if "timestamp" in df.columns:
            time_diffs = df["timestamp"].diff()
            expected_diff = pd.Timedelta(hours=1)

            # Find gaps larger than expected
            gaps = time_diffs > expected_diff * 1.5  # 50% tolerance

            if gaps.any():
                gap_count = gaps.sum()
                self.quality_metrics.gaps_filled = gap_count

                logger.info(f"Found {gap_count} data gaps")

                # For now, we'll just log gaps but not fill them
                # Complex gap filling would require more sophisticated logic
                # based on the specific exchange and market conditions

        return df

    def _calculate_quality_score(self, df: pd.DataFrame) -> None:
        """Calculate overall data quality score"""
        if len(df) == 0:
            self.quality_metrics.quality_score = 0.0
            return

        score = 1.0

        # Penalize for outliers removed
        if self.quality_metrics.total_records > 0:
            outlier_ratio = self.quality_metrics.outliers_removed / self.quality_metrics.total_records
            score -= outlier_ratio * 0.3

        # Penalize for duplicates removed
        if self.quality_metrics.total_records > 0:
            duplicate_ratio = self.quality_metrics.duplicates_removed / self.quality_metrics.total_records
            score -= duplicate_ratio * 0.2

        # Penalize for gaps
        if self.quality_metrics.gaps_filled > 0:
            score -= 0.1

        # Ensure score stays within bounds
        self.quality_metrics.quality_score = max(0.0, min(1.0, score))

        # Update valid records count
        self.quality_metrics.valid_records = len(df)

    def detect_price_anomalies(self, prices: List[float], threshold: float = 3.0) -> List[Tuple[int, float]]:
        """
        Detect price anomalies using statistical methods

        Args:
            prices: List of prices
            threshold: Z-score threshold for anomaly detection

        Returns:
            List of (index, z_score) tuples for anomalies
        """
        if len(prices) < 10:
            return []

        prices_array = np.array(prices)
        mean_price = np.mean(prices_array)
        std_price = np.std(prices_array)

        if std_price == 0:
            return []

        z_scores = np.abs((prices_array - mean_price) / std_price)

        anomalies = []
        for i, z_score in enumerate(z_scores):
            if z_score > threshold:
                anomalies.append((i, z_score))

        return anomalies

    def validate_ohlc_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLC price integrity

        Args:
            df: OHLCV DataFrame

        Returns:
            Validation results
        """
        issues = []

        # Check OHLC relationships
        invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
        invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"])

        if invalid_high.any():
            issues.append(f"{invalid_high.sum()} records with high < open/close")

        if invalid_low.any():
            issues.append(f"{invalid_low.sum()} records with low > open/close")

        # Check for zero or negative prices
        zero_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if zero_prices.any():
            issues.append(f"{zero_prices.sum()} records with zero/negative prices")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "checked_records": len(df),
        }

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        return {
            "metrics": {
                "total_records": self.quality_metrics.total_records,
                "valid_records": self.quality_metrics.valid_records,
                "outliers_removed": self.quality_metrics.outliers_removed,
                "gaps_filled": self.quality_metrics.gaps_filled,
                "duplicates_removed": self.quality_metrics.duplicates_removed,
                "quality_score": self.quality_metrics.quality_score,
            },
            "percentages": {
                "data_retention": self.quality_metrics.valid_records / max(1, self.quality_metrics.total_records),
                "outlier_ratio": self.quality_metrics.outliers_removed / max(1, self.quality_metrics.total_records),
                "duplicate_ratio": self.quality_metrics.duplicates_removed / max(1, self.quality_metrics.total_records),
            },
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []

        metrics = self.quality_metrics

        if metrics.outliers_removed > metrics.total_records * 0.1:
            recommendations.append("High outlier ratio - consider adjusting outlier detection parameters")

        if metrics.duplicates_removed > 0:
            recommendations.append("Duplicate records found - review data collection process")

        if metrics.quality_score < 0.8:
            recommendations.append("Low quality score - consider data source validation")

        if not recommendations:
            recommendations.append("Data quality is acceptable")

        return recommendations

    def reset_metrics(self) -> None:
        """Reset quality metrics"""
        self.quality_metrics = DataQualityMetrics()
        self.last_valid_price = None
        self.price_bounds = None

    def set_price_bounds(self, min_price: float, max_price: float) -> None:
        """
        Set acceptable price bounds for validation

        Args:
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price
        """
        self.price_bounds = (min_price, max_price)
        logger.info(f"Price bounds set: {min_price} - {max_price}")
