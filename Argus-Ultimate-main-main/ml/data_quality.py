"""
ML Training Data Quality Gate.

Validates training data quality before model training.
Checks for missing values, outliers, data drift, and ensures clean data.

Usage:
    from ml.data_quality import DataQualityPipeline, DataQualityConfig
    
    pipeline = DataQualityPipeline(DataQualityConfig())
    
    # Validate before training
    passed, report = pipeline.validate(train_df)
    if not passed:
        print(f"Data quality issues: {report['issues']}")
    
    # Clean data
    clean_df = pipeline.clean(train_df)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Outlier detection methods."""
    IQR = "iqr"           # Interquartile range
    ZSCORE = "zscore"     # Z-score threshold
    ISOLATION_FOREST = "isolation_forest"
    PERCENTILE = "percentile"


class MissingMethod(Enum):
    """Missing data handling methods."""
    DROP = "drop"         # Drop rows with missing
    FILL_MEAN = "mean"    # Fill with mean
    FILL_MEDIAN = "median"  # Fill with median
    FILL_FORWARD = "forward"  # Forward fill (time series)
    FILL_ZERO = "zero"    # Fill with zero


@dataclass
class DataQualityConfig:
    """Configuration for data quality checks."""
    
    # Basic checks
    min_samples: int = 100              # Minimum rows required
    max_missing_pct: float = 0.1        # Max 10% missing values
    min_columns: int = 2                # Minimum columns required
    
    # Outlier detection
    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_threshold: float = 3.0      # Z-score or IQR multiplier
    max_outlier_pct: float = 0.05       # Max 5% outliers
    
    # Data drift detection
    drift_threshold: float = 0.3        # KS test p-value threshold
    drift_window: int = 100             # Window for drift calculation
    
    # Feature quality
    min_variance: float = 1e-10         # Min feature variance (remove constants)
    max_correlation: float = 0.99       # Remove highly correlated features
    
    # Time series specific
    check_timestamps: bool = True       # Check for timestamp gaps
    max_timestamp_gap: float = 3600.0   # Max gap in seconds
    
    # Cleaning options
    missing_method: MissingMethod = MissingMethod.FILL_FORWARD
    clip_outliers: bool = True          # Clip outliers to bounds
    normalize: bool = False             # Apply normalization
    
    # Gate behavior
    strict_mode: bool = True            # Fail on any issue
    quality_threshold: float = 0.7      # Min quality score to pass


@dataclass
class DataQualityReport:
    """Report from data quality validation."""
    
    passed: bool
    quality_score: float
    n_samples: int
    n_features: int
    
    # Detailed metrics
    missing_pct: float
    outlier_pct: float
    drift_score: float
    variance_issues: List[str]
    
    # Issues found
    issues: List[str]
    warnings: List[str]
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'quality_score': self.quality_score,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'missing_pct': self.missing_pct,
            'outlier_pct': self.outlier_pct,
            'drift_score': self.drift_score,
            'issues': self.issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
        }


class DataQualityPipeline:
    """
    ML Training Data Quality Pipeline.
    
    Validates and cleans training data before model training.
    
    Args:
        config: DataQualityConfig with validation parameters
    
    Example:
        >>> config = DataQualityConfig(
        ...     min_samples=1000,
        ...     max_missing_pct=0.05,
        ...     outlier_method=OutlierMethod.IQR,
        ... )
        >>> pipeline = DataQualityPipeline(config)
        >>> 
        >>> # Validate
        >>> passed, report = pipeline.validate(train_df)
        >>> if not passed:
        ...     print(f"Issues: {report.issues}")
        ...     
        >>> # Clean
        >>> clean_df = pipeline.clean(train_df)
    """
    
    def __init__(self, config: Optional[DataQualityConfig] = None):
        self.config = config or DataQualityConfig()
        self._reference_stats: Optional[Dict] = None
        
        logger.info(f"DataQualityPipeline initialized: threshold={self.config.quality_threshold}, "
                    f"outlier_method={self.config.outlier_method.value}")
    
    def validate(
        self, 
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, DataQualityReport]:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            reference_df: Optional reference for drift detection
            
        Returns:
            Tuple of (passed, report)
        """
        issues = []
        warnings = []
        recommendations = []
        
        n_samples = len(df)
        n_features = len(df.columns)
        
        # 1. Check minimum samples
        if n_samples < self.config.min_samples:
            issues.append(f"Insufficient samples: {n_samples} < {self.config.min_samples}")
        
        # 2. Check minimum columns
        if n_features < self.config.min_columns:
            issues.append(f"Insufficient features: {n_features} < {self.config.min_columns}")
        
        # 3. Check missing data
        missing_pct = df.isnull().sum().sum() / (n_samples * n_features + 1e-10)
        if missing_pct > self.config.max_missing_pct:
            issues.append(f"High missing data: {missing_pct:.1%} > {self.config.max_missing_pct:.1%}")
            recommendations.append("Consider imputation or removing columns with high missing rate")
        elif missing_pct > self.config.max_missing_pct * 0.5:
            warnings.append(f"Moderate missing data: {missing_pct:.1%}")
        
        # 4. Check outliers
        outlier_count, outlier_cols = self._count_outliers(df)
        outlier_pct = outlier_count / (n_samples * n_features + 1e-10)
        if outlier_pct > self.config.max_outlier_pct:
            issues.append(f"High outlier rate: {outlier_pct:.1%} > {self.config.max_outlier_pct:.1%}")
            recommendations.append(f"Review outliers in: {outlier_cols[:5]}")
        
        # 5. Check constant features
        constant_cols = self._find_constant_features(df)
        if constant_cols:
            warnings.append(f"Constant features (no variance): {constant_cols}")
            recommendations.append("Remove constant features before training")
        
        # 6. Check high correlation
        high_corr_pairs = self._find_high_correlation(df)
        if high_corr_pairs:
            warnings.append(f"Highly correlated feature pairs: {len(high_corr_pairs)}")
        
        # 7. Check data drift (if reference provided)
        drift_score = 0.0
        if reference_df is not None:
            drift_score = self._check_drift(df, reference_df)
            if drift_score > self.config.drift_threshold:
                issues.append(f"Data drift detected: {drift_score:.2f} > {self.config.drift_threshold}")
                recommendations.append("Retrain model or investigate data distribution shift")
        
        # 8. Check for NaN/Inf
        nan_count = df.isnull().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if nan_count > 0:
            issues.append(f"NaN values found: {nan_count}")
        if inf_count > 0:
            issues.append(f"Inf values found: {inf_count}")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            missing_pct=missing_pct,
            outlier_pct=outlier_pct,
            drift_score=drift_score,
            n_issues=len(issues),
            n_warnings=len(warnings),
        )
        
        # Determine pass/fail
        if self.config.strict_mode:
            passed = len(issues) == 0 and quality_score >= self.config.quality_threshold
        else:
            passed = quality_score >= self.config.quality_threshold
        
        report = DataQualityReport(
            passed=passed,
            quality_score=quality_score,
            n_samples=n_samples,
            n_features=n_features,
            missing_pct=missing_pct,
            outlier_pct=outlier_pct,
            drift_score=drift_score,
            variance_issues=constant_cols,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
        )
        
        if passed:
            logger.info(f"Data quality PASSED: score={quality_score:.2f}")
        else:
            logger.warning(f"Data quality FAILED: score={quality_score:.2f}, issues={len(issues)}")
        
        return passed, report
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # 1. Handle NaN/Inf
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Handle missing values
        if self.config.missing_method == MissingMethod.DROP:
            df = df.dropna()
        elif self.config.missing_method == MissingMethod.FILL_MEAN:
            df = df.fillna(df.mean(numeric_only=True))
        elif self.config.missing_method == MissingMethod.FILL_MEDIAN:
            df = df.fillna(df.median(numeric_only=True))
        elif self.config.missing_method == MissingMethod.FILL_FORWARD:
            df = df.ffill().bfill()  # Forward fill then backward fill remaining
        elif self.config.missing_method == MissingMethod.FILL_ZERO:
            df = df.fillna(0)
        
        # 3. Remove constant features
        constant_cols = self._find_constant_features(df)
        if constant_cols:
            df = df.drop(columns=constant_cols)
            logger.info(f"Removed constant columns: {constant_cols}")
        
        # 4. Handle outliers
        if self.config.clip_outliers:
            df = self._clip_outliers(df)
        
        return df
    
    def _count_outliers(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """Count outliers using configured method."""
        outlier_count = 0
        outlier_cols = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            if self.config.outlier_method == OutlierMethod.IQR:
                q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_outliers = ((col_data < lower) | (col_data > upper)).sum()
            
            elif self.config.outlier_method == OutlierMethod.ZSCORE:
                z_scores = np.abs((col_data - col_data.mean()) / (col_data.std() + 1e-10))
                n_outliers = (z_scores > self.config.outlier_threshold).sum()
            
            elif self.config.outlier_method == OutlierMethod.PERCENTILE:
                lower = col_data.quantile(0.01)
                upper = col_data.quantile(0.99)
                n_outliers = ((col_data < lower) | (col_data > upper)).sum()
            
            else:
                n_outliers = 0
            
            if n_outliers > 0:
                outlier_count += n_outliers
                outlier_cols.append(col)
        
        return outlier_count, outlier_cols
    
    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers to bounds."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            
            df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    def _find_constant_features(self, df: pd.DataFrame) -> List[str]:
        """Find features with zero or near-zero variance."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        constant_cols = []
        
        for col in numeric_cols:
            if df[col].var() < self.config.min_variance:
                constant_cols.append(col)
        
        return constant_cols
    
    def _find_high_correlation(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find highly correlated feature pairs."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        high_corr_pairs = []
        
        if len(numeric_cols) < 2:
            return high_corr_pairs
        
        corr_matrix = df[numeric_cols].corr().abs()
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if corr_matrix.iloc[i, j] > self.config.max_correlation:
                    high_corr_pairs.append((numeric_cols[i], numeric_cols[j]))
        
        return high_corr_pairs
    
    def _check_drift(self, df: pd.DataFrame, reference_df: pd.DataFrame) -> float:
        """Check data drift using KS test."""
        try:
            from scipy import stats
            
            drift_scores = []
            common_cols = set(df.select_dtypes(include=[np.number]).columns) & \
                         set(reference_df.select_dtypes(include=[np.number]).columns)
            
            for col in common_cols:
                current = df[col].dropna().values
                reference = reference_df[col].dropna().values
                
                if len(current) > 10 and len(reference) > 10:
                    # Kolmogorov-Smirnov test
                    ks_stat, _ = stats.ks_2samp(current, reference)
                    drift_scores.append(ks_stat)
            
            return np.mean(drift_scores) if drift_scores else 0.0
            
        except ImportError:
            logger.warning("scipy not available for drift detection")
            return 0.0
    
    def _calculate_quality_score(
        self,
        missing_pct: float,
        outlier_pct: float,
        drift_score: float,
        n_issues: int,
        n_warnings: int,
    ) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0
        
        # Penalize missing data
        score -= min(0.3, missing_pct * 2)
        
        # Penalize outliers
        score -= min(0.2, outlier_pct * 3)
        
        # Penalize drift
        score -= min(0.2, drift_score)
        
        # Penalize issues
        score -= min(0.2, n_issues * 0.1)
        
        # Penalize warnings
        score -= min(0.1, n_warnings * 0.02)
        
        return max(0.0, min(1.0, score))
    
    def set_reference(self, df: pd.DataFrame) -> None:
        """Set reference statistics for drift detection."""
        self._reference_stats = {
            col: {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
            }
            for col in df.select_dtypes(include=[np.number]).columns
        }
        logger.info("Reference statistics set for drift detection")


class QuickDataValidator:
    """
    Quick validation for ML training data.
    
    Simpler version for fast checks.
    
    Example:
        >>> validator = QuickDataValidator()
        >>> if validator.is_valid(df):
        ...     train_model(df)
    """
    
    @staticmethod
    def is_valid(
        df: pd.DataFrame,
        min_samples: int = 100,
        max_missing_pct: float = 0.1,
    ) -> bool:
        """Quick validity check."""
        if len(df) < min_samples:
            return False
        
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns) + 1e-10)
        if missing_pct > max_missing_pct:
            return False
        
        if df.select_dtypes(include=[np.number]).shape[1] == 0:
            return False
        
        return True
    
    @staticmethod
    def get_quick_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Get quick data statistics."""
        return {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'n_numeric': df.select_dtypes(include=[np.number]).shape[1],
            'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns) + 1e-10),
            'has_inf': np.isinf(df.select_dtypes(include=[np.number])).any().any(),
        }
