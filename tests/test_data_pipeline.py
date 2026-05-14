"""
Tests for the ARGUS data pipeline: quality validation, historical ingestion,
feature store, and model lineage.

Covers data/quality.py, data/historical_ingester.py, ml/feature_store.py,
and lineage extensions in ml/model_manager.py.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 100, freq: str = "1h", start: str = "2025-01-01") -> pd.DataFrame:
    """Generate a clean synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    opn = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.uniform(100, 10000, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# =========================================================================
# 1. DataQualityValidator tests
# =========================================================================


class TestDataQualityValidator:

    def setup_method(self):
        import pytest; pytest.importorskip("data.quality")
        from data.quality import DataQualityValidator
        self.validator = DataQualityValidator()

    # --- validate_ohlcv ---

    def test_clean_data_passes(self):
        df = _make_ohlcv()
        report = self.validator.validate_ohlcv(df)
        assert report.passed is True
        assert report.rows_checked == 100
        assert report.rows_failed == 0
        assert report.issues == []

    def test_empty_dataframe_fails(self):
        df = pd.DataFrame()
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert "empty" in report.issues[0].lower()

    def test_nan_detected(self):
        df = _make_ohlcv(50)
        df.loc[5, "close"] = np.nan
        df.loc[10, "open"] = np.nan
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert report.rows_failed >= 2
        assert any("NaN" in i for i in report.issues)

    def test_inf_detected(self):
        df = _make_ohlcv(50)
        df.loc[3, "high"] = np.inf
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert any("Inf" in i for i in report.issues)

    def test_high_less_than_low(self):
        df = _make_ohlcv(30)
        df.loc[2, "high"] = df.loc[2, "low"] - 1
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert any("high < low" in i for i in report.issues)

    def test_close_outside_range(self):
        df = _make_ohlcv(30)
        df.loc[5, "close"] = df.loc[5, "high"] + 10
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert any("close outside" in i for i in report.issues)

    def test_negative_volume(self):
        df = _make_ohlcv(30)
        df.loc[7, "volume"] = -100
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert any("negative volume" in i for i in report.issues)

    def test_duplicate_timestamps(self):
        df = _make_ohlcv(30)
        df.loc[5, "timestamp"] = df.loc[4, "timestamp"]
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert any("duplicate" in i.lower() for i in report.issues)

    def test_non_monotonic_timestamps(self):
        df = _make_ohlcv(30)
        # Swap two timestamps
        df.loc[10, "timestamp"] = df.loc[5, "timestamp"] - pd.Timedelta("1h")
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert any("monotonic" in i.lower() for i in report.issues)

    def test_timestamp_gaps(self):
        df = _make_ohlcv(30, freq="1h")
        # Create a 10-hour gap (5x median interval of 1h)
        df.loc[15, "timestamp"] = df.loc[14, "timestamp"] + pd.Timedelta("10h")
        # Fix remaining timestamps to be after the gap
        for i in range(16, 30):
            df.loc[i, "timestamp"] = df.loc[i - 1, "timestamp"] + pd.Timedelta("1h")
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert any("gap" in i.lower() for i in report.issues)

    def test_missing_column(self):
        df = _make_ohlcv(10)
        df = df.drop(columns=["volume"])
        report = self.validator.validate_ohlcv(df)
        assert report.passed is False
        assert any("missing column" in i for i in report.issues)

    # --- clean_ohlcv ---

    def test_clean_fills_nan(self):
        df = _make_ohlcv(20)
        df.loc[5, "close"] = np.nan
        df.loc[10, "open"] = np.nan
        cleaned = self.validator.clean_ohlcv(df)
        assert cleaned["close"].isna().sum() == 0
        assert cleaned["open"].isna().sum() == 0

    def test_clean_clips_negative_volume(self):
        df = _make_ohlcv(20)
        df.loc[3, "volume"] = -500
        df.loc[7, "volume"] = -1
        cleaned = self.validator.clean_ohlcv(df)
        assert (cleaned["volume"] >= 0).all()

    def test_clean_removes_duplicates(self):
        df = _make_ohlcv(20)
        df = pd.concat([df, df.iloc[[5, 10]]], ignore_index=True)
        assert len(df) == 22
        cleaned = self.validator.clean_ohlcv(df)
        assert len(cleaned) == 20

    def test_clean_preserves_valid_data(self):
        df = _make_ohlcv(50)
        cleaned = self.validator.clean_ohlcv(df)
        assert len(cleaned) == 50
        pd.testing.assert_frame_equal(cleaned, df)

    # --- detect_outliers ---

    def test_outliers_zscore(self):
        s = pd.Series(np.concatenate([np.zeros(100), [100.0]]))
        outliers = self.validator.detect_outliers(s, method="zscore", threshold=3.0)
        assert 100 in outliers

    def test_outliers_iqr(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        data = np.append(data, 1000.0)
        s = pd.Series(data)
        outliers = self.validator.detect_outliers(s, method="iqr", threshold=1.5)
        assert 100 in outliers

    def test_outliers_empty_series(self):
        s = pd.Series(dtype=float)
        outliers = self.validator.detect_outliers(s)
        assert outliers == []

    def test_outliers_invalid_method(self):
        s = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown outlier method"):
            self.validator.detect_outliers(s, method="invalid")

    def test_outliers_constant_series(self):
        s = pd.Series([5.0] * 50)
        outliers = self.validator.detect_outliers(s, method="zscore")
        assert outliers == []

    def test_datetime_index_timestamps(self):
        """Validator should handle DatetimeIndex as timestamps."""
        df = _make_ohlcv(30)
        df = df.set_index("timestamp")
        report = self.validator.validate_ohlcv(df)
        assert report.passed is True


# =========================================================================
# 2. HistoricalDataIngester tests
# =========================================================================


class TestHistoricalDataIngester:

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def _make_ingester(self):
        from data.historical_ingester import HistoricalDataIngester
        return HistoricalDataIngester(exchange_id="kraken", data_dir=self._tmpdir)

    def test_save_and_load(self):
        ingester = self._make_ingester()
        df = _make_ohlcv(50)
        ingester.save("BTC/USD", "1h", df)
        loaded = ingester.load("BTC/USD", "1h")
        assert loaded is not None
        assert len(loaded) == 50

    def test_load_nonexistent(self):
        ingester = self._make_ingester()
        result = ingester.load("ETH/USD", "1d")
        assert result is None

    def test_save_deduplicates(self):
        ingester = self._make_ingester()
        df = _make_ohlcv(30)
        # Duplicate a few rows by timestamp
        df2 = pd.concat([df, df.iloc[:5]], ignore_index=True)
        ingester.save("BTC/USD", "1h", df2)
        loaded = ingester.load("BTC/USD", "1h")
        assert len(loaded) == 30

    def test_save_cleans_data(self):
        ingester = self._make_ingester()
        df = _make_ohlcv(20)
        df.loc[3, "volume"] = -50
        df.loc[7, "close"] = np.nan
        ingester.save("BTC/USD", "1h", df)
        loaded = ingester.load("BTC/USD", "1h")
        assert (loaded["volume"] >= 0).all()
        assert loaded["close"].isna().sum() == 0

    def test_parquet_path_sanitises_symbol(self):
        ingester = self._make_ingester()
        path = ingester._parquet_path("BTC/USD", "1h")
        assert "BTC_USD_1h.parquet" in str(path)

    def test_save_creates_directory(self):
        subdir = os.path.join(self._tmpdir, "nested", "deep")
        from data.historical_ingester import HistoricalDataIngester
        ingester = HistoricalDataIngester(data_dir=subdir)
        assert Path(subdir).exists()


# =========================================================================
# 3. FeatureStore tests
# =========================================================================


class TestFeatureStore:

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def _make_store(self):
        from ml.feature_store import FeatureStore
        return FeatureStore(cache_dir=self._tmpdir)

    def test_compute_and_cache(self):
        store = self._make_store()
        df = _make_ohlcv(100)
        features = store.compute_and_cache("BTC/USD", df)
        assert "returns" in features.columns
        assert "volatility" in features.columns
        assert "rsi" in features.columns
        assert "macd" in features.columns
        assert "bollinger_width" in features.columns
        assert "volume_ratio" in features.columns
        assert "spread" in features.columns

    def test_get_features_from_cache(self):
        store = self._make_store()
        df = _make_ohlcv(100)
        store.compute_and_cache("BTC/USD", df)
        cached = store.get_features("BTC/USD")
        assert cached is not None
        assert len(cached) == 100

    def test_get_features_missing(self):
        store = self._make_store()
        result = store.get_features("ETH/USD")
        assert result is None

    def test_get_features_stale(self):
        store = self._make_store()
        df = _make_ohlcv(50)
        store.compute_and_cache("BTC/USD", df)
        # Manually backdate the file
        path = store._cache_path("BTC/USD", "default")
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(str(path), (old_time, old_time))
        result = store.get_features("BTC/USD")
        assert result is None

    def test_feature_values_reasonable(self):
        store = self._make_store()
        df = _make_ohlcv(200)
        features = store.compute_and_cache("BTC/USD", df)
        # RSI should be between 0 and 100 (after warmup)
        rsi_valid = features["rsi"].dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_unknown_feature_set_raises(self):
        store = self._make_store()
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match="Unknown feature_set"):
            store.compute_and_cache("BTC/USD", df, feature_set="unknown_set")

    def test_drift_no_drift(self):
        store = self._make_store()
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"feat_a": rng.standard_normal(500), "feat_b": rng.uniform(0, 1, 500)})
        cur = pd.DataFrame({"feat_a": rng.standard_normal(500), "feat_b": rng.uniform(0, 1, 500)})
        report = store.check_drift(cur, ref)
        # Same distribution — should not flag significant drift
        assert report.is_significant is False
        assert len(report.drifted_features) == 0

    def test_drift_detects_shift(self):
        store = self._make_store()
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"feat_a": rng.standard_normal(500)})
        cur = pd.DataFrame({"feat_a": rng.standard_normal(500) + 5.0})  # big shift
        report = store.check_drift(cur, ref)
        assert report.is_significant is True
        assert "feat_a" in report.drifted_features
        assert "feat_a" in report.ks_statistics

    def test_drift_report_fields(self):
        from ml.feature_store import DriftReport
        report = DriftReport()
        assert report.drifted_features == []
        assert report.ks_statistics == {}
        assert report.is_significant is False


# =========================================================================
# 4. ModelManager lineage tests
# =========================================================================


class TestModelManagerLineage:

    def _make_manager(self):
        from ml.model_manager import ModelManager
        return ModelManager()

    def test_record_lineage(self):
        mgr = self._make_manager()
        mgr.record_training_lineage(
            "regime_classifier",
            data_hash="abc123def456",
            rows=5000,
            feature_set="default",
        )
        meta = mgr.registry["regime_classifier"]
        assert meta.training_data_hash == "abc123def456"
        assert meta.training_rows == 5000
        assert meta.feature_set == "default"
        assert meta.training_date != ""

    def test_get_lineage(self):
        mgr = self._make_manager()
        mgr.record_training_lineage(
            "alpha_model",
            data_hash="hash_xyz",
            rows=2000,
            feature_set="advanced",
        )
        lineage = mgr.get_lineage("alpha_model")
        assert lineage["training_data_hash"] == "hash_xyz"
        assert lineage["training_rows"] == 2000
        assert lineage["feature_set"] == "advanced"
        assert lineage["training_date"] != ""

    def test_lineage_unknown_model_raises(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="Unknown model"):
            mgr.record_training_lineage("nonexistent", "h", 1, "f")

    def test_get_lineage_unknown_model_raises(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="Unknown model"):
            mgr.get_lineage("nonexistent")

    def test_lineage_defaults_empty(self):
        mgr = self._make_manager()
        lineage = mgr.get_lineage("regime_classifier")
        assert lineage["training_data_hash"] == ""
        assert lineage["training_rows"] == 0
        assert lineage["training_date"] == ""
        assert lineage["feature_set"] == ""

    def test_lineage_in_snapshot(self):
        mgr = self._make_manager()
        mgr.record_training_lineage(
            "regime_classifier", data_hash="snap_hash", rows=999, feature_set="snap_fs"
        )
        snap = mgr.snapshot()
        model_snap = snap["models"]["regime_classifier"]
        assert model_snap["training_data_hash"] == "snap_hash"
        assert model_snap["training_rows"] == 999
        assert model_snap["feature_set"] == "snap_fs"

    def test_lineage_overwrite(self):
        mgr = self._make_manager()
        mgr.record_training_lineage("rl_agent", "hash1", 100, "v1")
        mgr.record_training_lineage("rl_agent", "hash2", 200, "v2")
        lineage = mgr.get_lineage("rl_agent")
        assert lineage["training_data_hash"] == "hash2"
        assert lineage["training_rows"] == 200


# =========================================================================
# 5. DataQualityReport dataclass tests
# =========================================================================


class TestDataQualityReport:

    def test_default_values(self):
        import pytest; pytest.importorskip("data.quality")
        from data.quality import DataQualityReport
        report = DataQualityReport()
        assert report.passed is True
        assert report.issues == []
        assert report.rows_checked == 0
        assert report.rows_failed == 0
        assert report.fill_forward_count == 0

    def test_custom_values(self):
        import pytest; pytest.importorskip("data.quality")
        from data.quality import DataQualityReport
        report = DataQualityReport(
            passed=False, issues=["bad"], rows_checked=10,
            rows_failed=3, fill_forward_count=2,
        )
        assert report.passed is False
        assert len(report.issues) == 1
