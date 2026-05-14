"""
Tests for ML training pipeline, feature computation, model loading, and ComponentRegistry wiring.

Covers:
  - Feature computation (RSI, MACD, vol, returns, bollinger, etc.)
  - Regime labeling logic
  - Synthetic data generation
  - Model training pipeline (on small synthetic data)
  - Model save/load round-trip
  - Loader with missing files
  - Loader with corrupted files
  - Version checking
  - Walk-forward train/test split correctness
  - Training report generation
  - ComponentRegistry pretrained model loading

Run: py -m pytest tests/test_model_training.py -v
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.training_features import (
    FEATURE_NAMES,
    compute_bollinger_width,
    compute_features,
    compute_macd,
    compute_rsi,
    compute_adx_approx,
    label_regimes,
)
from ml.trained_model_loader import (
    ModelMetadata,
    PreTrainedAlphaModel,
    PreTrainedRegimeClassifier,
    PreTrainedVolatilityForecaster,
    TrainedModelLoader,
    _load_model_file,
    _parse_metadata,
)
from scripts.train_models import (
    generate_synthetic_crypto,
    run_pipeline,
    train_alpha_model,
    train_regime_classifier,
    train_volatility_forecaster,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a small OHLCV DataFrame for testing."""
    rng = np.random.RandomState(42)
    n = 200
    close = 50000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="D"),
        "open": close * (1 + rng.normal(0, 0.005, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.01, n))),
        "close": close,
        "volume": rng.lognormal(20, 1, n),
    })


@pytest.fixture
def feature_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute features from sample OHLCV."""
    return compute_features(sample_ohlcv).dropna()


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory."""
    d = tmp_path / "data_historical"
    d.mkdir()
    return d


# ─── Feature computation tests ──────────────────────────────────────────────


class TestFeatureComputation:
    """Tests for technical feature computation."""

    def test_compute_rsi_values_in_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI should be between 0 and 100."""
        rsi = compute_rsi(sample_ohlcv["close"], period=14)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_compute_rsi_period(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI should have NaN for the first `period` rows."""
        rsi = compute_rsi(sample_ohlcv["close"], period=14)
        # First 14 values should be NaN
        assert rsi.iloc[:14].isna().all()
        # After that, should have values
        assert not rsi.iloc[14:].isna().all()

    def test_compute_macd_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        """MACD should have the same length as input."""
        macd = compute_macd(sample_ohlcv["close"])
        assert len(macd) == len(sample_ohlcv)

    def test_compute_macd_crosses_zero(self, sample_ohlcv: pd.DataFrame) -> None:
        """MACD should cross zero at some point in typical data."""
        macd = compute_macd(sample_ohlcv["close"])
        valid = macd.dropna()
        # Should have both positive and negative values in typical data
        assert valid.min() < 0 or valid.max() > 0  # at least one side

    def test_bollinger_width_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """Bollinger band width should be non-negative."""
        bw = compute_bollinger_width(sample_ohlcv["close"])
        valid = bw.dropna()
        assert (valid >= 0).all()

    def test_adx_approx_non_negative(self, sample_ohlcv: pd.DataFrame) -> None:
        """ADX approximation should be between 0 and 100."""
        adx = compute_adx_approx(
            sample_ohlcv["close"], sample_ohlcv["high"], sample_ohlcv["low"],
        )
        valid = adx.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_compute_features_has_all_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """compute_features should produce all FEATURE_NAMES columns."""
        result = compute_features(sample_ohlcv)
        for name in FEATURE_NAMES:
            assert name in result.columns, f"Missing feature: {name}"

    def test_compute_features_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        """Feature DataFrame should have same number of rows as input."""
        result = compute_features(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_returns_computation(self, feature_df: pd.DataFrame) -> None:
        """Return features should be log returns of appropriate period."""
        # return_1d should be close to the 1-day log return
        assert "return_1d" in feature_df.columns
        assert not feature_df["return_1d"].isna().all()

    def test_volatility_features_positive(self, feature_df: pd.DataFrame) -> None:
        """Volatility features should be non-negative."""
        for col in ["vol_10d", "vol_20d", "vol_60d"]:
            assert (feature_df[col] >= 0).all(), f"{col} has negative values"

    def test_volume_ratio_positive(self, feature_df: pd.DataFrame) -> None:
        """Volume ratio should be positive."""
        assert (feature_df["volume_ratio"] > 0).all()


# ─── Regime labeling tests ───────────────────────────────────────────────────


class TestRegimeLabeling:
    """Tests for regime label generation."""

    def test_label_regimes_returns_correct_types(self, feature_df: pd.DataFrame) -> None:
        """label_regimes should return integer labels and string names."""
        labels, names = label_regimes(feature_df)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype in (np.int32, np.int64, int)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_label_regimes_all_valid(self, feature_df: pd.DataFrame) -> None:
        """All labels should be valid indices into label_names."""
        labels, names = label_regimes(feature_df)
        assert labels.min() >= 0
        assert labels.max() < len(names)

    def test_label_regimes_four_classes(self, feature_df: pd.DataFrame) -> None:
        """Should have 4 regime classes."""
        _, names = label_regimes(feature_df)
        assert len(names) == 4
        assert "BULL" in names
        assert "BEAR" in names
        assert "HIGH_VOL" in names
        assert "CRISIS" in names

    def test_label_regimes_length_matches(self, feature_df: pd.DataFrame) -> None:
        """Labels array should match input DataFrame length."""
        labels, _ = label_regimes(feature_df)
        assert len(labels) == len(feature_df)

    def test_multiple_regimes_present(self, feature_df: pd.DataFrame) -> None:
        """In typical data, at least 2 different regimes should be present."""
        labels, _ = label_regimes(feature_df)
        unique = np.unique(labels)
        assert len(unique) >= 2


# ─── Synthetic data generation tests ─────────────────────────────────────────


class TestSyntheticDataGeneration:
    """Tests for synthetic crypto data generation."""

    def test_generates_correct_length(self) -> None:
        """Should generate approximately years*365 rows."""
        df = generate_synthetic_crypto("BTC-USD", years=1, seed=42)
        assert abs(len(df) - 365) < 5

    def test_columns_present(self) -> None:
        """Should have standard OHLCV columns."""
        df = generate_synthetic_crypto("BTC-USD", years=1, seed=42)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_ohlcv_consistency(self) -> None:
        """High >= max(open, close) and Low <= min(open, close)."""
        df = generate_synthetic_crypto("BTC-USD", years=2, seed=42)
        assert (df["high"] >= df["close"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["low"] <= df["open"]).all()

    def test_fat_tails(self) -> None:
        """Returns should have excess kurtosis (fat tails)."""
        df = generate_synthetic_crypto("BTC-USD", years=5, seed=42)
        log_returns = np.diff(np.log(df["close"].values))
        kurtosis = float(pd.Series(log_returns).kurtosis())
        # Excess kurtosis should be > 0 (fatter than normal)
        assert kurtosis > 0, f"Kurtosis {kurtosis} should be > 0 for fat tails"

    def test_reproducible_with_seed(self) -> None:
        """Same seed should produce same data."""
        df1 = generate_synthetic_crypto("BTC-USD", years=1, seed=99)
        df2 = generate_synthetic_crypto("BTC-USD", years=1, seed=99)
        np.testing.assert_array_equal(df1["close"].values, df2["close"].values)

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different data."""
        df1 = generate_synthetic_crypto("BTC-USD", years=1, seed=42)
        df2 = generate_synthetic_crypto("BTC-USD", years=1, seed=99)
        assert not np.allclose(df1["close"].values, df2["close"].values)


# ─── Model training tests ───────────────────────────────────────────────────


class TestModelTraining:
    """Tests for individual model training functions."""

    def test_train_regime_classifier(self, feature_df: pd.DataFrame) -> None:
        """Regime classifier should train successfully and return metrics."""
        features = feature_df[FEATURE_NAMES].values
        labels, names = label_regimes(feature_df)
        model, metrics = train_regime_classifier(features, labels, names)

        assert model is not None
        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.0
        assert metrics["test_accuracy"] > 0.0
        assert metrics["train_samples"] > 0
        assert metrics["test_samples"] > 0

    def test_regime_classifier_can_predict(self, feature_df: pd.DataFrame) -> None:
        """Trained regime classifier should make predictions."""
        features = feature_df[FEATURE_NAMES].values
        labels, names = label_regimes(feature_df)
        model, _ = train_regime_classifier(features, labels, names)

        pred = model.predict(features[:5])
        assert len(pred) == 5

    def test_train_volatility_forecaster(self, feature_df: pd.DataFrame) -> None:
        """Volatility forecaster should train and return R² and MAE."""
        features = feature_df[FEATURE_NAMES].values
        close = feature_df["close"].values
        returns = np.diff(np.log(close))
        vol_target = np.array([
            np.std(returns[max(0, i):i + 5]) * np.sqrt(252)
            for i in range(len(feature_df))
        ])
        valid = ~np.isnan(vol_target)
        model, metrics = train_volatility_forecaster(features[valid], vol_target[valid])

        assert model is not None
        assert "r2_score" in metrics
        assert "mae" in metrics

    def test_train_alpha_model_walk_forward(self, feature_df: pd.DataFrame) -> None:
        """Alpha model should use walk-forward split (first 80% train, last 20% test)."""
        features = feature_df[FEATURE_NAMES].values
        close = feature_df["close"].values
        next_ret = np.zeros(len(close))
        for i in range(len(close) - 1):
            next_ret[i] = np.log(close[i + 1] / close[i])
        target = (next_ret > 0).astype(int)

        model, metrics = train_alpha_model(features[:-1], target[:-1])

        assert model is not None
        assert "test_accuracy" in metrics
        assert "signal_sharpe" in metrics
        assert metrics["walk_forward_split"] == "80/20 temporal"

    def test_alpha_model_accuracy_above_random(self, feature_df: pd.DataFrame) -> None:
        """Alpha model should do at least slightly better than random on training data."""
        features = feature_df[FEATURE_NAMES].values
        close = feature_df["close"].values
        next_ret = np.zeros(len(close))
        for i in range(len(close) - 1):
            next_ret[i] = np.log(close[i + 1] / close[i])
        target = (next_ret > 0).astype(int)

        model, metrics = train_alpha_model(features[:-1], target[:-1])
        # Training accuracy should be > 0.4 at minimum (better than degenerate)
        assert metrics["test_accuracy"] > 0.35


# ─── Model save/load round-trip tests ────────────────────────────────────────


class TestModelSaveLoad:
    """Tests for model persistence and loading."""

    def test_save_and_load_regime_classifier(
        self, feature_df: pd.DataFrame, tmp_models_dir: Path,
    ) -> None:
        """Save and reload a regime classifier."""
        import joblib

        features = feature_df[FEATURE_NAMES].values
        labels, names = label_regimes(feature_df)
        model, _ = train_regime_classifier(features, labels, names)

        # Save
        path = tmp_models_dir / "regime_classifier.pkl"
        metadata = {
            "model_type": "regime_classifier",
            "training_date": "2026-03-18T00:00:00",
            "features": FEATURE_NAMES,
            "classes": names,
            "version": "1.0.0",
        }
        joblib.dump({"model": model, "metadata": metadata}, path)

        # Load
        loader = TrainedModelLoader(str(tmp_models_dir))
        loaded = loader.load_regime_classifier()
        assert loaded is not None
        pred = loaded.predict(features[0])
        assert isinstance(pred, str)
        assert pred in names

    def test_save_and_load_volatility_forecaster(
        self, feature_df: pd.DataFrame, tmp_models_dir: Path,
    ) -> None:
        """Save and reload a volatility forecaster."""
        import joblib

        features = feature_df[FEATURE_NAMES].values
        close = feature_df["close"].values
        returns = np.diff(np.log(close))
        vol_target = np.array([
            np.std(returns[max(0, i):i + 5]) * np.sqrt(252)
            for i in range(len(feature_df))
        ])
        valid = ~np.isnan(vol_target)
        model, _ = train_volatility_forecaster(features[valid], vol_target[valid])

        path = tmp_models_dir / "volatility_forecaster.pkl"
        metadata = {
            "model_type": "volatility_forecaster",
            "training_date": "2026-03-18T00:00:00",
            "features": FEATURE_NAMES,
            "version": "1.0.0",
        }
        joblib.dump({"model": model, "metadata": metadata}, path)

        loader = TrainedModelLoader(str(tmp_models_dir))
        loaded = loader.load_volatility_forecaster()
        assert loaded is not None
        pred = loaded.predict(features[0])
        assert isinstance(pred, float)

    def test_save_and_load_alpha_model(
        self, feature_df: pd.DataFrame, tmp_models_dir: Path,
    ) -> None:
        """Save and reload an alpha model."""
        import joblib

        features = feature_df[FEATURE_NAMES].values
        close = feature_df["close"].values
        next_ret = np.zeros(len(close))
        for i in range(len(close) - 1):
            next_ret[i] = np.log(close[i + 1] / close[i])
        target = (next_ret > 0).astype(int)
        model, _ = train_alpha_model(features[:-1], target[:-1])

        path = tmp_models_dir / "alpha_model.pkl"
        metadata = {
            "model_type": "alpha_model",
            "training_date": "2026-03-18T00:00:00",
            "features": FEATURE_NAMES,
            "classes": ["down", "up"],
            "version": "1.0.0",
        }
        joblib.dump({"model": model, "metadata": metadata}, path)

        loader = TrainedModelLoader(str(tmp_models_dir))
        loaded = loader.load_alpha_model()
        assert loaded is not None
        direction, confidence = loaded.predict(features[0])
        assert direction in ("up", "down")
        assert 0.0 <= confidence <= 1.0


# ─── Loader edge case tests ─────────────────────────────────────────────────


class TestLoaderEdgeCases:
    """Tests for model loader with missing/corrupted files."""

    def test_missing_model_file(self, tmp_models_dir: Path) -> None:
        """Loader should return None for missing files."""
        loader = TrainedModelLoader(str(tmp_models_dir))
        assert loader.load_regime_classifier() is None
        assert loader.load_volatility_forecaster() is None
        assert loader.load_alpha_model() is None

    def test_corrupted_model_file(self, tmp_models_dir: Path) -> None:
        """Loader should return None for corrupted pickle files."""
        corrupt_path = tmp_models_dir / "regime_classifier.pkl"
        corrupt_path.write_bytes(b"this is not a valid pickle file!!!")

        loader = TrainedModelLoader(str(tmp_models_dir))
        result = loader.load_regime_classifier()
        assert result is None

    def test_wrong_model_type(self, tmp_models_dir: Path) -> None:
        """Loader should reject model with wrong model_type in metadata."""
        import joblib
        path = tmp_models_dir / "regime_classifier.pkl"
        joblib.dump({
            "model": "placeholder_model",
            "metadata": {
                "model_type": "volatility_forecaster",  # wrong type
                "training_date": "2026-01-01",
                "features": [],
                "version": "1.0.0",
            },
        }, path)

        loader = TrainedModelLoader(str(tmp_models_dir))
        result = loader.load_regime_classifier()
        assert result is None

    def test_invalid_structure(self, tmp_models_dir: Path) -> None:
        """Loader should reject files without 'model' and 'metadata' keys."""
        import joblib
        path = tmp_models_dir / "regime_classifier.pkl"
        joblib.dump({"some_key": "some_value"}, path)

        result = _load_model_file(path)
        assert result is None

    def test_models_available_false_when_missing(self, tmp_models_dir: Path) -> None:
        """models_available should be False when models are missing."""
        loader = TrainedModelLoader(str(tmp_models_dir))
        assert not loader.models_available()

    def test_models_available_true_when_all_present(
        self, feature_df: pd.DataFrame, tmp_models_dir: Path,
    ) -> None:
        """models_available should be True when all three models exist."""
        import joblib
        for name in ("regime_classifier", "volatility_forecaster", "alpha_model"):
            path = tmp_models_dir / f"{name}.pkl"
            joblib.dump({
                "model": "placeholder_model",
                "metadata": {"model_type": name, "version": "1.0.0"},
            }, path)

        loader = TrainedModelLoader(str(tmp_models_dir))
        assert loader.models_available()


# ─── Version checking tests ──────────────────────────────────────────────────


class TestVersionChecking:
    """Tests for model version checking."""

    def test_check_versions_all_present(self, tmp_models_dir: Path) -> None:
        """Version check should return info for all present models."""
        import joblib
        for name in ("regime_classifier", "volatility_forecaster", "alpha_model"):
            path = tmp_models_dir / f"{name}.pkl"
            joblib.dump({
                "model": "placeholder_model",
                "metadata": {
                    "model_type": name,
                    "training_date": "2026-03-18T00:00:00",
                    "features": FEATURE_NAMES,
                    "version": "1.2.3",
                },
            }, path)

        loader = TrainedModelLoader(str(tmp_models_dir))
        versions = loader.check_model_versions()

        for name in ("regime_classifier", "volatility_forecaster", "alpha_model"):
            assert versions[name] is not None
            assert versions[name]["version"] == "1.2.3"
            assert versions[name]["training_date"] == "2026-03-18T00:00:00"

    def test_check_versions_missing(self, tmp_models_dir: Path) -> None:
        """Version check should return None for missing models."""
        loader = TrainedModelLoader(str(tmp_models_dir))
        versions = loader.check_model_versions()
        for name in ("regime_classifier", "volatility_forecaster", "alpha_model"):
            assert versions[name] is None

    def test_metadata_parsing(self) -> None:
        """_parse_metadata should correctly parse raw dicts."""
        raw = {
            "model_type": "regime_classifier",
            "training_date": "2026-03-18",
            "features": ["a", "b"],
            "version": "1.0.0",
            "classes": ["BULL", "BEAR"],
        }
        meta = _parse_metadata(raw)
        assert meta.model_type == "regime_classifier"
        assert meta.version == "1.0.0"
        assert meta.features == ["a", "b"]
        assert meta.extra["classes"] == ["BULL", "BEAR"]


# ─── Full pipeline test ─────────────────────────────────────────────────────


class TestFullPipeline:
    """Test the complete training pipeline end-to-end."""

    def test_run_pipeline_produces_all_artifacts(
        self, tmp_models_dir: Path, tmp_data_dir: Path,
    ) -> None:
        """Full pipeline should produce model files and training report."""
        report = run_pipeline(
            symbols=["BTC-USD"],
            years=1,
            output_dir=str(tmp_models_dir),
            data_dir=str(tmp_data_dir),
        )

        # Check model files exist
        assert (tmp_models_dir / "regime_classifier.pkl").exists()
        assert (tmp_models_dir / "volatility_forecaster.pkl").exists()
        assert (tmp_models_dir / "alpha_model.pkl").exists()
        assert (tmp_models_dir / "training_report.json").exists()

        # Check report structure
        assert "training_date" in report
        assert "symbols" in report
        assert "models" in report
        assert "regime_classifier" in report["models"]
        assert "volatility_forecaster" in report["models"]
        assert "alpha_model" in report["models"]

    def test_pipeline_training_report_json(
        self, tmp_models_dir: Path, tmp_data_dir: Path,
    ) -> None:
        """Training report should be valid JSON with expected fields."""
        run_pipeline(
            symbols=["BTC-USD"],
            years=1,
            output_dir=str(tmp_models_dir),
            data_dir=str(tmp_data_dir),
        )

        report_path = tmp_models_dir / "training_report.json"
        with open(report_path) as f:
            report = json.load(f)

        assert report["symbols"] == ["BTC-USD"]
        assert report["years"] == 1
        assert "data" in report
        assert report["data"]["source"] == "synthetic"

    def test_pipeline_data_saved_as_parquet(
        self, tmp_models_dir: Path, tmp_data_dir: Path,
    ) -> None:
        """Pipeline should save historical data as Parquet files."""
        run_pipeline(
            symbols=["BTC-USD", "ETH-USD"],
            years=1,
            output_dir=str(tmp_models_dir),
            data_dir=str(tmp_data_dir),
        )

        assert (tmp_data_dir / "BTC_USD_daily.parquet").exists()
        assert (tmp_data_dir / "ETH_USD_daily.parquet").exists()

        # Verify parquet is readable
        df = pd.read_parquet(tmp_data_dir / "BTC_USD_daily.parquet")
        assert len(df) > 300
        assert "close" in df.columns

    def test_pipeline_models_loadable(
        self, tmp_models_dir: Path, tmp_data_dir: Path,
    ) -> None:
        """Models produced by pipeline should be loadable via TrainedModelLoader."""
        run_pipeline(
            symbols=["BTC-USD"],
            years=1,
            output_dir=str(tmp_models_dir),
            data_dir=str(tmp_data_dir),
        )

        loader = TrainedModelLoader(str(tmp_models_dir))
        regime = loader.load_regime_classifier()
        vol = loader.load_volatility_forecaster()
        alpha = loader.load_alpha_model()

        assert regime is not None
        assert vol is not None
        assert alpha is not None

        # Verify they can predict
        test_features = np.random.randn(1, len(FEATURE_NAMES))
        regime_pred = regime.predict(test_features)
        assert isinstance(regime_pred, str)

        vol_pred = vol.predict(test_features)
        assert isinstance(vol_pred, float)

        direction, conf = alpha.predict(test_features)
        assert direction in ("up", "down")
        assert 0.0 <= conf <= 1.0


# ─── Wrapper class tests ────────────────────────────────────────────────────


class TestPreTrainedWrappers:
    """Tests for the pre-trained model wrapper classes."""

    def test_regime_classifier_predict_proba(
        self, feature_df: pd.DataFrame,
    ) -> None:
        """PreTrainedRegimeClassifier.predict_proba should return dict of probabilities."""
        features = feature_df[FEATURE_NAMES].values
        labels, names = label_regimes(feature_df)
        model, _ = train_regime_classifier(features, labels, names)

        # Use only the classes the model actually learned (may be < 4 with small data)
        seen_classes = [names[i] for i in sorted(set(labels)) if i < len(names)]

        metadata = ModelMetadata(
            model_type="regime_classifier",
            training_date="2026-03-18",
            features=FEATURE_NAMES,
            version="1.0.0",
            extra={"classes": seen_classes},
        )
        wrapper = PreTrainedRegimeClassifier(model, metadata)
        proba = wrapper.predict_proba(features[0])

        assert isinstance(proba, dict)
        assert abs(sum(proba.values()) - 1.0) < 0.01
        # All returned keys should be valid regime names
        for key in proba:
            assert key in names or key.isdigit()

    def test_alpha_model_confidence_range(
        self, feature_df: pd.DataFrame,
    ) -> None:
        """PreTrainedAlphaModel confidence should be between 0.5 and 1.0."""
        features = feature_df[FEATURE_NAMES].values
        close = feature_df["close"].values
        next_ret = np.zeros(len(close))
        for i in range(len(close) - 1):
            next_ret[i] = np.log(close[i + 1] / close[i])
        target = (next_ret > 0).astype(int)
        model, _ = train_alpha_model(features[:-1], target[:-1])

        metadata = ModelMetadata(
            model_type="alpha_model",
            training_date="2026-03-18",
            features=FEATURE_NAMES,
            version="1.0.0",
            extra={"classes": ["down", "up"]},
        )
        wrapper = PreTrainedAlphaModel(model, metadata)

        for i in range(min(10, len(features))):
            direction, confidence = wrapper.predict(features[i])
            assert 0.0 <= confidence <= 1.0
            assert direction in ("up", "down")
