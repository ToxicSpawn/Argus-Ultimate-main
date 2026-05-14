"""
tests_unified/test_deep_lob.py — Unit tests for ml/deep_lob.py

8 tests covering DeepLOBFeatures, DeepLOBModel, and OnlineLOBPredictor.
Torch is NOT required for the majority of tests.
"""

from __future__ import annotations

import sys
import os
import math
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.deep_lob import (
    DeepLOBFeatures,
    DeepLOBModel,
    OnlineLOBPredictor,
    DIRECTION_LABELS,
    _TORCH_AVAILABLE,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_bids():
    """10 bid levels, sorted descending."""
    base = 45000.0
    return [(base - i * 0.5, 1.0 + i * 0.2) for i in range(10)]


@pytest.fixture
def sample_asks():
    """10 ask levels, sorted ascending."""
    base = 45001.0
    return [(base + i * 0.5, 1.0 + i * 0.2) for i in range(10)]


@pytest.fixture
def feature_extractor():
    return DeepLOBFeatures()


@pytest.fixture
def sample_features(feature_extractor, sample_bids, sample_asks):
    return feature_extractor.extract(sample_bids, sample_asks, n_levels=10)


# ─── Test 1: Feature extraction shape ────────────────────────────────────────

def test_feature_extraction_shape(feature_extractor, sample_bids, sample_asks):
    """DeepLOBFeatures.extract should return shape (40,) for 10 levels."""
    feat = feature_extractor.extract(sample_bids, sample_asks, n_levels=10)
    assert feat is not None, "Features should not be None with valid data"
    assert feat.shape == (40,), f"Expected shape (40,), got {feat.shape}"
    assert feat.dtype == np.float32


# ─── Test 2: Feature normalisation ───────────────────────────────────────────

def test_feature_normalisation(feature_extractor, sample_bids, sample_asks):
    """Price features should be centred near zero (normalised by mid)."""
    feat = feature_extractor.extract(sample_bids, sample_asks, n_levels=10)
    assert feat is not None

    bid_prices = feat[:10]
    ask_prices = feat[10:20]

    # Best bid normalised price should be slightly negative (below mid)
    assert bid_prices[0] < 0.0, "Best bid normalised price should be < 0"
    # Best ask normalised price should be slightly positive (above mid)
    assert ask_prices[0] > 0.0, "Best ask normalised price should be > 0"

    # Size features should sum to 1.0 (normalised)
    bid_sizes = feat[20:30]
    ask_sizes = feat[30:40]
    total_size = bid_sizes.sum() + ask_sizes.sum()
    assert abs(total_size - 1.0) < 1e-5, f"Sizes should sum to 1.0, got {total_size}"


# ─── Test 3: Feature extraction with insufficient data ───────────────────────

def test_feature_extraction_insufficient_data(feature_extractor):
    """extract() should return None when data is insufficient."""
    # Empty bids
    result = feature_extractor.extract([], [(45001.0, 1.0)])
    assert result is None, "Should return None with empty bids"

    # Empty asks
    result = feature_extractor.extract([(45000.0, 1.0)], [])
    assert result is None, "Should return None with empty asks"

    # Invalid prices
    result = feature_extractor.extract(
        [(45000.0, 1.0)],
        [(44999.0, 1.0)],  # ask < bid — invalid
    )
    assert result is None, "Should return None when ask <= bid"


# ─── Test 4: Feature padding with fewer than n_levels ────────────────────────

def test_feature_padding(feature_extractor):
    """extract() should pad to n_levels even with fewer input levels."""
    bids = [(45000.0, 1.0), (44999.5, 0.5)]  # only 2 levels
    asks = [(45001.0, 1.0), (45001.5, 0.5)]

    feat = feature_extractor.extract(bids, asks, n_levels=10)
    assert feat is not None
    assert feat.shape == (40,)

    # Padded prices should be 0.0 normalised to (0-mid)/mid (< -1)
    # but sizes should be 0 for padded slots
    bid_sizes = feat[20:30]
    # Only the first 2 slots should be non-zero
    assert bid_sizes[2] == 0.0, "Padded size slots should be 0"


# ─── Test 5: DeepLOBModel predict fallback (no torch) ────────────────────────

def test_model_predict_no_torch_fallback(sample_features):
    """When torch is unavailable, predict() returns a FLAT fallback signal."""
    model = DeepLOBModel()

    if not _TORCH_AVAILABLE:
        result = model.predict(sample_features)
        assert result["direction"] == "FLAT"
        assert "fallback" in result.get("source", "")
    else:
        # Torch available — model is untrained, result should still be a valid dict
        result = model.predict(sample_features)
        assert "direction" in result
        assert result["direction"] in ("UP", "FLAT", "DOWN")
        assert 0.0 <= result["probability"] <= 1.0


# ─── Test 6: DeepLOBModel predict output schema ───────────────────────────────

def test_model_predict_schema(sample_features):
    """predict() must return a dict with required keys."""
    model = DeepLOBModel()
    result = model.predict(sample_features)

    required_keys = {"direction", "probability", "confidence"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )
    assert result["direction"] in ("UP", "FLAT", "DOWN")
    assert 0.0 <= result["probability"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


# ─── Test 7: OnlineLOBPredictor OBI fallback ─────────────────────────────────

def test_online_predictor_obi_fallback():
    """Without a trained model, OnlineLOBPredictor falls back to OBI threshold."""
    predictor = OnlineLOBPredictor()

    # Strong buy pressure
    bids = [(45000.0, 10.0), (44999.5, 8.0), (44999.0, 6.0)]
    asks = [(45001.0, 0.5),  (45001.5, 0.5), (45002.0, 0.5)]
    signal = predictor.predict({"bids": bids, "asks": asks})

    assert "direction" in signal
    assert "confidence" in signal
    assert signal["direction"] in ("UP", "FLAT", "DOWN")
    # Strong bid imbalance → should signal UP or FLAT (OBI fallback)
    obi = signal.get("obi", 0.0)
    # OBI should be very positive (bids >> asks)
    assert obi > 0.0, f"OBI should be positive for bid-heavy book, got {obi}"


# ─── Test 8: OnlineLOBPredictor update buffer and retrain ────────────────────

def test_online_predictor_update_and_buffer():
    """OnlineLOBPredictor.update() should accumulate samples and trigger retrain."""
    predictor = OnlineLOBPredictor(buffer_size=50, retrain_every=10, mini_batch=8)
    extractor = DeepLOBFeatures()

    bids = [(45000.0 - i * 0.5, 1.0 + i * 0.1) for i in range(10)]
    asks = [(45001.0 + i * 0.5, 1.0 + i * 0.1) for i in range(10)]
    features = extractor.extract(bids, asks)

    # Add enough samples to trigger retrain threshold
    for i in range(20):
        predictor.update(features, outcome=i % 3)

    # Buffer should have 20 entries
    assert len(predictor._buffer) == 20
    assert predictor._sample_count == 20

    # predict() should still work and return valid schema
    signal = predictor.predict({"bids": bids, "asks": asks})
    assert "direction" in signal
    assert signal["direction"] in ("UP", "FLAT", "DOWN")
    assert "timestamp_ms" in signal
    assert isinstance(signal["timestamp_ms"], int)
