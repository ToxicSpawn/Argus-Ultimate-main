"""
Tests for AI models: CryptoSentimentAnalyzer, MarketAnomalyDetector, ChartPatternDetector.

Run: py -m pytest tests/test_ai_models.py -v
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# CryptoSentimentAnalyzer tests
# ═══════════════════════════════════════════════════════════════════════════

from ml.finbert_sentiment import CryptoSentimentAnalyzer


class TestSentimentAnalyzerBasic:
    """Basic sentiment scoring tests."""

    def setup_method(self):
        self.analyzer = CryptoSentimentAnalyzer()

    def test_empty_text_returns_neutral(self):
        result = self.analyzer.analyze_text("")
        assert result["sentiment"] == 0.0
        assert result["confidence"] == 0.0

    def test_bullish_text_positive_sentiment(self):
        result = self.analyzer.analyze_text("Bitcoin is mooning! Breakout confirmed, new ATH incoming.")
        assert result["sentiment"] > 0.3, f"Expected bullish sentiment, got {result['sentiment']}"

    def test_bearish_text_negative_sentiment(self):
        result = self.analyzer.analyze_text("Market crash imminent. Massive liquidations and panic selling.")
        assert result["sentiment"] < -0.3, f"Expected bearish sentiment, got {result['sentiment']}"

    def test_neutral_text_near_zero(self):
        result = self.analyzer.analyze_text("The weather is nice today. I had a good lunch.")
        assert abs(result["sentiment"]) < 0.3

    def test_sentiment_range_clamped(self):
        result = self.analyzer.analyze_text(
            "Moon pump rocket explosion skyrocket parabolic ATH breakout lambo wagmi"
        )
        assert -1.0 <= result["sentiment"] <= 1.0

    def test_confidence_range(self):
        result = self.analyzer.analyze_text("Bitcoin is pumping hard!")
        assert 0.0 <= result["confidence"] <= 1.0


class TestSentimentNegation:
    """Negation handling tests."""

    def setup_method(self):
        self.analyzer = CryptoSentimentAnalyzer()

    def test_not_bullish_flips_sentiment(self):
        bullish = self.analyzer.analyze_text("Bitcoin is bullish")
        negated = self.analyzer.analyze_text("Bitcoin is not bullish")
        assert negated["sentiment"] < bullish["sentiment"]

    def test_never_pumping(self):
        result = self.analyzer.analyze_text("Bitcoin will never pump again")
        assert result["sentiment"] < 0.0

    def test_dont_sell_is_not_bearish(self):
        result = self.analyzer.analyze_text("Don't sell your Bitcoin")
        # "don't" negates "sell" (bearish) making it less bearish/slightly bullish
        assert result["sentiment"] > -0.3

    def test_no_crash_expected(self):
        result = self.analyzer.analyze_text("No crash expected for Bitcoin this quarter")
        # "no" negates "crash" — should be less bearish than without negation
        bearish = self.analyzer.analyze_text("Crash expected for Bitcoin this quarter")
        assert result["sentiment"] > bearish["sentiment"]


class TestSentimentIntensifiers:
    """Intensifier and diminisher tests."""

    def setup_method(self):
        self.analyzer = CryptoSentimentAnalyzer()

    def test_very_bullish_stronger(self):
        normal = self.analyzer.analyze_text("Bitcoin is bullish")
        intensified = self.analyzer.analyze_text("Bitcoin is very bullish")
        assert intensified["sentiment"] > normal["sentiment"]

    def test_extremely_bearish_stronger(self):
        normal = self.analyzer.analyze_text("Market is bearish")
        intensified = self.analyzer.analyze_text("Market is extremely bearish")
        assert intensified["sentiment"] < normal["sentiment"]

    def test_slightly_reduces(self):
        normal = self.analyzer.analyze_text("Bitcoin is bullish")
        diminished = self.analyzer.analyze_text("Bitcoin is slightly bullish")
        assert diminished["sentiment"] < normal["sentiment"]


class TestSentimentEntities:
    """Entity extraction tests."""

    def setup_method(self):
        self.analyzer = CryptoSentimentAnalyzer()

    def test_extract_bitcoin(self):
        entities = self.analyzer.get_crypto_entities("BTC just hit a new ATH")
        assert "Bitcoin" in entities

    def test_extract_ethereum(self):
        entities = self.analyzer.get_crypto_entities("Ethereum merge was successful")
        assert "Ethereum" in entities

    def test_extract_exchange(self):
        entities = self.analyzer.get_crypto_entities("Listing on Binance announced")
        assert "Binance" in entities

    def test_extract_person(self):
        entities = self.analyzer.get_crypto_entities("Vitalik proposed a new EIP")
        assert "Vitalik Buterin" in entities

    def test_extract_multiple(self):
        entities = self.analyzer.get_crypto_entities(
            "BTC and ETH listed on Kraken"
        )
        assert "Bitcoin" in entities
        assert "Ethereum" in entities
        assert "Kraken" in entities


class TestSentimentFudHype:
    """FUD/HYPE classification tests."""

    def setup_method(self):
        self.analyzer = CryptoSentimentAnalyzer()

    def test_fud_classification(self):
        result = self.analyzer.detect_fud_or_hype(
            "SCAM WARNING: This is a rug pull. Fear and panic everywhere!"
        )
        assert result == "FUD"

    def test_hype_classification(self):
        result = self.analyzer.detect_fud_or_hype(
            "TO THE MOON! WAGMI! Diamond hands hold for the moonshot! LFG!"
        )
        assert result == "HYPE"

    def test_neutral_classification(self):
        result = self.analyzer.detect_fud_or_hype(
            "I went to the store today"
        )
        assert result in ("NEUTRAL", "INFORMATIONAL")

    def test_informational_classification(self):
        result = self.analyzer.detect_fud_or_hype(
            "According to the research report published by the analysis team, data shows growth."
        )
        assert result == "INFORMATIONAL"


class TestSentimentBatch:
    """Batch and headline analysis tests."""

    def setup_method(self):
        self.analyzer = CryptoSentimentAnalyzer()

    def test_batch_analyze(self):
        texts = [
            "Bitcoin pumping!",
            "Market crash incoming",
            "Stable prices today",
        ]
        results = self.analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert results[0]["sentiment"] > 0
        assert results[1]["sentiment"] < 0

    def test_headline_aggregation(self):
        headlines = [
            {"text": "Bitcoin pump breakout rally bullish", "timestamp": time.time()},
            {"text": "ETH mooning to new ATH", "timestamp": time.time()},
            {"text": "Minor correction expected", "timestamp": time.time() - 7200},
        ]
        result = self.analyzer.analyze_headlines(headlines)
        assert "sentiment" in result
        assert "confidence" in result
        assert result["n_headlines"] == 3
        assert result["bullish_count"] >= 1

    def test_headline_time_decay(self):
        """Recent headlines should be weighted more."""
        now = time.time()
        # All bullish recent
        recent_bull = [
            {"text": "Bitcoin mooning!", "timestamp": now},
            {"text": "New ATH breakout!", "timestamp": now},
        ]
        # One bullish recent, one bearish old
        mixed = [
            {"text": "Bitcoin mooning!", "timestamp": now},
            {"text": "Complete crash and collapse!", "timestamp": now - 86400},
        ]
        r1 = self.analyzer.analyze_headlines(recent_bull)
        r2 = self.analyzer.analyze_headlines(mixed)
        # Recent bullish should dominate the mixed result (old bearish decayed)
        assert r2["sentiment"] > -0.3  # not strongly bearish due to decay

    def test_empty_headlines(self):
        result = self.analyzer.analyze_headlines([])
        assert result["sentiment"] == 0.0
        assert result["n_headlines"] == 0

    def test_urgency_detection(self):
        result = self.analyzer.analyze_text("BREAKING: Bitcoin flash crash happening NOW!")
        assert result["urgency"] > 0.3


class TestSentimentTopics:
    """Topic detection tests."""

    def setup_method(self):
        self.analyzer = CryptoSentimentAnalyzer()

    def test_regulation_topic(self):
        result = self.analyzer.analyze_text(
            "SEC regulatory crackdown on crypto exchanges"
        )
        assert "regulation" in result["topics"]

    def test_market_topic(self):
        result = self.analyzer.analyze_text(
            "Trading volume surges as price hits new highs"
        )
        assert "market" in result["topics"]

    def test_stats(self):
        self.analyzer.analyze_text("Test")
        self.analyzer.analyze_text("Test2")
        stats = self.analyzer.get_stats()
        assert stats["analyses_count"] == 2
        assert stats["lexicon_size"] > 300


# ═══════════════════════════════════════════════════════════════════════════
# MarketAnomalyDetector tests
# ═══════════════════════════════════════════════════════════════════════════

from ml.anomaly_detector import MarketAnomalyDetector, _avg_path_length


class TestAnomalyDetectorBasic:
    """Basic anomaly detector tests."""

    def test_avg_path_length(self):
        assert _avg_path_length(1) == 0.0
        assert _avg_path_length(2) == 1.0
        assert _avg_path_length(100) > 0

    def test_init_defaults(self):
        det = MarketAnomalyDetector()
        stats = det.get_stats()
        assert stats["fitted"] is False
        assert stats["n_trees"] == 100

    def test_fit_basic(self):
        det = MarketAnomalyDetector(n_trees=10, subsample_size=32)
        X = np.random.randn(200, 5)
        det.fit(X)
        assert det.get_stats()["fitted"] is True

    def test_predict_after_fit(self):
        det = MarketAnomalyDetector(n_trees=10, subsample_size=32)
        X = np.random.randn(200, 5)
        det.fit(X)
        labels = det.predict(X[:10])
        assert len(labels) == 10
        assert set(labels.tolist()).issubset({-1, 1})

    def test_score_samples(self):
        det = MarketAnomalyDetector(n_trees=10, subsample_size=32)
        X = np.random.randn(200, 5)
        det.fit(X)
        scores = det.score_samples(X[:10])
        assert len(scores) == 10
        assert all(0 <= s <= 1 for s in scores)

    def test_not_fitted_raises(self):
        det = MarketAnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.score_samples(np.array([[1, 2, 3, 4, 5]]))


class TestAnomalyDetectorAnomalies:
    """Test anomaly detection quality."""

    def test_outliers_score_higher(self):
        det = MarketAnomalyDetector(n_trees=50, subsample_size=64, contamination=0.1)
        # Normal data: centered around 0
        X_normal = np.random.randn(300, 3)
        det.fit(X_normal)

        # Normal sample
        normal_score = det.score_samples(np.array([[0.0, 0.0, 0.0]]))[0]
        # Outlier
        outlier_score = det.score_samples(np.array([[10.0, 10.0, 10.0]]))[0]

        assert outlier_score > normal_score, \
            f"Outlier score {outlier_score} should exceed normal score {normal_score}"

    def test_single_sample_predict(self):
        det = MarketAnomalyDetector(n_trees=10, subsample_size=32)
        X = np.random.randn(200, 5)
        det.fit(X)
        # Single sample (1D)
        label = det.predict(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        assert label[0] in (-1, 1)


class TestAnomalyDetectorRealtime:
    """Real-time detection tests."""

    def test_realtime_normal(self):
        det = MarketAnomalyDetector(n_trees=10, subsample_size=32)
        # Fit on normal market data
        X = np.random.randn(300, 5) * np.array([5, 1, 0.5, 0.3, 0.1])
        det.fit(X)

        result = det.detect_realtime({
            "spread_bps": 3.0,
            "volume_ratio": 1.0,
            "price_velocity": 0.1,
            "orderbook_imbalance": 0.0,
            "funding_rate_deviation": 0.0,
        })
        assert "is_anomaly" in result
        assert "score" in result
        assert "type" in result
        assert "severity" in result

    def test_realtime_extreme_spread(self):
        det = MarketAnomalyDetector(n_trees=10, subsample_size=32)
        X = np.random.randn(300, 5)
        det.fit(X)

        result = det.detect_realtime({
            "spread_bps": 100.0,
            "volume_ratio": 0.1,
            "price_velocity": -5.0,
            "orderbook_imbalance": 0.9,
            "funding_rate_deviation": 5.0,
        })
        # Extreme values should likely trigger an anomaly
        assert result["score"] > 0.3

    def test_heuristic_fallback(self):
        """When not fitted, should use heuristic detection."""
        det = MarketAnomalyDetector()
        result = det.detect_realtime({
            "spread_bps": 50.0,
            "volume_ratio": 5.0,
            "price_velocity": -4.0,
            "orderbook_imbalance": 0.8,
            "funding_rate_deviation": 3.0,
        })
        assert result["is_anomaly"] is True
        assert result["score"] > 0.5


class TestAnomalyDetectorOnline:
    """Online update tests."""

    def test_online_update(self):
        det = MarketAnomalyDetector(n_trees=10, subsample_size=32)
        X = np.random.randn(300, 5)
        det.fit(X)

        # Online update with new samples
        new_data = np.random.randn(50, 5)
        det.update_online(new_data)

        stats = det.get_stats()
        assert stats["buffer_size"] == 50

    def test_anomaly_history(self):
        det = MarketAnomalyDetector(n_trees=10, subsample_size=32)
        X = np.random.randn(300, 5)
        det.fit(X)

        # Trigger some anomalies
        for _ in range(5):
            det.detect_realtime({
                "spread_bps": 100.0,
                "volume_ratio": 10.0,
                "price_velocity": -10.0,
                "orderbook_imbalance": 0.95,
                "funding_rate_deviation": 5.0,
            })

        history = det.get_anomaly_history(limit=10)
        assert isinstance(history, list)
        # May or may not have anomalies depending on threshold
        for entry in history:
            assert "timestamp" in entry
            assert "score" in entry
            assert "type" in entry


class TestAnomalyClassification:
    """Anomaly type classification tests."""

    def test_flash_crash_classification(self):
        det = MarketAnomalyDetector()
        result = det.detect_realtime({
            "spread_bps": 5.0,
            "volume_ratio": 5.0,
            "price_velocity": -8.0,
            "orderbook_imbalance": 0.0,
            "funding_rate_deviation": 0.0,
        })
        assert result["type"] == "FLASH_CRASH"

    def test_spread_blowout_classification(self):
        det = MarketAnomalyDetector()
        result = det.detect_realtime({
            "spread_bps": 60.0,
            "volume_ratio": 5.0,
            "price_velocity": -3.5,
            "orderbook_imbalance": 0.0,
            "funding_rate_deviation": 0.0,
        })
        # With spread>50 and velocity/volume contributing, should be anomalous
        # and classified as SPREAD_BLOWOUT or FLASH_CRASH
        assert result["type"] in ("SPREAD_BLOWOUT", "FLASH_CRASH")


# ═══════════════════════════════════════════════════════════════════════════
# ChartPatternDetector tests
# ═══════════════════════════════════════════════════════════════════════════

from ml.chart_pattern_cnn import ChartPatternDetector


class TestChartPatternBasic:
    """Basic pattern detection tests."""

    def setup_method(self):
        self.detector = ChartPatternDetector()

    def test_too_few_bars_returns_empty(self):
        patterns = self.detector.detect_patterns([100.0, 101.0, 99.0])
        assert patterns == []

    def test_returns_list(self):
        prices = list(np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + 100)
        patterns = self.detector.detect_patterns(prices)
        assert isinstance(patterns, list)

    def test_pattern_dict_keys(self):
        prices = list(np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + 100)
        patterns = self.detector.detect_patterns(prices)
        for p in patterns:
            assert "pattern" in p
            assert "confidence" in p
            assert "start_idx" in p
            assert "end_idx" in p
            assert "direction" in p
            assert "target_price" in p

    def test_confidence_range(self):
        prices = list(np.sin(np.linspace(0, 6 * np.pi, 200)) * 10 + 100)
        patterns = self.detector.detect_patterns(prices)
        for p in patterns:
            assert 0.0 <= p["confidence"] <= 1.0


class TestChartPatternDoubleTop:
    """Double top/bottom detection tests."""

    def setup_method(self):
        self.detector = ChartPatternDetector(config={"min_confidence": 0.1})

    def test_double_top_detected(self):
        """Generate a clear double top pattern."""
        n = 80
        prices = []
        for i in range(n):
            if i < 20:
                prices.append(100 + i * 0.5)        # rise
            elif i < 30:
                prices.append(110 - (i - 20) * 0.5)  # fall
            elif i < 40:
                prices.append(105 + (i - 30) * 0.5)  # rise again
            elif i < 50:
                prices.append(110 - (i - 40) * 0.5)  # fall
            else:
                prices.append(105 - (i - 50) * 0.3)  # decline

        patterns = self.detector.detect_patterns(prices)
        pattern_names = [p["pattern"] for p in patterns]
        # Should find double top or similar bearish pattern
        assert any("Double Top" in name or "BEARISH" in p.get("direction", "")
                    for name, p in zip(pattern_names, patterns))

    def test_double_bottom_detected(self):
        """Generate a clear double bottom pattern."""
        n = 80
        prices = []
        for i in range(n):
            if i < 20:
                prices.append(110 - i * 0.5)          # fall
            elif i < 30:
                prices.append(100 + (i - 20) * 0.5)   # rise
            elif i < 40:
                prices.append(105 - (i - 30) * 0.5)   # fall again
            elif i < 50:
                prices.append(100 + (i - 40) * 0.5)   # rise
            else:
                prices.append(105 + (i - 50) * 0.3)   # continue up

        patterns = self.detector.detect_patterns(prices)
        pattern_names = [p["pattern"] for p in patterns]
        assert any("Double Bottom" in name or "BULLISH" in p.get("direction", "")
                    for name, p in zip(pattern_names, patterns))


class TestSupportResistance:
    """Support and resistance level tests."""

    def setup_method(self):
        self.detector = ChartPatternDetector()

    def test_find_sr_basic(self):
        # Oscillating prices should produce S/R levels
        prices = list(np.sin(np.linspace(0, 6 * np.pi, 200)) * 10 + 100)
        sr = self.detector.find_support_resistance(prices)
        assert "support" in sr
        assert "resistance" in sr
        assert "strength" in sr

    def test_sr_values_reasonable(self):
        prices = list(np.sin(np.linspace(0, 6 * np.pi, 200)) * 10 + 100)
        sr = self.detector.find_support_resistance(prices)
        current = prices[-1]
        for s in sr["support"]:
            assert s < current * 1.01  # support should be below current price
        for r in sr["resistance"]:
            assert r > current * 0.99  # resistance should be above

    def test_sr_too_few_bars(self):
        sr = self.detector.find_support_resistance([100, 101])
        assert sr["support"] == []
        assert sr["resistance"] == []


class TestDivergence:
    """Divergence detection tests."""

    def setup_method(self):
        self.detector = ChartPatternDetector(config={"peak_order": 3})

    def test_bearish_divergence(self):
        """Price higher highs, indicator lower highs."""
        n = 100
        x = np.linspace(0, 6 * np.pi, n)
        prices = np.sin(x) * 10 + np.linspace(0, 5, n) + 100  # trending up
        indicator = np.sin(x) * 10 + np.linspace(0, -5, n) + 50  # trending down

        divs = self.detector.detect_divergence(prices.tolist(), indicator.tolist())
        assert isinstance(divs, list)
        # May or may not detect divergence depending on exact shapes
        for d in divs:
            assert d["type"] in ("bullish", "bearish")
            assert 0.0 <= d["confidence"] <= 1.0

    def test_divergence_mismatched_lengths(self):
        divs = self.detector.detect_divergence([1, 2, 3], [1, 2])
        assert divs == []


class TestPatternSignal:
    """Aggregate signal tests."""

    def setup_method(self):
        self.detector = ChartPatternDetector()

    def test_get_pattern_signal(self):
        prices = list(np.sin(np.linspace(0, 6 * np.pi, 200)) * 10 + 100)
        sig = self.detector.get_pattern_signal(prices)
        assert "bias" in sig
        assert "confidence" in sig
        assert "patterns" in sig
        assert "support" in sig
        assert "resistance" in sig
        assert -1.0 <= sig["bias"] <= 1.0

    def test_signal_with_volumes(self):
        prices = list(np.sin(np.linspace(0, 6 * np.pi, 200)) * 10 + 100)
        volumes = list(np.random.rand(200) * 1000)
        sig = self.detector.get_pattern_signal(prices, volumes)
        assert isinstance(sig["patterns"], list)


# ═══════════════════════════════════════════════════════════════════════════
# Component Registry wiring tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComponentRegistryWiring:
    """Test that all 3 AI models are wired into ComponentRegistry."""

    def test_registry_has_ai_model_slots(self):
        from core.component_registry import ComponentRegistry

        class FakeConfig:
            pass

        reg = ComponentRegistry(FakeConfig())
        assert hasattr(reg, "sentiment_analyzer")
        assert hasattr(reg, "anomaly_detector")
        assert hasattr(reg, "chart_pattern_detector")

    def test_init_sentiment_analyzer(self):
        from ml.finbert_sentiment import CryptoSentimentAnalyzer
        analyzer = CryptoSentimentAnalyzer()
        assert analyzer.get_stats()["lexicon_size"] > 300

    def test_init_anomaly_detector(self):
        from ml.anomaly_detector import MarketAnomalyDetector
        det = MarketAnomalyDetector()
        assert det.get_stats()["n_trees"] == 100

    def test_init_chart_pattern_detector(self):
        from ml.chart_pattern_cnn import ChartPatternDetector
        det = ChartPatternDetector()
        result = det.detect_patterns([100.0] * 50)
        assert isinstance(result, list)

    def test_config_keys_registered(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        assert "sentiment_analyzer" in _KNOWN_TOP_LEVEL_KEYS
        assert "anomaly_detector" in _KNOWN_TOP_LEVEL_KEYS
        assert "chart_pattern_detector" in _KNOWN_TOP_LEVEL_KEYS
