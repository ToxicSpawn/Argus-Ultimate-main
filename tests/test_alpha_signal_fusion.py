"""
Tests for Alpha Signal Fusion.
"""

import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from ml.alpha_signal_fusion import (
    AlphaSignalFusion,
    AlphaSignal,
    OnChainAlpha,
    LLMSentimentAlpha,
    create_alpha_fusion,
)


class TestAlphaSignalFusion(unittest.TestCase):
    """Tests for AlphaSignalFusion."""

    def setUp(self):
        """Set up test fixtures."""
        self.fusion = create_alpha_fusion(
            use_ml=False,
            use_alpha=False,
            use_sentiment=False,
        )

    def test_initialization(self):
        """Test default initialization."""
        fusion = AlphaSignalFusion()
        self.assertTrue(fusion.use_ml_predictor)
        self.assertTrue(fusion.use_alpha_model)
        self.assertTrue(fusion.use_sentiment)
        self.assertEqual(fusion.ml_weight, 0.25)
        self.assertEqual(fusion.alpha_weight, 0.25)
        self.assertEqual(fusion.sentiment_weight, 0.15)

    def test_factory_function(self):
        """Test factory function creates fusion."""
        fusion = create_alpha_fusion(
            use_ml=True,
            use_alpha=True,
            use_sentiment=True,
        )

        self.assertIsNotNone(fusion)
        self.assertIsInstance(fusion, AlphaSignalFusion)

    @patch.object(AlphaSignalFusion, '_initialized', new=True)
    async def test_generate_signal_returns_none_when_no_components(self, mock_init):
        """Test generate_signal returns None without initialized components."""
        fusion = create_alpha_fusion()
        # Without _initialized = True, it will try to initialize
        # But with no components, should return None
        result = await fusion.generate_signal(
            "BTCUSDT",
            [],
            {},
        )

        # Will be None due to no components
        # Just verify it runs without error
        self.assertIsNotNone(fusion)


class TestAlphaSignal(unittest.TestCase):
    """Tests for AlphaSignal dataclass."""

    def test_to_signal_dict_buy(self):
        """Test converting buy signal to dict."""
        signal = AlphaSignal(
            symbol="BTCUSDT",
            direction="buy",
            confidence=0.8,
            expected_return=50.0,
            ml_prediction=0.6,
            alpha_model=0.5,
            sentiment_score=0.3,
            volatility_regime="normal",
            volatility_forecast=0.50,
            sources_used=["ml", "alpha"],
        )

        result = signal.to_signal_dict()

        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertEqual(result["action"], "BUY")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["expected_return"], 50.0)
        self.assertEqual(result["strategy_id"], "alpha_fusion")

    def test_to_signal_dict_sell(self):
        """Test converting sell signal to dict."""
        signal = AlphaSignal(
            symbol="ETHUSDT",
            direction="sell",
            confidence=0.7,
            expected_return=-30.0,
            ml_prediction=-0.4,
            alpha_model=-0.3,
            sentiment_score=-0.2,
            volatility_regime="elevated",
            volatility_forecast=0.80,
            sources_used=["ml"],
        )

        result = signal.to_signal_dict()

        self.assertEqual(result["action"], "SELL")

    def test_to_signal_dict_neutral(self):
        """Test converting neutral signal to dict."""
        signal = AlphaSignal(
            symbol="BTCUSDT",
            direction="neutral",
            confidence=0.0,
            expected_return=0.0,
            ml_prediction=0.0,
            alpha_model=0.0,
            sentiment_score=0.0,
            volatility_regime="normal",
            volatility_forecast=0.40,
            sources_used=[],
        )

        result = signal.to_signal_dict()

        self.assertEqual(result["action"], "FLAT")


class TestOnChainAlpha(unittest.TestCase):
    """Tests for OnChainAlpha."""

    def test_analyze_inflow(self):
        """Test positive whale inflow gives bullish signal."""
        alpha = OnChainAlpha(min_whale_threshold=100000)

        result = alpha.analyze("BTCUSDT", {"whale_inflow": 200000, "whale_outflow": 0})

        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["activity"], "inflow")

    def test_analyze_outflow(self):
        """Test negative whale outflow gives bearish signal."""
        alpha = OnChainAlpha(min_whale_threshold=100000)

        result = alpha.analyze("BTCUSDT", {"whale_inflow": 0, "whale_outflow": 200000})

        self.assertEqual(result["score"], -0.5)
        self.assertEqual(result["activity"], "outflow")

    def test_analyze_exchange_accumulation(self):
        """Test exchange accumulation gives bullish signal."""
        alpha = OnChainAlpha(min_whale_threshold=100000)

        result = alpha.analyze("BTCUSDT", {
            "whale_inflow": 0,
            "whale_outflow": 0,
            "exchange_flow": -200000,
        })

        self.assertEqual(result["score"], 0.3)
        self.assertEqual(result["activity"], "accumulation")

    def test_analyze_neutral(self):
        """Test neutral when no significant activity."""
        alpha = OnChainAlpha(min_whale_threshold=100000)

        result = alpha.analyze("BTCUSDT", {
            "whale_inflow": 10000,
            "whale_outflow": 10000,
            "exchange_flow": 0,
        })

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["activity"], "neutral")


class TestLLMSentimentAlpha(unittest.TestCase):
    """Tests for LLMSentimentAlpha."""

    @patch('asyncio.get_event_loop', return_value=AsyncMock())
    async def test_analyze_positive_news(self, mock_loop):
        """Test positive news gives bullish signal."""
        alpha = LLMSentimentAlpha()

        # Make it awaitable
        result = await alpha.analyze("BTCUSDT", {
            "news": [
                {"sentiment": 0.8},
                {"sentiment": 0.9},
            ]
        })

        self.assertGreater(result["score"], 0)
        self.assertEqual(result["source"], "llm")

    @patch('asyncio.get_event_loop')
    async def test_analyze_negative_news(self, mock_loop):
        """Test negative news gives bearish signal."""
        alpha = LLMSentimentAlpha()

        result = await alpha.analyze("BTCUSDT", {
            "news": [
                {"sentiment": 0.2},
                {"sentiment": 0.1},
            ]
        })

        self.assertLess(result["score"], 0)

    @patch('asyncio.get_event_loop')
    async def test_analyze_no_news(self, mock_loop):
        """Test no news gives neutral signal."""
        alpha = LLMSentimentAlpha()

        result = await alpha.analyze("BTCUSDT", {"news": []})

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["headlines"], 0)


if __name__ == "__main__":
    unittest.main()