"""Tests for new data source modules (batch — March 2026).

Covers:
  1. data/defi/tvl_tracker.py — TVLTracker
  2. data/etf/bitcoin_etf_flows.py — BitcoinETFFlowTracker
  3. data/mining/hash_rate_tracker.py — HashRateTracker
  4. data/sentiment/reddit_sentiment.py — RedditSentimentTracker
  5. data/onchain/gas_tracker.py — GasTracker
  6. data/derivatives/options_sentiment.py — OptionsSentimentTracker
  7. data/market/stablecoin_dominance.py — StablecoinDominanceTracker

60+ tests — all offline, no network calls.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path."""
    return str(tmp_path / "test.db")


# ===========================================================================
# 1. TVLTracker
# ===========================================================================

class TestTVLTracker:
    """Tests for data.defi.tvl_tracker.TVLTracker."""

    def test_import(self):
        from data.defi.tvl_tracker import TVLTracker, TVLSnapshot
        assert TVLTracker is not None
        assert TVLSnapshot is not None

    def test_tvl_snapshot_defaults(self):
        from data.defi.tvl_tracker import TVLSnapshot
        snap = TVLSnapshot(protocol="aave", tvl_usd=1e9, change_24h_pct=5.0, change_7d_pct=10.0)
        assert snap.protocol == "aave"
        assert snap.tvl_usd == 1e9
        assert snap.timestamp > 0

    def test_fetch_tvl_fallback_on_error(self):
        """When the API is unreachable, fetch_tvl returns a neutral snapshot."""
        from data.defi.tvl_tracker import TVLTracker
        tracker = TVLTracker()
        # Patch aiohttp to raise
        with patch("data.defi.tvl_tracker.TVLTracker._fetch_tvl_from_api", side_effect=Exception("no network")):
            snap = run_async(tracker.fetch_tvl("aave"))
        assert snap.tvl_usd == 0.0
        assert snap.change_24h_pct == 0.0

    def test_tvl_signal_neutral_when_no_data(self):
        from data.defi.tvl_tracker import TVLTracker
        tracker = TVLTracker()
        with patch.object(tracker, "_fetch_tvl_from_api", side_effect=Exception("no net")):
            signal = run_async(tracker.get_tvl_signal("aave"))
        assert signal == 0.0

    def test_tvl_signal_bullish(self):
        """Rising TVL should produce a positive signal."""
        from data.defi.tvl_tracker import TVLTracker, TVLSnapshot
        tracker = TVLTracker()
        # Pre-cache a bullish snapshot
        snap = TVLSnapshot(protocol="aave", tvl_usd=1e9, change_24h_pct=15.0, change_7d_pct=10.0)
        tracker._tvl_cache["aave"] = (snap, time.time())
        signal = run_async(tracker.get_tvl_signal("aave"))
        assert signal > 0.5

    def test_tvl_signal_bearish(self):
        """Falling TVL should produce a negative signal."""
        from data.defi.tvl_tracker import TVLTracker, TVLSnapshot
        tracker = TVLTracker()
        snap = TVLSnapshot(protocol="aave", tvl_usd=1e9, change_24h_pct=-15.0, change_7d_pct=-10.0)
        tracker._tvl_cache["aave"] = (snap, time.time())
        signal = run_async(tracker.get_tvl_signal("aave"))
        assert signal < -0.5

    def test_tvl_signal_clamped(self):
        """Signal should be clamped to [-1, 1]."""
        from data.defi.tvl_tracker import TVLTracker, TVLSnapshot
        tracker = TVLTracker()
        snap = TVLSnapshot(protocol="x", tvl_usd=1e9, change_24h_pct=100.0, change_7d_pct=100.0)
        tracker._tvl_cache["x"] = (snap, time.time())
        signal = run_async(tracker.get_tvl_signal("x"))
        assert signal == 1.0

    def test_cache_hit(self):
        """Second call should use cached data."""
        from data.defi.tvl_tracker import TVLTracker, TVLSnapshot
        tracker = TVLTracker()
        snap = TVLSnapshot(protocol="aave", tvl_usd=5e9, change_24h_pct=1.0, change_7d_pct=2.0)
        tracker._tvl_cache["aave"] = (snap, time.time())
        result = run_async(tracker.fetch_tvl("aave"))
        assert result.tvl_usd == 5e9

    def test_get_top_movers_empty(self):
        from data.defi.tvl_tracker import TVLTracker
        tracker = TVLTracker()
        with patch.object(tracker, "_fetch_protocols_list", new_callable=AsyncMock, return_value=[]):
            movers = run_async(tracker.get_top_movers(5))
        assert movers == []

    def test_get_top_movers_with_data(self):
        from data.defi.tvl_tracker import TVLTracker
        tracker = TVLTracker()
        mock_protocols = [
            {"slug": "aave", "change_1d": 5.0},
            {"slug": "compound", "change_1d": -10.0},
            {"slug": "uniswap", "change_1d": 2.0},
        ]
        with patch.object(tracker, "_fetch_protocols_list", new_callable=AsyncMock, return_value=mock_protocols):
            movers = run_async(tracker.get_top_movers(2))
        assert len(movers) == 2
        assert movers[0][0] == "compound"  # largest absolute change


# ===========================================================================
# 2. BitcoinETFFlowTracker
# ===========================================================================

class TestBitcoinETFFlowTracker:
    """Tests for data.etf.bitcoin_etf_flows.BitcoinETFFlowTracker."""

    def test_import(self):
        from data.etf.bitcoin_etf_flows import BitcoinETFFlowTracker, ETFFlowSummary, KNOWN_ETFS
        assert BitcoinETFFlowTracker is not None
        assert len(KNOWN_ETFS) == 6

    def test_record_and_get_signal(self, tmp_db):
        from data.etf.bitcoin_etf_flows import BitcoinETFFlowTracker
        tracker = BitcoinETFFlowTracker(db_path=tmp_db)
        tracker.record_flow("IBIT", 250_000_000, "2026-03-20")
        tracker.record_flow("GBTC", -80_000_000, "2026-03-20")
        signal = tracker.get_signal()
        assert signal > 0  # net inflow -> bullish

    def test_net_flow_summary(self, tmp_db):
        from data.etf.bitcoin_etf_flows import BitcoinETFFlowTracker
        tracker = BitcoinETFFlowTracker(db_path=tmp_db)
        tracker.record_flow("IBIT", 300e6, "2026-03-20")
        tracker.record_flow("FBTC", 100e6, "2026-03-20")
        tracker.record_flow("GBTC", -50e6, "2026-03-20")
        summary = tracker.get_net_flow(lookback_days=7)
        assert summary.direction == "INFLOW"
        assert summary.net_flow_usd == 350e6
        assert summary.largest_inflow[0] == "IBIT"
        assert summary.largest_outflow[0] == "GBTC"

    def test_empty_db_returns_neutral(self, tmp_db):
        from data.etf.bitcoin_etf_flows import BitcoinETFFlowTracker
        tracker = BitcoinETFFlowTracker(db_path=tmp_db)
        summary = tracker.get_net_flow()
        assert summary.direction == "NEUTRAL"
        assert summary.signal_bias == 0.0

    def test_bearish_signal(self, tmp_db):
        from data.etf.bitcoin_etf_flows import BitcoinETFFlowTracker
        tracker = BitcoinETFFlowTracker(db_path=tmp_db)
        tracker.record_flow("GBTC", -400e6, "2026-03-20")
        signal = tracker.get_signal()
        assert signal < 0  # net outflow -> bearish

    def test_signal_saturation(self, tmp_db):
        from data.etf.bitcoin_etf_flows import BitcoinETFFlowTracker
        tracker = BitcoinETFFlowTracker(db_path=tmp_db)
        tracker.record_flow("IBIT", 1e9, "2026-03-20")
        signal = tracker.get_signal()
        assert signal == 1.0  # saturated at +1

    def test_update_flow_alias(self, tmp_db):
        from data.etf.bitcoin_etf_flows import BitcoinETFFlowTracker
        tracker = BitcoinETFFlowTracker(db_path=tmp_db)
        tracker.update_flow("ARKB", 50e6, "2026-03-20")
        summary = tracker.get_net_flow()
        assert summary.net_flow_usd == 50e6

    def test_upsert_same_etf_date(self, tmp_db):
        """Recording the same ETF+date should replace the previous value."""
        from data.etf.bitcoin_etf_flows import BitcoinETFFlowTracker
        tracker = BitcoinETFFlowTracker(db_path=tmp_db)
        tracker.record_flow("IBIT", 100e6, "2026-03-20")
        tracker.record_flow("IBIT", 200e6, "2026-03-20")  # replace
        summary = tracker.get_net_flow()
        assert summary.net_flow_usd == 200e6


# ===========================================================================
# 3. HashRateTracker
# ===========================================================================

class TestHashRateTracker:
    """Tests for data.mining.hash_rate_tracker.HashRateTracker."""

    def test_import(self):
        from data.mining.hash_rate_tracker import HashRateTracker, MiningStats
        assert HashRateTracker is not None
        assert MiningStats is not None

    def test_mining_stats_defaults(self):
        from data.mining.hash_rate_tracker import MiningStats
        stats = MiningStats(hash_rate=500.0, difficulty=1e13, block_time_avg=600.0, miner_revenue_usd=5e7)
        assert stats.hash_rate == 500.0
        assert stats.timestamp > 0

    def test_fetch_stats_fallback(self):
        from data.mining.hash_rate_tracker import HashRateTracker
        tracker = HashRateTracker()
        with patch.object(tracker, "_fetch_from_api", side_effect=Exception("no net")):
            stats = run_async(tracker.fetch_stats())
        assert stats.hash_rate == 0.0
        assert stats.block_time_avg == 600.0

    def test_signal_neutral_no_history(self):
        from data.mining.hash_rate_tracker import HashRateTracker
        tracker = HashRateTracker()
        assert tracker.get_signal() == 0.0

    def test_signal_bullish_rising_hash_rate(self):
        from data.mining.hash_rate_tracker import HashRateTracker, MiningStats
        tracker = HashRateTracker()
        # Simulate 25 readings: hash rate rising from 500 to 550 (10% rise)
        for i in range(25):
            hr = 500.0 + (i * 2.0)
            tracker._history.append(MiningStats(hash_rate=hr, difficulty=1e13, block_time_avg=600, miner_revenue_usd=5e7))
        signal = tracker.get_signal()
        assert signal > 0  # rising hash rate is bullish

    def test_signal_bearish_falling_hash_rate(self):
        from data.mining.hash_rate_tracker import HashRateTracker, MiningStats
        tracker = HashRateTracker()
        for i in range(25):
            hr = 550.0 - (i * 2.0)
            tracker._history.append(MiningStats(hash_rate=hr, difficulty=1e13, block_time_avg=600, miner_revenue_usd=5e7))
        signal = tracker.get_signal()
        assert signal < 0

    def test_miner_capitulation_false_when_stable(self):
        from data.mining.hash_rate_tracker import HashRateTracker, MiningStats
        tracker = HashRateTracker()
        for _ in range(10):
            tracker._history.append(MiningStats(hash_rate=500.0, difficulty=1e13, block_time_avg=600, miner_revenue_usd=5e7))
        assert tracker.get_miner_capitulation() is False

    def test_miner_capitulation_true_on_drop(self):
        from data.mining.hash_rate_tracker import HashRateTracker, MiningStats
        tracker = HashRateTracker()
        # First reading at 500, last at 440 (12% drop)
        tracker._history.append(MiningStats(hash_rate=500.0, difficulty=1e13, block_time_avg=600, miner_revenue_usd=5e7))
        tracker._history.append(MiningStats(hash_rate=440.0, difficulty=1e13, block_time_avg=600, miner_revenue_usd=5e7))
        assert tracker.get_miner_capitulation() is True

    def test_miner_capitulation_no_history(self):
        from data.mining.hash_rate_tracker import HashRateTracker
        tracker = HashRateTracker()
        assert tracker.get_miner_capitulation() is False


# ===========================================================================
# 4. RedditSentimentTracker
# ===========================================================================

class TestRedditSentimentTracker:
    """Tests for data.sentiment.reddit_sentiment.RedditSentimentTracker."""

    def test_import(self):
        from data.sentiment.reddit_sentiment import RedditSentimentTracker, RedditSentiment
        assert RedditSentimentTracker is not None

    def test_sentiment_fallback_on_error(self):
        from data.sentiment.reddit_sentiment import RedditSentimentTracker
        tracker = RedditSentimentTracker()
        with patch.object(tracker, "_fetch_and_analyse", side_effect=Exception("no net")):
            sentiment = run_async(tracker.get_sentiment("cryptocurrency"))
        assert sentiment.neutral_pct == 100.0
        assert sentiment.signal_bias == 0.0

    def test_bullish_sentiment(self):
        from data.sentiment.reddit_sentiment import RedditSentimentTracker, RedditSentiment
        tracker = RedditSentimentTracker()
        mock_sentiment = RedditSentiment(
            bullish_pct=70.0, bearish_pct=10.0, neutral_pct=20.0,
            post_count=100, top_mentions=[("BTC", 50)], signal_bias=0.6,
        )
        with patch.object(tracker, "_fetch_and_analyse", new_callable=AsyncMock, return_value=mock_sentiment):
            result = run_async(tracker.get_sentiment("cryptocurrency"))
        assert result.bullish_pct == 70.0
        assert result.signal_bias == 0.6

    def test_cache_hit(self):
        from data.sentiment.reddit_sentiment import RedditSentimentTracker, RedditSentiment
        tracker = RedditSentimentTracker()
        mock_sentiment = RedditSentiment(
            bullish_pct=50.0, bearish_pct=30.0, neutral_pct=20.0,
            post_count=50, top_mentions=[], signal_bias=0.2,
        )
        tracker._cache["cryptocurrency"] = (mock_sentiment, time.time())
        result = run_async(tracker.get_sentiment("cryptocurrency"))
        assert result.signal_bias == 0.2

    def test_top_mentioned_coins(self):
        from data.sentiment.reddit_sentiment import RedditSentimentTracker, RedditSentiment
        tracker = RedditSentimentTracker()
        mock_sentiment = RedditSentiment(
            bullish_pct=50.0, bearish_pct=30.0, neutral_pct=20.0,
            post_count=50,
            top_mentions=[("BTC", 30), ("ETH", 20), ("SOL", 10)],
            signal_bias=0.2,
        )
        tracker._cache["cryptocurrency"] = (mock_sentiment, time.time())
        coins = run_async(tracker.get_top_mentioned_coins(n=2))
        assert len(coins) == 2
        assert coins[0] == ("BTC", 30)

    def test_keyword_classification(self):
        """Test that the keyword lists are populated."""
        from data.sentiment.reddit_sentiment import _BULLISH_WORDS, _BEARISH_WORDS
        assert "moon" in _BULLISH_WORDS
        assert "crash" in _BEARISH_WORDS
        assert len(_BULLISH_WORDS) > 10
        assert len(_BEARISH_WORDS) > 10

    def test_coin_patterns(self):
        from data.sentiment.reddit_sentiment import _COIN_PATTERNS
        assert "BTC" in _COIN_PATTERNS
        assert _COIN_PATTERNS["BTC"].search("Bitcoin is great")
        assert _COIN_PATTERNS["ETH"].search("Ethereum pumping")


# ===========================================================================
# 5. GasTracker
# ===========================================================================

class TestGasTracker:
    """Tests for data.onchain.gas_tracker.GasTracker."""

    def test_import(self):
        from data.onchain.gas_tracker import GasTracker, GasStats
        assert GasTracker is not None

    def test_gas_stats_defaults(self):
        from data.onchain.gas_tracker import GasStats
        stats = GasStats(fast_gwei=30.0, standard_gwei=20.0, slow_gwei=10.0, base_fee=15.0)
        assert stats.timestamp > 0

    def test_fetch_gas_default_no_api_key(self):
        from data.onchain.gas_tracker import GasTracker
        tracker = GasTracker(api_key="")
        gas = run_async(tracker.fetch_gas())
        assert gas.fast_gwei == 20.0
        assert gas.standard_gwei == 15.0

    def test_activity_signal_low_gas(self):
        from data.onchain.gas_tracker import GasTracker, GasStats
        tracker = GasTracker()
        tracker._cache = (GasStats(fast_gwei=5.0, standard_gwei=5.0, slow_gwei=3.0, base_fee=3.0), time.time())
        assert tracker.get_activity_signal() == 0.0

    def test_activity_signal_high_gas(self):
        from data.onchain.gas_tracker import GasTracker, GasStats
        tracker = GasTracker()
        tracker._cache = (GasStats(fast_gwei=150.0, standard_gwei=120.0, slow_gwei=80.0, base_fee=100.0), time.time())
        assert tracker.get_activity_signal() == 1.0

    def test_activity_signal_mid_gas(self):
        from data.onchain.gas_tracker import GasTracker, GasStats
        tracker = GasTracker()
        tracker._cache = (GasStats(fast_gwei=70.0, standard_gwei=55.0, slow_gwei=40.0, base_fee=45.0), time.time())
        signal = tracker.get_activity_signal()
        assert 0.3 < signal < 0.7

    def test_is_congested_true(self):
        from data.onchain.gas_tracker import GasTracker, GasStats
        tracker = GasTracker()
        tracker._cache = (GasStats(fast_gwei=100.0, standard_gwei=80.0, slow_gwei=60.0, base_fee=70.0), time.time())
        assert tracker.is_congested(threshold_gwei=50) is True

    def test_is_congested_false(self):
        from data.onchain.gas_tracker import GasTracker, GasStats
        tracker = GasTracker()
        tracker._cache = (GasStats(fast_gwei=20.0, standard_gwei=15.0, slow_gwei=10.0, base_fee=10.0), time.time())
        assert tracker.is_congested(threshold_gwei=50) is False

    def test_is_congested_no_cache(self):
        from data.onchain.gas_tracker import GasTracker
        tracker = GasTracker()
        assert tracker.is_congested() is False

    def test_activity_signal_no_cache(self):
        from data.onchain.gas_tracker import GasTracker
        tracker = GasTracker()
        assert tracker.get_activity_signal() == 0.5


# ===========================================================================
# 6. OptionsSentimentTracker
# ===========================================================================

class TestOptionsSentimentTracker:
    """Tests for data.derivatives.options_sentiment.OptionsSentimentTracker."""

    def test_import(self):
        from data.derivatives.options_sentiment import OptionsSentimentTracker, OptionsSnapshot
        assert OptionsSentimentTracker is not None

    def test_update_and_put_call_ratio(self, tmp_db):
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        tracker.update("BTC", put_volume=15000, call_volume=20000, max_pain_price=68000, iv_30d=0.55)
        ratio = tracker.get_put_call_ratio("BTC")
        assert abs(ratio - 0.75) < 0.01

    def test_put_call_ratio_no_data(self, tmp_db):
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        assert tracker.get_put_call_ratio("BTC") == 1.0

    def test_max_pain(self, tmp_db):
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        tracker.update("BTC", put_volume=10000, call_volume=10000, max_pain_price=65000, iv_30d=0.5)
        assert tracker.get_max_pain("BTC") == 65000.0

    def test_max_pain_no_data(self, tmp_db):
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        assert tracker.get_max_pain("BTC") == 0.0

    def test_iv_percentile_no_data(self, tmp_db):
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        assert tracker.get_iv_percentile("BTC") == 50.0

    def test_iv_percentile_with_data(self, tmp_db):
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        base_ts = time.time() - 86400 * 100  # 100 days ago
        # Insert 100 readings with IV from 0.3 to 0.7
        for i in range(100):
            iv = 0.3 + (i * 0.004)  # 0.3 to 0.696
            tracker.update("BTC", 10000, 10000, 65000, iv, timestamp=base_ts + i * 86400)
        pct = tracker.get_iv_percentile("BTC", lookback_days=365)
        # Current IV is the last one (~0.696), should be near 100th percentile
        assert pct > 90.0

    def test_signal_neutral(self, tmp_db):
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        tracker.update("BTC", put_volume=10000, call_volume=10000, max_pain_price=65000, iv_30d=0.5)
        signal = tracker.get_signal("BTC")
        # With PC=1.0 and single IV reading (50th percentile), should be near 0
        assert -0.3 < signal < 0.3

    def test_signal_bearish_high_calls(self, tmp_db):
        """Many calls + high IV = bearish contrarian signal."""
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        # Low PC ratio + high IV -> greed + expensive premiums -> bearish
        tracker.update("BTC", put_volume=5000, call_volume=20000, max_pain_price=65000, iv_30d=0.9)
        signal = tracker.get_signal("BTC")
        assert signal < 0

    def test_signal_clamped(self, tmp_db):
        from data.derivatives.options_sentiment import OptionsSentimentTracker
        tracker = OptionsSentimentTracker(db_path=tmp_db)
        tracker.update("BTC", put_volume=50000, call_volume=1, max_pain_price=65000, iv_30d=0.01)
        signal = tracker.get_signal("BTC")
        assert -1.0 <= signal <= 1.0


# ===========================================================================
# 7. StablecoinDominanceTracker
# ===========================================================================

class TestStablecoinDominanceTracker:
    """Tests for data.market.stablecoin_dominance.StablecoinDominanceTracker."""

    def test_import(self):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker, DominanceSnapshot
        assert StablecoinDominanceTracker is not None

    def test_update_and_get_dominance(self, tmp_db):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        tracker.update(total_crypto_mcap=2.5e12, stablecoin_mcap=150e9)
        dom = tracker.get_dominance()
        assert abs(dom - 6.0) < 0.01

    def test_get_dominance_no_data(self, tmp_db):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        assert tracker.get_dominance() == 0.0

    def test_signal_neutral_no_data(self, tmp_db):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        assert tracker.get_signal() == 0.0

    def test_signal_bearish_rising_dominance(self, tmp_db):
        """Rising stablecoin dominance -> bearish."""
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        base_ts = time.time() - 86400 * 8
        # Week of falling dominance then a spike
        for i in range(7):
            tracker.update(2.5e12, 125e9, timestamp=base_ts + i * 86400)  # 5%
        # Current: dominance jumps to 8% (above 7d average of 5%)
        tracker.update(2.5e12, 200e9, timestamp=time.time())
        signal = tracker.get_signal()
        assert signal < 0  # rising dominance is bearish

    def test_signal_bullish_falling_dominance(self, tmp_db):
        """Falling stablecoin dominance -> bullish."""
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        base_ts = time.time() - 86400 * 8
        for i in range(7):
            tracker.update(2.5e12, 200e9, timestamp=base_ts + i * 86400)  # 8%
        # Current: dominance drops to 5%
        tracker.update(2.5e12, 125e9, timestamp=time.time())
        signal = tracker.get_signal()
        assert signal > 0  # falling dominance is bullish

    def test_trend_stable(self, tmp_db):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        base_ts = time.time() - 86400 * 30
        for i in range(30):
            tracker.update(2.5e12, 150e9, timestamp=base_ts + i * 86400)
        assert tracker.get_trend() == "stable"

    def test_trend_rising(self, tmp_db):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        base_ts = time.time() - 86400 * 30
        for i in range(30):
            # Dominance rising from 5% to 8%
            mcap_stable = 150e9 + i * 3.33e9
            tracker.update(2.5e12, mcap_stable, timestamp=base_ts + i * 86400)
        assert tracker.get_trend() == "rising"

    def test_trend_falling(self, tmp_db):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        base_ts = time.time() - 86400 * 30
        for i in range(30):
            # Dominance falling from 8% to 5%
            mcap_stable = 200e9 - i * 3.33e9
            tracker.update(2.5e12, mcap_stable, timestamp=base_ts + i * 86400)
        assert tracker.get_trend() == "falling"

    def test_trend_insufficient_data(self, tmp_db):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        tracker.update(2.5e12, 150e9)
        assert tracker.get_trend() == "stable"

    def test_update_rejects_zero_mcap(self, tmp_db):
        """Zero total mcap should be rejected."""
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        tracker.update(total_crypto_mcap=0, stablecoin_mcap=100e9)
        assert tracker.get_dominance() == 0.0  # nothing recorded

    def test_signal_clamped(self, tmp_db):
        from data.market.stablecoin_dominance import StablecoinDominanceTracker
        tracker = StablecoinDominanceTracker(db_path=tmp_db)
        signal = tracker.get_signal()
        assert -1.0 <= signal <= 1.0
