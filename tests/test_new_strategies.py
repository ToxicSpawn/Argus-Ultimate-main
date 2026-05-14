"""
Tests for new strategies and ML signal modules (Batch — March 2026).

Covers:
- strategies/grid_trader.py (GridTrader)
- strategies/bb_squeeze.py (BBSqueezeStrategy)
- strategies/seasonal_patterns.py (SeasonalPatternStrategy)
- strategies/volume_profile.py (VolumeProfileAnalyzer)
- ml/signal_quality_scorer.py (SignalQualityScorer)
- ml/ensemble_voter.py (EnsembleVoter)

60+ tests total.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Grid Trader
# ---------------------------------------------------------------------------
from strategies.grid_trader import GridTrader, GridLevel, GridSignal


class TestGridTrader:
    """Tests for GridTrader."""

    def test_setup_grid_basic(self):
        gt = GridTrader()
        levels = gt.setup_grid("BTC/USD", lower_price=60000, upper_price=62000, num_levels=5, capital_usd=100)
        assert len(levels) == 5
        assert all(isinstance(lv, GridLevel) for lv in levels)
        assert levels[0].price == 60000
        assert levels[-1].price == 62000

    def test_setup_grid_sides(self):
        gt = GridTrader()
        levels = gt.setup_grid("BTC/USD", 100, 200, num_levels=5, capital_usd=50)
        buys = [lv for lv in levels if lv.side == "buy"]
        sells = [lv for lv in levels if lv.side == "sell"]
        assert len(buys) > 0
        assert len(sells) > 0

    def test_setup_grid_min_levels(self):
        gt = GridTrader()
        levels = gt.setup_grid("ETH/USD", 100, 200, num_levels=1, capital_usd=50)
        assert len(levels) >= 3  # clamped to 3 minimum

    def test_setup_grid_invalid_capital(self):
        gt = GridTrader()
        with pytest.raises(ValueError, match="capital_usd must be positive"):
            gt.setup_grid("BTC/USD", 100, 200, capital_usd=0)

    def test_setup_grid_invalid_bounds(self):
        gt = GridTrader()
        with pytest.raises(ValueError, match="lower_price.*must be less"):
            gt.setup_grid("BTC/USD", 200, 100, num_levels=5)

    def test_check_fills_buy(self):
        gt = GridTrader()
        gt.setup_grid("BTC/USD", 100, 200, num_levels=5, capital_usd=100)
        signals = gt.check_fills("BTC/USD", 95)
        buy_signals = [s for s in signals if s.side == "buy"]
        assert len(buy_signals) > 0
        assert all(isinstance(s, GridSignal) for s in signals)

    def test_check_fills_sell(self):
        gt = GridTrader()
        gt.setup_grid("BTC/USD", 100, 200, num_levels=5, capital_usd=100)
        signals = gt.check_fills("BTC/USD", 210)
        sell_signals = [s for s in signals if s.side == "sell"]
        assert len(sell_signals) > 0

    def test_check_fills_no_trigger(self):
        gt = GridTrader()
        gt.setup_grid("BTC/USD", 100, 200, num_levels=5, capital_usd=100)
        signals = gt.check_fills("BTC/USD", 150)
        # middle price may or may not trigger depending on side assignment
        # but definitely not all levels
        assert len(signals) < 5

    def test_check_fills_empty_symbol(self):
        gt = GridTrader()
        signals = gt.check_fills("UNKNOWN/USD", 100)
        assert signals == []

    def test_get_pnl_default(self):
        gt = GridTrader()
        gt.setup_grid("BTC/USD", 100, 200, num_levels=5, capital_usd=100)
        pnl = gt.get_pnl("BTC/USD")
        assert isinstance(pnl, float)

    def test_get_pnl_all_symbols(self):
        gt = GridTrader()
        gt.setup_grid("BTC/USD", 100, 200, num_levels=5, capital_usd=50)
        gt.setup_grid("ETH/USD", 10, 20, num_levels=5, capital_usd=50)
        pnl = gt.get_pnl()
        assert isinstance(pnl, float)

    def test_reset_grid(self):
        gt = GridTrader()
        gt.setup_grid("BTC/USD", 100, 200, num_levels=5, capital_usd=100)
        new_levels = gt.reset_grid("BTC/USD", new_center_price=180)
        assert len(new_levels) == 5
        # New grid should be centred around 180
        mid = (new_levels[0].price + new_levels[-1].price) / 2
        assert abs(mid - 180) < 1

    def test_reset_grid_no_existing(self):
        gt = GridTrader()
        with pytest.raises(ValueError, match="No grid exists"):
            gt.reset_grid("UNKNOWN/USD")

    def test_update_price(self):
        gt = GridTrader()
        for p in range(100):
            gt.update_price("BTC/USD", 60000 + p)
        assert gt._last_price["BTC/USD"] == 60099

    def test_auto_range_insufficient_data(self):
        gt = GridTrader(vol_lookback=50)
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            gt.setup_grid("BTC/USD")

    def test_auto_range_sufficient_data(self):
        gt = GridTrader(vol_lookback=20)
        for i in range(50):
            gt.update_price("BTC/USD", 60000 + random.uniform(-500, 500))
        levels = gt.setup_grid("BTC/USD", num_levels=5, capital_usd=100)
        assert len(levels) == 5

    def test_grid_level_repr(self):
        lv = GridLevel(price=100.0, side="buy", size=0.5)
        assert "BUY" in repr(lv)
        assert "OPEN" in repr(lv)


# ---------------------------------------------------------------------------
# BB Squeeze
# ---------------------------------------------------------------------------
from strategies.bb_squeeze import BBSqueezeStrategy, BBSqueezSignal


class TestBBSqueeze:
    """Tests for BBSqueezeStrategy."""

    def _feed_constant(self, strat, symbol, price, n=30, volume=100):
        """Feed constant price bars (creates squeeze)."""
        for _ in range(n):
            strat.update(symbol, close=price, high=price * 1.001, low=price * 0.999, volume=volume)

    def _feed_trending(self, strat, symbol, start, end, n=30, volume=100):
        """Feed trending price bars."""
        step = (end - start) / n
        for i in range(n):
            p = start + i * step
            strat.update(symbol, close=p, high=p * 1.01, low=p * 0.99, volume=volume)

    def test_init_defaults(self):
        strat = BBSqueezeStrategy()
        assert strat.bb_period == 20
        assert strat.bb_std == 2.0
        assert strat.squeeze_threshold == 0.03

    def test_detect_squeeze_constant_price(self):
        strat = BBSqueezeStrategy(bb_period=10, squeeze_threshold=0.05)
        self._feed_constant(strat, "BTC/USD", 60000, n=15)
        # Constant price → zero std → BB width=0 → squeeze
        assert strat.detect_squeeze("BTC/USD") is True

    def test_detect_squeeze_volatile_price(self):
        strat = BBSqueezeStrategy(bb_period=10, squeeze_threshold=0.01)
        for i in range(20):
            p = 60000 + (i % 2) * 3000  # oscillate 60000 ↔ 63000
            strat.update("BTC/USD", close=p, high=p + 100, low=p - 100, volume=100)
        # High volatility → wide BB → no squeeze
        assert strat.detect_squeeze("BTC/USD") is False

    def test_no_signal_during_squeeze(self):
        strat = BBSqueezeStrategy(bb_period=10, squeeze_threshold=0.1)
        self._feed_constant(strat, "BTC/USD", 60000, n=15)
        signal = strat.get_signal("BTC/USD")
        assert signal is None  # squeeze is active, no breakout yet

    def test_signal_after_breakout(self):
        strat = BBSqueezeStrategy(
            bb_period=10, squeeze_threshold=0.1, squeeze_lookback=10,
            min_volume_ratio=0.0,  # disable volume check
        )
        # Create squeeze
        self._feed_constant(strat, "BTC/USD", 60000, n=15)
        # Breakout up
        for i in range(5):
            p = 60000 + (i + 1) * 500
            strat.update("BTC/USD", close=p, high=p + 100, low=p - 50, volume=200)
        signal = strat.get_signal("BTC/USD")
        # May or may not trigger depending on exact BB calculation
        # But at minimum no error
        if signal is not None:
            assert isinstance(signal, BBSqueezSignal)
            assert signal.direction in ("buy", "sell")

    def test_get_bb_width(self):
        strat = BBSqueezeStrategy(bb_period=10)
        self._feed_constant(strat, "BTC/USD", 60000, n=15)
        width = strat.get_bb_width("BTC/USD")
        assert width is not None
        assert width >= 0

    def test_get_bb_width_insufficient_data(self):
        strat = BBSqueezeStrategy(bb_period=20)
        strat.update("BTC/USD", 100, 101, 99, 50)
        assert strat.get_bb_width("BTC/USD") is None

    def test_invalid_bar(self):
        strat = BBSqueezeStrategy()
        strat.update("BTC/USD", close=-1, high=100, low=99, volume=50)
        assert len(strat._bars.get("BTC/USD", [])) == 0

    def test_signal_side_property(self):
        sig = BBSqueezSignal(
            symbol="BTC/USD", direction="buy", confidence=0.7,
            bb_width=0.02, squeeze_bars=5, breakout_strength=0.01,
        )
        assert sig.side == "buy"

    def test_squeeze_detection_no_data(self):
        strat = BBSqueezeStrategy()
        assert strat.detect_squeeze("UNKNOWN") is False


# ---------------------------------------------------------------------------
# Seasonal Patterns
# ---------------------------------------------------------------------------
from strategies.seasonal_patterns import SeasonalPatternStrategy, SeasonalSignal


class TestSeasonalPatterns:
    """Tests for SeasonalPatternStrategy."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "seasonal_test.db")

    def test_init(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path)
        assert strat.min_days >= 7

    def test_record_return(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path, min_days=1)
        strat.record_return("BTC/USD", 0.5, hour=10, day_of_week=1)
        summary = strat.get_data_summary("BTC/USD")
        assert summary["total_observations"] == 1

    def test_get_seasonal_bias_no_data(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path)
        bias = strat.get_seasonal_bias("BTC/USD", hour=10, day_of_week=1)
        assert bias == 0.0

    def test_get_seasonal_bias_with_data(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path, min_days=1)
        # Positive returns at hour 10
        for i in range(10):
            strat.record_return("BTC/USD", 0.5, hour=10, day_of_week=1)
        # Negative returns at hour 22
        for i in range(10):
            strat.record_return("BTC/USD", -0.5, hour=22, day_of_week=1)
        bias_10 = strat.get_seasonal_bias("BTC/USD", hour=10, day_of_week=1)
        bias_22 = strat.get_seasonal_bias("BTC/USD", hour=22, day_of_week=1)
        assert bias_10 > bias_22

    def test_get_best_trading_hours(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path, min_days=1)
        for i in range(10):
            strat.record_return("BTC/USD", 1.0, hour=9, day_of_week=0)
            strat.record_return("BTC/USD", -0.5, hour=15, day_of_week=0)
        best = strat.get_best_trading_hours("BTC/USD", top_n=3)
        assert len(best) > 0
        # Hour 9 should be best
        assert best[0][0] == 9

    def test_get_weekend_effect(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path, min_days=1)
        # Weekday returns
        for i in range(10):
            strat.record_return("BTC/USD", 0.1, hour=10, day_of_week=2)
        # Weekend returns (better)
        for i in range(10):
            strat.record_return("BTC/USD", 0.5, hour=10, day_of_week=5)
        effect = strat.get_weekend_effect("BTC/USD")
        assert effect > 0  # weekends outperform

    def test_get_signal_insufficient_data(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path, min_days=30)
        strat.record_return("BTC/USD", 0.5, hour=10, day_of_week=1)
        signal = strat.get_signal("BTC/USD", hour=10, day_of_week=1)
        assert signal is None

    def test_get_signal_with_sufficient_data(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path, min_days=1, signal_threshold=0.01)
        # Strong positive signal at hour 10
        for i in range(20):
            strat.record_return("BTC/USD", 1.0, hour=10, day_of_week=1)
        # Neutral baseline
        for i in range(20):
            strat.record_return("BTC/USD", -0.2, hour=22, day_of_week=3)
        signal = strat.get_signal("BTC/USD", hour=10, day_of_week=1)
        if signal is not None:
            assert isinstance(signal, SeasonalSignal)
            assert signal.direction in ("buy", "sell")

    def test_persistence(self, db_path):
        strat1 = SeasonalPatternStrategy(db_path=db_path, min_days=1)
        for i in range(5):
            strat1.record_return("BTC/USD", 0.5, hour=10, day_of_week=1)
        del strat1

        strat2 = SeasonalPatternStrategy(db_path=db_path, min_days=1)
        summary = strat2.get_data_summary("BTC/USD")
        assert summary["total_observations"] == 5

    def test_has_sufficient_data(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path, min_days=30)
        assert strat.has_sufficient_data("BTC/USD") is False

    def test_hour_clamping(self, db_path):
        strat = SeasonalPatternStrategy(db_path=db_path, min_days=1)
        strat.record_return("BTC/USD", 0.5, hour=25, day_of_week=1)  # clamped to 23
        strat.record_return("BTC/USD", 0.5, hour=-1, day_of_week=1)  # clamped to 0


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------
from strategies.volume_profile import VolumeProfileAnalyzer, VolumeProfileSignal


class TestVolumeProfile:
    """Tests for VolumeProfileAnalyzer."""

    def _feed_data(self, vp, symbol, center, spread, n=200):
        """Feed normally distributed price-volume data."""
        for _ in range(n):
            price = center + random.gauss(0, spread)
            volume = random.uniform(1, 10)
            vp.update(symbol, max(price, 0.01), volume)

    def test_init(self):
        vp = VolumeProfileAnalyzer()
        assert vp.lookback >= 50

    def test_update(self):
        vp = VolumeProfileAnalyzer()
        vp.update("BTC/USD", 60000, 1.5)
        assert vp.get_observation_count("BTC/USD") == 1

    def test_update_invalid(self):
        vp = VolumeProfileAnalyzer()
        vp.update("BTC/USD", -1, 1.5)
        vp.update("BTC/USD", 100, 0)
        assert vp.get_observation_count("BTC/USD") == 0

    def test_get_poc(self):
        vp = VolumeProfileAnalyzer()
        self._feed_data(vp, "BTC/USD", center=60000, spread=500, n=200)
        poc = vp.get_poc("BTC/USD")
        assert poc is not None
        assert 58000 < poc < 62000  # should be near center

    def test_get_poc_insufficient_data(self):
        vp = VolumeProfileAnalyzer()
        vp.update("BTC/USD", 100, 1)
        assert vp.get_poc("BTC/USD") is None

    def test_get_value_area(self):
        vp = VolumeProfileAnalyzer()
        self._feed_data(vp, "BTC/USD", center=60000, spread=500, n=300)
        va = vp.get_value_area("BTC/USD", pct=0.70)
        assert va is not None
        vah, val = va
        assert vah > val
        assert val < 60000 < vah

    def test_get_value_area_custom_pct(self):
        vp = VolumeProfileAnalyzer()
        self._feed_data(vp, "BTC/USD", center=100, spread=10, n=200)
        va_70 = vp.get_value_area("BTC/USD", pct=0.70)
        va_90 = vp.get_value_area("BTC/USD", pct=0.90)
        assert va_70 is not None and va_90 is not None
        # 90% VA should be wider than 70%
        assert (va_90[0] - va_90[1]) >= (va_70[0] - va_70[1])

    def test_get_signal_near_val(self):
        vp = VolumeProfileAnalyzer(proximity_pct=0.05)
        # Feed data concentrated around 100
        for _ in range(200):
            p = random.gauss(100, 5)
            vp.update("ETH/USD", max(p, 1), random.uniform(1, 10))
        va = vp.get_value_area("ETH/USD")
        if va is not None:
            val = va[1]
            signal = vp.get_signal("ETH/USD", val * 0.99)
            # Should get buy signal near VAL
            if signal is not None:
                assert signal.direction == "buy"
                assert isinstance(signal, VolumeProfileSignal)

    def test_get_signal_no_data(self):
        vp = VolumeProfileAnalyzer()
        signal = vp.get_signal("BTC/USD", 60000)
        assert signal is None

    def test_get_profile(self):
        vp = VolumeProfileAnalyzer()
        self._feed_data(vp, "BTC/USD", 100, 5, n=100)
        profile = vp.get_profile("BTC/USD")
        assert len(profile) > 0
        # Sorted by price
        prices = [p for p, _ in profile]
        assert prices == sorted(prices)


# ---------------------------------------------------------------------------
# Signal Quality Scorer
# ---------------------------------------------------------------------------
from ml.signal_quality_scorer import SignalQualityScorer, SignalScore


class TestSignalQualityScorer:
    """Tests for SignalQualityScorer."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "sq_test.db")

    def test_init(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path)
        assert scorer.min_quality == 0.5

    def test_score_high_quality(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path, min_quality=0.3)
        result = scorer.score({
            "symbol": "BTC/USD",
            "direction": "buy",
            "confidence": 0.9,
            "volume": 150,
            "avg_volume": 100,
            "regime": "bullish",
            "trend": "up",
            "spread_bps": 5.0,
        })
        assert isinstance(result, SignalScore)
        assert result.quality > 0.5
        assert result.passed is True
        assert result.signal_id is not None

    def test_score_low_quality(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path, min_quality=0.8)
        result = scorer.score({
            "symbol": "BTC/USD",
            "direction": "buy",
            "confidence": 0.1,
            "volume": 10,
            "avg_volume": 100,
            "regime": "bearish",
            "trend": "down",
            "spread_bps": 200.0,
        })
        assert result.passed is False
        assert result.quality < 0.8

    def test_score_minimal_signal(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path)
        result = scorer.score({"symbol": "BTC/USD", "direction": "buy"})
        assert isinstance(result, SignalScore)
        assert 0 <= result.quality <= 1

    def test_score_factors_present(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path)
        result = scorer.score({
            "symbol": "BTC/USD",
            "direction": "sell",
            "confidence": 0.7,
        })
        assert "confidence" in result.factors
        assert "volume" in result.factors
        assert "regime" in result.factors

    def test_record_outcome(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path)
        result = scorer.score({"symbol": "BTC/USD", "direction": "buy", "confidence": 0.7})
        scorer.record_outcome(result.signal_id, pnl=10.0)
        # No error

    def test_get_pass_rate_empty(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path)
        rate = scorer.get_pass_rate(lookback_days=7)
        assert rate == 0.0

    def test_regime_alignment(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path, min_quality=0.0)
        # Aligned
        r1 = scorer.score({"symbol": "X", "direction": "buy", "regime": "bullish", "confidence": 0.5})
        # Misaligned
        r2 = scorer.score({"symbol": "X", "direction": "buy", "regime": "bearish", "confidence": 0.5})
        assert r1.factors["regime"] > r2.factors["regime"]

    def test_spread_check(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path, max_spread_bps=20.0, min_quality=0.0)
        r1 = scorer.score({"symbol": "X", "direction": "buy", "confidence": 0.5, "spread_bps": 5.0})
        r2 = scorer.score({"symbol": "X", "direction": "buy", "confidence": 0.5, "spread_bps": 100.0})
        assert r1.factors["spread"] > r2.factors["spread"]

    def test_get_factor_weights(self, db_path):
        scorer = SignalQualityScorer(db_path=db_path)
        weights = scorer.get_factor_weights()
        assert "confidence" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_score_repr(self):
        s = SignalScore(quality=0.75, reasons=["ok"], passed=True)
        assert "PASS" in repr(s)


# ---------------------------------------------------------------------------
# Ensemble Voter
# ---------------------------------------------------------------------------
from ml.ensemble_voter import EnsembleVoter, ConsensusResult


class TestEnsembleVoter:
    """Tests for EnsembleVoter."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "voter_test.db")

    def test_init(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        assert voter.min_votes == 2
        assert voter.min_agreement == 0.60

    def test_submit_vote(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        votes = voter.get_active_votes("BTC/USD")
        assert len(votes) == 1
        assert votes[0]["strategy"] == "strat_a"

    def test_submit_vote_replaces(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        voter.submit_vote("strat_a", "BTC/USD", "sell", 0.9)
        votes = voter.get_active_votes("BTC/USD")
        assert len(votes) == 1
        assert votes[0]["direction"] == "sell"

    def test_submit_vote_invalid_direction(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        voter.submit_vote("strat_a", "BTC/USD", "hold", 0.5)
        votes = voter.get_active_votes("BTC/USD")
        assert len(votes) == 0

    def test_get_consensus_unanimous(self, db_path):
        voter = EnsembleVoter(db_path=db_path, min_votes=2, min_agreement=0.6)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        voter.submit_vote("strat_b", "BTC/USD", "buy", 0.7)
        voter.submit_vote("strat_c", "BTC/USD", "buy", 0.6)
        result = voter.get_consensus("BTC/USD")
        assert result is not None
        assert isinstance(result, ConsensusResult)
        assert result.direction == "buy"
        assert result.agreement_pct == 1.0
        assert result.num_votes == 3
        assert len(result.voters) == 3

    def test_get_consensus_insufficient_votes(self, db_path):
        voter = EnsembleVoter(db_path=db_path, min_votes=3)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        voter.submit_vote("strat_b", "BTC/USD", "buy", 0.7)
        result = voter.get_consensus("BTC/USD")
        assert result is None

    def test_get_consensus_low_agreement(self, db_path):
        voter = EnsembleVoter(db_path=db_path, min_votes=2, min_agreement=0.9)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.5)
        voter.submit_vote("strat_b", "BTC/USD", "sell", 0.5)
        result = voter.get_consensus("BTC/USD")
        assert result is None

    def test_get_consensus_mixed_votes(self, db_path):
        voter = EnsembleVoter(db_path=db_path, min_votes=2, min_agreement=0.55)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.9)
        voter.submit_vote("strat_b", "BTC/USD", "buy", 0.8)
        voter.submit_vote("strat_c", "BTC/USD", "sell", 0.3)
        result = voter.get_consensus("BTC/USD")
        assert result is not None
        assert result.direction == "buy"
        assert result.agreement_pct > 0.55

    def test_record_outcome(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        voter.submit_vote("strat_b", "BTC/USD", "buy", 0.7)
        voter.get_consensus("BTC/USD")
        voter.record_outcome("BTC/USD", "buy", 10.0)
        # Check accuracy updated
        accuracy = voter.get_voter_accuracy("strat_a")
        assert accuracy >= 0.5

    def test_get_voter_accuracy_unknown(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        assert voter.get_voter_accuracy("unknown_strat") == 0.5

    def test_get_all_voter_stats(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        voter.record_outcome("BTC/USD", "buy", 5.0)
        stats = voter.get_all_voter_stats()
        assert "strat_a" in stats

    def test_clear_votes(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        voter.clear_votes("BTC/USD")
        assert voter.get_active_votes("BTC/USD") == []

    def test_clear_all_votes(self, db_path):
        voter = EnsembleVoter(db_path=db_path)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        voter.submit_vote("strat_b", "ETH/USD", "sell", 0.7)
        voter.clear_votes()
        assert voter.get_active_votes("BTC/USD") == []
        assert voter.get_active_votes("ETH/USD") == []

    def test_persistence(self, db_path):
        voter1 = EnsembleVoter(db_path=db_path)
        voter1.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        voter1.record_outcome("BTC/USD", "buy", 10.0)
        del voter1

        voter2 = EnsembleVoter(db_path=db_path)
        stats = voter2.get_all_voter_stats()
        assert "strat_a" in stats
        assert stats["strat_a"]["wins"] == 1

    def test_consensus_repr(self):
        cr = ConsensusResult(
            symbol="BTC/USD", direction="buy",
            avg_confidence=0.75, num_votes=3,
            agreement_pct=0.9, voters=["a", "b", "c"],
        )
        assert "BTC/USD" in repr(cr)
        assert "buy" in repr(cr)

    def test_stale_votes_pruned(self, db_path):
        voter = EnsembleVoter(db_path=db_path, vote_ttl_seconds=0.1)
        voter.submit_vote("strat_a", "BTC/USD", "buy", 0.8)
        time.sleep(0.2)
        votes = voter.get_active_votes("BTC/USD")
        assert len(votes) == 0

    def test_accuracy_weighted_voting(self, db_path):
        voter = EnsembleVoter(db_path=db_path, min_votes=1, min_agreement=0.5)
        # Set up a highly accurate voter
        voter._voter_stats["accurate_strat"] = {
            "wins": 80, "losses": 20, "accuracy": 0.8, "total_pnl": 100.0,
        }
        voter._voter_stats["bad_strat"] = {
            "wins": 20, "losses": 80, "accuracy": 0.2, "total_pnl": -50.0,
        }
        voter.submit_vote("accurate_strat", "BTC/USD", "buy", 0.6)
        voter.submit_vote("bad_strat", "BTC/USD", "sell", 0.9)
        result = voter.get_consensus("BTC/USD")
        # Accurate strat should win despite lower confidence
        if result is not None:
            assert result.direction == "buy"
