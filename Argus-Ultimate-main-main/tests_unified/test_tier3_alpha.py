"""
test_tier3_alpha.py — Test suite for Tier 3 Microstructure Alpha modules.

Covers:
  - LiveOFIStream: OFI computation, z-score, balanced trades
  - LiveVPINStream: bucket accumulation, alert threshold
  - DeepLOBLiveBridge: graceful no-model, feature extraction
  - EventCalendar: blackout window, next_event
  - MicropriceDriftSignal: bullish drift, quote skew
"""
from __future__ import annotations

import math
import sys
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha.microstructure.live_ofi_stream import LiveOFIStream, OFISignal
from alpha.microstructure.live_vpin_stream import LiveVPINStream
from alpha.microstructure.deeplob_live_bridge import (
    DeepLOBLiveBridge,
    DeepLOBPrediction,
    _LOBFeatureExtractor,
    FEATURE_DIM,
    N_LEVELS,
)
from infra.event_calendar import EventCalendar, EventRecord
from alpha.microstructure.microprice_drift import MicropriceDriftSignal, DriftState


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_bids(n: int = 10, best: float = 30000.0, step: float = 1.0) -> list:
    """Generate n bid levels descending from best."""
    return [(best - i * step, 1.0 + i * 0.1) for i in range(n)]


def _make_asks(n: int = 10, best: float = 30001.0, step: float = 1.0) -> list:
    """Generate n ask levels ascending from best."""
    return [(best + i * step, 1.0 + i * 0.1) for i in range(n)]


# ─── LiveOFIStream tests ──────────────────────────────────────────────────────


class TestLiveOFIStream:

    def test_ofi_stream_buy_heavy(self):
        """10 buy trades should produce OFI > 0."""
        stream = LiveOFIStream(window_trades=100, alpha=0.94)
        ts = time.time_ns()
        for i in range(10):
            stream.on_trade("BTCUSDT", "buy", 1.0, 30000.0 + i, ts + i)

        ofi = stream.get_ofi("BTCUSDT")
        assert ofi > 0.0, f"Expected OFI > 0 for all-buy trades, got {ofi}"
        assert -1.0 <= ofi <= 1.0, f"OFI out of range [-1, 1]: {ofi}"

    def test_ofi_stream_sell_heavy(self):
        """10 sell trades should produce OFI < 0."""
        stream = LiveOFIStream(window_trades=100, alpha=0.94)
        ts = time.time_ns()
        for i in range(10):
            stream.on_trade("BTCUSDT", "sell", 1.0, 29999.0 - i, ts + i)

        ofi = stream.get_ofi("BTCUSDT")
        assert ofi < 0.0, f"Expected OFI < 0 for all-sell trades, got {ofi}"

    def test_ofi_stream_balanced(self):
        """Equal buys and sells → OFI ≈ 0."""
        stream = LiveOFIStream(window_trades=100, alpha=1.0)  # alpha=1.0 = no decay
        ts = time.time_ns()
        # 5 buys and 5 sells of equal size
        for i in range(5):
            stream.on_trade("BTCUSDT", "buy", 1.0, 30000.0, ts + i * 2)
            stream.on_trade("BTCUSDT", "sell", 1.0, 30000.0, ts + i * 2 + 1)

        ofi = stream.get_ofi("BTCUSDT")
        assert abs(ofi) < 0.05, f"Expected OFI ≈ 0 for balanced trades, got {ofi}"

    def test_ofi_zscore_computed(self):
        """Z-score should be computable without error after enough samples."""
        stream = LiveOFIStream(window_trades=200, alpha=0.94)
        ts = time.time_ns()
        # Inject 120 trades alternating sides to build z-score history
        for i in range(120):
            side = "buy" if i % 3 != 0 else "sell"
            stream.on_trade("ETHUSDT", side, 0.5, 2000.0 + i * 0.01, ts + i * 1000)

        zscore = stream.get_ofi_zscore("ETHUSDT", lookback=100)
        assert isinstance(zscore, float), f"Expected float z-score, got {type(zscore)}"
        assert not math.isnan(zscore), "Z-score should not be NaN"
        assert not math.isinf(zscore), "Z-score should not be inf"

    def test_ofi_signal_dataclass_fields(self):
        """get_signal() should return an OFISignal with all required fields."""
        stream = LiveOFIStream()
        ts = time.time_ns()
        stream.on_trade("BTCUSDT", "buy", 0.1, 30000.0, ts)
        signal = stream.get_signal("BTCUSDT")

        assert isinstance(signal, OFISignal)
        assert hasattr(signal, "ofi")
        assert hasattr(signal, "ofi_zscore")
        assert hasattr(signal, "signed_imbalance")
        assert hasattr(signal, "aggressive_ratio")
        assert hasattr(signal, "window_trades")
        assert hasattr(signal, "timestamp_ns")
        assert signal.window_trades == 1

    def test_classify_aggressor_buy(self):
        """Trade at ask price should be classified as buy aggressor."""
        result = LiveOFIStream.classify_aggressor(30001.0, 30000.0, 30001.0)
        assert result == "buy"

    def test_classify_aggressor_sell(self):
        """Trade at bid price should be classified as sell aggressor."""
        result = LiveOFIStream.classify_aggressor(30000.0, 30000.0, 30001.0)
        assert result == "sell"

    def test_classify_aggressor_unknown_no_lob(self):
        """No LOB data → unknown aggressor."""
        result = LiveOFIStream.classify_aggressor(30000.5, 0.0, 0.0)
        assert result == "unknown"

    def test_signed_imbalance_buy_heavy(self):
        """All buys → signed_imbalance close to 1.0."""
        stream = LiveOFIStream(alpha=1.0)
        ts = time.time_ns()
        for i in range(20):
            stream.on_trade("X", "buy", 1.0, 100.0, ts + i)
        imbalance = stream.get_signed_trade_imbalance("X")
        assert imbalance > 0.9, f"Expected imbalance near 1.0, got {imbalance}"


# ─── LiveVPINStream tests ─────────────────────────────────────────────────────


class TestLiveVPINStream:

    def test_vpin_stream_accumulates(self):
        """Trades should fill buckets and produce a non-zero VPIN."""
        stream = LiveVPINStream(bucket_volume=1.0, n_buckets=10)
        ts = time.time_ns()
        # 20 buys of 0.5 each = 10 buckets completed
        for i in range(20):
            stream.on_trade("BTCUSDT", "buy", 0.5, 30000.0, ts + i * 1000)

        stats = stream.get_stats("BTCUSDT")
        assert stats["bucket_count"] >= 5, (
            f"Expected at least 5 completed buckets, got {stats['bucket_count']}"
        )

    def test_vpin_range(self):
        """VPIN should always be in [0, 1]."""
        stream = LiveVPINStream(bucket_volume=0.5, n_buckets=20)
        ts = time.time_ns()
        for i in range(50):
            side = "buy" if i < 40 else "sell"
            stream.on_trade("BTCUSDT", side, 0.3, 30000.0, ts + i * 1000)

        vpin = stream.get_vpin("BTCUSDT")
        assert 0.0 <= vpin <= 1.0, f"VPIN {vpin} out of range [0, 1]"

    def test_vpin_alert_threshold(self):
        """High imbalance (all buys) should trigger VPIN alert."""
        stream = LiveVPINStream(bucket_volume=0.1, n_buckets=10)
        ts = time.time_ns()
        spike_fired = []

        def on_spike(symbol, vpin, threshold):
            spike_fired.append((symbol, vpin, threshold))

        stream.register_spike_callback(on_spike)

        # Inject highly imbalanced trades (all buy)
        for i in range(100):
            stream.on_trade("BTCUSDT", "buy", 0.1, 30000.0, ts + i * 100)

        vpin = stream.get_vpin("BTCUSDT")
        # VPIN for all-one-sided should be 1.0
        assert vpin > 0.7, f"Expected VPIN > 0.7 for highly imbalanced flow, got {vpin}"
        assert stream.get_vpin_alert("BTCUSDT", threshold=0.7), (
            "Expected alert to be True"
        )

    def test_vpin_balanced_low(self):
        """Perfectly balanced buys/sells → low VPIN."""
        stream = LiveVPINStream(bucket_volume=1.0, n_buckets=20)
        ts = time.time_ns()
        for i in range(60):
            side = "buy" if i % 2 == 0 else "sell"
            stream.on_trade("BTCUSDT", side, 0.5, 30000.0, ts + i * 1000)

        vpin = stream.get_vpin("BTCUSDT")
        assert vpin < 0.3, f"Expected low VPIN for balanced flow, got {vpin}"

    def test_vpin_trend_rising(self):
        """Fill first half with balanced trades, second half all-buy → rising."""
        stream = LiveVPINStream(bucket_volume=0.5, n_buckets=20)
        ts = time.time_ns()
        # First 20 balanced
        for i in range(20):
            side = "buy" if i % 2 == 0 else "sell"
            stream.on_trade("BTCUSDT", side, 0.5, 30000.0, ts + i * 100)
        # Next 20 all buy
        for i in range(20, 60):
            stream.on_trade("BTCUSDT", "buy", 0.5, 30000.0, ts + i * 100)

        trend = stream.get_vpin_trend("BTCUSDT")
        # Should be rising or stable (depending on window fill) — not "falling"
        assert trend in ("rising", "stable"), f"Unexpected trend: {trend}"

    def test_vpin_stats_structure(self):
        """get_stats() should return expected keys."""
        stream = LiveVPINStream()
        stats = stream.get_stats("UNKNOWN")
        expected_keys = {"current_vpin", "bucket_count", "alert", "trend", "last_update_ns"}
        assert expected_keys.issubset(set(stats.keys())), (
            f"Missing keys in stats: {expected_keys - set(stats.keys())}"
        )


# ─── DeepLOBLiveBridge tests ──────────────────────────────────────────────────


class TestDeepLOBLiveBridge:

    def test_deeplob_bridge_graceful_no_model(self):
        """Missing model file → bridge returns neutral signal without crashing."""
        bridge = DeepLOBLiveBridge(
            model_path="/tmp/this_file_definitely_does_not_exist_12345.pt",
            device="cpu",
        )
        assert not bridge.model_loaded, "Model should not be loaded"

        bids = _make_bids(10)
        asks = _make_asks(10)
        ts = time.time_ns()

        # Should not raise
        bridge.on_book_update("BTCUSDT", bids, asks, ts)
        pred = bridge.get_prediction("BTCUSDT")

        assert isinstance(pred, DeepLOBPrediction)
        assert pred.direction in ("up", "down", "neutral")
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.raw_logits) == 3

    def test_deeplob_feature_extraction(self):
        """10-level book → 40-feature vector."""
        extractor = _LOBFeatureExtractor()
        bids = _make_bids(10, best=30000.0)
        asks = _make_asks(10, best=30001.0)

        features = extractor.extract(bids, asks)

        assert features is not None, "Feature extractor should return non-None"
        assert features.shape == (FEATURE_DIM,), (
            f"Expected shape ({FEATURE_DIM},), got {features.shape}"
        )
        assert features.dtype.name == "float32"

    def test_deeplob_feature_dimension_constant(self):
        """FEATURE_DIM should equal 4 × N_LEVELS = 40."""
        assert FEATURE_DIM == 4 * N_LEVELS == 40

    def test_deeplob_feature_price_normalisation(self):
        """Normalised bid prices should be <= 0, ask prices >= 0."""
        extractor = _LOBFeatureExtractor()
        bids = [(30000.0 - i, 1.0) for i in range(10)]
        asks = [(30001.0 + i, 1.0) for i in range(10)]

        features = extractor.extract(bids, asks)
        bid_prices = features[:N_LEVELS]
        ask_prices = features[N_LEVELS:2 * N_LEVELS]

        assert bid_prices[0] <= 0.0, f"Best bid price should be <= 0 after normalisation"
        assert ask_prices[0] >= 0.0, f"Best ask price should be >= 0 after normalisation"

    def test_deeplob_signal_for_mm_range(self):
        """get_signal_for_mm() should return a float in [-1, 1]."""
        bridge = DeepLOBLiveBridge(model_path="/nonexistent.pt")
        bids = _make_bids(10)
        asks = _make_asks(10)
        bridge.on_book_update("BTCUSDT", bids, asks, time.time_ns())
        signal = bridge.get_signal_for_mm("BTCUSDT")
        assert -1.0 <= signal <= 1.0, f"MM signal out of range: {signal}"

    def test_deeplob_window_accumulates(self):
        """Window depth should grow with each on_book_update call."""
        bridge = DeepLOBLiveBridge(model_path="/nonexistent.pt")
        bids = _make_bids(10)
        asks = _make_asks(10)
        ts = time.time_ns()

        for i in range(5):
            bridge.on_book_update("BTCUSDT", bids, asks, ts + i * 1000)

        depth = bridge.window_depth("BTCUSDT")
        assert depth == 5, f"Expected window depth 5, got {depth}"

    def test_deeplob_bad_lob_no_crash(self):
        """Empty or invalid LOB should not crash the bridge."""
        bridge = DeepLOBLiveBridge(model_path="/nonexistent.pt")
        bridge.on_book_update("BTCUSDT", [], [], time.time_ns())
        bridge.on_book_update("BTCUSDT", [(0.0, 0.0)], [(0.0, 0.0)], time.time_ns())
        # No assertion needed — just verify no exception raised


# ─── EventCalendar tests ──────────────────────────────────────────────────────


class TestEventCalendar:

    def _calendar_with_event(
        self,
        offset_seconds: float,
        blackout_before: int = 120,
        blackout_after: int = 300,
    ) -> tuple:
        """
        Create a calendar with one high-impact event at now + offset_seconds.

        Returns (calendar, event).
        """
        cal = EventCalendar(data_dir="/tmp/test_event_calendar")
        cal._events = []  # clear hardcoded fallback events

        event_time = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
        event = EventRecord(
            name="TEST_EVENT",
            datetime_utc=event_time,
            impact="high",
            symbol_filter=None,
            blackout_before_s=blackout_before,
            blackout_after_s=blackout_after,
        )
        cal.add_event(event)
        return cal, event

    def test_event_calendar_blackout_window_inside(self):
        """
        Event at T=now+60s; check at now (within 120s blackout window before)
        → is_blackout_now() should be True.
        """
        cal, _ = self._calendar_with_event(offset_seconds=60)
        # We are 60s before the event, well within the 120s pre-blackout window
        assert cal.is_blackout_now() is True, (
            "Expected blackout True when inside pre-event window"
        )

    def test_event_calendar_blackout_window_outside(self):
        """
        Event at T=now+60s with blackout_after=300s.
        Check at T+301s (just after the blackout ends) → should be False.

        We simulate this by placing the event 361 seconds in the past
        (past the 300s after-window).
        """
        cal, _ = self._calendar_with_event(offset_seconds=-361, blackout_after=300)
        assert cal.is_blackout_now() is False, (
            "Expected blackout False when 361s after event with 300s post-window"
        )

    def test_event_calendar_blackout_t_minus_1min(self):
        """
        Event at T=now+60s → is_blackout_now() True (within pre-event window).
        """
        cal, _ = self._calendar_with_event(offset_seconds=60, blackout_before=120)
        assert cal.is_blackout_now() is True

    def test_event_calendar_blackout_t_plus_6min(self):
        """
        Event at T=now-6min with blackout_after=300s (5min) → should be False.
        """
        cal, _ = self._calendar_with_event(offset_seconds=-361, blackout_after=300)
        assert cal.is_blackout_now() is False

    def test_event_calendar_next_event(self):
        """next_event() should return the soonest future event."""
        cal = EventCalendar(data_dir="/tmp/test_event_calendar_next")
        cal._events = []

        now = datetime.now(timezone.utc)
        ev_soon = EventRecord(
            name="SOON",
            datetime_utc=now + timedelta(hours=1),
            impact="high",
        )
        ev_later = EventRecord(
            name="LATER",
            datetime_utc=now + timedelta(hours=48),
            impact="medium",
        )
        ev_past = EventRecord(
            name="PAST",
            datetime_utc=now - timedelta(hours=1),
            impact="high",
        )
        cal._events = [ev_later, ev_soon, ev_past]

        nxt = cal.next_event()
        assert nxt is not None
        assert nxt.name == "SOON", f"Expected next event 'SOON', got '{nxt.name}'"

    def test_event_calendar_next_event_none_when_empty(self):
        """next_event() should return None when no future events."""
        cal = EventCalendar(data_dir="/tmp/test_event_calendar_none")
        cal._events = [
            EventRecord(
                name="OLD",
                datetime_utc=datetime.now(timezone.utc) - timedelta(hours=1),
            )
        ]
        assert cal.next_event() is None

    def test_event_calendar_time_to_next(self):
        """time_to_next_event_seconds() should return a positive float."""
        cal = EventCalendar(data_dir="/tmp/test_event_calendar_time")
        cal._events = []
        ev = EventRecord(
            name="FOMC",
            datetime_utc=datetime.now(timezone.utc) + timedelta(hours=2),
            impact="high",
        )
        cal._events = [ev]
        secs = cal.time_to_next_event_seconds()
        assert secs is not None
        assert secs > 0, f"Expected positive seconds, got {secs}"
        # Roughly 2 hours = 7200 seconds (allow ±60s for test execution)
        assert 7140 <= secs <= 7260, f"Unexpected seconds: {secs}"

    def test_event_calendar_fomc_fallback_populated(self):
        """Hardcoded FOMC fallback should have 8 FOMC events for 2026."""
        cal = EventCalendar(data_dir="/tmp/test_event_calendar_fomc")
        fomc_events = [e for e in cal._events if "FOMC" in e.name]
        assert len(fomc_events) >= 8, (
            f"Expected at least 8 FOMC events, got {len(fomc_events)}"
        )

    def test_event_record_blackout_boundaries(self):
        """EventRecord.is_in_blackout() should respect before/after windows."""
        event_time = datetime(2026, 6, 10, 19, 0, 0, tzinfo=timezone.utc)
        ev = EventRecord(
            name="FOMC",
            datetime_utc=event_time,
            blackout_before_s=120,
            blackout_after_s=300,
        )
        # 90s before → in blackout
        assert ev.is_in_blackout(event_time - timedelta(seconds=90))
        # 121s before → outside blackout
        assert not ev.is_in_blackout(event_time - timedelta(seconds=121))
        # 299s after → in blackout
        assert ev.is_in_blackout(event_time + timedelta(seconds=299))
        # 301s after → outside blackout
        assert not ev.is_in_blackout(event_time + timedelta(seconds=301))


# ─── MicropriceDriftSignal tests ──────────────────────────────────────────────


class TestMicropriceDriftSignal:

    def test_microprice_drift_bullish(self):
        """15/20 snapshots with microprice > mid → bullish signal."""
        sig = MicropriceDriftSignal(window=20, threshold=0.6)
        ts = time.time_ns()
        mid = 30000.5

        # 15 snapshots where microprice > mid
        for i in range(15):
            micro = mid + 0.1  # micro > mid
            sig.on_book_update("BTCUSDT", micro, mid, ts + i * 1000)

        # 5 snapshots where microprice <= mid
        for i in range(15, 20):
            micro = mid - 0.1  # micro < mid
            sig.on_book_update("BTCUSDT", micro, mid, ts + i * 1000)

        drift = sig.get_drift("BTCUSDT")
        signal = sig.get_drift_signal("BTCUSDT")

        assert drift == pytest.approx(15 / 20, rel=1e-6)
        assert signal == "bullish", f"Expected bullish, got {signal}"

    def test_microprice_drift_bearish(self):
        """15/20 snapshots with microprice < mid → bearish signal."""
        sig = MicropriceDriftSignal(window=20, threshold=0.6)
        ts = time.time_ns()
        mid = 30000.5

        for i in range(15):
            micro = mid - 0.1  # micro < mid
            sig.on_book_update("BTCUSDT", micro, mid, ts + i * 1000)
        for i in range(15, 20):
            micro = mid + 0.1
            sig.on_book_update("BTCUSDT", micro, mid, ts + i * 1000)

        signal = sig.get_drift_signal("BTCUSDT")
        assert signal == "bearish", f"Expected bearish, got {signal}"

    def test_microprice_drift_neutral(self):
        """10 above / 10 below → neutral signal."""
        sig = MicropriceDriftSignal(window=20, threshold=0.6)
        ts = time.time_ns()
        mid = 30000.5

        for i in range(20):
            micro = mid + (0.1 if i % 2 == 0 else -0.1)
            sig.on_book_update("BTCUSDT", micro, mid, ts + i * 1000)

        signal = sig.get_drift_signal("BTCUSDT")
        assert signal == "neutral", f"Expected neutral, got {signal}"

    def test_microprice_drift_quote_skew(self):
        """Bullish drift → positive quote_skew."""
        sig = MicropriceDriftSignal(window=20, threshold=0.6)
        ts = time.time_ns()
        mid = 30000.5

        # All microprice > mid → magnitude = 1.0 → skew = +2.0
        for i in range(20):
            sig.on_book_update("BTCUSDT", mid + 0.1, mid, ts + i * 1000)

        skew = sig.get_quote_skew("BTCUSDT")
        assert skew > 0.0, f"Expected positive skew for bullish drift, got {skew}"
        assert skew == pytest.approx(2.0, rel=1e-6), (
            f"Expected skew = 2.0 for full bullish, got {skew}"
        )

    def test_microprice_drift_quote_skew_bearish(self):
        """Bearish drift → negative quote_skew."""
        sig = MicropriceDriftSignal(window=20, threshold=0.6)
        ts = time.time_ns()
        mid = 30000.5

        for i in range(20):
            sig.on_book_update("BTCUSDT", mid - 0.1, mid, ts + i * 1000)

        skew = sig.get_quote_skew("BTCUSDT")
        assert skew < 0.0, f"Expected negative skew for bearish drift, got {skew}"
        assert skew == pytest.approx(-2.0, rel=1e-6)

    def test_microprice_drift_magnitude_range(self):
        """get_drift_magnitude() should always be in [-1, 1]."""
        sig = MicropriceDriftSignal(window=20, threshold=0.6)
        ts = time.time_ns()
        mid = 30000.5

        for i in range(20):
            micro = mid + (0.1 if i < 12 else -0.1)
            sig.on_book_update("X", micro, mid, ts + i * 1000)

        mag = sig.get_drift_magnitude("X")
        assert -1.0 <= mag <= 1.0, f"Drift magnitude out of range: {mag}"

    def test_microprice_drift_reset(self):
        """reset() should clear all history for the symbol."""
        sig = MicropriceDriftSignal(window=20)
        ts = time.time_ns()
        mid = 30000.5

        for i in range(20):
            sig.on_book_update("BTCUSDT", mid + 0.1, mid, ts + i * 1000)

        sig.reset("BTCUSDT")
        assert sig.get_drift("BTCUSDT") == 0.5, "Expected neutral drift after reset"
        assert sig.get_drift_signal("BTCUSDT") == "neutral"

    def test_microprice_drift_state_dataclass(self):
        """get_drift_state() should return a complete DriftState dataclass."""
        sig = MicropriceDriftSignal(window=10)
        ts = time.time_ns()
        mid = 1000.0
        for i in range(8):
            sig.on_book_update("ETH", mid + 0.05, mid, ts + i * 1000)

        state = sig.get_drift_state("ETH")
        assert isinstance(state, DriftState)
        assert state.symbol == "ETH"
        assert state.window == 8
        assert state.signal in ("bullish", "bearish", "neutral")
        assert -1.0 <= state.magnitude <= 1.0
        assert -2.0 <= state.quote_skew <= 2.0

    def test_microprice_compute_utility(self):
        """compute_microprice() static method should produce valid microprice."""
        micro = MicropriceDriftSignal.compute_microprice(
            best_bid=30000.0,
            best_ask=30002.0,
            bid_size=2.0,
            ask_size=1.0,
        )
        # With more bid volume, microprice should be closer to ask
        mid = (30000.0 + 30002.0) / 2.0
        # micro = (2 × 30002 + 1 × 30000) / 3 = 90004/3 + 30000/3 = 30001.333
        assert abs(micro - 30001.333) < 0.01
        # microprice should be between bid and ask
        assert 30000.0 <= micro <= 30002.0
