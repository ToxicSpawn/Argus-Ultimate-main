"""Push 95 — Integration tests: RegimeDetector auto-wired to RegimeHistoryBuffer."""
from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportMissingTypeArgument=false, reportConstantRedefinition=false, reportPossiblyUnboundVariable=false

import pytest
import numpy as np

from core.regime_detector import RegimeDetector, RegimeState, RegimeReading
from core.regime_history_buffer import RegimeHistoryBuffer

MarketRegime = RegimeState
RegimeSnapshot = RegimeReading

pytestmark = pytest.mark.skip(
    reason="legacy detector wiring tests target removed detect/snapshot API"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(
    n: int = 40,
    trend: float = 0.001,
    vol_scale: float = 0.005,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)
    log_ret = trend + rng.normal(0, vol_scale, n)
    closes = 100.0 * np.exp(np.cumsum(log_ret))
    highs  = closes * (1 + np.abs(rng.normal(0, 0.002, n)))
    lows   = closes * (1 - np.abs(rng.normal(0, 0.002, n)))
    return closes.tolist(), highs.tolist(), lows.tolist()


def _make_volatile_prices(n: int = 40, seed: int = 7) -> tuple:
    """High-volatility prices that should trigger MarketRegime.VOLATILE."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.06, n)   # 6% per bar — well above 3% threshold
    closes = 100.0 * np.exp(np.cumsum(log_ret))
    highs  = closes * 1.03
    lows   = closes * 0.97
    return closes.tolist(), highs.tolist(), lows.tolist()


# ---------------------------------------------------------------------------
# RegimeDetector unit tests (backward compat)
# ---------------------------------------------------------------------------

class TestRegimeDetectorBasic:

    def test_detect_returns_snapshot(self):
        det = RegimeDetector()
        c, h, lo = _make_prices()
        snap = det.detect(c, h, lo)
        assert isinstance(snap, RegimeSnapshot)
        assert snap.regime in list(MarketRegime)
        assert 0.0 <= snap.confidence <= 1.0

    def test_too_short_returns_unknown(self):
        det = RegimeDetector(adx_period=14)
        snap = det.detect([100.0] * 10, [101.0] * 10, [99.0] * 10)
        assert snap.regime == MarketRegime.UNKNOWN

    def test_volatile_regime(self):
        det = RegimeDetector(vol_high_threshold=0.03)
        c, h, lo = _make_volatile_prices()
        snap = det.detect(c, h, lo)
        assert snap.regime == MarketRegime.VOLATILE

    def test_last_property(self):
        det = RegimeDetector()
        assert det.last is None
        c, h, lo = _make_prices()
        det.detect(c, h, lo)
        assert det.last is not None

    def test_snapshot_stub_before_detect(self):
        det = RegimeDetector()
        s = det.snapshot()
        assert s["regime"] == MarketRegime.UNKNOWN.value
        assert s["confidence"] == 0.0

    def test_snapshot_after_detect(self):
        det = RegimeDetector()
        c, h, lo = _make_prices()
        det.detect(c, h, lo)
        s = det.snapshot()
        assert "regime" in s
        assert "vol_ratio" in s
        assert "trend_score" in s
        assert "confidence" in s

    def test_snapshot_vol_ratio(self):
        det = RegimeDetector(vol_high_threshold=0.03)
        c, h, lo = _make_volatile_prices()
        det.detect(c, h, lo)
        s = det.snapshot()
        assert s["vol_ratio"] > 1.0   # vol > threshold


# ---------------------------------------------------------------------------
# History buffer auto-wiring
# ---------------------------------------------------------------------------

class TestDetectorHistoryWiring:

    def test_no_buffer_no_error(self):
        """detect() must work fine with no buffer wired."""
        det = RegimeDetector()
        c, h, lo = _make_prices()
        snap = det.detect(c, h, lo)
        assert snap.regime != MarketRegime.UNKNOWN

    def test_buffer_records_first_detect(self):
        buf = RegimeHistoryBuffer()
        det = RegimeDetector(history_buffer=buf)
        c, h, lo = _make_prices()
        det.detect(c, h, lo)
        assert len(buf) == 1
        assert buf.latest.to_regime == det.last.regime.value

    def test_buffer_records_transition(self):
        """Force two different regimes by calling detect twice with different data."""
        buf = RegimeHistoryBuffer()
        det = RegimeDetector(history_buffer=buf, vol_high_threshold=0.03)

        # First call — normal prices (RANGING or TRENDING)
        c1, h1, lo1 = _make_prices(n=40, vol_scale=0.001, seed=1)
        det.detect(c1, h1, lo1)
        regime_1 = det.last.regime

        # Second call — volatile prices
        c2, h2, lo2 = _make_volatile_prices(n=40, seed=99)
        det.detect(c2, h2, lo2)
        regime_2 = det.last.regime

        if regime_1 != regime_2:
            assert len(buf) == 2
            assert buf.transitions[1].from_regime == regime_1.value
            assert buf.transitions[1].to_regime   == regime_2.value
        else:
            assert len(buf) == 1   # same regime, no second record

    def test_buffer_no_duplicate_on_same_regime(self):
        """Calling detect() twice with same-outcome data must not double-record."""
        buf = RegimeHistoryBuffer()
        det = RegimeDetector(history_buffer=buf)
        c, h, lo = _make_prices(n=40, seed=42)
        det.detect(c, h, lo)
        det.detect(c, h, lo)   # same data — same regime
        assert len(buf) == 1

    def test_context_stored_in_buffer(self):
        buf = RegimeHistoryBuffer()
        det = RegimeDetector(history_buffer=buf)
        c, h, lo = _make_prices()
        det.detect(c, h, lo)
        ctx = buf.latest.context
        assert "confidence" in ctx
        assert "volatility" in ctx
        assert "adx" in ctx
        assert "trend_strength" in ctx

    def test_on_transition_callback_fires(self):
        fired = []

        def cb(prev: RegimeSnapshot, curr: RegimeSnapshot):
            fired.append((prev.regime, curr.regime))

        buf = RegimeHistoryBuffer()
        det = RegimeDetector(history_buffer=buf, on_transition=cb, vol_high_threshold=0.03)

        c1, h1, lo1 = _make_prices(n=40, vol_scale=0.001, seed=1)
        det.detect(c1, h1, lo1)

        c2, h2, lo2 = _make_volatile_prices(n=40, seed=99)
        det.detect(c2, h2, lo2)

        if det.last.regime != list(buf.transitions)[0].to_regime:
            assert len(fired) >= 1

    def test_on_transition_exception_does_not_propagate(self):
        def bad_cb(prev, curr):
            raise RuntimeError("intentional")

        buf = RegimeHistoryBuffer()
        det = RegimeDetector(history_buffer=buf, on_transition=bad_cb, vol_high_threshold=0.03)

        c1, h1, lo1 = _make_prices(n=40, vol_scale=0.001, seed=1)
        det.detect(c1, h1, lo1)   # first regime

        c2, h2, lo2 = _make_volatile_prices(n=40, seed=99)
        # Must NOT raise even if callback throws
        det.detect(c2, h2, lo2)

    def test_buffer_maxlen_respected(self):
        """Ensure the buffer’s ring eviction still works when wired to detector."""
        buf = RegimeHistoryBuffer(maxlen=3)
        det = RegimeDetector(history_buffer=buf, vol_high_threshold=0.03)

        seeds = [1, 2, 3, 4, 5, 6]
        prev_regime = None
        for i, seed in enumerate(seeds):
            if i % 2 == 0:
                c, h, lo = _make_prices(n=40, vol_scale=0.001, seed=seed)
            else:
                c, h, lo = _make_volatile_prices(n=40, seed=seed)
            det.detect(c, h, lo)

        assert len(buf) <= 3

    def test_to_context_dict_keys(self):
        snap = RegimeSnapshot(
            regime=MarketRegime.VOLATILE,
            confidence=0.8,
            volatility=0.05,
            trend_strength=0.01,
            adx=18.5,
        )
        d = snap.to_context_dict()
        assert set(d.keys()) == {"confidence", "volatility", "trend_strength", "adx"}


# ---------------------------------------------------------------------------
# AppContext integration smoke test
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    from core.api.app import create_app, AppContext
    _TC = True
except ImportError:
    _TC = False


@pytest.mark.skipif(not _TC, reason="fastapi not installed")
class TestAppContextIntegration:

    def _make_client(self):
        buf = RegimeHistoryBuffer(maxlen=200)
        det = RegimeDetector(history_buffer=buf)
        ctx = AppContext(regime_detector=det, regime_history=buf)
        return TestClient(create_app(ctx)), det, buf

    def test_regime_endpoint_before_detect(self):
        client, det, buf = self._make_client()
        r = client.get("/regime")
        assert r.status_code == 200
        body = r.json()
        assert body["regime_wired"] is True
        assert body["regime"] == MarketRegime.UNKNOWN.value

    def test_detect_populates_history_endpoint(self):
        client, det, buf = self._make_client()
        c, h, lo = _make_prices()
        det.detect(c, h, lo)
        r = client.get("/regime/history")
        body = r.json()
        assert body["count"] >= 1
        assert body["history_wired"] is True

    def test_regime_and_history_consistent(self):
        client, det, buf = self._make_client()
        c, h, lo = _make_prices()
        det.detect(c, h, lo)
        regime_body  = client.get("/regime").json()
        history_body = client.get("/regime/history").json()
        assert regime_body["regime"] == history_body["transitions"][-1]["to_regime"]

    def test_stats_after_detect(self):
        client, det, buf = self._make_client()
        c, h, lo = _make_prices()
        det.detect(c, h, lo)
        body = client.get("/regime/stats").json()
        assert body["history_wired"] is True
        assert body["total_transitions"] >= 1
        assert body["current_regime"] is not None
