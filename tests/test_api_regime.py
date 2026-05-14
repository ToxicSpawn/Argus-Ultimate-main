"""
Push 93 — unit tests for /regime, /sizer, /bandit endpoints
and WS /ws/regime.

Uses TestClient from httpx / starlette; mocks all dependencies.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

from core.api.app import create_app, AppContext


def _make_snap(regime="RANGING", vol_ratio=1.0, trend_score=0.001,
               bb_pos=0.5, autocorr=0.05, confidence=0.4,
               tick_count=100, ts=0.0):
    snap = MagicMock()
    snap.regime = MagicMock()
    snap.regime.value = regime
    snap.vol_ratio    = vol_ratio
    snap.trend_score  = trend_score
    snap.bb_pos       = bb_pos
    snap.autocorr     = autocorr
    snap.confidence   = confidence
    snap.tick_count   = tick_count
    snap.ts           = ts
    return snap


@pytest.fixture
def client_no_ctx():
    if not _FASTAPI:
        pytest.skip("fastapi not installed")
    app = create_app(AppContext())
    return TestClient(app)


@pytest.fixture
def client_full():
    if not _FASTAPI:
        pytest.skip("fastapi not installed")
    det = MagicMock()
    det.snapshot.return_value = _make_snap("HIGH_VOL", vol_ratio=3.1, confidence=0.9)

    from core.risk.regime_sizer import RegimeAwareSizer
    sizer = RegimeAwareSizer(detector=det)

    bandit = MagicMock()
    bandit.allocations.return_value = {"mom_BTCUSDT": 0.6, "mr_BTCUSDT": 0.4}
    bandit.summary.return_value     = {"mom_BTCUSDT": {"sharpe": 1.2}}

    ctx = AppContext(
        regime_detector=det,
        regime_sizer=sizer,
        bandit_router=bandit,
    )
    app = create_app(ctx)
    return TestClient(app)


# -----------------------------------------------------------------------
# /regime
# -----------------------------------------------------------------------

class TestRegimeEndpoint:
    def test_regime_no_detector(self, client_no_ctx):
        r = client_no_ctx.get("/regime")
        assert r.status_code == 200
        data = r.json()
        assert data["regime"] == "UNKNOWN"
        assert data["detector_wired"] is False

    def test_regime_with_detector(self, client_full):
        r = client_full.get("/regime")
        assert r.status_code == 200
        data = r.json()
        assert data["regime"] == "HIGH_VOL"
        assert data["detector_wired"] is True
        assert data["vol_ratio"] == pytest.approx(3.1)
        assert data["confidence"] == pytest.approx(0.9)


# -----------------------------------------------------------------------
# POST /regime/scalars
# -----------------------------------------------------------------------

class TestRegimeScalars:
    def test_update_scalar(self, client_full):
        r = client_full.post("/regime/scalars",
                             json={"regime": "HIGH_VOL", "scalar": 0.20})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["regime"] == "HIGH_VOL"
        assert data["scalar"] == pytest.approx(0.20)

    def test_invalid_regime(self, client_full):
        r = client_full.post("/regime/scalars",
                             json={"regime": "MOON", "scalar": 0.5})
        assert r.status_code == 400

    def test_no_sizer(self, client_no_ctx):
        r = client_no_ctx.post("/regime/scalars",
                               json={"regime": "HIGH_VOL", "scalar": 0.5})
        assert r.status_code == 503


# -----------------------------------------------------------------------
# GET /sizer
# -----------------------------------------------------------------------

class TestSizerEndpoint:
    def test_sizer_no_ctx(self, client_no_ctx):
        r = client_no_ctx.get("/sizer")
        assert r.status_code == 200
        assert r.json()["sizer_wired"] is False

    def test_sizer_with_ctx(self, client_full):
        r = client_full.get("/sizer")
        assert r.status_code == 200
        data = r.json()
        assert data["sizer_wired"] is True
        assert "current_regime"  in data
        assert "regime_scalars"  in data
        assert "current_scalar"  in data


# -----------------------------------------------------------------------
# GET /bandit
# -----------------------------------------------------------------------

class TestBanditEndpoint:
    def test_bandit_no_ctx(self, client_no_ctx):
        r = client_no_ctx.get("/bandit")
        assert r.status_code == 200
        assert r.json()["bandit_wired"] is False

    def test_bandit_with_ctx(self, client_full):
        r = client_full.get("/bandit")
        assert r.status_code == 200
        data = r.json()
        assert data["bandit_wired"] is True
        assert "allocations" in data
        assert "regime"      in data

    def test_bandit_regime_override(self, client_full):
        r = client_full.get("/bandit?regime=RANGING")
        assert r.status_code == 200
        assert r.json()["regime"] == "RANGING"


# -----------------------------------------------------------------------
# Existing routes still work (regression)
# -----------------------------------------------------------------------

class TestExistingRoutes:
    def test_health(self, client_no_ctx):
        r = client_no_ctx.get("/health")
        assert r.status_code == 200
        assert r.json()["version"] == "8.29.0"

    def test_status(self, client_no_ctx):
        r = client_no_ctx.get("/status")
        assert r.status_code == 200

    def test_positions(self, client_no_ctx):
        r = client_no_ctx.get("/positions")
        assert r.status_code == 200

    def test_orders(self, client_no_ctx):
        r = client_no_ctx.get("/orders")
        assert r.status_code == 200

    def test_metrics(self, client_no_ctx):
        r = client_no_ctx.get("/metrics")
        assert r.status_code == 200
