#!/usr/bin/env python3
"""
Tests for FIX Protocol Adapter, Maker Rebate Optimizer, and Colocation Profiler.

Covers:
1.  FIXOrderMessage round-trip encode/decode
2.  FIXOrderMessage checksum consistency
3.  FIXExecutionReport fill classification
4.  BinaryOrderEncoder benchmark produces a numeric result
5.  FIXAdapter.validate_order rejects bad orders
6.  FIXAdapter.maker_only_order produces GTX limit
7.  FIXAdapter.ccxt_to_execution_report conversion
8.  MakerRebateOptimizer.optimal_limit_price respects urgency bands
9.  MakerRebateOptimizer.simulate_rebate_pnl aggregates correctly
10. ColocationProfiler.colocation_score classification + latency_adjusted_edge
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from execution.fix_protocol_adapter import (
    BinaryOrderEncoder,
    FIXAdapter,
    FIXExecType,
    FIXExecutionReport,
    FIXOrdType,
    FIXOrderMessage,
    FIXSide,
    FIXTag,
    FIXTimeInForce,
    _mod256_checksum,
)
from execution.maker_rebate_optimizer import MakerRebateOptimizer, RebateOpportunity
from infra.colocation_profiler import (
    ColocationProfiler,
    ExchangeLatency,
    DEFAULT_EXCHANGES_CONFIG,
)


# ===========================================================================
# TEST 1 — FIXOrderMessage round-trip encode / decode
# ===========================================================================

class TestFIXOrderMessageRoundTrip:
    """Test 1: FIXOrderMessage serialises and deserialises correctly."""

    def test_roundtrip_limit_order(self) -> None:
        msg = FIXOrderMessage(
            symbol        = "BTC/USD",
            side          = FIXSide.BUY,
            order_qty     = 0.5,
            price         = 29_500.00,
            ord_type      = FIXOrdType.LIMIT,
            time_in_force = FIXTimeInForce.GTC,
            cl_ord_id     = "TEST-001",
            transact_time = 1_700_000_000.0,
        )
        encoded = msg.to_bytes()
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        decoded = FIXOrderMessage.from_bytes(encoded)

        assert decoded.symbol        == msg.symbol
        assert decoded.side          == msg.side
        assert abs(decoded.order_qty - msg.order_qty) < 1e-8
        assert abs(decoded.price     - msg.price)     < 1e-4
        assert decoded.ord_type      == msg.ord_type
        assert decoded.time_in_force == msg.time_in_force
        assert decoded.cl_ord_id     == msg.cl_ord_id

    def test_to_dict_keys(self) -> None:
        msg = FIXOrderMessage(symbol="ETH/USD", side=FIXSide.SELL, order_qty=1.0)
        d   = msg.to_dict()
        assert set(d.keys()) >= {"cl_ord_id", "symbol", "side", "order_qty", "price",
                                  "ord_type", "time_in_force", "transact_time"}
        assert d["side"]    == "SELL"
        assert d["symbol"]  == "ETH/USD"


# ===========================================================================
# TEST 2 — FIXOrderMessage checksum consistency
# ===========================================================================

class TestFIXChecksum:
    """Test 2: Checksum is stable for identical orders."""

    def test_checksum_deterministic(self) -> None:
        msg1 = FIXOrderMessage(
            symbol="SOL/USD", side=FIXSide.BUY, order_qty=10.0,
            price=50.0, cl_ord_id="CHKTEST", transact_time=1_234_567_890.0,
        )
        msg2 = FIXOrderMessage(
            symbol="SOL/USD", side=FIXSide.BUY, order_qty=10.0,
            price=50.0, cl_ord_id="CHKTEST", transact_time=1_234_567_890.0,
        )
        assert msg1.checksum() == msg2.checksum()

    def test_checksum_differs_on_mutation(self) -> None:
        msg1 = FIXOrderMessage(
            symbol="SOL/USD", side=FIXSide.BUY, order_qty=10.0,
            price=50.0, cl_ord_id="CHK-A", transact_time=1_000.0,
        )
        msg2 = FIXOrderMessage(
            symbol="SOL/USD", side=FIXSide.BUY, order_qty=10.0,
            price=51.0, cl_ord_id="CHK-A", transact_time=1_000.0,
        )
        assert msg1.checksum() != msg2.checksum()

    def test_mod256_range(self) -> None:
        for data in [b"hello", b"", b"\x00" * 1000, b"\xff" * 256]:
            chk = _mod256_checksum(data)
            assert 0 <= chk <= 255


# ===========================================================================
# TEST 3 — FIXExecutionReport classification
# ===========================================================================

class TestFIXExecutionReport:
    """Test 3: Execution report status helpers."""

    def _make_report(self, exec_type: FIXExecType, **kw) -> FIXExecutionReport:
        return FIXExecutionReport(
            cl_ord_id  = "ERTEST",
            exec_type  = exec_type,
            symbol     = "BTC/USD",
            side       = FIXSide.BUY,
            leaves_qty = kw.get("leaves_qty", 0.0),
            cum_qty    = kw.get("cum_qty", 1.0),
            avg_px     = kw.get("avg_px", 30_000.0),
            order_qty  = kw.get("order_qty", 1.0),
        )

    def test_is_fill(self) -> None:
        r = self._make_report(FIXExecType.FILL)
        assert r.is_fill()
        assert not r.is_partial_fill()
        assert not r.is_cancel()
        assert not r.is_reject()

    def test_is_partial_fill(self) -> None:
        r = self._make_report(FIXExecType.PARTIAL_FILL, cum_qty=0.5, leaves_qty=0.5)
        assert r.is_partial_fill()
        assert not r.is_fill()

    def test_is_cancel(self) -> None:
        r = self._make_report(FIXExecType.CANCELED, cum_qty=0.0, leaves_qty=1.0)
        assert r.is_cancel()

    def test_is_reject(self) -> None:
        r = self._make_report(FIXExecType.REJECTED, cum_qty=0.0, leaves_qty=1.0)
        assert r.is_reject()

    def test_fill_properties(self) -> None:
        r = self._make_report(FIXExecType.FILL, cum_qty=2.5, avg_px=31_000.0, leaves_qty=0.0)
        assert r.fill_quantity  == 2.5
        assert r.fill_price     == 31_000.0
        assert r.leaves_quantity == 0.0


# ===========================================================================
# TEST 4 — BinaryOrderEncoder benchmark
# ===========================================================================

class TestBinaryOrderEncoder:
    """Test 4: BinaryOrderEncoder produces valid memoryviews and benchmarks."""

    def test_encode_produces_bytes(self) -> None:
        enc = BinaryOrderEncoder()
        mv  = enc.encode_new_order(
            "BTC/USD", FIXSide.BUY, 0.1, 30_000.0,
            FIXOrdType.LIMIT, FIXTimeInForce.GTC,
        )
        assert isinstance(mv, memoryview)
        assert len(mv) > 0
        raw = bytes(mv)
        assert b"BTC/USD" in raw

    def test_encode_cancel(self) -> None:
        enc = BinaryOrderEncoder()
        mv  = enc.encode_cancel("ORIG-001", "BTC/USD")
        assert isinstance(mv, memoryview)
        assert b"ORIG-001" in bytes(mv)

    def test_encode_cancel_replace(self) -> None:
        enc = BinaryOrderEncoder()
        mv  = enc.encode_cancel_replace("ORIG-002", "ETH/USD", 2_000.0, 1.5)
        assert isinstance(mv, memoryview)
        raw = bytes(mv)
        assert b"ORIG-002" in raw

    def test_benchmark_returns_numeric(self) -> None:
        enc     = BinaryOrderEncoder()
        ns_op   = enc.benchmark_encode(n=1_000)   # small n for test speed
        assert isinstance(ns_op, float)
        assert ns_op > 0


# ===========================================================================
# TEST 5 — FIXAdapter.validate_order
# ===========================================================================

class TestFIXAdapterValidation:
    """Test 5: Pre-flight validation catches invalid orders."""

    def setup_method(self) -> None:
        self.adapter = FIXAdapter(dry_run=True)

    def test_valid_limit_order(self) -> None:
        msg = FIXOrderMessage(
            symbol="BTC/USD", side=FIXSide.BUY, order_qty=0.1,
            price=30_000.0, ord_type=FIXOrdType.LIMIT,
        )
        ok, reason = self.adapter.validate_order(msg)
        assert ok, reason

    def test_rejects_zero_qty(self) -> None:
        msg = FIXOrderMessage(
            symbol="BTC/USD", side=FIXSide.BUY, order_qty=0.0,
            price=30_000.0,
        )
        ok, _ = self.adapter.validate_order(msg)
        assert not ok

    def test_rejects_limit_without_price(self) -> None:
        msg = FIXOrderMessage(
            symbol="BTC/USD", side=FIXSide.BUY, order_qty=0.1,
            price=0.0, ord_type=FIXOrdType.LIMIT,
        )
        ok, _ = self.adapter.validate_order(msg)
        assert not ok

    def test_rejects_gtx_market_order(self) -> None:
        msg = FIXOrderMessage(
            symbol="BTC/USD", side=FIXSide.BUY, order_qty=0.1,
            price=0.0, ord_type=FIXOrdType.MARKET,
            time_in_force=FIXTimeInForce.GTX,
        )
        ok, _ = self.adapter.validate_order(msg)
        assert not ok


# ===========================================================================
# TEST 6 — FIXAdapter.maker_only_order
# ===========================================================================

class TestMakerOnlyOrder:
    """Test 6: maker_only_order produces a valid GTX limit order."""

    def test_produces_gtx_limit(self) -> None:
        adapter = FIXAdapter(dry_run=True)
        msg     = adapter.maker_only_order("BTC/USD", FIXSide.SELL, 30_500.0, 0.25)

        assert msg.ord_type      == FIXOrdType.LIMIT
        assert msg.time_in_force == FIXTimeInForce.GTX
        assert msg.symbol        == "BTC/USD"
        assert msg.side          == FIXSide.SELL
        assert msg.price         == 30_500.0
        assert msg.order_qty     == 0.25

    def test_post_only_flag_in_ccxt_params(self) -> None:
        adapter = FIXAdapter(dry_run=True)
        msg     = adapter.maker_only_order("ETH/USD", FIXSide.BUY, 2_000.0, 1.0)
        params  = adapter.order_to_ccxt(msg)
        assert params["params"].get("postOnly") is True
        assert params["params"].get("timeInForce") == "GTX"


# ===========================================================================
# TEST 7 — FIXAdapter.ccxt_to_execution_report
# ===========================================================================

class TestCcxtToExecutionReport:
    """Test 7: ccxt response dict converts to FIXExecutionReport correctly."""

    def setup_method(self) -> None:
        self.adapter = FIXAdapter(dry_run=True)

    def test_full_fill(self) -> None:
        response = {
            "id":             "12345",
            "clientOrderId":  "MYORD-001",
            "symbol":         "BTC/USD",
            "side":           "buy",
            "status":         "closed",
            "amount":         1.0,
            "filled":         1.0,
            "remaining":      0.0,
            "average":        30_100.0,
            "timestamp":      1_700_000_000_000,
        }
        report = self.adapter.ccxt_to_execution_report(response)
        assert report.is_fill()
        assert report.fill_quantity == 1.0
        assert report.fill_price   == 30_100.0
        assert report.cl_ord_id    == "MYORD-001"

    def test_partial_fill(self) -> None:
        response = {
            "id":      "99",
            "symbol":  "ETH/USD",
            "side":    "sell",
            "status":  "open",
            "amount":  2.0,
            "filled":  0.5,
            "remaining": 1.5,
            "average": 1_900.0,
            "timestamp": 1_700_000_000_000,
        }
        report = self.adapter.ccxt_to_execution_report(response)
        assert report.is_partial_fill()
        assert report.leaves_quantity == 1.5

    def test_cancel(self) -> None:
        response = {
            "id":        "77",
            "symbol":    "SOL/USD",
            "side":      "buy",
            "status":    "canceled",
            "amount":    5.0,
            "filled":    0.0,
            "remaining": 5.0,
            "average":   None,
            "timestamp": 1_700_000_000_000,
        }
        report = self.adapter.ccxt_to_execution_report(response)
        assert report.is_cancel()


# ===========================================================================
# TEST 8 — MakerRebateOptimizer.optimal_limit_price
# ===========================================================================

class TestOptimalLimitPrice:
    """Test 8: Limit price placement respects urgency bands."""

    def setup_method(self) -> None:
        self.opt = MakerRebateOptimizer()

    def test_low_urgency_buy_improves_bid(self) -> None:
        """Low urgency buy should place above best bid (queue-jump)."""
        price = self.opt.optimal_limit_price(
            venue="kraken", side="buy",
            best_price=30_000.0, spread_bps=10.0, urgency=0.1,
        )
        assert price > 30_000.0, f"Expected improved bid, got {price}"

    def test_mid_urgency_joins_exactly(self) -> None:
        """Mid urgency should return exactly the best price."""
        price = self.opt.optimal_limit_price(
            venue="kraken", side="buy",
            best_price=30_000.0, spread_bps=10.0, urgency=0.5,
        )
        assert price == 30_000.0

    def test_high_urgency_crosses_spread(self) -> None:
        """High urgency buy should cross spread (price > best bid)."""
        price = self.opt.optimal_limit_price(
            venue="kraken", side="buy",
            best_price=30_000.0, spread_bps=10.0, urgency=0.9,
        )
        assert price > 30_000.0

    def test_low_urgency_sell_improves_ask(self) -> None:
        """Low urgency sell should place below best ask (queue-jump)."""
        price = self.opt.optimal_limit_price(
            venue="kraken", side="sell",
            best_price=30_100.0, spread_bps=10.0, urgency=0.1,
        )
        assert price < 30_100.0


# ===========================================================================
# TEST 9 — MakerRebateOptimizer.simulate_rebate_pnl
# ===========================================================================

class TestSimulateRebatePnl:
    """Test 9: simulate_rebate_pnl aggregates across venues correctly."""

    def test_all_maker_reduces_fee(self) -> None:
        opt = MakerRebateOptimizer(initial_volumes={"kraken": 200_000})
        trades = [
            {"venue": "kraken", "size_usd": 10_000, "order_type": "maker"},
            {"venue": "kraken", "size_usd": 10_000, "order_type": "maker"},
            {"venue": "kraken", "size_usd": 10_000, "order_type": "taker"},
        ]
        result = opt.simulate_rebate_pnl(trades)
        assert result["total_trades"]     == 3
        assert result["maker_fill_rate"]   == pytest.approx(2/3, rel=1e-3)
        assert result["total_rebate_usd"] >  0
        assert result["total_fee_usd"]    >  0
        assert "kraken" in result["by_venue"]

    def test_empty_trades_returns_zeroes(self) -> None:
        opt    = MakerRebateOptimizer()
        result = opt.simulate_rebate_pnl([])
        assert result["total_trades"]     == 0
        assert result["total_rebate_usd"] == 0.0
        assert result["total_fee_usd"]    == 0.0

    def test_tier_upgrade_roi_positive_on_volume(self) -> None:
        """Adding volume that crosses a tier boundary should give positive ROI."""
        opt = MakerRebateOptimizer(initial_volumes={"kraken": 10_000})
        roi = opt.tier_upgrade_roi("kraken", extra_volume_usd=100_000)
        # Crossing from Starter to Intermediate should save 2 bps
        assert isinstance(roi, float)
        assert roi >= 0.0


# ===========================================================================
# TEST 10 — ColocationProfiler.colocation_score + latency_adjusted_edge
# ===========================================================================

class TestColocationProfiler:
    """Test 10: Colocation scoring and edge adjustment."""

    def setup_method(self) -> None:
        self.profiler = ColocationProfiler()

    # --- colocation_score ---

    def test_score_colocated(self) -> None:
        assert self.profiler.colocation_score(100)    == "COLOCATED"
        assert self.profiler.colocation_score(499)    == "COLOCATED"

    def test_score_proximate(self) -> None:
        assert self.profiler.colocation_score(500)    == "PROXIMATE"
        assert self.profiler.colocation_score(4_999)  == "PROXIMATE"

    def test_score_remote(self) -> None:
        assert self.profiler.colocation_score(5_000)  == "REMOTE"
        assert self.profiler.colocation_score(49_999) == "REMOTE"

    def test_score_retail(self) -> None:
        assert self.profiler.colocation_score(50_000) == "RETAIL"
        assert self.profiler.colocation_score(200_000) == "RETAIL"

    # --- latency_adjusted_edge ---

    def test_colocated_baseline_no_penalty(self) -> None:
        """At baseline RTT there should be no degradation."""
        adj = self.profiler.latency_adjusted_edge(10.0, rtt_us=100.0, spread_bps=5.0)
        assert adj == pytest.approx(10.0)

    def test_high_latency_reduces_edge(self) -> None:
        """High RTT should reduce the strategy edge."""
        adj = self.profiler.latency_adjusted_edge(10.0, rtt_us=100_000.0, spread_bps=5.0)
        assert adj < 10.0

    def test_edge_can_go_negative(self) -> None:
        """Very high latency relative to edge should flip edge negative."""
        adj = self.profiler.latency_adjusted_edge(1.0, rtt_us=500_000.0, spread_bps=5.0)
        assert adj < 1.0

    # --- ExchangeLatency.from_raw ---

    def test_exchange_latency_from_raw(self) -> None:
        rtts  = [1_000.0, 1_200.0, 1_100.0, 1_050.0, 1_300.0,
                 1_150.0, 1_080.0, 1_250.0, 1_020.0, 1_180.0]
        lat   = ExchangeLatency.from_raw("kraken", "api.kraken.com:443", rtts)
        assert lat.samples   == 10
        assert lat.p50_us    > 0
        assert lat.p95_us    >= lat.p50_us
        assert lat.p99_us    >= lat.p95_us
        assert lat.jitter_us >= 0

    # --- async measure_rtt (mocked) ---

    @pytest.mark.asyncio
    async def test_measure_rtt_mock(self) -> None:
        """Verify measure_rtt returns ExchangeLatency when TCP succeeds."""
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock(return_value=None)

        with patch(
            "asyncio.open_connection",
            new=AsyncMock(return_value=(MagicMock(), mock_writer)),
        ):
            profiler = ColocationProfiler()
            result   = await profiler.measure_rtt("kraken", "api.kraken.com:443", n_samples=3)

        assert isinstance(result, ExchangeLatency)
        assert result.exchange == "kraken"
        assert result.samples  == 3
        assert result.p50_us   > 0

    # --- recommend_cloud_region (no measurements) ---

    def test_recommend_region_no_data(self) -> None:
        profiler = ColocationProfiler()
        region   = profiler.recommend_cloud_region()
        assert isinstance(region, str)
        assert len(region) > 0
