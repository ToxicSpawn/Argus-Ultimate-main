"""
Tests for compliance enforcement in the trade flow.

Covers:
  - AUSTRAC TTR auto-reporting on large trades
  - Wash sale pre-trade detection and blocking/warning
  - CGT acquisition/disposal recording on buy/sell
  - FX event recording with AUD conversion
  - Compliance audit trail integration
"""
from __future__ import annotations

import time
import pytest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeAuditChain:
    """Minimal audit chain that captures events for assertions."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def append_event(self, kind: str, payload: dict):
        self.events.append({"kind": kind, "payload": payload})
        return SimpleNamespace(kind=kind)

    def get_events(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        if kind is None:
            return list(self.events)
        return [e for e in self.events if e["kind"] == kind]


def _make_config(**overrides):
    """Build a minimal config namespace for ComponentRegistry."""
    defaults = {
        "aud_to_usd": 0.65,
        "entity_name": "ARGUS Test",
        "starting_capital_aud": 1000.0,
        "compliance_wash_sale_block": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_registry(config=None, with_audit=True):
    """Build a ComponentRegistry with AUSTRAC + ATO CGT initialised."""
    from core.component_registry import ComponentRegistry

    cfg = config or _make_config()
    reg = ComponentRegistry(cfg)

    # Manually init compliance components (skip full async initialize)
    from compliance.austrac import AUSTRACReporter
    from compliance.ato_cgt import ATOCapitalGainsTracker
    from pathlib import Path
    import tempfile

    reg.austrac = AUSTRACReporter(
        output_dir=Path(tempfile.mkdtemp()),
        entity_name="ARGUS Test",
    )
    reg.ato_cgt = ATOCapitalGainsTracker()

    if with_audit:
        audit = FakeAuditChain()
        reg.set_audit_chain(audit)
        return reg, audit
    return reg


def _make_trade_result(
    symbol="BTC/USD",
    side="buy",
    price=50000.0,
    quantity=0.1,
    exchange="kraken",
    order_id=None,
    **kwargs,
) -> Dict[str, Any]:
    """Build a trade_result dict matching what on_fill() expects."""
    return {
        "symbol": symbol,
        "side": side,
        "price": price,
        "quantity": quantity,
        "exchange": exchange,
        "order_id": order_id or f"test-{time.time():.0f}",
        "timestamp": kwargs.pop("timestamp", time.time()),
        "commission": kwargs.pop("commission", 0.0),
        "expected_price": price,
        **kwargs,
    }


# ===========================================================================
# AUSTRAC TTR Tests
# ===========================================================================

class TestAUSTRACTTR:
    """AUSTRAC Threshold Transaction Report enforcement."""

    def test_ttr_triggers_on_large_trade(self):
        """Trade with AUD value >= $10,000 must generate TTR + audit event."""
        reg, audit = _make_registry()
        # price=80000 * qty=0.1 = $8000 USD. At AUD/USD=0.65, AUD = 8000/0.65 ≈ 12,307
        trade = _make_trade_result(price=80000.0, quantity=0.1)
        reg.on_fill(trade)

        ttr_events = audit.get_events("compliance.austrac_ttr")
        assert len(ttr_events) == 1
        payload = ttr_events[0]["payload"]
        assert payload["amount_aud"] >= 10000
        assert payload["asset"] == "BTC"
        assert payload["threshold_aud"] == 10000.0

    def test_ttr_does_not_trigger_on_small_trade(self):
        """Trade with AUD value < $10,000 must NOT generate TTR event."""
        reg, audit = _make_registry()
        # price=50000 * qty=0.001 = $50 USD → ~$76.92 AUD — well under threshold
        trade = _make_trade_result(price=50000.0, quantity=0.001)
        reg.on_fill(trade)

        ttr_events = audit.get_events("compliance.austrac_ttr")
        assert len(ttr_events) == 0

    def test_ttr_triggers_on_sell(self):
        """SELL side also triggers TTR if above threshold."""
        reg, audit = _make_registry()
        trade = _make_trade_result(side="sell", price=80000.0, quantity=0.2)
        reg.on_fill(trade)

        ttr_events = audit.get_events("compliance.austrac_ttr")
        assert len(ttr_events) == 1
        assert ttr_events[0]["payload"]["direction"] == "SELL"

    def test_ttr_exact_threshold(self):
        """Trade at exactly $10,000 AUD must trigger TTR."""
        reg, audit = _make_registry()
        # Need AUD value = 10,000. With AUD/USD=0.65: USD = 10000 * 0.65 = 6500
        # price * qty = 6500; e.g. price=65000, qty=0.1
        trade = _make_trade_result(price=65000.0, quantity=0.1)
        reg.on_fill(trade)

        ttr_events = audit.get_events("compliance.austrac_ttr")
        assert len(ttr_events) == 1

    def test_ttr_records_transaction_in_austrac_reporter(self):
        """AUSTRAC reporter must have the transaction recorded."""
        reg, audit = _make_registry()
        trade = _make_trade_result(price=80000.0, quantity=0.2)
        reg.on_fill(trade)

        assert len(reg.austrac._transactions) == 1
        pending = reg.austrac.get_pending_ttrs()
        assert len(pending) == 1

    def test_ttr_not_created_for_sub_threshold_in_reporter(self):
        """AUSTRAC reporter must NOT create TTR for small trades."""
        reg, audit = _make_registry()
        trade = _make_trade_result(price=100.0, quantity=0.01)
        reg.on_fill(trade)

        assert len(reg.austrac._transactions) == 1
        pending = reg.austrac.get_pending_ttrs()
        assert len(pending) == 0

    def test_multiple_trades_multiple_ttrs(self):
        """Each large trade gets its own TTR event."""
        reg, audit = _make_registry()
        for i in range(3):
            trade = _make_trade_result(
                price=80000.0, quantity=0.2, order_id=f"big-{i}"
            )
            reg.on_fill(trade)

        ttr_events = audit.get_events("compliance.austrac_ttr")
        assert len(ttr_events) == 3


# ===========================================================================
# Wash Sale Tests
# ===========================================================================

class TestWashSale:
    """Wash sale (bed-and-breakfast) detection in pre-trade checks."""

    def _sell_at_loss(self, reg, symbol="BTC/USD", days_ago=10):
        """Helper: record a BUY then a SELL at a lower price to create a loss."""
        now = time.time()
        buy_ts = now - (days_ago + 5) * 86400
        sell_ts = now - days_ago * 86400

        # Buy at $60,000
        reg.ato_cgt.record_acquisition(
            asset="BTC", quantity=0.1, cost_base_aud=9230.77,
            timestamp=buy_ts, exchange="kraken",
        )
        # Sell at $50,000 (loss)
        reg.ato_cgt.record_disposal(
            asset="BTC", quantity=0.1, proceeds_aud=7692.31,
            timestamp=sell_ts, exchange="kraken",
        )

    def test_wash_sale_detected_on_recent_loss_rebuy(self):
        """Buying BTC after selling BTC at a loss within 30 days = wash sale warning."""
        reg, audit = _make_registry()
        self._sell_at_loss(reg, days_ago=10)

        result = reg.pre_order_check("BTC/USD", "buy", 5000.0)
        assert any("Wash sale risk" in r for r in result["reasons"])

    def test_wash_sale_not_triggered_for_different_symbol(self):
        """Selling BTC at a loss then buying ETH is NOT a wash sale."""
        reg, audit = _make_registry()
        self._sell_at_loss(reg, days_ago=10)

        result = reg.pre_order_check("ETH/USD", "buy", 5000.0)
        assert not any("Wash sale" in r for r in result["reasons"])

    def test_wash_sale_not_triggered_on_sell_side(self):
        """Wash sale check only applies to BUY side."""
        reg, audit = _make_registry()
        self._sell_at_loss(reg, days_ago=10)

        result = reg.pre_order_check("BTC/USD", "sell", 5000.0)
        assert not any("Wash sale" in r for r in result["reasons"])

    def test_wash_sale_warning_does_not_block_by_default(self):
        """Default config: wash sale warning but allow=True."""
        cfg = _make_config(compliance_wash_sale_block=False)
        reg, audit = _make_registry(config=cfg)
        self._sell_at_loss(reg, days_ago=10)

        result = reg.pre_order_check("BTC/USD", "buy", 5000.0)
        assert result["allow"] is True
        assert any("WARNING" in r for r in result["reasons"])

    def test_wash_sale_blocks_when_configured(self):
        """When compliance_wash_sale_block=True, wash sale blocks the trade."""
        cfg = _make_config(compliance_wash_sale_block=True)
        reg, audit = _make_registry(config=cfg)
        self._sell_at_loss(reg, days_ago=10)

        result = reg.pre_order_check("BTC/USD", "buy", 5000.0)
        assert result["allow"] is False
        assert any("BLOCKED" in r for r in result["reasons"])

    def test_wash_sale_not_triggered_outside_lookback_window(self):
        """Loss disposal > 30 days ago should NOT trigger wash sale."""
        reg, audit = _make_registry()
        self._sell_at_loss(reg, days_ago=35)

        result = reg.pre_order_check("BTC/USD", "buy", 5000.0)
        assert not any("Wash sale" in r for r in result["reasons"])

    def test_wash_sale_not_triggered_for_profit_disposal(self):
        """Selling at a profit then rebuying is NOT a wash sale."""
        reg, audit = _make_registry()
        now = time.time()
        # Buy at $50k, sell at $60k (profit)
        reg.ato_cgt.record_acquisition(
            asset="BTC", quantity=0.1, cost_base_aud=7692.31,
            timestamp=now - 15 * 86400, exchange="kraken",
        )
        reg.ato_cgt.record_disposal(
            asset="BTC", quantity=0.1, proceeds_aud=9230.77,
            timestamp=now - 5 * 86400, exchange="kraken",
        )

        result = reg.pre_order_check("BTC/USD", "buy", 5000.0)
        assert not any("Wash sale" in r for r in result["reasons"])

    def test_wash_sale_audit_trail_event(self):
        """Wash sale flag must be recorded to audit trail."""
        reg, audit = _make_registry()
        self._sell_at_loss(reg, days_ago=10)

        reg.pre_order_check("BTC/USD", "buy", 5000.0)

        ws_events = audit.get_events("compliance.wash_sale_flag")
        assert len(ws_events) == 1
        assert ws_events[0]["payload"]["symbol"] == "BTC"
        assert ws_events[0]["payload"]["action"] == "warned"

    def test_wash_sale_block_audit_trail_records_blocked(self):
        """When blocking, audit trail records action=blocked."""
        cfg = _make_config(compliance_wash_sale_block=True)
        reg, audit = _make_registry(config=cfg)
        self._sell_at_loss(reg, days_ago=10)

        reg.pre_order_check("BTC/USD", "buy", 5000.0)

        ws_events = audit.get_events("compliance.wash_sale_flag")
        assert len(ws_events) == 1
        assert ws_events[0]["payload"]["action"] == "blocked"


# ===========================================================================
# CGT Recording Tests
# ===========================================================================

class TestCGTRecording:
    """CGT acquisition/disposal auto-recording on buy/sell fills."""

    def test_cgt_acquisition_recorded_on_buy(self):
        """BUY fill must record an acquisition in ATO CGT tracker."""
        reg, audit = _make_registry()
        trade = _make_trade_result(side="buy", price=50000.0, quantity=0.1)
        reg.on_fill(trade)

        # Check CGT tracker has the acquisition
        assert "BTC" in reg.ato_cgt._acquisitions
        assert len(reg.ato_cgt._acquisitions["BTC"]) == 1
        acq = reg.ato_cgt._acquisitions["BTC"][0]
        assert acq.quantity == 0.1
        # 50000 * 0.1 / 0.65 ≈ 7692.31
        assert abs(acq.cost_base_aud - 7692.31) < 1.0

    def test_cgt_disposal_recorded_on_sell(self):
        """SELL fill must record a disposal in ATO CGT tracker."""
        reg, audit = _make_registry()
        # First buy to have acquisition lots
        buy_trade = _make_trade_result(
            side="buy", price=50000.0, quantity=0.1,
            timestamp=time.time() - 86400,
        )
        reg.on_fill(buy_trade)

        # Then sell
        sell_trade = _make_trade_result(side="sell", price=55000.0, quantity=0.1)
        reg.on_fill(sell_trade)

        assert len(reg.ato_cgt._disposals) == 1
        disposal = reg.ato_cgt._disposals[0]
        assert disposal.asset == "BTC"
        assert disposal.quantity == 0.1

    def test_cgt_acquisition_audit_event(self):
        """BUY fill must emit compliance.cgt_acquisition audit event."""
        reg, audit = _make_registry()
        trade = _make_trade_result(side="buy", price=50000.0, quantity=0.1)
        reg.on_fill(trade)

        acq_events = audit.get_events("compliance.cgt_acquisition")
        assert len(acq_events) == 1
        assert acq_events[0]["payload"]["asset"] == "BTC"
        assert acq_events[0]["payload"]["quantity"] == 0.1

    def test_cgt_disposal_audit_event(self):
        """SELL fill must emit compliance.cgt_disposal audit event."""
        reg, audit = _make_registry()
        # Buy first
        reg.on_fill(_make_trade_result(
            side="buy", price=50000.0, quantity=0.1,
            timestamp=time.time() - 86400,
        ))
        # Sell
        reg.on_fill(_make_trade_result(side="sell", price=55000.0, quantity=0.1))

        disp_events = audit.get_events("compliance.cgt_disposal")
        assert len(disp_events) == 1
        payload = disp_events[0]["payload"]
        assert payload["asset"] == "BTC"
        assert "capital_gain_aud" in payload

    def test_cgt_capital_gain_computed(self):
        """Disposal must compute capital gain correctly."""
        reg, audit = _make_registry()
        now = time.time()
        # Buy at $50k → AUD 7692.31
        reg.on_fill(_make_trade_result(
            side="buy", price=50000.0, quantity=0.1,
            timestamp=now - 86400,
        ))
        # Sell at $60k → AUD 9230.77
        reg.on_fill(_make_trade_result(
            side="sell", price=60000.0, quantity=0.1,
            timestamp=now,
        ))

        disposal = reg.ato_cgt._disposals[0]
        # Gain should be positive (sold higher than bought)
        assert disposal.capital_gain_aud > 0


# ===========================================================================
# FX Event Tests
# ===========================================================================

class TestFXEvent:
    """FX gain/loss recording with AUD conversion."""

    def test_fx_event_recorded_on_buy(self):
        """BUY fill must emit compliance.fx_event audit event."""
        reg, audit = _make_registry()
        trade = _make_trade_result(side="buy", price=50000.0, quantity=0.1)
        reg.on_fill(trade)

        fx_events = audit.get_events("compliance.fx_event")
        assert len(fx_events) == 1
        payload = fx_events[0]["payload"]
        assert payload["amount_usd"] == 5000.0
        assert payload["aud_usd_rate"] == 0.65
        assert payload["side"] == "buy"

    def test_fx_event_recorded_on_sell(self):
        """SELL fill must also emit compliance.fx_event."""
        reg, audit = _make_registry()
        # Buy first so disposal works
        reg.on_fill(_make_trade_result(
            side="buy", price=50000.0, quantity=0.1,
            timestamp=time.time() - 86400,
        ))
        reg.on_fill(_make_trade_result(side="sell", price=55000.0, quantity=0.1))

        fx_events = audit.get_events("compliance.fx_event")
        # One for buy, one for sell
        assert len(fx_events) == 2

    def test_fx_aud_equivalent_correct(self):
        """AUD equivalent must match price * qty / aud_to_usd."""
        reg, audit = _make_registry()
        trade = _make_trade_result(side="buy", price=50000.0, quantity=0.1)
        reg.on_fill(trade)

        fx_events = audit.get_events("compliance.fx_event")
        payload = fx_events[0]["payload"]
        expected_aud = 50000.0 * 0.1 / 0.65
        assert abs(payload["aud_equivalent"] - round(expected_aud, 2)) < 1.0


# ===========================================================================
# Compliance Audit Trail Tests
# ===========================================================================

class TestComplianceAuditTrail:
    """All compliance events must be recorded in the audit trail."""

    def test_cgt_events_in_audit_trail(self):
        """Both acquisition and disposal events appear in audit trail."""
        reg, audit = _make_registry()
        now = time.time()
        reg.on_fill(_make_trade_result(
            side="buy", price=50000.0, quantity=0.1,
            timestamp=now - 86400,
        ))
        reg.on_fill(_make_trade_result(
            side="sell", price=55000.0, quantity=0.1,
            timestamp=now,
        ))

        all_kinds = [e["kind"] for e in audit.events]
        assert "compliance.cgt_acquisition" in all_kinds
        assert "compliance.cgt_disposal" in all_kinds

    def test_austrac_ttr_in_audit_trail(self):
        """TTR threshold crossing appears in audit trail."""
        reg, audit = _make_registry()
        trade = _make_trade_result(price=80000.0, quantity=0.2)
        reg.on_fill(trade)

        all_kinds = [e["kind"] for e in audit.events]
        assert "compliance.austrac_ttr" in all_kinds

    def test_wash_sale_flag_in_audit_trail(self):
        """Wash sale flag appears in audit trail via pre_order_check."""
        reg, audit = _make_registry()
        now = time.time()
        reg.ato_cgt.record_acquisition(
            asset="ETH", quantity=1.0, cost_base_aud=5000.0,
            timestamp=now - 20 * 86400,
        )
        reg.ato_cgt.record_disposal(
            asset="ETH", quantity=1.0, proceeds_aud=3000.0,
            timestamp=now - 5 * 86400,
        )

        reg.pre_order_check("ETH/USD", "buy", 3000.0)

        all_kinds = [e["kind"] for e in audit.events]
        assert "compliance.wash_sale_flag" in all_kinds

    def test_no_audit_trail_without_chain(self):
        """Without audit chain, compliance events are silently skipped."""
        reg = _make_registry(with_audit=False)
        trade = _make_trade_result(price=80000.0, quantity=0.2)
        # Should not raise
        reg.on_fill(trade)
        # Verify AUSTRAC still records internally
        assert len(reg.austrac._transactions) == 1

    def test_compliance_events_prefixed(self):
        """All compliance audit events must be prefixed with 'compliance.'."""
        reg, audit = _make_registry()
        now = time.time()
        # Create wash sale scenario
        reg.ato_cgt.record_acquisition(
            asset="BTC", quantity=0.1, cost_base_aud=10000.0,
            timestamp=now - 20 * 86400,
        )
        reg.ato_cgt.record_disposal(
            asset="BTC", quantity=0.1, proceeds_aud=7000.0,
            timestamp=now - 5 * 86400,
        )
        # Trigger wash sale check
        reg.pre_order_check("BTC/USD", "buy", 5000.0)
        # Trigger TTR
        reg.on_fill(_make_trade_result(price=80000.0, quantity=0.2))

        for event in audit.events:
            assert event["kind"].startswith("compliance."), (
                f"Event kind '{event['kind']}' missing 'compliance.' prefix"
            )

    def test_fx_event_in_audit_trail(self):
        """FX events must appear in audit trail."""
        reg, audit = _make_registry()
        trade = _make_trade_result(side="buy", price=50000.0, quantity=0.1)
        reg.on_fill(trade)

        all_kinds = [e["kind"] for e in audit.events]
        assert "compliance.fx_event" in all_kinds


# ===========================================================================
# check_wash_sale_risk (ATOCapitalGainsTracker direct tests)
# ===========================================================================

class TestCheckWashSaleRisk:
    """Direct unit tests for ATOCapitalGainsTracker.check_wash_sale_risk()."""

    def test_returns_none_when_no_disposals(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        assert tracker.check_wash_sale_risk("BTC") is None

    def test_returns_none_for_profit_disposal(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        now = time.time()
        tracker.record_acquisition("BTC", 1.0, 5000.0, now - 20 * 86400)
        tracker.record_disposal("BTC", 1.0, 8000.0, now - 5 * 86400)
        assert tracker.check_wash_sale_risk("BTC") is None

    def test_returns_risk_for_loss_disposal(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        now = time.time()
        tracker.record_acquisition("BTC", 1.0, 8000.0, now - 20 * 86400)
        tracker.record_disposal("BTC", 1.0, 5000.0, now - 5 * 86400)
        result = tracker.check_wash_sale_risk("BTC")
        assert result is not None
        assert result["symbol"] == "BTC"
        assert result["potential_disallowed_loss"] == 3000.0
        assert result["days_since_disposal"] < 10

    def test_returns_none_for_old_loss(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        now = time.time()
        tracker.record_acquisition("BTC", 1.0, 8000.0, now - 60 * 86400)
        tracker.record_disposal("BTC", 1.0, 5000.0, now - 35 * 86400)
        assert tracker.check_wash_sale_risk("BTC") is None

    def test_different_asset_no_risk(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        now = time.time()
        tracker.record_acquisition("BTC", 1.0, 8000.0, now - 20 * 86400)
        tracker.record_disposal("BTC", 1.0, 5000.0, now - 5 * 86400)
        assert tracker.check_wash_sale_risk("ETH") is None
