"""
Integration tests for current production paths.

Tests the unified architecture: config_manager, audit_trail, risk_manager,
exchange_manager, trade_ledger. Replaces the previous file that depended on
pre-restructuring modules (core.data_feed, core.execution_unified, etc.)
which were moved to archive during restructuring.
"""
from __future__ import annotations

import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Audit trail chain integrity
# ---------------------------------------------------------------------------

def test_audit_trail_chain_integrity():
    """10 sequential appends must produce a valid, unbroken hash chain."""
    from monitoring.audit_trail import AuditTrail

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        trail = AuditTrail(db_path=db_path)
        for i in range(10):
            trail.append("test_event", {"index": i, "value": f"item_{i}"})
        result = trail.verify_chain()
        assert result["ok"] is True, f"Chain invalid: {result}"
        assert result["total"] == 10
        assert result["first_bad_seq"] is None
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_audit_trail_tamper_detection():
    """Directly corrupting a row must be detected by verify_chain."""
    import sqlite3
    from monitoring.audit_trail import AuditTrail

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        trail = AuditTrail(db_path=db_path)
        for i in range(5):
            trail.append("event", {"i": i})

        # Tamper: directly overwrite a payload in the DB
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE audit_events SET payload_json = '{\"tampered\": true}' WHERE seq = 3")
        conn.commit()
        conn.close()

        result = trail.verify_chain()
        assert result["ok"] is False
        assert result["first_bad_seq"] == 3
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Risk manager initialization
# ---------------------------------------------------------------------------

def test_risk_manager_initializes_with_capital():
    """Risk manager must accept and store initial capital."""
    from risk.unified_risk_manager import UnifiedRiskManager

    rm = UnifiedRiskManager(initial_capital=5000.0)
    assert rm.initial_capital == 5000.0
    assert rm.current_capital == 5000.0


def test_risk_manager_records_trade_loss():
    """Recording a loss trade must increment consecutive_losses."""
    from risk.unified_risk_manager import UnifiedRiskManager

    rm = UnifiedRiskManager(initial_capital=1000.0)
    rm.record_trade(pnl=-20.0)
    assert rm.consecutive_losses == 1


def test_risk_manager_resets_consecutive_losses_on_win():
    from risk.unified_risk_manager import UnifiedRiskManager

    rm = UnifiedRiskManager(initial_capital=1000.0)
    rm.record_trade(pnl=-10.0)
    rm.record_trade(pnl=-10.0)
    assert rm.consecutive_losses == 2
    rm.record_trade(pnl=50.0)
    assert rm.consecutive_losses == 0


# ---------------------------------------------------------------------------
# Trade ledger
# ---------------------------------------------------------------------------

def test_trade_ledger_records_and_retrieves():
    """record_trade followed by get_trades must return the inserted trade."""
    from monitoring.trade_ledger import TradeLedger

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        ledger = TradeLedger(db_path=db_path)
        ledger.record_trade({
            "order_id": "test-123",
            "symbol": "BTC/USD",
            "side": "buy",
            "size": 0.01,
            "price": 50000.0,
            "status": "filled",
        })
        trades = ledger.get_trades(symbol="BTC/USD", limit=10)
        assert len(trades) >= 1
        assert trades[0]["symbol"] == "BTC/USD"
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Exchange manager
# ---------------------------------------------------------------------------

def test_exchange_manager_failover_no_connected_attr():
    """Verify ExchangeManager can be constructed with AdapterRegistry API."""
    from unittest.mock import MagicMock
    from core.exchange_manager import ExchangeManager

    mock_registry = MagicMock()
    mgr = ExchangeManager(registry=mock_registry)
    # Verify the new API surface exists
    assert hasattr(mgr, "get_adapter")
    assert mgr is not None


# ---------------------------------------------------------------------------
# Config manager
# ---------------------------------------------------------------------------

def test_config_manager_loads_unified_config():
    """resolve_unified_config_path must return a path to the unified config."""
    from core.config_manager import resolve_unified_config_path
    path = resolve_unified_config_path()
    assert path.exists(), f"unified_config.yaml not found at {path}"
    assert path.suffix in {".yaml", ".yml"}
