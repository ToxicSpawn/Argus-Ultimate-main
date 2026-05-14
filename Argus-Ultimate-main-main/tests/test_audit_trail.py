"""
Tests for monitoring/audit_trail.py

Covers: hash chain integrity, concurrent append safety, verify_chain,
tamper detection, and query deserialization.
"""
from __future__ import annotations

import sqlite3
import threading

import pytest


def _make_trail(db_path: str):
    """Create an AuditTrail backed by the given path."""
    from monitoring.audit_trail import AuditTrail
    return AuditTrail(db_path=db_path)


class TestAuditTrailBasic:
    def test_append_returns_hash(self, tmp_path):
        trail = _make_trail(str(tmp_path / "audit.db"))
        result = trail.append("order", {"symbol": "BTC/USD", "side": "buy"})
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_sequential_appends_verify(self, tmp_path):
        trail = _make_trail(str(tmp_path / "audit.db"))
        for i in range(20):
            trail.append("fill", {"i": i})
        r = trail.verify_chain()
        assert r["ok"] is True
        assert r["total"] == 20

    def test_empty_chain_verifies(self, tmp_path):
        trail = _make_trail(str(tmp_path / "audit.db"))
        r = trail.verify_chain()
        assert r["ok"] is True
        assert r["total"] == 0


class TestAuditTrailTamperDetection:
    def test_payload_tamper_detected(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        trail = _make_trail(db_path)
        for i in range(5):
            trail.append("event", {"i": i})
        # Directly tamper with row at seq=3
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE audit_events SET payload_json = '{\"x\": 999}' WHERE seq = 3")
        conn.commit()
        conn.close()
        r = trail.verify_chain()
        assert r["ok"] is False
        assert r["first_bad_seq"] == 3
        assert r["error"] == "hash_mismatch"

    def test_seq_deletion_breaks_chain(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        trail = _make_trail(db_path)
        for i in range(5):
            trail.append("event", {"i": i})
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM audit_events WHERE seq = 2")
        conn.commit()
        conn.close()
        r = trail.verify_chain()
        # Chain is broken because prev_hash of seq=3 won't match seq=1's hash
        assert r["ok"] is False


class TestAuditTrailConcurrency:
    def test_concurrent_appends_produce_valid_chain(self, tmp_path):
        """50 concurrent threaded appends must produce no seq gaps or chain breaks."""
        trail = _make_trail(str(tmp_path / "audit.db"))
        errors = []

        def append_fn(i: int):
            try:
                trail.append("concurrent", {"i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=append_fn, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent appends: {errors}"
        r = trail.verify_chain()
        assert r["ok"] is True, f"Chain invalid after concurrent writes: {r}"
        assert r["total"] == 50


class TestAuditTrailQuery:
    def test_query_returns_events(self, tmp_path):
        trail = _make_trail(str(tmp_path / "audit.db"))
        trail.append("order", {"sym": "BTC/USD"})
        trail.append("fill", {"sym": "ETH/USD"})
        trail.append("order", {"sym": "SOL/USD"})
        results = trail.query(kind="order")
        assert len(results) == 2
        for r in results:
            assert r["kind"] == "order"
            assert isinstance(r["payload"], dict)

    def test_query_handles_corrupted_json_gracefully(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        trail = _make_trail(db_path)
        trail.append("event", {"k": "v"})
        # Corrupt the JSON directly
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE audit_events SET payload_json = 'NOT_JSON' WHERE seq = 1")
        conn.commit()
        conn.close()
        # Should not raise — returns empty dict for corrupted payload
        results = trail.query()
        assert len(results) == 1
        assert results[0]["payload"] == {}
