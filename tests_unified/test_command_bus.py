from __future__ import annotations

import tempfile
import time
import unittest
import shutil

from execution.command_bus import (
    LocalInstructionBus,
    deterministic_instruction_id,
    instruction_to_signal,
    sign_instruction_payload,
    validate_instruction_payload,
    verify_instruction_payload,
)


class TestCommandBus(unittest.TestCase):
    def test_sign_verify_and_tamper_detection(self) -> None:
        payload = {
            "run_id": "run_a",
            "trace_id": "trace_a",
            "cycle_id": 1,
            "symbol": "BTC/USD",
            "action": "BUY",
            "quantity": 0.1,
            "entry_price": 100.0,
            "expires_ts": time.time() + 10.0,
        }
        secret = "test_secret"
        sig = sign_instruction_payload(payload, secret)
        self.assertTrue(verify_instruction_payload(payload, sig, secret))

        tampered = dict(payload)
        tampered["quantity"] = 0.2
        self.assertFalse(verify_instruction_payload(tampered, sig, secret))

    def test_publish_claim_consume_and_reject_flow(self) -> None:
        td = tempfile.mkdtemp()
        try:
            db_path = f"{td}/command_bus.db"
            bus = LocalInstructionBus(db_path=db_path, queue="q1")

            payload_1 = {
                "run_id": "run_1",
                "trace_id": "trace_1",
                "cycle_id": 1,
                "symbol": "BTC/USD",
                "action": "BUY",
                "quantity": 0.01,
                "entry_price": 100.0,
                "expires_ts": time.time() + 60.0,
            }
            iid_1 = deterministic_instruction_id(payload_1)
            bus.publish(payload=payload_1, instruction_id=iid_1, signature="sig1")
            # duplicate publish should be ignored by PK constraint
            bus.publish(payload=payload_1, instruction_id=iid_1, signature="sig1")
            metrics = bus.metrics()
            self.assertEqual(metrics.get("PENDING"), 1)

            claimed_1 = bus.claim_pending(limit=10)
            self.assertEqual(len(claimed_1), 1)
            self.assertEqual(claimed_1[0]["instruction_id"], iid_1)
            bus.mark_consumed(iid_1)
            metrics = bus.metrics()
            self.assertEqual(metrics.get("CONSUMED"), 1)
            self.assertEqual(metrics.get("PENDING"), 0)

            payload_2 = {
                "run_id": "run_2",
                "trace_id": "trace_2",
                "cycle_id": 2,
                "symbol": "ETH/USD",
                "action": "SELL",
                "quantity": 0.02,
                "entry_price": 200.0,
                "expires_ts": time.time() + 60.0,
            }
            iid_2 = deterministic_instruction_id(payload_2)
            bus.publish(payload=payload_2, instruction_id=iid_2, signature="sig2")
            claimed_2 = bus.claim_pending(limit=10)
            self.assertEqual(len(claimed_2), 1)
            self.assertEqual(claimed_2[0]["instruction_id"], iid_2)
            bus.mark_rejected(iid_2, "invalid_signature")
            metrics = bus.metrics()
            self.assertEqual(metrics.get("REJECTED"), 1)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_validation_stale_and_max_notional(self) -> None:
        now_ts = time.time()
        stale_payload = {
            "symbol": "BTC/USD",
            "action": "BUY",
            "quantity": 1.0,
            "entry_price": 100.0,
            "expires_ts": now_ts - 1.0,
        }
        stale = validate_instruction_payload(stale_payload, now_ts=now_ts)
        self.assertFalse(stale.ok)
        self.assertEqual(stale.reason, "stale_instruction")

        too_large_payload = {
            "symbol": "BTC/USD",
            "action": "BUY",
            "quantity": 1.0,
            "entry_price": 100.0,
            "expires_ts": now_ts + 10.0,
        }
        too_large = validate_instruction_payload(
            too_large_payload,
            now_ts=now_ts,
            max_notional_aud=120.0,
            aud_to_usd=0.65,
        )
        self.assertFalse(too_large.ok)
        self.assertEqual(too_large.reason, "max_notional_exceeded")

    def test_instruction_to_signal_conversion(self) -> None:
        payload = {
            "symbol": "BTC/USD",
            "action": "BUY",
            "quantity": 0.01,
            "entry_price": 100.0,
            "confidence": 0.8,
            "strategy": "s1",
            "trace_id": "trace_1",
        }
        sig, reason = instruction_to_signal(payload)
        self.assertIsNotNone(sig)
        self.assertEqual(reason, "ok")
        self.assertEqual(getattr(sig, "side", ""), "BUY")

        bad_payload = dict(payload)
        bad_payload["action"] = "HOLD"
        sig2, reason2 = instruction_to_signal(bad_payload)
        self.assertIsNone(sig2)
        self.assertEqual(reason2, "invalid_signal_payload")


if __name__ == "__main__":
    unittest.main()
