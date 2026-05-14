"""Tests for latency compensator."""
import time
import unittest
from core.latency_compensator import LatencyCompensator, LatencyCompensation


class TestLatencyCompensator(unittest.TestCase):
    def test_fresh_signal_executes(self):
        lc = LatencyCompensator(expected_rtt_ms=280)
        now_ms = time.time() * 1000
        comp = lc.compensate(now_ms - 50, current_volatility=0.02)
        self.assertFalse(comp.is_stale)
        self.assertIn(comp.recommendation, ("EXECUTE", "WIDEN_LIMIT", "REDUCE_AND_EXECUTE"))

    def test_stale_signal_skipped(self):
        lc = LatencyCompensator(expected_rtt_ms=280, max_signal_age_ms=500)
        old_ms = time.time() * 1000 - 600
        comp = lc.compensate(old_ms)
        self.assertTrue(comp.is_stale)
        self.assertEqual(comp.recommendation, "SKIP_STALE")

    def test_high_latency_reduces_size(self):
        lc = LatencyCompensator(expected_rtt_ms=400)
        now_ms = time.time() * 1000
        comp = lc.compensate(now_ms - 100)
        self.assertLess(comp.size_multiplier, 1.0)

    def test_low_latency_full_size(self):
        lc = LatencyCompensator(expected_rtt_ms=5)
        now_ms = time.time() * 1000
        comp = lc.compensate(now_ms)
        self.assertGreater(comp.size_multiplier, 0.9)

    def test_limit_offset_wider_with_latency(self):
        lc_fast = LatencyCompensator(expected_rtt_ms=5)
        lc_slow = LatencyCompensator(expected_rtt_ms=400)
        now_ms = time.time() * 1000
        comp_fast = lc_fast.compensate(now_ms, base_spread_bps=2.0)
        comp_slow = lc_slow.compensate(now_ms, base_spread_bps=2.0)
        self.assertGreater(comp_slow.limit_offset_bps, comp_fast.limit_offset_bps)

    def test_record_rtt_updates(self):
        lc = LatencyCompensator(expected_rtt_ms=280)
        for _ in range(10):
            lc.record_rtt(150.0)
        self.assertLess(lc._current_rtt, 280)

    def test_order_type_high_latency(self):
        lc = LatencyCompensator()
        self.assertEqual(lc.get_optimal_order_type("low", 300), "limit")
        self.assertEqual(lc.get_optimal_order_type("critical", 450), "market")

    def test_order_type_low_latency(self):
        lc = LatencyCompensator()
        self.assertEqual(lc.get_optimal_order_type("low", 5), "limit")

    def test_get_stats(self):
        lc = LatencyCompensator()
        lc.record_rtt(280)
        stats = lc.get_stats()
        self.assertEqual(stats["samples"], 1)
        self.assertGreater(stats["current_rtt_ms"], 0)


if __name__ == "__main__":
    unittest.main()
