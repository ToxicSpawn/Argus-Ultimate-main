"""Tests for 10GbE network engine — Solarflare SFN8522-PLUS."""
import struct
import unittest
from core.network_10gbe import (
    NetworkConfig, LatencyStats, SolarflareOptimizer,
    DataChannelProtocol, NetworkHealthMonitor,
)


class TestNetworkConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = NetworkConfig()
        self.assertEqual(cfg.rt_port, 9100)
        self.assertEqual(cfg.bulk_port, 9200)
        self.assertEqual(cfg.bulk_mtu, 9000)
        self.assertTrue(cfg.tcp_nodelay)

    def test_dual_channel_ips(self):
        cfg = NetworkConfig()
        self.assertNotEqual(cfg.rt_local_ip, cfg.bulk_local_ip)
        self.assertNotEqual(cfg.rt_remote_ip, cfg.bulk_remote_ip)


class TestLatencyStats(unittest.TestCase):
    def test_record_and_stats(self):
        ls = LatencyStats()
        for i in range(100):
            ls.record(float(i + 5))
        self.assertEqual(ls.samples, 100)
        self.assertAlmostEqual(ls.min_us, 5.0)
        self.assertAlmostEqual(ls.max_us, 104.0)
        self.assertGreater(ls.avg_us, 0)
        self.assertGreater(ls.p99_us, 0)

    def test_jitter(self):
        ls = LatencyStats()
        # Constant latency = 0 jitter
        for _ in range(50):
            ls.record(10.0)
        self.assertLess(ls.jitter_us, 0.1)

        # Variable latency = high jitter
        ls2 = LatencyStats()
        for i in range(50):
            ls2.record(5.0 if i % 2 == 0 else 15.0)
        self.assertGreater(ls2.jitter_us, 5.0)


class TestDataChannelProtocol(unittest.TestCase):
    def test_encode_decode_header(self):
        payload = b"hello"
        encoded = DataChannelProtocol.encode(DataChannelProtocol.MSG_TICK, payload)
        msg_type, length, ts = DataChannelProtocol.decode_header(encoded)
        self.assertEqual(msg_type, DataChannelProtocol.MSG_TICK)
        self.assertEqual(length, len(payload))
        self.assertGreater(ts, 0)

    def test_header_size(self):
        self.assertEqual(DataChannelProtocol.HEADER_SIZE, 13)

    def test_encode_tick(self):
        data = DataChannelProtocol.encode_tick("BTC/USD", 50000.0, 0.1, "buy")
        msg_type, length, ts = DataChannelProtocol.decode_header(data)
        self.assertEqual(msg_type, DataChannelProtocol.MSG_TICK)
        self.assertGreater(length, 0)
        # Payload should be valid JSON
        import json
        payload = data[DataChannelProtocol.HEADER_SIZE:]
        parsed = json.loads(payload)
        self.assertEqual(parsed["s"], "BTC/USD")
        self.assertAlmostEqual(parsed["p"], 50000.0)

    def test_encode_heartbeat(self):
        data = DataChannelProtocol.encode_heartbeat()
        msg_type, length, ts = DataChannelProtocol.decode_header(data)
        self.assertEqual(msg_type, DataChannelProtocol.MSG_HEARTBEAT)

    def test_all_message_types_unique(self):
        types = [
            DataChannelProtocol.MSG_TICK,
            DataChannelProtocol.MSG_OHLCV,
            DataChannelProtocol.MSG_ORDER,
            DataChannelProtocol.MSG_BACKTEST_REQ,
            DataChannelProtocol.MSG_BACKTEST_RES,
            DataChannelProtocol.MSG_MODEL_SYNC,
            DataChannelProtocol.MSG_EVOLUTION,
            DataChannelProtocol.MSG_HEARTBEAT,
        ]
        self.assertEqual(len(types), len(set(types)))


class TestSolarflareOptimizer(unittest.TestCase):
    def test_create_rt_socket(self):
        opt = SolarflareOptimizer()
        sock = opt.create_rt_socket()
        self.assertIsNotNone(sock)
        sock.close()

    def test_create_bulk_socket(self):
        opt = SolarflareOptimizer()
        sock = opt.create_bulk_socket()
        self.assertIsNotNone(sock)
        sock.close()

    def test_optimize_system(self):
        opt = SolarflareOptimizer()
        results = opt.optimize_system()
        self.assertGreater(len(results), 0)
        self.assertTrue(opt._optimized)

    def test_get_stats(self):
        opt = SolarflareOptimizer()
        stats = opt.get_stats()
        self.assertIn("config", stats)
        self.assertIn("latency", stats)

    def test_latency_unreachable(self):
        opt = SolarflareOptimizer()
        lat = opt.measure_latency("192.168.255.255", 1)  # unreachable
        self.assertEqual(lat, -1.0)


class TestNetworkHealthMonitor(unittest.TestCase):
    def test_check_returns_status(self):
        mon = NetworkHealthMonitor()
        result = mon.check()
        self.assertIn("rt_channel", result)
        self.assertIn("bulk_channel", result)
        self.assertIn("both_healthy", result)

    def test_unhealthy_when_down(self):
        cfg = NetworkConfig(rt_remote_ip="192.168.255.255", bulk_remote_ip="192.168.255.255")
        mon = NetworkHealthMonitor(config=cfg)
        result = mon.check()
        self.assertFalse(result["both_healthy"])
        self.assertFalse(mon.healthy)

    def test_get_stats(self):
        mon = NetworkHealthMonitor()
        stats = mon.get_stats()
        self.assertIn("rt_healthy", stats)
        self.assertIn("checks", stats)


if __name__ == "__main__":
    unittest.main()
