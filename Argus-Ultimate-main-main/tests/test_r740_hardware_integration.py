"""
Tests for R740 hardware-specific integration:
  1. docker-compose.r740.yml — NVMe RAID1 + SATA bulk volume split
  2. core/network_10gbe.py — 25GbE link detection and config
  3. infra/setup_r740.sh — BOSS + NVMe + SATA verification block
"""
from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml


ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────────────────
# docker-compose.r740.yml — NVMe/SATA split
# ──────────────────────────────────────────────────────────────────────────────

class TestDockerComposeStorageTiers(unittest.TestCase):
    """Verify hot data is on NVMe, bulk data is on SATA."""

    @classmethod
    def setUpClass(cls):
        with open(ROOT / "docker-compose.r740.yml", "r", encoding="utf-8") as f:
            cls.compose = yaml.safe_load(f)

    def _volume_device(self, volume_name: str) -> str:
        vol = self.compose["volumes"].get(volume_name, {})
        return vol.get("driver_opts", {}).get("device", "")

    def test_timescaledb_on_nvme(self):
        device = self._volume_device("timescaledb_data")
        self.assertIn("ARGUS_NVME_MOUNT", device)
        self.assertIn("timescaledb", device)

    def test_redis_on_nvme(self):
        device = self._volume_device("redis_data")
        self.assertIn("ARGUS_NVME_MOUNT", device)

    def test_kafka_on_nvme(self):
        device = self._volume_device("kafka_data")
        self.assertIn("ARGUS_NVME_MOUNT", device)

    def test_zookeeper_on_nvme(self):
        device = self._volume_device("zookeeper_data")
        self.assertIn("ARGUS_NVME_MOUNT", device)

    def test_ollama_on_nvme(self):
        device = self._volume_device("ollama_data")
        self.assertIn("ARGUS_NVME_MOUNT", device)

    def test_prometheus_on_sata(self):
        device = self._volume_device("prometheus_data")
        self.assertIn("ARGUS_SATA_MOUNT", device)

    def test_grafana_on_sata(self):
        device = self._volume_device("grafana_data")
        self.assertIn("ARGUS_SATA_MOUNT", device)

    def test_loki_on_sata(self):
        device = self._volume_device("loki_data")
        self.assertIn("ARGUS_SATA_MOUNT", device)

    def test_minio_on_sata(self):
        device = self._volume_device("minio_data")
        self.assertIn("ARGUS_SATA_MOUNT", device)

    def test_mlflow_on_sata(self):
        device = self._volume_device("mlflow_data")
        self.assertIn("ARGUS_SATA_MOUNT", device)

    def test_nvme_mount_has_default(self):
        """Mount path should have a default value like ${ARGUS_NVME_MOUNT:-/mnt/nvme}."""
        device = self._volume_device("timescaledb_data")
        self.assertRegex(device, r"\$\{ARGUS_NVME_MOUNT:-/mnt/nvme\}")

    def test_sata_mount_has_default(self):
        device = self._volume_device("prometheus_data")
        self.assertRegex(device, r"\$\{ARGUS_SATA_MOUNT:-/mnt/sata\}")

    def test_hot_data_services_never_on_sata(self):
        """TimescaleDB, Redis, Kafka, Zookeeper must NEVER be on SATA."""
        hot_volumes = ["timescaledb_data", "redis_data", "kafka_data", "zookeeper_data"]
        for vol in hot_volumes:
            device = self._volume_device(vol)
            self.assertNotIn("ARGUS_SATA_MOUNT", device,
                             f"{vol} must be on NVMe, not SATA")

    def test_storage_topology_in_header_comment(self):
        """Header comment should document the storage topology."""
        text = (ROOT / "docker-compose.r740.yml").read_text(encoding="utf-8")
        self.assertIn("BOSS card", text)
        self.assertIn("NVMe", text)
        self.assertIn("SATA", text)
        self.assertIn("RAID1", text)


# ──────────────────────────────────────────────────────────────────────────────
# core/network_10gbe.py — 25GbE detection
# ──────────────────────────────────────────────────────────────────────────────

class TestLinkInfo(unittest.TestCase):
    """Test the LinkInfo dataclass tiers and helpers."""

    def _make_info(self, speed_mbps: int):
        from core.network_10gbe import LinkInfo
        return LinkInfo(
            interface="ens1f0",
            speed_mbps=speed_mbps,
            link_up=True,
        )

    def test_10gbe_tier(self):
        info = self._make_info(10000)
        self.assertEqual(info.tier, "10GbE")
        self.assertTrue(info.is_10gbe)
        self.assertFalse(info.is_25gbe)

    def test_25gbe_tier(self):
        info = self._make_info(25000)
        self.assertEqual(info.tier, "25GbE")
        self.assertFalse(info.is_10gbe)
        self.assertTrue(info.is_25gbe)

    def test_40gbe_tier(self):
        info = self._make_info(40000)
        self.assertEqual(info.tier, "40GbE")
        self.assertFalse(info.is_10gbe)
        # is_25gbe means "25GbE or higher" — 40GbE qualifies
        self.assertTrue(info.is_25gbe)

    def test_100gbe_tier(self):
        info = self._make_info(100000)
        self.assertEqual(info.tier, "100GbE")

    def test_1gbe_tier(self):
        info = self._make_info(1000)
        self.assertEqual(info.tier, "1GbE")

    def test_sub_gigabit_tier(self):
        info = self._make_info(100)
        self.assertEqual(info.tier, "100Mbps")

    def test_speed_gbps_conversion(self):
        info = self._make_info(25000)
        self.assertEqual(info.speed_gbps, 25.0)


class TestDetectLinkSpeed(unittest.TestCase):
    """Test link speed detection with mocked ethtool output."""

    def test_non_posix_returns_stub(self):
        from core.network_10gbe import detect_link_speed
        with patch("os.name", "nt"):
            info = detect_link_speed("ens1f0")
        self.assertEqual(info.speed_mbps, 0)
        self.assertFalse(info.link_up)

    def test_ethtool_not_installed(self):
        from core.network_10gbe import detect_link_speed
        with patch("os.name", "posix"), patch("shutil.which", return_value=None):
            info = detect_link_speed("ens1f0")
        self.assertEqual(info.speed_mbps, 0)

    def test_parses_25gbe_output(self):
        from core.network_10gbe import detect_link_speed
        ethtool_25g = """Settings for ens1f0:
        Supported ports: [ FIBRE ]
        Speed: 25000Mb/s
        Duplex: Full
        Port: Direct Attach Copper
        Link detected: yes
"""
        ethtool_i = "driver: ice\nversion: 1.11.14\n"
        with patch("os.name", "posix"), \
             patch("shutil.which", return_value="/usr/sbin/ethtool"), \
             patch("subprocess.run") as run:
            run.side_effect = [
                MagicMock(stdout=ethtool_25g, returncode=0),
                MagicMock(stdout=ethtool_i, returncode=0),
            ]
            info = detect_link_speed("ens1f0")

        self.assertEqual(info.speed_mbps, 25000)
        self.assertTrue(info.link_up)
        self.assertEqual(info.duplex, "full")
        self.assertEqual(info.driver, "ice")
        self.assertTrue(info.is_25gbe)

    def test_parses_10gbe_output(self):
        from core.network_10gbe import detect_link_speed
        ethtool_10g = "Speed: 10000Mb/s\nLink detected: yes\nDuplex: Full\n"
        ethtool_i = "driver: sfc\n"
        with patch("os.name", "posix"), \
             patch("shutil.which", return_value="/usr/sbin/ethtool"), \
             patch("subprocess.run") as run:
            run.side_effect = [
                MagicMock(stdout=ethtool_10g, returncode=0),
                MagicMock(stdout=ethtool_i, returncode=0),
            ]
            info = detect_link_speed("ens1f0")

        self.assertEqual(info.speed_mbps, 10000)
        self.assertTrue(info.is_10gbe)
        self.assertFalse(info.is_25gbe)
        self.assertEqual(info.driver, "sfc")

    def test_handles_link_down(self):
        from core.network_10gbe import detect_link_speed
        ethtool_down = "Speed: Unknown!\nLink detected: no\n"
        ethtool_i = "driver: i40e\n"
        with patch("os.name", "posix"), \
             patch("shutil.which", return_value="/usr/sbin/ethtool"), \
             patch("subprocess.run") as run:
            run.side_effect = [
                MagicMock(stdout=ethtool_down, returncode=0),
                MagicMock(stdout=ethtool_i, returncode=0),
            ]
            info = detect_link_speed("ens1f0")

        self.assertFalse(info.link_up)


class TestNetworkConfigAutodetect(unittest.TestCase):
    """Test NetworkConfig.autodetect() scales with link speed."""

    def test_default_config_10gbe_buffer(self):
        from core.network_10gbe import NetworkConfig
        cfg = NetworkConfig()
        self.assertEqual(cfg.link_speed_mbps, 10000)
        self.assertFalse(cfg.is_25gbe)
        self.assertEqual(cfg.socket_buffer_size, 16 * 1024 * 1024)

    def test_25gbe_config_has_larger_buffer(self):
        from core.network_10gbe import NetworkConfig
        cfg = NetworkConfig(link_speed_mbps=25000)
        self.assertTrue(cfg.is_25gbe)
        self.assertEqual(cfg.socket_buffer_size, 32 * 1024 * 1024)

    def test_autodetect_with_mocked_detector(self):
        from core.network_10gbe import NetworkConfig, LinkInfo
        mock_info = LinkInfo(interface="ens1f0", speed_mbps=25000, link_up=True,
                             duplex="full", driver="ice")
        with patch("core.network_10gbe.detect_primary_link", return_value=mock_info):
            cfg = NetworkConfig.autodetect()
        self.assertEqual(cfg.link_speed_mbps, 25000)
        self.assertEqual(cfg.rt_interface, "ens1f0")
        self.assertEqual(cfg.rt_mtu, 9000)  # jumbo frames at 25GbE

    def test_autodetect_with_10gbe_keeps_standard_mtu(self):
        from core.network_10gbe import NetworkConfig, LinkInfo
        mock_info = LinkInfo(interface="ens1f0", speed_mbps=10000, link_up=True,
                             duplex="full", driver="sfc")
        with patch("core.network_10gbe.detect_primary_link", return_value=mock_info):
            cfg = NetworkConfig.autodetect()
        self.assertEqual(cfg.rt_mtu, 1500)  # standard for 10GbE real-time

    def test_autodetect_overrides(self):
        from core.network_10gbe import NetworkConfig, LinkInfo
        mock_info = LinkInfo(interface="ens1f0", speed_mbps=25000, link_up=True)
        with patch("core.network_10gbe.detect_primary_link", return_value=mock_info):
            cfg = NetworkConfig.autodetect(rt_port=9999)
        self.assertEqual(cfg.rt_port, 9999)
        self.assertEqual(cfg.link_speed_mbps, 25000)


class TestSolarflareOptimizerScalesWithSpeed(unittest.TestCase):
    """Test that socket buffer sizes scale with link speed."""

    def test_stats_exposes_link_speed(self):
        from core.network_10gbe import SolarflareOptimizer, NetworkConfig
        opt = SolarflareOptimizer(NetworkConfig(link_speed_mbps=25000))
        stats = opt.get_stats()
        self.assertEqual(stats["config"]["link_speed_mbps"], 25000)
        self.assertEqual(stats["config"]["link_tier"], "25GbE")
        self.assertEqual(stats["config"]["socket_buffer_mb"], 32)

    def test_stats_exposes_10gbe(self):
        from core.network_10gbe import SolarflareOptimizer, NetworkConfig
        opt = SolarflareOptimizer(NetworkConfig(link_speed_mbps=10000))
        stats = opt.get_stats()
        self.assertEqual(stats["config"]["link_tier"], "10GbE")
        self.assertEqual(stats["config"]["socket_buffer_mb"], 16)


class TestSwitchConfigScalesWithSpeed(unittest.TestCase):
    """Test that switch config enables jumbo frames at 25GbE."""

    def test_10gbe_rt_switch_has_standard_frames(self):
        from core.network_10gbe import generate_all_switch_configs
        configs = generate_all_switch_configs(link_speed_mbps=10000)
        self.assertNotIn("enable jumbo-frame ports 49-50", configs["switch_rt"])

    def test_25gbe_rt_switch_has_jumbo_frames(self):
        from core.network_10gbe import generate_all_switch_configs
        configs = generate_all_switch_configs(link_speed_mbps=25000)
        self.assertIn("enable jumbo-frame", configs["switch_rt"])

    def test_bulk_switch_always_has_jumbo(self):
        from core.network_10gbe import generate_all_switch_configs
        for speed in (10000, 25000):
            configs = generate_all_switch_configs(link_speed_mbps=speed)
            self.assertIn("enable jumbo-frame", configs["switch_bulk"])


# ──────────────────────────────────────────────────────────────────────────────
# infra/setup_r740.sh — BOSS + NVMe + SATA verification
# ──────────────────────────────────────────────────────────────────────────────

class TestSetupScriptHardwareCheck(unittest.TestCase):
    """Verify setup_r740.sh checks for BOSS, NVMe, SATA mounts."""

    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "infra" / "setup_r740.sh").read_text(encoding="utf-8")

    def test_has_verify_hardware_function(self):
        self.assertIn("verify_hardware()", self.script)
        self.assertIn("=== Hardware verification ===", self.script)

    def test_checks_boss_boot_device(self):
        self.assertIn("findmnt -n -o SOURCE /", self.script)
        # Should check model contains "boss" or "dellboss"
        self.assertRegex(self.script, r"boss\|dellboss", msg="Should match BOSS card model strings")

    def test_checks_nvme_mount_exists(self):
        self.assertIn('ARGUS_NVME_MOUNT', self.script)
        self.assertIn("does not exist", self.script)

    def test_checks_nvme_is_actually_mounted(self):
        """Just having the directory isn't enough — must be a real mountpoint."""
        self.assertIn("mountpoint -q", self.script)

    def test_warns_if_nvme_is_rotational(self):
        self.assertIn("rotational", self.script)

    def test_checks_sata_mount(self):
        self.assertIn("ARGUS_SATA_MOUNT", self.script)

    def test_has_skip_hardware_check_env_var(self):
        """Users should be able to bypass checks in edge cases."""
        self.assertIn("SKIP_HARDWARE_CHECK", self.script)

    def test_uses_nvme_mount_in_volume_dirs(self):
        """Volume dirs should go on NVMe, not /srv/argus."""
        self.assertIn("NVME_DIRS=", self.script)
        self.assertIn("$ARGUS_NVME_MOUNT/argus/timescaledb", self.script)
        self.assertIn("$ARGUS_NVME_MOUNT/argus/redis", self.script)
        self.assertIn("$ARGUS_NVME_MOUNT/argus/kafka", self.script)

    def test_uses_sata_mount_for_bulk(self):
        self.assertIn("SATA_DIRS=", self.script)
        self.assertIn("$ARGUS_SATA_MOUNT/argus/grafana", self.script)
        self.assertIn("$ARGUS_SATA_MOUNT/argus/loki", self.script)
        self.assertIn("$ARGUS_SATA_MOUNT/argus/prometheus", self.script)

    def test_nfs_logs_path_on_sata(self):
        """Logs are high volume, append-only — should go on SATA, not NVMe."""
        self.assertIn("NFS_LOGS_PATH=", self.script)
        self.assertIn("$ARGUS_SATA_MOUNT/argus/logs", self.script)

    def test_detects_nic_link_speed(self):
        """Setup should inform user if NIC is 10GbE vs 25GbE."""
        self.assertIn("ethtool", self.script)
        self.assertIn("Speed:", self.script)

    def test_creates_legacy_srv_argus_symlink(self):
        """Backward compat for any code still using /srv/argus."""
        self.assertIn("/srv/argus", self.script)
        self.assertIn("ln -sfn", self.script)


if __name__ == "__main__":
    unittest.main()
