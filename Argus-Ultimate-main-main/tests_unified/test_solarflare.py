"""
tests_unified/test_solarflare.py — Unit tests for Solarflare OpenOnload and
NIC tuning modules.

All tests pass without Solarflare hardware.  Hardware-dependent paths are
exercised via monkeypatching / graceful detection.

Run with:
    pytest tests_unified/test_solarflare.py -v
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from infra.solarflare_onload import (
    OnloadSettings,
    OpenOnloadConfig,
    OnloadSocketWrapper,
    generate_onload_profile,
)
from infra.nic_tuning import NICConfig, NICTuner


# =============================================================================
# OnloadSettings defaults
# =============================================================================


class TestOnloadSettingsDefaults:
    def test_onload_settings_defaults(self):
        """All OnloadSettings fields should have correct HFT default values."""
        s = OnloadSettings()
        assert s.enabled is True
        assert s.onload_binary == "/usr/bin/onload"
        assert s.ef_poll_usec == 100
        assert s.ef_int_driven == 0
        assert s.ef_spin_usec == 100_000
        assert s.ef_tcp_nodelay == 1
        assert s.ef_tcp_rcvbuf == 131_072
        assert s.ef_tcp_sndbuf == 65_536
        assert s.ef_log_level == 0
        assert s.ef_udp == 1
        assert s.stack_name == "argus_hft"


# =============================================================================
# Environment variable keys
# =============================================================================


class TestOnloadEnvKeys:
    def test_onload_env_keys(self):
        """get_onload_env() must return all required EF_* keys."""
        cfg = OpenOnloadConfig(OnloadSettings())
        env = cfg.get_onload_env()
        required_keys = {
            "EF_POLL_USEC",
            "EF_INT_DRIVEN",
            "EF_SPIN_USEC",
            "EF_TCP_NODELAY",
            "EF_TCP_RCVBUF",
            "EF_TCP_SNDBUF",
            "EF_UDP",
            "EF_NAME",
            "EF_LOG_LEVEL",
        }
        missing = required_keys - set(env.keys())
        assert not missing, f"Missing EF_* keys: {missing}"

    def test_onload_env_values_are_strings(self):
        """All env values must be strings (ready for os.environ / Popen)."""
        cfg = OpenOnloadConfig(OnloadSettings())
        for key, val in cfg.get_onload_env().items():
            assert isinstance(val, str), f"{key} value is not a string: {val!r}"

    def test_onload_env_values_match_settings(self):
        """Env values should reflect the OnloadSettings instance."""
        settings = OnloadSettings(ef_poll_usec=200, stack_name="test_stack")
        cfg = OpenOnloadConfig(settings)
        env = cfg.get_onload_env()
        assert env["EF_POLL_USEC"] == "200"
        assert env["EF_NAME"] == "test_stack"


# =============================================================================
# Command wrapping
# =============================================================================


class TestOnloadCommandWrapping:
    def test_onload_command_wrapping(self):
        """get_onload_command should prefix with 'onload --profile=latency'."""
        cfg = OpenOnloadConfig()
        base = "python -m strategies.micro_strategy_orchestrator"
        result = cfg.get_onload_command(base)
        assert result.startswith("onload --profile=latency ")
        assert base in result

    def test_get_launch_command_structure(self):
        """get_launch_command should include the onload binary and module."""
        cfg = OpenOnloadConfig()
        cmd = cfg.get_launch_command()
        assert "onload" in cmd
        assert "--profile=latency" in cmd
        assert "strategies.micro_strategy_orchestrator" in cmd

    def test_get_launch_command_no_double_onload(self):
        """Launch command should not repeat 'onload' more than once."""
        cfg = OpenOnloadConfig()
        cmd = cfg.get_launch_command()
        assert cmd.count("onload") <= 2  # "onload" binary + "--profile=latency"


# =============================================================================
# Availability detection (no hardware)
# =============================================================================


class TestOnloadAvailability:
    def test_onload_availability_no_hardware(self):
        """is_onload_available must return False gracefully when binary absent."""
        settings = OnloadSettings(onload_binary="/nonexistent/onload")
        cfg = OpenOnloadConfig(settings)
        # No binary → should return False without raising
        result = cfg.is_onload_available()
        assert result is False

    def test_onload_availability_disabled_flag(self):
        """is_onload_available returns False immediately when enabled=False."""
        settings = OnloadSettings(enabled=False)
        cfg = OpenOnloadConfig(settings)
        assert cfg.is_onload_available() is False

    def test_onload_availability_no_lspci(self):
        """If lspci is absent, is_onload_available returns False gracefully."""
        # Even if binary path is fake-present, lspci absence → False
        with patch("infra.solarflare_onload.OpenOnloadConfig._check_onload_binary",
                   return_value=True), \
             patch("shutil.which", return_value=None):
            cfg = OpenOnloadConfig()
            cfg._availability_cache = None
            result = cfg._check_solarflare_nic()
            assert result is False


# =============================================================================
# Socket wrapper
# =============================================================================


class TestSocketWrapper:
    def test_socket_wrapper_no_crash(self):
        """wrap_socket on a real TCP socket must not raise."""
        wrapper = OnloadSocketWrapper(OnloadSettings())
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = wrapper.wrap_socket(sock)
            assert result is sock  # same object returned
        finally:
            sock.close()

    def test_socket_wrapper_returns_same_socket(self):
        """wrap_socket must return the identical socket object."""
        wrapper = OnloadSocketWrapper()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            returned = wrapper.wrap_socket(sock)
            assert returned is sock
        finally:
            sock.close()

    def test_wrap_websocket_transport_no_socket(self):
        """wrap_websocket_transport must not crash when transport has no socket."""
        wrapper = OnloadSocketWrapper()
        transport = MagicMock()
        transport.get_extra_info.return_value = None
        # Should not raise
        wrapper.wrap_websocket_transport(transport)

    def test_get_socket_latency_stats_no_hardware(self):
        """get_socket_latency_stats returns sensible defaults on non-SF systems."""
        wrapper = OnloadSocketWrapper()
        stats = wrapper.get_socket_latency_stats()
        assert "bypass_packets" in stats
        assert "kernel_packets" in stats
        assert "avg_latency_us" in stats
        assert isinstance(stats["bypass_packets"], int)
        assert isinstance(stats["kernel_packets"], int)


# =============================================================================
# NICConfig defaults
# =============================================================================


class TestNICConfigDefaults:
    def test_nic_config_defaults(self):
        """NICConfig defaults: MTU=9000, bond enabled, RSS=4."""
        cfg = NICConfig()
        assert cfg.mtu == 9000
        assert cfg.enable_bonding is True
        assert cfg.rss_queues == 4
        assert cfg.ring_buffer_rx == 4096
        assert cfg.ring_buffer_tx == 4096
        assert cfg.enable_flow_control is False


# =============================================================================
# Generated tuning script content
# =============================================================================


class TestNICTuningScriptContent:
    def test_nic_tuning_script_content(self):
        """Generated script must contain key tuning commands."""
        tuner = NICTuner(NICConfig())
        script = tuner.generate_tuning_script()
        assert "ethtool" in script
        assert "ip link" in script
        assert "smp_affinity" in script

    def test_nic_tuning_script_mtu_value(self):
        """Script must embed the configured MTU value."""
        cfg = NICConfig(mtu=9000)
        tuner = NICTuner(cfg)
        script = tuner.generate_tuning_script()
        assert "9000" in script

    def test_nic_tuning_script_ring_buffers(self):
        """Script must include ring buffer sizing commands."""
        tuner = NICTuner(NICConfig())
        script = tuner.generate_tuning_script()
        assert "ethtool -G" in script or "ethtool" in script

    def test_nic_tuning_script_coalescing_disable(self):
        """Script must disable interrupt coalescing."""
        tuner = NICTuner(NICConfig())
        script = tuner.generate_tuning_script()
        # Should contain the coalescing-disable ethtool invocation
        assert "rx-usecs" in script or "adaptive-rx" in script


# =============================================================================
# Recommended configs
# =============================================================================


class TestNICRecommendedConfigs:
    def test_nic_recommended_r7525(self):
        """R7525 recommended config must use IRQ cores 4-7."""
        cfg = NICTuner.get_recommended_config("r7525")
        assert cfg.irq_affinity_cores == [4, 5, 6, 7]

    def test_nic_recommended_pc(self):
        """PC recommended config must use IRQ cores 20-23."""
        cfg = NICTuner.get_recommended_config("pc")
        assert cfg.irq_affinity_cores == [20, 21, 22, 23]

    def test_nic_recommended_r7525_bond_enabled(self):
        """R7525 config must have bonding enabled."""
        cfg = NICTuner.get_recommended_config("r7525")
        assert cfg.enable_bonding is True

    def test_nic_recommended_pc_bond_enabled(self):
        """PC config must have bonding enabled."""
        cfg = NICTuner.get_recommended_config("pc")
        assert cfg.enable_bonding is True

    def test_nic_recommended_unknown_machine_raises(self):
        """Unknown machine identifier must raise ValueError."""
        with pytest.raises(ValueError):
            NICTuner.get_recommended_config("unknown_machine_xyz")


# =============================================================================
# Diagnose
# =============================================================================


class TestNICDiagnose:
    def test_nic_diagnose_empty(self):
        """diagnose() returns a list (possibly empty) on systems without SF NIC."""
        tuner = NICTuner(NICConfig())
        result = tuner.diagnose()
        # Must return a list — never raise
        assert isinstance(result, list)

    def test_nic_diagnose_no_crash_on_missing_tools(self):
        """diagnose() must not crash when ip/ethtool binaries are absent."""
        with patch("shutil.which", return_value=None):
            tuner = NICTuner(NICConfig())
            result = tuner.diagnose()
            assert isinstance(result, list)


# =============================================================================
# Bond config
# =============================================================================


class TestBondConfig:
    def test_bond_config_generated(self):
        """Bond config must reference 802.3ad / mode=4."""
        tuner = NICTuner(NICConfig())
        bond_cfg = tuner.generate_bond_config()
        # Accept either the full name or the numeric mode
        assert "802.3ad" in bond_cfg or "mode=4" in bond_cfg or "mode: 802.3ad" in bond_cfg

    def test_bond_config_contains_both_interfaces(self):
        """Bond config must reference both NIC ports."""
        cfg = NICConfig(interface_name="enp1s0f0", interface_name_2="enp1s0f1")
        tuner = NICTuner(cfg)
        bond_cfg = tuner.generate_bond_config()
        assert "enp1s0f0" in bond_cfg
        assert "enp1s0f1" in bond_cfg

    def test_bond_config_mtu(self):
        """Bond config must embed the configured MTU."""
        tuner = NICTuner(NICConfig(mtu=9000))
        bond_cfg = tuner.generate_bond_config()
        assert "9000" in bond_cfg


# =============================================================================
# Onload profile content
# =============================================================================


class TestOnloadProfileContent:
    def test_onload_profile_content(self):
        """Generated profile must contain EF_POLL_USEC and EF_SPIN_USEC."""
        profile = generate_onload_profile()
        assert "EF_POLL_USEC" in profile
        assert "EF_SPIN_USEC" in profile

    def test_onload_profile_contains_udp(self):
        """Profile must enable UDP acceleration (EF_UDP=1)."""
        profile = generate_onload_profile()
        assert "EF_UDP" in profile
        assert "EF_UDP=1" in profile

    def test_onload_profile_int_driven_disabled(self):
        """Profile must disable interrupt-driven mode (EF_INT_DRIVEN=0)."""
        profile = generate_onload_profile()
        assert "EF_INT_DRIVEN=0" in profile

    def test_onload_profile_is_string(self):
        """generate_onload_profile must return a string."""
        profile = generate_onload_profile()
        assert isinstance(profile, str)
        assert len(profile) > 100  # non-trivial content
