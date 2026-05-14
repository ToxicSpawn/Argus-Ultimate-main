"""
Tests for R740 infrastructure configuration files.

Validates:
  - Docker Compose YAML structure (healthchecks, restart policies, networks)
  - Redis config parsing
  - PgBouncer config parsing
  - Backup script syntax
  - Failover monitor logic (heartbeat timeout, state transitions)
  - Ansible playbook YAML validity
  - WireGuard config parsing
  - Systemd service file parsing
  - Prometheus config structure
  - Grafana dashboard JSON validity
  - Nginx rate limit config
  - Maintenance script syntax

Run: py -m pytest tests/test_r740_infrastructure.py -v
"""

import configparser
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


def _bash_syntax_check(script_path: Path) -> subprocess.CompletedProcess:
    """Run bash -n syntax check, handling Windows path issues.

    On Windows, subprocess.run(['bash', '-n', path]) may fail to resolve
    paths. We use 'bash -c' with a redirected read to avoid path issues.
    """
    if platform.system() == "Windows":
        # Read the script content and pipe it to bash -n via stdin
        content = script_path.read_bytes()
        return subprocess.run(
            ["bash", "-n"],
            input=content,
            capture_output=True,
        )
    else:
        return subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True, text=True,
        )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFRA_DIR = PROJECT_ROOT / "infra"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.r740.yml"


# ===========================================================================
# Docker Compose
# ===========================================================================

class TestDockerCompose:
    """Validate docker-compose.r740.yml structure."""

    @pytest.fixture(autouse=True)
    def load_compose(self):
        assert COMPOSE_FILE.exists(), f"Missing {COMPOSE_FILE}"
        with open(COMPOSE_FILE) as f:
            self.compose = yaml.safe_load(f)
        self.services = self.compose.get("services", {})

    def test_yaml_is_valid(self):
        """Docker compose file parses as valid YAML."""
        assert isinstance(self.compose, dict)
        assert "services" in self.compose

    def test_all_expected_services_present(self):
        """All required services are defined."""
        expected = [
            "timescaledb", "redis", "kafka", "zookeeper", "prometheus",
            "grafana", "loki", "promtail", "caddy", "node-exporter",
            "pgbouncer", "ollama", "jupyter", "minio", "portainer",
            "uptime-kuma", "cadvisor", "watchtower", "mlflow", "registry",
        ]
        for svc in expected:
            assert svc in self.services, f"Service '{svc}' missing from docker-compose"

    def test_all_services_have_restart_policy(self):
        """Every long-running service has a restart policy."""
        # Exclude one-shot init containers
        skip = {"kafka-init", "minio-init"}
        for name, svc in self.services.items():
            if name in skip:
                continue
            restart = svc.get("restart")
            assert restart is not None, f"Service '{name}' missing restart policy"
            assert restart in ("unless-stopped", "always", "on-failure"), \
                f"Service '{name}' has unexpected restart policy: {restart}"

    def test_all_services_have_healthchecks(self):
        """Every long-running service has a healthcheck."""
        skip = {"kafka-init", "minio-init"}
        # Some services use profiles and may not have healthchecks
        profiles_only = {"pgadmin", "redis-insight"}
        for name, svc in self.services.items():
            if name in skip or name in profiles_only:
                continue
            hc = svc.get("healthcheck")
            assert hc is not None, f"Service '{name}' missing healthcheck"
            assert "test" in hc, f"Service '{name}' healthcheck missing 'test'"

    def test_services_on_argus_network(self):
        """Services are connected to argus-infra network."""
        skip = {"kafka-init", "minio-init"}
        for name, svc in self.services.items():
            if name in skip:
                continue
            networks = svc.get("networks")
            if networks is None:
                continue
            if isinstance(networks, list):
                assert "argus-infra" in networks, \
                    f"Service '{name}' not on argus-infra network"
            elif isinstance(networks, dict):
                assert "argus-infra" in networks, \
                    f"Service '{name}' not on argus-infra network"

    def test_memory_limits_present(self):
        """Key services have memory limits to prevent OOM."""
        memory_required = [
            "timescaledb", "redis", "kafka", "prometheus", "grafana",
            "loki", "ollama", "mlflow", "jupyter",
        ]
        for name in memory_required:
            svc = self.services.get(name, {})
            deploy = svc.get("deploy", {})
            resources = deploy.get("resources", {})
            limits = resources.get("limits", {})
            assert "memory" in limits, \
                f"Service '{name}' missing memory limit in deploy.resources.limits"

    def test_volumes_defined(self):
        """Volume definitions exist for persistent services."""
        volumes = self.compose.get("volumes", {})
        expected_volumes = [
            "timescaledb_data", "redis_data", "kafka_data",
            "prometheus_data", "grafana_data", "loki_data",
            "ollama_data", "minio_data", "mlflow_data",
        ]
        for vol in expected_volumes:
            assert vol in volumes, f"Volume '{vol}' not defined"

    def test_pgbouncer_depends_on_timescaledb(self):
        """PgBouncer depends on healthy TimescaleDB."""
        pgb = self.services.get("pgbouncer", {})
        depends = pgb.get("depends_on", {})
        assert "timescaledb" in depends, "PgBouncer should depend on timescaledb"

    def test_mlflow_depends_on_timescaledb_and_minio(self):
        """MLflow depends on both TimescaleDB and MinIO."""
        mlf = self.services.get("mlflow", {})
        depends = mlf.get("depends_on", {})
        assert "timescaledb" in depends, "MLflow should depend on timescaledb"
        assert "minio" in depends, "MLflow should depend on minio"


# ===========================================================================
# Redis Config
# ===========================================================================

class TestRedisConfig:
    """Validate Redis configuration file."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        self.config_path = INFRA_DIR / "redis" / "redis.conf"
        assert self.config_path.exists(), f"Missing {self.config_path}"
        self.content = self.config_path.read_text()
        self.lines = [
            line.strip() for line in self.content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def _get_value(self, key):
        """Get value for a redis config key."""
        for line in self.lines:
            parts = line.split(None, 1)
            if len(parts) >= 2 and parts[0] == key:
                return parts[1]
        return None

    def test_maxmemory_set(self):
        assert self._get_value("maxmemory") == "4gb"

    def test_maxmemory_policy(self):
        assert self._get_value("maxmemory-policy") == "allkeys-lru"

    def test_rdb_persistence(self):
        saves = [line for line in self.lines if line.startswith("save ")]
        assert len(saves) >= 3, "Expected at least 3 RDB save rules"

    def test_aof_enabled(self):
        assert self._get_value("appendonly") == "yes"

    def test_tcp_keepalive(self):
        assert self._get_value("tcp-keepalive") == "60"


# ===========================================================================
# PgBouncer Config
# ===========================================================================

class TestPgBouncerConfig:
    """Validate PgBouncer configuration file."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        self.config_path = INFRA_DIR / "pgbouncer" / "pgbouncer.ini"
        assert self.config_path.exists(), f"Missing {self.config_path}"
        self.config = configparser.ConfigParser()
        self.config.read(str(self.config_path))

    def test_has_databases_section(self):
        assert "databases" in self.config.sections()

    def test_has_pgbouncer_section(self):
        assert "pgbouncer" in self.config.sections()

    def test_pool_mode_transaction(self):
        assert self.config.get("pgbouncer", "pool_mode") == "transaction"

    def test_max_client_conn(self):
        assert int(self.config.get("pgbouncer", "max_client_conn")) == 200

    def test_default_pool_size(self):
        assert int(self.config.get("pgbouncer", "default_pool_size")) == 25

    def test_min_pool_size(self):
        assert int(self.config.get("pgbouncer", "min_pool_size")) == 5

    def test_reserve_pool_size(self):
        assert int(self.config.get("pgbouncer", "reserve_pool_size")) == 5


# ===========================================================================
# Backup Script
# ===========================================================================

class TestBackupScript:
    """Validate backup script."""

    @pytest.fixture(autouse=True)
    def load_script(self):
        self.script_path = INFRA_DIR / "backup" / "backup.sh"
        assert self.script_path.exists(), f"Missing {self.script_path}"
        self.content = self.script_path.read_text()

    def test_bash_syntax(self):
        """Verify script has valid bash syntax."""
        result = _bash_syntax_check(self.script_path)
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_contains_pg_dump(self):
        assert "pg_dump" in self.content

    def test_contains_redis_bgsave(self):
        assert "BGSAVE" in self.content

    def test_contains_rotation_logic(self):
        assert "DAILY_KEEP" in self.content
        assert "WEEKLY_KEEP" in self.content
        assert "MONTHLY_KEEP" in self.content

    def test_contains_b2_upload(self):
        assert "B2_KEY" in self.content
        assert "b2" in self.content


# ===========================================================================
# Failover Monitor
# ===========================================================================

class TestFailoverMonitor:
    """Test failover monitor state machine logic."""

    @pytest.fixture(autouse=True)
    def setup_monitor(self):
        # Import the module
        sys.path.insert(0, str(INFRA_DIR / "failover"))
        from failover_monitor import FailoverMonitor, FailoverState
        self.FailoverMonitor = FailoverMonitor
        self.FailoverState = FailoverState

    def _make_monitor(self, **kwargs):
        defaults = dict(
            redis_url="redis://localhost:6379/0",
            redis_password="test",
            failover_timeout=300,
            heartbeat_channel="argus:heartbeat",
            check_interval=1,
        )
        defaults.update(kwargs)
        m = self.FailoverMonitor(**defaults)
        m._redis = MagicMock()
        m._redis.ping.return_value = True
        return m

    def test_initial_state_is_standby(self):
        m = self._make_monitor()
        assert m.state == self.FailoverState.STANDBY

    def test_heartbeat_received_stays_standby(self):
        m = self._make_monitor(failover_timeout=300)
        m._redis.get.return_value = "alive"
        m.process_cycle()
        assert m.state == self.FailoverState.STANDBY

    def test_no_heartbeat_transitions_to_pending(self):
        m = self._make_monitor(failover_timeout=0)
        m._redis.get.return_value = None
        m.last_heartbeat = None  # never received
        m.process_cycle()
        assert m.state == self.FailoverState.FAILOVER_PENDING

    def test_heartbeat_recovery_cancels_pending(self):
        m = self._make_monitor(failover_timeout=300)
        m.state = self.FailoverState.FAILOVER_PENDING
        m._redis.get.return_value = "alive"
        m.last_heartbeat = time.time()  # just received
        m.process_cycle()
        assert m.state == self.FailoverState.STANDBY

    def test_timeout_activates_failover(self):
        m = self._make_monitor(failover_timeout=0)
        m.state = self.FailoverState.FAILOVER_PENDING
        m._redis.get.return_value = None
        m.last_heartbeat = None
        with patch.object(m, '_run_command', return_value=True):
            m.process_cycle()
        assert m.state == self.FailoverState.ACTIVE

    def test_heartbeat_during_active_triggers_handoff(self):
        m = self._make_monitor(failover_timeout=300)
        m.state = self.FailoverState.ACTIVE
        m._redis.get.return_value = "alive"
        m.last_heartbeat = time.time()
        with patch.object(m, '_run_command', return_value=True):
            m.process_cycle()
        assert m.state == self.FailoverState.STANDBY

    def test_seconds_since_heartbeat_infinity_when_none(self):
        m = self._make_monitor()
        m.last_heartbeat = None
        assert m._seconds_since_heartbeat() == float("inf")

    def test_seconds_since_heartbeat_recent(self):
        m = self._make_monitor()
        m.last_heartbeat = time.time() - 5
        elapsed = m._seconds_since_heartbeat()
        assert 4 <= elapsed <= 7


# ===========================================================================
# Ansible Playbook
# ===========================================================================

class TestAnsiblePlaybook:
    """Validate Ansible playbook YAML."""

    @pytest.fixture(autouse=True)
    def load_playbook(self):
        self.playbook_path = INFRA_DIR / "ansible" / "playbook.yml"
        assert self.playbook_path.exists(), f"Missing {self.playbook_path}"
        with open(self.playbook_path) as f:
            self.playbook = yaml.safe_load(f)

    def test_yaml_valid(self):
        assert isinstance(self.playbook, list)
        assert len(self.playbook) >= 1

    def test_has_tasks(self):
        play = self.playbook[0]
        assert "tasks" in play
        assert len(play["tasks"]) > 0

    def test_installs_docker(self):
        """Playbook installs Docker."""
        tasks = self.playbook[0]["tasks"]
        task_names = [t.get("name", "") for t in tasks]
        assert any("Docker" in n for n in task_names), \
            "No Docker installation task found"

    def test_configures_ufw(self):
        """Playbook configures UFW firewall."""
        tasks = self.playbook[0]["tasks"]
        task_names = [t.get("name", "") for t in tasks]
        assert any("UFW" in n or "ufw" in n or "SSH" in n for n in task_names)

    def test_creates_zfs_pool(self):
        """Playbook creates ZFS pool."""
        tasks = self.playbook[0]["tasks"]
        task_names = [t.get("name", "") for t in tasks]
        assert any("ZFS" in n or "zfs" in n.lower() for n in task_names)

    def test_installs_wireguard(self):
        """Playbook installs WireGuard."""
        tasks = self.playbook[0]["tasks"]
        task_names = [t.get("name", "") for t in tasks]
        assert any("WireGuard" in n or "wireguard" in n.lower() for n in task_names)

    def test_installs_fail2ban(self):
        """Playbook installs Fail2ban."""
        tasks = self.playbook[0]["tasks"]
        task_names = [t.get("name", "") for t in tasks]
        assert any("Fail2ban" in n or "fail2ban" in n.lower() for n in task_names)

    def test_creates_argus_user(self):
        """Playbook creates argus user."""
        tasks = self.playbook[0]["tasks"]
        task_names = [t.get("name", "") for t in tasks]
        assert any("argus user" in n.lower() for n in task_names)


# ===========================================================================
# WireGuard Config
# ===========================================================================

class TestWireGuardConfig:
    """Validate WireGuard configuration."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        self.config_path = INFRA_DIR / "wireguard" / "wg0.conf"
        assert self.config_path.exists(), f"Missing {self.config_path}"
        self.content = self.config_path.read_text()

    def test_has_interface_section(self):
        assert "[Interface]" in self.content

    def test_has_peer_section(self):
        assert "[Peer]" in self.content

    def test_server_address(self):
        assert "10.0.0.1/24" in self.content

    def test_listen_port(self):
        assert "ListenPort = 51820" in self.content

    def test_post_up_iptables(self):
        assert "PostUp" in self.content
        assert "iptables" in self.content

    def test_post_down_iptables(self):
        assert "PostDown" in self.content

    def test_client_allowed_ips(self):
        assert "10.0.0.2/32" in self.content


# ===========================================================================
# Systemd Service
# ===========================================================================

class TestSystemdService:
    """Validate systemd service file."""

    @pytest.fixture(autouse=True)
    def load_service(self):
        self.service_path = INFRA_DIR / "systemd" / "argus.service"
        assert self.service_path.exists(), f"Missing {self.service_path}"
        self.content = self.service_path.read_text()

    def test_has_unit_section(self):
        assert "[Unit]" in self.content

    def test_has_service_section(self):
        assert "[Service]" in self.content

    def test_has_install_section(self):
        assert "[Install]" in self.content

    def test_exec_start(self):
        assert "ExecStart" in self.content
        assert "docker compose" in self.content

    def test_exec_stop(self):
        assert "ExecStop" in self.content

    def test_restart_on_failure(self):
        assert "Restart=on-failure" in self.content

    def test_wanted_by_multi_user(self):
        assert "WantedBy=multi-user.target" in self.content

    def test_after_docker(self):
        assert "After=docker.service" in self.content


# ===========================================================================
# Prometheus Config
# ===========================================================================

class TestPrometheusConfig:
    """Validate Prometheus configuration."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        self.config_path = INFRA_DIR / "prometheus" / "prometheus.yml"
        assert self.config_path.exists()
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def test_yaml_valid(self):
        assert isinstance(self.config, dict)

    def test_has_scrape_configs(self):
        assert "scrape_configs" in self.config
        assert len(self.config["scrape_configs"]) > 0

    def test_scrapes_node_exporter(self):
        jobs = [sc["job_name"] for sc in self.config["scrape_configs"]]
        assert "r740-node" in jobs

    def test_scrapes_cadvisor(self):
        jobs = [sc["job_name"] for sc in self.config["scrape_configs"]]
        assert "cadvisor" in jobs

    def test_scrapes_redis_exporter(self):
        jobs = [sc["job_name"] for sc in self.config["scrape_configs"]]
        assert "redis" in jobs

    def test_scrapes_pgbouncer(self):
        jobs = [sc["job_name"] for sc in self.config["scrape_configs"]]
        assert "pgbouncer" in jobs


# ===========================================================================
# Grafana Dashboard
# ===========================================================================

class TestGrafanaDashboard:
    """Validate Grafana dashboard JSON."""

    @pytest.fixture(autouse=True)
    def load_dashboard(self):
        self.dashboard_path = INFRA_DIR / "grafana" / "dashboards" / "argus_trading.json"
        assert self.dashboard_path.exists()
        with open(self.dashboard_path) as f:
            self.dashboard = json.load(f)

    def test_json_valid(self):
        assert isinstance(self.dashboard, dict)

    def test_has_panels(self):
        assert "panels" in self.dashboard
        assert len(self.dashboard["panels"]) >= 6

    def test_has_portfolio_value_panel(self):
        titles = [p.get("title", "") for p in self.dashboard["panels"]]
        assert "Portfolio Value" in titles

    def test_has_pnl_panel(self):
        titles = [p.get("title", "") for p in self.dashboard["panels"]]
        assert any("P&L" in t for t in titles)

    def test_has_risk_metrics_panel(self):
        titles = [p.get("title", "") for p in self.dashboard["panels"]]
        assert "Risk Metrics" in titles

    def test_has_system_health_panel(self):
        titles = [p.get("title", "") for p in self.dashboard["panels"]]
        assert "System Health" in titles


# ===========================================================================
# Nginx Rate Limit
# ===========================================================================

class TestNginxRateLimit:
    """Validate nginx rate limiting config."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        self.config_path = INFRA_DIR / "nginx" / "rate_limit.conf"
        assert self.config_path.exists()
        self.content = self.config_path.read_text()

    def test_has_api_rate_limit(self):
        assert "api_limit" in self.content
        assert "10r/s" in self.content

    def test_has_dashboard_rate_limit(self):
        assert "dashboard_limit" in self.content
        assert "30r/s" in self.content

    def test_has_connection_limit(self):
        assert "limit_conn_zone" in self.content
        assert "conn_per_ip" in self.content


# ===========================================================================
# Maintenance Script
# ===========================================================================

class TestMaintenanceScript:
    """Validate maintenance window script."""

    @pytest.fixture(autouse=True)
    def load_script(self):
        self.script_path = INFRA_DIR / "maintenance" / "maintenance_window.sh"
        assert self.script_path.exists()
        self.content = self.script_path.read_text()

    def test_bash_syntax(self):
        result = _bash_syntax_check(self.script_path)
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_creates_maintenance_flag(self):
        assert "MAINTENANCE_MODE" in self.content

    def test_runs_vacuum(self):
        assert "VACUUM" in self.content

    def test_rotates_logs(self):
        assert "gzip" in self.content

    def test_compresses_chunks(self):
        assert "compress_chunk" in self.content

    def test_sends_discord_notification(self):
        assert "DISCORD_WEBHOOK" in self.content


# ===========================================================================
# Setup Script
# ===========================================================================

class TestSetupScript:
    """Validate updated setup_r740.sh."""

    @pytest.fixture(autouse=True)
    def load_script(self):
        self.script_path = INFRA_DIR / "setup_r740.sh"
        assert self.script_path.exists()
        self.content = self.script_path.read_text()

    def test_bash_syntax(self):
        result = _bash_syntax_check(self.script_path)
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_installs_zfs(self):
        assert "zfsutils-linux" in self.content

    def test_installs_wireguard(self):
        assert "wireguard" in self.content

    def test_installs_fail2ban(self):
        assert "fail2ban" in self.content

    def test_creates_ssl_cert(self):
        assert "openssl" in self.content
        assert "argus-r740" in self.content

    def test_installs_systemd_service(self):
        assert "argus.service" in self.content
        assert "systemctl" in self.content
