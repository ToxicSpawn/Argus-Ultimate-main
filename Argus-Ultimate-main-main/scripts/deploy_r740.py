"""
ARGUS R740 One-Command Deployment — deploy the full infrastructure stack to
the Dell PowerEdge R740 server from the workstation.

Steps:
  1. SSH to R740 (configurable host/user)
  2. Copy all infra/ files + docker-compose
  3. Run docker-compose up -d
  4. Wait for health checks (TimescaleDB, Redis, Kafka, Grafana, MinIO)
  5. Create Kafka topics
  6. Initialize TimescaleDB schema
  7. Configure Grafana datasources (via provisioning)
  8. Print status summary

Usage:
  py scripts/deploy_r740.py
  py scripts/deploy_r740.py --host 192.168.1.100 --user argus
  py scripts/deploy_r740.py --host r740.local --dry-run
  py scripts/deploy_r740.py --skip-copy          # just restart + health check

Environment variables (override CLI args):
  R740_HOST       default: 192.168.1.100
  R740_USER       default: argus
  R740_SSH_KEY    default: ~/.ssh/id_ed25519
  R740_DEPLOY_DIR default: /srv/argus
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.deploy")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_HOST = os.environ.get("R740_HOST", "192.168.1.100")
DEFAULT_USER = os.environ.get("R740_USER", "argus")
DEFAULT_SSH_KEY = os.environ.get("R740_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
DEFAULT_DEPLOY_DIR = os.environ.get("R740_DEPLOY_DIR", "/srv/argus")

# Project root (3 levels up from scripts/deploy_r740.py or found via git)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Files/dirs to copy to the R740
DEPLOY_ARTIFACTS = [
    "docker-compose.r740.yml",
    "infra/",
    ".env.example",
    "requirements.txt",
]

# Services expected to be healthy
EXPECTED_SERVICES = [
    "timescaledb",
    "redis",
    "kafka",
    "zookeeper",
    "prometheus",
    "grafana",
    "loki",
    "minio",
    "caddy",
]

# Kafka topics to create
KAFKA_TOPICS = [
    ("argus.trades", 3, 1),       # (name, partitions, replication)
    ("argus.signals", 3, 1),
    ("argus.prices", 6, 1),       # more partitions for high-volume prices
    ("argus.alerts", 1, 1),
    ("argus.regime", 1, 1),
    ("argus.dlq", 1, 1),          # dead letter queue
]

# Health check ports
HEALTH_PORTS = {
    "timescaledb": 5432,
    "redis": 6379,
    "kafka": 9093,
    "grafana": 3000,
    "prometheus": 9090,
    "minio": 9000,
    "caddy": 80,
    "loki": 3100,
}


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

class SSHRunner:
    """Execute commands on the R740 via SSH."""

    def __init__(self, host: str, user: str, key_path: str, dry_run: bool = False):
        self.host = host
        self.user = user
        self.key_path = key_path
        self.dry_run = dry_run
        self._ssh_opts = [
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
        ]
        if key_path and Path(key_path).exists():
            self._ssh_opts.extend(["-i", key_path])

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def run(self, cmd: str, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command on the remote host via SSH."""
        full_cmd = ["ssh"] + self._ssh_opts + [self.target, cmd]
        logger.debug("SSH: %s", cmd)

        if self.dry_run:
            logger.info("[DRY RUN] ssh %s '%s'", self.target, cmd)
            return subprocess.CompletedProcess(full_cmd, 0, stdout="", stderr="")

        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if check and result.returncode != 0:
                logger.error("SSH command failed (rc=%d): %s\nstderr: %s", result.returncode, cmd, result.stderr[:300])
            return result
        except subprocess.TimeoutExpired:
            logger.error("SSH command timed out (%ds): %s", timeout, cmd)
            raise
        except FileNotFoundError:
            logger.error("ssh not found — ensure OpenSSH is installed")
            raise

    def rsync(self, local_path: str, remote_path: str) -> bool:
        """Rsync a file or directory to the remote host."""
        ssh_cmd = f"ssh {' '.join(self._ssh_opts)}"
        if self.key_path and Path(self.key_path).exists():
            # Already included in _ssh_opts
            pass

        cmd = [
            "rsync", "-avz", "--delete",
            "-e", ssh_cmd,
            local_path,
            f"{self.target}:{remote_path}",
        ]

        if self.dry_run:
            logger.info("[DRY RUN] rsync %s → %s:%s", local_path, self.target, remote_path)
            return True

        logger.info("Syncing %s → %s:%s", local_path, self.target, remote_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error("rsync failed: %s", result.stderr[:300])
                return False
            return True
        except FileNotFoundError:
            # rsync not available — fall back to scp
            return self._scp_fallback(local_path, remote_path)
        except Exception as exc:
            logger.error("rsync error: %s", exc)
            return False

    def _scp_fallback(self, local_path: str, remote_path: str) -> bool:
        """Fallback to scp when rsync is not available."""
        cmd = ["scp", "-r"] + self._ssh_opts + [local_path, f"{self.target}:{remote_path}"]

        if self.dry_run:
            logger.info("[DRY RUN] scp %s → %s:%s", local_path, self.target, remote_path)
            return True

        logger.info("Falling back to scp: %s → %s:%s", local_path, self.target, remote_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception as exc:
            logger.error("scp failed: %s", exc)
            return False

    def test_connection(self) -> bool:
        """Test SSH connection to host."""
        logger.info("Testing SSH connection to %s...", self.target)
        try:
            result = self.run("echo ok", timeout=10, check=False)
            if result.returncode == 0 and "ok" in (result.stdout or ""):
                logger.info("SSH connection OK")
                return True
            logger.error("SSH connection failed: %s", result.stderr[:200])
            return False
        except Exception as exc:
            logger.error("SSH connection error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Deployment steps
# ---------------------------------------------------------------------------

def step_copy_files(ssh: SSHRunner, deploy_dir: str) -> bool:
    """Copy infrastructure files to R740."""
    logger.info("=== Step 1: Copy files to R740 ===")

    # Ensure deploy directory exists
    ssh.run(f"mkdir -p {deploy_dir}", check=False)

    success = True
    for artifact in DEPLOY_ARTIFACTS:
        local_path = PROJECT_ROOT / artifact
        if not local_path.exists():
            logger.warning("Artifact not found: %s — skipping", local_path)
            continue

        remote_path = f"{deploy_dir}/{artifact}"
        if local_path.is_dir():
            remote_path = f"{deploy_dir}/"
            ok = ssh.rsync(str(local_path) + "/", f"{deploy_dir}/{artifact}/")
        else:
            ok = ssh.rsync(str(local_path), remote_path)

        if not ok:
            success = False
            logger.error("Failed to copy: %s", artifact)

    return success


def step_docker_compose_up(ssh: SSHRunner, deploy_dir: str) -> bool:
    """Run docker-compose up -d on the R740."""
    logger.info("=== Step 2: Start docker-compose ===")

    # Pull latest images first
    ssh.run(
        f"cd {deploy_dir} && docker compose -f docker-compose.r740.yml pull",
        timeout=600,
        check=False,
    )

    # Start all services
    result = ssh.run(
        f"cd {deploy_dir} && docker compose -f docker-compose.r740.yml up -d",
        timeout=300,
    )

    if result.returncode != 0:
        logger.error("docker-compose up failed")
        return False

    logger.info("docker-compose up -d complete")
    return True


def step_wait_health(ssh: SSHRunner, max_wait: int = 120) -> Dict[str, bool]:
    """Wait for all services to become healthy."""
    logger.info("=== Step 3: Waiting for health checks (max %ds) ===", max_wait)

    health: Dict[str, bool] = {svc: False for svc in EXPECTED_SERVICES}
    start = time.time()

    while time.time() - start < max_wait:
        all_healthy = True

        for service in EXPECTED_SERVICES:
            if health[service]:
                continue

            port = HEALTH_PORTS.get(service)
            if not port:
                continue

            # Check if port is listening
            result = ssh.run(
                f"timeout 2 bash -c '</dev/tcp/localhost/{port}' 2>/dev/null && echo UP || echo DOWN",
                timeout=10,
                check=False,
            )

            if "UP" in (result.stdout or ""):
                health[service] = True
                logger.info("  %s (:%d) — UP", service, port)
            else:
                all_healthy = False

        if all_healthy:
            logger.info("All services healthy!")
            return health

        remaining = [s for s, ok in health.items() if not ok]
        elapsed = int(time.time() - start)
        logger.info("  Waiting... (%ds) — pending: %s", elapsed, ", ".join(remaining))
        time.sleep(5)

    # Log final state
    for svc, ok in health.items():
        if not ok:
            logger.warning("  %s — TIMEOUT (not healthy after %ds)", svc, max_wait)

    return health


def step_create_kafka_topics(ssh: SSHRunner) -> bool:
    """Create Kafka topics via kafka-topics.sh inside the container."""
    logger.info("=== Step 4: Create Kafka topics ===")

    success = True
    for topic_name, partitions, replication in KAFKA_TOPICS:
        cmd = (
            f"docker exec argus-kafka kafka-topics.sh "
            f"--create --if-not-exists "
            f"--bootstrap-server localhost:9092 "
            f"--topic {topic_name} "
            f"--partitions {partitions} "
            f"--replication-factor {replication}"
        )
        result = ssh.run(cmd, timeout=30, check=False)
        if result.returncode == 0:
            logger.info("  Topic: %s (%d partitions) — OK", topic_name, partitions)
        else:
            # Topic may already exist
            if "already exists" in (result.stdout or "") + (result.stderr or ""):
                logger.info("  Topic: %s — already exists", topic_name)
            else:
                logger.error("  Topic: %s — FAILED: %s", topic_name, (result.stderr or "")[:200])
                success = False

    return success


def step_init_timescaledb(ssh: SSHRunner, deploy_dir: str) -> bool:
    """Initialize TimescaleDB schema."""
    logger.info("=== Step 5: Initialize TimescaleDB schema ===")

    schema_file = f"{deploy_dir}/infra/timescaledb_schema.sql"

    # Check if schema file was copied
    result = ssh.run(f"test -f {schema_file} && echo EXISTS || echo MISSING", check=False)
    if "MISSING" in (result.stdout or ""):
        logger.warning("Schema file not found at %s — skipping", schema_file)
        return False

    # Run schema via psql in the TimescaleDB container
    result = ssh.run(
        f"docker exec -i argus-timescaledb psql -U argus -d argus < {schema_file}",
        timeout=60,
        check=False,
    )

    if result.returncode == 0:
        logger.info("TimescaleDB schema initialized successfully")
        return True
    else:
        # Errors like "already exists" are normal on re-deploy
        stderr = (result.stderr or "")
        if "already exists" in stderr or "IF NOT EXISTS" in stderr:
            logger.info("TimescaleDB schema already initialized (idempotent)")
            return True
        logger.error("TimescaleDB schema init failed: %s", stderr[:300])
        return False


def step_configure_grafana(ssh: SSHRunner, deploy_dir: str) -> bool:
    """
    Verify Grafana datasources are configured.
    Grafana auto-loads from provisioning YAML files (copied with infra/).
    """
    logger.info("=== Step 6: Verify Grafana datasources ===")

    # Check Grafana API
    result = ssh.run(
        "curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/api/health",
        timeout=10,
        check=False,
    )

    http_code = (result.stdout or "").strip()
    if http_code == "200":
        logger.info("Grafana API responding (HTTP 200)")

        # List datasources
        result = ssh.run(
            "curl -s -u admin:admin http://localhost:3000/api/datasources",
            timeout=10,
            check=False,
        )
        try:
            import json
            datasources = json.loads(result.stdout or "[]")
            for ds in datasources:
                logger.info("  Datasource: %s (%s)", ds.get("name"), ds.get("type"))
        except Exception:
            logger.info("  Datasources configured via provisioning")

        return True
    else:
        logger.warning("Grafana not responding (HTTP %s) — datasources will load on next restart", http_code)
        return False


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------

def print_summary(
    health: Dict[str, bool],
    kafka_ok: bool,
    tsdb_ok: bool,
    grafana_ok: bool,
    host: str,
) -> None:
    """Print final deployment status."""
    print("\n" + "=" * 70)
    print("  ARGUS R740 DEPLOYMENT SUMMARY")
    print("=" * 70)
    print(f"  Host:           {host}")
    print(f"  Deploy time:    {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print()

    # Service health
    print("  Service Health:")
    for svc, ok in sorted(health.items()):
        status = "UP" if ok else "DOWN"
        port = HEALTH_PORTS.get(svc, "?")
        icon = "[OK]" if ok else "[!!]"
        print(f"    {icon} {svc:<20} :{port:<6} {status}")

    print()
    print(f"  Kafka topics:     {'OK' if kafka_ok else 'FAILED'}")
    print(f"  TimescaleDB:      {'OK' if tsdb_ok else 'FAILED'}")
    print(f"  Grafana:          {'OK' if grafana_ok else 'NOT READY'}")

    # Access URLs
    print()
    print("  Access URLs:")
    print(f"    Grafana:        http://{host}:3000     (admin/admin)")
    print(f"    Prometheus:     http://{host}:9090")
    print(f"    MinIO Console:  http://{host}:9001     (argus/argus-secret)")
    print(f"    Caddy Proxy:    http://{host}")
    print(f"    TimescaleDB:    postgresql://argus@{host}:5432/argus")
    print(f"    Redis:          redis://{host}:6379")
    print(f"    Kafka:          {host}:9093")

    healthy_count = sum(1 for ok in health.values() if ok)
    total_count = len(health)
    overall = "SUCCESS" if healthy_count == total_count else "PARTIAL"
    print()
    print(f"  Overall:  {overall} ({healthy_count}/{total_count} services healthy)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def deploy(
    host: str = DEFAULT_HOST,
    user: str = DEFAULT_USER,
    ssh_key: str = DEFAULT_SSH_KEY,
    deploy_dir: str = DEFAULT_DEPLOY_DIR,
    dry_run: bool = False,
    skip_copy: bool = False,
) -> bool:
    """
    Execute full R740 deployment. Returns True if all steps succeed.
    """
    ssh = SSHRunner(host, user, ssh_key, dry_run)

    # Test connection
    if not dry_run:
        if not ssh.test_connection():
            logger.error("Cannot connect to %s — check SSH config", ssh.target)
            return False

    # Step 1: Copy files
    if not skip_copy:
        if not step_copy_files(ssh, deploy_dir):
            logger.error("File copy failed — aborting")
            return False
    else:
        logger.info("Skipping file copy (--skip-copy)")

    # Step 2: Docker compose up
    if not step_docker_compose_up(ssh, deploy_dir):
        logger.error("docker-compose up failed — check logs on R740")
        return False

    # Step 3: Wait for health
    health = step_wait_health(ssh)

    # Step 4: Create Kafka topics (only if Kafka is healthy)
    kafka_ok = False
    if health.get("kafka", False):
        kafka_ok = step_create_kafka_topics(ssh)
    else:
        logger.warning("Kafka not healthy — skipping topic creation")

    # Step 5: Init TimescaleDB (only if healthy)
    tsdb_ok = False
    if health.get("timescaledb", False):
        tsdb_ok = step_init_timescaledb(ssh, deploy_dir)
    else:
        logger.warning("TimescaleDB not healthy — skipping schema init")

    # Step 6: Grafana
    grafana_ok = False
    if health.get("grafana", False):
        grafana_ok = step_configure_grafana(ssh, deploy_dir)

    # Summary
    print_summary(health, kafka_ok, tsdb_ok, grafana_ok, host)

    all_healthy = all(health.values()) and kafka_ok and tsdb_ok
    return all_healthy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARGUS R740 One-Command Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py scripts/deploy_r740.py
  py scripts/deploy_r740.py --host 192.168.1.100 --user argus
  py scripts/deploy_r740.py --dry-run
  py scripts/deploy_r740.py --skip-copy
        """,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"R740 hostname/IP (default: {DEFAULT_HOST})")
    parser.add_argument("--user", default=DEFAULT_USER, help=f"SSH user (default: {DEFAULT_USER})")
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY, help=f"SSH key path (default: {DEFAULT_SSH_KEY})")
    parser.add_argument("--deploy-dir", default=DEFAULT_DEPLOY_DIR, help=f"Remote deploy dir (default: {DEFAULT_DEPLOY_DIR})")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--skip-copy", action="store_true", help="Skip file copy (just restart + health check)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    success = deploy(
        host=args.host,
        user=args.user,
        ssh_key=args.ssh_key,
        deploy_dir=args.deploy_dir,
        dry_run=args.dry_run,
        skip_copy=args.skip_copy,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
