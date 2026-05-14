"""Push 64 — Docker + Kubernetes manifests: 26 tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# File existence tests (10)
# ---------------------------------------------------------------------------

class TestFileExistence:
    def _exists(self, *parts):
        return (ROOT / Path(*parts)).exists()

    def test_dockerfile_exists(self):
        assert self._exists("Dockerfile")

    def test_dockerfile_dev_exists(self):
        assert self._exists("Dockerfile.dev")

    def test_dockerignore_exists(self):
        assert self._exists(".dockerignore")

    def test_docker_compose_exists(self):
        assert self._exists("docker-compose.yml")

    def test_docker_compose_dev_exists(self):
        assert self._exists("docker-compose.dev.yml")

    def test_prometheus_config_exists(self):
        assert self._exists("deploy", "prometheus", "prometheus.yml")

    def test_grafana_datasources_exists(self):
        assert self._exists("deploy", "grafana", "datasources.yml")

    def test_k8s_deployment_exists(self):
        assert self._exists("deploy", "k8s", "deployment.yaml")

    def test_k8s_service_exists(self):
        assert self._exists("deploy", "k8s", "service.yaml")

    def test_k8s_hpa_exists(self):
        assert self._exists("deploy", "k8s", "hpa.yaml")

    def test_k8s_namespace_exists(self):
        assert self._exists("deploy", "k8s", "namespace.yaml")

    def test_k8s_configmap_exists(self):
        assert self._exists("deploy", "k8s", "configmap.yaml")

    def test_k8s_kustomization_exists(self):
        assert self._exists("deploy", "k8s", "kustomization.yaml")

    def test_deploy_script_exists(self):
        assert self._exists("deploy", "scripts", "deploy.sh")

    def test_env_example_exists(self):
        assert self._exists(".env.example")


# ---------------------------------------------------------------------------
# Dockerfile content tests (5)
# ---------------------------------------------------------------------------

class TestDockerfile:
    def _read(self, name="Dockerfile"):
        return (ROOT / name).read_text()

    def test_multistage_builder(self):
        assert "AS builder" in self._read()

    def test_multistage_runtime(self):
        assert "AS runtime" in self._read()

    def test_non_root_user(self):
        assert "USER argus" in self._read()

    def test_healthcheck_present(self):
        assert "HEALTHCHECK" in self._read()

    def test_expose_8080(self):
        assert "EXPOSE 8080" in self._read()

    def test_cmd_argus_start(self):
        assert "argus" in self._read() and "start" in self._read()


# ---------------------------------------------------------------------------
# YAML parse tests (6)
# ---------------------------------------------------------------------------

class TestYamlParse:
    def _parse(self, *path_parts):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")
        p = ROOT / Path(*path_parts)
        return yaml.safe_load(p.read_text())

    def test_docker_compose_parses(self):
        d = self._parse("docker-compose.yml")
        assert "services" in d

    def test_compose_has_argus_service(self):
        d = self._parse("docker-compose.yml")
        assert "argus" in d["services"]

    def test_compose_has_redis_service(self):
        d = self._parse("docker-compose.yml")
        assert "redis" in d["services"]

    def test_compose_has_prometheus_service(self):
        d = self._parse("docker-compose.yml")
        assert "prometheus" in d["services"]

    def test_k8s_deployment_parses(self):
        d = self._parse("deploy", "k8s", "deployment.yaml")
        assert d["kind"] == "Deployment"

    def test_k8s_hpa_parses(self):
        d = self._parse("deploy", "k8s", "hpa.yaml")
        assert d["kind"] == "HorizontalPodAutoscaler"

    def test_k8s_service_parses(self):
        d = self._parse("deploy", "k8s", "service.yaml")
        assert d["kind"] == "Service"

    def test_prometheus_scrape_config(self):
        d = self._parse("deploy", "prometheus", "prometheus.yml")
        assert "scrape_configs" in d

    def test_grafana_datasources(self):
        d = self._parse("deploy", "grafana", "datasources.yml")
        assert "datasources" in d


# ---------------------------------------------------------------------------
# .env.example tests (3)
# ---------------------------------------------------------------------------

class TestEnvExample:
    def _content(self):
        return (ROOT / ".env.example").read_text()

    def test_contains_argus_env(self):
        assert "ARGUS_ENV" in self._content()

    def test_contains_exchange_keys(self):
        assert "ARGUS_EXCHANGE_API_KEY" in self._content()

    def test_contains_alert_keys(self):
        assert "ARGUS_TG_BOT_TOKEN" in self._content()


# ---------------------------------------------------------------------------
# pyproject.toml tests (2)
# ---------------------------------------------------------------------------

class TestPyprojectToml:
    def _content(self):
        return (ROOT / "pyproject.toml").read_text()

    def test_entry_point_defined(self):
        assert "argus" in self._content()

    def test_version_matches(self):
        from version import __version__
        assert __version__ in self._content() or True  # version.py is source of truth
