"""tests/test_push49_grafana_provisioning.py — Push 49.

14 tests verifying Grafana provisioning, docker-compose fix,
Loki config, and regime dashboard JSON are correct.
"""

from __future__ import annotations

import json
import pathlib
import unittest

import yaml


class TestDockerCompose(unittest.TestCase):

    def setUp(self):
        self.compose = yaml.safe_load(pathlib.Path("docker-compose.yml").read_text())

    def test_grafana_provisioning_mount_correct(self):
        """Must mount ./grafana/provisioning not ./grafana."""
        grafana = self.compose["services"]["grafana"]
        volumes = grafana["volumes"]
        mounts = [v for v in volumes if "provisioning" in v]
        self.assertTrue(any("grafana/provisioning" in m for m in mounts),
                        f"Expected grafana/provisioning mount, got: {volumes}")

    def test_loki_service_present(self):
        self.assertIn("loki", self.compose["services"])

    def test_loki_port_3100(self):
        loki = self.compose["services"]["loki"]
        ports = loki.get("ports", [])
        self.assertTrue(any("3100" in str(p) for p in ports))

    def test_argus_metrics_port_8001(self):
        argus = self.compose["services"]["argus"]
        ports = argus.get("ports", [])
        self.assertTrue(any("8001" in str(p) for p in ports))

    def test_loki_volume_declared(self):
        volumes = self.compose.get("volumes", {})
        self.assertIn("loki_data", volumes)


class TestDatasourcesYml(unittest.TestCase):

    def setUp(self):
        path = pathlib.Path("grafana/provisioning/datasources/datasources.yml")
        self.ds = yaml.safe_load(path.read_text())

    def test_prometheus_datasource_present(self):
        names = [d["name"] for d in self.ds["datasources"]]
        self.assertIn("Prometheus", names)

    def test_loki_datasource_present(self):
        names = [d["name"] for d in self.ds["datasources"]]
        self.assertIn("Loki", names)


class TestLokiConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = yaml.safe_load(pathlib.Path("infra/loki-config.yml").read_text())

    def test_auth_disabled(self):
        self.assertFalse(self.cfg["auth_enabled"])

    def test_http_port_3100(self):
        self.assertEqual(self.cfg["server"]["http_listen_port"], 3100)

    def test_analytics_reporting_disabled(self):
        self.assertFalse(self.cfg["analytics"]["reporting_enabled"])


class TestRegimeDashboard(unittest.TestCase):

    def setUp(self):
        path = pathlib.Path("grafana/provisioning/dashboards/argus-regime.json")
        self.dash = json.loads(path.read_text())

    def test_dashboard_uid(self):
        self.assertEqual(self.dash["uid"], "argus-regime")

    def test_four_panels(self):
        self.assertEqual(len(self.dash["panels"]), 4)

    def test_regime_label_panel_present(self):
        titles = [p["title"] for p in self.dash["panels"]]
        self.assertIn("HMM Regime", titles)

    def test_regime_probs_panel_present(self):
        titles = [p["title"] for p in self.dash["panels"]]
        self.assertIn("Regime State Probabilities", titles)


if __name__ == "__main__":
    unittest.main()
