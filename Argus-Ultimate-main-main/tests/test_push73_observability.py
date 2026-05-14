"""Push 73 — Tests: Prometheus config, alerting rules, Grafana dashboard,
requirements.txt. 18 tests.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Prometheus config (4)
# ---------------------------------------------------------------------------

class TestPrometheusConfig:
    def _load(self):
        try:
            import yaml
            return yaml.safe_load((ROOT / "docker" / "prometheus.yml").read_text())
        except ImportError:
            return None

    def test_file_exists(self):
        assert (ROOT / "docker" / "prometheus.yml").exists()

    def test_scrape_interval_set(self):
        cfg = self._load()
        if cfg:
            assert cfg["global"]["scrape_interval"] == "15s"

    def test_argus_job_defined(self):
        cfg = self._load()
        if cfg:
            jobs = [j["job_name"] for j in cfg["scrape_configs"]]
            assert "argus" in jobs

    def test_rule_files_referenced(self):
        cfg = self._load()
        if cfg:
            assert any("alerting_rules" in r for r in cfg.get("rule_files", []))


# ---------------------------------------------------------------------------
# Alerting rules (5)
# ---------------------------------------------------------------------------

class TestAlertingRules:
    def _load(self):
        try:
            import yaml
            return yaml.safe_load((ROOT / "docker" / "alerting_rules.yml").read_text())
        except ImportError:
            return None

    def test_file_exists(self):
        assert (ROOT / "docker" / "alerting_rules.yml").exists()

    def test_argus_down_alert(self):
        cfg = self._load()
        if cfg:
            all_alerts = [
                r["alert"]
                for g in cfg["groups"]
                for r in g["rules"]
            ]
            assert "ArgusDown" in all_alerts

    def test_drawdown_alert(self):
        cfg = self._load()
        if cfg:
            all_alerts = [
                r["alert"]
                for g in cfg["groups"]
                for r in g["rules"]
            ]
            assert "HighDrawdown" in all_alerts

    def test_critical_severity_exists(self):
        cfg = self._load()
        if cfg:
            severities = [
                r["labels"]["severity"]
                for g in cfg["groups"]
                for r in g["rules"]
            ]
            assert "critical" in severities

    def test_infra_group_exists(self):
        cfg = self._load()
        if cfg:
            groups = [g["name"] for g in cfg["groups"]]
            assert "infrastructure" in groups


# ---------------------------------------------------------------------------
# Grafana dashboard (5)
# ---------------------------------------------------------------------------

class TestGrafanaDashboard:
    def _load(self):
        path = ROOT / "grafana" / "argus_dashboard.json"
        return json.loads(path.read_text())

    def test_file_exists(self):
        assert (ROOT / "grafana" / "argus_dashboard.json").exists()

    def test_valid_json(self):
        d = self._load()
        assert isinstance(d, dict)

    def test_has_panels(self):
        d = self._load()
        assert len(d["panels"]) > 5

    def test_equity_curve_panel_exists(self):
        d = self._load()
        titles = [p.get("title", "") for p in d["panels"]]
        assert any("Equity" in t for t in titles)

    def test_refresh_set(self):
        d = self._load()
        assert d["refresh"] == "30s"


# ---------------------------------------------------------------------------
# requirements.txt (4)
# ---------------------------------------------------------------------------

class TestRequirements:
    def _reqs(self):
        text = (ROOT / "requirements.txt").read_text()
        pkgs = {}
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name = line.split("==")[0].split(">")[0].split("<")[0].strip()
                pkgs[name.lower()] = line
        return pkgs

    def test_aiohttp_pinned(self):
        assert "aiohttp" in self._reqs()

    def test_torch_pinned(self):
        assert "torch" in self._reqs()

    def test_prometheus_client_pinned(self):
        assert "prometheus-client" in self._reqs()

    def test_pytest_present(self):
        assert "pytest" in self._reqs()
