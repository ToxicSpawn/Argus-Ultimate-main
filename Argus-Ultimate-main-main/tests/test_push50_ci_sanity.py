"""tests/test_push50_ci_sanity.py — Push 50.

10 tests verifying the CI pipeline additions and final v6.6.0 sanity.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml


class TestCIWorkflow(unittest.TestCase):
    """ci.yml contains all Push 50 additions."""

    def setUp(self):
        self.content = pathlib.Path(".github/workflows/ci.yml").read_text()

    def test_hmmlearn_in_ci_install(self):
        self.assertIn("hmmlearn", self.content)

    def test_ccxtpro_in_ci_install(self):
        self.assertIn("ccxtpro", self.content)

    def test_regime_smoke_job_present(self):
        self.assertIn("regime-smoke", self.content)

    def test_regime_classifier_imported_in_smoke(self):
        self.assertIn("RegimeClassifier", self.content)

    def test_docker_build_job_present(self):
        self.assertIn("docker-build", self.content)

    def test_backtest_smoke_job_present(self):
        self.assertIn("backtest-smoke", self.content)


class TestRegressionGates(unittest.TestCase):

    def setUp(self):
        self.content = pathlib.Path(
            ".github/workflows/argus-regression-gates.yml"
        ).read_text()

    def test_hmmlearn_in_regression_install(self):
        self.assertIn("hmmlearn", self.content)

    def test_push45_in_regression_filter(self):
        self.assertIn("push45", self.content)


class TestFinalVersionSanity(unittest.TestCase):

    def test_version_is_660(self):
        import sys
        sys.modules.pop("version", None)
        import version
        self.assertEqual(version.__version__, "6.6.0")

    def test_changelog_has_push50_entry(self):
        content = pathlib.Path("CHANGELOG.md").read_text()
        self.assertIn("Push 50", content)


if __name__ == "__main__":
    unittest.main()
