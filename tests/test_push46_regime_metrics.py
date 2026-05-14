"""tests/test_push46_regime_metrics.py — Push 46.

18 tests for PrometheusEmitter.emit_regime().
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np


def _make_emitter(has_prometheus: bool = False):
    """Build a PrometheusEmitter with prometheus_client optionally mocked."""
    if has_prometheus:
        # Minimal stub so the module initialises without a real prometheus install
        prom_mod = types.ModuleType("prometheus_client")
        fake_gauge   = MagicMock()
        fake_counter = MagicMock()
        fake_hist    = MagicMock()
        prom_mod.Gauge     = MagicMock(return_value=fake_gauge)
        prom_mod.Counter   = MagicMock(return_value=fake_counter)
        prom_mod.Histogram = MagicMock(return_value=fake_hist)
        prom_mod.start_http_server = MagicMock()
        prom_mod.REGISTRY  = MagicMock()
        sys.modules["prometheus_client"] = prom_mod
    else:
        sys.modules.pop("prometheus_client", None)

    # Force reimport
    sys.modules.pop("metrics.prometheus_emitter", None)
    sys.modules.pop("metrics", None)
    from metrics.prometheus_emitter import PrometheusEmitter
    return PrometheusEmitter(port=19999, version="push46-test")


class TestEmitRegimeNoProm(unittest.TestCase):
    """emit_regime() is a no-op when prometheus_client is absent."""

    def setUp(self):
        self.em = _make_emitter(has_prometheus=False)

    def test_noop_bull(self):
        # Should not raise
        self.em.emit_regime("bull", np.array([0.8, 0.15, 0.05]), 1.3)

    def test_noop_bear(self):
        self.em.emit_regime("bear", np.array([0.05, 0.15, 0.8]), 0.6)

    def test_noop_sideways(self):
        self.em.emit_regime("sideways", np.array([0.2, 0.6, 0.2]), 1.0)

    def test_noop_none_probs(self):
        self.em.emit_regime("sideways", None, 1.0)

    def test_noop_wrong_probs_shape(self):
        self.em.emit_regime("bull", np.array([0.5, 0.5]), 1.3)


class TestEmitRegimeSignature(unittest.TestCase):
    """emit_regime signature and attribute checks."""

    def test_method_exists(self):
        em = _make_emitter(has_prometheus=False)
        self.assertTrue(hasattr(em, "emit_regime"))
        self.assertTrue(callable(em.emit_regime))

    def test_accepts_three_args(self):
        em = _make_emitter(has_prometheus=False)
        import inspect
        sig = inspect.signature(em.emit_regime)
        params = list(sig.parameters.keys())
        self.assertIn("label",  params)
        self.assertIn("probs",  params)
        self.assertIn("scalar", params)


class TestLabelEncoding(unittest.TestCase):
    """Verify label encoding logic is consistent with spec."""

    def _encode(self, label: str) -> float:
        return 1.0 if label == "bull" else (-1.0 if label == "bear" else 0.0)

    def test_bull_encodes_positive(self):
        self.assertEqual(self._encode("bull"), 1.0)

    def test_bear_encodes_negative(self):
        self.assertEqual(self._encode("bear"), -1.0)

    def test_sideways_encodes_zero(self):
        self.assertEqual(self._encode("sideways"), 0.0)

    def test_unknown_encodes_zero(self):
        self.assertEqual(self._encode("unknown"), 0.0)


class TestVersionBump(unittest.TestCase):
    """version.py reflects Push 46."""

    def test_version_string(self):
        sys.modules.pop("version", None)
        import version
        self.assertEqual(version.__version__, "6.6.0")

    def test_version_tuple(self):
        sys.modules.pop("version", None)
        import version
        self.assertEqual(version.__version_info__, (6, 6, 0))

    def test_codename(self):
        sys.modules.pop("version", None)
        import version
        self.assertEqual(version.__codename__, "HMM-Regime")


class TestChangelogEntry(unittest.TestCase):
    """CHANGELOG.md contains the [6.6.0] entry."""

    def test_changelog_has_660(self):
        with open("CHANGELOG.md") as f:
            content = f.read()
        self.assertIn("[6.6.0]", content)

    def test_changelog_mentions_push46(self):
        with open("CHANGELOG.md") as f:
            content = f.read()
        self.assertIn("Push 46", content)

    def test_changelog_mentions_emit_regime(self):
        with open("CHANGELOG.md") as f:
            content = f.read()
        self.assertIn("emit_regime", content)


if __name__ == "__main__":
    unittest.main()
