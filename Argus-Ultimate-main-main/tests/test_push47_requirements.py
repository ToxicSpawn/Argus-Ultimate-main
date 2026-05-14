"""tests/test_push47_requirements.py — Push 47.

12 tests verifying requirements.txt contains all deps introduced in Pushes 42-46.
"""

from __future__ import annotations

import pathlib
import unittest

REQ_PATH = pathlib.Path("requirements.txt")


class TestRequirementsFile(unittest.TestCase):

    def setUp(self):
        self.content = REQ_PATH.read_text()

    # Push 45: hmmlearn
    def test_hmmlearn_present(self):
        self.assertIn("hmmlearn", self.content)

    def test_hmmlearn_version_pinned(self):
        self.assertIn("hmmlearn>=0.3.2", self.content)

    # Push 42: ccxtpro
    def test_ccxtpro_present(self):
        self.assertIn("ccxtpro", self.content)

    def test_ccxtpro_version_pinned(self):
        self.assertIn("ccxtpro>=4.2.0", self.content)

    # Pre-existing core deps still present
    def test_ccxt_present(self):
        self.assertIn("ccxt>=4.2.0", self.content)

    def test_numpy_present(self):
        self.assertIn("numpy", self.content)

    def test_pandas_present(self):
        self.assertIn("pandas", self.content)

    def test_torch_present(self):
        self.assertIn("torch", self.content)

    def test_scikit_learn_present(self):
        self.assertIn("scikit-learn", self.content)

    def test_prometheus_client_present(self):
        self.assertIn("prometheus-client", self.content)

    def test_river_present(self):
        self.assertIn("river", self.content)

    def test_hmmlearn_comment_explains_use(self):
        self.assertIn("regime_classifier", self.content)


if __name__ == "__main__":
    unittest.main()
