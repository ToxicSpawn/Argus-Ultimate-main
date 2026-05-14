import unittest


class TestQuantumPackagesImportSafe(unittest.TestCase):
    def test_imports_do_not_crash(self) -> None:
        # These should be import-safe even if optional deps are missing.
        import quantum  # noqa: F401
        import quantum_walk  # noqa: F401
        import quantum_simulator  # noqa: F401


if __name__ == "__main__":
    unittest.main()

