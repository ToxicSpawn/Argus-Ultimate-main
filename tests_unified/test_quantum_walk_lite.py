import unittest


class TestQuantumWalkLite(unittest.TestCase):
    def test_quantum_walk_import_safe(self) -> None:
        # Should not require SciPy/NetworkX/Qiskit to import.
        from quantum_walk import QuantumWalkLite  # noqa: F401

    def test_quantum_walk_lite_basic(self) -> None:
        from quantum_walk import QuantumWalkLite

        qwl = QuantumWalkLite(correlation_threshold=0.0, steps=10, damping=0.2)
        res = qwl.analyze(
            price_history={
                "A": [100, 101, 102, 103, 104, 105],
                "B": [50, 49, 51, 50, 52, 53],
                "C": [10, 10, 10.1, 10.2, 10.15, 10.3],
            },
            start_symbol="A",
        )

        self.assertEqual(set(res.symbols), {"A", "B", "C"})
        self.assertEqual(res.correlation_matrix.shape, (3, 3))
        s = sum(res.visitation_probabilities.values())
        self.assertAlmostEqual(float(s), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()

