import unittest


class TestQuantumSimulatorBasic(unittest.TestCase):
    def test_bell_state_counts(self) -> None:
        from quantum_simulator import QuantumCircuit, simulate

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        qc.measure_all()
        res = simulate(qc, shots=2000, seed=123)
        counts = res["counts"]

        # In a Bell state, outcomes should concentrate on 00 and 11.
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
        self.assertEqual(set(k for k, _ in top), {"00", "11"})
        self.assertGreaterEqual(sum(counts.values()), 2000)


if __name__ == "__main__":
    unittest.main()

