import unittest


class TestQuantumSimulatorMps(unittest.TestCase):
    def test_auto_backend_uses_mps_for_large_n(self) -> None:
        from quantum_simulator import QuantumCircuit, simulate

        qc = QuantumCircuit(30)
        for i in range(30):
            qc.h(i)
        for i in range(0, 29, 2):
            qc.cnot(i, i + 1)
        qc.measure_all()

        res = simulate(qc, shots=200, seed=123, backend="auto")
        self.assertEqual(res["backend"], "mps")
        self.assertEqual(res["num_qubits"], 30)
        self.assertEqual(len(res["marginals_p1"]), 30)


if __name__ == "__main__":
    unittest.main()

