import unittest


class TestQuantumQmlBasic(unittest.TestCase):
    def test_quantum_qml_import_and_fit(self) -> None:
        from quantum.qml import VariationalQuantumClassifier

        # Simple separable dataset
        X = [
            [0.0, 0.0],
            [0.2, 0.1],
            [1.0, 1.0],
            [1.2, 0.9],
        ]
        y = [0, 0, 1, 1]

        m = VariationalQuantumClassifier(steps=200, lr=0.3, seed=123).fit(X, y)
        pred = m.predict(X).tolist()
        self.assertEqual(len(pred), 4)
        self.assertEqual(pred, [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()

