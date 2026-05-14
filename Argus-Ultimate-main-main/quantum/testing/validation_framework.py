"""
Quantum validation framework (restored, lightweight).

This is not a deep verification of quantum advantage; it's a pragmatic
import-/runtime-smoke validation so the repo's quantum components remain usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class QuantumValidationReport:
    ok: bool
    details: Dict[str, Any]


def run_validation(*, seed: int = 123) -> QuantumValidationReport:
    details: Dict[str, Any] = {}
    ok = True

    # QML basic check
    try:
        from quantum.qml import VariationalQuantumClassifier

        rng = np.random.default_rng(seed)
        X = rng.normal(size=(200, 4))
        y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(int)
        m = VariationalQuantumClassifier(steps=200, lr=0.2, seed=seed).fit(X, y)
        pred = m.predict(X)
        acc = float(np.mean(pred == y))
        details["qml_vqc_accuracy"] = acc
        if acc < 0.7:
            ok = False
            details["qml_vqc_warning"] = "unexpectedly low accuracy"
    except Exception as e:
        ok = False
        details["qml_error"] = repr(e)

    # Vendors base check
    try:
        from quantum.vendors import SimulatorVendor, QuantumJobRequest

        v = SimulatorVendor()
        res = v.submit(QuantumJobRequest(vendor=v.name, problem_type="noop", payload={"x": 1}, shots=100))
        details["vendors_simulator_status"] = res.status
    except Exception as e:
        ok = False
        details["vendors_error"] = repr(e)

    # Production quantum simulator status (optional)
    try:
        from quantum import get_quantum_simulator_status

        status = get_quantum_simulator_status()  # type: ignore[misc]
        details["production_simulator_status"] = status
    except Exception as e:
        # This is optional, do not fail validation.
        details["production_simulator_status"] = "unavailable"
        details["production_simulator_note"] = repr(e)

    return QuantumValidationReport(ok=ok, details=details)


def main() -> None:
    rep = run_validation()
    print({"ok": rep.ok, "details": rep.details})


if __name__ == "__main__":
    main()
