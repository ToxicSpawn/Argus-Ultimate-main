from __future__ import annotations


def test_quantum_status_report_is_import_safe_and_honest():
    import quantum

    report = quantum.quantum_status_report()
    data = report.to_dict()
    assert data["default_execution_mode"] == "classical_simulation"
    assert data["hardware_enabled"] is False
    assert data["supported_capabilities"]
    claims = " ".join(cap["honest_claim"] for cap in data["supported_capabilities"])
    assert "no speedup" in claims.lower() or "not a hardware quantum advantage" in claims.lower()
    assert any("No current module proves hardware quantum advantage" in w for w in data["warnings"])


def test_recommended_max_statevector_qubits_is_conservative():
    from quantum.status import recommended_max_statevector_qubits

    assert recommended_max_statevector_qubits(memory_gb=1.0) >= 20
    assert recommended_max_statevector_qubits(memory_gb=16.0) >= 28


def test_qmc_metadata_does_not_claim_quantum_advantage():
    import asyncio

    import numpy as np

    from quantum.algorithms.quantum_monte_carlo import QuantumMonteCarlo

    returns = np.array([0.01, -0.02, 0.005, -0.04, 0.02, -0.01])
    result = asyncio.run(QuantumMonteCarlo(n_qubits=4, n_samples=128).simulate(returns))
    assert result["quantum_advantage"] == 1.0
    assert result["quantum_advantage_claimed"] is False
    assert "no hardware quantum advantage" in result["qmc_note"]
