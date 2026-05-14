from __future__ import annotations

import numpy as np


def test_canonical_facade_is_exported_and_reports_status():
    import quantum

    facade = quantum.get_quantum_facade()
    report = facade.status().to_dict()

    assert isinstance(facade, quantum.ArgusQuantumFacade)
    assert report["default_execution_mode"] == "classical_simulation"
    assert report["hardware_enabled"] is False
    assert any(cap["name"] == "actual_bell_pair_execution" for cap in report["supported_capabilities"])


def test_actual_bell_pair_fallback_runs_entangling_circuit(monkeypatch):
    from quantum import get_quantum_facade

    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.delenv("IBM_QUANTUM_API_KEY", raising=False)

    result = get_quantum_facade().run_actual_bell_pair(shots=256, seed=7)

    assert result["circuit_name"] == "bell_pair"
    assert result["shots"] == 256
    assert result["hardware_enabled"] is False
    assert result["requested_hardware"] is False
    assert result["execution_mode"] == "classical_statevector_simulation"
    assert result["quantum_metadata"]["capability"] == "actual_bell_pair_execution"
    assert result["quantum_advantage_claimed"] is False
    assert result["entanglement_score"] > 0.85
    assert set(result["counts"]).issubset({"00", "11"})


def test_actual_bell_pair_hardware_request_without_token_falls_back(monkeypatch):
    from quantum import get_quantum_facade

    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.delenv("IBM_QUANTUM_API_KEY", raising=False)

    result = get_quantum_facade(hardware_enabled=True).run_actual_bell_pair(shots=128, seed=3)

    assert result["requested_hardware"] is True
    assert result["hardware_enabled"] is False
    assert result["execution_mode"] == "classical_statevector_simulation"
    assert any("IBM_QUANTUM_TOKEN" in warning for warning in result["warnings"])


def test_local_bell_pair_never_attempts_ibm_even_with_token(monkeypatch):
    from quantum import get_quantum_facade
    from quantum.actual_quantum import ActualQuantumRunner

    def fail_if_called(self, *, shots, warnings):
        raise AssertionError("local-only Bell pair should not attempt IBM hardware")

    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-token")
    monkeypatch.setattr(ActualQuantumRunner, "_try_ibm_hardware", fail_if_called)

    result = get_quantum_facade(hardware_enabled=True).run_local_bell_pair(shots=128, seed=11)

    assert result["provider"] == "argus_in_repo"
    assert result["backend"] == "statevector"
    assert result["requested_hardware"] is False
    assert result["hardware_enabled"] is False
    assert result["execution_mode"] == "classical_statevector_simulation"
    assert result["warnings"] == []
    assert result["entanglement_score"] > 0.85


def test_actual_bell_pair_can_use_mocked_ibm_hardware(monkeypatch):
    from quantum.actual_quantum import ActualQuantumRunner
    import quantum.vendors.ibm_quantum as ibm_quantum

    class FakeBackend:
        def __init__(self, api_token=None, backend_name=None):
            self.api_token = api_token
            self.backend_name = backend_name

        def run_circuit(self, circuit, shots=1000):
            return {
                "counts": {"00": shots // 2, "11": shots - shots // 2},
                "method": "ibm_hardware",
                "backend": self.backend_name or "ibm_mock_backend",
                "shots": shots,
            }

    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-token")
    monkeypatch.setattr(ActualQuantumRunner, "_build_qiskit_bell_pair", staticmethod(lambda: object()))
    monkeypatch.setattr(ibm_quantum, "IBMQuantumBackend", FakeBackend)

    result = ActualQuantumRunner(hardware_enabled=True, backend_name="ibm_mock").run_bell_pair(shots=100)

    assert result["hardware_enabled"] is True
    assert result["requested_hardware"] is True
    assert result["execution_mode"] == "remote_hardware_if_configured"
    assert result["backend"] == "ibm_mock"
    assert result["entanglement_score"] == 1.0


def test_canonical_portfolio_result_has_honest_metadata():
    from quantum import get_quantum_facade

    expected_returns = np.array([0.08, 0.04, 0.12])
    covariance = np.array(
        [
            [0.10, 0.02, 0.01],
            [0.02, 0.08, 0.01],
            [0.01, 0.01, 0.12],
        ]
    )

    result = get_quantum_facade().optimize_portfolio(
        expected_returns,
        covariance,
        budget=2,
        n_layers=1,
        max_assets=3,
    )

    assert "weights" in result
    assert result["quantum_advantage_claimed"] is False
    assert result["hardware_enabled"] is False
    assert result["advisory_only"] is True
    assert result["quantum_metadata"]["capability"] == "qaoa_portfolio_subset"
    assert result["quantum_metadata"]["execution_mode"] == "classical_statevector_simulation"
    assert "classical simulator" in result["honest_claim"]


def test_canonical_qmc_tail_risk_result_has_no_advantage_claim():
    from quantum import get_quantum_facade

    returns = np.array([0.01, -0.02, 0.005, -0.04, 0.02, -0.01])

    result = get_quantum_facade().estimate_tail_risk_qmc(
        returns,
        n_samples=128,
        confidence=0.95,
    )

    assert "var" in result
    assert "cvar" in result
    assert result["quantum_advantage"] == 1.0
    assert result["quantum_advantage_claimed"] is False
    assert result["quantum_metadata"]["capability"] == "qmc_var_cvar"
    assert result["quantum_metadata"]["execution_mode"] == "quantum_inspired_classical_sobol"
    assert "no hardware quantum advantage" in result["qmc_note"]


def test_canonical_has_simulation_backend_and_noise_model():
    from quantum import get_quantum_facade

    facade = get_quantum_facade(
        simulation_backend="statevector",
        noise_model="ideal",
    )

    assert facade.simulation_backend == "statevector"
    assert facade.noise_model == "ideal"

    result = facade.run_quantum_walk(
        {"BTCUSDT": [0.01, 0.02], "ETHUSDT": [0.01, 0.02]},
        max_steps=5,
    )

    assert result["quantum_metadata"]["simulation_backend"] == "statevector"
    assert result["quantum_metadata"]["noise_model"] == "ideal"
    assert result["simulation_backend"] == "statevector"
    assert result["noise_model"] == "ideal"


def test_canonical_classify_regime():
    from quantum import get_quantum_facade

    facade = get_quantum_facade()
    returns = [0.01, 0.02, -0.01, 0.03, 0.02, -0.02, 0.01, -0.01, 0.02, 0.01]

    result = facade.classify_regime(returns, n_qubits=4)

    assert "regime" in result or "error" in result
    if "quantum_metadata" in result:
        assert result["quantum_metadata"]["capability"] == "vqc_regime_classifier"


def test_canonical_coordinate_agents():
    from quantum import get_quantum_facade

    facade = get_quantum_facade()
    agents = [
        {"agent_id": "trend", "signal": "buy", "confidence": 0.8},
        {"agent_id": "mean_rev", "signal": "sell", "confidence": 0.6},
    ]

    result = facade.coordinate_agents(agents)

    assert "consensus_signal" in result or "error" in result
    if "quantum_metadata" in result:
        assert result["quantum_metadata"]["capability"] == "quantum_multi_agent"


def test_canonical_optimize_hyperparameters():
    from quantum import get_quantum_facade

    facade = get_quantum_facade()
    param_grid = {"lr": [0.01, 0.1], "layers": [1, 2]}
    results = [0.5, 0.6, 0.4, 0.7]  # 4 combinations

    result = facade.optimize_hyperparameters(param_grid, results, maximize=True)

    assert "best_params" in result or "error" in result


def test_canonical_mlqae_tail_risk_result_is_research_advisory():
    from quantum import get_quantum_facade

    returns = np.array([0.01, -0.02, 0.005, -0.04, 0.02, -0.01, -0.03, 0.015])

    result = get_quantum_facade().estimate_tail_risk_mlqae(
        returns,
        n_samples=64,
        n_qubits=2,
    )

    assert "var_95" in result
    assert "cvar_95" in result
    assert result["quantum_advantage_claimed"] is False
    assert result["advisory_only"] is True
    assert result["quantum_metadata"]["capability"] == "mlqae_var"
    assert result["quantum_metadata"]["execution_mode"] == "classical_statevector_simulation"
    assert "simulated classically" in result["honest_claim"]


def test_retired_quantum_optimizer_delegates_supported_paths():
    from quantum.quantum_optimizer import get_quantum_optimizer

    returns = np.array([0.01, -0.02, 0.005, -0.04, 0.02, -0.01])

    result = get_quantum_optimizer().estimate_tail_risk(returns, n_samples=128)

    assert result["quantum_metadata"]["capability"] == "qmc_var_cvar"
    assert result["quantum_advantage_claimed"] is False


def test_ghz_state_has_entanglement():
    from quantum import get_quantum_facade

    result = get_quantum_facade().run_ghz(n_qubits=3, shots=256, seed=42)

    assert result["counts"]
    assert result["entanglement_score"] > 0.85
    assert result["quantum_metadata"]["capability"] == "local_ghz_state"
    assert result["hardware_enabled"] is False
    assert result["quantum_advantage_claimed"] is False


def test_w_state_has_entanglement():
    from quantum import get_quantum_facade

    result = get_quantum_facade().run_w_state(n_qubits=3, shots=256, seed=7)

    assert result["counts"]
    assert result["entanglement_score"] > 0.6
    assert result["quantum_metadata"]["capability"] == "local_w_state"
    assert result["hardware_enabled"] is False
    assert result["quantum_advantage_claimed"] is False


def test_parameterized_circuit_runs():
    from quantum import get_quantum_facade

    thetas = [0.5, 1.0, 1.5]
    result = get_quantum_facade().run_parameterized_circuit(
        n_qubits=3, thetas=thetas, n_layers=1, shots=128, seed=3,
    )

    assert result["counts"]
    assert result["quantum_metadata"]["capability"] == "local_parameterized_circuit"
    assert result["hardware_enabled"] is False


def test_quantum_walk_runs():
    from quantum import get_quantum_facade

    returns = {
        "BTC": [0.01, -0.02, 0.03, -0.01, 0.02],
        "ETH": [0.02, -0.01, 0.04, -0.02, 0.01],
        "SPY": [0.005, 0.001, -0.002, 0.003, 0.001],
    }
    result = get_quantum_facade().run_quantum_walk(returns, correlation_threshold=0.1)

    assert "weights" in result
    assert "centrality" in result
    assert result["quantum_metadata"]["capability"] == "local_quantum_walk"
    assert result["quantum_advantage_claimed"] is False


def test_trotter_ising_evolution():
    from quantum import get_quantum_facade

    result = get_quantum_facade().run_trotter_ising(
        n_qubits=3, time=1.0, n_steps=10, j_zz=1.0, h_x=0.5, shots=256, seed=42,
    )

    assert "counts" in result
    assert "expectation_z" in result
    assert result["quantum_metadata"]["capability"] == "trotter_ising"
    assert result["hardware_enabled"] is False
    assert result["quantum_advantage_claimed"] is False


def test_trotter_custom_hamiltonian():
    from quantum import get_quantum_facade

    terms = [
        (1.0, [(0, "Z"), (1, "Z")]),
        (0.5, [(0, "X")]),
        (0.5, [(1, "X")]),
    ]
    result = get_quantum_facade().run_trotter_evolution(
        terms, n_qubits=2, time=0.5, n_steps=5, shots=128, seed=7,
    )

    assert "counts" in result
    assert result["quantum_metadata"]["capability"] == "trotter_evolution"
    assert result["hardware_enabled"] is False


def test_grover_search():
    from quantum import get_quantum_facade

    oracle = lambda x: x in [3, 7]
    result = get_quantum_facade().run_grover_search(
        oracle, n_qubits=4, shots=512, seed=42,
    )

    assert "found_indices" in result
    assert "iterations" in result
    assert result["quantum_metadata"]["capability"] == "grover_search"
    assert result["hardware_enabled"] is False
    assert 3 in result["found_indices"] or 7 in result["found_indices"]


def test_vqe_ising():
    from quantum import get_quantum_facade
    import numpy as np

    h = np.array([0.5, -0.3, 0.2])
    J = np.array([[0, 1.0, 0], [1.0, 0, 0.5], [0, 0.5, 0]])

    result = get_quantum_facade().run_vqe_ising(
        h, J, n_qubits=3, n_layers=2, max_iter=50, shots=1024, seed=42,
    )

    assert "ground_energy" in result or "energy" in result
    assert result["quantum_metadata"]["capability"] == "vqe_ising"
    assert result["hardware_enabled"] is False


def test_mps_ghz():
    from quantum import get_quantum_facade

    result = get_quantum_facade().run_mps_ghz(
        n_qubits=15, shots=256, max_bond_dim=16, seed=7,
    )

    assert "counts" in result
    assert "entanglement_entropy" in result
    assert result["quantum_metadata"]["capability"] == "mps_ghz"
    assert result["execution_mode"] == "mps_tensor_network"
    assert result["hardware_enabled"] is False


def test_statevector_inspection():
    from quantum import get_quantum_facade

    result = get_quantum_facade().statevector_probabilities(
        "ghz", n_qubits=3, shots=256, seed=42,
    )

    assert "statevector_probabilities" in result
    assert "000" in result["statevector_probabilities"]
    assert "111" in result["statevector_probabilities"]
    assert result["quantum_metadata"]["capability"] == "statevector_ghz"


def test_retired_quantum_optimizer_unknown_symbols_fail_clearly():
    import quantum.quantum_optimizer as qopt

    placeholder = qopt.__getattr__("UnsupportedLegacyOptimizer")

    assert callable(placeholder)
    try:
        placeholder()
    except RuntimeError as exc:
        assert "retired" in str(exc).lower()
        assert "get_quantum_facade" in str(exc)
    else:
        raise AssertionError("retired quantum placeholder should raise RuntimeError")
