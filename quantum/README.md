# Quantum Computing Module -- Honest Status

## What is actually implemented

### Real implementations (classical simulation, no quantum hardware required)

These modules contain working code that runs on a CPU by simulating quantum
dynamics with numpy statevectors.  No quantum advantage is claimed -- the
value is the mathematical framework, not speed.

The approved application API is `quantum.get_quantum_facade()`. Runtime trading
code should use that facade instead of importing experimental modules directly.

| Module | What it does | Status |
|--------|-------------|--------|
| `actual_quantum.py` | Bell-state circuit execution via IBM Quantum when explicitly enabled; simulator fallback otherwise | Hardware-ready |
| `algorithms/qaoa.py` | QAOA for portfolio optimisation (Markowitz) | Working, classical sim |
| `algorithms/quantum_amplitude_estimation.py` | QAE for VaR/CVaR tail probabilities | Working, classical sim |
| `algorithms/quantum_monte_carlo.py` | Quasi-Monte Carlo with Sobol sequences | Working, genuinely useful |
| `qml/quantum_kernel.py` | Quantum kernel SVM classifier | Working, classical sim |
| `qml/quantum_reservoir.py` | Quantum reservoir computing for time series | Working, classical sim |
| `qml/models.py` | VQC, QSVM, QNN, QBM, QKernel wrappers | Working, classical sim |
| `vendors/dwave_solver.py` | D-Wave quantum annealing client | Requires D-Wave API key |
| `vendors/ibm_quantum.py` | IBM Quantum backend client | Requires IBM Quantum token |
| `vendors/dwave_provider.py` | D-Wave provider abstraction | Requires D-Wave API key |
| `vendors/ibm_provider.py` | IBM provider abstraction | Requires IBM Quantum token |

### Retired or research-only files

Hype-era startup modules and duplicate optimizers are no longer loaded by the
main application. If kept for compatibility, they should either delegate to the
canonical facade or fail with a clear retired-feature message.

| Module | Dependencies | Notes |
|--------|-------------|-------|
| `quantum_optimizer.py` | none | Retired adapter; delegates supported paths to `get_quantum_facade()` |
| `advanced_quantum_ml.py` | qiskit, pennylane | Research only; not loaded at startup |
| `production_quantum_infrastructure.py` | qiskit, pennylane | Enterprise integration layer |
| `production_quantum_simulator.py` | qiskit | Portfolio/strategy/risk via quantum sim |
| `hybrid_quantum_classical.py` | qiskit, pennylane | Hybrid workflow engine |
| `variational_quantum_financial.py` | qiskit | VQE for financial problems |
| `quantum_error_correction.py` | qiskit | Error correction codes |
| `quantum_cloud_integration.py` | various | Multi-cloud quantum backends |
| `quantum_nisq_optimizer.py` | qiskit | NISQ-aware circuit optimisation |

### Classical simulation vs real hardware

Everything in this module runs as **classical simulation** unless you
explicitly configure API keys for real quantum hardware.  Classical
simulation is exact but limited to ~12-14 qubits before memory becomes
an issue.  For the problem sizes ARGUS handles (5-20 assets), classical
simulation is perfectly adequate and often faster than network round-trips
to real quantum hardware.

To run only on this machine and never contact IBM, use
`quantum.get_quantum_facade().run_local_bell_pair()`. This is a real quantum
circuit model executed by ARGUS' local statevector simulator, not physical
quantum hardware.

The concrete real-hardware path is
`quantum.get_quantum_facade(hardware_enabled=True).run_actual_bell_pair()`.
It attempts IBM Quantum only when `IBM_QUANTUM_TOKEN` is configured and falls
back to the in-repo simulator with explicit metadata when hardware is not
available.

The quantum reservoir computer (`qml/quantum_reservoir.py`) simulates
a 2^n dimensional Hilbert space as a nonlinear feature expansion for
time series prediction.  This is legitimate applied mathematics -- the
quantum formalism provides a rich feature space -- but there is no
quantum speedup involved.

### Placeholder files (56 files)

The following files are **empty stubs** that exist only so that existing
`import` statements elsewhere in the codebase do not crash.  They contain
a `_Placeholder` class that raises `RuntimeError` if you try to call
anything on them.

See the docstring at the top of any of these files for details.  A
non-exhaustive list:

- `quantum_ultimate.py`, `quantum_unified.py`, `quantum_optimizer.py`
- `free_quantum_supremacy.py`, `free_optimization.py`, `free_extreme_performance.py`
- `free_peak_performance.py`, `free_performance_boost.py`, `free_realistic_quantum.py`
- `enhanced_quantum_system.py`, `advanced_algorithms.py`, `benchmarking.py`
- `game_theory_advanced.py`, `market_making.py`, `meta_learning.py`
- `hybrid/__init__.py`, `hybrid/gradients.py`, `hybrid/variational.py`
- `optimization/grover.py`, `optimization/qaoa.py` (the real QAOA is `algorithms/qaoa.py`)
- `backtesting/__init__.py`, `backtesting/monte_carlo.py`
- ... and ~35 more

These will never be implemented.  They are the remnants of an earlier
auto-generated scaffold.

## Setting up API keys for real quantum hardware

```bash
# D-Wave (quantum annealing -- best for combinatorial optimisation)
export DWAVE_API_KEY="your-dwave-api-key"

# IBM Quantum (gate-based -- QAOA, VQE, kernel methods)
export IBM_QUANTUM_TOKEN="your-ibm-quantum-token"
```

Other vendors (IonQ, Google, Rigetti) have provider stubs in `vendors/`
but are untested.

## Honest benchmarks

**Where quantum-inspired methods help:**
- Quasi-Monte Carlo (`algorithms/quantum_monte_carlo.py`) genuinely
  converges faster than standard MC for VaR/CVaR estimation.
- Quantum kernel methods can capture nonlinear feature interactions
  that classical RBF kernels miss, but training is O(n^2) in samples.
- Reservoir computing provides a principled nonlinear feature expansion
  that can beat simple baselines on noisy time series.

**Where quantum does NOT help (today):**
- Portfolio optimisation with <20 assets is solved faster by classical
  quadratic programming than by any quantum or quantum-simulated method.
- QAOA on a classical simulator is strictly slower than scipy.optimize.
- Claims of "120x speedup" or "+200-500% alpha" are not supported by
  evidence.  On current hardware and problem sizes, classical methods
  dominate.

## Running tests

```bash
# Quantum kernel classifier
py -m pytest tests/test_quantum_kernel.py -v

# Quantum reservoir computing
py -m pytest tests/test_quantum_reservoir.py -v

# Core algorithm tests (if they exist)
py -m pytest tests/quantum/ -v
```
