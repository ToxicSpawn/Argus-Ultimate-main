#!/usr/bin/env python3
"""
Quantum algorithm benchmark suite.

Runs each quantum algorithm against its classical baseline on fixed seed
problems and emits a JSON report. Used for:

- CI regression guard (call from CI to detect performance regressions)
- Honest assessment in docs (records the WALL-CLOCK comparison so we never
  accidentally claim simulation-based quantum speedups)

Usage:
    py scripts/quantum_benchmark.py
    py scripts/quantum_benchmark.py --output reports/quantum_benchmark.json
    py scripts/quantum_benchmark.py --quick    # smaller problem sizes
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add repo root to sys.path so we can import quantum/, strategies/, etc.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Benchmarks
# ═════════════════════════════════════════════════════════════════════════════


def benchmark_qaoa(n_assets: int, seed: int = 42) -> Dict[str, Any]:
    """Benchmark QAOA portfolio optimization vs classical Markowitz."""
    from quantum.algorithms.qaoa import QAOAPortfolioOptimizer

    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.15, n_assets)
    A = rng.standard_normal((n_assets, n_assets)) * 0.1
    sigma = A.T @ A + np.eye(n_assets) * 0.01

    opt = QAOAPortfolioOptimizer(n_layers=3, max_assets=n_assets)
    bench = opt.benchmark_vs_classical(mu, sigma, risk_aversion=0.5)

    return {
        "algorithm": "QAOA",
        "problem": "portfolio_optimization",
        "n_assets": n_assets,
        "quantum_method": bench.get("qaoa_method"),
        "classical_baseline": bench.get("classical_method"),
        "quantum_sharpe": bench.get("qaoa_sharpe"),
        "classical_sharpe": bench.get("classical_sharpe"),
        "quantum_time_ms": bench.get("qaoa_time_ms"),
        "classical_time_ms": bench.get("classical_time_ms"),
        "improvement_pct": bench.get("improvement_pct"),
        "honest_notes": bench.get("honest_assessment"),
    }


def benchmark_vqe(n_qubits: int, seed: int = 42) -> Dict[str, Any]:
    """Benchmark VQE on a random Ising Hamiltonian vs exact diagonalization."""
    from quantum.algorithms.vqe import VQESolver, exact_ising_ground_energy

    rng = np.random.default_rng(seed)
    h = rng.uniform(-0.5, 0.5, n_qubits)
    J = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J[i, j] = rng.uniform(-0.5, 0.5)

    t0 = time.perf_counter()
    exact_e, exact_bits = exact_ising_ground_energy(h, J)
    classical_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    solver = VQESolver(n_qubits=n_qubits, n_layers=4)
    result = solver.solve_ising(h, J, max_iter=80, shots=2048, n_restarts=3)
    quantum_time = (time.perf_counter() - t0) * 1000

    error_pct = (
        abs(result["ground_energy"] - exact_e) / max(abs(exact_e), 1e-9) * 100
    )

    return {
        "algorithm": "VQE",
        "problem": "ising_ground_state",
        "n_qubits": n_qubits,
        "quantum_method": result["method"],
        "classical_baseline": "exact_diagonalization",
        "quantum_energy": result["ground_energy"],
        "classical_energy": exact_e,
        "energy_error_pct": error_pct,
        "quantum_time_ms": quantum_time,
        "classical_time_ms": classical_time,
        "honest_notes": (
            "VQE on classical simulation cannot beat exact diagonalization for "
            f"n={n_qubits} qubits ({1 << n_qubits} states). Value is "
            "hardware-portability and architectural correctness."
        ),
    }


def benchmark_grover(n_qubits: int, n_marked: int = 1, seed: int = 42) -> Dict[str, Any]:
    """Benchmark Grover search vs classical linear scan."""
    from quantum.algorithms.grover import GroverSearch

    N = 1 << n_qubits
    # Pick n_marked random marked indices
    rng = np.random.default_rng(seed)
    marked = set(rng.choice(N, size=n_marked, replace=False).tolist())

    def oracle(x: int) -> bool:
        return x in marked

    grover = GroverSearch(n_qubits=n_qubits)

    t0 = time.perf_counter()
    quantum_result = grover.search(oracle, n_items=N, n_solutions=n_marked)
    quantum_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    classical_found = [x for x in range(N) if oracle(x)]
    classical_time = (time.perf_counter() - t0) * 1000

    return {
        "algorithm": "Grover",
        "problem": "unstructured_search",
        "n_qubits": n_qubits,
        "search_space": N,
        "n_marked": n_marked,
        "quantum_method": quantum_result["method"],
        "classical_baseline": "linear_scan",
        "quantum_iterations": quantum_result["iterations"],
        "quantum_oracle_calls": quantum_result["n_oracle_calls"],
        "classical_oracle_calls": N,
        "oracle_call_speedup": quantum_result["speedup_vs_classical"],
        "quantum_time_ms": quantum_time,
        "classical_time_ms": classical_time,
        "success_probability": quantum_result["success_probability"],
        "honest_notes": (
            f"Grover gives O(sqrt(N/M)) QUERY complexity ({quantum_result['n_oracle_calls']} "
            f"vs {N} classical). Wall-clock simulation is O(N) per iteration, so it does NOT "
            "beat linear scan on a CPU. Hardware-portable structure."
        ),
    }


def benchmark_mlqae(n_returns: int = 500, seed: int = 42) -> Dict[str, Any]:
    """Benchmark MLQAE VaR estimation vs classical empirical percentile."""
    from quantum.algorithms.quantum_amplitude_estimation import (
        QuantumAmplitudeEstimatorVaR,
    )

    rng = np.random.default_rng(seed)
    returns = np.concatenate(
        [
            rng.normal(0.0005, 0.02, int(n_returns * 0.95)),
            rng.normal(-0.08, 0.01, int(n_returns * 0.05)),
        ]
    )
    rng.shuffle(returns)

    classical_var = float(np.percentile(returns, 5.0))

    est = QuantumAmplitudeEstimatorVaR(n_qubits=4)

    t0 = time.perf_counter()
    result = est.estimate_var(returns, confidence=0.95, n_samples=5000)
    quantum_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = float(np.percentile(returns, 5.0))
    classical_time = (time.perf_counter() - t0) * 1000

    error_pct = (
        abs(result["var_95"] - classical_var) / max(abs(classical_var), 1e-9) * 100
    )

    return {
        "algorithm": "MLQAE",
        "problem": "value_at_risk",
        "n_returns": n_returns,
        "quantum_method": result.get("method"),
        "classical_baseline": "numpy_percentile",
        "quantum_var_95": result.get("var_95"),
        "classical_var_95": classical_var,
        "error_pct": error_pct,
        "quantum_time_ms": quantum_time,
        "classical_time_ms": classical_time,
        "variance_reduction_factor": result.get("variance_reduction_factor"),
        "honest_notes": (
            "MLQAE on classical simulation provably cannot beat classical "
            "Monte Carlo wall-clock — each oracle query is O(2^n). Value is "
            "framing/correctness/hardware-readiness, not speedup."
        ),
    }


def benchmark_grover_arb_search() -> Dict[str, Any]:
    """Benchmark the Grover-driven arbitrage search strategy."""
    from strategies.quantum_arb_search import QuantumArbSearcher, VenuePrice

    prices = {
        "kraken": {
            "BTC/USD": VenuePrice("kraken", "BTC/USD", bid=50000, ask=50001, fee_bps=5),
            "ETH/USD": VenuePrice("kraken", "ETH/USD", bid=3000, ask=3001, fee_bps=5),
            "XRP/USD": VenuePrice("kraken", "XRP/USD", bid=0.50, ask=0.5005, fee_bps=5),
        },
        "coinbase": {
            "BTC/USD": VenuePrice("coinbase", "BTC/USD", bid=50500, ask=50501, fee_bps=5),
            "ETH/USD": VenuePrice("coinbase", "ETH/USD", bid=3000.5, ask=3001.5, fee_bps=5),
            "XRP/USD": VenuePrice("coinbase", "XRP/USD", bid=0.5050, ask=0.5052, fee_bps=5),
        },
    }

    searcher = QuantumArbSearcher(threshold_multiplier=1.2, min_edge_bps=3.0)

    t0 = time.perf_counter()
    signals = searcher.find_opportunities(prices)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "algorithm": "Grover_arb_search",
        "problem": "arbitrage_detection",
        "n_venues": len(prices),
        "n_symbols": 3,
        "n_signals": len(signals),
        "elapsed_ms": elapsed_ms,
        "snapshot": searcher.snapshot(),
        "top_signals": [
            {
                "symbol": s.symbol,
                "buy@": s.venue_buy,
                "sell@": s.venue_sell,
                "edge_bps": s.expected_edge_bps,
            }
            for s in signals[:3]
        ],
        "honest_notes": (
            "Grover for arb detection: hardware-portable O(sqrt(N)) query "
            "complexity vs O(N) classical scan. Wall-clock on simulator is "
            "comparable since each iteration costs O(N)."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum benchmark suite")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run smaller problem sizes",
    )
    args = parser.parse_args()

    print("Running quantum benchmark suite...", file=sys.stderr)

    benchmarks: List[Dict[str, Any]] = []

    if args.quick:
        # Smaller problems for fast feedback
        n_qaoa_assets = [4]
        n_vqe_qubits = [3]
        n_grover_qubits = [4]
    else:
        n_qaoa_assets = [4, 6, 8]
        n_vqe_qubits = [3, 4, 5]
        n_grover_qubits = [4, 6, 8]

    print("  QAOA portfolio optimization...", file=sys.stderr)
    for n in n_qaoa_assets:
        benchmarks.append(benchmark_qaoa(n))

    print("  VQE Ising ground state...", file=sys.stderr)
    for n in n_vqe_qubits:
        benchmarks.append(benchmark_vqe(n))

    print("  Grover unstructured search...", file=sys.stderr)
    for n in n_grover_qubits:
        benchmarks.append(benchmark_grover(n, n_marked=1))

    print("  MLQAE value-at-risk...", file=sys.stderr)
    benchmarks.append(benchmark_mlqae())

    print("  Grover arbitrage search...", file=sys.stderr)
    benchmarks.append(benchmark_grover_arb_search())

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_benchmarks": len(benchmarks),
        "benchmarks": benchmarks,
        "global_honest_notes": (
            "All quantum algorithms in this suite run on the in-repo classical "
            "simulator. Quantum query complexity speedups are real on hardware "
            "but do NOT translate to wall-clock speedups in simulation. The "
            "value of these implementations is correctness, framing, and "
            "hardware-portability."
        ),
    }

    output_json = json.dumps(report, indent=2, default=str)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json)
        print(f"Benchmark report written to {out_path}", file=sys.stderr)
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
