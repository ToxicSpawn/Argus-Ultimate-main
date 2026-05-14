"""
IBM Simulator Comparison & Benchmark Suite
Compares basic vs enhanced simulator, benchmarks against theoretical limits
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict

# Import both simulators
from quantum.advanced_local_ibm_simulator import get_ibm_simulator, AdvancedLocalIBMSimulator
from quantum.enhanced_ibm_simulator import get_enhanced_ibm_simulator, EnhancedIBMSimulator

logger = logging.getLogger(__name__)


class SimulatorComparator:
    """Compare and benchmark IBM simulators"""
    
    def __init__(self):
        self.results = {}
    
    def compare_on_circuit(
        self,
        circuit: List[Dict],
        device: str = "ibmq_manila",
        shots: int = 8192
    ) -> Dict[str, Any]:
        """Run same circuit on both simulators and compare"""
        
        print(f"\n{'='*80}")
        print(f"COMPARING SIMULATORS: {device}")
        print(f"Circuit: {len(circuit)} gates, {shots} shots")
        print('='*80)
        
        # Run basic simulator
        print("\n1. Running Basic Simulator...")
        basic_sim = get_ibm_simulator(device, realistic_noise=True)
        basic_start = time.time()
        basic_result = basic_sim.execute(circuit, shots, simulate_queue=False)
        basic_time = time.time() - basic_start
        
        # Run enhanced simulator
        print("\n2. Running Enhanced Simulator...")
        enhanced_sim = get_enhanced_ibm_simulator(device, noise_level='realistic')
        enhanced_start = time.time()
        enhanced_result = enhanced_sim.execute(circuit, shots)
        enhanced_time = time.time() - enhanced_start
        
        # Compare results
        comparison = self._analyze_comparison(
            basic_result, enhanced_result, basic_time, enhanced_time
        )
        
        # Print comparison
        self._print_comparison(comparison)
        
        return comparison
    
    def _analyze_comparison(
        self,
        basic: Dict,
        enhanced: Dict,
        basic_time: float,
        enhanced_time: float
    ) -> Dict[str, Any]:
        """Analyze and compare results"""
        
        basic_counts = basic['results'][0]['data']['counts']
        enhanced_counts = enhanced['results'][0]['data']['counts']
        
        # Calculate distribution similarity
        fidelity = self._distribution_fidelity(basic_counts, enhanced_counts, 8192)
        
        # Get metadata
        basic_meta = basic['header']['metadata']
        enhanced_meta = enhanced['header']['metadata']
        
        return {
            'basic': {
                'execution_time': basic_time,
                'gates_transpiled': basic_meta['transpilation']['n_gates_transpiled'],
                'fidelity_estimate': 0.9,  # Basic doesn't calculate this
            },
            'enhanced': {
                'execution_time': enhanced_time,
                'gates_transpiled': enhanced_meta['circuit']['transpiled_gates'],
                'gates_routed': enhanced_meta['circuit']['routed_gates'],
                'swaps_inserted': enhanced_meta['circuit']['swap_gates_inserted'],
                'circuit_duration_ns': enhanced_meta['timing']['circuit_duration_ns'],
                'fidelity_estimate': enhanced_meta['noise']['estimated_fidelity'],
                'decoherence_error': enhanced_meta['noise']['total_decoherence_error'],
                'gate_error': enhanced_meta['noise']['total_gate_error'],
            },
            'comparison': {
                'fidelity_between_simulators': fidelity,
                'speedup': basic_time / enhanced_time if enhanced_time > 0 else 1.0,
                'time_difference_ms': (basic_time - enhanced_time) * 1000,
            }
        }
    
    def _distribution_fidelity(
        self,
        counts1: Dict[str, int],
        counts2: Dict[str, int],
        shots: int
    ) -> float:
        """Calculate fidelity between two count distributions"""
        all_keys = set(counts1.keys()) | set(counts2.keys())
        
        fidelity = 0.0
        for key in all_keys:
            p1 = counts1.get(key, 0) / shots
            p2 = counts2.get(key, 0) / shots
            fidelity += np.sqrt(p1 * p2)
        
        return fidelity ** 2
    
    def _print_comparison(self, comp: Dict):
        """Print formatted comparison"""
        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print('='*80)
        
        print("\n📊 Basic Simulator:")
        print(f"  Execution time: {comp['basic']['execution_time']*1000:.1f} ms")
        print(f"  Gates transpiled: {comp['basic']['gates_transpiled']}")
        
        print("\n⚡ Enhanced Simulator:")
        print(f"  Execution time: {comp['enhanced']['execution_time']*1000:.1f} ms")
        print(f"  Gates transpiled: {comp['enhanced']['gates_transpiled']}")
        print(f"  Gates routed: {comp['enhanced']['gates_routed']}")
        print(f"  SWAPs inserted: {comp['enhanced']['swaps_inserted']}")
        print(f"  Circuit duration: {comp['enhanced']['circuit_duration_ns']:.0f} ns")
        print(f"  Estimated fidelity: {comp['enhanced']['fidelity_estimate']:.2%}")
        print(f"  Decoherence error: {comp['enhanced']['decoherence_error']:.4f}")
        print(f"  Gate error: {comp['enhanced']['gate_error']:.4f}")
        
        print("\n🔄 Comparison:")
        print(f"  Inter-simulator fidelity: {comp['comparison']['fidelity_between_simulators']:.4f}")
        print(f"  Speedup: {comp['comparison']['speedup']:.2f}x")
        print(f"  Time diff: {comp['comparison']['time_difference_ms']:+.1f} ms")
        
        if comp['comparison']['fidelity_between_simulators'] > 0.95:
            print("\n  ✅ EXCELLENT: Simulators agree >95%")
        elif comp['comparison']['fidelity_between_simulators'] > 0.90:
            print("\n  ✅ GOOD: Simulators agree >90%")
        else:
            print("\n  ⚠️  WARNING: Significant divergence between simulators")


class SimulatorBenchmark:
    """Benchmark simulator performance"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_device(
        self,
        device: str,
        n_qubits: int,
        circuit_depths: List[int],
        shots: int = 1024
    ) -> Dict[str, Any]:
        """Benchmark simulator at various circuit depths"""
        
        print(f"\n{'='*80}")
        print(f"BENCHMARKING: {device} ({n_qubits} qubits)")
        print('='*80)
        
        results = {
            'device': device,
            'n_qubits': n_qubits,
            'shots': shots,
            'depths': [],
            'basic_times': [],
            'enhanced_times': [],
            'fidelities': []
        }
        
        for depth in circuit_depths:
            # Generate random circuit
            circuit = self._generate_random_circuit(n_qubits, depth)
            
            # Benchmark both simulators
            basic_time, enhanced_time, fidelity = self._benchmark_circuit(
                circuit, device, shots
            )
            
            results['depths'].append(depth)
            results['basic_times'].append(basic_time)
            results['enhanced_times'].append(enhanced_time)
            results['fidelities'].append(fidelity)
            
            print(f"  Depth {depth:3d}: Basic={basic_time*1000:6.1f}ms, "
                  f"Enhanced={enhanced_time*1000:6.1f}ms, "
                  f"Fidelity={fidelity:.3f}")
        
        # Calculate statistics
        results['avg_speedup'] = np.mean([
            b/e for b, e in zip(results['basic_times'], results['enhanced_times'])
        ])
        results['avg_fidelity'] = np.mean(results['fidelities'])
        
        print(f"\n  Average speedup: {results['avg_speedup']:.2f}x")
        print(f"  Average fidelity: {results['avg_fidelity']:.3f}")
        
        return results
    
    def _generate_random_circuit(self, n_qubits: int, depth: int) -> List[Dict]:
        """Generate random circuit for benchmarking"""
        circuit = []
        
        for _ in range(depth):
            # Random single-qubit gate
            q = np.random.randint(0, n_qubits)
            gate_type = np.random.choice(['H', 'X', 'RZ', 'SX'])
            
            if gate_type == 'RZ':
                circuit.append({
                    'type': 'RZ',
                    'qubits': [q],
                    'params': [np.random.uniform(0, 2*np.pi)]
                })
            else:
                circuit.append({'type': gate_type, 'qubits': [q]})
            
            # Random two-qubit gate (with probability)
            if np.random.random() < 0.3 and n_qubits > 1:
                control = np.random.randint(0, n_qubits)
                target = np.random.randint(0, n_qubits)
                if control != target:
                    circuit.append({
                        'type': 'CX',
                        'qubits': [control, target]
                    })
        
        return circuit
    
    def _benchmark_circuit(
        self,
        circuit: List[Dict],
        device: str,
        shots: int
    ) -> Tuple[float, float, float]:
        """Benchmark single circuit, return (basic_time, enhanced_time, fidelity)"""
        
        # Basic simulator
        basic_sim = get_ibm_simulator(device, realistic_noise=True)
        start = time.time()
        basic_result = basic_sim.execute(circuit, shots, simulate_queue=False)
        basic_time = time.time() - start
        
        # Enhanced simulator
        enhanced_sim = get_enhanced_ibm_simulator(device, noise_level='realistic')
        start = time.time()
        enhanced_result = enhanced_sim.execute(circuit, shots)
        enhanced_time = time.time() - start
        
        # Calculate fidelity between results
        basic_counts = basic_result['results'][0]['data']['counts']
        enhanced_counts = enhanced_result['results'][0]['data']['counts']
        
        fidelity = self._calculate_fidelity(basic_counts, enhanced_counts, shots)
        
        return basic_time, enhanced_time, fidelity
    
    def _calculate_fidelity(
        self,
        counts1: Dict[str, int],
        counts2: Dict[str, int],
        shots: int
    ) -> float:
        """Calculate Hellinger fidelity"""
        all_keys = set(counts1.keys()) | set(counts2.keys())
        fidelity = 0.0
        
        for key in all_keys:
            p1 = counts1.get(key, 0) / shots
            p2 = counts2.get(key, 0) / shots
            fidelity += np.sqrt(p1 * p2)
        
        return fidelity ** 2


def run_full_comparison():
    """Run comprehensive comparison between simulators"""
    
    print("=" * 80)
    print("⚛️  IBM SIMULATOR COMPARISON SUITE")
    print("=" * 80)
    print("\nThis will compare the basic and enhanced IBM simulators")
    print("to verify accuracy and measure performance improvements.")
    
    comparator = SimulatorComparator()
    benchmark = SimulatorBenchmark()
    
    # Test circuits
    test_circuits = {
        "Bell State": [
            {'type': 'H', 'qubits': [0]},
            {'type': 'CX', 'qubits': [0, 1]},
        ],
        "GHZ State": [
            {'type': 'H', 'qubits': [0]},
            {'type': 'CX', 'qubits': [0, 1]},
            {'type': 'CX', 'qubits': [1, 2]},
            {'type': 'CX', 'qubits': [2, 3]},
        ],
        "QFT": [
            {'type': 'H', 'qubits': [0]},
            {'type': 'RZ', 'qubits': [1], 'params': [np.pi/4]},
            {'type': 'CX', 'qubits': [0, 1]},
            {'type': 'H', 'qubits': [1]},
            {'type': 'RZ', 'qubits': [2], 'params': [np.pi/8]},
            {'type': 'CX', 'qubits': [0, 2]},
            {'type': 'RZ', 'qubits': [2], 'params': [np.pi/4]},
            {'type': 'CX', 'qubits': [1, 2]},
            {'type': 'H', 'qubits': [2]},
        ],
        "Variational": [
            {'type': 'RY', 'qubits': [0], 'params': [0.5]},
            {'type': 'RY', 'qubits': [1], 'params': [0.8]},
            {'type': 'CX', 'qubits': [0, 1]},
            {'type': 'RZ', 'qubits': [0], 'params': [1.2]},
            {'type': 'RZ', 'qubits': [1], 'params': [0.9]},
            {'type': 'CX', 'qubits': [0, 1]},
            {'type': 'RY', 'qubits': [0], 'params': [0.3]},
            {'type': 'RY', 'qubits': [1], 'params': [0.6]},
        ]
    }
    
    # Run circuit comparisons
    print("\n" + "=" * 80)
    print("PART 1: CIRCUIT COMPARISONS")
    print("=" * 80)
    
    all_comparisons = {}
    for name, circuit in test_circuits.items():
        comp = comparator.compare_on_circuit(circuit, "ibmq_manila", 8192)
        all_comparisons[name] = comp
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("PART 2: PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    benchmark_results = benchmark.benchmark_device(
        device="ibmq_manila",
        n_qubits=5,
        circuit_depths=[10, 20, 50, 100],
        shots=1024
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    avg_fidelity = np.mean([
        comp['comparison']['fidelity_between_simulators']
        for comp in all_comparisons.values()
    ])
    avg_speedup = np.mean([
        comp['comparison']['speedup']
        for comp in all_comparisons.values()
    ])
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Average inter-simulator fidelity: {avg_fidelity:.4f}")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Benchmark fidelity: {benchmark_results['avg_fidelity']:.4f}")
    print(f"  Benchmark speedup: {benchmark_results['avg_speedup']:.2f}x")
    
    if avg_fidelity > 0.95:
        print("\n  ✅ EXCELLENT: Simulators highly consistent (>95%)")
    elif avg_fidelity > 0.90:
        print("\n  ✅ GOOD: Simulators consistent (>90%)")
    else:
        print("\n  ⚠️  WARNING: Some divergence detected")
    
    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE")
    print('='*80)
    
    return {
        'comparisons': all_comparisons,
        'benchmark': benchmark_results
    }


if __name__ == '__main__':
    results = run_full_comparison()
