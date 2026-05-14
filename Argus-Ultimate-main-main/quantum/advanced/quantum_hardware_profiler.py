"""
Quantum Hardware Profiler

This module profiles and optimizes quantum algorithms for specific hardware backends.
Key features include:
- Hardware capability analysis
- Circuit optimization for specific backends
- Quantum resource estimation
- Hardware-specific parameter tuning
- Performance benchmarking
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class QuantumBackendType(Enum):
    """Supported quantum backend types"""
    SIMULATOR = auto()
    IBM_QISKIT = auto()
    DWAVE = auto()
    RIGETTI = auto()
    IONQ = auto()
    QUERA = auto()
    CUSTOM = auto()

@dataclass
class QuantumBackendProfile:
    """Profile of a quantum hardware backend"""
    backend_type: QuantumBackendType
    name: str
    version: str
    qubits: int
    topology: str
    gate_error_rate: float
    readout_error_rate: float
    qubit_coherence_time_us: float
    gate_time_ns: float
    max_circuit_depth: int
    queue_depth: int
    queue_time_ms: float
    supported_gates: List[str]
    
    def calculate_quantum_volume(self) -> int:
        """Calculate quantum volume metric"""
        # Simplified quantum volume calculation
        # In reality, this would involve running specific benchmark circuits
        return min(self.qubits, 
                  int(np.sqrt(self.max_circuit_depth) * 
                      (1 - self.gate_error_rate) * 100))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'backend_type': self.backend_type.name,
            'name': self.name,
            'version': self.version,
            'qubits': self.qubits,
            'topology': self.topology,
            'gate_error_rate': self.gate_error_rate,
            'readout_error_rate': self.readout_error_rate,
            'qubit_coherence_time_us': self.qubit_coherence_time_us,
            'gate_time_ns': self.gate_time_ns,
            'max_circuit_depth': self.max_circuit_depth,
            'queue_depth': self.queue_depth,
            'queue_time_ms': self.queue_time_ms,
            'supported_gates': self.supported_gates,
            'quantum_volume': self.calculate_quantum_volume()
        }

@dataclass
class QuantumCircuitProfile:
    """Profile of a quantum circuit for hardware optimization"""
    circuit_id: str
    num_qubits: int
    depth: int
    gate_count: int
    gate_types: Dict[str, int]  # Gate type to count mapping
    connectivity: List[Tuple[int, int]]  # Qubit connectivity requirements
    
    def calculate_circuit_complexity(self) -> float:
        """Calculate circuit complexity score"""
        # Complexity based on qubits, depth, and gate diversity
        gate_diversity = len(self.gate_types)
        return (self.num_qubits * self.depth * gate_diversity) / 1000

@dataclass
class HardwareOptimizationResult:
    """Result of hardware optimization"""
    backend: QuantumBackendProfile
    circuit: QuantumCircuitProfile
    optimized_circuit: QuantumCircuitProfile
    optimization_metrics: Dict[str, float]
    execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'backend': self.backend.name,
            'circuit_id': self.circuit.circuit_id,
            'original_qubits': self.circuit.num_qubits,
            'optimized_qubits': self.optimized_circuit.num_qubits,
            'original_depth': self.circuit.depth,
            'optimized_depth': self.optimized_circuit.depth,
            'original_gates': self.circuit.gate_count,
            'optimized_gates': self.optimized_circuit.gate_count,
            'optimization_metrics': self.optimization_metrics,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp.isoformat()
        }

class QuantumHardwareProfiler:
    """Quantum hardware profiler and optimizer"""
    
    def __init__(self):
        # Predefined backend profiles
        self.backend_profiles = {
            'ibm_lagos': QuantumBackendProfile(
                backend_type=QuantumBackendType.IBM_QISKIT,
                name='ibm_lagos',
                version='1.0.0',
                qubits=7,
                topology='heavy-hex',
                gate_error_rate=0.001,
                readout_error_rate=0.05,
                qubit_coherence_time_us=100,
                gate_time_ns=50,
                max_circuit_depth=1000,
                queue_depth=50,
                queue_time_ms=1000,
                supported_gates=['rx', 'ry', 'rz', 'cx', 'id', 'measure']
            ),
            'ibm_nairobi': QuantumBackendProfile(
                backend_type=QuantumBackendType.IBM_QISKIT,
                name='ibm_nairobi',
                version='1.0.0',
                qubits=7,
                topology='heavy-hex',
                gate_error_rate=0.0008,
                readout_error_rate=0.04,
                qubit_coherence_time_us=120,
                gate_time_ns=45,
                max_circuit_depth=1200,
                queue_depth=30,
                queue_time_ms=1500,
                supported_gates=['rx', 'ry', 'rz', 'cx', 'id', 'measure']
            ),
            'dwave_advantage': QuantumBackendProfile(
                backend_type=QuantumBackendType.DWAVE,
                name='dwave_advantage',
                version='5.0',
                qubits=5000,
                topology='chimera',
                gate_error_rate=0.005,
                readout_error_rate=0.1,
                qubit_coherence_time_us=50,
                gate_time_ns=100,
                max_circuit_depth=50,  # Different metric for annealers
                queue_depth=100,
                queue_time_ms=5000,
                supported_gates=['qubit_bias', 'coupler_strength']
            ),
            'simulator_statevector': QuantumBackendProfile(
                backend_type=QuantumBackendType.SIMULATOR,
                name='simulator_statevector',
                version='1.0',
                qubits=1024,
                topology='all-to-all',
                gate_error_rate=0.0,
                readout_error_rate=0.0,
                qubit_coherence_time_us=float('inf'),
                gate_time_ns=10,
                max_circuit_depth=100000,
                queue_depth=0,
                queue_time_ms=0,
                supported_gates=['rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'h', 'x', 'y', 'z', 's', 't', 'measure']
            )
        }
        
        self.optimization_history = []
    
    def get_backend_profile(self, backend_name: str) -> QuantumBackendProfile:
        """Get profile for a specific backend"""
        if backend_name not in self.backend_profiles:
            raise ValueError(f"Backend {backend_name} not found")
        return self.backend_profiles[backend_name]
    
    def add_custom_backend(self, profile: QuantumBackendProfile):
        """Add a custom backend profile"""
        self.backend_profiles[profile.name] = profile
    
    def profile_backend(self, backend_name: str, circuit: QuantumCircuitProfile) -> Dict[str, Any]:
        """
        Profile a quantum circuit on a specific backend
        
        Args:
            backend_name: Name of the backend
            circuit: Quantum circuit to profile
            
        Returns:
            Profiling results
        """
        if backend_name not in self.backend_profiles:
            raise ValueError(f"Backend {backend_name} not found")
            
        backend = self.backend_profiles[backend_name]
        
        # Calculate compatibility metrics
        qubit_compatibility = min(1.0, circuit.num_qubits / backend.qubits)
        depth_compatibility = min(1.0, circuit.depth / backend.max_circuit_depth)
        
        # Calculate expected performance metrics
        expected_fidelity = self._calculate_expected_fidelity(backend, circuit)
        expected_execution_time = self._calculate_expected_execution_time(backend, circuit)
        queue_time_adjusted = expected_execution_time + backend.queue_time_ms
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(backend, circuit)
        
        return {
            'backend': backend.to_dict(),
            'circuit': {
                'id': circuit.circuit_id,
                'qubits': circuit.num_qubits,
                'depth': circuit.depth,
                'gates': circuit.gate_count,
                'complexity': circuit.calculate_circuit_complexity()
            },
            'compatibility': {
                'qubit_compatibility': qubit_compatibility,
                'depth_compatibility': depth_compatibility,
                'overall_compatibility': min(qubit_compatibility, depth_compatibility),
                'supported_gates': all(gate in backend.supported_gates 
                                      for gate in circuit.gate_types.keys())
            },
            'performance': {
                'expected_fidelity': expected_fidelity,
                'expected_execution_time_ms': expected_execution_time,
                'queue_time_adjusted_ms': queue_time_adjusted,
                'quantum_volume_utilization': circuit.calculate_circuit_complexity() / backend.calculate_quantum_volume()
            },
            'optimization_potential': optimization_potential,
            'recommendations': self._generate_optimization_recommendations(backend, circuit)
        }
    
    def _calculate_expected_fidelity(self, backend: QuantumBackendProfile, circuit: QuantumCircuitProfile) -> float:
        """Calculate expected fidelity for circuit on backend"""
        # Base fidelity based on backend error rates
        if backend.backend_type == QuantumBackendType.SIMULATOR:
            return 1.0
        
        # Fidelity degrades with circuit depth and gate count
        depth_penalty = (1 - backend.gate_error_rate) ** circuit.depth
        gate_penalty = (1 - backend.gate_error_rate) ** circuit.gate_count
        readout_penalty = (1 - backend.readout_error_rate) ** circuit.num_qubits
        
        # Coherence time impact
        total_time_ns = circuit.depth * backend.gate_time_ns
        coherence_penalty = np.exp(-total_time_ns / (backend.qubit_coherence_time_us * 1000))
        
        return depth_penalty * gate_penalty * readout_penalty * coherence_penalty
    
    def _calculate_expected_execution_time(self, backend: QuantumBackendProfile, circuit: QuantumCircuitProfile) -> float:
        """Calculate expected execution time"""
        if backend.backend_type == QuantumBackendType.SIMULATOR:
            # Simulator time scales with circuit complexity
            complexity = circuit.calculate_circuit_complexity()
            return max(1, complexity * 0.1)  # 0.1ms per complexity unit
        else:
            # Real hardware: depth * gate time + overhead
            base_time = circuit.depth * backend.gate_time_ns / 1_000_000  # Convert to ms
            return max(1, base_time + 5)  # Add 5ms overhead
    
    def _calculate_optimization_potential(self, backend: QuantumBackendProfile, circuit: QuantumCircuitProfile) -> Dict[str, float]:
        """Calculate potential for optimization"""
        potential = {}
        
        # Qubit reduction potential
        if circuit.num_qubits > backend.qubits * 0.8:
            potential['qubit_reduction'] = min(0.5, (circuit.num_qubits - backend.qubits * 0.8) / circuit.num_qubits)
        else:
            potential['qubit_reduction'] = 0.0
        
        # Depth reduction potential
        if circuit.depth > backend.max_circuit_depth * 0.7:
            potential['depth_reduction'] = min(0.5, (circuit.depth - backend.max_circuit_depth * 0.7) / circuit.depth)
        else:
            potential['depth_reduction'] = 0.0
        
        # Gate reduction potential (based on gate diversity)
        gate_diversity = len(circuit.gate_types)
        if gate_diversity > 3:  # More than 3 gate types suggests optimization potential
            potential['gate_reduction'] = min(0.3, (gate_diversity - 3) / 10)
        else:
            potential['gate_reduction'] = 0.0
        
        # Topology-aware optimization potential
        if backend.topology != 'all-to-all':
            # Count non-local gates (would require SWAPs on limited topology)
            non_local_gates = 0
            for (q1, q2) in circuit.connectivity:
                if abs(q1 - q2) > 1:  # Not adjacent qubits
                    non_local_gates += 1
            
            if non_local_gates > 0:
                potential['topology_optimization'] = min(0.4, non_local_gates / circuit.gate_count)
            else:
                potential['topology_optimization'] = 0.0
        else:
            potential['topology_optimization'] = 0.0
        
        # Overall potential
        potential['overall'] = sum(potential.values())
        
        return potential
    
    def _generate_optimization_recommendations(self, backend: QuantumBackendProfile, circuit: QuantumCircuitProfile) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Qubit recommendations
        if circuit.num_qubits > backend.qubits * 0.9:
            recommendations.append(f"Reduce qubit count from {circuit.num_qubits} to ≤{int(backend.qubits * 0.9)} (90% of backend capacity)")
        
        # Depth recommendations
        if circuit.depth > backend.max_circuit_depth * 0.8:
            recommendations.append(f"Reduce circuit depth from {circuit.depth} to ≤{int(backend.max_circuit_depth * 0.8)} (80% of max depth)")
        
        # Gate recommendations
        if len(circuit.gate_types) > 5:
            recommendations.append(f"Simplify gate set (currently {len(circuit.gate_types)} gate types)")
        
        # Topology recommendations
        if backend.topology != 'all-to-all':
            non_local_gates = 0
            for (q1, q2) in circuit.connectivity:
                if abs(q1 - q2) > 1:
                    non_local_gates += 1
            
            if non_local_gates > circuit.gate_count * 0.1:
                recommendations.append(f"Optimize qubit connectivity for {backend.topology} topology (currently {non_local_gates} non-local gates)")
        
        # Error mitigation recommendations
        if backend.gate_error_rate > 0.001:
            recommendations.append(f"Apply error mitigation techniques (backend gate error: {backend.gate_error_rate:.3f})")
        
        # Queue time recommendations
        if backend.queue_time_ms > 1000:
            recommendations.append(f"Consider batching circuits to amortize queue time ({backend.queue_time_ms}ms)")
        
        return recommendations
    
    def optimize_for_backend(self, backend_name: str, circuit: QuantumCircuitProfile) -> HardwareOptimizationResult:
        """
        Optimize a quantum circuit for a specific backend
        
        Args:
            backend_name: Name of the backend
            circuit: Quantum circuit to optimize
            
        Returns:
            Optimization result
        """
        if backend_name not in self.backend_profiles:
            raise ValueError(f"Backend {backend_name} not found")
            
        backend = self.backend_profiles[backend_name]
        
        # Start timing
        start_time = time.time()
        
        # Create optimized circuit (simplified for demo - real implementation would use actual optimization)
        optimized_circuit = QuantumCircuitProfile(
            circuit_id=circuit.circuit_id + "_optimized",
            num_qubits=min(circuit.num_qubits, int(backend.qubits * 0.9)),
            depth=min(circuit.depth, int(backend.max_circuit_depth * 0.8)),
            gate_count=int(circuit.gate_count * 0.8),  # 20% gate reduction
            gate_types=circuit.gate_types,  # Keep same gate types for simplicity
            connectivity=self._optimize_connectivity(backend, circuit)
        )
        
        # Calculate optimization metrics
        qubit_reduction = 1 - (optimized_circuit.num_qubits / circuit.num_qubits)
        depth_reduction = 1 - (optimized_circuit.depth / circuit.depth)
        gate_reduction = 1 - (optimized_circuit.gate_count / circuit.gate_count)
        
        # Calculate expected performance improvement
        original_fidelity = self._calculate_expected_fidelity(backend, circuit)
        optimized_fidelity = self._calculate_expected_fidelity(backend, optimized_circuit)
        
        original_time = self._calculate_expected_execution_time(backend, circuit)
        optimized_time = self._calculate_expected_execution_time(backend, optimized_circuit)
        
        optimization_metrics = {
            'qubit_reduction': qubit_reduction,
            'depth_reduction': depth_reduction,
            'gate_reduction': gate_reduction,
            'fidelity_improvement': optimized_fidelity - original_fidelity,
            'time_reduction': original_time - optimized_time,
            'complexity_reduction': circuit.calculate_circuit_complexity() - optimized_circuit.calculate_circuit_complexity()
        }
        
        # Create result
        result = HardwareOptimizationResult(
            backend=backend,
            circuit=circuit,
            optimized_circuit=optimized_circuit,
            optimization_metrics=optimization_metrics,
            execution_time_ms=(time.time() - start_time) * 1000  # Convert to ms
        )
        
        # Store in history
        self.optimization_history.append(result)
        
        return result
    
    def _optimize_connectivity(self, backend: QuantumBackendProfile, circuit: QuantumCircuitProfile) -> List[Tuple[int, int]]:
        """Optimize qubit connectivity for specific backend topology"""
        if backend.topology == 'all-to-all':
            return circuit.connectivity  # No optimization needed
        
        # For limited topologies, we need to optimize connectivity
        # This is a simplified version - real implementation would use more sophisticated algorithms
        optimized_connectivity = []
        
        # Sort qubits by degree (number of connections)
        qubit_degrees = {q: 0 for q in range(circuit.num_qubits)}
        for (q1, q2) in circuit.connectivity:
            qubit_degrees[q1] += 1
            qubit_degrees[q2] += 1
        
        sorted_qubits = sorted(qubit_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Try to place high-degree qubits close to each other
        qubit_mapping = {old: new for new, (old, _) in enumerate(sorted_qubits)}
        
        # Remap connectivity
        for (q1, q2) in circuit.connectivity:
            new_q1 = qubit_mapping[q1]
            new_q2 = qubit_mapping[q2]
            
            # For non-all-to-all topologies, only keep local connections
            if backend.topology == 'heavy-hex' and abs(new_q1 - new_q2) <= 1:
                optimized_connectivity.append((new_q1, new_q2))
            elif backend.topology == 'chimera' and (abs(new_q1 - new_q2) <= 1 or 
                (new_q1 % 2 == 0 and new_q2 == new_q1 + 1) or
                (new_q2 % 2 == 0 and new_q1 == new_q2 + 1)):
                optimized_connectivity.append((new_q1, new_q2))
            elif backend.topology == 'linear' and abs(new_q1 - new_q2) == 1:
                optimized_connectivity.append((new_q1, new_q2))
        
        return optimized_connectivity
    
    def compare_backends(self, circuit: QuantumCircuitProfile, backend_names: List[str]) -> Dict[str, Any]:
        """
        Compare a circuit across multiple backends
        
        Args:
            circuit: Quantum circuit to compare
            backend_names: List of backend names to compare
            
        Returns:
            Comparison results
        """
        results = {}
        
        for backend_name in backend_names:
            if backend_name not in self.backend_profiles:
                logger.warning(f"Backend {backend_name} not found, skipping")
                continue
                
            results[backend_name] = self.profile_backend(backend_name, circuit)
        
        return {
            'circuit': {
                'id': circuit.circuit_id,
                'qubits': circuit.num_qubits,
                'depth': circuit.depth,
                'gates': circuit.gate_count,
                'complexity': circuit.calculate_circuit_complexity()
            },
            'backends': results,
            'best_backend': self._select_best_backend(results) if results else None
        }
    
    def _select_best_backend(self, comparison_results: Dict[str, Any]) -> str:
        """Select the best backend based on comparison results"""
        if not comparison_results:
            return None
            
        # Score backends based on multiple factors
        best_backend = None
        best_score = -float('inf')
        
        for backend_name, result in comparison_results.items():
            # Calculate score based on:
            # 1. Compatibility (50% weight)
            # 2. Expected fidelity (30% weight)
            # 3. Execution time (20% weight)
            
            compatibility = result['compatibility']['overall_compatibility']
            fidelity = result['performance']['expected_fidelity']
            exec_time = 1 / (result['performance']['expected_execution_time_ms'] + 1)  # Invert for score
            
            score = (
                compatibility * 0.5 +
                fidelity * 0.3 +
                exec_time * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_backend = backend_name
        
        return best_backend
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent optimization history"""
        return [result.to_dict() for result in self.optimization_history[-limit:]]
    
    def generate_hardware_report(self) -> Dict[str, Any]:
        """Generate a report of all available hardware backends"""
        return {
            'backends': {name: profile.to_dict() for name, profile in self.backend_profiles.items()},
            'total_backends': len(self.backend_profiles),
            'best_quantum_volume': max(profile.calculate_quantum_volume() 
                                     for profile in self.backend_profiles.values())
        }
    
    def estimate_quantum_resource_requirements(self, circuit: QuantumCircuitProfile) -> Dict[str, Any]:
        """
        Estimate quantum resources required for a circuit
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            Resource requirements estimate
        """
        return {
            'qubits': circuit.num_qubits,
            'min_qubits_recommended': max(5, int(circuit.num_qubits * 1.2)),  # 20% buffer
            'depth': circuit.depth,
            'gates': circuit.gate_count,
            'estimated_quantum_volume': circuit.calculate_circuit_complexity() * 10,
            'error_correction_overhead': {
                'qubits': int(circuit.num_qubits * 3) if circuit.num_qubits > 20 else circuit.num_qubits,
                'depth': int(circuit.depth * 5) if circuit.depth > 100 else circuit.depth
            },
            'suitable_backends': [
                name for name, profile in self.backend_profiles.items()
                if circuit.num_qubits <= profile.qubits and 
                   circuit.depth <= profile.max_circuit_depth
            ]
        }

@dataclass
class QuantumHardwareSelector:
    """Intelligent quantum hardware selector"""
    
    def __init__(self, profiler: QuantumHardwareProfiler):
        self.profiler = profiler
        self.selection_history = []
    
    def select_best_backend(self, circuit: QuantumCircuitProfile,
                           optimization_goal: str = 'balanced') -> Tuple[str, Dict[str, Any]]:
        """
        Select the best quantum backend for a given circuit
        
        Args:
            circuit: Quantum circuit to run
            optimization_goal: Optimization goal ('speed', 'fidelity', 'balanced')
            
        Returns:
            Tuple of (backend_name, selection_reasoning)
        """
        # Get all available backends
        backend_names = list(self.profiler.backend_profiles.keys())
        
        # Profile circuit on all backends
        comparison = self.profiler.compare_backends(circuit, backend_names)
        
        if not comparison['backends']:
            return None, {"error": "No suitable backends found"}
        
        # Select based on optimization goal
        if optimization_goal == 'speed':
            best_backend = min(
                comparison['backends'].items(),
                key=lambda x: x[1]['performance']['expected_execution_time_ms'] + x[1]['performance']['queue_time_adjusted_ms']
            )[0]
        elif optimization_goal == 'fidelity':
            best_backend = max(
                comparison['backends'].items(),
                key=lambda x: x[1]['performance']['expected_fidelity']
            )[0]
        else:  # balanced
            best_backend = self.profiler._select_best_backend(comparison['backends'])
            
        # Get the selected backend's profile
        backend_profile = comparison['backends'][best_backend]
        
        # Record selection
        self.selection_history.append({
            'circuit_id': circuit.circuit_id,
            'selected_backend': best_backend,
            'optimization_goal': optimization_goal,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'expected_fidelity': backend_profile['performance']['expected_fidelity'],
                'expected_time_ms': backend_profile['performance']['expected_execution_time_ms'],
                'compatibility': backend_profile['compatibility']['overall_compatibility']
            }
        })
        
        return best_backend, {
            'reasoning': f"Selected {best_backend} for {optimization_goal} optimization",
            'expected_fidelity': backend_profile['performance']['expected_fidelity'],
            'expected_time_ms': backend_profile['performance']['expected_execution_time_ms'],
            'queue_time_ms': backend_profile['performance']['queue_time_adjusted_ms'],
            'compatibility': backend_profile['compatibility']['overall_compatibility'],
            'recommendations': backend_profile['recommendations']
        }
    
    def get_selection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent backend selection history"""
        return self.selection_history[-limit:]
    
    def analyze_selection_patterns(self) -> Dict[str, Any]:
        """Analyze backend selection patterns"""
        if not self.selection_history:
            return {"status": "no_selections"}
            
        # Count selections by backend
        backend_counts = {}
        for selection in self.selection_history:
            backend = selection['selected_backend']
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        
        # Calculate selection trends
        optimization_goals = [s['optimization_goal'] for s in self.selection_history]
        goal_counts = {}
        for goal in optimization_goals:
            goal_counts[goal] = goal_counts.get(goal, 0) + 1
        
        return {
            'total_selections': len(self.selection_history),
            'backend_distribution': backend_counts,
            'most_used_backend': max(backend_counts.items(), key=lambda x: x[1])[0] if backend_counts else None,
            'optimization_goal_distribution': goal_counts,
            'recent_trend': self.selection_history[-1]['optimization_goal'] if self.selection_history else None
        }

@dataclass
class QuantumExecutionManager:
    """Quantum execution manager with fallback capabilities"""
    
    def __init__(self, profiler: QuantumHardwareProfiler, selector: QuantumHardwareSelector):
        self.profiler = profiler
        self.selector = selector
        self.execution_history = []
        self.fallback_count = 0
    
    def execute_circuit(self, circuit: QuantumCircuitProfile,
                       optimization_goal: str = 'balanced',
                       max_retries: int = 3) -> Dict[str, Any]:
        """
        Execute a quantum circuit with automatic backend selection and fallback
        
        Args:
            circuit: Quantum circuit to execute
            optimization_goal: Optimization goal for backend selection
            max_retries: Maximum number of retries on failure
            
        Returns:
            Execution result
        """
        start_time = time.time()
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # Select best backend
                backend_name, selection_info = self.selector.select_best_backend(circuit, optimization_goal)
                
                if not backend_name:
                    raise RuntimeError("No suitable backend found")
                
                # Get backend profile
                backend = self.profiler.get_backend_profile(backend_name)
                
                # Simulate execution (in real implementation, this would call actual quantum hardware)
                execution_result = self._simulate_quantum_execution(backend, circuit)
                
                # Record successful execution
                execution_time = (time.time() - start_time) * 1000  # ms
                
                result = {
                    'status': 'success',
                    'backend': backend_name,
                    'circuit_id': circuit.circuit_id,
                    'execution_time_ms': execution_time,
                    'quantum_results': execution_result,
                    'selection_info': selection_info,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.execution_history.append(result)
                return result
                
            except Exception as e:
                last_error = str(e)
                retries += 1
                logger.warning(f"Execution attempt {retries} failed: {e}")
                
                # If this was our last retry, fall back to simulator
                if retries >= max_retries:
                    logger.warning("Max retries reached, falling back to simulator")
                    self.fallback_count += 1
                    
                    # Use simulator as fallback
                    simulator = self.profiler.get_backend_profile('simulator_statevector')
                    execution_result = self._simulate_quantum_execution(simulator, circuit)
                    
                    execution_time = (time.time() - start_time) * 1000  # ms
                    
                    result = {
                        'status': 'fallback_success',
                        'backend': 'simulator_statevector',
                        'circuit_id': circuit.circuit_id,
                        'execution_time_ms': execution_time,
                        'quantum_results': execution_result,
                        'original_error': last_error,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.execution_history.append(result)
                    return result
                
                # Wait before retrying
                time.sleep(0.1 * retries)  # Exponential backoff
        
        # Should never reach here
        raise RuntimeError(f"Failed to execute circuit after {max_retries} retries: {last_error}")
    
    def _simulate_quantum_execution(self, backend: QuantumBackendProfile, circuit: QuantumCircuitProfile) -> Dict[str, Any]:
        """Simulate quantum execution (replace with actual hardware calls in production)"""
        # Simulate execution results based on backend and circuit characteristics
        
        # Calculate expected fidelity
        fidelity = self.profiler._calculate_expected_fidelity(backend, circuit)
        
        # Simulate measurement results (random for simulation)
        measurements = np.random.randint(0, 2, circuit.num_qubits)
        
        # Simulate execution metadata
        return {
            'fidelity': fidelity,
            'measurements': measurements.tolist(),
            'execution_metadata': {
                'qubits_used': circuit.num_qubits,
                'circuit_depth': circuit.depth,
                'gate_count': circuit.gate_count,
                'backend_quantum_volume': backend.calculate_quantum_volume(),
                'circuit_complexity': circuit.calculate_circuit_complexity(),
                'expected_fidelity': fidelity,
                'simulated': True  # Mark as simulated for testing
            }
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.execution_history:
            return {"status": "no_executions"}
            
        successful = [e for e in self.execution_history if e['status'] == 'success']
        fallbacks = [e for e in self.execution_history if e['status'] == 'fallback_success']
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful),
            'fallback_executions': len(fallbacks),
            'fallback_rate': len(fallbacks) / len(self.execution_history) if self.execution_history else 0,
            'avg_execution_time_ms': np.mean([e['execution_time_ms'] for e in self.execution_history]) if self.execution_history else 0,
            'backend_distribution': self._calculate_backend_distribution(),
            'recent_fidelity': self.execution_history[-1]['quantum_results']['fidelity'] if self.execution_history else 0
        }
    
    def _calculate_backend_distribution(self) -> Dict[str, int]:
        """Calculate backend usage distribution"""
        backend_counts = {}
        for execution in self.execution_history:
            backend = execution['backend']
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        return backend_counts

@dataclass
class QuantumResourceManager:
    """Quantum resource manager for tracking and allocating quantum resources"""
    
    def __init__(self, profiler: QuantumHardwareProfiler):
        self.profiler = profiler
        self.resource_allocation = {}
        self.usage_history = []
        
    def allocate_resources(self, circuit: QuantumCircuitProfile,
                          priority: int = 1) -> Dict[str, Any]:
        """
        Allocate quantum resources for a circuit
        
        Args:
            circuit: Quantum circuit to allocate resources for
            priority: Priority level (1-5, 1 being highest)
            
        Returns:
            Allocation result
        """
        # Find suitable backends
        suitable_backends = []
        for backend_name, backend in self.profiler.backend_profiles.items():
            if (circuit.num_qubits <= backend.qubits and
                circuit.depth <= backend.max_circuit_depth):
                suitable_backends.append((backend_name, backend))
        
        if not suitable_backends:
            return {
                'status': 'failed',
                'reason': 'No suitable backends available for circuit requirements',
                'circuit_requirements': {
                    'qubits': circuit.num_qubits,
                    'depth': circuit.depth
                }
            }
        
        # Sort by suitability (quantum volume, queue depth, error rates)
        suitable_backends.sort(
            key=lambda x: (
                -x[1].calculate_quantum_volume(),  # Higher quantum volume first
                x[1].queue_depth,                   # Lower queue depth first
                x[1].gate_error_rate               # Lower error rate first
            )
        )
        
        # Select best available backend
        best_backend_name, best_backend = suitable_backends[0]
        
        # Check if backend is already allocated
        if best_backend_name in self.resource_allocation:
            current_usage = sum(
                self.resource_allocation[best_backend_name].values()
            )
            if current_usage + circuit.num_qubits > best_backend.qubits * 0.9:
                # Try to find another backend
                for backend_name, backend in suitable_backends[1:]:
                    if backend_name not in self.resource_allocation or (
                        sum(self.resource_allocation[backend_name].values()) +
                        circuit.num_qubits <= backend.qubits * 0.9
                    ):
                        best_backend_name = backend_name
                        best_backend = backend
                        break
        
        # Allocate resources
        if best_backend_name not in self.resource_allocation:
            self.resource_allocation[best_backend_name] = {}
            
        self.resource_allocation[best_backend_name][circuit.circuit_id] = {
            'qubits': circuit.num_qubits,
            'depth': circuit.depth,
            'priority': priority,
            'allocated_at': datetime.now().isoformat()
        }
        
        # Record allocation
        self.usage_history.append({
            'circuit_id': circuit.circuit_id,
            'backend': best_backend_name,
            'qubits_allocated': circuit.num_qubits,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'status': 'allocated',
            'backend': best_backend_name,
            'backend_profile': best_backend.to_dict(),
            'circuit_id': circuit.circuit_id,
            'qubits_allocated': circuit.num_qubits,
            'priority': priority,
            'allocation_time': datetime.now().isoformat()
        }
    
    def release_resources(self, circuit_id: str) -> Dict[str, Any]:
        """Release allocated quantum resources"""
        # Find and remove the allocation
        backend_found = None
        for backend_name in self.resource_allocation:
            if circuit_id in self.resource_allocation[backend_name]:
                backend_found = backend_name
                del self.resource_allocation[backend_name][circuit_id]
                
                # If no more allocations for this backend, remove it
                if not self.resource_allocation[backend_name]:
                    del self.resource_allocation[backend_name]
                
                break
        
        if not backend_found:
            return {
                'status': 'failed',
                'reason': f'Circuit {circuit_id} not found in allocations'
            }
        
        # Record release
        self.usage_history.append({
            'circuit_id': circuit_id,
            'backend': backend_found,
            'action': 'release',
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'status': 'released',
            'circuit_id': circuit_id,
            'backend': backend_found,
            'release_time': datetime.now().isoformat()
        }
    
    def get_resource_allocation(self) -> Dict[str, Any]:
        """Get current resource allocation status"""
        return {
            'allocations': {
                backend: {
                    'total_qubits': sum(allocation['qubits'] for allocation in allocations.values()),
                    'max_qubits': self.profiler.backend_profiles[backend].qubits,
                    'utilization': sum(allocation['qubits'] for allocation in allocations.values()) /
                                   self.profiler.backend_profiles[backend].qubits,
                    'circuits': list(allocations.keys())
                }
                for backend, allocations in self.resource_allocation.items()
            },
            'total_utilization': sum(
                sum(allocation['qubits'] for allocation in allocations.values())
                for allocations in self.resource_allocation.values()
            ) / sum(
                self.profiler.backend_profiles[backend].qubits
                for backend in self.resource_allocation.keys()
            ) if self.resource_allocation else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_usage_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent resource usage history"""
        return self.usage_history[-limit:]
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize current resource allocation"""
        if not self.resource_allocation:
            return {"status": "no_allocations_to_optimize"}
            
        # Simple optimization: try to balance load across backends
        # In a real implementation, this would be more sophisticated
        
        optimization_actions = []
        
        # Calculate current load per backend
        backend_loads = {}
        for backend_name, allocations in self.resource_allocation.items():
            total_qubits = sum(allocation['qubits'] for allocation in allocations.values())
            backend_profile = self.profiler.backend_profiles[backend_name]
            backend_loads[backend_name] = total_qubits / backend_profile.qubits
        
        # Find most and least loaded backends
        if len(backend_loads) < 2:
            return {"status": "only_one_backend_in_use"}
            
        most_loaded = max(backend_loads.items(), key=lambda x: x[1])
        least_loaded = min(backend_loads.items(), key=lambda x: x[1])
        
        # If the difference is significant, suggest rebalancing
        if most_loaded[1] - least_loaded[1] > 0.3:
            # Find a circuit to move
            for circuit_id, allocation in self.resource_allocation[most_loaded[0]].items():
                if allocation['priority'] > 2:  # Only move lower priority circuits
                    # Check if circuit can fit on least loaded backend
                    least_loaded_backend = self.profiler.backend_profiles[least_loaded[0]]
                    if allocation['qubits'] + sum(
                        self.resource_allocation[least_loaded[0]][c]['qubits']
                        for c in self.resource_allocation[least_loaded[0]].keys()
                    ) <= least_loaded_backend.qubits * 0.9:
                        
                        optimization_actions.append({
                            'action': 'reallocate',
                            'circuit_id': circuit_id,
                            'from_backend': most_loaded[0],
                            'to_backend': least_loaded[0],
                            'qubits': allocation['qubits'],
                            'reason': f'Balance load (current: {most_loaded[1]:.2f} vs {least_loaded[1]:.2f})'
                        })
                        break
        
        return {
            'status': 'optimization_suggested',
            'current_load_balance': {k: f"{v:.2f}" for k, v in backend_loads.items()},
            'suggested_actions': optimization_actions,
            'timestamp': datetime.now().isoformat()
        }

class QuantumHardwareMonitor:
    """Monitor quantum hardware performance and availability"""
    
    def __init__(self, profiler: QuantumHardwareProfiler):
        self.profiler = profiler
        self.availability_history = []
        self.performance_history = []
    
    def update_backend_availability(self, backend_name: str, available: bool,
                                    queue_depth: Optional[int] = None,
                                    estimated_wait_time_ms: Optional[float] = None) -> Dict[str, Any]:
        """
        Update availability status for a backend
        
        Args:
            backend_name: Name of the backend
            available: Availability status
            queue_depth: Current queue depth
            estimated_wait_time_ms: Estimated wait time in milliseconds
            
        Returns:
            Update result
        """
        if backend_name not in self.profiler.backend_profiles:
            return {
                'status': 'failed',
                'reason': f'Backend {backend_name} not found'
            }
        
        # Update availability
        self.availability_history.append({
            'backend': backend_name,
            'available': available,
            'queue_depth': queue_depth,
            'estimated_wait_time_ms': estimated_wait_time_ms,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'status': 'updated',
            'backend': backend_name,
            'available': available,
            'timestamp': datetime.now().isoformat()
        }
    
    def record_execution_performance(self, backend_name: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record execution performance metrics
        
        Args:
            backend_name: Name of the backend
            execution_result: Execution result from QuantumExecutionManager
            
        Returns:
            Recording result
        """
        if backend_name not in self.profiler.backend_profiles:
            return {
                'status': 'failed',
                'reason': f'Backend {backend_name} not found'
            }
        
        # Record performance
        self.performance_history.append({
            'backend': backend_name,
            'circuit_id': execution_result['circuit_id'],
            'execution_time_ms': execution_result['execution_time_ms'],
            'fidelity': execution_result['quantum_results']['fidelity'],
            'timestamp': execution_result['timestamp'],
            'status': execution_result['status']
        })
        
        return {
            'status': 'recorded',
            'backend': backend_name,
            'circuit_id': execution_result['circuit_id'],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_backend_status(self, backend_name: str) -> Dict[str, Any]:
        """Get current status of a backend"""
        if backend_name not in self.profiler.backend_profiles:
            return {
                'status': 'failed',
                'reason': f'Backend {backend_name} not found'
            }
        
        # Get recent availability
        recent_availability = [
            entry for entry in self.availability_history
            if entry['backend'] == backend_name
        ][-1] if self.availability_history else None
        
        # Get recent performance
        recent_performance = [
            entry for entry in self.performance_history
            if entry['backend'] == backend_name
        ][-5:] if self.performance_history else []  # Last 5 executions
        
        # Calculate metrics
        avg_fidelity = np.mean([p['fidelity'] for p in recent_performance]) if recent_performance else 0
        avg_time = np.mean([p['execution_time_ms'] for p in recent_performance]) if recent_performance else 0
        success_rate = sum(1 for p in recent_performance if p['status'] == 'success') / len(recent_performance) if recent_performance else 1.0
        
        return {
            'backend': backend_name,
            'profile': self.profiler.get_backend_profile(backend_name).to_dict(),
            'available': recent_availability['available'] if recent_availability else True,
            'queue_depth': recent_availability['queue_depth'] if recent_availability else 0,
            'estimated_wait_time_ms': recent_availability['estimated_wait_time_ms'] if recent_availability else 0,
            'recent_performance': {
                'avg_fidelity': avg_fidelity,
                'avg_execution_time_ms': avg_time,
                'success_rate': success_rate,
                'recent_executions': len(recent_performance)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        # Get all backend names
        backend_names = list(self.profiler.backend_profiles.keys())
        
        # Get status for each backend
        backend_statuses = {}
        for backend_name in backend_names:
            backend_statuses[backend_name] = self.get_backend_status(backend_name)
        
        # Calculate system metrics
        available_backends = sum(
            1 for status in backend_statuses.values()
            if status.get('available', False)
        )
        
        total_executions = len(self.performance_history)
        successful_executions = sum(
            1 for entry in self.performance_history
            if entry['status'] == 'success'
        )
        
        avg_fidelity = np.mean([
            entry['fidelity'] for entry in self.performance_history
            if entry['status'] == 'success'
        ]) if self.performance_history else 0
        
        avg_time = np.mean([
            entry['execution_time_ms'] for entry in self.performance_history
            if entry['status'] == 'success'
        ]) if self.performance_history else 0
        
        return {
            'total_backends': len(backend_names),
            'available_backends': available_backends,
            'backend_statuses': backend_statuses,
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 1.0,
            'avg_fidelity': avg_fidelity,
            'avg_execution_time_ms': avg_time,
            'recent_availability': [
                entry for entry in self.availability_history[-5:]
            ] if self.availability_history else [],
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a performance report for the specified period"""
        # Filter history for the time period
        cutoff_time = datetime.now().timestamp() * 1000 - days * 24 * 60 * 60 * 1000
        
        recent_performance = [
            entry for entry in self.performance_history
            if datetime.fromisoformat(entry['timestamp']).timestamp() * 1000 >= cutoff_time
        ]
        
        recent_availability = [
            entry for entry in self.availability_history
            if datetime.fromisoformat(entry['timestamp']).timestamp() * 1000 >= cutoff_time
        ]
        
        # Calculate metrics
        if not recent_performance:
            return {"status": "no_recent_data"}
        
        # Group by backend
        backend_metrics = {}
        for entry in recent_performance:
            backend = entry['backend']
            if backend not in backend_metrics:
                backend_metrics[backend] = {
                    'executions': 0,
                    'successful': 0,
                    'total_fidelity': 0.0,
                    'total_time': 0.0
                }
            
            backend_metrics[backend]['executions'] += 1
            if entry['status'] == 'success':
                backend_metrics[backend]['successful'] += 1
                backend_metrics[backend]['total_fidelity'] += entry['fidelity']
                backend_metrics[backend]['total_time'] += entry['execution_time_ms']
        
        # Calculate per-backend metrics
        backend_results = {}
        for backend, metrics in backend_metrics.items():
            backend_results[backend] = {
                'executions': metrics['executions'],
                'success_rate': metrics['successful'] / metrics['executions'] if metrics['executions'] > 0 else 0,
                'avg_fidelity': metrics['total_fidelity'] / metrics['successful'] if metrics['successful'] > 0 else 0,
                'avg_time_ms': metrics['total_time'] / metrics['successful'] if metrics['successful'] > 0 else 0
            }
        
        # Calculate availability metrics
        availability_by_backend = {}
        for entry in recent_availability:
            backend = entry['backend']
            if backend not in availability_by_backend:
                availability_by_backend[backend] = {
                    'total': 0,
                    'available': 0
                }
            availability_by_backend[backend]['total'] += 1
            if entry['available']:
                availability_by_backend[backend]['available'] += 1
        
        availability_results = {}
        for backend, metrics in availability_by_backend.items():
            availability_results[backend] = {
                'availability_rate': metrics['available'] / metrics['total'] if metrics['total'] > 0 else 1.0,
                'total_checks': metrics['total']
            }
        
        return {
            'period_days': days,
            'start_date': datetime.fromtimestamp(cutoff_time / 1000).isoformat(),
            'end_date': datetime.now().isoformat(),
            'total_executions': len(recent_performance),
            'successful_executions': sum(1 for e in recent_performance if e['status'] == 'success'),
            'overall_success_rate': sum(1 for e in recent_performance if e['status'] == 'success') / len(recent_performance) if recent_performance else 0,
            'overall_avg_fidelity': np.mean([e['fidelity'] for e in recent_performance if e['status'] == 'success']) if recent_performance else 0,
            'overall_avg_time_ms': np.mean([e['execution_time_ms'] for e in recent_performance if e['status'] == 'success']) if recent_performance else 0,
            'by_backend': {
                'performance': backend_results,
                'availability': availability_results
            },
            'timestamp': datetime.now().isoformat()
        }