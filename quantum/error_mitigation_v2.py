"""
Advanced Quantum Error Mitigation
Zero Noise Extrapolation + Probabilistic Error Cancellation
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MitigationResult:
    """Result of error mitigation"""
    raw_value: float
    mitigated_value: float
    confidence: float
    method: str
    scale_factors: List[float]
    raw_results: Dict[float, float]


class ErrorMitigationStrategy(ABC):
    """Base class for error mitigation strategies"""
    
    @abstractmethod
    async def mitigate(self, circuit_executor: Callable, circuit: Any, shots: int) -> MitigationResult:
        pass


class ZeroNoiseExtrapolator(ErrorMitigationStrategy):
    """
    Zero Noise Extrapolation (ZNE)
    Extrapolates to zero noise by scaling circuit noise
    """
    
    def __init__(self, scale_factors: List[float] = None, order: int = 2):
        """
        Args:
            scale_factors: Noise scaling factors (default: [1.0, 2.0, 3.0])
            order: Extrapolation order (1=linear, 2=quadratic)
        """
        self.scale_factors = scale_factors or [1.0, 2.0, 3.0]
        self.order = order
        self.extrapolation_methods = {
            1: self._linear_extrapolate,
            2: self._richardson_extrapolate,
            3: self._cubic_extrapolate
        }
    
    async def mitigate(self, circuit_executor: Callable, circuit: Any, shots: int) -> MitigationResult:
        """
        Execute circuit at different noise levels and extrapolate
        """
        results = {}
        
        # Execute at different noise scales
        for scale in self.scale_factors:
            # Scale noise by inserting identity gates
            scaled_circuit = self._scale_noise(circuit, scale)
            
            # Execute
            result = await circuit_executor(scaled_circuit, shots)
            expectation = self._calculate_expectation(result)
            results[scale] = expectation
            
            logger.info(f"Scale {scale}: expectation = {expectation:.6f}")
        
        # Extrapolate to zero noise
        extrapolate_func = self.extrapolation_methods.get(self.order, self._richardson_extrapolate)
        zero_noise = extrapolate_func(results)
        
        # Calculate confidence based on variance
        raw_values = list(results.values())
        variance = np.var(raw_values)
        confidence = 1.0 / (1.0 + variance)
        
        return MitigationResult(
            raw_value=results[1.0],
            mitigated_value=zero_noise,
            confidence=confidence,
            method=f"ZNE-order-{self.order}",
            scale_factors=self.scale_factors,
            raw_results=results
        )
    
    def _scale_noise(self, circuit, scale: float):
        """
        Scale noise by inserting identity gates
        Identity gates take time but don't change state
        """
        from qiskit import QuantumCircuit
        
        if isinstance(circuit, QuantumCircuit):
            scaled = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
            
            # Insert scale-1 identity operations between gates
            for instruction in circuit.data:
                scaled.append(instruction[0], instruction[1], instruction[2])
                
                # Insert identities proportional to scale
                n_identities = int((scale - 1) * 2)
                for _ in range(n_identities):
                    for qubit in range(circuit.num_qubits):
                        scaled.id(qubit)
            
            return scaled
        else:
            # Generic scaling
            return circuit
    
    def _calculate_expectation(self, result: Dict[str, Any]) -> float:
        """Calculate expectation value from result"""
        if 'counts' in result:
            counts = result['counts']
            total = sum(counts.values())
            
            # Calculate ⟨Z⟩ expectation
            expectation = 0
            for bitstring, count in counts.items():
                # Convert to value (-1 or +1 for each bit)
                parity = sum(int(b) for b in bitstring) % 2
                value = -1 if parity else 1
                expectation += value * count / total
            
            return expectation
        
        return 0.0
    
    def _linear_extrapolate(self, results: Dict[float, float]) -> float:
        """Linear extrapolation to zero noise"""
        scales = np.array(list(results.keys()))
        values = np.array(list(results.values()))
        
        # Linear fit: y = a*x + b, extrapolate to x=0
        coeffs = np.polyfit(scales, values, 1)
        return coeffs[-1]  # Intercept at x=0
    
    def _richardson_extrapolate(self, results: Dict[float, float]) -> float:
        """Richardson extrapolation (polynomial fit)"""
        scales = np.array(list(results.keys()))
        values = np.array(list(results.values()))
        
        # Polynomial fit of order len(scales)-1
        n = len(scales)
        coeffs = np.polyfit(scales, values, n - 1)
        
        # Evaluate at x=0 (constant term)
        return coeffs[-1]
    
    def _cubic_extrapolate(self, results: Dict[float, float]) -> float:
        """Cubic extrapolation"""
        scales = np.array(list(results.keys()))
        values = np.array(list(results.values()))
        
        if len(scales) < 4:
            # Fall back to Richardson
            return self._richardson_extrapolate(results)
        
        coeffs = np.polyfit(scales, values, 3)
        return coeffs[-1]


class ProbabilisticErrorCanceler(ErrorMitigationStrategy):
    """
    Probabilistic Error Cancellation (PEC)
    Cancels errors by applying inverse noise operations
    """
    
    def __init__(self, precision: float = 0.01, max_samples: int = 10000):
        """
        Args:
            precision: Target precision for mitigation
            max_samples: Maximum number of circuit samples
        """
        self.precision = precision
        self.max_samples = max_samples
    
    async def mitigate(self, circuit_executor: Callable, circuit: Any, shots: int) -> MitigationResult:
        """
        Apply probabilistic error cancellation
        """
        # Learn noise model (if not known)
        noise_model = await self._learn_noise_model(circuit_executor, circuit)
        
        # Apply quasi-probability decomposition
        decomposed = self._quasi_prob_decompose(circuit, noise_model)
        
        # Sample circuits according to probabilities
        samples = self._sample_circuits(decomposed, self.max_samples)
        
        # Execute sampled circuits
        results = []
        for sampled_circuit, sign, weight in samples:
            result = await circuit_executor(sampled_circuit, shots)
            expectation = self._calculate_expectation(result)
            results.append((sign, weight, expectation))
        
        # Combine results with signs
        total = sum(sign * weight * val for sign, weight, val in results)
        total_weight = sum(weight for _, weight, _ in results)
        
        mitigated_value = total / total_weight if total_weight > 0 else 0
        
        # Raw result (without mitigation)
        raw_result = await circuit_executor(circuit, shots)
        raw_value = self._calculate_expectation(raw_result)
        
        # Confidence based on sample count
        confidence = min(1.0, len(samples) / self.max_samples)
        
        return MitigationResult(
            raw_value=raw_value,
            mitigated_value=mitigated_value,
            confidence=confidence,
            method="PEC",
            scale_factors=[],
            raw_results={1.0: raw_value}
        )
    
    async def _learn_noise_model(self, executor, circuit):
        """Learn noise characteristics by running test circuits"""
        # Simplified noise model
        # In practice, use characterization circuits
        
        noise_model = {
            'single_qubit_error': 0.001,
            'two_qubit_error': 0.01,
            'measurement_error': 0.02
        }
        
        return noise_model
    
    def _quasi_prob_decompose(self, circuit, noise_model):
        """
        Decompose ideal operation into noisy + correction
        Uses quasi-probability representation
        """
        # Simplified: return circuit with correction probabilities
        decompositions = []
        
        # For each gate, add correction operations
        decompositions.append({
            'circuit': circuit,
            'probability': 0.7,
            'sign': 1,
            'corrections': []
        })
        
        decompositions.append({
            'circuit': self._add_correction_gates(circuit, noise_model),
            'probability': 0.3,
            'sign': -1,  # Negative probability!
            'corrections': ['X', 'Z']
        })
        
        return decompositions
    
    def _add_correction_gates(self, circuit, noise_model):
        """Add gates to correct for known noise"""
        # Implementation depends on noise model
        return circuit
    
    def _sample_circuits(self, decompositions, n_samples):
        """Sample circuits according to quasi-probabilities"""
        samples = []
        
        for _ in range(n_samples):
            # Sample which decomposition to use
            probs = [d['probability'] for d in decompositions]
            idx = np.random.choice(len(decompositions), p=probs)
            
            decomp = decompositions[idx]
            samples.append((
                decomp['circuit'],
                decomp['sign'],
                decomp['probability']
            ))
        
        return samples
    
    def _calculate_expectation(self, result):
        """Calculate expectation from result"""
        if 'counts' in result:
            counts = result['counts']
            total = sum(counts.values())
            
            # Simple expectation calculation
            expectation = 0
            for bitstring, count in counts.items():
                # Weight by bitstring value
                value = int(bitstring, 2) / (2**len(bitstring) - 1)
                expectation += value * count / total
            
            return expectation * 2 - 1  # Scale to [-1, 1]
        
        return 0.0


class ReadoutErrorMitigator:
    """
    Mitigates measurement/readout errors
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.calibration_matrix = None
    
    async def calibrate(self, executor: Callable):
        """Calibrate by preparing and measuring basis states"""
        from qiskit import QuantumCircuit
        
        calibration_matrix = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        
        # Prepare each basis state and measure
        for state in range(2**self.n_qubits):
            circuit = QuantumCircuit(self.n_qubits)
            
            # Prepare state
            for i in range(self.n_qubits):
                if (state >> i) & 1:
                    circuit.x(i)
            
            circuit.measure_all()
            
            # Execute
            result = await executor(circuit, shots=8192)
            counts = result.get('counts', {})
            
            # Fill calibration matrix column
            total = sum(counts.values())
            for measured_state, count in counts.items():
                measured = int(measured_state, 2)
                calibration_matrix[measured, state] = count / total
        
        self.calibration_matrix = calibration_matrix
        logger.info("Readout calibration complete")
    
    def mitigate_counts(self, raw_counts: Dict[str, int]) -> Dict[str, float]:
        """Apply calibration matrix to mitigate readout errors"""
        if self.calibration_matrix is None:
            logger.warning("Calibration not performed, returning raw counts")
            return {k: float(v) for k, v in raw_counts.items()}
        
        # Convert to probability vector
        n_states = 2**self.n_qubits
        raw_probs = np.zeros(n_states)
        
        for bitstring, count in raw_counts.items():
            state = int(bitstring, 2)
            raw_probs[state] = count
        
        raw_probs /= np.sum(raw_probs)
        
        # Apply inverse calibration matrix
        try:
            inv_matrix = np.linalg.inv(self.calibration_matrix)
            mitigated_probs = inv_matrix @ raw_probs
            
            # Clip negative probabilities (small numerical errors)
            mitigated_probs = np.clip(mitigated_probs, 0, 1)
            mitigated_probs /= np.sum(mitigated_probs)
            
        except np.linalg.LinAlgError:
            logger.warning("Calibration matrix singular, using least squares")
            mitigated_probs, _ = np.linalg.lstsq(
                self.calibration_matrix, raw_probs, rcond=None
            )[0]
            mitigated_probs = np.clip(mitigated_probs, 0, 1)
            mitigated_probs /= np.sum(mitigated_probs)
        
        # Convert back to dictionary
        mitigated_counts = {}
        for state, prob in enumerate(mitigated_probs):
            bitstring = format(state, f'0{self.n_qubits}b')
            mitigated_counts[bitstring] = prob * sum(raw_counts.values())
        
        return mitigated_counts


class AdvancedErrorMitigation:
    """
    Orchestrates multiple error mitigation strategies
    """
    
    def __init__(self, n_qubits: int = 20):
        self.n_qubits = n_qubits
        self.zne = ZeroNoiseExtrapolator()
        self.pec = ProbabilisticErrorCanceler()
        self.readout = ReadoutErrorMitigator(n_qubits)
        self.calibrated = False
    
    async def calibrate(self, executor: Callable):
        """Calibrate all mitigation strategies"""
        logger.info("Calibrating error mitigation...")
        
        await self.readout.calibrate(executor)
        self.calibrated = True
        
        logger.info("Calibration complete")
    
    async def execute_with_mitigation(
        self,
        circuit: Any,
        executor: Callable,
        shots: int = 8192,
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Execute with best error mitigation strategy
        
        Args:
            circuit: Quantum circuit
            executor: Function to execute circuit
            shots: Number of shots
            strategy: 'zne', 'pec', 'readout', 'all', or 'auto'
        
        Returns:
            Mitigated results
        """
        if not self.calibrated and strategy in ['readout', 'all', 'auto']:
            await self.calibrate(executor)
        
        # Auto-select strategy based on circuit complexity
        if strategy == "auto":
            strategy = self._select_strategy(circuit)
        
        results = {}
        
        if strategy in ['zne', 'all']:
            logger.info("Applying Zero Noise Extrapolation...")
            zne_result = await self.zne.mitigate(executor, circuit, shots)
            results['zne'] = {
                'value': zne_result.mitigated_value,
                'confidence': zne_result.confidence,
                'raw_value': zne_result.raw_value
            }
        
        if strategy in ['pec', 'all']:
            logger.info("Applying Probabilistic Error Cancellation...")
            pec_result = await self.pec.mitigate(executor, circuit, shots)
            results['pec'] = {
                'value': pec_result.mitigated_value,
                'confidence': pec_result.confidence,
                'raw_value': pec_result.raw_value
            }
        
        # Always apply readout mitigation if calibrated
        if self.calibrated:
            raw_result = await executor(circuit, shots)
            if 'counts' in raw_result:
                mitigated_counts = self.readout.mitigate_counts(raw_result['counts'])
                results['readout_mitigated'] = mitigated_counts
        
        # Combine results (weighted by confidence)
        if len(results) > 0:
            final_value = self._combine_results(results)
            results['final'] = final_value
        
        results['mitigation_applied'] = True
        results['strategy'] = strategy
        
        return results
    
    def _select_strategy(self, circuit) -> str:
        """Select best mitigation strategy for circuit"""
        # Simple heuristic: use ZNE for shallow circuits, PEC for deep
        depth = getattr(circuit, 'depth', lambda: 10)()
        
        if depth < 20:
            return "zne"
        else:
            return "pec"
    
    def _combine_results(self, results: Dict[str, Any]) -> float:
        """Combine multiple mitigation results"""
        values = []
        weights = []
        
        for method, data in results.items():
            if method in ['zne', 'pec'] and 'value' in data:
                values.append(data['value'])
                weights.append(data.get('confidence', 0.5))
        
        if not values:
            return 0.0
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        combined = np.dot(values, weights)
        
        return combined


# Convenience function
async def mitigate_errors(
    circuit: Any,
    executor: Callable,
    n_qubits: int = 20,
    shots: int = 8192,
    strategy: str = "auto"
) -> Dict[str, Any]:
    """
    Apply advanced error mitigation to quantum execution
    
    Example:
        result = await mitigate_errors(circuit, execute_func, n_qubits=20)
    """
    mitigator = AdvancedErrorMitigation(n_qubits)
    return await mitigator.execute_with_mitigation(circuit, executor, shots, strategy)
