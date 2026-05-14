"""
Quantum Hardware Selector

This module provides intelligent selection of quantum hardware backends
based on circuit requirements, performance metrics, and availability.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import quantum hardware profiler components
from quantum.advanced.quantum_hardware_profiler import (
    QuantumHardwareProfiler,
    QuantumBackendProfile,
    QuantumCircuitProfile,
    QuantumHardwareType,
    HardwareOptimizationResult,
    QuantumHardwareSelector as BaseHardwareSelector,
    QuantumExecutionManager,
    QuantumResourceManager,
    QuantumHardwareMonitor
)

class QuantumTaskType(Enum):
    """Types of quantum tasks"""
    PORTFOLIO_OPTIMIZATION = auto()
    RISK_ANALYSIS = auto()
    STRATEGY_OPTIMIZATION = auto()
    REGIME_DETECTION = auto()
    FEATURE_EXTRACTION = auto()
    CIRCUIT_OPTIMIZATION = auto()
    GENERAL_COMPUTATION = auto()

@dataclass
class QuantumTaskRequirements:
    """Requirements for a quantum task"""
    task_type: QuantumTaskType
    num_qubits: int
    circuit_depth: int
    gate_count: int
    required_fidelity: float
    max_execution_time_ms: float
    hardware_preferences: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_type': self.task_type.name,
            'num_qubits': self.num_qubits,
            'circuit_depth': self.circuit_depth,
            'gate_count': self.gate_count,
            'required_fidelity': self.required_fidelity,
            'max_execution_time_ms': self.max_execution_time_ms,
            'hardware_preferences': self.hardware_preferences or []
        }

@dataclass
class QuantumBackendSelection:
    """Result of quantum backend selection"""
    task_requirements: QuantumTaskRequirements
    selected_backend: str
    backend_profile: QuantumBackendProfile
    optimization_result: Optional[HardwareOptimizationResult] = None
    selection_reasoning: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_requirements': self.task_requirements.to_dict(),
            'selected_backend': self.selected_backend,
            'backend_profile': self.backend_profile.to_dict(),
            'optimization_result': self.optimization_result.to_dict() if self.optimization_result else None,
            'selection_reasoning': self.selection_reasoning,
            'timestamp': self.timestamp.isoformat()
        }

class AdvancedQuantumHardwareSelector:
    """Advanced quantum hardware selector with dynamic optimization"""
    
    def __init__(self, hardware_profiler: QuantumHardwareProfiler):
        self.hardware_profiler = hardware_profiler
        self.selection_history = []
        self.performance_history = []
        self.fallback_count = 0
        
        # Task type to hardware preferences mapping
        self.task_preferences = {
            QuantumTaskType.PORTFOLIO_OPTIMIZATION: ["ibm_nairobi", "ibm_lagos", "simulator_statevector"],
            QuantumTaskType.RISK_ANALYSIS: ["ibm_nairobi", "simulator_statevector", "ibm_lagos"],
            QuantumTaskType.STRATEGY_OPTIMIZATION: ["simulator_statevector", "ibm_nairobi", "ibm_lagos"],
            QuantumTaskType.REGIME_DETECTION: ["simulator_statevector", "ibm_lagos", "ibm_nairobi"],
            QuantumTaskType.FEATURE_EXTRACTION: ["simulator_statevector", "ibm_nairobi", "ibm_lagos"],
            QuantumTaskType.CIRCUIT_OPTIMIZATION: ["ibm_lagos", "ibm_nairobi", "simulator_statevector"],
            QuantumTaskType.GENERAL_COMPUTATION: ["simulator_statevector", "ibm_lagos", "ibm_nairobi"]
        }
        
        logger.info("Advanced Quantum Hardware Selector initialized")
    
    def select_backend(self, task_requirements: QuantumTaskRequirements,
                      optimization_goal: str = 'balanced') -> QuantumBackendSelection:
        """
        Select the best quantum backend for a given task
        
        Args:
            task_requirements: Task requirements
            optimization_goal: Optimization goal ('speed', 'fidelity', 'balanced')
            
        Returns:
            Quantum backend selection
        """
        logger.info(f"Selecting backend for task: {task_requirements.task_type.name}")
        logger.info(f"Requirements: {task_requirements.to_dict()}")
        
        # Get all available backends
        available_backends = list(self.hardware_profiler.backend_profiles.keys())
        
        if not available_backends:
            raise ValueError("No quantum backends available")
        
        # Filter backends that meet minimum requirements
        suitable_backends = []
        for backend_name in available_backends:
            backend = self.hardware_profiler.get_backend_profile(backend_name)
            
            # Check if backend meets requirements
            if (task_requirements.num_qubits <= backend.qubits and
                task_requirements.circuit_depth <= backend.max_circuit_depth and
                task_requirements.required_fidelity <= self._estimate_achievable_fidelity(backend, task_requirements)):
                suitable_backends.append(backend_name)
        
        if not suitable_backends:
            logger.warning("No backends meet minimum requirements, falling back to simulator")
            suitable_backends = ["simulator_statevector"]
        
        # Apply task-specific preferences
        preferred_backends = self._apply_task_preferences(task_requirements, suitable_backends)
        
        # Profile each suitable backend
        backend_profiles = {}
        for backend_name in preferred_backends:
            backend = self.hardware_profiler.get_backend_profile(backend_name)
            
            # Create circuit profile for this backend
            circuit_profile = QuantumCircuitProfile(
                circuit_id=f"selection_{task_requirements.task_type.name}_{backend_name}",
                num_qubits=task_requirements.num_qubits,
                depth=task_requirements.circuit_depth,
                gate_count=task_requirements.gate_count,
                gate_types={"rx": task_requirements.gate_count // 3, 
                          "ry": task_requirements.gate_count // 3, 
                          "cx": task_requirements.gate_count // 3},
                connectivity=[(i, i+1) for i in range(task_requirements.num_qubits-1)]
            )
            
            # Profile backend
            profile_result = self.hardware_profiler.profile_backend(backend_name, circuit_profile)
            backend_profiles[backend_name] = {
                'backend': backend,
                'profile_result': profile_result,
                'circuit_profile': circuit_profile
            }
        
        # Select best backend based on optimization goal
        if optimization_goal == 'speed':
            best_backend = self._select_fastest_backend(backend_profiles)
        elif optimization_goal == 'fidelity':
            best_backend = self._select_highest_fidelity_backend(backend_profiles)
        else:  # balanced
            best_backend = self._select_balanced_backend(backend_profiles)
        
        # Optimize circuit for selected backend
        optimization_result = None
        if best_backend:
            optimization_result = self.hardware_profiler.optimize_for_backend(
                best_backend,
                backend_profiles[best_backend]['circuit_profile']
            )
        
        # Create selection result
        selection = QuantumBackendSelection(
            task_requirements=task_requirements,
            selected_backend=best_backend,
            backend_profile=self.hardware_profiler.get_backend_profile(best_backend),
            optimization_result=optimization_result,
            selection_reasoning=self._generate_selection_reasoning(
                best_backend, backend_profiles, optimization_goal
            )
        )
        
        # Record selection
        self.selection_history.append(selection)
        
        logger.info(f"Selected backend: {best_backend}")
        logger.info(f"Selection reasoning: {selection.selection_reasoning}")
        
        return selection
    
    def _estimate_achievable_fidelity(self, backend: QuantumBackendProfile,
                                     requirements: QuantumTaskRequirements) -> float:
        """Estimate achievable fidelity for a given backend and task"""
        # Base fidelity based on backend error rates
        base_fidelity = 1.0 - backend.gate_error_rate - backend.readout_error_rate
        
        # Adjust for circuit complexity
        complexity_penalty = 1.0 - (requirements.circuit_depth / backend.max_circuit_depth) * 0.1
        qubit_penalty = 1.0 - (requirements.num_qubits / backend.qubits) * 0.1
        
        # Calculate estimated fidelity
        estimated_fidelity = base_fidelity * complexity_penalty * qubit_penalty
        
        return max(0.0, min(1.0, estimated_fidelity))
    
    def _apply_task_preferences(self, task_requirements: QuantumTaskRequirements,
                               suitable_backends: List[str]) -> List[str]:
        """Apply task-specific backend preferences"""
        # Get preferences for this task type
        preferences = self.task_preferences.get(task_requirements.task_type, [])
        
        # Reorder backends based on preferences
        preferred_backends = []
        remaining_backends = []
        
        for backend in suitable_backends:
            if backend in preferences:
                preferred_backends.append(backend)
            else:
                remaining_backends.append(backend)
        
        # Sort preferred backends by preference order
        preferred_backends.sort(key=lambda x: preferences.index(x) if x in preferences else len(preferences))
        
        # Combine preferred and remaining backends
        return preferred_backends + remaining_backends
    
    def _select_fastest_backend(self, backend_profiles: Dict[str, Any]) -> str:
        """Select the fastest backend"""
        # Find backend with lowest expected execution time
        best_backend = min(
            backend_profiles.items(),
            key=lambda x: x[1]['profile_result']['performance']['expected_execution_time_ms']
        )[0]
        
        return best_backend
    
    def _select_highest_fidelity_backend(self, backend_profiles: Dict[str, Any]) -> str:
        """Select the backend with highest expected fidelity"""
        # Find backend with highest expected fidelity
        best_backend = max(
            backend_profiles.items(),
            key=lambda x: x[1]['profile_result']['performance']['expected_fidelity']
        )[0]
        
        return best_backend
    
    def _select_balanced_backend(self, backend_profiles: Dict[str, Any]) -> str:
        """Select a balanced backend considering multiple factors"""
        # Calculate score for each backend
        backend_scores = {}
        
        for backend_name, profile in backend_profiles.items():
            # Get metrics
            fidelity = profile['profile_result']['performance']['expected_fidelity']
            exec_time = profile['profile_result']['performance']['expected_execution_time_ms']
            compatibility = profile['profile_result']['compatibility']['overall_compatibility']
            
            # Calculate score (weighted average)
            # Weights: fidelity 40%, speed 30%, compatibility 30%
            score = (
                0.4 * fidelity +
                0.3 * (1.0 / (exec_time + 1)) +
                0.3 * compatibility
            )
            
            backend_scores[backend_name] = score
        
        # Select backend with highest score
        best_backend = max(backend_scores.items(), key=lambda x: x[1])[0]
        
        return best_backend
    
    def _generate_selection_reasoning(self, selected_backend: str,
                                     backend_profiles: Dict[str, Any],
                                     optimization_goal: str) -> Dict[str, Any]:
        """Generate reasoning for backend selection"""
        selected_profile = backend_profiles[selected_backend]
        
        reasoning = {
            'optimization_goal': optimization_goal,
            'selected_backend': selected_backend,
            'expected_fidelity': selected_profile['profile_result']['performance']['expected_fidelity'],
            'expected_execution_time_ms': selected_profile['profile_result']['performance']['expected_execution_time_ms'],
            'compatibility': selected_profile['profile_result']['compatibility']['overall_compatibility'],
            'quantum_volume_utilization': selected_profile['profile_result']['performance']['quantum_volume_utilization'],
            'recommendations': selected_profile['profile_result']['recommendations']
        }
        
        # Add comparison with other backends
        comparisons = []
        for backend_name, profile in backend_profiles.items():
            if backend_name != selected_backend:
                comparisons.append({
                    'backend': backend_name,
                    'expected_fidelity': profile['profile_result']['performance']['expected_fidelity'],
                    'expected_execution_time_ms': profile['profile_result']['performance']['expected_execution_time_ms'],
                    'compatibility': profile['profile_result']['compatibility']['overall_compatibility']
                })
        
        reasoning['comparisons'] = comparisons
        
        return reasoning
    
    def execute_task(self, task_requirements: QuantumTaskRequirements,
                    execution_func: callable,
                    optimization_goal: str = 'balanced',
                    max_retries: int = 3) -> Dict[str, Any]:
        """
        Execute a quantum task with automatic backend selection and fallback
        
        Args:
            task_requirements: Task requirements
            execution_func: Function to execute (takes backend_name as parameter)
            optimization_goal: Optimization goal for backend selection
            max_retries: Maximum number of retries on failure
            
        Returns:
            Execution result
        """
        logger.info(f"Executing quantum task: {task_requirements.task_type.name}")
        
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # Select best backend
                selection = self.select_backend(task_requirements, optimization_goal)
                
                # Execute task on selected backend
                start_time = time.time()
                result = execution_func(selection.selected_backend)
                execution_time = (time.time() - start_time) * 1000  # ms
                
                # Record successful execution
                execution_result = {
                    'status': 'success',
                    'backend': selection.selected_backend,
                    'execution_time_ms': execution_time,
                    'task_requirements': task_requirements.to_dict(),
                    'selection_reasoning': selection.selection_reasoning,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.performance_history.append(execution_result)
                
                return execution_result
                
            except Exception as e:
                last_error = str(e)
                retries += 1
                logger.warning(f"Execution attempt {retries} failed: {e}")
                
                # If this was our last retry, fall back to simulator
                if retries >= max_retries:
                    logger.warning("Max retries reached, falling back to simulator")
                    self.fallback_count += 1
                    
                    # Update task requirements for simulator
                    simulator_requirements = QuantumTaskRequirements(
                        task_type=task_requirements.task_type,
                        num_qubits=min(task_requirements.num_qubits, 20),  # Simulator can handle more
                        circuit_depth=task_requirements.circuit_depth,
                        gate_count=task_requirements.gate_count,
                        required_fidelity=task_requirements.required_fidelity,
                        max_execution_time_ms=task_requirements.max_execution_time_ms
                    )
                    
                    # Execute on simulator
                    start_time = time.time()
                    result = execution_func("simulator_statevector")
                    execution_time = (time.time() - start_time) * 1000  # ms
                    
                    execution_result = {
                        'status': 'fallback_success',
                        'backend': "simulator_statevector",
                        'execution_time_ms': execution_time,
                        'task_requirements': simulator_requirements.to_dict(),
                        'original_error': last_error,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.performance_history.append(execution_result)
                    return execution_result
                
                # Wait before retrying
                time.sleep(0.1 * retries)  # Exponential backoff
        
        # Should never reach here
        raise RuntimeError(f"Failed to execute task after {max_retries} retries: {last_error}")
    
    def get_selection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent selection history"""
        return [selection.to_dict() for selection in self.selection_history[-limit:]]
    
    def get_performance_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance history"""
        return self.performance_history[-limit:]
    
    def get_backend_statistics(self) -> Dict[str, Any]:
        """Get statistics about backend usage"""
        if not self.selection_history:
            return {"status": "no_data"}
        
        # Count selections by backend
        backend_counts = {}
        for selection in self.selection_history:
            backend = selection.selected_backend
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        
        # Calculate success rates
        success_rates = {}
        for execution in self.performance_history:
            backend = execution['backend']
            success_rates[backend] = success_rates.get(backend, {'total': 0, 'success': 0})
            success_rates[backend]['total'] += 1
            if execution['status'] == 'success':
                success_rates[backend]['success'] += 1
        
        # Prepare statistics
        stats = {}
        for backend, count in backend_counts.items():
            stats[backend] = {
                'selections': count,
                'success_rate': success_rates.get(backend, {}).get('success', 0) / success_rates.get(backend, {}).get('total', 1) if success_rates.get(backend) else 0,
                'avg_execution_time': np.mean([
                    e['execution_time_ms'] for e in self.performance_history
                    if e['backend'] == backend and e['status'] == 'success'
                ]) if any(e['backend'] == backend and e['status'] == 'success' for e in self.performance_history) else 0
            }
        
        return {
            'total_selections': len(self.selection_history),
            'total_executions': len(self.performance_history),
            'fallback_count': self.fallback_count,
            'fallback_rate': self.fallback_count / len(self.performance_history) if self.performance_history else 0,
            'by_backend': stats,
            'most_used_backend': max(backend_counts.items(), key=lambda x: x[1])[0] if backend_counts else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_task_type_statistics(self) -> Dict[str, Any]:
        """Get statistics by task type"""
        if not self.selection_history:
            return {"status": "no_data"}
        
        # Group by task type
        by_task_type = {}
        for selection in self.selection_history:
            task_type = selection.task_requirements.task_type.name
            if task_type not in by_task_type:
                by_task_type[task_type] = {
                    'count': 0,
                    'backends': {}
                }
            by_task_type[task_type]['count'] += 1
            
            backend = selection.selected_backend
            if backend not in by_task_type[task_type]['backends']:
                by_task_type[task_type]['backends'][backend] = 0
            by_task_type[task_type]['backends'][backend] += 1
        
        # Calculate performance by task type
        task_performance = {}
        for execution in self.performance_history:
            task_type = execution['task_requirements']['task_type']
            if task_type not in task_performance:
                task_performance[task_type] = {
                    'total': 0,
                    'success': 0,
                    'execution_times': []
                }
            task_performance[task_type]['total'] += 1
            if execution['status'] == 'success':
                task_performance[task_type]['success'] += 1
                task_performance[task_type]['execution_times'].append(execution['execution_time_ms'])
        
        # Combine statistics
        final_stats = {}
        for task_type in by_task_type:
            final_stats[task_type] = {
                'selections': by_task_type[task_type]['count'],
                'backend_distribution': by_task_type[task_type]['backends'],
                'most_used_backend': max(
                    by_task_type[task_type]['backends'].items(),
                    key=lambda x: x[1]
                )[0] if by_task_type[task_type]['backends'] else None,
                'success_rate': task_performance.get(task_type, {}).get('success', 0) / task_performance.get(task_type, {}).get('total', 1) if task_performance.get(task_type) else 0,
                'avg_execution_time': np.mean(task_performance.get(task_type, {}).get('execution_times', [])) if task_performance.get(task_type) and task_performance[task_type].get('execution_times') else 0
            }
        
        return {
            'by_task_type': final_stats,
            'total_task_types': len(final_stats),
            'timestamp': datetime.now().isoformat()
        }
    
    def optimize_task_requirements(self, task_requirements: QuantumTaskRequirements) -> QuantumTaskRequirements:
        """
        Optimize task requirements for better hardware compatibility
        
        Args:
            task_requirements: Original task requirements
            
        Returns:
            Optimized task requirements
        """
        # Get all available backends
        available_backends = list(self.hardware_profiler.backend_profiles.keys())
        
        # Find backend with best compatibility
        best_compatibility = 0
        best_backend = None
        
        for backend_name in available_backends:
            backend = self.hardware_profiler.get_backend_profile(backend_name)
            
            # Calculate compatibility
            qubit_compatibility = min(1.0, task_requirements.num_qubits / backend.qubits)
            depth_compatibility = min(1.0, task_requirements.circuit_depth / backend.max_circuit_depth)
            overall_compatibility = min(qubit_compatibility, depth_compatibility)
            
            if overall_compatibility > best_compatibility:
                best_compatibility = overall_compatibility
                best_backend = backend_name
        
        if not best_backend:
            return task_requirements  # No optimization possible
        
        # Get best backend profile
        best_backend_profile = self.hardware_profiler.get_backend_profile(best_backend)
        
        # Optimize requirements
        optimized_qubits = min(
            task_requirements.num_qubits,
            int(best_backend_profile.qubits * 0.9)  # Use 90% of backend capacity
        )
        
        optimized_depth = min(
            task_requirements.circuit_depth,
            int(best_backend_profile.max_circuit_depth * 0.8)  # Use 80% of max depth
        )
        
        # Create optimized requirements
        optimized_requirements = QuantumTaskRequirements(
            task_type=task_requirements.task_type,
            num_qubits=optimized_qubits,
            circuit_depth=optimized_depth,
            gate_count=int(task_requirements.gate_count * (optimized_qubits / task_requirements.num_qubits)),
            required_fidelity=task_requirements.required_fidelity,
            max_execution_time_ms=task_requirements.max_execution_time_ms,
            hardware_preferences=task_requirements.hardware_preferences
        )
        
        return optimized_requirements
    
    def get_hardware_recommendations(self, task_requirements: QuantumTaskRequirements) -> Dict[str, Any]:
        """
        Get hardware recommendations for a given task
        
        Args:
            task_requirements: Task requirements
            
        Returns:
            Hardware recommendations
        """
        # Get all available backends
        available_backends = list(self.hardware_profiler.backend_profiles.keys())
        
        # Filter suitable backends
        suitable_backends = []
        for backend_name in available_backends:
            backend = self.hardware_profiler.get_backend_profile(backend_name)
            
            if (task_requirements.num_qubits <= backend.qubits and
                task_requirements.circuit_depth <= backend.max_circuit_depth):
                suitable_backends.append(backend_name)
        
        if not suitable_backends:
            return {
                'status': 'no_suitable_backends',
                'task_requirements': task_requirements.to_dict()
            }
        
        # Profile each suitable backend
        backend_profiles = {}
        for backend_name in suitable_backends:
            backend = self.hardware_profiler.get_backend_profile(backend_name)
            
            # Create circuit profile
            circuit_profile = QuantumCircuitProfile(
                circuit_id=f"recommendation_{task_requirements.task_type.name}_{backend_name}",
                num_qubits=task_requirements.num_qubits,
                depth=task_requirements.circuit_depth,
                gate_count=task_requirements.gate_count,
                gate_types={"rx": task_requirements.gate_count // 3, 
                          "ry": task_requirements.gate_count // 3, 
                          "cx": task_requirements.gate_count // 3},
                connectivity=[(i, i+1) for i in range(task_requirements.num_qubits-1)]
            )
            
            # Profile backend
            profile_result = self.hardware_profiler.profile_backend(backend_name, circuit_profile)
            backend_profiles[backend_name] = {
                'backend': backend,
                'profile_result': profile_result,
                'circuit_profile': circuit_profile
            }
        
        # Generate recommendations
        recommendations = []
        
        for backend_name, profile in backend_profiles.items():
            recommendations.append({
                'backend': backend_name,
                'quantum_volume': profile['backend'].calculate_quantum_volume(),
                'expected_fidelity': profile['profile_result']['performance']['expected_fidelity'],
                'expected_execution_time_ms': profile['profile_result']['performance']['expected_execution_time_ms'],
                'compatibility': profile['profile_result']['compatibility']['overall_compatibility'],
                'recommendations': profile['profile_result']['recommendations']
            })
        
        # Sort recommendations by suitability
        recommendations.sort(
            key=lambda x: (
                -x['compatibility'] * 0.4 -
                x['expected_fidelity'] * 0.3 -
                (1.0 / (x['expected_execution_time_ms'] + 1)) * 0.3
            )
        )
        
        return {
            'status': 'success',
            'task_requirements': task_requirements.to_dict(),
            'recommendations': recommendations,
            'best_recommendation': recommendations[0] if recommendations else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_execution_plan(self, task_requirements: QuantumTaskRequirements,
                          optimization_goal: str = 'balanced') -> Dict[str, Any]:
        """
        Generate a complete execution plan for a quantum task
        
        Args:
            task_requirements: Task requirements
            optimization_goal: Optimization goal
            
        Returns:
            Execution plan
        """
        # Optimize task requirements
        optimized_requirements = self.optimize_task_requirements(task_requirements)
        
        # Select backend
        selection = self.select_backend(optimized_requirements, optimization_goal)
        
        # Generate recommendations
        recommendations = self.get_hardware_recommendations(optimized_requirements)
        
        # Create execution plan
        plan = {
            'original_requirements': task_requirements.to_dict(),
            'optimized_requirements': optimized_requirements.to_dict(),
            'backend_selection': selection.to_dict(),
            'hardware_recommendations': recommendations,
            'optimization_goal': optimization_goal,
            'timestamp': datetime.now().isoformat()
        }
        
        return plan
    
    def execute_with_plan(self, execution_plan: Dict[str, Any],
                          execution_func: callable,
                          max_retries: int = 3) -> Dict[str, Any]:
        """
        Execute a quantum task using a pre-generated execution plan
        
        Args:
            execution_plan: Execution plan from get_execution_plan
            execution_func: Function to execute
            max_retries: Maximum number of retries
            
        Returns:
            Execution result
        """
        # Extract requirements from plan
        optimized_requirements = QuantumTaskRequirements(
            task_type=QuantumTaskType[execution_plan['optimized_requirements']['task_type']],
            num_qubits=execution_plan['optimized_requirements']['num_qubits'],
            circuit_depth=execution_plan['optimized_requirements']['circuit_depth'],
            gate_count=execution_plan['optimized_requirements']['gate_count'],
            required_fidelity=execution_plan['optimized_requirements']['required_fidelity'],
            max_execution_time_ms=execution_plan['optimized_requirements']['max_execution_time_ms']
        )
        
        # Execute task
        return self.execute_task(
            task_requirements=optimized_requirements,
            execution_func=execution_func,
            optimization_goal=execution_plan['optimization_goal'],
            max_retries=max_retries
        )

class QuantumTaskExecutor:
    """Executes quantum tasks with automatic hardware selection and optimization"""
    
    def __init__(self, hardware_selector: AdvancedQuantumHardwareSelector):
        self.hardware_selector = hardware_selector
        self.execution_history = []
    
    def execute_circuit_optimization(self, circuit_profile: QuantumCircuitProfile,
                                     optimization_goal: str = 'balanced') -> Dict[str, Any]:
        """
        Execute quantum circuit optimization with automatic hardware selection
        
        Args:
            circuit_profile: Circuit profile to optimize
            optimization_goal: Optimization goal
            
        Returns:
            Execution result
        """
        # Create task requirements
        task_requirements = QuantumTaskRequirements(
            task_type=QuantumTaskType.CIRCUIT_OPTIMIZATION,
            num_qubits=circuit_profile.num_qubits,
            circuit_depth=circuit_profile.depth,
            gate_count=circuit_profile.gate_count,
            required_fidelity=0.95,  # High fidelity requirement
            max_execution_time_ms=5000  # 5 seconds max
        )
        
        # Define execution function
        def execution_func(backend_name: str) -> Dict[str, Any]:
            # Get hardware profiler from selector
            hardware_profiler = self.hardware_selector.hardware_profiler
            
            # Optimize circuit
            optimization_result = hardware_profiler.optimize_for_backend(backend_name, circuit_profile)
            
            return {
                'optimization_result': optimization_result.to_dict(),
                'backend': backend_name
            }
        
        # Execute with hardware selection
        return self.hardware_selector.execute_task(
            task_requirements=task_requirements,
            execution_func=execution_func,
            optimization_goal=optimization_goal
        )
    
    def execute_quantum_neural_network(self, architecture: QNNArchitecture,
                                       training_data: Tuple[np.ndarray, np.ndarray],
                                       epochs: int = 10,
                                       optimization_goal: str = 'balanced') -> Dict[str, Any]:
        """
        Execute quantum neural network training with automatic hardware selection
        
        Args:
            architecture: QNN architecture
            training_data: Training data (X, y)
            epochs: Number of training epochs
            optimization_goal: Optimization goal
            
        Returns:
            Execution result
        """
        # Estimate task requirements
        num_qubits = max(layer.num_qubits for layer in architecture.layers)
        circuit_depth = epochs * 10  # Rough estimate
        gate_count = sum(layer.num_parameters for layer in architecture.layers) * epochs
        
        task_requirements = QuantumTaskRequirements(
            task_type=QuantumTaskType.GENERAL_COMPUTATION,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            gate_count=gate_count,
            required_fidelity=0.90,  # Moderate fidelity requirement
            max_execution_time_ms=10000  # 10 seconds max
        )
        
        # Define execution function
        def execution_func(backend_name: str) -> Dict[str, Any]:
            # Create and train QNN
            qnn = QuantumNeuralNetwork(
                architecture=architecture,
                training_mode=QuantumTrainingMode.ADAPTIVE,
                hardware_backend=backend_name
            )
            
            # Train
            training_history = qnn.train(training_data[0], training_data[1], epochs=epochs, learning_rate=0.01)
            
            return {
                'qnn_model': qnn.to_dict(),
                'training_history': training_history,
                'backend': backend_name
            }
        
        # Execute with hardware selection
        return self.hardware_selector.execute_task(
            task_requirements=task_requirements,
            execution_func=execution_func,
            optimization_goal=optimization_goal
        )
    
    def execute_quantum_regime_detection(self, training_data: List[Tuple[MarketDataFeatures, MarketRegime]],
                                          test_features: MarketDataFeatures,
                                          optimization_goal: str = 'balanced') -> Dict[str, Any]:
        """
        Execute quantum regime detection with automatic hardware selection
        
        Args:
            training_data: Training data
            test_features: Features to detect regime for
            optimization_goal: Optimization goal
            
        Returns:
            Execution result
        """
        # Estimate task requirements
        num_qubits = 4  # Fixed for regime detection
        circuit_depth = len(training_data)  # Rough estimate
        gate_count = len(training_data) * 10  # Rough estimate
        
        task_requirements = QuantumTaskRequirements(
            task_type=QuantumTaskType.REGIME_DETECTION,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            gate_count=gate_count,
            required_fidelity=0.85,  # Moderate fidelity requirement
            max_execution_time_ms=2000  # 2 seconds max
        )
        
        # Define execution function
        def execution_func(backend_name: str) -> Dict[str, Any]:
            # Create and train detector
            detector = QuantumRegimeDetector(num_qubits=num_qubits, hardware_backend=backend_name)
            
            # Train
            train_result = detector.train_detector(training_data, epochs=20, learning_rate=0.01)
            
            # Detect regime
            detection = detector.detect_regime(test_features)
            
            return {
                'train_result': train_result,
                'detection': detection.to_dict(),
                'backend': backend_name
            }
        
        # Execute with hardware selection
        return self.hardware_selector.execute_task(
            task_requirements=task_requirements,
            execution_func=execution_func,
            optimization_goal=optimization_goal
        )
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about quantum task executions"""
        if not self.execution_history:
            return {"status": "no_data"}
        
        # Count by task type
        task_type_counts = {}
        for execution in self.execution_history:
            task_type = execution['task_requirements']['task_type']
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        # Count by backend
        backend_counts = {}
        for execution in self.execution_history:
            backend = execution['backend']
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        
        # Calculate success rates
        success_rates = {}
        for execution in self.execution_history:
            backend = execution['backend']
            success_rates[backend] = success_rates.get(backend, {'total': 0, 'success': 0})
            success_rates[backend]['total'] += 1
            if execution['status'] == 'success':
                success_rates[backend]['success'] += 1
        
        # Calculate execution time statistics
        execution_times = {}
        for execution in self.execution_history:
            if execution['status'] == 'success':
                backend = execution['backend']
                if backend not in execution_times:
                    execution_times[backend] = []
                execution_times[backend].append(execution['execution_time_ms'])
        
        # Prepare statistics
        backend_stats = {}
        for backend, count in backend_counts.items():
            backend_stats[backend] = {
                'executions': count,
                'success_rate': success_rates.get(backend, {}).get('success', 0) / success_rates.get(backend, {}).get('total', 1) if success_rates.get(backend) else 0,
                'avg_execution_time': np.mean(execution_times.get(backend, [])) if execution_times.get(backend) else 0,
                'min_execution_time': min(execution_times.get(backend, [])) if execution_times.get(backend) else 0,
                'max_execution_time': max(execution_times.get(backend, [])) if execution_times.get(backend) else 0
            }
        
        return {
            'total_executions': len(self.execution_history),
            'by_task_type': task_type_counts,
            'by_backend': backend_stats,
            'timestamp': datetime.now().isoformat()
        }

class QuantumHardwareOrchestrator:
    """Orchestrates quantum hardware selection and execution"""
    
    def __init__(self):
        # Initialize components
        self.hardware_profiler = QuantumHardwareProfiler()
        self.hardware_selector = AdvancedQuantumHardwareSelector(self.hardware_profiler)
        self.task_executor = QuantumTaskExecutor(self.hardware_selector)
        
        # Initialize monitoring
        self.hardware_monitor = QuantumHardwareMonitor(self.hardware_profiler)
        
        logger.info("Quantum Hardware Orchestrator initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'hardware_profiler': {
                'backends': list(self.hardware_profiler.backend_profiles.keys()),
                'total_backends': len(self.hardware_profiler.backend_profiles)
            },
            'hardware_selector': {
                'selection_history': len(self.hardware_selector.selection_history),
                'performance_history': len(self.hardware_selector.performance_history),
                'fallback_count': self.hardware_selector.fallback_count
            },
            'task_executor': {
                'execution_history': len(self.task_executor.execution_history)
            },
            'hardware_monitor': {
                'availability_history': len(self.hardware_monitor.availability_history),
                'performance_history': len(self.hardware_monitor.performance_history)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_backend_recommendations(self, task_type: QuantumTaskType,
                                    num_qubits: int = 5,
                                    circuit_depth: int = 50,
                                    gate_count: int = 200) -> Dict[str, Any]:
        """
        Get hardware recommendations for a specific task type
        
        Args:
            task_type: Type of quantum task
            num_qubits: Number of qubits required
            circuit_depth: Circuit depth required
            gate_count: Number of gates required
            
        Returns:
            Hardware recommendations
        """
        # Create task requirements
        task_requirements = QuantumTaskRequirements(
            task_type=task_type,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            gate_count=gate_count,
            required_fidelity=0.90,
            max_execution_time_ms=5000
        )
        
        # Get recommendations
        return self.hardware_selector.get_hardware_recommendations(task_requirements)
    
    def create_execution_plan(self, task_type: QuantumTaskType,
                              num_qubits: int = 5,
                              circuit_depth: int = 50,
                              gate_count: int = 200,
                              optimization_goal: str = 'balanced') -> Dict[str, Any]:
        """
        Create an execution plan for a quantum task
        
        Args:
            task_type: Type of quantum task
            num_qubits: Number of qubits required
            circuit_depth: Circuit depth required
            gate_count: Number of gates required
            optimization_goal: Optimization goal
            
        Returns:
            Execution plan
        """
        # Create task requirements
        task_requirements = QuantumTaskRequirements(
            task_type=task_type,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            gate_count=gate_count,
            required_fidelity=0.90,
            max_execution_time_ms=5000
        )
        
        # Create execution plan
        return self.hardware_selector.get_execution_plan(task_requirements, optimization_goal)
    
    def execute_circuit_optimization(self, circuit_profile: QuantumCircuitProfile,
                                     optimization_goal: str = 'balanced') -> Dict[str, Any]:
        """
        Execute circuit optimization with automatic hardware selection
        
        Args:
            circuit_profile: Circuit profile to optimize
            optimization_goal: Optimization goal
            
        Returns:
            Execution result
        """
        return self.task_executor.execute_circuit_optimization(circuit_profile, optimization_goal)
    
    def execute_quantum_neural_network(self, architecture: QNNArchitecture,
                                       training_data: Tuple[np.ndarray, np.ndarray],
                                       epochs: int = 10,
                                       optimization_goal: str = 'balanced') -> Dict[str, Any]:
        """
        Execute quantum neural network training with automatic hardware selection
        
        Args:
            architecture: QNN architecture
            training_data: Training data (X, y)
            epochs: Number of training epochs
            optimization_goal: Optimization goal
            
        Returns:
            Execution result
        """
        return self.task_executor.execute_quantum_neural_network(architecture, training_data, epochs, optimization_goal)
    
    def execute_quantum_regime_detection(self, training_data: List[Tuple[MarketDataFeatures, MarketRegime]],
                                          test_features: MarketDataFeatures,
                                          optimization_goal: str = 'balanced') -> Dict[str, Any]:
        """
        Execute quantum regime detection with automatic hardware selection
        
        Args:
            training_data: Training data
            test_features: Features to detect regime for
            optimization_goal: Optimization goal
            
        Returns:
            Execution result
        """
        return self.task_executor.execute_quantum_regime_detection(training_data, test_features, optimization_goal)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about quantum task executions"""
        return {
            'task_executor': self.task_executor.get_execution_statistics(),
            'hardware_selector': self.hardware_selector.get_backend_statistics(),
            'hardware_selector_task_types': self.hardware_selector.get_task_type_statistics(),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Create and test the quantum hardware orchestrator
    orchestrator = QuantumHardwareOrchestrator()
    
    # Print system status
    print("=== QUANTUM HARDWARE ORCHESTRATOR STATUS ===")
    status = orchestrator.get_system_status()
    print(f"Hardware Profiler: {status['hardware_profiler']['total_backends']} backends")
    print(f"Hardware Selector: {status['hardware_selector']['selection_history']} selections")
    print(f"Task Executor: {status['task_executor']['execution_history']} executions")
    
    # Get recommendations for different task types
    print("\n=== HARDWARE RECOMMENDATIONS ===")
    for task_type in QuantumTaskType:
        recommendations = orchestrator.get_backend_recommendations(task_type)
        best = recommendations['best_recommendation']
        print(f"{task_type.name}:")
        print(f"  Best backend: {best['backend'] if best else 'None'}")
        print(f"  Quantum volume: {best['quantum_volume'] if best else 'N/A'}")
        print(f"  Expected fidelity: {best['expected_fidelity']:.2%} if best else 'N/A'")
        print(f"  Expected execution time: {best['expected_execution_time_ms']:.1f}ms if best else 'N/A'")
        print()