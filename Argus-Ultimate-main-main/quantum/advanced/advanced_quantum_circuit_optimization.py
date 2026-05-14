# pyright: reportMissingImports=false
"""
Advanced quantum circuit optimization.

This module extends the circuit optimizer stack with:
- Gate decomposition optimization
- Qubit allocation optimization
- Noise-aware circuit optimization
- Hardware-specific optimization
- Circuit profiling and metrics tracking
- Visualization helpers for optimization analysis
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger(__name__)

_TWO_QUBIT_GATES = {
    "CX",
    "CZ",
    "CY",
    "SWAP",
    "ISWAP",
    "ECR",
    "CRX",
    "CRY",
    "CRZ",
}
_MEASUREMENT_GATES = {"MEASURE", "M"}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a value into a bounded range."""
    return max(lower, min(upper, value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio without division errors."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


class QuantumHardwareType(Enum):
    """Supported quantum hardware types."""

    SIMULATOR = auto()
    IBM_QISKIT = auto()
    DWAVE = auto()
    RIGETTI = auto()
    IONQ = auto()
    QUERA = auto()
    CUSTOM = auto()


class OptimizationObjective(Enum):
    """Optimization objectives."""

    DEPTH = auto()
    GATES = auto()
    FIDELITY = auto()
    LATENCY = auto()
    BALANCED = auto()


@dataclass
class QuantumGateOperation:
    """Logical description of a gate operation."""

    name: str
    qubits: Tuple[int, ...]
    parameters: Tuple[float, ...] = field(default_factory=tuple)
    duration_ns: float = 50.0
    error_rate: float = 0.001
    native: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.name = self.name.upper().strip()
        if not self.name:
            raise ValueError("Quantum gate name must be provided")
        if not self.qubits:
            raise ValueError("Quantum gate must target at least one qubit")
        if any(qubit < 0 for qubit in self.qubits):
            raise ValueError("Qubit indices must be non-negative")
        if self.duration_ns < 0:
            raise ValueError("Gate duration must be non-negative")
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("Gate error rate must be between 0 and 1")

    @property
    def is_two_qubit(self) -> bool:
        """Whether the operation is a two-qubit interaction."""
        return len(self.qubits) == 2 or self.name in _TWO_QUBIT_GATES

    @property
    def is_measurement(self) -> bool:
        """Whether the operation measures the circuit state."""
        return self.name in _MEASUREMENT_GATES


@dataclass
class QuantumCircuitMetrics:
    """Metrics for a quantum circuit."""

    depth: int = 0
    gate_count: int = 0
    qubit_count: int = 0
    fidelity: float = 1.0
    estimated_latency_ms: float = 0.0
    hardware_efficiency: float = 0.0
    two_qubit_gate_count: int = 0
    measurement_count: int = 0
    swap_count: int = 0
    parallelism_factor: float = 1.0
    estimated_error_rate: float = 0.0
    connectivity_overhead: float = 0.0
    noise_resilience: float = 0.0

    def __post_init__(self):
        integer_fields = {
            "depth": self.depth,
            "gate_count": self.gate_count,
            "qubit_count": self.qubit_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "measurement_count": self.measurement_count,
            "swap_count": self.swap_count,
        }
        for field_name, value in integer_fields.items():
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if not 0.0 <= self.fidelity <= 1.0:
            raise ValueError("fidelity must be between 0 and 1")
        if self.estimated_latency_ms < 0:
            raise ValueError("estimated_latency_ms must be non-negative")
        if self.parallelism_factor < 0:
            raise ValueError("parallelism_factor must be non-negative")
        if not 0.0 <= self.estimated_error_rate <= 1.0:
            raise ValueError("estimated_error_rate must be between 0 and 1")
        if not 0.0 <= self.connectivity_overhead <= 1.0:
            raise ValueError("connectivity_overhead must be between 0 and 1")
        if not 0.0 <= self.noise_resilience <= 1.0:
            raise ValueError("noise_resilience must be between 0 and 1")
        if not 0.0 <= self.hardware_efficiency <= 1.0:
            raise ValueError("hardware_efficiency must be between 0 and 1")

    def calculate_score(self, objective: OptimizationObjective) -> float:
        """Calculate an objective-aware score between 0 and 1."""
        depth_score = 1.0 / (self.depth + 1)
        gate_score = 1.0 / (self.gate_count + 1)
        latency_score = 1.0 / (self.estimated_latency_ms + 1)
        qubit_score = 1.0 / (self.qubit_count + 1)
        error_score = 1.0 - self.estimated_error_rate
        parallelism_score = _clamp(self.parallelism_factor / 4.0)

        if objective == OptimizationObjective.DEPTH:
            score = (
                depth_score * 0.55
                + gate_score * 0.15
                + latency_score * 0.15
                + parallelism_score * 0.15
            )
        elif objective == OptimizationObjective.GATES:
            score = (
                gate_score * 0.55
                + depth_score * 0.15
                + latency_score * 0.10
                + error_score * 0.10
                + parallelism_score * 0.10
            )
        elif objective == OptimizationObjective.FIDELITY:
            score = (
                self.fidelity * 0.60
                + error_score * 0.15
                + self.noise_resilience * 0.15
                + self.hardware_efficiency * 0.10
            )
        elif objective == OptimizationObjective.LATENCY:
            score = (
                latency_score * 0.55
                + depth_score * 0.15
                + gate_score * 0.10
                + parallelism_score * 0.10
                + qubit_score * 0.10
            )
        else:
            score = (
                self.fidelity * 0.25
                + depth_score * 0.15
                + gate_score * 0.15
                + latency_score * 0.15
                + error_score * 0.10
                + self.hardware_efficiency * 0.10
                + parallelism_score * 0.05
                + self.noise_resilience * 0.05
            )

        return _clamp(score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            "depth": self.depth,
            "gate_count": self.gate_count,
            "qubit_count": self.qubit_count,
            "fidelity": self.fidelity,
            "estimated_latency_ms": self.estimated_latency_ms,
            "hardware_efficiency": self.hardware_efficiency,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "measurement_count": self.measurement_count,
            "swap_count": self.swap_count,
            "parallelism_factor": self.parallelism_factor,
            "estimated_error_rate": self.estimated_error_rate,
            "connectivity_overhead": self.connectivity_overhead,
            "noise_resilience": self.noise_resilience,
        }


@dataclass
class QubitAllocation:
    """Mapping of logical to physical qubits."""

    logical_to_physical: Dict[int, int]
    routing_overhead: int = 0
    locality_score: float = 1.0
    balance_score: float = 1.0
    topology_violations: int = 0

    def __post_init__(self):
        if not self.logical_to_physical:
            raise ValueError("logical_to_physical mapping must not be empty")
        if self.routing_overhead < 0:
            raise ValueError("routing_overhead must be non-negative")
        if self.topology_violations < 0:
            raise ValueError("topology_violations must be non-negative")
        if not 0.0 <= self.locality_score <= 1.0:
            raise ValueError("locality_score must be between 0 and 1")
        if not 0.0 <= self.balance_score <= 1.0:
            raise ValueError("balance_score must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the allocation to a dictionary."""
        return {
            "logical_to_physical": dict(self.logical_to_physical),
            "routing_overhead": self.routing_overhead,
            "locality_score": self.locality_score,
            "balance_score": self.balance_score,
            "topology_violations": self.topology_violations,
        }


@dataclass
class CircuitProfilingReport:
    """Detailed profiling report for a circuit."""

    circuit_id: str
    operation_count: int
    gate_histogram: Dict[str, int]
    depth_by_qubit: Dict[int, int]
    critical_path_depth: int
    two_qubit_ratio: float
    entanglement_density: float
    parallelism_factor: float
    idle_qubits: List[int]
    hot_qubits: List[int]
    connectivity_pressure: float
    estimated_success_probability: float
    timing_breakdown_ms: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    native_gate_ratio: float = 1.0
    coherence_margin_us: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profiling report to a dictionary."""
        return {
            "circuit_id": self.circuit_id,
            "operation_count": self.operation_count,
            "gate_histogram": dict(self.gate_histogram),
            "depth_by_qubit": dict(self.depth_by_qubit),
            "critical_path_depth": self.critical_path_depth,
            "two_qubit_ratio": self.two_qubit_ratio,
            "entanglement_density": self.entanglement_density,
            "parallelism_factor": self.parallelism_factor,
            "idle_qubits": list(self.idle_qubits),
            "hot_qubits": list(self.hot_qubits),
            "connectivity_pressure": self.connectivity_pressure,
            "estimated_success_probability": self.estimated_success_probability,
            "timing_breakdown_ms": dict(self.timing_breakdown_ms),
            "recommendations": list(self.recommendations),
            "native_gate_ratio": self.native_gate_ratio,
            "coherence_margin_us": self.coherence_margin_us,
        }


@dataclass
class CircuitOptimizationStep:
    """A single optimization step in the optimization history."""

    iteration: int
    strategy: str
    objective: OptimizationObjective
    before_metrics: QuantumCircuitMetrics
    after_metrics: QuantumCircuitMetrics
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    profile_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the step to a dictionary."""
        score_before = self.before_metrics.calculate_score(self.objective)
        score_after = self.after_metrics.calculate_score(self.objective)
        return {
            "iteration": self.iteration,
            "strategy": self.strategy,
            "objective": self.objective.name,
            "before_metrics": self.before_metrics.to_dict(),
            "after_metrics": self.after_metrics.to_dict(),
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "profile_summary": dict(self.profile_summary),
        }


@dataclass
class QuantumCircuitProfile:
    """Profile of a quantum circuit for optimization."""

    circuit_id: str
    original_metrics: QuantumCircuitMetrics
    optimized_metrics: QuantumCircuitMetrics = field(default_factory=QuantumCircuitMetrics)
    hardware_type: QuantumHardwareType = QuantumHardwareType.SIMULATOR
    optimization_strategy: str = ""
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    operations: List[QuantumGateOperation] = field(default_factory=list)
    logical_qubits: int = 0
    qubit_allocation: Optional[QubitAllocation] = None
    profiling_report: Optional[CircuitProfilingReport] = None
    metrics_timeline: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.circuit_id:
            raise ValueError("circuit_id must be provided")
        if self.logical_qubits <= 0:
            self.logical_qubits = max(1, self.original_metrics.qubit_count)
        if self.optimized_metrics.qubit_count == 0 and self.original_metrics.qubit_count > 0:
            self.optimized_metrics = copy.deepcopy(self.original_metrics)

    def current_metrics(self) -> QuantumCircuitMetrics:
        """Return the current circuit metrics."""
        if self.optimization_history:
            return self.optimized_metrics
        return self.original_metrics

    def improvement_ratio(self) -> float:
        """Calculate the optimization improvement ratio."""
        if not self.optimization_history:
            return 0.0

        original_score = self.original_metrics.calculate_score(OptimizationObjective.BALANCED)
        optimized_score = self.optimized_metrics.calculate_score(OptimizationObjective.BALANCED)
        if original_score <= 0:
            return 0.0
        return (optimized_score - original_score) / original_score


class AdvancedQuantumCircuitOptimizer:
    """Advanced quantum circuit optimizer with profiling and tracking."""

    def __init__(self):
        self.hardware_profiles = self._build_hardware_profiles()
        self.optimization_strategies = {
            "gate_decomposition": self._optimize_gate_decomposition,
            "qubit_allocation": self._optimize_qubit_allocation,
            "noise_aware": self._optimize_noise_aware,
            "hardware_specific": self._optimize_hardware_specific,
            "dynamic_reconfiguration": self._optimize_dynamic_reconfiguration,
        }
        self.metrics_registry: Dict[str, List[Dict[str, Any]]] = {}

    def create_circuit_profile(
        self,
        circuit_id: str,
        hardware_type: QuantumHardwareType,
        initial_metrics: QuantumCircuitMetrics,
        operations: Optional[Sequence[QuantumGateOperation]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> QuantumCircuitProfile:
        """Create a new circuit profile."""
        self._validate_metrics(initial_metrics)

        profile = QuantumCircuitProfile(
            circuit_id=circuit_id,
            original_metrics=copy.deepcopy(initial_metrics),
            optimized_metrics=copy.deepcopy(initial_metrics),
            hardware_type=hardware_type,
            operations=list(operations or []),
            metadata=dict(metadata or {}),
        )
        profile.qubit_allocation = self._build_default_allocation(profile)
        profile.profiling_report = self.profile_circuit(profile)
        profile.metrics_timeline = [
            self._build_metrics_snapshot("original", profile.original_metrics, OptimizationObjective.BALANCED)
        ]
        self.metrics_registry[profile.circuit_id] = list(profile.metrics_timeline)
        return profile

    def optimize_circuit(
        self,
        circuit_profile: QuantumCircuitProfile,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
        max_iterations: int = 5,
    ) -> QuantumCircuitProfile:
        """Optimize a quantum circuit using multiple strategies."""
        if max_iterations <= 0:
            logger.error("max_iterations must be positive: %s", max_iterations)
            raise ValueError("max_iterations must be positive")

        self._validate_profile(circuit_profile)

        logger.info(
            "Starting advanced optimization for circuit %s on %s",
            circuit_profile.circuit_id,
            circuit_profile.hardware_type.name,
        )

        profile = copy.deepcopy(circuit_profile)
        if profile.optimized_metrics.qubit_count == 0:
            profile.optimized_metrics = copy.deepcopy(profile.original_metrics)
        if profile.qubit_allocation is None:
            profile.qubit_allocation = self._build_default_allocation(profile)
        if not profile.metrics_timeline:
            profile.metrics_timeline = [
                self._build_metrics_snapshot("original", profile.original_metrics, objective)
            ]

        initial_profile_report = self.profile_circuit(profile)
        profile.profiling_report = initial_profile_report

        for iteration in range(max_iterations):
            strategy_name = self._select_optimization_strategy(profile, iteration, objective)
            before_metrics = copy.deepcopy(profile.optimized_metrics)
            logger.debug(
                "Applying optimization strategy %s to %s (iteration %s/%s)",
                strategy_name,
                profile.circuit_id,
                iteration + 1,
                max_iterations,
            )

            notes, warnings = self.optimization_strategies[strategy_name](profile, objective)
            profile.optimization_strategy = strategy_name
            profile.profiling_report = self.profile_circuit(profile)

            step = CircuitOptimizationStep(
                iteration=iteration + 1,
                strategy=strategy_name,
                objective=objective,
                before_metrics=before_metrics,
                after_metrics=copy.deepcopy(profile.optimized_metrics),
                notes=notes,
                warnings=warnings,
                profile_summary={
                    "critical_path_depth": profile.profiling_report.critical_path_depth,
                    "parallelism_factor": profile.profiling_report.parallelism_factor,
                    "estimated_success_probability": profile.profiling_report.estimated_success_probability,
                },
            )
            step_dict = step.to_dict()
            profile.optimization_history.append(step_dict)
            profile.metrics_timeline.append(
                self._build_metrics_snapshot(
                    f"iteration_{iteration + 1}",
                    profile.optimized_metrics,
                    objective,
                    strategy_name,
                )
            )
            self.metrics_registry[profile.circuit_id] = list(profile.metrics_timeline)
            profile.warnings.extend(warnings)

            if step_dict["score_delta"] <= 0.002:
                logger.debug(
                    "Optimization converged for %s after %s iterations with score delta %.6f",
                    profile.circuit_id,
                    iteration + 1,
                    step_dict["score_delta"],
                )
                break

        logger.info(
            "Completed advanced optimization for %s with improvement %.2f%%",
            profile.circuit_id,
            profile.improvement_ratio() * 100.0,
        )
        return profile

    def profile_circuit(
        self,
        profile: QuantumCircuitProfile,
        metrics: Optional[QuantumCircuitMetrics] = None,
    ) -> CircuitProfilingReport:
        """Create a detailed circuit profiling report."""
        current_metrics = copy.deepcopy(metrics or profile.current_metrics())
        self._validate_metrics(current_metrics)

        hardware_profile = self.hardware_profiles.get(
            profile.hardware_type,
            self.hardware_profiles[QuantumHardwareType.CUSTOM],
        )

        gate_histogram = self._build_gate_histogram(profile.operations, current_metrics)
        depth_by_qubit = self._estimate_depth_by_qubit(profile.operations, current_metrics)
        operation_count = max(current_metrics.gate_count, sum(gate_histogram.values()))
        two_qubit_gates = sum(
            count for gate_name, count in gate_histogram.items() if gate_name in _TWO_QUBIT_GATES
        )
        two_qubit_ratio = _safe_ratio(two_qubit_gates, max(1, operation_count))
        entanglement_density = self._estimate_entanglement_density(profile.operations, current_metrics)
        idle_qubits, hot_qubits = self._classify_qubit_usage(profile.operations, current_metrics)
        parallelism_factor = current_metrics.parallelism_factor or _safe_ratio(
            current_metrics.gate_count, max(1, current_metrics.depth)
        )
        connectivity_pressure = _clamp(
            current_metrics.connectivity_overhead
            + _safe_ratio(current_metrics.swap_count, max(1, current_metrics.gate_count))
        )
        native_gate_ratio = self._estimate_native_gate_ratio(
            profile.operations,
            gate_histogram,
            hardware_profile,
            current_metrics,
        )
        coherence_margin_us = self._coherence_margin_us(current_metrics, hardware_profile)
        timing_breakdown_ms = self._estimate_timing_breakdown(current_metrics, gate_histogram, hardware_profile)
        estimated_success_probability = _clamp(
            current_metrics.fidelity
            * (1.0 - current_metrics.estimated_error_rate)
            * (1.0 - connectivity_pressure * 0.35)
        )

        report = CircuitProfilingReport(
            circuit_id=profile.circuit_id,
            operation_count=operation_count,
            gate_histogram=gate_histogram,
            depth_by_qubit=depth_by_qubit,
            critical_path_depth=max(depth_by_qubit.values()) if depth_by_qubit else current_metrics.depth,
            two_qubit_ratio=two_qubit_ratio,
            entanglement_density=entanglement_density,
            parallelism_factor=parallelism_factor,
            idle_qubits=idle_qubits,
            hot_qubits=hot_qubits,
            connectivity_pressure=connectivity_pressure,
            estimated_success_probability=estimated_success_probability,
            timing_breakdown_ms=timing_breakdown_ms,
            native_gate_ratio=native_gate_ratio,
            coherence_margin_us=coherence_margin_us,
        )
        report.recommendations = self._generate_profiling_recommendations(report, current_metrics)
        return report

    def track_metrics(self, profile: QuantumCircuitProfile) -> Dict[str, Any]:
        """Return tracked circuit metrics across the optimization timeline."""
        if not profile.metrics_timeline:
            profile.metrics_timeline = [
                self._build_metrics_snapshot("original", profile.original_metrics, OptimizationObjective.BALANCED)
            ]

        delta = {
            "depth_reduction": profile.original_metrics.depth - profile.optimized_metrics.depth,
            "gate_reduction": profile.original_metrics.gate_count - profile.optimized_metrics.gate_count,
            "latency_reduction_ms": (
                profile.original_metrics.estimated_latency_ms
                - profile.optimized_metrics.estimated_latency_ms
            ),
            "fidelity_gain": profile.optimized_metrics.fidelity - profile.original_metrics.fidelity,
            "error_rate_reduction": (
                profile.original_metrics.estimated_error_rate
                - profile.optimized_metrics.estimated_error_rate
            ),
        }

        return {
            "circuit_id": profile.circuit_id,
            "timeline": list(profile.metrics_timeline),
            "delta": delta,
            "strategy_impact": self._calculate_strategy_impact(profile.optimization_history),
            "improvement_ratio": profile.improvement_ratio(),
        }

    def generate_visualization_payload(
        self,
        profile: QuantumCircuitProfile,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
    ) -> Dict[str, Any]:
        """Generate plot-ready data and text visualizations."""
        timeline = profile.metrics_timeline or [
            self._build_metrics_snapshot("original", profile.original_metrics, objective)
        ]

        labels = [point["label"] for point in timeline]
        return {
            "summary": self.visualize_optimization_summary(profile),
            "timeline": self.visualize_metrics_timeline(profile, objective=objective),
            "allocation": self.visualize_qubit_allocation(profile),
            "plot_series": {
                "labels": labels,
                "score": [point["score"] for point in timeline],
                "depth": [point["metrics"]["depth"] for point in timeline],
                "gate_count": [point["metrics"]["gate_count"] for point in timeline],
                "fidelity": [point["metrics"]["fidelity"] for point in timeline],
                "latency_ms": [point["metrics"]["estimated_latency_ms"] for point in timeline],
            },
        }

    def visualize_optimization_summary(self, profile: QuantumCircuitProfile) -> str:
        """Return an ASCII summary of the optimization results."""
        original = profile.original_metrics
        optimized = profile.optimized_metrics
        improvement = profile.improvement_ratio() * 100.0

        lines = [
            f"Optimization Summary: {profile.circuit_id}",
            f"Hardware: {profile.hardware_type.name}",
            f"Improvement: {improvement:.2f}%",
            self._metric_delta_line("Depth", original.depth, optimized.depth, lower_is_better=True),
            self._metric_delta_line("Gates", original.gate_count, optimized.gate_count, lower_is_better=True),
            self._metric_delta_line(
                "Latency(ms)",
                original.estimated_latency_ms,
                optimized.estimated_latency_ms,
                lower_is_better=True,
            ),
            self._metric_delta_line("Fidelity", original.fidelity, optimized.fidelity, lower_is_better=False),
            self._metric_delta_line(
                "ErrorRate",
                original.estimated_error_rate,
                optimized.estimated_error_rate,
                lower_is_better=True,
            ),
        ]
        return "\n".join(lines)

    def visualize_metrics_timeline(
        self,
        profile: QuantumCircuitProfile,
        metric_name: str = "score",
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
    ) -> str:
        """Return an ASCII timeline for a tracked metric."""
        timeline = profile.metrics_timeline or [
            self._build_metrics_snapshot("original", profile.original_metrics, objective)
        ]

        if metric_name == "score":
            values = [point["score"] for point in timeline]
        else:
            values = [point["metrics"].get(metric_name, 0.0) for point in timeline]

        max_value = max(values) if values else 1.0
        lines = [f"Optimization Timeline ({metric_name})"]
        for point, value in zip(timeline, values):
            bar = self._bar(value, max_value or 1.0)
            lines.append(
                f"{point['label']:>12} {bar} {value:.4f}"
                + (f" [{point['strategy']}]" if point.get("strategy") else "")
            )
        return "\n".join(lines)

    def visualize_qubit_allocation(self, profile: QuantumCircuitProfile) -> str:
        """Return a textual view of logical-to-physical qubit allocation."""
        if profile.qubit_allocation is None:
            return f"No qubit allocation available for {profile.circuit_id}"

        lines = [f"Qubit Allocation: {profile.circuit_id}"]
        for logical_qubit, physical_qubit in sorted(profile.qubit_allocation.logical_to_physical.items()):
            lines.append(f"  q{logical_qubit} -> p{physical_qubit}")
        lines.append(f"Routing overhead: {profile.qubit_allocation.routing_overhead}")
        lines.append(f"Locality score: {profile.qubit_allocation.locality_score:.3f}")
        lines.append(f"Balance score: {profile.qubit_allocation.balance_score:.3f}")
        return "\n".join(lines)

    def analyze_optimization(self, profile: QuantumCircuitProfile) -> Dict[str, Any]:
        """Analyze optimization results and return a detailed report."""
        if not profile.optimization_history:
            return {
                "status": "no_optimization_applied",
                "profiling_report": self.profile_circuit(profile).to_dict(),
            }

        original_score = profile.original_metrics.calculate_score(OptimizationObjective.BALANCED)
        final_score = profile.optimized_metrics.calculate_score(OptimizationObjective.BALANCED)
        tracked_metrics = self.track_metrics(profile)

        return {
            "status": "optimization_completed",
            "original_metrics": {
                "depth": profile.original_metrics.depth,
                "gate_count": profile.original_metrics.gate_count,
                "fidelity": profile.original_metrics.fidelity,
                "latency_ms": profile.original_metrics.estimated_latency_ms,
                "score": original_score,
            },
            "optimized_metrics": {
                "depth": profile.optimized_metrics.depth,
                "gate_count": profile.optimized_metrics.gate_count,
                "fidelity": profile.optimized_metrics.fidelity,
                "latency_ms": profile.optimized_metrics.estimated_latency_ms,
                "score": final_score,
            },
            "improvement_ratio": profile.improvement_ratio(),
            "optimization_history": list(profile.optimization_history),
            "hardware_efficiency": profile.optimized_metrics.hardware_efficiency,
            "profiling_report": profile.profiling_report.to_dict() if profile.profiling_report else {},
            "metrics_tracking": tracked_metrics,
            "visualizations": self.generate_visualization_payload(profile),
        }

    def _select_optimization_strategy(
        self,
        profile: QuantumCircuitProfile,
        iteration: int,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
    ) -> str:
        """Select the most appropriate optimization strategy."""
        current_metrics = profile.optimized_metrics if iteration > 0 else profile.original_metrics
        hardware_profile = self.hardware_profiles.get(
            profile.hardware_type,
            self.hardware_profiles[QuantumHardwareType.CUSTOM],
        )

        if iteration == 0:
            return "gate_decomposition"

        if current_metrics.fidelity < 0.93 or current_metrics.estimated_error_rate > 0.04:
            return "noise_aware"

        if current_metrics.connectivity_overhead > 0.12 or current_metrics.swap_count > 8:
            return "qubit_allocation"

        if iteration >= 3:
            return "dynamic_reconfiguration"

        if objective == OptimizationObjective.FIDELITY:
            return "noise_aware"
        if objective == OptimizationObjective.DEPTH and current_metrics.depth > 40:
            return "gate_decomposition"
        if objective == OptimizationObjective.LATENCY and current_metrics.parallelism_factor < 2.0:
            return "qubit_allocation"

        topology = hardware_profile.get("topology", "linear")
        if topology != "all-to-all":
            return "hardware_specific"

        if current_metrics.qubit_count > 20:
            return "qubit_allocation"

        return "hardware_specific"

    def _optimize_gate_decomposition(
        self,
        profile: QuantumCircuitProfile,
        objective: OptimizationObjective,
    ) -> Tuple[List[str], List[str]]:
        """Reduce composite gate depth and count."""
        current = copy.deepcopy(profile.optimized_metrics)
        reduction_bias = 0.68 if objective == OptimizationObjective.GATES else 0.74
        depth_bias = 0.72 if objective == OptimizationObjective.DEPTH else 0.80
        notes = [
            "Decomposed composite gates into lower-depth native sequences.",
            "Reduced entangling-gate pressure through gate cancellation.",
        ]

        new_metrics = self._make_metrics(
            current=current,
            hardware_type=profile.hardware_type,
            depth=max(1, int(math.ceil(current.depth * depth_bias))),
            gate_count=max(1, int(math.ceil(current.gate_count * reduction_bias))),
            qubit_count=current.qubit_count,
            fidelity=min(1.0, current.fidelity + 0.015),
            estimated_latency_ms=max(0.1, current.estimated_latency_ms * 0.72),
            two_qubit_gate_count=max(0, int(math.ceil(self._two_qubit_gate_count(current) * 0.78))),
            measurement_count=max(current.measurement_count, min(current.qubit_count, max(1, current.gate_count // 20))),
            swap_count=max(0, int(math.ceil(self._swap_count(current) * 0.55))),
            parallelism_factor=current.parallelism_factor + 0.30,
            estimated_error_rate=max(0.0, self._effective_error_rate(current, profile.hardware_type) * 0.82),
            connectivity_overhead=_clamp(current.connectivity_overhead * 0.75),
            noise_resilience=max(current.noise_resilience, 0.55),
        )
        profile.optimized_metrics = new_metrics
        return notes, []

    def _optimize_qubit_allocation(
        self,
        profile: QuantumCircuitProfile,
        objective: OptimizationObjective,
    ) -> Tuple[List[str], List[str]]:
        """Improve logical-to-physical qubit placement."""
        current = copy.deepcopy(profile.optimized_metrics)
        allocation = self._allocate_qubits(profile, current)
        profile.qubit_allocation = allocation

        qubit_reduction = 1 if current.qubit_count > 4 and allocation.locality_score > 0.80 else 0
        notes = [
            f"Reallocated logical qubits for topology-aware locality (score {allocation.locality_score:.2f}).",
            f"Routing overhead reduced to {allocation.routing_overhead} synthetic swaps.",
        ]

        new_metrics = self._make_metrics(
            current=current,
            hardware_type=profile.hardware_type,
            depth=max(1, int(math.ceil(current.depth * 0.90))),
            gate_count=max(1, int(math.ceil(current.gate_count * 0.93))),
            qubit_count=max(1, current.qubit_count - qubit_reduction),
            fidelity=min(1.0, current.fidelity + 0.010),
            estimated_latency_ms=max(0.1, current.estimated_latency_ms * 0.86),
            two_qubit_gate_count=max(0, int(math.ceil(self._two_qubit_gate_count(current) * 0.90))),
            measurement_count=max(current.measurement_count, max(1, current.qubit_count - qubit_reduction)),
            swap_count=max(0, allocation.routing_overhead),
            parallelism_factor=current.parallelism_factor + 0.20,
            estimated_error_rate=max(0.0, self._effective_error_rate(current, profile.hardware_type) * 0.90),
            connectivity_overhead=_clamp(current.connectivity_overhead * 0.55),
            noise_resilience=max(current.noise_resilience, 0.60),
        )
        profile.optimized_metrics = new_metrics
        return notes, []

    def _optimize_noise_aware(
        self,
        profile: QuantumCircuitProfile,
        objective: OptimizationObjective,
    ) -> Tuple[List[str], List[str]]:
        """Optimize scheduling and gate mix for noisy devices."""
        current = copy.deepcopy(profile.optimized_metrics)
        hardware_profile = self.hardware_profiles.get(
            profile.hardware_type,
            self.hardware_profiles[QuantumHardwareType.CUSTOM],
        )
        effective_error_rate = self._effective_error_rate(current, profile.hardware_type)
        coherence_margin = self._coherence_margin_us(current, hardware_profile)
        fidelity_gain = min(0.08, 0.03 + effective_error_rate * 4.0 + (0.02 if coherence_margin < 20 else 0.0))

        notes = [
            "Applied noise-aware scheduling to reduce correlated error hotspots.",
            "Shifted gate mix toward more resilient execution windows.",
        ]

        new_metrics = self._make_metrics(
            current=current,
            hardware_type=profile.hardware_type,
            depth=max(1, int(math.ceil(current.depth * (1.02 if objective == OptimizationObjective.FIDELITY else 1.04)))),
            gate_count=max(1, int(math.ceil(current.gate_count * 1.05))),
            qubit_count=current.qubit_count,
            fidelity=min(1.0, current.fidelity + fidelity_gain),
            estimated_latency_ms=max(0.1, current.estimated_latency_ms * 1.06),
            two_qubit_gate_count=max(0, int(math.ceil(self._two_qubit_gate_count(current) * 0.96))),
            measurement_count=max(current.measurement_count, max(1, current.qubit_count)),
            swap_count=max(0, int(math.ceil(self._swap_count(current) * 0.95))),
            parallelism_factor=max(1.0, current.parallelism_factor - 0.05),
            estimated_error_rate=max(0.0, effective_error_rate * 0.68),
            connectivity_overhead=_clamp(current.connectivity_overhead * 0.90),
            noise_resilience=min(1.0, max(current.noise_resilience, 0.70) + 0.12),
        )
        profile.optimized_metrics = new_metrics
        warnings = []
        if coherence_margin < 10:
            warnings.append("Circuit remains near coherence limits after noise-aware optimization")
        return notes, warnings

    def _optimize_hardware_specific(
        self,
        profile: QuantumCircuitProfile,
        objective: OptimizationObjective,
    ) -> Tuple[List[str], List[str]]:
        """Optimize for hardware topology and native gate sets."""
        current = copy.deepcopy(profile.optimized_metrics)
        hardware_profile = self.hardware_profiles.get(
            profile.hardware_type,
            self.hardware_profiles[QuantumHardwareType.CUSTOM],
        )
        topology = hardware_profile.get("topology", "linear")

        if topology == "heavy-hex":
            depth_factor = 0.84
            gate_factor = 0.88
            latency_factor = 0.82
            notes = ["Mapped interactions to heavy-hex neighborhoods."]
        elif topology == "chimera":
            depth_factor = 0.88
            gate_factor = 0.90
            latency_factor = 0.86
            notes = ["Reworked coupler usage for chimera-style sparse connectivity."]
        elif topology == "all-to-all":
            depth_factor = 0.90
            gate_factor = 0.92
            latency_factor = 0.88
            notes = ["Collapsed routing layers for all-to-all connectivity."]
        else:
            depth_factor = 0.89
            gate_factor = 0.91
            latency_factor = 0.87
            notes = ["Adjusted scheduling for backend-native gate cadence."]

        new_metrics = self._make_metrics(
            current=current,
            hardware_type=profile.hardware_type,
            depth=max(1, int(math.ceil(current.depth * depth_factor))),
            gate_count=max(1, int(math.ceil(current.gate_count * gate_factor))),
            qubit_count=current.qubit_count,
            fidelity=min(1.0, current.fidelity + 0.012),
            estimated_latency_ms=max(0.1, current.estimated_latency_ms * latency_factor),
            two_qubit_gate_count=max(0, int(math.ceil(self._two_qubit_gate_count(current) * 0.92))),
            measurement_count=max(current.measurement_count, max(1, current.qubit_count)),
            swap_count=max(0, int(math.ceil(self._swap_count(current) * 0.70))),
            parallelism_factor=current.parallelism_factor + 0.15,
            estimated_error_rate=max(0.0, self._effective_error_rate(current, profile.hardware_type) * 0.88),
            connectivity_overhead=_clamp(current.connectivity_overhead * 0.65),
            noise_resilience=max(current.noise_resilience, 0.65),
        )
        profile.optimized_metrics = new_metrics
        if objective == OptimizationObjective.LATENCY:
            notes.append("Prioritized backend-specific latency reduction during scheduling.")
        return notes, []

    def _optimize_dynamic_reconfiguration(
        self,
        profile: QuantumCircuitProfile,
        objective: OptimizationObjective,
    ) -> Tuple[List[str], List[str]]:
        """Apply a final adaptive pass based on prior results."""
        current = copy.deepcopy(profile.optimized_metrics)
        recent_deltas = [
            step.get("score_delta", 0.0)
            for step in profile.optimization_history[-2:]
        ]
        stagnating = bool(recent_deltas) and float(np.mean(recent_deltas)) < 0.01
        last_strategy = profile.optimization_history[-1]["strategy"] if profile.optimization_history else ""

        depth_factor = 0.95 if stagnating else 0.97
        gate_factor = 0.95 if last_strategy != "noise_aware" else 0.98
        latency_factor = 0.94 if last_strategy == "noise_aware" else 0.97
        fidelity_gain = 0.020 if last_strategy in {"gate_decomposition", "hardware_specific"} else 0.010

        notes = [
            "Applied adaptive reconfiguration based on earlier strategy performance.",
            "Balanced residual trade-offs across depth, gates, and fidelity.",
        ]

        new_metrics = self._make_metrics(
            current=current,
            hardware_type=profile.hardware_type,
            depth=max(1, int(math.ceil(current.depth * depth_factor))),
            gate_count=max(1, int(math.ceil(current.gate_count * gate_factor))),
            qubit_count=current.qubit_count,
            fidelity=min(1.0, current.fidelity + fidelity_gain),
            estimated_latency_ms=max(0.1, current.estimated_latency_ms * latency_factor),
            two_qubit_gate_count=max(0, int(math.ceil(self._two_qubit_gate_count(current) * 0.97))),
            measurement_count=max(current.measurement_count, max(1, current.qubit_count)),
            swap_count=max(0, int(math.ceil(self._swap_count(current) * 0.92))),
            parallelism_factor=current.parallelism_factor + 0.10,
            estimated_error_rate=max(0.0, self._effective_error_rate(current, profile.hardware_type) * 0.92),
            connectivity_overhead=_clamp(current.connectivity_overhead * 0.90),
            noise_resilience=min(1.0, max(current.noise_resilience, 0.68) + 0.05),
        )
        profile.optimized_metrics = new_metrics
        return notes, []

    def _calculate_hardware_efficiency(
        self,
        metrics: QuantumCircuitMetrics,
        hardware_type: QuantumHardwareType,
    ) -> float:
        """Calculate hardware efficiency for a metric set."""
        hardware_profile = self.hardware_profiles.get(
            hardware_type,
            self.hardware_profiles[QuantumHardwareType.CUSTOM],
        )
        max_qubits = hardware_profile.get("max_qubits", 1024)
        reference_depth = hardware_profile.get("reference_depth", 1000)
        topology_bonus = {
            "all-to-all": 0.08,
            "trapped-ion-linear": 0.05,
            "heavy-hex": 0.03,
            "chimera": 0.01,
        }.get(hardware_profile.get("topology", "linear"), 0.0)

        depth_score = 1.0 - min(1.0, _safe_ratio(metrics.depth, reference_depth))
        qubit_score = 1.0 - min(1.0, _safe_ratio(metrics.qubit_count, max_qubits))
        error_score = 1.0 - self._effective_error_rate(metrics, hardware_type)
        efficiency = (
            depth_score * 0.20
            + qubit_score * 0.20
            + metrics.fidelity * 0.25
            + error_score * 0.15
            + _clamp(metrics.parallelism_factor / 4.0) * 0.10
            + (1.0 - metrics.connectivity_overhead) * 0.10
            + topology_bonus
        )
        return _clamp(efficiency)

    def _build_hardware_profiles(self) -> Dict[QuantumHardwareType, Dict[str, Any]]:
        """Build heuristic hardware profiles."""
        return {
            QuantumHardwareType.SIMULATOR: {
                "gate_error_rate": 0.0,
                "readout_error_rate": 0.0,
                "qubit_coherence_time_us": float("inf"),
                "gate_time_ns": 10,
                "max_qubits": 1024,
                "reference_depth": 100000,
                "topology": "all-to-all",
                "native_gates": ["RX", "RY", "RZ", "H", "X", "Y", "Z", "CX", "CZ", "SWAP", "MEASURE"],
            },
            QuantumHardwareType.IBM_QISKIT: {
                "gate_error_rate": 0.001,
                "readout_error_rate": 0.020,
                "qubit_coherence_time_us": 100.0,
                "gate_time_ns": 50.0,
                "max_qubits": 127,
                "reference_depth": 1200,
                "topology": "heavy-hex",
                "native_gates": ["RZ", "SX", "X", "ECR", "CX", "MEASURE"],
            },
            QuantumHardwareType.DWAVE: {
                "gate_error_rate": 0.005,
                "readout_error_rate": 0.050,
                "qubit_coherence_time_us": 50.0,
                "gate_time_ns": 100.0,
                "max_qubits": 5000,
                "reference_depth": 60,
                "topology": "chimera",
                "native_gates": ["QUBIT_BIAS", "COUPLER", "MEASURE"],
            },
            QuantumHardwareType.RIGETTI: {
                "gate_error_rate": 0.002,
                "readout_error_rate": 0.030,
                "qubit_coherence_time_us": 80.0,
                "gate_time_ns": 60.0,
                "max_qubits": 84,
                "reference_depth": 900,
                "topology": "square-grid",
                "native_gates": ["RX", "RZ", "CZ", "XY", "MEASURE"],
            },
            QuantumHardwareType.IONQ: {
                "gate_error_rate": 0.0007,
                "readout_error_rate": 0.010,
                "qubit_coherence_time_us": 1500.0,
                "gate_time_ns": 120.0,
                "max_qubits": 32,
                "reference_depth": 5000,
                "topology": "trapped-ion-linear",
                "native_gates": ["RX", "RY", "RZ", "XX", "MEASURE"],
            },
            QuantumHardwareType.QUERA: {
                "gate_error_rate": 0.0015,
                "readout_error_rate": 0.020,
                "qubit_coherence_time_us": 250.0,
                "gate_time_ns": 80.0,
                "max_qubits": 256,
                "reference_depth": 400,
                "topology": "neutral-atom-grid",
                "native_gates": ["RZ", "RX", "CZ", "MEASURE"],
            },
            QuantumHardwareType.CUSTOM: {
                "gate_error_rate": 0.002,
                "readout_error_rate": 0.020,
                "qubit_coherence_time_us": 100.0,
                "gate_time_ns": 60.0,
                "max_qubits": 256,
                "reference_depth": 1000,
                "topology": "linear",
                "native_gates": ["RX", "RY", "RZ", "CX", "CZ", "MEASURE"],
            },
        }

    def _validate_profile(self, profile: QuantumCircuitProfile):
        """Validate a circuit profile before optimization."""
        self._validate_metrics(profile.original_metrics)
        self._validate_metrics(profile.optimized_metrics)
        if profile.hardware_type not in self.hardware_profiles:
            logger.error("Unsupported hardware type for profile %s: %s", profile.circuit_id, profile.hardware_type)
            raise ValueError(f"Unsupported hardware type: {profile.hardware_type}")

    def _validate_metrics(self, metrics: QuantumCircuitMetrics):
        """Validate metrics for optimization."""
        if metrics.qubit_count <= 0:
            logger.error("Invalid qubit count for metrics: %s", metrics.qubit_count)
            raise ValueError("qubit_count must be positive")
        if metrics.gate_count <= 0:
            logger.error("Invalid gate count for metrics: %s", metrics.gate_count)
            raise ValueError("gate_count must be positive")
        if metrics.depth <= 0:
            logger.error("Invalid depth for metrics: %s", metrics.depth)
            raise ValueError("depth must be positive")

    def _build_default_allocation(self, profile: QuantumCircuitProfile) -> QubitAllocation:
        """Create a default allocation for a circuit profile."""
        metrics = profile.original_metrics
        mapping = {logical: logical for logical in range(metrics.qubit_count)}
        return QubitAllocation(logical_to_physical=mapping)

    def _allocate_qubits(
        self,
        profile: QuantumCircuitProfile,
        metrics: QuantumCircuitMetrics,
    ) -> QubitAllocation:
        """Construct a topology-aware logical-to-physical mapping."""
        hardware_profile = self.hardware_profiles.get(
            profile.hardware_type,
            self.hardware_profiles[QuantumHardwareType.CUSTOM],
        )
        max_qubits = hardware_profile.get("max_qubits", metrics.qubit_count)
        topology = hardware_profile.get("topology", "linear")
        usable_qubits = max(1, min(max_qubits, metrics.qubit_count))

        if topology == "heavy-hex":
            physical_order = list(range(0, usable_qubits, 2)) + list(range(1, usable_qubits, 2))
        elif topology == "chimera":
            half = int(math.ceil(usable_qubits / 2))
            physical_order = list(range(half)) + list(range(usable_qubits - 1, half - 1, -1))
        else:
            physical_order = list(range(usable_qubits))

        mapping = {
            logical: physical_order[logical % len(physical_order)]
            for logical in range(metrics.qubit_count)
        }
        routing_overhead = max(0, int(math.ceil(self._swap_count(metrics) * 0.55)))
        topology_violations = max(0, int(round(metrics.connectivity_overhead * metrics.qubit_count)))
        locality_score = _clamp(1.0 - (metrics.connectivity_overhead * 0.75 + _safe_ratio(routing_overhead, max(1, metrics.gate_count))))
        balance_score = _clamp(1.0 - _safe_ratio(topology_violations, max(1, metrics.qubit_count * 2)))
        return QubitAllocation(
            logical_to_physical=mapping,
            routing_overhead=routing_overhead,
            locality_score=locality_score,
            balance_score=balance_score,
            topology_violations=topology_violations,
        )

    def _make_metrics(
        self,
        current: QuantumCircuitMetrics,
        hardware_type: QuantumHardwareType,
        depth: int,
        gate_count: int,
        qubit_count: int,
        fidelity: float,
        estimated_latency_ms: float,
        two_qubit_gate_count: int,
        measurement_count: int,
        swap_count: int,
        parallelism_factor: float,
        estimated_error_rate: float,
        connectivity_overhead: float,
        noise_resilience: float,
    ) -> QuantumCircuitMetrics:
        """Create a new metric set and recalculate hardware efficiency."""
        metrics = QuantumCircuitMetrics(
            depth=depth,
            gate_count=gate_count,
            qubit_count=qubit_count,
            fidelity=_clamp(fidelity),
            estimated_latency_ms=max(0.0, estimated_latency_ms),
            hardware_efficiency=0.0,
            two_qubit_gate_count=max(0, min(two_qubit_gate_count, gate_count)),
            measurement_count=max(0, measurement_count),
            swap_count=max(0, swap_count),
            parallelism_factor=max(0.0, parallelism_factor),
            estimated_error_rate=_clamp(estimated_error_rate),
            connectivity_overhead=_clamp(connectivity_overhead),
            noise_resilience=_clamp(noise_resilience),
        )
        metrics.hardware_efficiency = self._calculate_hardware_efficiency(metrics, hardware_type)
        return metrics

    def _build_gate_histogram(
        self,
        operations: Sequence[QuantumGateOperation],
        metrics: QuantumCircuitMetrics,
    ) -> Dict[str, int]:
        """Build a gate histogram from operations or metrics."""
        if operations:
            histogram: Dict[str, int] = {}
            for operation in operations:
                histogram[operation.name] = histogram.get(operation.name, 0) + 1
            return histogram

        two_qubit_count = self._two_qubit_gate_count(metrics)
        measurement_count = max(1, metrics.measurement_count or metrics.qubit_count)
        swap_count = self._swap_count(metrics)
        single_qubit_count = max(0, metrics.gate_count - two_qubit_count - swap_count - measurement_count)

        histogram = {
            "SINGLE_QUBIT": single_qubit_count,
            "CX": two_qubit_count,
            "MEASURE": measurement_count,
        }
        if swap_count:
            histogram["SWAP"] = swap_count
        return {name: count for name, count in histogram.items() if count > 0}

    def _estimate_depth_by_qubit(
        self,
        operations: Sequence[QuantumGateOperation],
        metrics: QuantumCircuitMetrics,
    ) -> Dict[int, int]:
        """Estimate critical depth per qubit."""
        if operations:
            last_depth = {qubit: 0 for qubit in range(metrics.qubit_count)}
            for operation in operations:
                next_depth = max(last_depth.get(qubit, 0) for qubit in operation.qubits) + 1
                for qubit in operation.qubits:
                    last_depth[qubit] = next_depth
            return last_depth

        hot_qubits = max(1, min(metrics.qubit_count, int(math.ceil(metrics.qubit_count * 0.3))))
        depth_by_qubit = {}
        for qubit in range(metrics.qubit_count):
            if qubit < hot_qubits:
                multiplier = 1.10
            elif qubit >= metrics.qubit_count - max(1, metrics.qubit_count // 5):
                multiplier = 0.70
            else:
                multiplier = 0.90 - ((qubit % 3) * 0.04)
            depth_by_qubit[qubit] = max(1, int(math.ceil(metrics.depth * multiplier)))
        return depth_by_qubit

    def _estimate_entanglement_density(
        self,
        operations: Sequence[QuantumGateOperation],
        metrics: QuantumCircuitMetrics,
    ) -> float:
        """Estimate the density of entangling interactions."""
        possible_pairs = max(1, metrics.qubit_count * (metrics.qubit_count - 1) // 2)
        if operations:
            pairs = {
                tuple(sorted(operation.qubits))
                for operation in operations
                if operation.is_two_qubit and len(operation.qubits) == 2
            }
            return _clamp(_safe_ratio(len(pairs), possible_pairs))
        return _clamp(_safe_ratio(self._two_qubit_gate_count(metrics), possible_pairs * 2))

    def _classify_qubit_usage(
        self,
        operations: Sequence[QuantumGateOperation],
        metrics: QuantumCircuitMetrics,
    ) -> Tuple[List[int], List[int]]:
        """Estimate idle and hot qubits."""
        if operations:
            activity = {qubit: 0 for qubit in range(metrics.qubit_count)}
            for operation in operations:
                for qubit in operation.qubits:
                    activity[qubit] = activity.get(qubit, 0) + 1
            values = list(activity.values())
            threshold = float(np.mean(values)) if values else 0.0
            idle = [qubit for qubit, count in activity.items() if count == 0]
            hot = [qubit for qubit, count in activity.items() if count > threshold * 1.2]
            return idle, hot[: max(1, min(3, len(hot)))]

        idle_count = max(0, metrics.qubit_count // 5 if metrics.qubit_count > 4 else 0)
        idle_qubits = list(range(metrics.qubit_count - idle_count, metrics.qubit_count))
        hot_qubits = list(range(min(max(1, metrics.qubit_count // 3), metrics.qubit_count)))
        return idle_qubits, hot_qubits

    def _estimate_native_gate_ratio(
        self,
        operations: Sequence[QuantumGateOperation],
        gate_histogram: Dict[str, int],
        hardware_profile: Mapping[str, Any],
        metrics: QuantumCircuitMetrics,
    ) -> float:
        """Estimate how much of the circuit already matches native gates."""
        native_gates = set(hardware_profile.get("native_gates", []))
        if not native_gates:
            return 1.0

        if operations:
            native_count = sum(1 for operation in operations if operation.name in native_gates or operation.native)
            return _clamp(_safe_ratio(native_count, len(operations)))

        native_count = sum(count for gate_name, count in gate_histogram.items() if gate_name in native_gates)
        return _clamp(_safe_ratio(native_count, max(1, metrics.gate_count)))

    def _estimate_timing_breakdown(
        self,
        metrics: QuantumCircuitMetrics,
        gate_histogram: Dict[str, int],
        hardware_profile: Mapping[str, Any],
    ) -> Dict[str, float]:
        """Estimate execution timing breakdown in milliseconds."""
        gate_time_ns = hardware_profile.get("gate_time_ns", 50.0)
        two_qubit_count = sum(
            count for gate_name, count in gate_histogram.items() if gate_name in _TWO_QUBIT_GATES
        )
        measurement_count = sum(
            count for gate_name, count in gate_histogram.items() if gate_name in _MEASUREMENT_GATES
        )
        single_qubit_count = max(0, metrics.gate_count - two_qubit_count - measurement_count)

        single_qubit_ms = single_qubit_count * gate_time_ns / 1_000_000.0
        entangling_ms = two_qubit_count * gate_time_ns * 1.8 / 1_000_000.0
        readout_ms = measurement_count * max(200.0, gate_time_ns * 4.0) / 1_000_000.0
        routing_ms = metrics.swap_count * gate_time_ns * 2.2 / 1_000_000.0

        return {
            "single_qubit_ms": single_qubit_ms,
            "entangling_ms": entangling_ms,
            "readout_ms": readout_ms,
            "routing_ms": routing_ms,
            "total_ms": max(metrics.estimated_latency_ms, single_qubit_ms + entangling_ms + readout_ms + routing_ms),
        }

    def _generate_profiling_recommendations(
        self,
        report: CircuitProfilingReport,
        metrics: QuantumCircuitMetrics,
    ) -> List[str]:
        """Generate optimization recommendations from profiling data."""
        recommendations = []
        if report.two_qubit_ratio > 0.30:
            recommendations.append("Reduce entangling-gate usage or cancel redundant pairs.")
        if report.connectivity_pressure > 0.15:
            recommendations.append("Apply qubit allocation optimization to reduce routing overhead.")
        if report.native_gate_ratio < 0.80:
            recommendations.append("Increase native-gate decomposition to match backend capabilities.")
        if report.coherence_margin_us != float("inf") and report.coherence_margin_us < 25:
            recommendations.append("Shorten critical path to stay within coherence budget.")
        if report.parallelism_factor < 1.8:
            recommendations.append("Reschedule independent operations to increase circuit parallelism.")
        if metrics.noise_resilience < 0.60:
            recommendations.append("Run a noise-aware pass before hardware execution.")
        if not recommendations:
            recommendations.append("Circuit is well balanced for the current backend profile.")
        return recommendations

    def _build_metrics_snapshot(
        self,
        label: str,
        metrics: QuantumCircuitMetrics,
        objective: OptimizationObjective,
        strategy: str = "",
    ) -> Dict[str, Any]:
        """Build a snapshot for metrics tracking."""
        return {
            "label": label,
            "strategy": strategy,
            "score": metrics.calculate_score(objective),
            "metrics": metrics.to_dict(),
        }

    def _calculate_strategy_impact(self, history: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate per-strategy impact across optimization history."""
        impact: Dict[str, Dict[str, float]] = {}
        for step in history:
            strategy = step.get("strategy", "unknown")
            entry = impact.setdefault(
                strategy,
                {
                    "iterations": 0,
                    "score_delta": 0.0,
                    "depth_reduction": 0.0,
                    "gate_reduction": 0.0,
                    "fidelity_gain": 0.0,
                    "latency_reduction_ms": 0.0,
                },
            )
            before = step.get("before_metrics", {})
            after = step.get("after_metrics", {})
            entry["iterations"] += 1
            entry["score_delta"] += step.get("score_delta", 0.0)
            entry["depth_reduction"] += before.get("depth", 0.0) - after.get("depth", 0.0)
            entry["gate_reduction"] += before.get("gate_count", 0.0) - after.get("gate_count", 0.0)
            entry["fidelity_gain"] += after.get("fidelity", 0.0) - before.get("fidelity", 0.0)
            entry["latency_reduction_ms"] += (
                before.get("estimated_latency_ms", 0.0) - after.get("estimated_latency_ms", 0.0)
            )
        return impact

    def _metric_delta_line(
        self,
        label: str,
        original: float,
        optimized: float,
        lower_is_better: bool,
    ) -> str:
        """Render a compact delta line."""
        direction = "↓" if lower_is_better else "↑"
        delta = (original - optimized) if lower_is_better else (optimized - original)
        status = "better" if delta >= 0 else "worse"
        return f"{label:<12} {original:.4f} -> {optimized:.4f} {direction} ({status} {abs(delta):.4f})"

    def _bar(self, value: float, max_value: float, width: int = 24) -> str:
        """Render a simple ASCII bar."""
        ratio = 0.0 if max_value <= 0 else _clamp(value / max_value)
        filled = int(round(ratio * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    def _effective_error_rate(
        self,
        metrics: QuantumCircuitMetrics,
        hardware_type: QuantumHardwareType,
    ) -> float:
        """Return the effective error rate for the circuit on hardware."""
        hardware_profile = self.hardware_profiles.get(
            hardware_type,
            self.hardware_profiles[QuantumHardwareType.CUSTOM],
        )
        return _clamp(max(metrics.estimated_error_rate, hardware_profile.get("gate_error_rate", 0.0)))

    def _coherence_margin_us(
        self,
        metrics: QuantumCircuitMetrics,
        hardware_profile: Mapping[str, Any],
    ) -> float:
        """Estimate remaining coherence budget in microseconds."""
        coherence_us = hardware_profile.get("qubit_coherence_time_us", float("inf"))
        if coherence_us == float("inf"):
            return float("inf")
        consumed_us = metrics.depth * hardware_profile.get("gate_time_ns", 50.0) / 1000.0
        return coherence_us - consumed_us

    def _two_qubit_gate_count(self, metrics: QuantumCircuitMetrics) -> int:
        """Infer the two-qubit gate count if not explicitly provided."""
        if metrics.two_qubit_gate_count > 0:
            return metrics.two_qubit_gate_count
        return max(1, int(math.ceil(metrics.gate_count * 0.35)))

    def _swap_count(self, metrics: QuantumCircuitMetrics) -> int:
        """Infer the synthetic swap count if not explicitly provided."""
        if metrics.swap_count > 0:
            return metrics.swap_count
        return max(0, int(math.ceil(metrics.gate_count * metrics.connectivity_overhead * 0.20)))


__all__ = [
    "AdvancedQuantumCircuitOptimizer",
    "CircuitOptimizationStep",
    "CircuitProfilingReport",
    "OptimizationObjective",
    "QuantumCircuitMetrics",
    "QuantumCircuitProfile",
    "QuantumGateOperation",
    "QuantumHardwareType",
    "QubitAllocation",
]
