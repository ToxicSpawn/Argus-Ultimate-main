from __future__ import annotations

import importlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

np = importlib.import_module("numpy")

logger = logging.getLogger(__name__)

try:
    transpile = getattr(importlib.import_module("qiskit"), "transpile")
    AerSimulator = getattr(importlib.import_module("qiskit_aer"), "AerSimulator")
    _has_aer = True
except Exception:
    transpile = None
    AerSimulator = None
    _has_aer = False

try:
    runtime_module = importlib.import_module("qiskit_ibm_runtime")
    QiskitRuntimeService = getattr(runtime_module, "QiskitRuntimeService")
    SamplerV2 = getattr(runtime_module, "SamplerV2")
    _has_ibm_runtime = True
except Exception:
    QiskitRuntimeService = None
    SamplerV2 = None
    _has_ibm_runtime = False


@dataclass(slots=True)
class IBMQuantumJobRequest:
    circuit: Any = None
    shots: int = 1024
    backend_name: str | None = None
    use_hardware: bool = False
    num_qubits: int | None = None
    probability_map: Mapping[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IBMQuantumJob:
    job_id: str
    backend_name: str
    mode: str
    status: str
    created_at: float
    request: IBMQuantumJobRequest
    native_job: Any = None
    completed_at: float | None = None
    error: str | None = None


@dataclass(slots=True)
class IBMQuantumResult:
    job_id: str
    backend_name: str
    mode: str
    status: str
    counts: dict[str, int]
    quasi_probabilities: dict[str, float]
    shots: int
    execution_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class IBMQuantumClient:
    """IBM Quantum job client with hardware, Aer, and pure-python simulator paths."""

    def __init__(self, api_token: str | None = None, default_backend: str | None = None) -> None:
        self.api_token = api_token or os.environ.get("IBM_QUANTUM_TOKEN")
        self.default_backend = default_backend or "aer_simulator"
        self._service = None
        self._jobs: dict[str, IBMQuantumJob] = {}
        self._results: dict[str, IBMQuantumResult] = {}

        if _has_ibm_runtime and self.api_token:
            try:
                if QiskitRuntimeService is None:
                    raise RuntimeError("QiskitRuntimeService is unavailable")
                self._service = QiskitRuntimeService(channel="ibm_quantum", token=self.api_token)
            except Exception as exc:
                logger.warning("IBM Quantum runtime unavailable, simulator mode only: %s", exc)
                self._service = None

    def submit_job(self, request: IBMQuantumJobRequest) -> IBMQuantumJob:
        job_id = f"ibm-job-{uuid.uuid4().hex[:12]}"
        backend_name = request.backend_name or self.default_backend
        mode = "hardware" if request.use_hardware and self._service is not None else ("aer_simulator" if _has_aer else "python_simulator")
        job = IBMQuantumJob(
            job_id=job_id,
            backend_name=backend_name,
            mode=mode,
            status="queued",
            created_at=time.time(),
            request=request,
        )
        self._jobs[job_id] = job

        try:
            if mode == "hardware":
                self._submit_runtime_job(job)
            else:
                self._complete_local_job(job)
        except Exception as exc:
            job.status = "failed"
            job.completed_at = time.time()
            job.error = str(exc)
            logger.warning("IBM job %s failed: %s", job_id, exc)

        return job

    def check_status(self, job_id: str) -> str:
        job = self._get_job(job_id)
        if job.native_job is not None and job.status not in {"completed", "failed", "cancelled"}:
            try:
                native_status = str(job.native_job.status()).lower()
                if "done" in native_status:
                    self._collect_runtime_result(job)
                elif "error" in native_status or "fail" in native_status:
                    job.status = "failed"
                else:
                    job.status = native_status
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
        return job.status

    def get_result(self, job_id: str) -> IBMQuantumResult:
        status = self.check_status(job_id)
        if status != "completed":
            raise RuntimeError(f"Job {job_id} is not complete (status={status})")
        return self._results[job_id]

    def _submit_runtime_job(self, job: IBMQuantumJob) -> None:
        if self._service is None or SamplerV2 is None:
            raise RuntimeError("IBM Quantum runtime service is not available")
        if job.request.circuit is None:
            raise ValueError("A qiskit circuit is required for hardware execution")
        if transpile is None:
            raise RuntimeError("qiskit transpile is not available")

        backend = self._service.backend(job.request.backend_name or self.default_backend)
        circuit = job.request.circuit
        if getattr(circuit, "num_clbits", 0) == 0 and hasattr(circuit, "measure_all"):
            circuit = circuit.copy()
            circuit.measure_all()
        transpiled_circuit = transpile(circuit, backend=backend)
        sampler = SamplerV2(mode=backend)
        job.native_job = sampler.run([transpiled_circuit], shots=int(job.request.shots))
        job.backend_name = getattr(backend, "name", job.backend_name)
        job.status = "running"

    def _collect_runtime_result(self, job: IBMQuantumJob) -> None:
        if job.native_job is None:
            raise RuntimeError("No runtime job associated with the requested job id")
        t0 = time.perf_counter()
        runtime_result = job.native_job.result()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        counts: dict[str, int] = {}
        quasi_probabilities: dict[str, float] = {}
        first_pub = runtime_result[0]
        if hasattr(first_pub, "data"):
            for key in first_pub.data:
                register = first_pub.data[key]
                if hasattr(register, "get_counts"):
                    counts = dict(register.get_counts())
                    break
        total_shots = max(sum(counts.values()), 1)
        quasi_probabilities = {state: count / total_shots for state, count in counts.items()}

        result = IBMQuantumResult(
            job_id=job.job_id,
            backend_name=job.backend_name,
            mode=job.mode,
            status="completed",
            counts=counts,
            quasi_probabilities=quasi_probabilities,
            shots=job.request.shots,
            execution_time_ms=elapsed_ms,
            metadata=dict(job.request.metadata),
        )
        job.status = "completed"
        job.completed_at = time.time()
        self._results[job.job_id] = result

    def _complete_local_job(self, job: IBMQuantumJob) -> None:
        t0 = time.perf_counter()
        if _has_aer and job.request.circuit is not None:
            counts = self._run_aer(job.request.circuit, job.request.shots)
            backend_name = "aer_simulator"
        else:
            counts = self._run_python_simulator(job.request)
            backend_name = "python_simulator"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_shots = max(sum(counts.values()), 1)
        quasi_probabilities = {state: count / total_shots for state, count in counts.items()}

        result = IBMQuantumResult(
            job_id=job.job_id,
            backend_name=backend_name,
            mode=job.mode,
            status="completed",
            counts=counts,
            quasi_probabilities=quasi_probabilities,
            shots=job.request.shots,
            execution_time_ms=elapsed_ms,
            metadata=dict(job.request.metadata),
        )
        job.backend_name = backend_name
        job.status = "completed"
        job.completed_at = time.time()
        self._results[job.job_id] = result

    def _run_aer(self, circuit: Any, shots: int) -> dict[str, int]:
        if AerSimulator is None or transpile is None:
            raise RuntimeError("Qiskit Aer is not available")
        simulator = AerSimulator()
        working_circuit = circuit
        if getattr(working_circuit, "num_clbits", 0) == 0 and hasattr(working_circuit, "measure_all"):
            working_circuit = working_circuit.copy()
            working_circuit.measure_all()
        transpiled_circuit = transpile(working_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=int(shots)).result()
        return dict(result.get_counts())

    def _run_python_simulator(self, request: IBMQuantumJobRequest) -> dict[str, int]:
        rng = np.random.default_rng()
        probability_map = dict(request.probability_map or {})
        if probability_map:
            states = list(probability_map.keys())
            probabilities = np.asarray([max(float(probability_map[state]), 0.0) for state in states], dtype=float)
            total = float(np.sum(probabilities))
            if total <= 0.0:
                raise ValueError("probability_map must contain positive values")
            probabilities = probabilities / total
        else:
            num_qubits = int(request.num_qubits or 1)
            states = [format(index, f"0{num_qubits}b") for index in range(2 ** num_qubits)]
            probabilities = np.full(len(states), 1.0 / len(states), dtype=float)

        counts = {state: 0 for state in states}
        samples = rng.choice(states, size=int(request.shots), p=probabilities)
        for state in samples:
            counts[str(state)] = counts.get(str(state), 0) + 1
        return counts

    def _get_job(self, job_id: str) -> IBMQuantumJob:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown IBM Quantum job id: {job_id}")
        return self._jobs[job_id]
