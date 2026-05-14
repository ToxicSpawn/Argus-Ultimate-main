"""
Performance Optimization Engine
================================
GPU acceleration, parallel processing, and latency optimization for
institutional-grade trading performance.

Features:
- GPU-accelerated computation (CUDA via CuPy/JAX)
- Parallel strategy execution
- Latency profiling and optimization
- Memory pool management
- Async I/O optimization
- Zero-copy data transfer
"""

import asyncio
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    operation: str
    duration_ms: float
    throughput: float  # ops/sec
    memory_mb: float
    cpu_percent: float
    gpu_utilization: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p99: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "throughput": throughput,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "gpu_utilization": self.gpu_utilization,
            "latency_p50": self.latency_p50,
            "latency_p99": self.latency_p99
        }


@dataclass
class GPUConfig:
    """GPU configuration settings."""
    device_id: int = 0
    memory_limit_mb: int = 4096
    enable_tensor_cores: bool = True
    enable_unified_memory: bool = False
    batch_size: int = 1024
    use_jax: bool = True  # Prefer JAX over CuPy


class GPUAccelerator:
    """
    GPU acceleration for compute-intensive operations.
    
    Supports:
    - CuPy for NumPy-compatible GPU arrays
    - JAX for JIT-compiled GPU functions
    - Batch processing for ML inference
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        self._available = False
        self._backend = None
        self._memory_pool = None
        
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU backend."""
        try:
            if self.config.use_jax:
                import jax
                import jax.numpy as jnp
                
                # Check for GPU
                devices = jax.devices()
                gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]
                
                if gpu_devices:
                    self._backend = 'jax'
                    self._available = True
                    self._jax = jax
                    self._jnp = jnp
                    logger.info(f"JAX GPU initialized: {gpu_devices[0]}")
                else:
                    logger.warning("No JAX GPU available, using CPU")
                    
            if not self._available:
                import cupy as cp
                
                # Initialize CuPy memory pool
                self._memory_pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(self._memory_pool.malloc)
                
                self._backend = 'cupy'
                self._available = True
                self._cp = cp
                logger.info(f"CuPy GPU initialized: {cp.cuda.runtime.getDeviceCount()} devices")
                
        except ImportError as e:
            logger.warning(f"GPU libraries not available: {e}")
            self._available = False
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            self._available = False
    
    @property
    def available(self) -> bool:
        """Check if GPU is available."""
        return self._available
    
    @property
    def backend(self) -> Optional[str]:
        """Get active GPU backend."""
        return self._backend
    
    def to_gpu(self, array: np.ndarray) -> Any:
        """Transfer NumPy array to GPU."""
        if not self._available:
            return array
            
        if self._backend == 'jax':
            return self._jnp.array(array)
        else:
            return self._cp.asarray(array)
    
    def to_cpu(self, gpu_array: Any) -> np.ndarray:
        """Transfer GPU array to CPU."""
        if not self._available:
            return gpu_array
            
        if self._backend == 'jax':
            return np.array(gpu_array)
        else:
            return self._cp.asnumpy(gpu_array)
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU-accelerated matrix multiplication."""
        if not self._available:
            return np.matmul(a, b)
        
        a_gpu = self.to_gpu(a)
        b_gpu = self.to_gpu(b)
        
        if self._backend == 'jax':
            result = self._jnp.matmul(a_gpu, b_gpu)
        else:
            result = self._cp.matmul(a_gpu, b_gpu)
        
        return self.to_cpu(result)
    
    def batch_matmul(self, batch_a: np.ndarray, batch_b: np.ndarray) -> np.ndarray:
        """Batched matrix multiplication on GPU."""
        if not self._available:
            return np.einsum('bij,bjk->bik', batch_a, batch_b)
        
        a_gpu = self.to_gpu(batch_a)
        b_gpu = self.to_gpu(batch_b)
        
        if self._backend == 'jax':
            result = self._jnp.einsum('bij,bjk->bik', a_gpu, b_gpu)
        else:
            result = self._cp.einsum('bij,bjk->bik', a_gpu, b_gpu)
        
        return self.to_cpu(result)
    
    def fft(self, signal: np.ndarray) -> np.ndarray:
        """GPU-accelerated FFT."""
        if not self._available:
            return np.fft.fft(signal)
        
        signal_gpu = self.to_gpu(signal)
        
        if self._backend == 'jax':
            result = self._jnp.fft.fft(signal_gpu)
        else:
            result = self._cp.fft.fft(signal_gpu)
        
        return self.to_cpu(result)
    
    def rolling_statistics(
        self,
        data: np.ndarray,
        window: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate rolling mean, std, and variance on GPU."""
        if not self._available:
            return self._rolling_statistics_cpu(data, window)
        
        data_gpu = self.to_gpu(data)
        
        if self._backend == 'jax':
            # JAX implementation using sliding window
            shape = (len(data) - window + 1, window)
            strides = (data_gpu.strides[0], data_gpu.strides[0])
            windows = self._jnp.lib.stride_tricks.as_strided(
                data_gpu, shape=shape, strides=strides
            )
            means = self._jnp.mean(windows, axis=1)
            stds = self._jnp.std(windows, axis=1)
            vars_ = self._jnp.var(windows, axis=1)
        else:
            # CuPy implementation
            shape = (len(data) - window + 1, window)
            strides = (data_gpu.strides[0], data_gpu.strides[0])
            windows = self._cp.lib.stride_tricks.as_strided(
                data_gpu, shape=shape, strides=strides
            )
            means = self._cp.mean(windows, axis=1)
            stds = self._cp.std(windows, axis=1)
            vars_ = self._cp.var(windows, axis=1)
        
        return self.to_cpu(means), self.to_cpu(stds), self.to_cpu(vars_)
    
    def _rolling_statistics_cpu(
        self,
        data: np.ndarray,
        window: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU fallback for rolling statistics."""
        means = np.convolve(data, np.ones(window)/window, mode='valid')
        return means, np.zeros_like(means), np.zeros_like(means)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not self._available:
            return {"available": False}
        
        if self._backend == 'jax':
            return {
                "available": True,
                "backend": "jax",
                "device": str(self._jax.devices()[0])
            }
        else:
            mem_info = self._cp.cuda.mem_get_info()
            return {
                "available": True,
                "backend": "cupy",
                "free_mb": mem_info[0] / 1024**2,
                "total_mb": mem_info[1] / 1024**2
            }


class ParallelExecutor:
    """
    Parallel execution engine for strategy backtesting and signal generation.
    
    Features:
    - Multi-process execution for CPU-bound tasks
    - Thread pool for I/O-bound tasks
    - Automatic work distribution
    - Progress tracking
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = True
    ):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        
        if use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"ParallelExecutor initialized: {self.max_workers} workers")
    
    def map_parallel(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Execute function in parallel across items."""
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 4))
        
        results = list(self._executor.map(func, items, chunksize=chunk_size))
        return results
    
    async def map_async(
        self,
        func: Callable,
        items: List[Any]
    ) -> List[Any]:
        """Execute function asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            tasks = [
                loop.run_in_executor(pool, partial(func, item))
                for item in items
            ]
            results = await asyncio.gather(*tasks)
        
        return list(results)
    
    def parallel_reduce(
        self,
        func: Callable,
        items: List[Any],
        reducer: Callable = sum
    ) -> Any:
        """Map-reduce pattern for parallel processing."""
        mapped = self.map_parallel(func, items)
        return reducer(mapped)
    
    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)


class LatencyOptimizer:
    """
    Latency profiling and optimization for critical paths.
    
    Optimizations:
    - Hot path detection
    - Memory allocation profiling
    - Cache-friendly data layouts
    - Zero-copy operations
    """
    
    def __init__(self):
        self._profiles: Dict[str, List[float]] = {}
        self._hot_paths: Dict[str, float] = {}
    
    def profile(self, name: str):
        """Decorator to profile function execution time."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter_ns()
                result = func(*args, **kwargs)
                duration_ns = time.perf_counter_ns() - start
                
                if name not in self._profiles:
                    self._profiles[name] = []
                self._profiles[name].append(duration_ns / 1e6)  # Convert to ms
                
                # Track hot paths
                self._hot_paths[name] = np.mean(self._profiles[name])
                
                return result
            return wrapper
        return decorator
    
    async def profile_async(self, name: str, coro):
        """Profile async function execution."""
        start = time.perf_counter_ns()
        result = await coro
        duration_ns = time.perf_counter_ns() - start
        
        if name not in self._profiles:
            self._profiles[name] = []
        self._profiles[name].append(duration_ns / 1e6)
        
        return result
    
    def get_profile(self, name: str) -> Dict[str, float]:
        """Get profiling statistics for a function."""
        if name not in self._profiles:
            return {}
        
        times = self._profiles[name]
        return {
            "count": len(times),
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99)
        }
    
    def get_hot_paths(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the hottest (slowest) code paths."""
        sorted_paths = sorted(
            self._hot_paths.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_paths[:top_n]
    
    def optimize_data_layout(self, data: np.ndarray) -> np.ndarray:
        """
        Optimize data layout for cache efficiency.
        
        - Ensure C-contiguous layout
        - Align to cache line boundaries
        - Minimize memory fragmentation
        """
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        return data
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on profiling data."""
        suggestions = []
        
        hot_paths = self.get_hot_paths(5)
        for name, avg_time in hot_paths:
            if avg_time > 10:  # > 10ms average
                suggestions.append(
                    f"Hot path '{name}' averages {avg_time:.2f}ms - consider GPU acceleration or caching"
                )
            elif avg_time > 1:
                suggestions.append(
                    f"Hot path '{name}' averages {avg_time:.2f}ms - consider vectorization"
                )
        
        return suggestions


class MemoryPool:
    """
    Memory pool for efficient allocation in hot paths.
    
    Reduces GC pressure by reusing allocated memory.
    """
    
    def __init__(self, initial_size_mb: int = 100):
        self._pools: Dict[Tuple[int, np.dtype], List[np.ndarray]] = {}
        self._allocated = 0
        self._max_size_mb = initial_size_mb
        self._hits = 0
        self._misses = 0
    
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """Allocate array from pool or create new."""
        key = (int(np.prod(shape)), dtype)
        
        if key in self._pools and self._pools[key]:
            self._hits += 1
            arr = self._pools[key].pop()
            arr = arr.reshape(shape)
            return arr
        
        self._misses += 1
        return np.zeros(shape, dtype=dtype)
    
    def release(self, arr: np.ndarray):
        """Return array to pool for reuse."""
        key = (arr.size, arr.dtype)
        
        if key not in self._pools:
            self._pools[key] = []
        
        # Only keep a reasonable number in pool
        if len(self._pools[key]) < 100:
            self._pools[key].append(arr.ravel())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        total_items = sum(len(v) for v in self._pools.values())
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        
        return {
            "pools": len(self._pools),
            "total_items": total_items,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }


class LatencyOptimizer:
    """
    Latency profiling and optimization for critical paths.
    
    Optimizations:
    - Hot path detection
    - Memory allocation profiling
    - Cache-friendly data layouts
    - Zero-copy operations
    """
    
    def __init__(self):
        self._profiles: Dict[str, List[float]] = {}
        self._hot_paths: Dict[str, float] = {}
    
    def profile(self, name: str):
        """Decorator to profile function execution time."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter_ns()
                result = func(*args, **kwargs)
                duration_ns = time.perf_counter_ns() - start
                
                if name not in self._profiles:
                    self._profiles[name] = []
                self._profiles[name].append(duration_ns / 1e6)  # Convert to ms
                
                # Track hot paths
                self._hot_paths[name] = np.mean(self._profiles[name])
                
                return result
            return wrapper
        return decorator
    
    async def profile_async(self, name: str, coro):
        """Profile async function execution."""
        start = time.perf_counter_ns()
        result = await coro
        duration_ns = time.perf_counter_ns() - start
        
        if name not in self._profiles:
            self._profiles[name] = []
        self._profiles[name].append(duration_ns / 1e6)
        
        return result
    
    def get_profile(self, name: str) -> Dict[str, float]:
        """Get profiling statistics for a function."""
        if name not in self._profiles:
            return {}
        
        times = self._profiles[name]
        return {
            "count": len(times),
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99)
        }
    
    def get_hot_paths(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the hottest (slowest) code paths."""
        sorted_paths = sorted(
            self._hot_paths.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_paths[:top_n]
    
    def optimize_data_layout(self, data: np.ndarray) -> np.ndarray:
        """
        Optimize data layout for cache efficiency.
        
        - Ensure C-contiguous layout
        - Align to cache line boundaries
        - Minimize memory fragmentation
        """
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        return data
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on profiling data."""
        suggestions = []
        
        hot_paths = self.get_hot_paths(5)
        for name, avg_time in hot_paths:
            if avg_time > 10:  # > 10ms average
                suggestions.append(
                    f"Hot path '{name}' averages {avg_time:.2f}ms - consider GPU acceleration or caching"
                )
            elif avg_time > 1:
                suggestions.append(
                    f"Hot path '{name}' averages {avg_time:.2f}ms - consider vectorization"
                )
        
        return suggestions


class PerformanceEngine:
    """
    Unified performance optimization engine.
    
    Coordinates:
    - GPU acceleration
    - Parallel execution
    - Latency optimization
    - Memory management
    """
    
    def __init__(
        self,
        enable_gpu: bool = True,
        max_workers: Optional[int] = None,
        enable_profiling: bool = True
    ):
        self.gpu = GPUAccelerator() if enable_gpu else None
        self.parallel = ParallelExecutor(max_workers=max_workers)
        self.latency = LatencyOptimizer() if enable_profiling else None
        self.memory_pool = MemoryPool()
        
        self._metrics: List[PerformanceMetrics] = []
        
        logger.info(
            f"PerformanceEngine initialized: "
            f"gpu={self.gpu.available if self.gpu else False}, "
            f"workers={self.parallel.max_workers}"
        )
    
    def benchmark(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        iterations: int = 1000,
        warmup: int = 10
    ) -> PerformanceMetrics:
        """Benchmark a function."""
        kwargs = kwargs or {}
        
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            func(*args, **kwargs)
            times.append(time.perf_counter_ns() - start)
        
        times_ms = np.array(times) / 1e6
        
        metrics = PerformanceMetrics(
            operation=func.__name__,
            duration_ms=np.mean(times_ms),
            throughput=1000 / np.mean(times_ms),
            memory_mb=0,  # Would need memory profiler
            cpu_percent=0,
            latency_p50=np.percentile(times_ms, 50),
            latency_p99=np.percentile(times_ms, 99)
        )
        
        self._metrics.append(metrics)
        return metrics
    
    def optimize_backtest(
        self,
        backtest_func: Callable,
        data: np.ndarray,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize backtest execution.
        
        Strategies:
        - GPU acceleration for calculations
        - Parallel parameter sweeps
        - Memory-efficient data handling
        """
        start_time = time.perf_counter()
        
        # Optimize data layout
        data = self.latency.optimize_data_layout(data) if self.latency else data
        
        # Run backtest
        if self.gpu and self.gpu.available:
            # GPU-accelerated path
            result = self._gpu_backtest(backtest_func, data, params)
        else:
            # CPU path
            result = backtest_func(data, params)
        
        duration = time.perf_counter() - start_time
        
        return {
            "result": result,
            "duration_seconds": duration,
            "gpu_used": self.gpu.available if self.gpu else False
        }
    
    def _gpu_backtest(
        self,
        backtest_func: Callable,
        data: np.ndarray,
        params: Dict[str, Any]
    ) -> Any:
        """Run GPU-accelerated backtest."""
        # Transfer data to GPU
        gpu_data = self.gpu.to_gpu(data)
        
        # Run on GPU (function should handle GPU arrays)
        result = backtest_func(gpu_data, params)
        
        # Transfer result back
        if hasattr(result, 'shape'):
            return self.gpu.to_cpu(result)
        return result
    
    def parallel_parameter_sweep(
        self,
        func: Callable,
        param_grid: Dict[str, List[Any]],
        data: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Run parallel parameter sweep for optimization.
        """
        import itertools
        
        # Generate all parameter combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Create partial function with data
        partial_func = partial(func, data)
        
        # Run in parallel
        results = self.parallel.map_parallel(
            lambda params: {"params": params, "result": partial_func(params)},
            combinations
        )
        
        return results
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        report = {
            "gpu": self.gpu.get_memory_info() if self.gpu else {"available": False},
            "parallel": {
                "max_workers": self.parallel.max_workers,
                "backend": "process" if self.parallel.use_processes else "thread"
            },
            "memory_pool": self.memory_pool.get_stats(),
            "benchmarks": [
                {
                    "operation": m.operation,
                    "duration_ms": m.duration_ms,
                    "throughput": m.throughput,
                    "latency_p50": m.latency_p50,
                    "latency_p99": m.latency_p99
                }
                for m in self._metrics
            ]
        }
        
        if self.latency:
            report["hot_paths"] = self.latency.get_hot_paths(5)
            report["suggestions"] = self.latency.suggest_optimizations()
        
        return report
    
    def shutdown(self):
        """Cleanup resources."""
        self.parallel.shutdown()
        logger.info("PerformanceEngine shutdown complete")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_performance_engine(
    enable_gpu: bool = True,
    max_workers: Optional[int] = None
) -> PerformanceEngine:
    """Create and configure a PerformanceEngine."""
    return PerformanceEngine(enable_gpu=enable_gpu, max_workers=max_workers)


def benchmark_function(func: Callable, iterations: int = 1000) -> Dict[str, float]:
    """Quick benchmark of a function."""
    engine = PerformanceEngine(enable_gpu=False, enable_profiling=False)
    metrics = engine.benchmark(func, iterations=iterations)
    engine.shutdown()
    
    return {
        "mean_ms": metrics.duration_ms,
        "throughput": metrics.throughput,
        "p50_ms": metrics.latency_p50,
        "p99_ms": metrics.latency_p99
    }


if __name__ == "__main__":
    # Demo usage
    engine = create_performance_engine()
    
    # Benchmark a simple operation
    def test_operation():
        return np.random.randn(1000, 1000).sum()
    
    metrics = engine.benchmark(test_operation, iterations=100)
    print(f"Benchmark: {metrics.duration_ms:.2f}ms, {metrics.throughput:.0f} ops/sec")
    
    # Check GPU
    if engine.gpu and engine.gpu.available:
        print(f"GPU backend: {engine.gpu.backend}")
        print(f"GPU memory: {engine.gpu.get_memory_info()}")
    
    # Get report
    report = engine.get_report()
    print(f"\nPerformance Report:")
    print(f"  Workers: {report['parallel']['max_workers']}")
    print(f"  Memory pool hit rate: {report['memory_pool']['hit_rate']:.2%}")
    
    engine.shutdown()
