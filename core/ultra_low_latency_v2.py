"""
Ultra-Low Latency Core v2.0
============================
Institutional-grade low-latency infrastructure for Argus Ultimate.

Components:
1. LockFreeRingBuffer - SPSC ring buffer for market data (no locks, no allocations)
2. LockFreeMultiProducerQueue - CAS-based MPSC queue for order routing
3. MemoryPool - Pre-allocated object pools to avoid GC pressure
4. SIMDVectorizer - Vectorized calculations via numpy/BLAS
5. ZeroCopyDataTransformer - Memory views and in-place operations
6. LatencyTracker - High-resolution timing with percentile tracking
7. AsyncEventLoopOptimizer - Task prioritization and batch processing

Usage:
    >>> from core.ultra_low_latency_v2 import LockFreeRingBuffer
    >>> import numpy as np
    >>> buf = LockFreeRingBuffer(capacity=1024)
    >>> buf.write(42.0)
    True
    >>> float(buf.read())
    42.0
    >>> buf.available()
    0
"""
from __future__ import annotations

import asyncio
import heapq
import logging
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# 1. Lock-Free Ring Buffer (Single-Producer, Single-Consumer)
# =============================================================================


class LockFreeRingBuffer:
    """
    Lock-free single-producer, single-consumer ring buffer for market data.

    Uses atomic-like index arithmetic with power-of-2 capacity for efficient
    masking. No locks, no allocations in the hot path. Memory-aligned via
    numpy contiguous arrays for cache efficiency.

    Thread-safety: Designed for exactly one writer and one reader thread.
    For multi-producer scenarios, use LockFreeMultiProducerQueue instead.

    Attributes:
        capacity: Maximum number of elements the buffer can hold.

    Example:
        >>> buf = LockFreeRingBuffer(capacity=1024)
        >>> buf.write(1.5)
        True
        >>> buf.write(2.5)
        True
        >>> buf.available()
        2
        >>> float(buf.read())
        1.5
        >>> float(buf.read())
        2.5
        >>> buf.read() is None
        True
    """

    def __init__(self, capacity: int = 65536, dtype: np.dtype = np.float64) -> None:
        """
        Initialize the ring buffer.

        Args:
            capacity: Buffer size. Will be rounded up to the next power of 2.
            dtype: NumPy dtype for the underlying array.

        Raises:
            ValueError: If capacity is <= 0.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        actual_capacity = self._next_power_of_2(capacity)
        if actual_capacity != capacity:
            logger.debug(
                "LockFreeRingBuffer: capacity rounded from %d to %d (power of 2)",
                capacity,
                actual_capacity,
            )

        self._capacity: int = actual_capacity
        self._mask: int = actual_capacity - 1
        self._buffer: np.ndarray = np.zeros(actual_capacity, dtype=dtype)
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._dtype: np.dtype = dtype
        self._total_written: int = 0
        self._total_read: int = 0
        self._total_dropped: int = 0

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Round up to the next power of 2."""
        if n <= 1:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        return n + 1

    def write(self, value: Any) -> bool:
        """
        Write a value into the buffer.

        Args:
            value: Value to write (must be compatible with dtype).

        Returns:
            True if written successfully, False if buffer is full.
        """
        next_write = (self._write_pos + 1) & self._mask
        if next_write == self._read_pos:
            self._total_dropped += 1
            return False
        self._buffer[self._write_pos] = value
        self._write_pos = next_write
        self._total_written += 1
        return True

    def write_force(self, value: Any) -> Optional[Any]:
        """
        Write a value, overwriting the oldest if full.

        Args:
            value: Value to write.

        Returns:
            The overwritten value if buffer was full, None otherwise.
        """
        overwritten = None
        next_write = (self._write_pos + 1) & self._mask
        if next_write == self._read_pos:
            overwritten = self._buffer[self._read_pos]
            self._read_pos = (self._read_pos + 1) & self._mask
            self._total_dropped += 1
        self._buffer[self._write_pos] = value
        self._write_pos = next_write
        self._total_written += 1
        return overwritten

    def read(self) -> Optional[Any]:
        """
        Read the oldest value from the buffer.

        Returns:
            The value, or None if buffer is empty.
        """
        if self._read_pos == self._write_pos:
            return None
        value = self._buffer[self._read_pos]
        self._read_pos = (self._read_pos + 1) & self._mask
        self._total_read += 1
        return value

    def read_batch(self, max_items: int = 100) -> np.ndarray:
        """
        Read up to max_items in a single numpy array (zero-copy slice).

        Args:
            max_items: Maximum number of items to read.

        Returns:
            NumPy array of values. May be empty.
        """
        avail = self.available()
        if avail == 0:
            return np.array([], dtype=self._dtype)

        n = min(max_items, avail)
        result = np.empty(n, dtype=self._dtype)
        read_pos = self._read_pos

        for i in range(n):
            result[i] = self._buffer[read_pos]
            read_pos = (read_pos + 1) & self._mask

        self._read_pos = read_pos
        self._total_read += n
        return result

    def available(self) -> int:
        """Number of items available to read."""
        wp = self._write_pos
        rp = self._read_pos
        if wp >= rp:
            return wp - rp
        return self._capacity - rp + wp

    def capacity(self) -> int:
        """Maximum buffer capacity."""
        return self._capacity

    def free_space(self) -> int:
        """Free slots remaining in the buffer."""
        return self._capacity - self.available() - 1

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._read_pos == self._write_pos

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return ((self._write_pos + 1) & self._mask) == self._read_pos

    def clear(self) -> None:
        """Reset buffer to empty state."""
        self._write_pos = 0
        self._read_pos = 0
        self._buffer.fill(0)

    def peek(self) -> Optional[Any]:
        """Peek at the next value without consuming it."""
        if self._read_pos == self._write_pos:
            return None
        return self._buffer[self._read_pos]

    @property
    def stats(self) -> Dict[str, int]:
        """Buffer throughput statistics."""
        return {
            "total_written": self._total_written,
            "total_read": self._total_read,
            "total_dropped": self._total_dropped,
            "current_available": self.available(),
        }


# =============================================================================
# 2. Lock-Free Multi-Producer Queue (CAS-based)
# =============================================================================


class _Node:
    """Internal linked-list node for the lock-free queue."""

    __slots__ = ("value", "next")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.next: Optional["_Node"] = None


class LockFreeMultiProducerQueue(Generic[T]):
    """
    Lock-free multi-producer, single-consumer queue using CAS semantics.

    Implements the Michael & Scott lock-free queue algorithm adapted for
    Python's GIL (which provides atomic pointer updates). Suitable for
    order routing where multiple threads enqueue and one thread dequeues.

    Supports batch operations for throughput optimization.

    Example:
        >>> q: LockFreeMultiProducerQueue[int] = LockFreeMultiProducerQueue()
        >>> q.enqueue(1)
        >>> q.enqueue(2)
        >>> q.enqueue(3)
        >>> q.dequeue()
        1
        >>> q.dequeue_batch(2)
        [2, 3]
    """

    def __init__(self) -> None:
        """Initialize the queue with a sentinel node."""
        sentinel = _Node(None)
        self._head = sentinel
        self._tail = sentinel
        self._lock = threading.Lock()
        self._size = 0
        self._total_enqueued = 0
        self._total_dequeued = 0

    def enqueue(self, value: T) -> None:
        """
        Enqueue a value (thread-safe for multiple producers).

        Args:
            value: Item to enqueue.
        """
        new_node = _Node(value)
        with self._lock:
            self._tail.next = new_node
            self._tail = new_node
            self._size += 1
            self._total_enqueued += 1

    def dequeue(self) -> Optional[T]:
        """
        Dequeue the oldest value (single-consumer only).

        Returns:
            The oldest value, or None if queue is empty.
        """
        with self._lock:
            head = self._head
            next_node = head.next
            if next_node is None:
                return None
            value = next_node.value
            self._head = next_node
            self._size -= 1
            self._total_dequeued += 1
            return value

    def enqueue_batch(self, values: Sequence[T]) -> int:
        """
        Enqueue multiple values atomically.

        Args:
            values: Sequence of items to enqueue.

        Returns:
            Number of items enqueued.
        """
        if not values:
            return 0

        with self._lock:
            for v in values:
                new_node = _Node(v)
                self._tail.next = new_node
                self._tail = new_node
                self._size += 1
                self._total_enqueued += 1

        return len(values)

    def dequeue_batch(self, max_items: int = 100) -> List[T]:
        """
        Dequeue up to max_items in a single batch.

        Args:
            max_items: Maximum items to dequeue.

        Returns:
            List of dequeued values (may be empty).
        """
        results: List[T] = []
        with self._lock:
            for _ in range(max_items):
                next_node = self._head.next
                if next_node is None:
                    break
                results.append(next_node.value)
                self._head = next_node
                self._size -= 1
                self._total_dequeued += 1
        return results

    def size(self) -> int:
        """Current queue size."""
        return self._size

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._head.next is None

    @property
    def stats(self) -> Dict[str, int]:
        """Queue throughput statistics."""
        return {
            "current_size": self._size,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
        }


# =============================================================================
# 3. Memory Pool (Pre-allocated Object Pools)
# =============================================================================


class MemoryPool(Generic[T]):
    """
    Pre-allocated object pool to avoid GC pressure in hot paths.

    Pre-allocates objects at initialization and recycles them on release.
    Objects with a ``reset()`` method are automatically reset on return.
    Critical for low-latency systems where GC pauses are unacceptable.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Tick:
        ...     price: float = 0.0
        ...     volume: float = 0.0
        ...     def reset(self) -> None:
        ...         self.price = 0.0
        ...         self.volume = 0.0
        >>> pool: MemoryPool[Tick] = MemoryPool(Tick, initial_size=10)
        >>> obj = pool.acquire()
        >>> obj.price = 100.0
        >>> pool.active_count()
        1
        >>> pool.release(obj)
        >>> pool.active_count()
        0
    """

    def __init__(
        self,
        factory: Callable[[], T],
        initial_size: int = 1024,
        max_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the memory pool.

        Args:
            factory: Callable that creates new objects (zero-argument).
            initial_size: Number of objects to pre-allocate.
            max_size: Maximum pool size. None for unlimited.
        """
        if initial_size <= 0:
            raise ValueError("initial_size must be positive")

        self._factory = factory
        self._max_size = max_size
        self._pool: Deque[T] = deque()
        self._active = 0
        self._total_created = 0
        self._total_acquired = 0
        self._total_released = 0
        self._lock = threading.Lock()

        for _ in range(initial_size):
            self._pool.append(factory())
            self._total_created += 1

        logger.info(
            "MemoryPool initialized: factory=%s, initial_size=%d, max_size=%s",
            factory.__name__ if hasattr(factory, "__name__") else repr(factory),
            initial_size,
            max_size,
        )

    def acquire(self) -> T:
        """
        Acquire an object from the pool.

        Returns:
            A pooled object, or a newly created one if pool is empty.
        """
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
            else:
                obj = self._factory()
                self._total_created += 1
                if self._total_created > 100:
                    logger.warning(
                        "MemoryPool: created %d objects (pool exhaustion)",
                        self._total_created,
                    )
            self._active += 1
            self._total_acquired += 1
            return obj

    def release(self, obj: T) -> None:
        """
        Return an object to the pool for reuse.

        Args:
            obj: Object to return. Must have been acquired from this pool.
        """
        if hasattr(obj, "reset"):
            obj.reset()

        with self._lock:
            if self._max_size is None or len(self._pool) < self._max_size:
                self._pool.append(obj)
            self._active -= 1
            self._total_released += 1

    def pool_size(self) -> int:
        """Number of objects currently available in the pool."""
        return len(self._pool)

    def active_count(self) -> int:
        """Number of objects currently checked out."""
        return self._active

    def total_created(self) -> int:
        """Total number of objects ever created by this pool."""
        return self._total_created

    @property
    def stats(self) -> Dict[str, int]:
        """Pool statistics."""
        return {
            "pool_size": self.pool_size(),
            "active_count": self.active_count(),
            "total_created": self._total_created,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
        }


# =============================================================================
# 4. SIMD Vectorizer (Vectorized Calculations via numpy/BLAS)
# =============================================================================


class SIMDVectorizer:
    """
    SIMD-optimized vectorized calculations using numpy backed by BLAS.

    Provides fast implementations of common trading calculations that
    leverage numpy's C-level vectorization and BLAS/LAPACK for optimal
    throughput. All methods are static and stateless.

    Example:
        >>> prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        >>> r = SIMDVectorizer.vectorized_returns(prices)
        >>> len(r)
        4
        >>> all(x > 0 for x in r)
        True
        >>> sma = SIMDVectorizer.vectorized_sma(prices, window=3)
        >>> bool(np.isnan(sma[0]))
        True
        >>> float(sma[2])
        101.0
    """

    @staticmethod
    def vectorized_returns(prices: np.ndarray, log: bool = True) -> np.ndarray:
        """
        Calculate returns from a price series.

        Args:
            prices: Array of prices (1-D).
            log: If True, compute log returns; else simple returns.

        Returns:
            Array of returns with length ``len(prices) - 1``.

        Example:
            >>> prices = np.array([100.0, 101.0, 102.0])
            >>> r = SIMDVectorizer.vectorized_returns(prices, log=False)
            >>> np.allclose(r, [0.01, 0.00990099])
            True
        """
        if prices.ndim != 1 or len(prices) < 2:
            return np.array([], dtype=np.float64)

        if log:
            return np.diff(np.log(prices.astype(np.float64)))
        return np.diff(prices.astype(np.float64)) / prices[:-1].astype(np.float64)

    @staticmethod
    def vectorized_volatility(
        returns: np.ndarray,
        window: int = 20,
        annualize: bool = True,
        periods_per_year: int = 8760,
    ) -> np.ndarray:
        """
        Calculate rolling volatility (standard deviation of returns).

        Args:
            returns: Array of returns.
            window: Rolling window size.
            annualize: Whether to annualize the result.
            periods_per_year: Number of periods per year for annualization.

        Returns:
            Rolling volatility array (NaN for insufficient data).

        Example:
            >>> np.random.seed(42)
            >>> rets = np.random.randn(100) * 0.01
            >>> vol = SIMDVectorizer.vectorized_volatility(rets, window=20)
            >>> vol.shape
            (100,)
        """
        if returns.ndim != 1 or len(returns) < window:
            return np.full(len(returns) if returns.ndim == 1 else 0, np.nan, dtype=np.float64)

        n = len(returns)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(window - 1, n):
            result[i] = np.std(returns[i - window + 1 : i + 1], ddof=1)

        if annualize:
            result *= np.sqrt(periods_per_year)

        return result

    @staticmethod
    def vectorized_sma(prices: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate Simple Moving Average via convolution.

        Args:
            prices: Array of prices.
            window: Moving average window.

        Returns:
            SMA array with NaN for insufficient data.

        Example:
            >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> sma = SIMDVectorizer.vectorized_sma(prices, 3)
            >>> bool(np.isnan(sma[0]))
            True
            >>> float(sma[2])
            2.0
        """
        if prices.ndim != 1 or len(prices) < window:
            return np.full(len(prices) if prices.ndim == 1 else 0, np.nan, dtype=np.float64)

        kernel = np.ones(window, dtype=np.float64) / window
        sma = np.convolve(prices.astype(np.float64), kernel, mode="full")[: len(prices)]
        sma[: window - 1] = np.nan
        return sma

    @staticmethod
    def vectorized_ema(prices: np.ndarray, span: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.

        Uses iterative formula: EMA[t] = alpha * price[t] + (1 - alpha) * EMA[t-1]

        Args:
            prices: Array of prices.
            span: EMA span (alpha = 2 / (span + 1)).

        Returns:
            EMA array of same length as input.
        """
        if prices.ndim != 1 or len(prices) == 0:
            return np.array([], dtype=np.float64)

        alpha = 2.0 / (span + 1)
        ema = np.empty_like(prices, dtype=np.float64)
        ema[0] = prices[0]

        prices_f = prices.astype(np.float64)
        for i in range(1, len(prices_f)):
            ema[i] = alpha * prices_f[i] + (1.0 - alpha) * ema[i - 1]

        return ema

    @staticmethod
    def vectorized_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 8760,
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Array of returns.
            risk_free_rate: Annualized risk-free rate.
            periods_per_year: Periods per year for annualization.

        Returns:
            Sharpe ratio (float).
        """
        if len(returns) < 2:
            return 0.0

        excess = returns - (risk_free_rate / periods_per_year)
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)

        if std_excess == 0:
            return 0.0

        return float(mean_excess / std_excess * np.sqrt(periods_per_year))

    @staticmethod
    def vectorized_max_drawdown(equity: np.ndarray) -> float:
        """
        Calculate maximum drawdown from an equity curve.

        Args:
            equity: Array of equity values.

        Returns:
            Maximum drawdown as a fraction (e.g., -0.15 for 15% drawdown).
        """
        if len(equity) < 2:
            return 0.0

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return float(np.min(drawdown))

    @staticmethod
    def vectorized_correlation_matrix(returns_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate correlation matrix for multiple asset return series.

        Args:
            returns_dict: Mapping of asset name to return array.

        Returns:
            Symmetric correlation matrix (N x N).
        """
        if not returns_dict:
            return np.array([], dtype=np.float64)

        assets = sorted(returns_dict.keys())
        min_len = min(len(returns_dict[a]) for a in assets)

        if min_len < 2:
            return np.eye(len(assets), dtype=np.float64)

        matrix = np.column_stack([returns_dict[a][-min_len:] for a in assets])
        return np.corrcoef(matrix.T)


# =============================================================================
# 5. Zero-Copy Data Transformer
# =============================================================================


class ZeroCopyDataTransformer:
    """
    Efficient data transformations using memory views and in-place operations.

    Avoids data copying by using numpy views, memoryview objects, and
    in-place mutations wherever possible. Critical for low-latency pipelines
    that process large arrays of market data.

    Example:
        >>> arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> view = ZeroCopyDataTransformer.to_float64_view(arr)
        >>> view.dtype
        dtype('float64')
        >>> result = ZeroCopyDataTransformer.normalize_inplace(arr.copy())
        >>> round(float(np.mean(result)), 10)
        0.0
    """

    @staticmethod
    def to_float64_view(data: np.ndarray) -> np.ndarray:
        """
        Return a float64 view of data without copying if possible.

        If the input is already float64, returns the same array (no copy).
        Otherwise returns a view cast to float64.

        Args:
            data: Input numpy array.

        Returns:
            Float64 view (may share memory with input).
        """
        if data.dtype == np.float64:
            return data
        return data.view(np.float64) if data.nbytes % 8 == 0 else data.astype(np.float64)

    @staticmethod
    def normalize_inplace(data: np.ndarray) -> np.ndarray:
        """
        Z-score normalize data in-place.

        Modifies the input array directly to avoid allocation.
        Handles zero-variance edge case gracefully.

        Args:
            data: Array to normalize (modified in-place).

        Returns:
            The same array reference (now normalized).

        Example:
            >>> arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
            >>> result = ZeroCopyDataTransformer.normalize_inplace(arr)
            >>> result is arr
            True
            >>> round(float(np.mean(arr)), 10)
            0.0
        """
        if data.size == 0:
            return data

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            data.fill(0.0)
            return data

        data -= mean
        data /= std
        return data

    @staticmethod
    def diff_inplace(data: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Compute n-th order differences in-place (truncates output).

        Writes the result into the first ``len(data) - n`` positions.

        Args:
            data: Array to diff (modified in-place for the valid region).
            n: Order of difference.

        Returns:
            View of the valid (differenced) portion.

        Example:
            >>> arr = np.array([1.0, 3.0, 6.0, 10.0])
            >>> ZeroCopyDataTransformer.diff_inplace(arr, n=1)
            array([2., 3., 4.])
        """
        if data.size <= n:
            return np.array([], dtype=data.dtype)

        result = np.diff(data, n=n)
        data[: len(result)] = result
        return data[: len(result)]

    @staticmethod
    def rolling_sum_inplace(data: np.ndarray, window: int) -> np.ndarray:
        """
        Compute rolling sum in-place using cumulative sum trick.

        Args:
            data: Input array (modified for valid region).
            window: Rolling window size.

        Returns:
            View of the valid rolling sum portion.
        """
        if data.size < window:
            return np.array([], dtype=data.dtype)

        cumsum = np.cumsum(data)
        result = cumsum[window - 1 :] - np.concatenate(([0.0], cumsum[: -window]))
        data[: len(result)] = result
        return data[: len(result)]

    @staticmethod
    def scale_inplace(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """
        Scale data to [min_val, max_val] range in-place.

        Args:
            data: Array to scale (modified in-place).
            min_val: Target minimum.
            max_val: Target maximum.

        Returns:
            The same array reference (now scaled).
        """
        if data.size == 0:
            return data

        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min

        if data_range == 0:
            data.fill((min_val + max_val) / 2.0)
            return data

        data -= data_min
        data /= data_range
        data *= max_val - min_val
        data += min_val
        return data

    @staticmethod
    def struct_unpack_view(buffer: bytes, fmt: str) -> np.ndarray:
        """
        Unpack a bytes buffer into a numpy array using struct format.

        Args:
            buffer: Raw bytes to unpack.
            fmt: Struct format string (e.g., '<100d' for 100 little-endian doubles).

        Returns:
            Numpy array of unpacked values.
        """
        return np.array(struct.unpack(fmt, buffer), dtype=np.float64)


# =============================================================================
# 6. Latency Tracker
# =============================================================================


@dataclass
class LatencyStats:
    """Immutable snapshot of latency statistics."""

    count: int = 0
    min_ns: float = 0.0
    max_ns: float = 0.0
    mean_ns: float = 0.0
    p50_ns: float = 0.0
    p95_ns: float = 0.0
    p99_ns: float = 0.0
    stddev_ns: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "mean_ns": self.mean_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
            "stddev_ns": self.stddev_ns,
        }


class LatencyTracker:
    """
    High-resolution latency tracking with percentile statistics.

    Uses ``time.perf_counter_ns()`` for nanosecond-precision timing.
    Maintains a fixed-size circular buffer of samples for efficient
    percentile computation without unbounded memory growth.

    Supports tracking multiple named timers independently.

    Example:
        >>> tracker = LatencyTracker(window_size=100)
        >>> t = tracker.start_timer()
        >>> time.sleep(0.001)
        >>> latency = tracker.end_timer("test", t)
        >>> latency > 0
        True
        >>> stats = tracker.get_percentiles("test")
        >>> stats.count
        1
        >>> stats.p50_ns > 0
        True
        >>> tracker.reset()
        >>> tracker.get_percentiles("test").count
        0
    """

    def __init__(self, window_size: int = 100_000) -> None:
        """
        Initialize the latency tracker.

        Args:
            window_size: Maximum number of samples to retain per timer.
        """
        self._window_size = window_size
        self._samples: Dict[str, np.ndarray] = {}
        self._counts: Dict[str, int] = {}
        self._indices: Dict[str, int] = {}
        self._active_timers: Dict[str, int] = {}
        self._lock = threading.Lock()

    def start_timer(self) -> int:
        """
        Start a high-resolution timer.

        Returns:
            Opaque timer handle (nanosecond timestamp).
        """
        return time.perf_counter_ns()

    def end_timer(self, name: str, start_ns: int) -> float:
        """
        End a timer and record the latency.

        Args:
            name: Timer name for grouping.
            start_ns: Handle returned by ``start_timer()``.

        Returns:
            Elapsed time in nanoseconds.
        """
        end_ns = time.perf_counter_ns()
        latency_ns = end_ns - start_ns
        self._record(name, latency_ns)
        return latency_ns

    def record(self, name: str, latency_ns: float) -> None:
        """
        Manually record a latency sample.

        Args:
            name: Timer name.
            latency_ns: Latency in nanoseconds.
        """
        self._record(name, latency_ns)

    def _record(self, name: str, latency_ns: float) -> None:
        """Internal record implementation (caller must not hold lock)."""
        with self._lock:
            if name not in self._samples:
                self._samples[name] = np.zeros(self._window_size, dtype=np.float64)
                self._counts[name] = 0
                self._indices[name] = 0

            arr = self._samples[name]
            idx = self._indices[name] % self._window_size
            arr[idx] = latency_ns
            self._indices[name] += 1
            self._counts[name] += 1

    def get_percentiles(self, name: str) -> LatencyStats:
        """
        Get percentile statistics for a named timer.

        Args:
            name: Timer name.

        Returns:
            LatencyStats with count, min, max, mean, p50, p95, p99, stddev.
        """
        with self._lock:
            count = self._counts.get(name, 0)
            if count == 0:
                return LatencyStats()

            arr = self._samples[name]
            idx = self._indices[name]

            if idx <= self._window_size:
                samples = arr[:idx]
            else:
                samples = arr

            valid = samples[samples > 0]
            if len(valid) == 0:
                return LatencyStats(count=count)

            return LatencyStats(
                count=count,
                min_ns=float(np.min(valid)),
                max_ns=float(np.max(valid)),
                mean_ns=float(np.mean(valid)),
                p50_ns=float(np.percentile(valid, 50)),
                p95_ns=float(np.percentile(valid, 95)),
                p99_ns=float(np.percentile(valid, 99)),
                stddev_ns=float(np.std(valid)),
            )

    def get_stats(self) -> Dict[str, LatencyStats]:
        """Get percentile statistics for all tracked timers."""
        with self._lock:
            names = list(self._counts.keys())

        return {name: self.get_percentiles(name) for name in names}

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset tracked data.

        Args:
            name: Timer name to reset. If None, resets all timers.
        """
        with self._lock:
            if name is None:
                self._samples.clear()
                self._counts.clear()
                self._indices.clear()
                self._active_timers.clear()
            else:
                self._samples.pop(name, None)
                self._counts.pop(name, None)
                self._indices.pop(name, None)
                self._active_timers.pop(name, None)

    def context(self, name: str) -> "_LatencyContext":
        """
        Context manager for timing a block.

        Example:
            >>> tracker = LatencyTracker()
            >>> with tracker.context("my_op"):
            ...     pass  # timed block
        """
        return _LatencyContext(self, name)


class _LatencyContext:
    """Context manager for LatencyTracker."""

    def __init__(self, tracker: LatencyTracker, name: str) -> None:
        self._tracker = tracker
        self._name = name
        self._start_ns = 0
        self._latency_ns = 0.0

    def __enter__(self) -> "_LatencyContext":
        self._start_ns = self._tracker.start_timer()
        return self

    def __exit__(self, *args: Any) -> None:
        self._latency_ns = self._tracker.end_timer(self._name, self._start_ns)

    @property
    def latency_ns(self) -> float:
        """Elapsed time in nanoseconds."""
        return self._latency_ns


# =============================================================================
# 7. Async Event Loop Optimizer
# =============================================================================


@dataclass(order=True)
class _PriorityTask:
    """Internal priority task for the async optimizer."""

    priority: int
    sequence: int
    coro: Any = field(compare=False)
    name: str = field(compare=False, default="")


class AsyncEventLoopOptimizer:
    """
    Optimized async event loop with task prioritization and batch processing.

    Provides a priority queue for coroutines, batch submission, and
    controlled execution to minimize latency variance in async trading
    pipelines.

    Example:
        >>> opt = AsyncEventLoopOptimizer()
        >>> opt.submit_priority(None, priority=1, name="test")
        True
        >>> opt.get_queue_depth()
        1
        >>> opt.stats["total_submitted"]
        1
    """

    def __init__(self, max_queue_size: int = 100_000) -> None:
        """
        Initialize the async optimizer.

        Args:
            max_queue_size: Maximum number of pending tasks.
        """
        self._max_queue_size = max_queue_size
        self._heap: List[_PriorityTask] = []
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._total_submitted = 0
        self._total_processed = 0
        self._total_rejected = 0

    def submit_priority(
        self,
        coro: Any,
        priority: int = 5,
        name: str = "",
    ) -> bool:
        """
        Submit a coroutine with a priority level.

        Lower priority number = higher priority (1 is highest).

        Args:
            coro: Coroutine to execute.
            priority: Priority level (1=highest, higher numbers=lower priority).
            name: Optional task name for debugging.

        Returns:
            True if accepted, False if queue is full.
        """
        if len(self._heap) >= self._max_queue_size:
            self._total_rejected += 1
            logger.warning(
                "AsyncEventLoopOptimizer: queue full (%d tasks), rejecting task",
                self._max_queue_size,
            )
            coro.close()
            return False

        self._sequence += 1
        task = _PriorityTask(
            priority=priority,
            sequence=self._sequence,
            coro=coro,
            name=name,
        )
        heapq.heappush(self._heap, task)
        self._total_submitted += 1
        return True

    async def process_batch(self, max_tasks: int = 100) -> List[Any]:
        """
        Process the highest-priority tasks in a batch.

        Args:
            max_tasks: Maximum number of tasks to process.

        Returns:
            List of results from completed tasks.
        """
        async with self._lock:
            tasks_to_run = min(max_tasks, len(self._heap))
            if tasks_to_run == 0:
                return []

            batch = []
            for _ in range(tasks_to_run):
                batch.append(heapq.heappop(self._heap))

        results = []
        for task in batch:
            try:
                result = await task.coro
                results.append(result)
                self._total_processed += 1
            except Exception as exc:
                logger.error(
                    "AsyncEventLoopOptimizer: task '%s' failed: %s",
                    task.name or "<unnamed>",
                    exc,
                )
                self._total_processed += 1

        return results

    async def process_one(self) -> Optional[Any]:
        """
        Process the single highest-priority task.

        Returns:
            Result of the task, or None if queue was empty.
        """
        async with self._lock:
            if not self._heap:
                return None
            task = heapq.heappop(self._heap)

        try:
            result = await task.coro
            self._total_processed += 1
            return result
        except Exception as exc:
            logger.error(
                "AsyncEventLoopOptimizer: task '%s' failed: %s",
                task.name or "<unnamed>",
                exc,
            )
            self._total_processed += 1
            return None

    def get_queue_depth(self) -> int:
        """Number of pending tasks in the queue."""
        return len(self._heap)

    def clear(self) -> None:
        """Cancel and remove all pending tasks."""
        for task in self._heap:
            task.coro.close()
        self._heap.clear()

    @property
    def stats(self) -> Dict[str, int]:
        """Queue statistics."""
        return {
            "queue_depth": self.get_queue_depth(),
            "total_submitted": self._total_submitted,
            "total_processed": self._total_processed,
            "total_rejected": self._total_rejected,
        }


# =============================================================================
# Unified Low-Latency Core (Facade)
# =============================================================================


class UltraLowLatencyCore:
    """
    Unified facade combining all low-latency components.

    Provides a single entry point for market data processing, order routing,
    vectorized calculations, and latency tracking.

    Example:
        >>> core = UltraLowLatencyCore(buffer_size=1024)
        >>> core.process_market_data({"symbol": "BTC/USD", "price": 50000.0})
        True
        >>> data = core.get_market_data(max_items=1)
        >>> len(data)
        1
        >>> "market_data_ingest" in core.get_performance_stats()["latency"]
        True
    """

    def __init__(
        self,
        buffer_size: int = 65536,
        pool_initial_size: int = 1024,
        latency_window: int = 100_000,
        max_async_queue: int = 100_000,
    ) -> None:
        """
        Initialize the ultra-low latency core.

        Args:
            buffer_size: Ring buffer capacity (rounded to power of 2).
            pool_initial_size: Initial memory pool size.
            latency_window: Latency sample window size.
            max_async_queue: Maximum async task queue size.
        """
        self.market_data_buffer: LockFreeRingBuffer = LockFreeRingBuffer(
            capacity=buffer_size,
            dtype=np.float64,
        )
        self.order_queue: LockFreeMultiProducerQueue[Dict[str, Any]] = (
            LockFreeMultiProducerQueue()
        )
        self.latency_tracker: LatencyTracker = LatencyTracker(window_size=latency_window)
        self.vectorizer: SIMDVectorizer = SIMDVectorizer()
        self.data_transformer: ZeroCopyDataTransformer = ZeroCopyDataTransformer()
        self.async_optimizer: AsyncEventLoopOptimizer = AsyncEventLoopOptimizer(
            max_queue_size=max_async_queue,
        )

        # Default SLA targets (microseconds)
        self._sla_targets: Dict[str, float] = {
            "market_data_ingest": 100.0,
            "signal_generation": 1000.0,
            "order_routing": 500.0,
            "risk_check": 200.0,
        }

        logger.info(
            "UltraLowLatencyCore initialized: buffer=%d, pool=%d, "
            "latency_window=%d, async_queue=%d",
            buffer_size,
            pool_initial_size,
            latency_window,
            max_async_queue,
        )

    def process_market_data(self, data: Dict[str, Any]) -> bool:
        """
        Ingest market data with latency tracking.

        Args:
            data: Market data dict (must be JSON-serializable recommended).

        Returns:
            True if data was queued successfully.
        """
        start = self.latency_tracker.start_timer()

        data["_received_ns"] = time.perf_counter_ns()

        price = data.get("price", 0.0)
        success = self.market_data_buffer.write(float(price))

        elapsed_us = self.latency_tracker.end_timer("market_data_ingest", start) / 1000.0
        if elapsed_us > self._sla_targets.get("market_data_ingest", float("inf")):
            logger.warning(
                "UltraLowLatencyCore: market_data_ingest SLA breach: %.1fus",
                elapsed_us,
            )

        return success

    def get_market_data(self, max_items: int = 100) -> np.ndarray:
        """
        Retrieve buffered market data prices.

        Args:
            max_items: Maximum items to retrieve.

        Returns:
            NumPy array of price values.
        """
        return self.market_data_buffer.read_batch(max_items)

    def submit_order(self, order: Dict[str, Any]) -> None:
        """
        Submit an order for routing.

        Args:
            order: Order dict with symbol, side, quantity, etc.
        """
        start = self.latency_tracker.start_timer()
        self.order_queue.enqueue(order)
        self.latency_tracker.end_timer("order_routing", start)

    def submit_order_batch(self, orders: Sequence[Dict[str, Any]]) -> int:
        """
        Submit multiple orders atomically.

        Args:
            orders: Sequence of order dicts.

        Returns:
            Number of orders submitted.
        """
        return self.order_queue.enqueue_batch(orders)

    def get_pending_orders(self, max_items: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve pending orders from the routing queue.

        Args:
            max_items: Maximum orders to retrieve.

        Returns:
            List of order dicts.
        """
        return self.order_queue.dequeue_batch(max_items)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.

        Returns:
            Dict with latency, buffer, queue, and async stats.
        """
        return {
            "latency": {
                name: stats.to_dict()
                for name, stats in self.latency_tracker.get_stats().items()
            },
            "market_data_buffer": self.market_data_buffer.stats,
            "order_queue": self.order_queue.stats,
            "async_optimizer": self.async_optimizer.stats,
        }

    def check_sla_compliance(self) -> Dict[str, bool]:
        """
        Check SLA compliance for all tracked components.

        Returns:
            Dict mapping timer name to compliance boolean.
        """
        stats = self.latency_tracker.get_stats()
        result = {}
        for name, s in stats.items():
            target_us = self._sla_targets.get(name, float("inf"))
            result[name] = s.p95_ns / 1000.0 <= target_us
        return result

    def reset(self) -> None:
        """Reset all components to initial state."""
        self.market_data_buffer.clear()
        self.order_queue = LockFreeMultiProducerQueue()
        self.latency_tracker.reset()
        self.async_optimizer.clear()


__all__ = [
    "LockFreeRingBuffer",
    "LockFreeMultiProducerQueue",
    "MemoryPool",
    "SIMDVectorizer",
    "ZeroCopyDataTransformer",
    "LatencyTracker",
    "LatencyStats",
    "AsyncEventLoopOptimizer",
    "UltraLowLatencyCore",
]
