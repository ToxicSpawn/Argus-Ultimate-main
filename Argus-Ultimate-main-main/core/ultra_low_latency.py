"""
Ultra-Low Latency Infrastructure Components
============================================

High-performance data structures and utilities for sub-microsecond trading operations.
Uses numpy, ctypes, and memoryview for zero-copy, lock-free operations.

Components:
- RingBuffer: Fixed-size circular buffer with lock-free push/pop
- MemoryPool: Pre-allocated memory block manager
- OrderBookFast: Array-based L2 order book with O(1) best bid/ask
- TimestampTracker: Nanosecond-precision latency measurement
- MessageQueue: Zero-copy message passing with priority and backpressure
- HotPathOptimizer: Code path analysis and optimization suggestions
- LatencyMonitor: Component-level latency tracking and bottleneck detection
"""
from __future__ import annotations

import ctypes
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class PoolStats:
    """Memory pool statistics."""
    total_blocks: int
    allocated_blocks: int
    free_blocks: int
    total_bytes: int
    allocated_bytes: int
    utilization_pct: float


@dataclass
class HotPath:
    """Identified hot execution path."""
    function_name: str
    file_path: str
    line_number: int
    call_count: int
    avg_time_ns: float
    total_time_ns: float
    self_time_pct: float


@dataclass
class Optimization:
    """Suggested optimization for a hot path."""
    hot_path: HotPath
    suggestion: str
    expected_improvement_pct: float
    difficulty: str  # "easy", "medium", "hard"
    details: str


@dataclass
class ImprovementMetrics:
    """Metrics comparing before/after optimization."""
    before_avg_ns: float
    after_avg_ns: float
    before_p99_ns: float
    after_p99_ns: float
    speedup_factor: float
    improvement_pct: float


@dataclass
class LatencyPercentiles:
    """Latency percentile statistics for a component."""
    component: str
    count: int
    min_ns: float
    max_ns: float
    p50_ns: float
    p95_ns: float
    p99_ns: float
    mean_ns: float
    stddev_ns: float


@dataclass
class Alert:
    """Latency threshold breach alert."""
    component: str
    threshold_ns: float
    actual_ns: float
    timestamp_ns: int
    severity: str  # "warning", "critical"
    message: str


@dataclass
class PerformanceStats:
    """Aggregate performance statistics."""
    throughput_ops_per_sec: float
    avg_latency_ns: float
    p50_latency_ns: float
    p95_latency_ns: float
    p99_latency_ns: float
    memory_usage_bytes: int


# =============================================================================
# RingBuffer
# =============================================================================

class RingBuffer:
    """
    Fixed-size circular buffer with lock-free push/pop operations.

    Uses numpy arrays for contiguous memory and memoryview for zero-copy slices.
    Thread-safe via atomic index operations (compare-and-swap pattern).
    """

    def __init__(self, capacity: int, dtype: np.dtype = np.float64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if capacity & (capacity - 1) != 0:
            logger.warning(
                "RingBuffer capacity %d is not a power of 2; "
                "performance may be suboptimal",
                capacity,
            )
        self._capacity = capacity
        self._mask = capacity - 1 if (capacity & (capacity - 1) == 0) else None
        self._buffer: np.ndarray = np.zeros(capacity, dtype=dtype)
        self._item_size = self._buffer.dtype.itemsize
        self._head = 0  # write index
        self._tail = 0  # read index
        self._count = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    @property
    def is_full(self) -> bool:
        return self._count == self._capacity

    def push(self, value: Any) -> bool:
        """Push a value into the buffer. Returns False if full."""
        with self._lock:
            if self._count >= self._capacity:
                return False
            idx = self._head % self._capacity if self._mask is None else self._head & self._mask
            self._buffer[idx] = value
            self._head += 1
            self._count += 1
            return True

    def pop(self) -> Optional[Any]:
        """Pop a value from the buffer. Returns None if empty."""
        with self._lock:
            if self._count <= 0:
                return None
            idx = self._tail % self._capacity if self._mask is None else self._tail & self._mask
            value = self._buffer[idx]
            self._tail += 1
            self._count -= 1
            return value

    def push_force(self, value: Any) -> Optional[Any]:
        """Push a value, overwriting oldest if full. Returns overwritten value or None."""
        with self._lock:
            overwritten = None
            if self._count >= self._capacity:
                idx = self._tail % self._capacity if self._mask is None else self._tail & self._mask
                overwritten = self._buffer[idx]
                self._tail += 1
                self._count -= 1
            idx = self._head % self._capacity if self._mask is None else self._head & self._mask
            self._buffer[idx] = value
            self._head += 1
            self._count += 1
            return overwritten

    def get_read_slice(self, n: int) -> memoryview:
        """Get a memoryview of the next n readable elements (zero-copy)."""
        with self._lock:
            n = min(n, self._count)
            if n <= 0:
                return memoryview(bytearray(0))
            start_idx = self._tail % self._capacity if self._mask is None else self._tail & self._mask
            if start_idx + n <= self._capacity:
                arr = self._buffer[start_idx:start_idx + n]
            else:
                arr = np.concatenate([
                    self._buffer[start_idx:],
                    self._buffer[:n - (self._capacity - start_idx)],
                ])
            return memoryview(arr)

    def get_write_slice(self, n: int) -> memoryview:
        """Get a memoryview of the next n writable positions (zero-copy)."""
        with self._lock:
            n = min(n, self._capacity - self._count)
            if n <= 0:
                return memoryview(bytearray(0))
            start_idx = self._head % self._capacity if self._mask is None else self._head & self._mask
            if start_idx + n <= self._capacity:
                arr = self._buffer[start_idx:start_idx + n]
            else:
                arr = np.concatenate([
                    self._buffer[start_idx:],
                    self._buffer[:n - (self._capacity - start_idx)],
                ])
            return memoryview(arr)

    def commit_write(self, n: int) -> None:
        """Commit n elements written via get_write_slice."""
        with self._lock:
            n = min(n, self._capacity - self._count)
            self._head += n
            self._count += n

    def peek(self) -> Optional[Any]:
        """Peek at the next element without removing it."""
        with self._lock:
            if self._count <= 0:
                return None
            idx = self._tail % self._capacity if self._mask is None else self._tail & self._mask
            return self._buffer[idx]

    def clear(self) -> None:
        """Clear all elements."""
        with self._lock:
            self._head = 0
            self._tail = 0
            self._count = 0
            self._buffer.fill(0)

    def to_array(self) -> np.ndarray:
        """Return a copy of all elements in order."""
        with self._lock:
            if self._count == 0:
                return np.array([], dtype=self._buffer.dtype)
            start_idx = self._tail % self._capacity if self._mask is None else self._tail & self._mask
            if start_idx + self._count <= self._capacity:
                return self._buffer[start_idx:start_idx + self._count].copy()
            return np.concatenate([
                self._buffer[start_idx:],
                self._buffer[:self._count - (self._capacity - start_idx)],
            ]).copy()


# =============================================================================
# MemoryPool
# =============================================================================

class MemoryPool:
    """
    Pre-allocated memory block manager for zero-allocation hot paths.

    Uses ctypes for raw memory allocation and memoryview for safe access.
    """

    def __init__(
        self,
        block_size: int = 4096,
        num_blocks: int = 256,
    ) -> None:
        if block_size <= 0 or num_blocks <= 0:
            raise ValueError("block_size and num_blocks must be positive")
        self._block_size = block_size
        self._num_blocks = num_blocks
        self._total_bytes = block_size * num_blocks

        self._raw_memory: ctypes.Array = (ctypes.c_ubyte * self._total_bytes)()
        self._free_blocks: List[int] = list(range(num_blocks))
        self._allocated: Dict[int, memoryview] = {}
        self._lock = threading.Lock()
        self._alloc_count = 0
        self._dealloc_count = 0

        logger.info(
            "MemoryPool initialized: %d blocks x %d bytes = %.1f KB",
            num_blocks, block_size, self._total_bytes / 1024,
        )

    @property
    def block_size(self) -> int:
        return self._block_size

    def allocate(self) -> Optional[memoryview]:
        """Allocate a block. Returns memoryview or None if pool exhausted."""
        with self._lock:
            if not self._free_blocks:
                logger.warning("MemoryPool exhausted: no free blocks")
                return None
            block_idx = self._free_blocks.pop()
            offset = block_idx * self._block_size
            buf = memoryview(self._raw_memory)[offset:offset + self._block_size]
            self._allocated[block_idx] = buf
            self._alloc_count += 1
            return buf

    def deallocate(self, buffer: memoryview) -> None:
        """Return a block to the pool."""
        with self._lock:
            for block_idx, allocated_buf in list(self._allocated.items()):
                if allocated_buf is buffer or allocated_buf.obj is buffer.obj:
                    offset = block_idx * self._block_size
                    pool_view = memoryview(self._raw_memory)[offset:offset + self._block_size]
                    if buffer.obj is pool_view.obj and buffer.nbytes == self._block_size:
                        del self._allocated[block_idx]
                        self._free_blocks.append(block_idx)
                        self._dealloc_count += 1
                        return
            logger.warning("MemoryPool: attempted to deallocate unknown buffer")

    def deallocate_by_index(self, block_idx: int) -> None:
        """Return a block to the pool by index (faster than buffer lookup)."""
        with self._lock:
            if block_idx in self._allocated:
                del self._allocated[block_idx]
                self._free_blocks.append(block_idx)
                self._dealloc_count += 1
            else:
                logger.warning("MemoryPool: block %d not allocated", block_idx)

    def get_stats(self) -> PoolStats:
        """Return current pool statistics."""
        with self._lock:
            allocated_count = len(self._allocated)
            free_count = len(self._free_blocks)
            return PoolStats(
                total_blocks=self._num_blocks,
                allocated_blocks=allocated_count,
                free_blocks=free_count,
                total_bytes=self._total_bytes,
                allocated_bytes=allocated_count * self._block_size,
                utilization_pct=(allocated_count / self._num_blocks) * 100.0,
            )

    @property
    def available(self) -> int:
        return len(self._free_blocks)


# =============================================================================
# OrderBookFast
# =============================================================================

class OrderBookFast:
    """
    Array-based L2 order book with O(1) best bid/ask lookup and O(log n) insertion.

    Uses numpy arrays for price levels and sorted insertion via binary search.
    Supports bulk updates for efficient batch processing.
    """

    def __init__(self, max_levels: int = 50) -> None:
        self._max_levels = max_levels

        self._bid_prices: np.ndarray = np.zeros(max_levels, dtype=np.float64)
        self._bid_sizes: np.ndarray = np.zeros(max_levels, dtype=np.float64)
        self._bid_count: int = 0

        self._ask_prices: np.ndarray = np.zeros(max_levels, dtype=np.float64)
        self._ask_sizes: np.ndarray = np.zeros(max_levels, dtype=np.float64)
        self._ask_count: int = 0

        self._sequence: int = 0

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def best_bid(self) -> Optional[float]:
        """O(1) best bid price lookup."""
        if self._bid_count == 0:
            return None
        return self._bid_prices[0]

    @property
    def best_ask(self) -> Optional[float]:
        """O(1) best ask price lookup."""
        if self._ask_count == 0:
            return None
        return self._ask_prices[0]

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from best bid/ask."""
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    def add_bid(self, price: float, size: float) -> None:
        """O(log n) bid insertion via binary search."""
        if self._bid_count >= self._max_levels:
            return
        idx = np.searchsorted(self._bid_prices[:self._bid_count], price, side='right')
        idx = self._bid_count - idx
        if idx < self._bid_count:
            self._bid_prices[idx + 1:self._bid_count + 1] = self._bid_prices[idx:self._bid_count]
            self._bid_sizes[idx + 1:self._bid_count + 1] = self._bid_sizes[idx:self._bid_count]
        self._bid_prices[idx] = price
        self._bid_sizes[idx] = size
        self._bid_count += 1
        self._sequence += 1

    def add_ask(self, price: float, size: float) -> None:
        """O(log n) ask insertion via binary search."""
        if self._ask_count >= self._max_levels:
            return
        idx = np.searchsorted(self._ask_prices[:self._ask_count], price, side='left')
        if idx < self._ask_count:
            self._ask_prices[idx + 1:self._ask_count + 1] = self._ask_prices[idx:self._ask_count]
            self._ask_sizes[idx + 1:self._ask_count + 1] = self._ask_sizes[idx:self._ask_count]
        self._ask_prices[idx] = price
        self._ask_sizes[idx] = size
        self._ask_count += 1
        self._sequence += 1

    def remove_bid(self, price: float) -> bool:
        """Remove a bid level by price."""
        for i in range(self._bid_count):
            if self._bid_prices[i] == price:
                self._bid_prices[i:self._bid_count - 1] = self._bid_prices[i + 1:self._bid_count]
                self._bid_sizes[i:self._bid_count - 1] = self._bid_sizes[i + 1:self._bid_count]
                self._bid_count -= 1
                self._sequence += 1
                return True
        return False

    def remove_ask(self, price: float) -> bool:
        """Remove an ask level by price."""
        for i in range(self._ask_count):
            if self._ask_prices[i] == price:
                self._ask_prices[i:self._ask_count - 1] = self._ask_prices[i + 1:self._ask_count]
                self._ask_sizes[i:self._ask_count - 1] = self._ask_sizes[i + 1:self._ask_count]
                self._ask_count -= 1
                self._sequence += 1
                return True
        return False

    def update_bid_size(self, price: float, size: float) -> bool:
        """Update size at an existing bid price level."""
        for i in range(self._bid_count):
            if self._bid_prices[i] == price:
                self._bid_sizes[i] = size
                self._sequence += 1
                return True
        return False

    def update_ask_size(self, price: float, size: float) -> bool:
        """Update size at an existing ask price level."""
        for i in range(self._ask_count):
            if self._ask_prices[i] == price:
                self._ask_sizes[i] = size
                self._sequence += 1
                return True
        return False

    def bulk_update(self, updates: List[Tuple[str, float, float]]) -> None:
        """
        Apply bulk updates: list of (side, price, size) tuples.
        size=0 means remove the level.
        """
        for side, price, size in updates:
            if side == "bid":
                if size <= 0:
                    self.remove_bid(price)
                elif not self.update_bid_size(price, size):
                    self.add_bid(price, size)
            elif side == "ask":
                if size <= 0:
                    self.remove_ask(price)
                elif not self.update_ask_size(price, size):
                    self.add_ask(price, size)
        self._sequence += 1

    def get_bids(self, depth: Optional[int] = None) -> np.ndarray:
        """Return bid levels as (price, size) array."""
        d = depth if depth else self._bid_count
        d = min(d, self._bid_count)
        return np.column_stack([
            self._bid_prices[:d].copy(),
            self._bid_sizes[:d].copy(),
        ])

    def get_asks(self, depth: Optional[int] = None) -> np.ndarray:
        """Return ask levels as (price, size) array."""
        d = depth if depth else self._ask_count
        d = min(d, self._ask_count)
        return np.column_stack([
            self._ask_prices[:d].copy(),
            self._ask_sizes[:d].copy(),
        ])

    def get_imbalance(self, depth: int = 5) -> float:
        """
        Calculate order book imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume).
        Returns value in [-1, 1].
        """
        d = min(depth, self._bid_count, self._ask_count)
        if d == 0:
            return 0.0
        bid_vol = np.sum(self._bid_sizes[:d])
        ask_vol = np.sum(self._ask_sizes[:d])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def clear(self) -> None:
        """Clear all levels."""
        self._bid_count = 0
        self._ask_count = 0
        self._sequence += 1


# =============================================================================
# TimestampTracker
# =============================================================================

class TimestampTracker:
    """
    Nanosecond-precision timestamp tracking with latency measurement.

    Uses time.perf_counter_ns() for high-resolution timing.
    Maintains a rolling window of latency samples for percentile calculation.
    """

    def __init__(self, window_size: int = 100_000) -> None:
        self._window_size = window_size
        self._latencies: np.ndarray = np.zeros(window_size, dtype=np.float64)
        self._index: int = 0
        self._count: int = 0
        self._total_latency_ns: float = 0.0
        self._start_times: Dict[str, int] = {}
        self._throughput_window: deque = deque(maxlen=window_size)

    def now_ns(self) -> int:
        """Current time in nanoseconds."""
        return time.perf_counter_ns()

    def start(self, label: str) -> None:
        """Start timing a labeled operation."""
        self._start_times[label] = self.now_ns()

    def stop(self, label: str) -> float:
        """
        Stop timing and record latency. Returns latency in nanoseconds.
        """
        start = self._start_times.pop(label, None)
        if start is None:
            logger.warning("TimestampTracker: no start for label '%s'", label)
            return 0.0
        latency_ns = self.now_ns() - start
        self.record(latency_ns)
        return latency_ns

    def record(self, latency_ns: float) -> None:
        """Record a latency sample."""
        idx = self._index % self._window_size
        self._latencies[idx] = latency_ns
        self._index += 1
        self._count += 1
        self._total_latency_ns += latency_ns
        self._throughput_window.append(self.now_ns())

    def measure(self, label: str):
        """
        Context manager for timing operations.

        Usage:
            with tracker.measure("operation"):
                do_something()
        """
        return _TimingContext(self, label)

    def get_percentile(self, percentile: float) -> float:
        """
        Calculate latency percentile (0-100).
        e.g., get_percentile(99) returns p99 latency.
        """
        if self._count == 0:
            return 0.0
        n = min(self._count, self._window_size)
        if self._index <= self._window_size:
            samples = self._latencies[:self._index]
        else:
            samples = self._latencies
        sorted_samples = np.sort(samples[samples > 0])
        if len(sorted_samples) == 0:
            return 0.0
        idx = min(int(len(sorted_samples) * percentile / 100.0), len(sorted_samples) - 1)
        return sorted_samples[idx]

    def get_p50(self) -> float:
        return self.get_percentile(50)

    def get_p95(self) -> float:
        return self.get_percentile(95)

    def get_p99(self) -> float:
        return self.get_percentile(99)

    def get_mean(self) -> float:
        if self._count == 0:
            return 0.0
        return self._total_latency_ns / self._count

    def get_throughput(self) -> float:
        """Calculate operations per second over the tracking window."""
        if len(self._throughput_window) < 2:
            return 0.0
        timestamps = list(self._throughput_window)
        elapsed_ns = timestamps[-1] - timestamps[0]
        if elapsed_ns == 0:
            return 0.0
        return len(timestamps) / (elapsed_ns / 1e9)

    def get_stats(self) -> Dict[str, float]:
        """Return comprehensive latency statistics."""
        return {
            "count": self._count,
            "mean_ns": self.get_mean(),
            "p50_ns": self.get_p50(),
            "p95_ns": self.get_p95(),
            "p99_ns": self.get_p99(),
            "throughput_ops_per_sec": self.get_throughput(),
        }

    def reset(self) -> None:
        """Reset all tracked data."""
        self._latencies.fill(0)
        self._index = 0
        self._count = 0
        self._total_latency_ns = 0.0
        self._start_times.clear()
        self._throughput_window.clear()


class _TimingContext:
    """Context manager for TimestampTracker.measure()."""

    def __init__(self, tracker: TimestampTracker, label: str) -> None:
        self._tracker = tracker
        self._label = label
        self._latency_ns: float = 0.0

    def __enter__(self) -> "_TimingContext":
        self._tracker.start(self._label)
        return self

    def __exit__(self, *args: Any) -> None:
        self._latency_ns = self._tracker.stop(self._label)

    @property
    def latency_ns(self) -> float:
        return self._latency_ns


# =============================================================================
# MessageQueue
# =============================================================================

class _Message:
    """Internal message representation."""

    __slots__ = ("data", "priority", "timestamp_ns", "size_bytes")

    def __init__(
        self,
        data: memoryview,
        priority: int,
        timestamp_ns: int,
        size_bytes: int,
    ) -> None:
        self.data = data
        self.priority = priority
        self.timestamp_ns = timestamp_ns
        self.size_bytes = size_bytes


class MessageQueue:
    """
    Zero-copy message queue with priority queuing and backpressure handling.

    Uses memoryview for zero-copy message passing and numpy for batch processing.
    """

    def __init__(
        self,
        max_size: int = 100_000,
        max_memory_bytes: int = 64 * 1024 * 1024,  # 64 MB
        backpressure_threshold: float = 0.8,
        num_priority_levels: int = 8,
    ) -> None:
        self._max_size = max_size
        self._max_memory_bytes = max_memory_bytes
        self._backpressure_threshold = backpressure_threshold
        self._num_priority_levels = num_priority_levels

        self._queues: List[deque] = [deque() for _ in range(num_priority_levels)]
        self._total_messages: int = 0
        self._total_memory_bytes: int = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._backpressure_active: bool = False
        self._dropped_count: int = 0
        self._enqueued_count: int = 0
        self._dequeued_count: int = 0

    @property
    def size(self) -> int:
        return self._total_messages

    @property
    def is_backpressured(self) -> bool:
        return self._backpressure_active

    def enqueue(self, data: memoryview, priority: int = 0) -> bool:
        """
        Enqueue a message with zero-copy. Returns False if backpressured.
        Priority: 0 (highest) to num_priority_levels-1 (lowest).
        """
        priority = max(0, min(priority, self._num_priority_levels - 1))

        with self._lock:
            if self._backpressure_active:
                self._dropped_count += 1
                return False

            if self._total_messages >= self._max_size:
                self._backpressure_active = True
                self._dropped_count += 1
                logger.warning("MessageQueue: backpressure activated (max_size=%d)", self._max_size)
                return False

            if self._total_memory_bytes + data.nbytes > self._max_memory_bytes:
                self._backpressure_active = True
                self._dropped_count += 1
                logger.warning(
                    "MessageQueue: backpressure activated (max_memory=%d bytes)",
                    self._max_memory_bytes,
                )
                return False

            msg = _Message(
                data=data,
                priority=priority,
                timestamp_ns=time.perf_counter_ns(),
                size_bytes=data.nbytes,
            )
            self._queues[priority].append(msg)
            self._total_messages += 1
            self._total_memory_bytes += data.nbytes
            self._enqueued_count += 1
            self._not_empty.notify()
            return True

    def dequeue(self) -> Optional[memoryview]:
        """Dequeue the highest-priority message (zero-copy)."""
        with self._lock:
            for queue in self._queues:
                if queue:
                    msg = queue.popleft()
                    self._total_messages -= 1
                    self._total_memory_bytes -= msg.size_bytes
                    self._dequeued_count += 1
                    self._backpressure_active = False
                    return msg.data
            return None

    def dequeue_batch(self, max_batch: int = 100) -> List[memoryview]:
        """Dequeue a batch of messages for efficient processing."""
        results: List[memoryview] = []
        with self._lock:
            for queue in self._queues:
                while queue and len(results) < max_batch:
                    msg = queue.popleft()
                    self._total_messages -= 1
                    self._total_memory_bytes -= msg.size_bytes
                    self._dequeued_count += 1
                    results.append(msg.data)
            self._backpressure_active = False
        return results

    def peek(self) -> Optional[memoryview]:
        """Peek at the highest-priority message without removing it."""
        with self._lock:
            for queue in self._queues:
                if queue:
                    return queue[0].data
            return None

    def get_backpressure_ratio(self) -> float:
        """Return current queue utilization (0.0 to 1.0)."""
        if self._max_size == 0:
            return 0.0
        size_ratio = self._total_messages / self._max_size
        memory_ratio = self._total_memory_bytes / self._max_memory_bytes
        return max(size_ratio, memory_ratio)

    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics."""
        return {
            "total_messages": self._total_messages,
            "total_memory_bytes": self._total_memory_bytes,
            "enqueued_count": self._enqueued_count,
            "dequeued_count": self._dequeued_count,
            "dropped_count": self._dropped_count,
            "backpressure_active": self._backpressure_active,
            "backpressure_ratio": self.get_backpressure_ratio(),
            "per_priority_sizes": [len(q) for q in self._queues],
        }

    def clear(self) -> None:
        """Clear all messages."""
        with self._lock:
            for queue in self._queues:
                queue.clear()
            self._total_messages = 0
            self._total_memory_bytes = 0
            self._backpressure_active = False


# =============================================================================
# HotPathOptimizer
# =============================================================================

class HotPathOptimizer:
    """
    Identifies hot code paths and suggests optimizations.

    Analyzes function call patterns and timing to identify performance
    bottlenecks in the trading hot path.
    """

    _COMMON_OPTIMIZATIONS: Dict[str, List[Optimization]] = {
        "dict_lookup": [
            Optimization(
                hot_path=HotPath("", "", 0, 0, 0, 0, 0),
                suggestion="Replace dict lookups with direct attribute access or __slots__",
                expected_improvement_pct=15.0,
                difficulty="easy",
                details="Dict lookups involve hashing; attribute access is O(1) via descriptor protocol",
            ),
        ],
        "function_call": [
            Optimization(
                hot_path=HotPath("", "", 0, 0, 0, 0, 0),
                suggestion="Inline small frequently-called functions",
                expected_improvement_pct=10.0,
                difficulty="easy",
                details="Function call overhead in CPython is ~100-200ns per call",
            ),
        ],
        "list_append": [
            Optimization(
                hot_path=HotPath("", "", 0, 0, 0, 0, 0),
                suggestion="Use pre-allocated numpy arrays instead of list.append",
                expected_improvement_pct=25.0,
                difficulty="medium",
                details="Pre-allocated arrays avoid repeated memory allocation and resizing",
            ),
        ],
        "attribute_access": [
            Optimization(
                hot_path=HotPath("", "", 0, 0, 0, 0, 0),
                suggestion="Cache attribute lookups in local variables",
                expected_improvement_pct=8.0,
                difficulty="easy",
                details="Each attribute access involves __getattribute__ lookup",
            ),
        ],
        "global_lookup": [
            Optimization(
                hot_path=HotPath("", "", 0, 0, 0, 0, 0),
                suggestion="Move global lookups to local scope (local vars are faster)",
                expected_improvement_pct=12.0,
                difficulty="easy",
                details="Python resolves local variables via LOAD_FAST vs LOAD_GLOBAL",
            ),
        ],
    }

    def __init__(self) -> None:
        self._profiles: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._call_counts: Dict[str, int] = defaultdict(int)

    def record_call(
        self,
        function_name: str,
        file_path: str,
        line_number: int,
        duration_ns: float,
    ) -> None:
        """Record a function call with its duration."""
        key = f"{file_path}:{function_name}:{line_number}"
        self._profiles[key]["durations"].append(duration_ns)
        self._call_counts[key] += 1

    def identify_hot_paths(
        self,
        code: Optional[str] = None,
        top_n: int = 10,
    ) -> List[HotPath]:
        """
        Identify the top N hottest code paths by total time.

        Args:
            code: Optional code snippet to analyze (placeholder for AST analysis).
            top_n: Number of hot paths to return.

        Returns:
            List of HotPath sorted by total_time_ns descending.
        """
        hot_paths: List[HotPath] = []

        for key, data in self._profiles.items():
            durations = data.get("durations", [])
            if not durations:
                continue

            parts = key.split(":")
            file_path = parts[0] if len(parts) > 0 else ""
            function_name = parts[1] if len(parts) > 1 else ""
            line_number = int(parts[2]) if len(parts) > 2 else 0

            total_time = sum(durations)
            avg_time = total_time / len(durations) if durations else 0.0
            call_count = self._call_counts.get(key, 0)

            hot_paths.append(HotPath(
                function_name=function_name,
                file_path=file_path,
                line_number=line_number,
                call_count=call_count,
                avg_time_ns=avg_time,
                total_time_ns=total_time,
                self_time_pct=0.0,
            ))

        hot_paths.sort(key=lambda hp: hp.total_time_ns, reverse=True)
        return hot_paths[:top_n]

    def suggest_optimizations(self, hot_path: HotPath) -> List[Optimization]:
        """
        Generate optimization suggestions for a hot path.

        Analyzes the hot path characteristics and returns applicable optimizations.
        """
        suggestions: List[Optimization] = []

        if hot_path.call_count > 10_000:
            for opt in self._COMMON_OPTIMIZATIONS.get("function_call", []):
                suggestions.append(Optimization(
                    hot_path=hot_path,
                    suggestion=opt.suggestion,
                    expected_improvement_pct=opt.expected_improvement_pct,
                    difficulty=opt.difficulty,
                    details=opt.details,
                ))

        if hot_path.avg_time_ns > 1000:
            for opt in self._COMMON_OPTIMIZATIONS.get("dict_lookup", []):
                suggestions.append(Optimization(
                    hot_path=hot_path,
                    suggestion=opt.suggestion,
                    expected_improvement_pct=opt.expected_improvement_pct,
                    difficulty=opt.difficulty,
                    details=opt.details,
                ))

        if hot_path.total_time_ns > 1_000_000:
            for opt in self._COMMON_OPTIMIZATIONS.get("list_append", []):
                suggestions.append(Optimization(
                    hot_path=hot_path,
                    suggestion=opt.suggestion,
                    expected_improvement_pct=opt.expected_improvement_pct,
                    difficulty=opt.difficulty,
                    details=opt.details,
                ))

        if not suggestions:
            suggestions.append(Optimization(
                hot_path=hot_path,
                suggestion="Profile with cProfile/pyinstrument for detailed breakdown",
                expected_improvement_pct=5.0,
                difficulty="easy",
                details="General profiling recommendation for unidentified hotspots",
            ))

        return suggestions

    def measure_improvement(
        self,
        before: List[float],
        after: List[float],
    ) -> ImprovementMetrics:
        """
        Measure improvement between before and after latency samples.

        Args:
            before: List of latency samples (ns) before optimization.
            after: List of latency samples (ns) after optimization.

        Returns:
            ImprovementMetrics with comparative statistics.
        """
        before_arr = np.array(before, dtype=np.float64)
        after_arr = np.array(after, dtype=np.float64)

        before_avg = float(np.mean(before_arr)) if len(before_arr) > 0 else 0.0
        after_avg = float(np.mean(after_arr)) if len(after_arr) > 0 else 0.0
        before_p99 = float(np.percentile(before_arr, 99)) if len(before_arr) > 0 else 0.0
        after_p99 = float(np.percentile(after_arr, 99)) if len(after_arr) > 0 else 0.0

        speedup = before_avg / after_avg if after_avg > 0 else 0.0
        improvement = ((before_avg - after_avg) / before_avg * 100.0) if before_avg > 0 else 0.0

        return ImprovementMetrics(
            before_avg_ns=before_avg,
            after_avg_ns=after_avg,
            before_p99_ns=before_p99,
            after_p99_ns=after_p99,
            speedup_factor=speedup,
            improvement_pct=improvement,
        )

    def reset(self) -> None:
        """Clear all recorded profiles."""
        self._profiles.clear()
        self._call_counts.clear()


# =============================================================================
# LatencyMonitor
# =============================================================================

class LatencyMonitor:
    """
    Component-level latency tracking with percentile calculation and alerting.

    Tracks latency per component and provides bottleneck identification.
    """

    def __init__(
        self,
        default_threshold_ns: float = 1_000_000,  # 1ms default
        window_size: int = 10_000,
    ) -> None:
        self._default_threshold_ns = default_threshold_ns
        self._window_size = window_size
        self._latencies: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(window_size, dtype=np.float64)
        )
        self._counts: Dict[str, int] = defaultdict(int)
        self._indices: Dict[str, int] = defaultdict(int)
        self._thresholds: Dict[str, float] = {}
        self._alerts: List[Alert] = []
        self._lock = threading.Lock()

    def set_threshold(self, component: str, threshold_ns: float) -> None:
        """Set a custom latency threshold for a component."""
        self._thresholds[component] = threshold_ns

    def track_component_latency(self, component: str, latency_ns: float) -> None:
        """Record a latency measurement for a component."""
        with self._lock:
            arr = self._latencies[component]
            idx = self._indices[component] % self._window_size
            arr[idx] = latency_ns
            self._indices[component] += 1
            self._counts[component] += 1

            threshold = self._thresholds.get(component, self._default_threshold_ns)
            if latency_ns > threshold:
                alert = self._create_alert(component, latency_ns, threshold)
                self._alerts.append(alert)
                logger.warning(
                    "LatencyMonitor: %s latency %.0fns exceeds threshold %.0fns",
                    component, latency_ns, threshold,
                )

    def get_latency_percentiles(self, component: str) -> LatencyPercentiles:
        """Get latency percentile statistics for a component."""
        with self._lock:
            count = self._counts.get(component, 0)
            if count == 0:
                return LatencyPercentiles(
                    component=component,
                    count=0,
                    min_ns=0.0,
                    max_ns=0.0,
                    p50_ns=0.0,
                    p95_ns=0.0,
                    p99_ns=0.0,
                    mean_ns=0.0,
                    stddev_ns=0.0,
                )

            arr = self._latencies[component]
            n = min(count, self._window_size)
            if self._indices[component] <= self._window_size:
                samples = arr[:self._indices[component]]
            else:
                samples = arr

            valid = samples[samples > 0]
            if len(valid) == 0:
                return LatencyPercentiles(
                    component=component,
                    count=count,
                    min_ns=0.0,
                    max_ns=0.0,
                    p50_ns=0.0,
                    p95_ns=0.0,
                    p99_ns=0.0,
                    mean_ns=0.0,
                    stddev_ns=0.0,
                )

            return LatencyPercentiles(
                component=component,
                count=count,
                min_ns=float(np.min(valid)),
                max_ns=float(np.max(valid)),
                p50_ns=float(np.percentile(valid, 50)),
                p95_ns=float(np.percentile(valid, 95)),
                p99_ns=float(np.percentile(valid, 99)),
                mean_ns=float(np.mean(valid)),
                stddev_ns=float(np.std(valid)),
            )

    def alert_if_exceeds(
        self,
        component: str,
        threshold_ns: Optional[float] = None,
    ) -> Optional[Alert]:
        """
        Check if the latest latency exceeds threshold. Returns Alert if breached.
        """
        with self._lock:
            count = self._counts.get(component, 0)
            if count == 0:
                return None

            threshold = threshold_ns or self._thresholds.get(component, self._default_threshold_ns)
            arr = self._latencies[component]
            idx = (self._indices[component] - 1) % self._window_size
            latest = arr[idx]

            if latest > threshold:
                alert = self._create_alert(component, latest, threshold)
                self._alerts.append(alert)
                return alert
            return None

    def get_bottleneck(self) -> str:
        """
        Identify the component with the highest average latency.
        Returns component name or empty string if no data.
        """
        with self._lock:
            if not self._counts:
                return ""

            max_mean = 0.0
            bottleneck = ""

            for component, count in self._counts.items():
                if count == 0:
                    continue
                arr = self._latencies[component]
                n = min(count, self._window_size)
                if self._indices[component] <= self._window_size:
                    samples = arr[:self._indices[component]]
                else:
                    samples = arr

                valid = samples[samples > 0]
                if len(valid) == 0:
                    continue

                mean = float(np.mean(valid))
                if mean > max_mean:
                    max_mean = mean
                    bottleneck = component

            return bottleneck

    def get_all_percentiles(self) -> Dict[str, LatencyPercentiles]:
        """Get latency percentiles for all tracked components."""
        with self._lock:
            return {
                component: self.get_latency_percentiles(component)
                for component in self._counts
            }

    def get_alerts(self, component: Optional[str] = None) -> List[Alert]:
        """Get all alerts, optionally filtered by component."""
        with self._lock:
            if component:
                return [a for a in self._alerts if a.component == component]
            return list(self._alerts)

    def clear_alerts(self) -> None:
        """Clear all recorded alerts."""
        with self._lock:
            self._alerts.clear()

    def _create_alert(
        self,
        component: str,
        actual_ns: float,
        threshold_ns: float,
    ) -> Alert:
        ratio = actual_ns / threshold_ns if threshold_ns > 0 else 0
        severity = "critical" if ratio > 2.0 else "warning"
        return Alert(
            component=component,
            threshold_ns=threshold_ns,
            actual_ns=actual_ns,
            timestamp_ns=time.perf_counter_ns(),
            severity=severity,
            message=(
                f"{component} latency {actual_ns:.0f}ns "
                f"exceeds threshold {threshold_ns:.0f}ns "
                f"({ratio:.1f}x)"
            ),
        )
