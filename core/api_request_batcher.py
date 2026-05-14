#!/usr/bin/env python3
"""
API Request Batcher - S+ Tier
Batches API requests to optimize throughput and reduce rate limiting.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for request batching"""

    max_batch_size: int = 10
    max_wait_time: float = 0.1  # seconds
    rate_limit_per_second: int = 50
    retry_attempts: int = 3
    timeout: float = 30.0


class APIRequestBatcher:
    """
    API Request Batcher - S+ Tier
    Batches API requests to optimize throughput and reduce rate limiting.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.request_queue: deque = deque()
        self.pending_batches: Dict[str, List[Any]] = {}
        self.batch_results: Dict[str, Any] = {}
        self.is_running = False
        self.last_request_time = 0.0

        # Rate limiting
        self.request_count = 0
        self.rate_limit_window_start = time.time()

        logger.info("API Request Batcher initialized")

    async def start(self) -> None:
        """Start the batching processor"""
        if self.is_running:
            return

        self.is_running = True
        logger.info("API Request Batcher started")

        # Start processing loop
        asyncio.create_task(self._process_batches())

    async def stop(self) -> None:
        """Stop the batching processor"""
        self.is_running = False
        logger.info("API Request Batcher stopped")

    async def submit_request(self, request_params: Dict[str, Any], request_id: Optional[str] = None) -> str:
        """
        Submit a request for batching.

        Args:
            request_params: Request parameters
            request_id: Optional request ID

        Returns:
            Request ID for tracking
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"

        # Add to queue
        self.request_queue.append({"id": request_id, "params": request_params, "submitted_at": time.time()})

        return request_id

    async def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[Any]:
        """
        Get result for a request.

        Args:
            request_id: Request ID
            timeout: Timeout in seconds

        Returns:
            Request result or None if not ready
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if request_id in self.batch_results:
                result = self.batch_results[request_id]
                del self.batch_results[request_id]  # Clean up
                return result

            await asyncio.sleep(0.01)  # Small delay

        return None

    async def _process_batches(self) -> None:
        """Main batch processing loop"""
        while self.is_running:
            try:
                # Check rate limits
                await self._enforce_rate_limits()

                # Collect requests for batch
                batch_requests = []
                batch_start_time = time.time()

                while (
                    len(batch_requests) < self.config.max_batch_size
                    and time.time() - batch_start_time < self.config.max_wait_time
                ):

                    if self.request_queue:
                        request = self.request_queue.popleft()
                        batch_requests.append(request)
                    else:
                        # Wait for requests
                        await asyncio.sleep(0.01)

                    # Check timeout
                    if time.time() - batch_start_time >= self.config.max_wait_time:
                        break

                # Process batch if we have requests
                if batch_requests:
                    await self._execute_batch(batch_requests)

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(1.0)

    async def _execute_batch(self, batch_requests: List[Dict[str, Any]]) -> None:
        """
        Execute a batch of requests.

        Args:
            batch_requests: List of requests to execute
        """
        batch_id = f"batch_{int(time.time() * 1000000)}"

        try:
            # Extract parameters
            params_list = [req["params"] for req in batch_requests]

            # Execute batch (placeholder - would call actual API)
            # In real implementation, this would call the actual API endpoint
            batch_result = await self._mock_api_call(params_list)

            # Distribute results
            for i, request in enumerate(batch_requests):
                result = batch_result[i] if i < len(batch_result) else {"error": "No result"}
                self.batch_results[request["id"]] = result

            self.request_count += len(batch_requests)

            logger.debug(f"Processed batch {batch_id} with {len(batch_requests)} requests")

        except Exception as e:
            logger.error(f"Error executing batch {batch_id}: {e}")

            # Mark all requests in batch as failed
            for request in batch_requests:
                self.batch_results[request["id"]] = {"error": str(e)}

    async def _mock_api_call(self, params_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Mock API call for demonstration.

        Args:
            params_list: List of request parameters

        Returns:
            Mock results
        """
        # Simulate API delay
        await asyncio.sleep(0.01)

        # Return mock results
        results = []
        for params in params_list:
            # Simulate different response types
            if "price" in params:
                results.append({"price": params["price"] * 1.001, "status": "success"})
            elif "volume" in params:
                results.append({"volume": params["volume"], "status": "success"})
            else:
                results.append({"data": params, "status": "success"})

        return results

    async def _enforce_rate_limits(self) -> None:
        """Enforce rate limiting"""
        current_time = time.time()

        # Reset counter if window has passed
        if current_time - self.rate_limit_window_start >= 1.0:
            self.request_count = 0
            self.rate_limit_window_start = current_time

        # Check if we're at the limit
        if self.request_count >= self.config.rate_limit_per_second:
            # Wait until next window
            sleep_time = 1.0 - (current_time - self.rate_limit_window_start)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queue_size": len(self.request_queue),
            "pending_batches": len(self.pending_batches),
            "total_requests_processed": self.request_count,
            "is_running": self.is_running,
            "rate_limit_remaining": max(0, self.config.rate_limit_per_second - self.request_count),
        }

    async def flush_queue(self, timeout: float = 5.0) -> int:
        """
        Flush all pending requests.

        Args:
            timeout: Maximum time to wait for completion

        Returns:
            Number of requests flushed
        """
        start_time = time.time()
        flushed_count = 0

        while self.request_queue and (time.time() - start_time < timeout):
            # Process remaining batches
            if self.request_queue:
                batch_size = min(self.config.max_batch_size, len(self.request_queue))
                batch_requests = []

                for _ in range(batch_size):
                    if self.request_queue:
                        batch_requests.append(self.request_queue.popleft())

                if batch_requests:
                    await self._execute_batch(batch_requests)
                    flushed_count += len(batch_requests)

            await asyncio.sleep(0.01)

        return flushed_count

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_requests_processed": self.request_count,
            "current_queue_size": len(self.request_queue),
            "avg_batch_size": self.config.max_batch_size,
            "rate_limit_per_second": self.config.rate_limit_per_second,
            "is_running": self.is_running,
        }
