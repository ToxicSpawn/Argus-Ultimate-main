#!/usr/bin/env python3
"""
Rate Limiter - S+ Tier
Advanced rate limiting with burst handling and dynamic adjustments.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""

    requests_per_second: float = 10.0
    burst_size: int | None = None  # Burst allowance
    window_seconds: float = 1.0
    adaptive: bool = True


@dataclass
class RateLimitResult:
    """Rate limit check result"""

    allowed: bool
    wait_time: float
    remaining_requests: int
    reset_time: float


class TokenBucketRateLimiter:
    """
    Token Bucket Rate Limiter - S+ Tier
    Implements token bucket algorithm with burst handling.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.requests_per_second * config.window_seconds
        self.max_tokens = self.tokens
        self.last_update = time.time()

        # Burst handling
        if config.burst_size:
            self.max_tokens += config.burst_size

        # Statistics
        self.total_requests = 0
        self.throttled_requests = 0

    def check_limit(self) -> RateLimitResult:
        """Check if request can proceed"""
        current_time = time.time()
        time_passed = current_time - self.last_update

        # Refill tokens
        tokens_to_add = time_passed * self.config.requests_per_second
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)

        self.last_update = current_time
        self.total_requests += 1

        if self.tokens >= 1.0:
            # Allow request
            self.tokens -= 1.0
            remaining = int(self.tokens)
            reset_time = current_time + (1.0 / self.config.requests_per_second)

            return RateLimitResult(allowed=True, wait_time=0.0, remaining_requests=remaining, reset_time=reset_time)
        else:
            # Throttle request
            self.throttled_requests += 1
            wait_time = (1.0 - self.tokens) / self.config.requests_per_second
            reset_time = current_time + wait_time

            return RateLimitResult(allowed=False, wait_time=wait_time, remaining_requests=0, reset_time=reset_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        total_handled = self.total_requests - self.throttled_requests
        throttle_rate = self.throttled_requests / max(1, self.total_requests)

        return {
            "total_requests": self.total_requests,
            "throttled_requests": self.throttled_requests,
            "throttle_rate": throttle_rate,
            "current_tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "efficiency": total_handled / max(1, self.total_requests),
        }


class SlidingWindowRateLimiter:
    """
    Sliding Window Rate Limiter - S+ Tier
    Implements sliding window algorithm for precise rate limiting.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: List[float] = []  # Timestamps of requests

        # Statistics
        self.total_requests = 0
        self.throttled_requests = 0

    def check_limit(self) -> RateLimitResult:
        """Check if request can proceed"""
        current_time = time.time()
        self.total_requests += 1

        # Remove old requests outside the window
        window_start = current_time - self.config.window_seconds
        self.requests = [t for t in self.requests if t > window_start]

        # Check if under limit
        if len(self.requests) < self.config.requests_per_second * self.config.window_seconds:
            # Allow request
            self.requests.append(current_time)

            # Calculate reset time
            if self.requests:
                reset_time = self.requests[0] + self.config.window_seconds
            else:
                reset_time = current_time + self.config.window_seconds

            remaining = int((self.config.requests_per_second * self.config.window_seconds) - len(self.requests))

            return RateLimitResult(allowed=True, wait_time=0.0, remaining_requests=remaining, reset_time=reset_time)
        else:
            # Throttle request
            self.throttled_requests += 1

            # Calculate wait time until oldest request expires
            if self.requests:
                wait_time = self.requests[0] + self.config.window_seconds - current_time
                reset_time = self.requests[0] + self.config.window_seconds
            else:
                wait_time = 0.0
                reset_time = current_time

            return RateLimitResult(
                allowed=False, wait_time=max(0, wait_time), remaining_requests=0, reset_time=reset_time
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        total_handled = self.total_requests - self.throttled_requests
        throttle_rate = self.throttled_requests / max(1, self.total_requests)

        return {
            "total_requests": self.total_requests,
            "throttled_requests": self.throttled_requests,
            "throttle_rate": throttle_rate,
            "current_window_requests": len(self.requests),
            "window_capacity": int(self.config.requests_per_second * self.config.window_seconds),
            "efficiency": total_handled / max(1, self.total_requests),
        }


class AdaptiveRateLimiter:
    """
    Adaptive Rate Limiter - S+ Tier
    Dynamically adjusts rate limits based on system performance.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.base_rate = config.requests_per_second

        # Adaptive parameters
        self.current_rate = self.base_rate
        self.adjustment_factor = 1.1  # How much to adjust rates
        self.min_rate = self.base_rate * 0.1  # Minimum rate
        self.max_rate = self.base_rate * 10.0  # Maximum rate

        # Performance tracking
        self.response_times: List[float] = []
        self.error_rates: List[float] = []
        self.last_adjustment = time.time()

        # Use token bucket as base limiter
        self.limiter = TokenBucketRateLimiter(config)

        logger.info(f"Adaptive Rate Limiter initialized (base rate: {self.base_rate})")

    def check_limit(self) -> RateLimitResult:
        """Check rate limit with adaptation"""
        # Update configuration with current rate
        self.config.requests_per_second = self.current_rate

        result = self.limiter.check_limit()

        # Adapt rate based on performance (every 60 seconds)
        current_time = time.time()
        if current_time - self.last_adjustment > 60:
            self._adapt_rate()
            self.last_adjustment = current_time

        return result

    def _adapt_rate(self) -> None:
        """Adapt rate based on recent performance"""
        if len(self.response_times) < 10:
            return  # Need more data

        # Calculate metrics
        avg_response_time = sum(self.response_times[-10:]) / 10
        error_rate = sum(self.error_rates[-10:]) / 10

        # Adjust rate based on performance
        if avg_response_time > 1.0:  # Slow responses
            self.current_rate = max(self.min_rate, self.current_rate * 0.9)
            logger.info(f"Reducing rate due to slow responses: {self.current_rate:.1f}")
        elif error_rate > 0.1:  # High error rate
            self.current_rate = max(self.min_rate, self.current_rate * 0.95)
            logger.info(f"Reducing rate due to high errors: {self.current_rate:.1f}")
        elif avg_response_time < 0.1 and error_rate < 0.01:  # Good performance
            self.current_rate = min(self.max_rate, self.current_rate * 1.05)
            logger.info(f"Increasing rate due to good performance: {self.current_rate:.1f}")

    def record_performance(self, response_time: float, had_error: bool) -> None:
        """Record performance metrics"""
        self.response_times.append(response_time)
        self.error_rates.append(1.0 if had_error else 0.0)

        # Keep only recent data
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
            self.error_rates = self.error_rates[-100:]

    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive rate limiter statistics"""
        base_stats = self.limiter.get_stats()

        return {
            **base_stats,
            "current_rate": self.current_rate,
            "base_rate": self.base_rate,
            "rate_adjustment_ratio": self.current_rate / self.base_rate,
            "avg_response_time": sum(self.response_times) / max(1, len(self.response_times)),
            "error_rate": sum(self.error_rates) / max(1, len(self.error_rates)),
        }


class RateLimiterManager:
    """
    Rate Limiter Manager - S+ Tier
    Manages multiple rate limiters for different endpoints/services.
    """

    def __init__(self):
        self.limiters: Dict[str, Any] = {}  # endpoint -> limiter
        self.endpoint_configs: Dict[str, RateLimitConfig] = {}

        logger.info("Rate Limiter Manager initialized")

    def add_endpoint(self, endpoint: str, config: RateLimitConfig, algorithm: str = "token_bucket") -> None:
        """
        Add rate limiter for an endpoint.

        Args:
            endpoint: Endpoint identifier
            config: Rate limit configuration
            algorithm: Rate limiting algorithm ('token_bucket', 'sliding_window', 'adaptive')
        """
        self.endpoint_configs[endpoint] = config

        if algorithm == "token_bucket":
            self.limiters[endpoint] = TokenBucketRateLimiter(config)
        elif algorithm == "sliding_window":
            self.limiters[endpoint] = SlidingWindowRateLimiter(config)
        elif algorithm == "adaptive":
            self.limiters[endpoint] = AdaptiveRateLimiter(config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        logger.info(f"Added {algorithm} rate limiter for {endpoint}")

    def check_limit(self, endpoint: str) -> RateLimitResult:
        """
        Check rate limit for an endpoint.

        Args:
            endpoint: Endpoint to check

        Returns:
            Rate limit result
        """
        if endpoint not in self.limiters:
            # No limiter configured, allow request
            return RateLimitResult(allowed=True, wait_time=0.0, remaining_requests=999, reset_time=time.time() + 60)

        limiter = self.limiters[endpoint]
        result = limiter.check_limit()

        if not result.allowed:
            logger.warning(f"Rate limit exceeded for {endpoint}, wait {result.wait_time:.2f}s")

        return result

    async def wait_if_needed(self, endpoint: str) -> None:
        """
        Wait if rate limit would be exceeded.

        Args:
            endpoint: Endpoint to check
        """
        result = self.check_limit(endpoint)

        if not result.allowed and result.wait_time > 0:
            logger.info(f"Rate limited, waiting {result.wait_time:.2f}s")
            await asyncio.sleep(result.wait_time)

    def record_performance(self, endpoint: str, response_time: float, had_error: bool) -> None:
        """
        Record performance metrics for adaptive rate limiters.

        Args:
            endpoint: Endpoint identifier
            response_time: Response time in seconds
            had_error: Whether the request had an error
        """
        if endpoint in self.limiters:
            limiter = self.limiters[endpoint]
            if hasattr(limiter, "record_performance"):
                limiter.record_performance(response_time, had_error)

    def get_endpoint_stats(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for an endpoint.

        Args:
            endpoint: Endpoint identifier

        Returns:
            Statistics dictionary or None if endpoint not found
        """
        if endpoint in self.limiters:
            return self.limiters[endpoint].get_stats()

        return None

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all endpoints"""
        stats = {}

        for endpoint, limiter in self.limiters.items():
            stats[endpoint] = limiter.get_stats()

        return {
            "endpoints": stats,
            "total_endpoints": len(self.limiters),
            "timestamp": datetime.now().isoformat(),
        }

    def update_endpoint_config(self, endpoint: str, config: RateLimitConfig) -> None:
        """
        Update configuration for an endpoint.

        Args:
            endpoint: Endpoint identifier
            config: New configuration
        """
        if endpoint in self.limiters:
            self.endpoint_configs[endpoint] = config

            # Recreate limiter with new config
            algorithm = "token_bucket"  # Default, could be stored
            if hasattr(self.limiters[endpoint], "__class__"):
                if "Adaptive" in str(self.limiters[endpoint].__class__):
                    algorithm = "adaptive"
                elif "Sliding" in str(self.limiters[endpoint].__class__):
                    algorithm = "sliding_window"

            self.add_endpoint(endpoint, config, algorithm)
            logger.info(f"Updated configuration for {endpoint}")


# Global rate limiter manager instance
_rate_limiter_manager = RateLimiterManager()


def get_rate_limiter_manager() -> RateLimiterManager:
    """Get global rate limiter manager instance"""
    return _rate_limiter_manager


async def rate_limited_request(endpoint: str, request_func, *args, **kwargs):
    """
    Make a rate-limited request.

    Args:
        endpoint: Endpoint identifier
        request_func: Function to call
        *args: Arguments for request function
        **kwargs: Keyword arguments for request function

    Returns:
        Request result
    """
    manager = get_rate_limiter_manager()

    # Wait if needed
    await manager.wait_if_needed(endpoint)

    # Make request and record performance
    start_time = time.time()

    try:
        result = await request_func(*args, **kwargs)
        response_time = time.time() - start_time
        manager.record_performance(endpoint, response_time, False)
        return result

    except Exception as e:
        response_time = time.time() - start_time
        manager.record_performance(endpoint, response_time, True)
        raise e
