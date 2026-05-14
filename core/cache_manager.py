"""
Cache Manager - Performance Optimization Layer
=============================================

Provides caching for expensive operations to improve performance.
Supports TTL caching, LRU caching, and memoization.
"""

import time
import hashlib
import pickle
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from collections import OrderedDict

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    value: T
    created_at: datetime
    ttl_seconds: Optional[int] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds
    
    def touch(self):
        """Update last accessed time and increment count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class CacheManager:
    """
    Centralized cache manager for performance optimization.
    
    Features:
    - TTL-based expiration
    - LRU eviction
    - Size-based limits
    - Hit/miss statistics
    - Multiple cache namespaces
    """
    
    def __init__(
        self,
        default_ttl: int = 300,  # 5 minutes
        max_size: int = 1000,
        enable_stats: bool = True
    ):
        self._caches: Dict[str, Dict[str, CacheEntry]] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._enable_stats = enable_stats
        self._stats: Dict[str, Dict[str, int]] = {}
        
        logger.info(f"CacheManager initialized (ttl={default_ttl}s, max_size={max_size})")
    
    def get(
        self,
        namespace: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            default: Default value if not found or expired
            
        Returns:
            Cached value or default
        """
        if namespace not in self._caches:
            self._update_stats(namespace, 'miss')
            return default
        
        cache = self._caches[namespace]
        
        if key not in cache:
            self._update_stats(namespace, 'miss')
            return default
        
        entry = cache[key]
        
        # Check expiration
        if entry.is_expired():
            del cache[key]
            self._update_stats(namespace, 'expired')
            return default
        
        # Update access stats
        entry.touch()
        self._update_stats(namespace, 'hit')
        
        return entry.value
    
    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Set value in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
        """
        # Create namespace if doesn't exist
        if namespace not in self._caches:
            self._caches[namespace] = OrderedDict()
            self._stats[namespace] = {'hit': 0, 'miss': 0, 'expired': 0}
        
        cache = self._caches[namespace]
        
        # Check size limit and evict if necessary
        if len(cache) >= self._max_size:
            self._evict_lru(namespace)
        
        # Store entry
        cache[key] = CacheEntry(
            value=value,
            created_at=datetime.utcnow(),
            ttl_seconds=ttl or self._default_ttl
        )
        
        # Move to end (most recently used)
        cache.move_to_end(key)
    
    def delete(self, namespace: str, key: str) -> bool:
        """Delete key from cache. Returns True if key existed."""
        if namespace not in self._caches:
            return False
        
        cache = self._caches[namespace]
        
        if key in cache:
            del cache[key]
            return True
        
        return False
    
    def clear(self, namespace: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            namespace: Namespace to clear (None for all)
        """
        if namespace is None:
            self._caches.clear()
            self._stats.clear()
            logger.info("All caches cleared")
        elif namespace in self._caches:
            self._caches[namespace].clear()
            self._stats[namespace] = {'hit': 0, 'miss': 0, 'expired': 0}
            logger.info(f"Cache cleared: {namespace}")
    
    def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics."""
        if namespace:
            if namespace not in self._stats:
                return {}
            
            stats = self._stats[namespace].copy()
            total = stats['hit'] + stats['miss']
            stats['hit_rate'] = stats['hit'] / total if total > 0 else 0.0
            stats['size'] = len(self._caches.get(namespace, {}))
            return {namespace: stats}
        
        all_stats = {}
        for ns in self._stats:
            all_stats[ns] = self.get_stats(ns)[ns]
        
        return all_stats
    
    def cleanup_expired(self):
        """Remove all expired entries from all caches."""
        total_removed = 0
        
        for namespace, cache in self._caches.items():
            expired_keys = [
                key for key, entry in cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del cache[key]
                total_removed += 1
                self._update_stats(namespace, 'expired')
        
        if total_removed > 0:
            logger.info(f"Cleaned up {total_removed} expired cache entries")
        
        return total_removed
    
    def _evict_lru(self, namespace: str):
        """Evict least recently used entry."""
        cache = self._caches[namespace]
        
        if cache:
            # Remove first item (least recently used)
            oldest_key = next(iter(cache))
            del cache[oldest_key]
    
    def _update_stats(self, namespace: str, event: str):
        """Update cache statistics."""
        if not self._enable_stats:
            return
        
        if namespace not in self._stats:
            self._stats[namespace] = {'hit': 0, 'miss': 0, 'expired': 0}
        
        self._stats[namespace][event] += 1
    
    def cached(
        self,
        namespace: str,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            namespace: Cache namespace
            ttl: Time-to-live in seconds
            key_func: Function to generate cache key from arguments
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default: hash of arguments
                    key_data = pickle.dumps((args, kwargs))
                    cache_key = hashlib.md5(key_data).hexdigest()
                
                # Try to get from cache
                cached_value = self.get(namespace, cache_key)
                
                if cached_value is not None:
                    return cached_value
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(namespace, cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def memoize(
        self,
        maxsize: int = 128,
        typed: bool = False
    ):
        """
        Simple memoization decorator (LRU cache).
        
        Args:
            maxsize: Maximum cache size
            typed: If True, cache separately for different types
        """
        def decorator(func: Callable) -> Callable:
            cache = OrderedDict()
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                if typed:
                    key = tuple(
                        (type(arg), arg) for arg in args
                    ) + tuple(
                        (type(v), k, v) for k, v in sorted(kwargs.items())
                    )
                else:
                    key = (args, tuple(sorted(kwargs.items())))
                
                # Check cache
                if key in cache:
                    # Move to end (most recently used)
                    cache.move_to_end(key)
                    return cache[key]
                
                # Call function
                result = func(*args, **kwargs)
                
                # Store in cache
                cache[key] = result
                
                # Evict oldest if over limit
                if len(cache) > maxsize:
                    cache.popitem(last=False)
                
                return result
            
            # Attach cache clear function
            wrapper.cache_clear = lambda: cache.clear()
            wrapper.cache_info = lambda: {
                'size': len(cache),
                'maxsize': maxsize
            }
            
            return wrapper
        return decorator


# Global cache manager instance
cache = CacheManager()


def get_cache() -> CacheManager:
    """Get global cache manager instance."""
    return cache


# Convenience decorators
def cached(
    namespace: str = "default",
    ttl: int = 300,
    key_func: Optional[Callable] = None
):
    """Decorator for caching with global cache manager."""
    return cache.cached(namespace, ttl, key_func)


def memoize(maxsize: int = 128, typed: bool = False):
    """Simple memoization decorator."""
    return cache.memoize(maxsize, typed)


# Predefined namespaces for common use cases
CACHE_NAMESPACES = {
    'market_data': 'Market data (prices, OHLCV)',
    'indicators': 'Technical indicators',
    'models': 'ML model predictions',
    'risk': 'Risk calculations',
    'portfolio': 'Portfolio calculations',
    'execution': 'Execution results',
    'config': 'Configuration values'
}


class CachedProperty:
    """
    Property decorator that caches the result.
    
    Usage:
        class MyClass:
            @CachedProperty(ttl=60)
            def expensive_property(self):
                return expensive_calculation()
    """
    
    def __init__(self, func: Callable, ttl: int = 60):
        self.func = func
        self.ttl = ttl
        self.cache: Dict[int, tuple] = {}  # instance id -> (value, timestamp)
        self.name = func.__name__
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        instance_id = id(instance)
        now = time.time()
        
        # Check cache
        if instance_id in self.cache:
            value, timestamp = self.cache[instance_id]
            if now - timestamp < self.ttl:
                return value
        
        # Calculate and cache
        value = self.func(instance)
        self.cache[instance_id] = (value, now)
        
        return value
    
    def __set__(self, instance, value):
        raise AttributeError("can't set attribute")
    
    def __delete__(self, instance):
        instance_id = id(instance)
        if instance_id in self.cache:
            del self.cache[instance_id]


# Usage example
def cached_property(ttl: int = 60):
    """Create a cached property with specified TTL."""
    def decorator(func: Callable):
        return CachedProperty(func, ttl)
    return decorator
