"""
Argus Ultimate - Resource Manager
Comprehensive memory and resource management utilities.
"""

import logging
import gc
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import psutil
import tracemalloc

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    open_files: int = 0
    threads: int = 0
    objects_tracked: int = 0

@dataclass
class ResourceLeak:
    """Information about a detected resource leak"""
    resource_type: str
    location: str
    count: int
    size_mb: float
    first_detected: float
    last_detected: float

class ResourceManager:
    """Comprehensive resource management and leak detection"""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self._tracked_objects: Dict[str, Set[weakref.ref]] = {}
        self._resource_leaks: List[ResourceLeak] = []
        self._cleanup_callbacks: List[Callable] = []
        self._lock = threading.RLock()
        self._process = psutil.Process()
        self._start_time = time.time()
        
        # Enable memory tracking if available
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
        
        logger.info("ResourceManager initialized with monitoring: %s", enable_monitoring)
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics"""
        try:
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = self._process.memory_percent()
            cpu_percent = self._process.cpu_percent()
            open_files = len(self._process.open_files())
            threads = self._process.num_threads()
            
            # Get tracked objects count
            objects_tracked = 0
            for obj_set in self._tracked_objects.values():
                objects_tracked += len([ref for ref in obj_set if ref() is not None])
            
            return ResourceMetrics(
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                open_files=open_files,
                threads=threads,
                objects_tracked=objects_tracked
            )
        except Exception as e:
            logger.warning(f"Failed to get resource metrics: {e}")
            return ResourceMetrics()
    
    def track_object(self, obj: Any, category: str = "default") -> None:
        """Track an object for leak detection"""
        if not self.enable_monitoring:
            return
        
        try:
            with self._lock:
                if category not in self._tracked_objects:
                    self._tracked_objects[category] = set()
                
                # Use weak reference to avoid preventing garbage collection
                weak_ref = weakref.ref(obj, self._object_deleted_callback)
                self._tracked_objects[category].add(weak_ref)
        except Exception as e:
            logger.debug(f"Failed to track object in category {category}: {e}")
    
    def _object_deleted_callback(self, weak_ref: weakref.ref) -> None:
        """Callback when a tracked object is deleted"""
        try:
            with self._lock:
                for category, obj_set in self._tracked_objects.items():
                    if weak_ref in obj_set:
                        obj_set.remove(weak_ref)
                        break
        except Exception:
            pass  # Object cleanup, ignore errors
    
    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a cleanup callback to be called on shutdown"""
        self._cleanup_callbacks.append(callback)
    
    def detect_leaks(self, threshold_mb: float = 10.0, threshold_objects: int = 100) -> List[ResourceLeak]:
        """Detect potential resource leaks"""
        if not self.enable_monitoring:
            return []
        
        leaks = []
        current_time = time.time()
        
        try:
            with self._lock:
                for category, obj_set in self._tracked_objects.items():
                    # Count live objects
                    live_objects = [ref for ref in obj_set if ref() is not None]
                    count = len(live_objects)
                    
                    if count > threshold_objects:
                        # Estimate memory usage
                        size_mb = self._estimate_category_memory(live_objects)
                        
                        if size_mb > threshold_mb:
                            leak = ResourceLeak(
                                resource_type=category,
                                location=f"tracked_objects[{category}]",
                                count=count,
                                size_mb=size_mb,
                                first_detected=current_time,
                                last_detected=current_time
                            )
                            leaks.append(leak)
                            logger.warning(f"Potential leak detected: {category} - {count} objects, {size_mb:.1f}MB")
        
        except Exception as e:
            logger.error(f"Error during leak detection: {e}")
        
        return leaks
    
    def _estimate_category_memory(self, live_objects: List[weakref.ref]) -> float:
        """Estimate memory usage for a category of objects"""
        try:
            total_size = 0
            sample_size = min(10, len(live_objects))
            
            for ref in live_objects[:sample_size]:
                obj = ref()
                if obj is not None:
                    try:
                        # Use sys.getsizeof if available
                        import sys
                        size = sys.getsizeof(obj)
                        total_size += size
                    except:
                        # Fallback estimation
                        total_size += 1024  # 1KB per object estimate
            
            # Extrapolate to all objects
            if len(live_objects) > 0:
                avg_size = total_size / sample_size
                estimated_total = avg_size * len(live_objects)
                return estimated_total / 1024 / 1024  # Convert to MB
            
            return 0.0
        except Exception:
            return 0.0
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics"""
        try:
            # Get metrics before GC
            before_metrics = self.get_current_metrics()
            
            # Force garbage collection
            collected = gc.collect()
            
            # Get metrics after GC
            after_metrics = self.get_current_metrics()
            
            stats = {
                "objects_collected": collected,
                "memory_freed_mb": before_metrics.memory_mb - after_metrics.memory_mb,
                "before_metrics": before_metrics,
                "after_metrics": after_metrics
            }
            
            logger.info(f"Garbage collection completed: {collected} objects, "
                       f"{stats['memory_freed_mb']:.1f}MB freed")
            
            return stats
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return {"error": str(e)}
    
    def cleanup_resources(self) -> None:
        """Clean up all tracked resources"""
        logger.info("Starting resource cleanup...")
        
        try:
            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            # Clear tracked objects
            with self._lock:
                self._tracked_objects.clear()
            
            # Force garbage collection
            self.force_garbage_collection()
            
            # Stop memory tracing
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    @contextmanager
    def managed_resource(self, resource: Any, category: str = "default"):
        """Context manager for managed resources"""
        self.track_object(resource, category)
        try:
            yield resource
        finally:
            # Resource will be cleaned up automatically when it goes out of scope
            pass
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get a detailed memory snapshot"""
        try:
            if not tracemalloc.is_tracing():
                return {"error": "Memory tracing not enabled"}
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            
            # Get top memory allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            return {
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024,
                "top_allocations": [
                    {
                        "file": stat.traceback.format()[-1],
                        "size_mb": stat.size / 1024 / 1024,
                        "count": stat.count
                    }
                    for stat in top_stats
                ]
            }
        except Exception as e:
            logger.error(f"Error getting memory snapshot: {e}")
            return {"error": str(e)}
    
    def monitor_resources(self, interval_seconds: int = 60) -> None:
        """Start resource monitoring in a background thread"""
        if not self.enable_monitoring:
            return
        
        def monitor_loop():
            while self.enable_monitoring:
                try:
                    metrics = self.get_current_metrics()
                    
                    # Log warnings for high resource usage
                    if metrics.memory_mb > 1000:  # 1GB
                        logger.warning(f"High memory usage: {metrics.memory_mb:.1f}MB")
                    
                    if metrics.memory_percent > 90:
                        logger.warning(f"High memory percent: {metrics.memory_percent:.1f}%")
                    
                    if metrics.cpu_percent > 80:
                        logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
                    
                    # Check for leaks periodically
                    if int(time.time()) % 300 == 0:  # Every 5 minutes
                        leaks = self.detect_leaks()
                        if leaks:
                            logger.warning(f"Detected {len(leaks)} potential resource leaks")
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in resource monitoring loop: {e}")
                    time.sleep(interval_seconds)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"Resource monitoring started (interval: {interval_seconds}s)")

# Global resource manager instance
resource_manager = ResourceManager()
