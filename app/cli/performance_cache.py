"""
CLI Performance Optimization Cache

Enhanced caching system to achieve <500ms CLI response times by:
1. SimpleOrchestrator singleton caching
2. Configuration service caching with TTL
3. Connection pool management
4. Lazy import optimization
5. Memory-efficient storage

Target Performance Improvements:
- From 700-820ms to <500ms average execution time
- Cold start: <200ms
- Warm start: <100ms
- SimpleOrchestrator initialization: <50ms (cached)
"""

import time
import threading
import weakref
import sys
import os
from typing import Optional, Any, Dict, TYPE_CHECKING
from pathlib import Path
import json
import psutil

if TYPE_CHECKING:
    from ..core.simple_orchestrator import SimpleOrchestrator
    from ..core.configuration_service import ApplicationConfiguration

# Performance cache TTL settings
CONFIG_CACHE_TTL = 60        # 1 minute for config
ORCHESTRATOR_CACHE_TTL = 300 # 5 minutes for orchestrator
CONNECTION_CACHE_TTL = 600   # 10 minutes for connections


class PerformanceCacheEntry:
    """Cache entry with TTL and performance tracking."""
    
    def __init__(self, data: Any, ttl: float):
        self.data = data
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry."""
        return time.time() - self.created_at
    
    def access(self) -> Any:
        """Access cached data and update statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.data


class OrchestratorSingleton:
    """
    Thread-safe singleton for SimpleOrchestrator with lifecycle management.
    
    This ensures the heavy SimpleOrchestrator initialization (600ms+) only
    happens once per CLI session, dramatically improving performance.
    """
    
    _instance: Optional['SimpleOrchestrator'] = None
    _lock = threading.Lock()
    _created_at: Optional[float] = None
    _initialization_time: Optional[float] = None
    
    @classmethod
    def get_instance(cls, force_reinit: bool = False) -> 'SimpleOrchestrator':
        """Get cached SimpleOrchestrator instance with <50ms response time."""
        # Fast path for already initialized instance
        if cls._instance is not None and not force_reinit:
            return cls._instance
        
        with cls._lock:
            # Double-check locking pattern
            if cls._instance is not None and not force_reinit:
                return cls._instance
            
            # Performance measurement
            start_time = time.time()
            
            try:
                # Import only when needed to reduce startup time
                from ..core.simple_orchestrator import SimpleOrchestrator
                
                # Create lightweight instance with minimal dependencies
                cls._instance = SimpleOrchestrator(
                    enable_production_plugin=False  # Disable heavy plugins for CLI
                )
                
                cls._created_at = start_time
                cls._initialization_time = time.time() - start_time
                
                # Track performance metrics
                if cls._initialization_time > 0.05:  # 50ms threshold
                    print(f"⚠️ Orchestrator initialization took {cls._initialization_time:.3f}s")
                
            except ImportError as e:
                print(f"⚠️ Failed to import SimpleOrchestrator: {e}")
                return None
            except Exception as e:
                print(f"⚠️ Failed to create SimpleOrchestrator: {e}")
                return None
            
            return cls._instance
    
    @classmethod
    async def get_initialized_instance(cls) -> 'SimpleOrchestrator':
        """Get fully initialized SimpleOrchestrator instance."""
        instance = cls.get_instance()
        if not instance:
            return None
        
        # Only initialize if not already done
        if not getattr(instance, '_initialized', False):
            init_start = time.time()
            try:
                await instance.initialize()
                init_time = time.time() - init_start
                
                if init_time > 0.1:  # 100ms threshold
                    print(f"⚠️ Orchestrator full initialization took {init_time:.3f}s")
                    
            except Exception as e:
                print(f"⚠️ Failed to initialize orchestrator: {e}")
                return None
        
        return instance
    
    @classmethod
    def is_healthy(cls) -> bool:
        """Check if cached instance is still healthy."""
        if cls._instance is None:
            return False
        
        # Check if instance is too old (>10 minutes)
        if cls._created_at and time.time() - cls._created_at > 600:
            return False
        
        return True
    
    @classmethod
    def invalidate(cls):
        """Invalidate cached orchestrator instance."""
        with cls._lock:
            if cls._instance:
                # Graceful shutdown attempt
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(cls._instance.shutdown())
                    loop.close()
                except:
                    pass  # Best effort cleanup
                
                cls._instance = None
                cls._created_at = None
                cls._initialization_time = None
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'cached': cls._instance is not None,
            'healthy': cls.is_healthy(),
            'age_seconds': time.time() - cls._created_at if cls._created_at else None,
            'init_time_ms': cls._initialization_time * 1000 if cls._initialization_time else None,
            'initialized': getattr(cls._instance, '_initialized', False) if cls._instance else False
        }


class CLIPerformanceCache:
    """
    Enhanced performance cache for CLI operations targeting <500ms response times.
    
    Features:
    - SimpleOrchestrator singleton caching
    - Configuration caching with TTL
    - Connection pool reuse
    - Memory usage tracking
    - Performance metrics
    """
    
    def __init__(self):
        # Core caches
        self._cache: Dict[str, PerformanceCacheEntry] = {}
        self._cache_lock = threading.RLock()
        
        # Performance tracking
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_bytes': 0
        }
        
        # Startup time tracking
        self._startup_time = time.time()
        
    def get_cached_config(self):
        """Get cached configuration service with <20ms response time."""
        cache_key = "configuration_service"
        
        with self._cache_lock:
            entry = self._cache.get(cache_key)
            
            # Cache hit - fast path
            if entry and not entry.is_expired:
                self._stats['hits'] += 1
                return entry.access()
        
        # Cache miss - load configuration
        self._stats['misses'] += 1
        
        try:
            start_time = time.time()
            
            # Try lightweight configuration first for better CLI performance
            try:
                from .lightweight_config import get_fast_config_service
                config_service = get_fast_config_service()
                load_time = time.time() - start_time
                
                # Log performance for ultra-fast loads
                if load_time > 0.02:  # 20ms threshold
                    print(f"⚠️ Fast config loading took {load_time:.3f}s")
                elif load_time < 0.005:  # Under 5ms is excellent
                    print(f"⚡ Lightning config load: {load_time*1000:.1f}ms")
                    
            except ImportError:
                # Fallback to full configuration service
                from ..core.configuration_service import get_configuration_service
                config_service = get_configuration_service()
                load_time = time.time() - start_time
                
                if load_time > 0.02:  # 20ms threshold
                    print(f"⚠️ Full config loading took {load_time:.3f}s")
            
            # Cache for fast future access
            with self._cache_lock:
                self._cache[cache_key] = PerformanceCacheEntry(
                    config_service, 
                    CONFIG_CACHE_TTL
                )
            
            return config_service
            
        except ImportError as e:
            print(f"⚠️ Configuration service not available: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Failed to load configuration: {e}")
            return None
    
    def get_cached_orchestrator(self, initialized: bool = False) -> 'SimpleOrchestrator':
        """Get cached SimpleOrchestrator with <50ms response time."""
        if initialized:
            # This requires async, so we return a coroutine
            return OrchestratorSingleton.get_initialized_instance()
        else:
            # Fast synchronous access
            return OrchestratorSingleton.get_instance()
    
    def get_or_cache(self, key: str, factory_func, ttl: float = 300) -> Any:
        """Generic cache with factory function."""
        with self._cache_lock:
            entry = self._cache.get(key)
            
            if entry and not entry.is_expired:
                self._stats['hits'] += 1
                return entry.access()
        
        # Cache miss
        self._stats['misses'] += 1
        
        try:
            data = factory_func()
            
            with self._cache_lock:
                self._cache[key] = PerformanceCacheEntry(data, ttl)
            
            return data
            
        except Exception as e:
            print(f"⚠️ Factory function failed for {key}: {e}")
            return None
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and free memory."""
        expired_count = 0
        
        with self._cache_lock:
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                expired_count += 1
                self._stats['evictions'] += 1
        
        return expired_count
    
    def clear_cache(self):
        """Clear all caches - useful for testing or forced refresh."""
        with self._cache_lock:
            self._cache.clear()
        
        # Also clear orchestrator singleton
        OrchestratorSingleton.invalidate()
        
        # Reset stats
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_bytes': 0
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        uptime = time.time() - self._startup_time
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests) if total_requests > 0 else 0.0
        
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'cache_stats': {
                **self._stats,
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache)
            },
            'orchestrator_stats': OrchestratorSingleton.get_stats(),
            'system_stats': {
                'uptime_seconds': uptime,
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'pid': process.pid
            },
            'performance_targets': {
                'target_response_time_ms': 500,
                'config_load_target_ms': 20,
                'orchestrator_init_target_ms': 50
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and cleanup."""
        expired_count = self.cleanup_expired()
        orchestrator_healthy = OrchestratorSingleton.is_healthy()
        
        return {
            'status': 'healthy',
            'expired_entries_cleaned': expired_count,
            'orchestrator_healthy': orchestrator_healthy,
            'cache_size': len(self._cache),
            'uptime_seconds': time.time() - self._startup_time
        }


# Global enhanced cache instance
_cli_cache = CLIPerformanceCache()


def get_cached_config():
    """Get cached configuration service for CLI commands."""
    return _cli_cache.get_cached_config()


def get_cached_orchestrator(initialized: bool = False):
    """Get cached SimpleOrchestrator for CLI commands."""
    return _cli_cache.get_cached_orchestrator(initialized)


async def get_cached_initialized_orchestrator():
    """Get cached and initialized SimpleOrchestrator for CLI commands."""
    return await _cli_cache.get_cached_orchestrator(initialized=True)


def clear_cli_cache():
    """Clear CLI performance cache."""
    _cli_cache.clear_cache()


def get_cli_performance_metrics():
    """Get CLI performance metrics."""
    return _cli_cache.get_performance_metrics()


def perform_cli_health_check():
    """Perform CLI performance health check."""
    return _cli_cache.health_check()


# Performance measurement decorators
def measure_cli_performance(command_name: str):
    """Decorator to measure CLI command performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance results
                if execution_time > 0.5:  # Above 500ms target
                    print(f"🐌 {command_name} took {execution_time:.3f}s (target: <0.5s)")
                elif execution_time < 0.1:  # Excellent performance
                    print(f"⚡ {command_name} completed in {execution_time:.3f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"❌ {command_name} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def measure_async_cli_performance(command_name: str):
    """Decorator to measure async CLI command performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance results
                if execution_time > 0.5:  # Above 500ms target
                    print(f"🐌 {command_name} took {execution_time:.3f}s (target: <0.5s)")
                elif execution_time < 0.1:  # Excellent performance
                    print(f"⚡ {command_name} completed in {execution_time:.3f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"❌ {command_name} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


# Context manager for performance tracking
class CLIPerformanceTracker:
    """Context manager to track CLI command performance."""
    
    def __init__(self, command_name: str, target_ms: float = 500):
        self.command_name = command_name
        self.target_ms = target_ms
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        # Perform health check and cleanup
        _cli_cache.health_check()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        execution_time_ms = (self.end_time - self.start_time) * 1000
        
        # Log results based on performance
        if exc_type is None:
            if execution_time_ms > self.target_ms:
                print(f"🐌 {self.command_name}: {execution_time_ms:.1f}ms (target: <{self.target_ms}ms)")
            elif execution_time_ms < 100:
                print(f"⚡ {self.command_name}: {execution_time_ms:.1f}ms")
            # Optimal performance (100-500ms) - silent success
        else:
            print(f"❌ {self.command_name} failed after {execution_time_ms:.1f}ms")
    
    @property
    def execution_time_ms(self) -> Optional[float]:
        """Get execution time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None