"""
Mobile API Cache and Performance Optimization Layer
for LeanVibe Agent Hive 2.0

Provides intelligent caching and performance optimizations specifically
designed for mobile dashboard interfaces with <5ms response time targets.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
import structlog
import hashlib

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """Cache entry with mobile-optimized metadata."""
    key: str
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    mobile_optimized: bool = False
    priority: str = "medium"  # critical, high, medium, low
    size_bytes: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for mobile API optimization."""
    cache_hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    mobile_optimization_score: float = 0.0
    alert_relevance_score: float = 0.0
    total_requests: int = 0
    cache_size_mb: float = 0.0


class MobileAPICache:
    """High-performance cache optimized for mobile dashboard APIs."""
    
    def __init__(self, max_size_mb: float = 50, default_ttl_seconds: int = 30):
        self.max_size_mb = max_size_mb
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_times: Dict[str, List[float]] = {}
        self._performance_metrics = PerformanceMetrics()
        self._lock = None  # Will be initialized on first use
        self._cleanup_task = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure cache is properly initialized with async components."""
        if not self._initialized:
            self._lock = asyncio.Lock()
            # Start background cleanup task only if event loop is running
            try:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            except RuntimeError:
                # No event loop running - cleanup task will be started when needed
                pass
            self._initialized = True
    
    async def get(self, key: str, mobile_context: bool = False) -> Optional[Any]:
        """Get cached value with mobile optimization context."""
        await self._ensure_initialized()
        async with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            
            # Check expiration
            if datetime.utcnow() > entry.expires_at:
                del self._cache[key]
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            
            # Track performance for mobile optimization
            if mobile_context and not entry.mobile_optimized:
                logger.debug("Cache hit for non-mobile-optimized content", key=key)
            
            logger.debug("Cache hit", key=key, mobile_context=mobile_context)
            return entry.data
    
    async def set(
        self, 
        key: str, 
        data: Any, 
        ttl_seconds: Optional[int] = None,
        mobile_optimized: bool = False,
        priority: str = "medium"
    ) -> None:
        """Set cached value with mobile optimization flags."""
        await self._ensure_initialized()
        async with self._lock:
            ttl = ttl_seconds or self.default_ttl_seconds
            
            # Calculate data size
            data_str = json.dumps(data) if not isinstance(data, str) else data
            size_bytes = len(data_str.encode('utf-8'))
            
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=ttl),
                mobile_optimized=mobile_optimized,
                priority=priority,
                size_bytes=size_bytes
            )
            
            # Check cache size limits
            await self._ensure_space_available(size_bytes)
            
            self._cache[key] = entry
            logger.debug("Cache set", key=key, ttl=ttl, mobile_optimized=mobile_optimized)
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug("Cache invalidated", key=key)
                return True
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        async with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
            
            logger.debug("Cache pattern invalidated", pattern=pattern, count=len(keys_to_remove))
            return len(keys_to_remove)
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl_seconds: Optional[int] = None,
        mobile_optimized: bool = False,
        priority: str = "medium"
    ) -> Any:
        """Get from cache or set using factory function."""
        data = await self.get(key, mobile_context=mobile_optimized)
        if data is not None:
            return data
        
        # Generate new data
        start_time = time.time()
        data = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        generation_time = (time.time() - start_time) * 1000
        
        # Cache the result
        await self.set(key, data, ttl_seconds, mobile_optimized, priority)
        
        logger.debug("Cache miss - generated new data", 
                    key=key, generation_time_ms=generation_time)
        return data
    
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        async with self._lock:
            total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
            cache_size_mb = total_size_bytes / (1024 * 1024)
            
            # Calculate hit rate (simplified)
            mobile_optimized_entries = len([e for e in self._cache.values() if e.mobile_optimized])
            mobile_optimization_score = mobile_optimized_entries / max(len(self._cache), 1)
            
            # Calculate average access rate for recently accessed items
            recent_entries = [e for e in self._cache.values() 
                            if e.last_accessed and e.last_accessed > datetime.utcnow() - timedelta(minutes=5)]
            avg_access_count = sum(e.access_count for e in recent_entries) / max(len(recent_entries), 1)
            
            self._performance_metrics.cache_size_mb = cache_size_mb
            self._performance_metrics.mobile_optimization_score = mobile_optimization_score
            self._performance_metrics.total_requests = sum(e.access_count for e in self._cache.values())
            
            return self._performance_metrics
    
    async def optimize_for_mobile(self) -> Dict[str, Any]:
        """Optimize cache contents for mobile performance."""
        async with self._lock:
            optimization_results = {
                "entries_optimized": 0,
                "entries_removed": 0,
                "space_freed_mb": 0.0,
                "mobile_priority_promoted": 0
            }
            
            # Remove low-priority, rarely accessed entries
            entries_to_remove = []
            for key, entry in self._cache.items():
                if (entry.priority == "low" and 
                    entry.access_count < 2 and 
                    entry.created_at < datetime.utcnow() - timedelta(minutes=10)):
                    entries_to_remove.append(key)
            
            freed_bytes = 0
            for key in entries_to_remove:
                freed_bytes += self._cache[key].size_bytes
                del self._cache[key]
                optimization_results["entries_removed"] += 1
            
            optimization_results["space_freed_mb"] = freed_bytes / (1024 * 1024)
            
            # Promote frequently accessed mobile content
            for entry in self._cache.values():
                if (entry.access_count > 5 and 
                    not entry.mobile_optimized and 
                    "mobile" in entry.key.lower()):
                    entry.mobile_optimized = True
                    entry.priority = "high"
                    optimization_results["mobile_priority_promoted"] += 1
            
            logger.info("Cache optimized for mobile", **optimization_results)
            return optimization_results
    
    async def _ensure_space_available(self, required_bytes: int) -> None:
        """Ensure sufficient cache space is available."""
        current_size = sum(entry.size_bytes for entry in self._cache.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if current_size + required_bytes <= max_size_bytes:
            return
        
        # Remove least recently used entries
        entries_by_access = sorted(
            self._cache.items(),
            key=lambda x: (x[1].last_accessed or x[1].created_at, x[1].priority == "critical")
        )
        
        freed_bytes = 0
        for key, entry in entries_by_access:
            if entry.priority != "critical":  # Never remove critical entries
                freed_bytes += entry.size_bytes
                del self._cache[key]
                
                if current_size - freed_bytes + required_bytes <= max_size_bytes:
                    break
        
        logger.debug("Cache space freed", freed_mb=freed_bytes / (1024 * 1024))
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                async with self._lock:
                    now = datetime.utcnow()
                    expired_keys = [
                        key for key, entry in self._cache.items() 
                        if now > entry.expires_at
                    ]
                    
                    for key in expired_keys:
                        del self._cache[key]
                    
                    if expired_keys:
                        logger.debug("Cleaned up expired cache entries", count=len(expired_keys))
                
                # Optimize for mobile every 5 minutes
                if int(time.time()) % 300 == 0:
                    await self.optimize_for_mobile()
                    
            except Exception as e:
                logger.error("Cache cleanup error", error=str(e))
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate a consistent cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache cleared", entries_removed=count)
            return count
    
    async def stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        async with self._lock:
            mobile_optimized = len([e for e in self._cache.values() if e.mobile_optimized])
            by_priority = {}
            for entry in self._cache.values():
                by_priority[entry.priority] = by_priority.get(entry.priority, 0) + 1
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                "total_entries": len(self._cache),
                "mobile_optimized": mobile_optimized,
                "mobile_optimization_percentage": (mobile_optimized / max(len(self._cache), 1)) * 100,
                "entries_by_priority": by_priority,
                "total_size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_mb,
                "utilization_percentage": (total_size / (self.max_size_mb * 1024 * 1024)) * 100,
                "performance_metrics": asdict(await self.get_performance_metrics())
            }


# Global cache instance
_mobile_cache: Optional[MobileAPICache] = None


def get_mobile_cache() -> MobileAPICache:
    """Get the global mobile API cache instance."""
    global _mobile_cache
    if _mobile_cache is None:
        _mobile_cache = MobileAPICache()
    return _mobile_cache


def is_cache_available() -> bool:
    """Check if mobile cache is available without initialization."""
    global _mobile_cache
    return _mobile_cache is not None and _mobile_cache._initialized


async def cache_mobile_api_response(
    cache_key: str,
    response_data: Any,
    ttl_seconds: int = 30,
    priority: str = "medium"
) -> None:
    """Helper function to cache mobile API responses."""
    cache = get_mobile_cache()
    await cache.set(
        cache_key, 
        response_data, 
        ttl_seconds=ttl_seconds,
        mobile_optimized=True,
        priority=priority
    )


async def get_cached_mobile_response(cache_key: str) -> Optional[Any]:
    """Helper function to get cached mobile API responses."""
    cache = get_mobile_cache()
    return await cache.get(cache_key, mobile_context=True)


def mobile_cache_key(*args, **kwargs) -> str:
    """Generate mobile-specific cache key."""
    cache = get_mobile_cache()
    return f"mobile:{cache.generate_key(*args, **kwargs)}"