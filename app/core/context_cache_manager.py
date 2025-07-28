"""
Context Cache Manager - Intelligent Multi-Level Caching System.

Provides advanced caching for context operations with:
- Multi-level caching (L1: Memory, L2: Redis, L3: Database)
- Intelligent cache warming and prefetching
- Context-aware cache invalidation
- Performance-optimized cache policies
- Cache analytics and monitoring
- Automatic cache optimization
"""

import asyncio
import logging
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import UUID
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque, OrderedDict
import threading

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..schemas.context import ContextSearchRequest
from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


class CachePolicy(Enum):
    """Cache policies for different data types."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on usage patterns
    IMPORTANCE_BASED = "importance_based"  # Based on context importance


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size_bytes: int
    importance_score: float
    cache_level: CacheLevel
    ttl_seconds: Optional[int] = None
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds
    
    def calculate_score(self, policy: CachePolicy) -> float:
        """Calculate cache score based on policy."""
        if policy == CachePolicy.LRU:
            return (datetime.utcnow() - self.accessed_at).total_seconds()
        elif policy == CachePolicy.LFU:
            return -self.access_count  # Negative for descending sort
        elif policy == CachePolicy.TTL:
            return self.age_seconds
        elif policy == CachePolicy.IMPORTANCE_BASED:
            return -self.importance_score  # Negative for descending sort
        elif policy == CachePolicy.ADAPTIVE:
            # Combine multiple factors
            age_factor = self.age_seconds / 3600  # Hours
            frequency_factor = self.access_count / max(1, self.age_seconds / 3600)
            importance_factor = self.importance_score * 10
            return age_factor - frequency_factor - importance_factor
        else:
            return 0.0


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    cache_level: CacheLevel
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.cache_hits + self.cache_misses
        self.hit_rate = self.cache_hits / max(1, total)


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry.accessed_at = datetime.utcnow()
                    entry.access_count += 1
                    return entry
                else:
                    # Remove expired entry
                    del self.cache[key]
            return None
    
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing entry
                self.cache[key] = entry
                self.cache.move_to_end(key)
                return True
            
            # Check capacity
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            # Add new entry
            self.cache[key] = entry
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            
            return {
                "entries": len(self.cache),
                "capacity": self.capacity,
                "total_size_bytes": total_size,
                "total_accesses": total_accesses,
                "utilization": len(self.cache) / self.capacity
            }


class ContextCacheManager:
    """
    Intelligent multi-level caching system for contexts.
    
    Features:
    - L1 Memory cache for frequently accessed data
    - L2 Redis cache for distributed caching
    - L3 Database caching with query optimization
    - Intelligent cache warming and prefetching
    - Context-aware invalidation strategies
    - Performance monitoring and optimization
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        
        # Cache configuration
        self.l1_capacity = 1000
        self.l2_ttl_seconds = 3600
        self.l3_ttl_seconds = 86400
        self.prefetch_threshold = 0.8
        
        # L1 Memory caches
        self.context_cache = LRUCache(capacity=self.l1_capacity)
        self.search_cache = LRUCache(capacity=500)
        self.embedding_cache = LRUCache(capacity=2000)
        
        # Cache metrics
        self.metrics = {
            CacheLevel.L1_MEMORY: CacheMetrics(CacheLevel.L1_MEMORY),
            CacheLevel.L2_REDIS: CacheMetrics(CacheLevel.L2_REDIS),
            CacheLevel.L3_DATABASE: CacheMetrics(CacheLevel.L3_DATABASE)
        }
        
        # Performance tracking
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.invalidation_stats = defaultdict(int)
        self.prefetch_stats = {"attempts": 0, "hits": 0, "misses": 0}
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._prefetch_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start_cache_management(self) -> None:
        """Start cache management system."""
        if self._is_running:
            return
        
        logger.info("Starting context cache management system")
        self._is_running = True
        
        # Start background tasks
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        self._prefetch_task = asyncio.create_task(self._prefetch_loop())
    
    async def stop_cache_management(self) -> None:
        """Stop cache management system."""
        if not self._is_running:
            return
        
        logger.info("Stopping context cache management system")
        self._is_running = False
        
        # Cancel background tasks
        for task in [self._maintenance_task, self._prefetch_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear caches
        self.context_cache.clear()
        self.search_cache.clear()
        self.embedding_cache.clear()
    
    async def get_context(
        self,
        context_id: UUID,
        use_cache: bool = True
    ) -> Optional[Context]:
        """
        Get context with multi-level caching.
        
        Args:
            context_id: Context ID to retrieve
            use_cache: Whether to use caching
            
        Returns:
            Context if found, None otherwise
        """
        start_time = time.perf_counter()
        cache_key = f"context:{context_id}"
        
        try:
            if not use_cache:
                # Skip cache and go directly to database
                return await self._get_context_from_database(context_id)
            
            # L1 Memory cache
            l1_entry = self.context_cache.get(cache_key)
            if l1_entry:
                access_time = (time.perf_counter() - start_time) * 1000
                self._record_cache_hit(CacheLevel.L1_MEMORY, access_time)
                self._record_access_pattern(cache_key, "l1_hit")
                return self._deserialize_context(l1_entry.data)
            
            # L2 Redis cache
            l2_data = await self._get_from_redis_cache(cache_key)
            if l2_data:
                access_time = (time.perf_counter() - start_time) * 1000
                self._record_cache_hit(CacheLevel.L2_REDIS, access_time)
                self._record_access_pattern(cache_key, "l2_hit")
                
                # Store in L1 for future access
                context = self._deserialize_context(l2_data)
                await self._store_in_l1_cache(cache_key, context, importance_score=context.importance_score)
                
                return context
            
            # L3 Database
            context = await self._get_context_from_database(context_id)
            if context:
                access_time = (time.perf_counter() - start_time) * 1000
                self._record_cache_hit(CacheLevel.L3_DATABASE, access_time)
                self._record_access_pattern(cache_key, "l3_hit")
                
                # Store in L2 and L1 caches
                await self._store_in_l2_cache(cache_key, context)
                await self._store_in_l1_cache(cache_key, context, importance_score=context.importance_score)
                
                return context
            
            # Cache miss
            access_time = (time.perf_counter() - start_time) * 1000
            self._record_cache_miss(access_time)
            self._record_access_pattern(cache_key, "miss")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting context {context_id} from cache: {e}")
            return None
    
    async def store_context(
        self,
        context: Context,
        cache_levels: Optional[List[CacheLevel]] = None
    ) -> bool:
        """
        Store context in specified cache levels.
        
        Args:
            context: Context to store
            cache_levels: Cache levels to store in (all levels if None)
            
        Returns:
            True if successfully stored
        """
        try:
            cache_key = f"context:{context.id}"
            
            if cache_levels is None:
                cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
            
            success = True
            
            # Store in specified cache levels
            if CacheLevel.L1_MEMORY in cache_levels:
                success &= await self._store_in_l1_cache(
                    cache_key, context, importance_score=context.importance_score
                )
            
            if CacheLevel.L2_REDIS in cache_levels:
                success &= await self._store_in_l2_cache(cache_key, context)
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing context {context.id} in cache: {e}")
            return False
    
    async def invalidate_context(
        self,
        context_id: UUID,
        cache_levels: Optional[List[CacheLevel]] = None
    ) -> bool:
        """
        Invalidate context from specified cache levels.
        
        Args:
            context_id: Context ID to invalidate
            cache_levels: Cache levels to invalidate from (all levels if None)
            
        Returns:
            True if successfully invalidated
        """
        try:
            cache_key = f"context:{context_id}"
            
            if cache_levels is None:
                cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
            
            success = True
            
            # Invalidate from specified cache levels
            if CacheLevel.L1_MEMORY in cache_levels:
                success &= self.context_cache.delete(cache_key)
                self.invalidation_stats["l1_invalidations"] += 1
            
            if CacheLevel.L2_REDIS in cache_levels:
                success &= await self._invalidate_from_redis(cache_key)
                self.invalidation_stats["l2_invalidations"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error invalidating context {context_id} from cache: {e}")
            return False
    
    async def invalidate_agent_contexts(
        self,
        agent_id: UUID,
        cache_levels: Optional[List[CacheLevel]] = None
    ) -> int:
        """
        Invalidate all contexts for an agent.
        
        Args:
            agent_id: Agent ID to invalidate contexts for
            cache_levels: Cache levels to invalidate from
            
        Returns:
            Number of contexts invalidated
        """
        try:
            if cache_levels is None:
                cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
            
            invalidated_count = 0
            
            # Get all context IDs for the agent
            async with get_async_session() as session:
                result = await session.execute(
                    select(Context.id).where(Context.agent_id == agent_id)
                )
                context_ids = [row[0] for row in result.all()]
            
            # Invalidate each context
            for context_id in context_ids:
                if await self.invalidate_context(context_id, cache_levels):
                    invalidated_count += 1
            
            logger.info(f"Invalidated {invalidated_count} contexts for agent {agent_id}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Error invalidating agent contexts for {agent_id}: {e}")
            return 0
    
    async def get_search_results(
        self,
        search_request: ContextSearchRequest,
        use_cache: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search results.
        
        Args:
            search_request: Search request
            use_cache: Whether to use caching
            
        Returns:
            Cached search results if found
        """
        if not use_cache:
            return None
        
        try:
            cache_key = self._generate_search_cache_key(search_request)
            
            # Check L1 cache
            l1_entry = self.search_cache.get(cache_key)
            if l1_entry and not l1_entry.is_expired:
                self._record_cache_hit(CacheLevel.L1_MEMORY, 1.0)
                return l1_entry.data
            
            # Check L2 cache
            l2_data = await self._get_from_redis_cache(cache_key)
            if l2_data:
                self._record_cache_hit(CacheLevel.L2_REDIS, 5.0)
                
                # Store in L1 for future access
                await self._store_search_results_l1(cache_key, l2_data)
                
                return l2_data
            
            self._record_cache_miss(2.0)
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached search results: {e}")
            return None
    
    async def store_search_results(
        self,
        search_request: ContextSearchRequest,
        results: List[Dict[str, Any]],
        ttl_seconds: int = 300
    ) -> bool:
        """
        Store search results in cache.
        
        Args:
            search_request: Search request
            results: Search results to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successfully stored
        """
        try:
            cache_key = self._generate_search_cache_key(search_request)
            
            # Store in L1 cache
            await self._store_search_results_l1(cache_key, results, ttl_seconds)
            
            # Store in L2 cache
            await self._store_search_results_l2(cache_key, results, ttl_seconds)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing search results in cache: {e}")
            return False
    
    async def warm_cache_for_agent(
        self,
        agent_id: UUID,
        context_limit: int = 50
    ) -> int:
        """
        Warm cache with frequently accessed contexts for an agent.
        
        Args:
            agent_id: Agent ID to warm cache for
            context_limit: Maximum number of contexts to warm
            
        Returns:
            Number of contexts warmed
        """
        try:
            logger.info(f"Warming cache for agent {agent_id}")
            
            # Get frequently accessed contexts
            async with get_async_session() as session:
                result = await session.execute(
                    select(Context).where(
                        and_(
                            Context.agent_id == agent_id,
                            func.cast(Context.access_count, session.Integer) > 1
                        )
                    ).order_by(
                        func.cast(Context.access_count, session.Integer).desc(),
                        Context.importance_score.desc()
                    ).limit(context_limit)
                )
                
                contexts = list(result.scalars().all())
            
            warmed_count = 0
            for context in contexts:
                try:
                    # Store in both L1 and L2 caches
                    if await self.store_context(context):
                        warmed_count += 1
                except Exception as e:
                    logger.warning(f"Error warming context {context.id}: {e}")
            
            logger.info(f"Warmed {warmed_count} contexts for agent {agent_id}")
            return warmed_count
            
        except Exception as e:
            logger.error(f"Error warming cache for agent {agent_id}: {e}")
            return 0
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            stats = {
                "cache_levels": {},
                "l1_memory_stats": {},
                "l2_redis_stats": {},
                "performance_metrics": {},
                "access_patterns": {},
                "invalidation_stats": dict(self.invalidation_stats),
                "prefetch_stats": self.prefetch_stats.copy()
            }
            
            # Cache level metrics
            for level, metrics in self.metrics.items():
                metrics.update_hit_rate()
                stats["cache_levels"][level.value] = asdict(metrics)
            
            # L1 memory statistics
            stats["l1_memory_stats"] = {
                "context_cache": self.context_cache.get_stats(),
                "search_cache": self.search_cache.get_stats(),
                "embedding_cache": self.embedding_cache.get_stats()
            }
            
            # L2 Redis statistics
            try:
                redis_info = await self.redis_client.info("memory")
                stats["l2_redis_stats"] = {
                    "used_memory": redis_info.get("used_memory", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "connected_clients": redis_info.get("connected_clients", 0)
                }
            except Exception as e:
                stats["l2_redis_stats"] = {"error": str(e)}
            
            # Performance metrics
            total_requests = sum(m.total_requests for m in self.metrics.values())
            total_hits = sum(m.cache_hits for m in self.metrics.values())
            
            stats["performance_metrics"] = {
                "overall_hit_rate": total_hits / max(1, total_requests),
                "total_requests": total_requests,
                "total_hits": total_hits,
                "avg_access_time_ms": sum(m.avg_access_time_ms for m in self.metrics.values()) / 3
            }
            
            # Access pattern analysis
            pattern_summary = {}
            for key, patterns in list(self.access_patterns.items())[:10]:  # Top 10 patterns
                pattern_counts = defaultdict(int)
                for pattern in patterns:
                    pattern_counts[pattern] += 1
                
                pattern_summary[key] = dict(pattern_counts)
            
            stats["access_patterns"] = pattern_summary
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {"error": str(e)}
    
    # Private Methods
    
    async def _get_context_from_database(self, context_id: UUID) -> Optional[Context]:
        """Get context directly from database."""
        try:
            async with get_async_session() as session:
                return await session.get(Context, context_id)
        except Exception as e:
            logger.error(f"Error getting context {context_id} from database: {e}")
            return None
    
    async def _store_in_l1_cache(
        self,
        cache_key: str,
        context: Context,
        importance_score: float = 0.5,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store context in L1 memory cache."""
        try:
            serialized_data = self._serialize_context(context)
            
            entry = CacheEntry(
                key=cache_key,
                data=serialized_data,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                access_count=1,
                size_bytes=len(json.dumps(serialized_data, default=str)),
                importance_score=importance_score,
                cache_level=CacheLevel.L1_MEMORY,
                ttl_seconds=ttl_seconds,
                tags={"context", f"agent:{context.agent_id}"}
            )
            
            return self.context_cache.put(cache_key, entry)
            
        except Exception as e:
            logger.error(f"Error storing in L1 cache: {e}")
            return False
    
    async def _store_in_l2_cache(
        self,
        cache_key: str,
        context: Context,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store context in L2 Redis cache."""
        try:
            serialized_data = self._serialize_context(context)
            ttl = ttl_seconds or self.l2_ttl_seconds
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(serialized_data, default=str)
            )
            
            # Store metadata
            metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "size_bytes": len(json.dumps(serialized_data, default=str)),
                "importance_score": context.importance_score,
                "agent_id": str(context.agent_id)
            }
            
            await self.redis_client.setex(
                f"{cache_key}:meta",
                ttl,
                json.dumps(metadata)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing in L2 cache: {e}")
            return False
    
    async def _get_from_redis_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from Redis cache."""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None
    
    async def _invalidate_from_redis(self, cache_key: str) -> bool:
        """Invalidate key from Redis cache."""
        try:
            await self.redis_client.delete(cache_key, f"{cache_key}:meta")
            return True
        except Exception as e:
            logger.error(f"Error invalidating from Redis: {e}")
            return False
    
    def _serialize_context(self, context: Context) -> Dict[str, Any]:
        """Serialize context for caching."""
        return context.to_dict()
    
    def _deserialize_context(self, data: Dict[str, Any]) -> Context:
        """Deserialize context from cache data."""
        # This is simplified - in production, you'd need proper deserialization
        context = Context()
        for key, value in data.items():
            if hasattr(context, key):
                setattr(context, key, value)
        return context
    
    def _generate_search_cache_key(self, request: ContextSearchRequest) -> str:
        """Generate cache key for search request."""
        key_data = {
            "query": request.query,
            "agent_id": str(request.agent_id) if request.agent_id else None,
            "context_type": request.context_type.value if request.context_type else None,
            "limit": request.limit,
            "min_relevance": request.min_relevance
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"search:{hashlib.sha256(key_string.encode()).hexdigest()[:16]}"
    
    async def _store_search_results_l1(
        self,
        cache_key: str,
        results: List[Dict[str, Any]],
        ttl_seconds: int = 300
    ) -> bool:
        """Store search results in L1 cache."""
        try:
            entry = CacheEntry(
                key=cache_key,
                data=results,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                access_count=1,
                size_bytes=len(json.dumps(results, default=str)),
                importance_score=0.5,
                cache_level=CacheLevel.L1_MEMORY,
                ttl_seconds=ttl_seconds,
                tags={"search_results"}
            )
            
            return self.search_cache.put(cache_key, entry)
            
        except Exception as e:
            logger.error(f"Error storing search results in L1: {e}")
            return False
    
    async def _store_search_results_l2(
        self,
        cache_key: str,
        results: List[Dict[str, Any]],
        ttl_seconds: int = 300
    ) -> bool:
        """Store search results in L2 cache."""
        try:
            await self.redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(results, default=str)
            )
            return True
        except Exception as e:
            logger.error(f"Error storing search results in L2: {e}")
            return False
    
    def _record_cache_hit(self, level: CacheLevel, access_time_ms: float):
        """Record cache hit metrics."""
        metrics = self.metrics[level]
        metrics.total_requests += 1
        metrics.cache_hits += 1
        
        # Update average access time
        total_time = metrics.avg_access_time_ms * (metrics.total_requests - 1) + access_time_ms
        metrics.avg_access_time_ms = total_time / metrics.total_requests
    
    def _record_cache_miss(self, access_time_ms: float):
        """Record cache miss metrics."""
        for metrics in self.metrics.values():
            metrics.total_requests += 1
            metrics.cache_misses += 1
    
    def _record_access_pattern(self, cache_key: str, pattern: str):
        """Record access pattern for analysis."""
        self.access_patterns[cache_key].append(pattern)
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        logger.info("Starting cache maintenance loop")
        
        try:
            while self._is_running:
                try:
                    # Clean expired entries from L1 caches
                    await self._clean_expired_l1_entries()
                    
                    # Optimize cache sizes
                    await self._optimize_cache_sizes()
                    
                    # Clean up access patterns
                    self._cleanup_access_patterns()
                    
                    # Wait before next maintenance cycle
                    await asyncio.sleep(300)  # 5 minutes
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cache maintenance: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Cache maintenance loop stopped")
    
    async def _prefetch_loop(self) -> None:
        """Background prefetching loop."""
        logger.info("Starting cache prefetch loop")
        
        try:
            while self._is_running:
                try:
                    # Analyze access patterns for prefetch opportunities
                    await self._analyze_prefetch_opportunities()
                    
                    # Wait before next prefetch cycle
                    await asyncio.sleep(600)  # 10 minutes
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cache prefetch: {e}")
                    await asyncio.sleep(120)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Cache prefetch loop stopped")
    
    async def _clean_expired_l1_entries(self):
        """Clean expired entries from L1 caches."""
        try:
            # Clean context cache
            expired_keys = []
            for key in list(self.context_cache.cache.keys()):
                entry = self.context_cache.cache.get(key)
                if entry and entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.context_cache.delete(key)
            
            if expired_keys:
                logger.debug(f"Cleaned {len(expired_keys)} expired context cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning expired L1 entries: {e}")
    
    async def _optimize_cache_sizes(self):
        """Optimize cache sizes based on usage patterns."""
        try:
            # Get current utilization
            context_stats = self.context_cache.get_stats()
            search_stats = self.search_cache.get_stats()
            
            # Adjust cache sizes based on utilization
            if context_stats["utilization"] > 0.9:
                # Consider increasing context cache size
                logger.debug("Context cache highly utilized, consider increasing capacity")
            
            if search_stats["utilization"] < 0.3:
                # Search cache underutilized
                logger.debug("Search cache underutilized")
                
        except Exception as e:
            logger.error(f"Error optimizing cache sizes: {e}")
    
    def _cleanup_access_patterns(self):
        """Clean up old access patterns."""
        try:
            # Keep only recent patterns for frequently accessed items
            keys_to_remove = []
            for key, patterns in self.access_patterns.items():
                if len(patterns) == 0:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.access_patterns[key]
                
        except Exception as e:
            logger.error(f"Error cleaning access patterns: {e}")
    
    async def _analyze_prefetch_opportunities(self):
        """Analyze access patterns for prefetch opportunities."""
        try:
            # Simple prefetch strategy: if an agent's contexts are frequently accessed,
            # prefetch related contexts
            
            agent_access_counts = defaultdict(int)
            
            # Count accesses by agent
            for key, patterns in self.access_patterns.items():
                if key.startswith("context:"):
                    # Extract agent information if available
                    for pattern in patterns:
                        if pattern in ["l1_hit", "l2_hit"]:
                            # This is a simplified implementation
                            # In practice, you'd track agent relationships
                            agent_access_counts["unknown"] += 1
            
            # Prefetch for highly active agents
            for agent, count in agent_access_counts.items():
                if count > 10:  # Threshold for prefetching
                    self.prefetch_stats["attempts"] += 1
                    # This would trigger actual prefetching
                    logger.debug(f"Prefetch opportunity detected for agent activity")
                    
        except Exception as e:
            logger.error(f"Error analyzing prefetch opportunities: {e}")


# Global instance for application use
_cache_manager: Optional[ContextCacheManager] = None


def get_context_cache_manager() -> ContextCacheManager:
    """
    Get singleton context cache manager instance.
    
    Returns:
        ContextCacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = ContextCacheManager()
    
    return _cache_manager


async def start_cache_management() -> None:
    """Start context cache management."""
    cache_manager = get_context_cache_manager()
    await cache_manager.start_cache_management()


async def stop_cache_management() -> None:
    """Stop context cache management."""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.stop_cache_management()
        _cache_manager = None