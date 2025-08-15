"""
Advanced Context Caching System for LeanVibe Agent Hive 2.0

Intelligent caching system with similarity-based retrieval, quality metrics tracking,
and adaptive cache management. Provides efficient storage and retrieval of context
optimization results with smart invalidation and performance monitoring.
"""

import asyncio
import json
import hashlib
import time
import pickle
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

import structlog
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .context_optimizer import ContextRequest, OptimizedContext, RelevanceScore
from .context_assembler import AssembledContext
from .models import FileAnalysisResult

logger = structlog.get_logger()


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class CacheEntryStatus(Enum):
    """Status of cache entries."""
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass
class CacheKey:
    """Structured cache key for context optimization results."""
    request_hash: str
    project_id: str
    task_type: str
    file_count: int
    timestamp: float
    version: str = "1.0"
    
    def to_string(self) -> str:
        """Convert to string representation."""
        return f"{self.project_id}_{self.task_type}_{self.request_hash}_{self.version}"
    
    @classmethod
    def from_request(
        cls,
        context_request: ContextRequest,
        project_id: str,
        file_results: List[FileAnalysisResult]
    ) -> 'CacheKey':
        """Create cache key from context request."""
        # Create deterministic hash from request
        request_data = {
            "task_description": context_request.task_description,
            "task_type": context_request.task_type.value,
            "files_mentioned": sorted(context_request.files_mentioned),
            "context_preferences": context_request.context_preferences,
            "file_count": len(file_results),
            "file_hashes": sorted([
                f"{f.file_path}:{f.file_hash}" for f in file_results
                if f.file_hash
            ][:50])  # Limit to 50 files for performance
        }
        
        request_str = json.dumps(request_data, sort_keys=True)
        request_hash = hashlib.sha256(request_str.encode()).hexdigest()[:16]
        
        return cls(
            request_hash=request_hash,
            project_id=project_id,
            task_type=context_request.task_type.value,
            file_count=len(file_results),
            timestamp=time.time()
        )


@dataclass
class CacheEntry:
    """Cache entry with metadata and quality metrics."""
    key: CacheKey
    context_result: Union[OptimizedContext, AssembledContext]
    creation_time: datetime
    last_accessed: datetime
    access_count: int
    quality_score: float
    similarity_vector: Optional[np.ndarray]
    invalidation_triggers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_fresh(self, max_age_hours: int = 24) -> bool:
        """Check if entry is fresh."""
        age = datetime.utcnow() - self.creation_time
        return age < timedelta(hours=max_age_hours)
    
    def is_valid(self, file_changes: Set[str] = None) -> bool:
        """Check if entry is still valid."""
        if file_changes:
            # Check if any invalidation triggers are in file changes
            for trigger in self.invalidation_triggers:
                if any(trigger in change for change in file_changes):
                    return False
        return True
    
    def calculate_age_hours(self) -> float:
        """Calculate age in hours."""
        age = datetime.utcnow() - self.creation_time
        return age.total_seconds() / 3600


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    similarity_hits: int = 0
    entries_created: int = 0
    entries_invalidated: int = 0
    entries_expired: int = 0
    total_size_bytes: int = 0
    avg_retrieval_time_ms: float = 0.0
    avg_quality_score: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits + self.similarity_hits) / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "similarity_hits": self.similarity_hits,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "entries_created": self.entries_created,
            "entries_invalidated": self.entries_invalidated,
            "entries_expired": self.entries_expired,
            "total_size_bytes": self.total_size_bytes,
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            "avg_quality_score": self.avg_quality_score
        }


@dataclass
class CacheConfiguration:
    """Configuration for context cache."""
    max_memory_entries: int = 1000
    max_disk_entries: int = 10000
    default_ttl_hours: int = 24
    similarity_threshold: float = 0.8
    quality_threshold: float = 0.6
    eviction_policy: str = "lru"  # lru, lfu, quality
    compression_enabled: bool = True
    disk_cache_path: str = "/tmp/context_cache"
    redis_enabled: bool = False
    redis_ttl_seconds: int = 86400  # 24 hours


class ContextCacheManager:
    """
    Advanced context cache manager with intelligent features.
    
    Features:
    - Multi-level caching (memory, Redis, disk)
    - Similarity-based cache retrieval
    - Quality-aware cache management
    - Adaptive cache invalidation
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: CacheConfiguration, redis_client=None):
        """Initialize cache manager."""
        self.config = config
        self.redis_client = redis_client
        
        # Memory cache (LRU)
        self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Similarity index for fast similarity search
        self.similarity_index: Dict[str, np.ndarray] = {}
        self.entry_vectors: Dict[str, np.ndarray] = {}
        
        # Statistics and monitoring
        self.statistics = CacheStatistics()
        self.performance_metrics = defaultdict(list)
        
        # Cache locks for thread safety
        self.cache_lock = asyncio.Lock()
        
        # Initialize disk cache directory
        self._initialize_disk_cache()
    
    async def get_cached_context(
        self,
        cache_key: CacheKey,
        similarity_threshold: Optional[float] = None
    ) -> Optional[Union[OptimizedContext, AssembledContext]]:
        """
        Retrieve cached context with similarity fallback.
        
        Args:
            cache_key: Cache key to look up
            similarity_threshold: Threshold for similarity matching
            
        Returns:
            Cached context result or None if not found
        """
        start_time = time.time()
        self.statistics.total_requests += 1
        
        try:
            async with self.cache_lock:
                # Try exact match first
                exact_result = await self._get_exact_match(cache_key)
                if exact_result:
                    self.statistics.cache_hits += 1
                    retrieval_time = (time.time() - start_time) * 1000
                    self._update_retrieval_metrics(retrieval_time)
                    return exact_result
                
                # Try similarity-based match
                threshold = similarity_threshold or self.config.similarity_threshold
                similar_result = await self._get_similar_match(cache_key, threshold)
                if similar_result:
                    self.statistics.similarity_hits += 1
                    retrieval_time = (time.time() - start_time) * 1000
                    self._update_retrieval_metrics(retrieval_time)
                    return similar_result
                
                # Cache miss
                self.statistics.cache_misses += 1
                return None
                
        except Exception as e:
            logger.error("Cache retrieval failed",
                        cache_key=cache_key.to_string(),
                        error=str(e))
            self.statistics.cache_misses += 1
            return None
    
    async def cache_context(
        self,
        cache_key: CacheKey,
        context_result: Union[OptimizedContext, AssembledContext],
        quality_score: Optional[float] = None
    ) -> bool:
        """
        Cache context optimization result.
        
        Args:
            cache_key: Cache key
            context_result: Context result to cache
            quality_score: Quality score for the result
            
        Returns:
            True if successfully cached
        """
        try:
            async with self.cache_lock:
                # Calculate quality score if not provided
                if quality_score is None:
                    quality_score = self._calculate_quality_score(context_result)
                
                # Skip caching low-quality results
                if quality_score < self.config.quality_threshold:
                    logger.debug("Skipping cache for low quality result",
                               quality_score=quality_score,
                               threshold=self.config.quality_threshold)
                    return False
                
                # Generate similarity vector
                similarity_vector = self._generate_similarity_vector(
                    cache_key, context_result
                )
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    context_result=context_result,
                    creation_time=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    access_count=0,
                    quality_score=quality_score,
                    similarity_vector=similarity_vector,
                    invalidation_triggers=self._extract_invalidation_triggers(context_result),
                    metadata={"cache_level": "memory"}
                )
                
                # Store in appropriate cache levels
                success = await self._store_cache_entry(entry)
                
                if success:
                    self.statistics.entries_created += 1
                    self._update_cache_statistics()
                
                return success
                
        except Exception as e:
            logger.error("Context caching failed",
                        cache_key=cache_key.to_string(),
                        error=str(e))
            return False
    
    async def invalidate_cache(
        self,
        project_id: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        pattern: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries based on criteria.
        
        Args:
            project_id: Project ID to invalidate
            file_paths: File paths that changed
            pattern: Key pattern to match
            
        Returns:
            Number of entries invalidated
        """
        invalidated_count = 0
        
        try:
            async with self.cache_lock:
                file_changes = set(file_paths or [])
                entries_to_remove = []
                
                # Check memory cache
                for key_str, entry in self.memory_cache.items():
                    should_invalidate = False
                    
                    if project_id and entry.key.project_id == project_id:
                        if not file_changes or not entry.is_valid(file_changes):
                            should_invalidate = True
                    
                    if pattern and pattern in key_str:
                        should_invalidate = True
                    
                    if should_invalidate:
                        entries_to_remove.append(key_str)
                
                # Remove invalidated entries
                for key_str in entries_to_remove:
                    await self._remove_cache_entry(key_str)
                    invalidated_count += 1
                
                # Invalidate Redis cache if enabled
                if self.redis_client and project_id:
                    redis_pattern = f"context_cache:{project_id}:*"
                    await self._invalidate_redis_pattern(redis_pattern)
                
                self.statistics.entries_invalidated += invalidated_count
                
                logger.info("Cache invalidation completed",
                           invalidated_count=invalidated_count,
                           project_id=project_id)
                
                return invalidated_count
                
        except Exception as e:
            logger.error("Cache invalidation failed", error=str(e))
            return 0
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """
        Optimize cache performance and storage.
        
        Returns:
            Optimization results and metrics
        """
        try:
            async with self.cache_lock:
                optimization_results = {
                    "before_size": len(self.memory_cache),
                    "expired_removed": 0,
                    "low_quality_removed": 0,
                    "evicted_entries": 0,
                    "after_size": 0
                }
                
                # Remove expired entries
                expired_keys = []
                for key_str, entry in self.memory_cache.items():
                    if not entry.is_fresh(self.config.default_ttl_hours):
                        expired_keys.append(key_str)
                
                for key_str in expired_keys:
                    await self._remove_cache_entry(key_str)
                    optimization_results["expired_removed"] += 1
                
                # Remove low-quality entries if cache is large
                if len(self.memory_cache) > self.config.max_memory_entries * 0.8:
                    low_quality_keys = []
                    for key_str, entry in self.memory_cache.items():
                        if entry.quality_score < self.config.quality_threshold * 1.2:
                            low_quality_keys.append(key_str)
                    
                    # Remove lowest quality entries
                    low_quality_keys.sort(
                        key=lambda k: self.memory_cache[k].quality_score
                    )
                    
                    remove_count = min(
                        len(low_quality_keys),
                        len(self.memory_cache) - self.config.max_memory_entries
                    )
                    
                    for key_str in low_quality_keys[:remove_count]:
                        await self._remove_cache_entry(key_str)
                        optimization_results["low_quality_removed"] += 1
                
                # Apply eviction policy if still over limit
                while len(self.memory_cache) > self.config.max_memory_entries:
                    evicted_key = self._apply_eviction_policy()
                    if evicted_key:
                        await self._remove_cache_entry(evicted_key)
                        optimization_results["evicted_entries"] += 1
                    else:
                        break
                
                optimization_results["after_size"] = len(self.memory_cache)
                
                # Update statistics
                self.statistics.entries_expired += optimization_results["expired_removed"]
                self._update_cache_statistics()
                
                logger.info("Cache optimization completed", **optimization_results)
                
                return optimization_results
                
        except Exception as e:
            logger.error("Cache optimization failed", error=str(e))
            return {}
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        # Update real-time statistics
        self._update_cache_statistics()
        return self.statistics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "cache_statistics": self.statistics.to_dict(),
            "memory_entries": len(self.memory_cache),
            "similarity_index_size": len(self.similarity_index),
            "average_entry_age_hours": self._calculate_average_entry_age(),
            "quality_distribution": self._calculate_quality_distribution(),
            "retrieval_time_percentiles": self._calculate_retrieval_percentiles()
        }
    
    # Private methods
    
    async def _get_exact_match(self, cache_key: CacheKey) -> Optional[Any]:
        """Get exact cache match."""
        key_str = cache_key.to_string()
        
        # Check memory cache first
        if key_str in self.memory_cache:
            entry = self.memory_cache[key_str]
            if entry.is_fresh() and entry.is_valid():
                # Move to end (LRU)
                self.memory_cache.move_to_end(key_str)
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                return entry.context_result
            else:
                # Remove stale entry
                await self._remove_cache_entry(key_str)
        
        # Check Redis cache
        if self.redis_client:
            redis_result = await self._get_from_redis(cache_key)
            if redis_result:
                # Promote to memory cache
                await self._promote_to_memory(cache_key, redis_result)
                return redis_result
        
        # Check disk cache
        disk_result = await self._get_from_disk(cache_key)
        if disk_result:
            # Promote to memory cache
            await self._promote_to_memory(cache_key, disk_result)
            return disk_result
        
        return None
    
    async def _get_similar_match(
        self,
        cache_key: CacheKey,
        threshold: float
    ) -> Optional[Any]:
        """Get similar cache match using vector similarity."""
        if not self.similarity_index:
            return None
        
        try:
            # Generate query vector for the request
            query_vector = self._generate_query_vector(cache_key)
            if query_vector is None:
                return None
            
            # Find most similar cached entry
            best_similarity = 0.0
            best_entry_key = None
            
            for entry_key, entry_vector in self.similarity_index.items():
                similarity = cosine_similarity(
                    query_vector.reshape(1, -1),
                    entry_vector.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_entry_key = entry_key
            
            if best_entry_key and best_entry_key in self.memory_cache:
                entry = self.memory_cache[best_entry_key]
                
                # Check if entry is still valid
                if entry.is_fresh() and entry.is_valid():
                    # Update access metrics
                    entry.last_accessed = datetime.utcnow()
                    entry.access_count += 1
                    self.memory_cache.move_to_end(best_entry_key)
                    
                    logger.debug("Similar cache hit",
                               similarity=best_similarity,
                               original_key=cache_key.to_string(),
                               similar_key=best_entry_key)
                    
                    return entry.context_result
                else:
                    # Remove stale entry
                    await self._remove_cache_entry(best_entry_key)
            
            return None
            
        except Exception as e:
            logger.warning("Similarity search failed", error=str(e))
            return None
    
    async def _store_cache_entry(self, entry: CacheEntry) -> bool:
        """Store cache entry in appropriate levels."""
        key_str = entry.key.to_string()
        
        try:
            # Store in memory cache
            self.memory_cache[key_str] = entry
            
            # Update similarity index
            if entry.similarity_vector is not None:
                self.similarity_index[key_str] = entry.similarity_vector
            
            # Apply memory cache size limit
            while len(self.memory_cache) > self.config.max_memory_entries:
                evicted_key = self._apply_eviction_policy()
                if evicted_key:
                    await self._remove_cache_entry(evicted_key)
                else:
                    break
            
            # Store in Redis if enabled
            if self.redis_client:
                await self._store_in_redis(entry)
            
            # Store in disk cache for persistence
            await self._store_in_disk(entry)
            
            return True
            
        except Exception as e:
            logger.error("Cache entry storage failed",
                        key=key_str,
                        error=str(e))
            return False
    
    async def _remove_cache_entry(self, key_str: str) -> bool:
        """Remove cache entry from all levels."""
        try:
            # Remove from memory
            if key_str in self.memory_cache:
                del self.memory_cache[key_str]
            
            # Remove from similarity index
            if key_str in self.similarity_index:
                del self.similarity_index[key_str]
            
            # Remove from Redis
            if self.redis_client:
                await self._remove_from_redis(key_str)
            
            return True
            
        except Exception as e:
            logger.warning("Cache entry removal failed",
                          key=key_str,
                          error=str(e))
            return False
    
    def _generate_similarity_vector(
        self,
        cache_key: CacheKey,
        context_result: Union[OptimizedContext, AssembledContext]
    ) -> Optional[np.ndarray]:
        """Generate similarity vector for cache entry."""
        try:
            # Extract features for similarity comparison
            features = []
            
            # Task type features
            task_types = ["feature", "bugfix", "refactoring", "analysis", "documentation"]
            task_vector = [1.0 if cache_key.task_type == t else 0.0 for t in task_types]
            features.extend(task_vector)
            
            # File count feature (normalized)
            file_count_normalized = min(1.0, cache_key.file_count / 100.0)
            features.append(file_count_normalized)
            
            # Context result features
            if hasattr(context_result, 'core_files'):
                # OptimizedContext
                core_file_count = len(context_result.core_files)
                supporting_file_count = len(context_result.supporting_files)
                
                features.extend([
                    min(1.0, core_file_count / 20.0),
                    min(1.0, supporting_file_count / 50.0)
                ])
                
                # Average relevance score
                if context_result.core_files:
                    avg_relevance = sum(
                        f.relevance_score for f in context_result.core_files
                    ) / len(context_result.core_files)
                    features.append(avg_relevance)
                else:
                    features.append(0.0)
                
            elif hasattr(context_result, 'layers'):
                # AssembledContext
                layer_count = len(context_result.layers)
                total_files = sum(len(layer.files) for layer in context_result.layers)
                
                features.extend([
                    min(1.0, layer_count / 10.0),
                    min(1.0, total_files / 100.0),
                    0.5  # Default relevance for assembled context
                ])
            
            else:
                # Unknown context type
                features.extend([0.0, 0.0, 0.5])
            
            # Pad or truncate to fixed size
            target_size = 20
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            elif len(features) > target_size:
                features = features[:target_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning("Similarity vector generation failed", error=str(e))
            return None
    
    def _generate_query_vector(self, cache_key: CacheKey) -> Optional[np.ndarray]:
        """Generate query vector for similarity search."""
        try:
            # Similar to _generate_similarity_vector but for query
            features = []
            
            # Task type features
            task_types = ["feature", "bugfix", "refactoring", "analysis", "documentation"]
            task_vector = [1.0 if cache_key.task_type == t else 0.0 for t in task_types]
            features.extend(task_vector)
            
            # File count feature
            file_count_normalized = min(1.0, cache_key.file_count / 100.0)
            features.append(file_count_normalized)
            
            # Default features for unknown query characteristics
            features.extend([0.5] * 8)  # 8 default features
            
            # Pad to target size
            target_size = 20
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            
            return np.array(features[:target_size], dtype=np.float32)
            
        except Exception as e:
            logger.warning("Query vector generation failed", error=str(e))
            return None
    
    def _calculate_quality_score(
        self,
        context_result: Union[OptimizedContext, AssembledContext]
    ) -> float:
        """Calculate quality score for context result."""
        try:
            quality_factors = []
            
            if hasattr(context_result, 'core_files'):
                # OptimizedContext
                if context_result.core_files:
                    # Average relevance score
                    avg_relevance = sum(
                        f.relevance_score for f in context_result.core_files
                    ) / len(context_result.core_files)
                    quality_factors.append(avg_relevance)
                    
                    # Confidence score
                    avg_confidence = sum(
                        f.confidence_score for f in context_result.core_files
                    ) / len(context_result.core_files)
                    quality_factors.append(avg_confidence)
                
                # Context summary quality
                if hasattr(context_result, 'context_summary'):
                    confidence = context_result.context_summary.get('confidence_score', 0.5)
                    quality_factors.append(confidence)
                
            elif hasattr(context_result, 'layers'):
                # AssembledContext
                if context_result.layers:
                    # Average layer quality
                    total_files = sum(len(layer.files) for layer in context_result.layers)
                    if total_files > 0:
                        # Estimate quality based on layer organization
                        organization_score = min(1.0, len(context_result.layers) / 5.0)
                        quality_factors.append(organization_score)
                    
                    # File distribution quality
                    file_counts = [len(layer.files) for layer in context_result.layers]
                    if file_counts:
                        # Good distribution = consistent layer sizes
                        mean_count = sum(file_counts) / len(file_counts)
                        variance = sum((c - mean_count) ** 2 for c in file_counts) / len(file_counts)
                        distribution_score = max(0.0, 1.0 - variance / (mean_count ** 2) if mean_count > 0 else 0.0)
                        quality_factors.append(distribution_score)
            
            # Default quality if no factors
            if not quality_factors:
                return 0.5
            
            # Calculate weighted average
            return sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.warning("Quality score calculation failed", error=str(e))
            return 0.5
    
    def _extract_invalidation_triggers(
        self,
        context_result: Union[OptimizedContext, AssembledContext]
    ) -> List[str]:
        """Extract file paths that would invalidate this cache entry."""
        triggers = []
        
        try:
            if hasattr(context_result, 'core_files'):
                # OptimizedContext
                for file_score in context_result.core_files + context_result.supporting_files:
                    triggers.append(file_score.file_path)
                    
            elif hasattr(context_result, 'layers'):
                # AssembledContext
                for layer in context_result.layers:
                    for file_score in layer.files:
                        triggers.append(file_score.file_path)
            
            return list(set(triggers))  # Remove duplicates
            
        except Exception as e:
            logger.warning("Invalidation trigger extraction failed", error=str(e))
            return []
    
    def _apply_eviction_policy(self) -> Optional[str]:
        """Apply eviction policy to select entry for removal."""
        if not self.memory_cache:
            return None
        
        if self.config.eviction_policy == "lru":
            # Remove least recently used (first item in OrderedDict)
            return next(iter(self.memory_cache))
            
        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used
            min_access_count = min(entry.access_count for entry in self.memory_cache.values())
            for key, entry in self.memory_cache.items():
                if entry.access_count == min_access_count:
                    return key
                    
        elif self.config.eviction_policy == "quality":
            # Remove lowest quality
            min_quality = min(entry.quality_score for entry in self.memory_cache.values())
            for key, entry in self.memory_cache.items():
                if entry.quality_score == min_quality:
                    return key
        
        # Default to LRU
        return next(iter(self.memory_cache))
    
    # Redis operations (if enabled)
    
    async def _get_from_redis(self, cache_key: CacheKey) -> Optional[Any]:
        """Get entry from Redis cache."""
        if not self.redis_client:
            return None
        
        try:
            redis_key = f"context_cache:{cache_key.to_string()}"
            serialized_data = await self.redis_client.get(redis_key)
            
            if serialized_data:
                # Deserialize
                if self.config.compression_enabled:
                    import gzip
                    serialized_data = gzip.decompress(serialized_data)
                
                cache_entry = pickle.loads(serialized_data)
                
                # Check if still valid
                if cache_entry.is_fresh() and cache_entry.is_valid():
                    return cache_entry.context_result
                else:
                    # Remove stale entry
                    await self.redis_client.delete(redis_key)
            
            return None
            
        except Exception as e:
            logger.warning("Redis cache retrieval failed",
                          cache_key=cache_key.to_string(),
                          error=str(e))
            return None
    
    async def _store_in_redis(self, entry: CacheEntry) -> bool:
        """Store entry in Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            redis_key = f"context_cache:{entry.key.to_string()}"
            
            # Serialize
            serialized_data = pickle.dumps(entry)
            
            if self.config.compression_enabled:
                import gzip
                serialized_data = gzip.compress(serialized_data)
            
            # Store with TTL
            await self.redis_client.setex(
                redis_key,
                self.config.redis_ttl_seconds,
                serialized_data
            )
            
            return True
            
        except Exception as e:
            logger.warning("Redis cache storage failed",
                          cache_key=entry.key.to_string(),
                          error=str(e))
            return False
    
    async def _remove_from_redis(self, key_str: str) -> bool:
        """Remove entry from Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            redis_key = f"context_cache:{key_str}"
            await self.redis_client.delete(redis_key)
            return True
            
        except Exception as e:
            logger.warning("Redis cache removal failed",
                          key=key_str,
                          error=str(e))
            return False
    
    async def _invalidate_redis_pattern(self, pattern: str) -> int:
        """Invalidate Redis entries matching pattern."""
        if not self.redis_client:
            return 0
        
        try:
            # Get all keys matching pattern
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Delete all matching keys
                await self.redis_client.delete(*keys)
                return len(keys)
            
            return 0
            
        except Exception as e:
            logger.warning("Redis pattern invalidation failed",
                          pattern=pattern,
                          error=str(e))
            return 0
    
    # Disk operations (for persistence)
    
    def _initialize_disk_cache(self):
        """Initialize disk cache directory."""
        try:
            import os
            os.makedirs(self.config.disk_cache_path, exist_ok=True)
        except Exception as e:
            logger.warning("Disk cache initialization failed", error=str(e))
    
    async def _get_from_disk(self, cache_key: CacheKey) -> Optional[Any]:
        """Get entry from disk cache."""
        try:
            import os
            
            file_path = os.path.join(
                self.config.disk_cache_path,
                f"{cache_key.to_string()}.cache"
            )
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    if self.config.compression_enabled:
                        import gzip
                        with gzip.open(f, 'rb') as gz_f:
                            cache_entry = pickle.load(gz_f)
                    else:
                        cache_entry = pickle.load(f)
                
                # Check if still valid
                if cache_entry.is_fresh() and cache_entry.is_valid():
                    return cache_entry.context_result
                else:
                    # Remove stale file
                    os.remove(file_path)
            
            return None
            
        except Exception as e:
            logger.warning("Disk cache retrieval failed",
                          cache_key=cache_key.to_string(),
                          error=str(e))
            return None
    
    async def _store_in_disk(self, entry: CacheEntry) -> bool:
        """Store entry in disk cache."""
        try:
            import os
            
            file_path = os.path.join(
                self.config.disk_cache_path,
                f"{entry.key.to_string()}.cache"
            )
            
            with open(file_path, 'wb') as f:
                if self.config.compression_enabled:
                    import gzip
                    with gzip.open(f, 'wb') as gz_f:
                        pickle.dump(entry, gz_f)
                else:
                    pickle.dump(entry, f)
            
            return True
            
        except Exception as e:
            logger.warning("Disk cache storage failed",
                          cache_key=entry.key.to_string(),
                          error=str(e))
            return False
    
    async def _promote_to_memory(self, cache_key: CacheKey, context_result: Any) -> bool:
        """Promote cache entry to memory cache."""
        try:
            # Create memory cache entry
            entry = CacheEntry(
                key=cache_key,
                context_result=context_result,
                creation_time=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                quality_score=0.8,  # Assume good quality for promoted entries
                similarity_vector=None,  # Will be generated if needed
                invalidation_triggers=[],
                metadata={"promoted": True}
            )
            
            # Store in memory
            self.memory_cache[cache_key.to_string()] = entry
            
            # Apply size limits
            while len(self.memory_cache) > self.config.max_memory_entries:
                evicted_key = self._apply_eviction_policy()
                if evicted_key:
                    await self._remove_cache_entry(evicted_key)
                else:
                    break
            
            return True
            
        except Exception as e:
            logger.warning("Cache promotion failed",
                          cache_key=cache_key.to_string(),
                          error=str(e))
            return False
    
    # Statistics and monitoring methods
    
    def _update_retrieval_metrics(self, retrieval_time_ms: float):
        """Update retrieval time metrics."""
        self.performance_metrics["retrieval_times"].append(retrieval_time_ms)
        
        # Keep only recent metrics (last 1000)
        if len(self.performance_metrics["retrieval_times"]) > 1000:
            self.performance_metrics["retrieval_times"] = \
                self.performance_metrics["retrieval_times"][-1000:]
        
        # Update average
        times = self.performance_metrics["retrieval_times"]
        self.statistics.avg_retrieval_time_ms = sum(times) / len(times)
    
    def _update_cache_statistics(self):
        """Update cache statistics."""
        # Calculate total size
        self.statistics.total_size_bytes = self._calculate_cache_size()
        
        # Calculate average quality
        if self.memory_cache:
            total_quality = sum(entry.quality_score for entry in self.memory_cache.values())
            self.statistics.avg_quality_score = total_quality / len(self.memory_cache)
    
    def _calculate_cache_size(self) -> int:
        """Calculate approximate cache size in bytes."""
        try:
            import sys
            total_size = 0
            
            # Estimate memory cache size
            for entry in self.memory_cache.values():
                total_size += sys.getsizeof(entry)
                if entry.similarity_vector is not None:
                    total_size += entry.similarity_vector.nbytes
            
            return total_size
            
        except Exception:
            return 0
    
    def _calculate_average_entry_age(self) -> float:
        """Calculate average age of cache entries in hours."""
        if not self.memory_cache:
            return 0.0
        
        total_age = sum(entry.calculate_age_hours() for entry in self.memory_cache.values())
        return total_age / len(self.memory_cache)
    
    def _calculate_quality_distribution(self) -> Dict[str, int]:
        """Calculate quality score distribution."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for entry in self.memory_cache.values():
            if entry.quality_score >= 0.8:
                distribution["high"] += 1
            elif entry.quality_score >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _calculate_retrieval_percentiles(self) -> Dict[str, float]:
        """Calculate retrieval time percentiles."""
        times = self.performance_metrics.get("retrieval_times", [])
        
        if not times:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        return {
            "p50": sorted_times[int(n * 0.5)] if n > 0 else 0.0,
            "p90": sorted_times[int(n * 0.9)] if n > 0 else 0.0,
            "p95": sorted_times[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_times[int(n * 0.99)] if n > 0 else 0.0
        }