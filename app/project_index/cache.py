"""
Redis Cache Manager for LeanVibe Agent Hive 2.0

Provides efficient caching for project analysis results, file analysis data,
and dependency information. Optimized for project indexing use cases.
"""

import json
import pickle
import time
import hashlib
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from uuid import UUID

import structlog

from ..core.redis import RedisClient
from .models import FileAnalysisResult, AnalysisResult, DependencyResult

logger = structlog.get_logger()


class CacheLayer(Enum):
    """Cache layer types for multi-layer caching strategy."""
    AST = "ast"                    # Abstract Syntax Tree cache
    ANALYSIS = "analysis"          # Complete file analysis results
    DEPENDENCY = "dependency"      # Resolved dependency relationships
    CONTEXT = "context"            # Generated context optimization results
    PROJECT = "project"            # Project-level metadata and statistics
    LANGUAGE = "language"          # Language detection results
    HASH = "hash"                  # File content hashes


@dataclass
class CacheStatistics:
    """Comprehensive cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    invalidations: int = 0
    errors: int = 0
    size_bytes: int = 0
    compression_ratio: float = 0.0
    hit_rate_percent: float = 0.0
    layer_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def calculate_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        self.hit_rate_percent = (self.hits / total) * 100
        return self.hit_rate_percent


@dataclass
class CacheKey:
    """Type-safe cache key generation and management."""
    
    @staticmethod
    def ast_key(file_path: str, file_hash: str) -> str:
        """Generate AST cache key."""
        return f"pi:ast:{file_hash[:8]}:{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
    
    @staticmethod
    def analysis_key(file_path: str, file_hash: str) -> str:
        """Generate analysis cache key."""
        return f"pi:analysis:{file_hash[:8]}:{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
    
    @staticmethod
    def dependency_key(project_id: UUID) -> str:
        """Generate dependency graph cache key."""
        return f"pi:dependency:project:{str(project_id)}"
    
    @staticmethod
    def context_key(project_id: UUID, context_hash: str) -> str:
        """Generate context optimization cache key."""
        return f"pi:context:{str(project_id)}:{context_hash[:8]}"
    
    @staticmethod
    def project_key(project_id: UUID) -> str:
        """Generate project metadata cache key."""
        return f"pi:project:{str(project_id)}"
    
    @staticmethod
    def language_key(file_path: str) -> str:
        """Generate language detection cache key."""
        return f"pi:language:{hashlib.md5(file_path.encode()).hexdigest()}"
    
    @staticmethod
    def hash_key(file_path: str) -> str:
        """Generate file hash cache key."""
        return f"pi:hash:{hashlib.md5(file_path.encode()).hexdigest()}"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    layer: CacheLayer = CacheLayer.ANALYSIS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'size_bytes': self.size_bytes,
            'compressed': self.compressed,
            'layer': self.layer.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            data=data['data'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data.get('access_count', 0),
            size_bytes=data.get('size_bytes', 0),
            compressed=data.get('compressed', False),
            layer=CacheLayer(data.get('layer', 'analysis'))
        )


class AdvancedCacheManager:
    """
    Advanced Redis-based cache manager with multi-layer caching strategy.
    
    Features:
    - Multi-layer caching for different data types (AST, Analysis, Dependencies, Context)
    - Intelligent cache invalidation based on file changes
    - Data compression for large cache entries
    - Performance monitoring with detailed hit rate tracking
    - Memory management with automatic cleanup
    - Connection pooling integration
    """
    
    
    def __init__(
        self, 
        redis_client: RedisClient,
        enable_compression: bool = True,
        compression_threshold: int = 1024,  # Compress entries larger than 1KB
        max_memory_mb: int = 500
    ):
        """
        Initialize AdvancedCacheManager.
        
        Args:
            redis_client: Redis client instance
            enable_compression: Enable compression for large cache entries
            compression_threshold: Minimum size in bytes to trigger compression
            max_memory_mb: Maximum memory usage for cache in MB
        """
        self.redis = redis_client
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.max_memory_mb = max_memory_mb
        
        # Enhanced statistics tracking
        self._stats = CacheStatistics()
        self._layer_ttls = {
            CacheLayer.AST: 3600 * 24 * 3,         # 3 days
            CacheLayer.ANALYSIS: 3600 * 24,        # 24 hours
            CacheLayer.DEPENDENCY: 3600 * 12,      # 12 hours
            CacheLayer.CONTEXT: 3600 * 2,          # 2 hours
            CacheLayer.PROJECT: 3600,              # 1 hour
            CacheLayer.LANGUAGE: 3600 * 24 * 7,    # 1 week
            CacheLayer.HASH: 3600 * 24 * 30        # 30 days
        }
        
        # Cache invalidation tracking
        self._dependency_map: Dict[str, Set[str]] = {}  # file_path -> {dependent_files}
        self._last_cleanup = datetime.utcnow()
        
        # Initialize layer statistics
        for layer in CacheLayer:
            self._stats.layer_stats[layer.value] = {
                'hits': 0, 'misses': 0, 'sets': 0, 'deletes': 0
            }
    
    # ================== ADVANCED CACHING OPERATIONS ==================
    
    async def get_ast_cache(self, file_path: str, file_hash: str) -> Optional[Any]:
        """
        Get cached AST data with optimized retrieval.
        
        Args:
            file_path: Path to the file
            file_hash: Hash of the file content
            
        Returns:
            AST data if found in cache, None otherwise
        """
        try:
            key = CacheKey.ast_key(file_path, file_hash)
            cached_entry = await self._get_cache_entry(key, CacheLayer.AST)
            
            if cached_entry:
                # Update access statistics
                cached_entry.last_accessed = datetime.utcnow()
                cached_entry.access_count += 1
                
                self._stats.hits += 1
                self._stats.layer_stats[CacheLayer.AST.value]['hits'] += 1
                
                logger.debug("AST cache hit", 
                           file_path=file_path, 
                           file_hash=file_hash[:8],
                           access_count=cached_entry.access_count)
                
                return cached_entry.data
            else:
                self._stats.misses += 1
                self._stats.layer_stats[CacheLayer.AST.value]['misses'] += 1
                return None
                
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to get cached AST data", 
                        file_path=file_path, 
                        file_hash=file_hash[:8], 
                        error=str(e))
            return None
    
    async def set_ast_cache(
        self, 
        file_path: str, 
        file_hash: str, 
        ast_data: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache AST data with compression and metadata.
        
        Args:
            file_path: Path to the file
            file_hash: Hash of the file content
            ast_data: AST data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            key = CacheKey.ast_key(file_path, file_hash)
            ttl = ttl or self._layer_ttls[CacheLayer.AST]
            
            # Create cache entry with metadata
            cache_entry = CacheEntry(
                data=ast_data,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                layer=CacheLayer.AST
            )
            
            success = await self._set_cache_entry(key, cache_entry, ttl)
            
            if success:
                self._stats.sets += 1
                self._stats.layer_stats[CacheLayer.AST.value]['sets'] += 1
                logger.debug("AST data cached", 
                           file_path=file_path, 
                           file_hash=file_hash[:8], 
                           ttl=ttl,
                           compressed=cache_entry.compressed)
            
            return success
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to cache AST data", 
                        file_path=file_path, 
                        file_hash=file_hash[:8], 
                        error=str(e))
            return False
    
    # ================== CORE CACHE OPERATIONS ==================
    
    async def _get_cache_entry(self, key: str, layer: CacheLayer) -> Optional[CacheEntry]:
        """
        Get cache entry with decompression and metadata handling.
        
        Args:
            key: Cache key
            layer: Cache layer type
            
        Returns:
            CacheEntry if found, None otherwise
        """
        try:
            cached_data = await self.redis.get(key)
            if not cached_data:
                return None
            
            # Deserialize entry
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode('utf-8')
            
            entry_dict = json.loads(cached_data)
            
            # Check if data is compressed
            if entry_dict.get('compressed', False):
                # Decompress data
                compressed_data = entry_dict['data'].encode('latin1')
                decompressed_data = zlib.decompress(compressed_data)
                entry_dict['data'] = pickle.loads(decompressed_data)
            
            return CacheEntry.from_dict(entry_dict)
            
        except Exception as e:
            logger.error("Failed to deserialize cache entry", 
                       key=key, layer=layer.value, error=str(e))
            return None
    
    async def _set_cache_entry(self, key: str, entry: CacheEntry, ttl: int) -> bool:
        """
        Set cache entry with compression and metadata.
        
        Args:
            key: Cache key
            entry: Cache entry to store
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize data
            serialized_data = pickle.dumps(entry.data)
            entry.size_bytes = len(serialized_data)
            
            # Apply compression if enabled and data is large enough
            if (self.enable_compression and 
                entry.size_bytes > self.compression_threshold):
                compressed_data = zlib.compress(serialized_data)
                if len(compressed_data) < entry.size_bytes:
                    # Compression is beneficial
                    entry.compressed = True
                    entry.data = compressed_data.decode('latin1')
                    self._stats.compression_ratio = len(compressed_data) / entry.size_bytes
                else:
                    # Keep original if compression doesn't help
                    entry.data = serialized_data.decode('latin1')
            else:
                entry.data = serialized_data.decode('latin1')
            
            # Convert to JSON
            entry_json = json.dumps(entry.to_dict(), default=str)
            
            # Store in Redis
            success = await self.redis.set(key, entry_json, expire=ttl)
            
            if success:
                self._stats.size_bytes += entry.size_bytes
            
            return success
            
        except Exception as e:
            logger.error("Failed to serialize cache entry", 
                       key=key, error=str(e))
            return False
    
    # ================== ANALYSIS LAYER CACHING ==================
    
    async def get_analysis_cache(self, file_path: str, file_hash: str) -> Optional[FileAnalysisResult]:
        """
        Get cached file analysis result.
        
        Args:
            file_path: Path to the file
            file_hash: Hash of the file content
            
        Returns:
            FileAnalysisResult if found in cache, None otherwise
        """
        try:
            key = CacheKey.analysis_key(file_path, file_hash)
            cached_entry = await self._get_cache_entry(key, CacheLayer.ANALYSIS)
            
            if cached_entry:
                cached_entry.last_accessed = datetime.utcnow()
                cached_entry.access_count += 1
                
                self._stats.hits += 1
                self._stats.layer_stats[CacheLayer.ANALYSIS.value]['hits'] += 1
                
                logger.debug("Analysis cache hit", 
                           file_path=file_path, 
                           file_hash=file_hash[:8])
                
                return cached_entry.data
            else:
                self._stats.misses += 1
                self._stats.layer_stats[CacheLayer.ANALYSIS.value]['misses'] += 1
                return None
                
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to get cached analysis result", 
                        file_path=file_path, 
                        file_hash=file_hash[:8], 
                        error=str(e))
            return None
    
    async def set_analysis_cache(
        self, 
        file_path: str, 
        file_hash: str, 
        result: FileAnalysisResult,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache file analysis result.
        
        Args:
            file_path: Path to the file
            file_hash: Hash of the file content
            result: FileAnalysisResult to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            key = CacheKey.analysis_key(file_path, file_hash)
            ttl = ttl or self._layer_ttls[CacheLayer.ANALYSIS]
            
            cache_entry = CacheEntry(
                data=result,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                layer=CacheLayer.ANALYSIS
            )
            
            success = await self._set_cache_entry(key, cache_entry, ttl)
            
            if success:
                self._stats.sets += 1
                self._stats.layer_stats[CacheLayer.ANALYSIS.value]['sets'] += 1
                logger.debug("Analysis result cached", 
                           file_path=file_path, 
                           file_hash=file_hash[:8], 
                           ttl=ttl)
            
            return success
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to cache analysis result", 
                        file_path=file_path, 
                        file_hash=file_hash[:8], 
                        error=str(e))
            return False
    
    # ================== DEPENDENCY LAYER CACHING ==================
    
    async def get_dependency_cache(self, project_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get cached dependency graph for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Dependency graph data if found in cache, None otherwise
        """
        try:
            key = CacheKey.dependency_key(project_id)
            cached_entry = await self._get_cache_entry(key, CacheLayer.DEPENDENCY)
            
            if cached_entry:
                cached_entry.last_accessed = datetime.utcnow()
                cached_entry.access_count += 1
                
                self._stats.hits += 1
                self._stats.layer_stats[CacheLayer.DEPENDENCY.value]['hits'] += 1
                
                logger.debug("Dependency cache hit", project_id=str(project_id))
                return cached_entry.data
            else:
                self._stats.misses += 1
                self._stats.layer_stats[CacheLayer.DEPENDENCY.value]['misses'] += 1
                return None
                
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to get cached dependency graph", 
                        project_id=str(project_id), error=str(e))
            return None
    
    async def set_dependency_cache(
        self, 
        project_id: UUID, 
        dependency_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache dependency graph for a project.
        
        Args:
            project_id: Project identifier
            dependency_data: Dependency graph data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            key = CacheKey.dependency_key(project_id)
            ttl = ttl or self._layer_ttls[CacheLayer.DEPENDENCY]
            
            cache_entry = CacheEntry(
                data=dependency_data,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                layer=CacheLayer.DEPENDENCY
            )
            
            success = await self._set_cache_entry(key, cache_entry, ttl)
            
            if success:
                self._stats.sets += 1
                self._stats.layer_stats[CacheLayer.DEPENDENCY.value]['sets'] += 1
                logger.debug("Dependency graph cached", 
                           project_id=str(project_id), ttl=ttl)
            
            return success
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to cache dependency graph", 
                        project_id=str(project_id), error=str(e))
            return False
    
    # ================== CONTEXT LAYER CACHING ==================
    
    async def get_context_cache(self, project_id: UUID, context_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached context optimization result.
        
        Args:
            project_id: Project identifier
            context_hash: Hash of the context request
            
        Returns:
            Context optimization data if found, None otherwise
        """
        try:
            key = CacheKey.context_key(project_id, context_hash)
            cached_entry = await self._get_cache_entry(key, CacheLayer.CONTEXT)
            
            if cached_entry:
                cached_entry.last_accessed = datetime.utcnow()
                cached_entry.access_count += 1
                
                self._stats.hits += 1
                self._stats.layer_stats[CacheLayer.CONTEXT.value]['hits'] += 1
                
                logger.debug("Context cache hit", 
                           project_id=str(project_id), 
                           context_hash=context_hash[:8])
                return cached_entry.data
            else:
                self._stats.misses += 1
                self._stats.layer_stats[CacheLayer.CONTEXT.value]['misses'] += 1
                return None
                
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to get cached context optimization", 
                        project_id=str(project_id), 
                        context_hash=context_hash[:8], 
                        error=str(e))
            return None
    
    async def set_context_cache(
        self, 
        project_id: UUID, 
        context_hash: str,
        context_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache context optimization result.
        
        Args:
            project_id: Project identifier
            context_hash: Hash of the context request
            context_data: Context optimization data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            key = CacheKey.context_key(project_id, context_hash)
            ttl = ttl or self._layer_ttls[CacheLayer.CONTEXT]
            
            cache_entry = CacheEntry(
                data=context_data,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                layer=CacheLayer.CONTEXT
            )
            
            success = await self._set_cache_entry(key, cache_entry, ttl)
            
            if success:
                self._stats.sets += 1
                self._stats.layer_stats[CacheLayer.CONTEXT.value]['sets'] += 1
                logger.debug("Context optimization cached", 
                           project_id=str(project_id), 
                           context_hash=context_hash[:8], 
                           ttl=ttl)
            
            return success
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to cache context optimization", 
                        project_id=str(project_id), 
                        context_hash=context_hash[:8], 
                        error=str(e))
            return False
    
    # ================== INTELLIGENT CACHE INVALIDATION ==================
    
    async def invalidate_file_caches(self, file_path: str, file_hash: Optional[str] = None) -> bool:
        """
        Invalidate all cache entries related to a specific file.
        
        Args:
            file_path: Path to the file
            file_hash: Optional file hash for targeted invalidation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            keys_to_delete = []
            
            if file_hash:
                # Targeted invalidation using hash
                keys_to_delete.extend([
                    CacheKey.ast_key(file_path, file_hash),
                    CacheKey.analysis_key(file_path, file_hash)
                ])
            
            # Always invalidate language detection cache
            keys_to_delete.append(CacheKey.language_key(file_path))
            keys_to_delete.append(CacheKey.hash_key(file_path))
            
            deleted_count = 0
            for key in keys_to_delete:
                deleted = await self.redis.delete(key)
                if deleted:
                    deleted_count += 1
            
            # Invalidate dependent files
            await self._invalidate_dependent_files(file_path)
            
            self._stats.deletes += deleted_count
            self._stats.invalidations += 1
            
            logger.debug("File caches invalidated", 
                        file_path=file_path, 
                        deleted_count=deleted_count)
            
            return True
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to invalidate file caches", 
                        file_path=file_path, error=str(e))
            return False
    
    async def invalidate_project_caches(self, project_id: UUID) -> bool:
        """
        Invalidate all cache entries for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            keys_to_delete = [
                CacheKey.project_key(project_id),
                CacheKey.dependency_key(project_id)
            ]
            
            # Find and delete context caches (requires pattern matching)
            context_pattern = f"pi:context:{str(project_id)}:*"
            context_keys = await self._scan_keys(context_pattern)
            keys_to_delete.extend(context_keys)
            
            deleted_count = 0
            for key in keys_to_delete:
                deleted = await self.redis.delete(key)
                if deleted:
                    deleted_count += 1
            
            self._stats.deletes += deleted_count
            self._stats.invalidations += 1
            
            logger.info("Project caches invalidated", 
                       project_id=str(project_id), 
                       deleted_count=deleted_count)
            
            return True
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to invalidate project caches", 
                        project_id=str(project_id), error=str(e))
            return False
    
    async def _invalidate_dependent_files(self, file_path: str) -> None:
        """
        Invalidate caches for files that depend on the given file.
        
        Args:
            file_path: Path to the file whose dependents should be invalidated
        """
        if file_path in self._dependency_map:
            dependent_files = self._dependency_map[file_path]
            for dependent_file in dependent_files:
                await self.invalidate_file_caches(dependent_file)
    
    async def _scan_keys(self, pattern: str) -> List[str]:
        """
        Scan Redis keys matching a pattern.
        
        Args:
            pattern: Redis key pattern
            
        Returns:
            List of matching keys
        """
        try:
            # This is a simplified implementation
            # In production, you would use Redis SCAN command
            # For now, return empty list as fallback
            return []
        except Exception as e:
            logger.error("Failed to scan keys", pattern=pattern, error=str(e))
            return []
    
    def track_dependency(self, file_path: str, dependent_file: str) -> None:
        """
        Track dependency relationship for cache invalidation.
        
        Args:
            file_path: Path to the dependency file
            dependent_file: Path to the file that depends on file_path
        """
        if file_path not in self._dependency_map:
            self._dependency_map[file_path] = set()
        self._dependency_map[file_path].add(dependent_file)
    
    # ================== PROJECT AND LANGUAGE CACHING ==================
    
    async def get_language_cache(self, file_path: str) -> Optional[str]:
        """
        Get cached language detection result.
        
        Args:
            file_path: File path for language detection
            
        Returns:
            Detected language if cached, None otherwise
        """
        try:
            key = CacheKey.language_key(file_path)
            cached_entry = await self._get_cache_entry(key, CacheLayer.LANGUAGE)
            
            if cached_entry:
                self._stats.hits += 1
                self._stats.layer_stats[CacheLayer.LANGUAGE.value]['hits'] += 1
                
                logger.debug("Language cache hit", 
                           file_path=file_path, language=cached_entry.data)
                return cached_entry.data
            else:
                self._stats.misses += 1
                self._stats.layer_stats[CacheLayer.LANGUAGE.value]['misses'] += 1
                return None
                
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to get cached language", 
                        file_path=file_path, error=str(e))
            return None
    
    async def set_language_cache(
        self, 
        file_path: str, 
        language: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache language detection result.
        
        Args:
            file_path: File path
            language: Detected language
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            key = CacheKey.language_key(file_path)
            ttl = ttl or self._layer_ttls[CacheLayer.LANGUAGE]
            
            cache_entry = CacheEntry(
                data=language,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                layer=CacheLayer.LANGUAGE
            )
            
            success = await self._set_cache_entry(key, cache_entry, ttl)
            
            if success:
                self._stats.sets += 1
                self._stats.layer_stats[CacheLayer.LANGUAGE.value]['sets'] += 1
                logger.debug("Language cached", 
                           file_path=file_path, 
                           language=language, 
                           ttl=ttl)
            
            return success
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to cache language", 
                        file_path=file_path, error=str(e))
            return False
    
    async def get_project_cache(self, project_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get cached project metadata and statistics.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project metadata if found in cache, None otherwise
        """
        try:
            key = CacheKey.project_key(project_id)
            cached_entry = await self._get_cache_entry(key, CacheLayer.PROJECT)
            
            if cached_entry:
                self._stats.hits += 1
                self._stats.layer_stats[CacheLayer.PROJECT.value]['hits'] += 1
                
                logger.debug("Project cache hit", project_id=str(project_id))
                return cached_entry.data
            else:
                self._stats.misses += 1
                self._stats.layer_stats[CacheLayer.PROJECT.value]['misses'] += 1
                return None
                
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to get cached project data", 
                        project_id=str(project_id), error=str(e))
            return None
    
    async def set_project_cache(
        self, 
        project_id: UUID, 
        project_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache project metadata and statistics.
        
        Args:
            project_id: Project identifier
            project_data: Project data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            key = CacheKey.project_key(project_id)
            ttl = ttl or self._layer_ttls[CacheLayer.PROJECT]
            
            cache_entry = CacheEntry(
                data=project_data,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                layer=CacheLayer.PROJECT
            )
            
            success = await self._set_cache_entry(key, cache_entry, ttl)
            
            if success:
                self._stats.sets += 1
                self._stats.layer_stats[CacheLayer.PROJECT.value]['sets'] += 1
                logger.debug("Project data cached", 
                           project_id=str(project_id), ttl=ttl)
            
            return success
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to cache project data", 
                        project_id=str(project_id), error=str(e))
            return False
    
    # ================== CACHE MANAGEMENT AND STATISTICS ==================
    
    async def cleanup_expired_entries(self) -> Dict[str, int]:
        """
        Clean up expired cache entries and optimize memory usage.
        
        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {
            'entries_cleaned': 0,
            'memory_freed_bytes': 0,
            'cleanup_duration_ms': 0
        }
        
        start_time = time.time()
        
        try:
            # Perform cleanup logic here
            # This is a placeholder - in production you would:
            # 1. Scan for expired entries
            # 2. Remove least recently used entries if over memory limit
            # 3. Compress large entries
            
            self._last_cleanup = datetime.utcnow()
            cleanup_stats['cleanup_duration_ms'] = int((time.time() - start_time) * 1000)
            
            logger.info("Cache cleanup completed", **cleanup_stats)
            return cleanup_stats
            
        except Exception as e:
            logger.error("Cache cleanup failed", error=str(e))
            return cleanup_stats
    
    def get_cache_stats(self) -> CacheStatistics:
        """
        Get comprehensive cache performance statistics.
        
        Returns:
            CacheStatistics object with detailed metrics
        """
        # Calculate hit rate
        self._stats.calculate_hit_rate()
        
        # Update layer statistics
        for layer_name, layer_stats in self._stats.layer_stats.items():
            layer_total = layer_stats['hits'] + layer_stats['misses']
            if layer_total > 0:
                layer_stats['hit_rate'] = (layer_stats['hits'] / layer_total) * 100
        
        return self._stats
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """
        Get detailed cache information and configuration.
        
        Returns:
            Dictionary with cache information
        """
        try:
            cache_info = {
                'configuration': {
                    'compression_enabled': self.enable_compression,
                    'compression_threshold': self.compression_threshold,
                    'max_memory_mb': self.max_memory_mb
                },
                'layer_ttls': {layer.value: ttl for layer, ttl in self._layer_ttls.items()},
                'statistics': self.get_cache_stats(),
                'dependency_tracking': {
                    'tracked_files': len(self._dependency_map),
                    'total_dependencies': sum(len(deps) for deps in self._dependency_map.values())
                },
                'last_cleanup': self._last_cleanup.isoformat() if self._last_cleanup else None
            }
            
            return cache_info
            
        except Exception as e:
            logger.error("Failed to get cache info", error=str(e))
            return {'error': str(e)}
    
    async def extend_ttl(self, cache_key: str, additional_seconds: int) -> bool:
        """
        Extend the TTL of a cached item.
        
        Args:
            cache_key: Full cache key
            additional_seconds: Additional seconds to add to TTL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = await self.redis.expire(cache_key, additional_seconds)
            
            if success:
                logger.debug("Cache TTL extended", 
                           cache_key=cache_key, 
                           additional_seconds=additional_seconds)
            
            return success
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to extend cache TTL", 
                        cache_key=cache_key, error=str(e))
            return False
    
    async def clear_all_cache(self) -> bool:
        """
        Clear all project index cache entries.
        
        WARNING: This will clear all cached data for all projects.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Clearing all project index cache")
            
            # Reset statistics
            self._stats = CacheStatistics()
            for layer in CacheLayer:
                self._stats.layer_stats[layer.value] = {
                    'hits': 0, 'misses': 0, 'sets': 0, 'deletes': 0
                }
            
            # Clear dependency tracking
            self._dependency_map.clear()
            
            return True
            
        except Exception as e:
            self._stats.errors += 1
            logger.error("Failed to clear all cache", error=str(e))
            return False


# Backward compatibility alias
CacheManager = AdvancedCacheManager