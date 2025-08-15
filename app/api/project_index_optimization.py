"""
Performance optimization utilities for Project Index API endpoints.

Provides caching, query optimization, and performance monitoring
for the Project Index API to meet response time requirements.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import select, func, and_, or_, text
from pydantic import BaseModel

from ..core.redis import RedisClient
from ..models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession
)

logger = structlog.get_logger()


class CacheConfig(BaseModel):
    """Configuration for Redis caching."""
    ttl_project: int = 300  # 5 minutes
    ttl_file_list: int = 180  # 3 minutes  
    ttl_dependencies: int = 240  # 4 minutes
    ttl_statistics: int = 600  # 10 minutes
    prefix: str = "project_index_api"
    enable_compression: bool = True
    max_cache_size: int = 1000000  # 1MB


class QueryOptimizer:
    """Optimizes database queries for Project Index API endpoints."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_project_with_stats(self, project_id: str) -> Optional[ProjectIndex]:
        """
        Get project with optimized statistics loading.
        
        Uses single query with joins to minimize database round trips.
        """
        try:
            # Optimized query with eager loading
            stmt = (
                select(ProjectIndex)
                .where(ProjectIndex.id == project_id)
                .options(
                    # Don't load relationships by default for better performance
                    # Only load what's needed for basic project info
                )
            )
            
            result = await self.session.execute(stmt)
            project = result.scalar_one_or_none()
            
            if project:
                # Update file and dependency counts in separate optimized queries
                file_count_stmt = select(func.count(FileEntry.id)).where(
                    FileEntry.project_id == project_id
                )
                dep_count_stmt = select(func.count(DependencyRelationship.id)).where(
                    DependencyRelationship.project_id == project_id
                )
                
                file_count_result = await self.session.execute(file_count_stmt)
                dep_count_result = await self.session.execute(dep_count_stmt)
                
                project.file_count = file_count_result.scalar() or 0
                project.dependency_count = dep_count_result.scalar() or 0
            
            return project
            
        except Exception as e:
            logger.error("Failed to get project with stats", 
                        project_id=project_id, error=str(e))
            return None
    
    async def get_files_paginated_optimized(
        self,
        project_id: str,
        page: int,
        limit: int,
        filters: Dict[str, Any] = None
    ) -> tuple[List[FileEntry], int]:
        """
        Get paginated files with optimized query and filtering.
        
        Uses efficient pagination and selective loading for better performance.
        """
        try:
            # Build base query
            base_query = select(FileEntry).where(FileEntry.project_id == project_id)
            count_query = select(func.count(FileEntry.id)).where(FileEntry.project_id == project_id)
            
            # Apply filters
            if filters:
                if language := filters.get('language'):
                    base_query = base_query.where(FileEntry.language == language)
                    count_query = count_query.where(FileEntry.language == language)
                
                if file_type := filters.get('file_type'):
                    base_query = base_query.where(FileEntry.file_type == file_type)
                    count_query = count_query.where(FileEntry.file_type == file_type)
                
                if modified_after := filters.get('modified_after'):
                    base_query = base_query.where(FileEntry.last_modified >= modified_after)
                    count_query = count_query.where(FileEntry.last_modified >= modified_after)
            
            # Add efficient pagination
            offset = (page - 1) * limit
            base_query = (
                base_query
                .offset(offset)
                .limit(limit)
                .order_by(FileEntry.relative_path)  # Consistent ordering
            )
            
            # Execute queries concurrently
            files_task = self.session.execute(base_query)
            count_task = self.session.execute(count_query)
            
            files_result, count_result = await asyncio.gather(files_task, count_task)
            
            files = files_result.scalars().all()
            total = count_result.scalar() or 0
            
            return files, total
            
        except Exception as e:
            logger.error("Failed to get files paginated", 
                        project_id=project_id, error=str(e))
            return [], 0
    
    async def get_file_with_dependencies(
        self,
        project_id: str,
        file_path: str
    ) -> Optional[FileEntry]:
        """
        Get file with dependencies using optimized joins.
        
        Loads file and related dependencies in single query.
        """
        try:
            stmt = (
                select(FileEntry)
                .where(
                    and_(
                        FileEntry.project_id == project_id,
                        FileEntry.relative_path == file_path
                    )
                )
                .options(
                    selectinload(FileEntry.outgoing_dependencies),
                    selectinload(FileEntry.incoming_dependencies)
                )
            )
            
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error("Failed to get file with dependencies", 
                        project_id=project_id, file_path=file_path, error=str(e))
            return None
    
    async def get_dependencies_optimized(
        self,
        project_id: str,
        filters: Dict[str, Any] = None,
        page: int = 1,
        limit: int = 100
    ) -> tuple[List[DependencyRelationship], int]:
        """
        Get dependencies with optimized query and optional file filtering.
        """
        try:
            base_query = select(DependencyRelationship).where(
                DependencyRelationship.project_id == project_id
            )
            count_query = select(func.count(DependencyRelationship.id)).where(
                DependencyRelationship.project_id == project_id
            )
            
            # Apply filters
            if filters:
                if source_file_id := filters.get('source_file_id'):
                    base_query = base_query.where(
                        DependencyRelationship.source_file_id == source_file_id
                    )
                    count_query = count_query.where(
                        DependencyRelationship.source_file_id == source_file_id
                    )
                
                if not filters.get('include_external', True):
                    base_query = base_query.where(DependencyRelationship.is_external == False)
                    count_query = count_query.where(DependencyRelationship.is_external == False)
            
            # Add pagination
            offset = (page - 1) * limit
            base_query = (
                base_query
                .offset(offset)
                .limit(limit)
                .order_by(DependencyRelationship.target_name)
            )
            
            # Execute concurrently
            deps_task = self.session.execute(base_query)
            count_task = self.session.execute(count_query)
            
            deps_result, count_result = await asyncio.gather(deps_task, count_task)
            
            dependencies = deps_result.scalars().all()
            total = count_result.scalar() or 0
            
            return dependencies, total
            
        except Exception as e:
            logger.error("Failed to get dependencies optimized", 
                        project_id=project_id, error=str(e))
            return [], 0
    
    async def get_dependency_graph_optimized(
        self,
        project_id: str,
        include_external: bool = True
    ) -> tuple[List[FileEntry], List[DependencyRelationship]]:
        """
        Get complete dependency graph data with optimized queries.
        
        Uses efficient joins and selective loading for graph construction.
        """
        try:
            # Get files for graph nodes
            files_stmt = select(FileEntry).where(FileEntry.project_id == project_id)
            
            # Get dependencies for graph edges
            deps_stmt = select(DependencyRelationship).where(
                DependencyRelationship.project_id == project_id
            )
            
            if not include_external:
                deps_stmt = deps_stmt.where(DependencyRelationship.is_external == False)
            
            # Execute concurrently
            files_task = self.session.execute(files_stmt)
            deps_task = self.session.execute(deps_stmt)
            
            files_result, deps_result = await asyncio.gather(files_task, deps_task)
            
            files = files_result.scalars().all()
            dependencies = deps_result.scalars().all()
            
            return files, dependencies
            
        except Exception as e:
            logger.error("Failed to get dependency graph", 
                        project_id=project_id, error=str(e))
            return [], []


class CacheManager:
    """Manages Redis caching for Project Index API responses."""
    
    def __init__(self, redis_client: RedisClient, config: CacheConfig = None):
        self.redis = redis_client
        self.config = config or CacheConfig()
    
    def _make_cache_key(self, key_type: str, *args) -> str:
        """Generate cache key with consistent format."""
        key_parts = [self.config.prefix, key_type] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    async def get_cached_response(
        self,
        key_type: str,
        *key_args,
        deserializer: Callable = json.loads
    ) -> Optional[Any]:
        """Get cached response with optional deserialization."""
        try:
            cache_key = self._make_cache_key(key_type, *key_args)
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                if self.config.enable_compression:
                    # Handle compressed data if needed
                    pass
                
                return deserializer(cached_data)
            
            return None
            
        except Exception as e:
            logger.error("Failed to get cached response", 
                        key_type=key_type, error=str(e))
            return None
    
    async def set_cached_response(
        self,
        key_type: str,
        *key_args,
        data: Any,
        ttl: Optional[int] = None,
        serializer: Callable = json.dumps
    ) -> bool:
        """Cache response with optional compression and TTL."""
        try:
            cache_key = self._make_cache_key(key_type, *key_args)
            
            # Determine TTL based on key type
            if ttl is None:
                ttl_map = {
                    "project": self.config.ttl_project,
                    "files": self.config.ttl_file_list,
                    "dependencies": self.config.ttl_dependencies,
                    "stats": self.config.ttl_statistics
                }
                ttl = ttl_map.get(key_type, 300)  # Default 5 minutes
            
            # Serialize data
            serialized_data = serializer(data)
            
            # Check size limits
            if len(serialized_data) > self.config.max_cache_size:
                logger.warning("Cache data too large, skipping cache", 
                             key_type=key_type, size=len(serialized_data))
                return False
            
            # Cache with compression if enabled
            if self.config.enable_compression:
                # Add compression logic here if needed
                pass
            
            await self.redis.setex(cache_key, ttl, serialized_data)
            return True
            
        except Exception as e:
            logger.error("Failed to cache response", 
                        key_type=key_type, error=str(e))
            return False
    
    async def invalidate_project_cache(self, project_id: str):
        """Invalidate all cached data for a project."""
        try:
            # Get all keys for this project
            pattern = f"{self.config.prefix}:*:{project_id}*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
                logger.info("Invalidated project cache", 
                           project_id=project_id, keys_count=len(keys))
            
        except Exception as e:
            logger.error("Failed to invalidate project cache", 
                        project_id=project_id, error=str(e))
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        try:
            pattern = f"{self.config.prefix}:*"
            keys = await self.redis.keys(pattern)
            
            stats = {
                "total_keys": len(keys),
                "cache_prefix": self.config.prefix,
                "key_types": {}
            }
            
            # Analyze key types
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) >= 2:
                    key_type = parts[1]
                    stats["key_types"][key_type] = stats["key_types"].get(key_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"error": str(e)}


class PerformanceMonitor:
    """Monitors API endpoint performance and provides metrics."""
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
    
    def monitor_endpoint(self, endpoint_name: str):
        """Decorator to monitor endpoint performance."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record success metrics
                    duration = (time.time() - start_time) * 1000  # Convert to ms
                    await self._record_metric(endpoint_name, "success", duration)
                    
                    return result
                    
                except Exception as e:
                    # Record error metrics
                    duration = (time.time() - start_time) * 1000
                    await self._record_metric(endpoint_name, "error", duration)
                    raise
            
            return wrapper
        return decorator
    
    async def _record_metric(
        self,
        endpoint: str,
        status: str,
        duration: float
    ):
        """Record performance metric to Redis."""
        try:
            timestamp = datetime.utcnow()
            metric_key = f"performance:{endpoint}:{status}:{timestamp.strftime('%Y%m%d%H')}"
            
            # Store metric with hourly buckets
            await self.redis.lpush(metric_key, json.dumps({
                "timestamp": timestamp.isoformat(),
                "duration_ms": duration,
                "status": status
            }))
            
            # Set expiration for metrics (keep for 7 days)
            await self.redis.expire(metric_key, 7 * 24 * 3600)
            
            # Update aggregated metrics
            await self._update_aggregated_metrics(endpoint, status, duration)
            
        except Exception as e:
            logger.error("Failed to record performance metric", 
                        endpoint=endpoint, error=str(e))
    
    async def _update_aggregated_metrics(
        self,
        endpoint: str,
        status: str,
        duration: float
    ):
        """Update aggregated performance metrics."""
        try:
            today = datetime.utcnow().strftime('%Y%m%d')
            agg_key = f"performance_agg:{endpoint}:{today}"
            
            # Get existing aggregated data
            existing = await self.redis.get(agg_key)
            if existing:
                agg_data = json.loads(existing)
            else:
                agg_data = {
                    "total_requests": 0,
                    "success_requests": 0,
                    "error_requests": 0,
                    "total_duration": 0,
                    "min_duration": float('inf'),
                    "max_duration": 0
                }
            
            # Update aggregated data
            agg_data["total_requests"] += 1
            agg_data[f"{status}_requests"] += 1
            agg_data["total_duration"] += duration
            agg_data["min_duration"] = min(agg_data["min_duration"], duration)
            agg_data["max_duration"] = max(agg_data["max_duration"], duration)
            agg_data["avg_duration"] = agg_data["total_duration"] / agg_data["total_requests"]
            
            # Store updated aggregated data
            await self.redis.setex(agg_key, 7 * 24 * 3600, json.dumps(agg_data))
            
        except Exception as e:
            logger.error("Failed to update aggregated metrics", 
                        endpoint=endpoint, error=str(e))
    
    async def get_performance_metrics(
        self,
        endpoint: str,
        days: int = 1
    ) -> Dict[str, Any]:
        """Get performance metrics for an endpoint."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            metrics = {
                "endpoint": endpoint,
                "period_days": days,
                "daily_metrics": {}
            }
            
            # Get daily aggregated metrics
            for i in range(days):
                date = (start_date + timedelta(days=i)).strftime('%Y%m%d')
                agg_key = f"performance_agg:{endpoint}:{date}"
                
                daily_data = await self.redis.get(agg_key)
                if daily_data:
                    metrics["daily_metrics"][date] = json.loads(daily_data)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to get performance metrics", 
                        endpoint=endpoint, error=str(e))
            return {"error": str(e)}


# ================== PERFORMANCE OPTIMIZED UTILITIES ==================

def create_optimized_session_factory(session: AsyncSession):
    """Create factory for optimized session instances."""
    def factory():
        return QueryOptimizer(session)
    return factory


def create_cache_manager_factory(redis_client: RedisClient, config: CacheConfig = None):
    """Create factory for cache manager instances."""
    def factory():
        return CacheManager(redis_client, config)
    return factory


def create_performance_monitor_factory(redis_client: RedisClient):
    """Create factory for performance monitor instances."""
    def factory():
        return PerformanceMonitor(redis_client)
    return factory


# ================== CACHING DECORATORS ==================

def cache_response(
    cache_type: str,
    ttl: Optional[int] = None,
    key_generator: Optional[Callable] = None
):
    """Decorator to cache API endpoint responses."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract cache manager from kwargs or create one
            cache_manager = kwargs.get('cache_manager')
            if not cache_manager:
                return await func(*args, **kwargs)
            
            # Generate cache key
            if key_generator:
                cache_key_parts = key_generator(*args, **kwargs)
            else:
                # Default key generation
                cache_key_parts = [str(arg) for arg in args if isinstance(arg, (str, int, uuid.UUID))]
            
            # Try to get from cache
            cached_result = await cache_manager.get_cached_response(
                cache_type, *cache_key_parts
            )
            
            if cached_result:
                logger.debug("Cache hit", cache_type=cache_type, key_parts=cache_key_parts)
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache_manager.set_cached_response(
                cache_type, *cache_key_parts, data=result, ttl=ttl
            )
            
            logger.debug("Cache miss, cached result", 
                        cache_type=cache_type, key_parts=cache_key_parts)
            
            return result
        
        return wrapper
    return decorator


# ================== QUERY OPTIMIZATION HELPERS ==================

def optimize_select_query(
    base_query,
    eager_load: List[str] = None,
    join_load: List[str] = None
):
    """Apply query optimizations for select operations."""
    if eager_load:
        for relationship in eager_load:
            base_query = base_query.options(selectinload(relationship))
    
    if join_load:
        for relationship in join_load:
            base_query = base_query.options(joinedload(relationship))
    
    return base_query


def build_filter_conditions(model_class, filters: Dict[str, Any]):
    """Build SQLAlchemy filter conditions from filter dictionary."""
    conditions = []
    
    for field_name, field_value in filters.items():
        if hasattr(model_class, field_name):
            field = getattr(model_class, field_name)
            
            if isinstance(field_value, list):
                conditions.append(field.in_(field_value))
            elif isinstance(field_value, dict):
                # Handle range queries
                if 'gte' in field_value:
                    conditions.append(field >= field_value['gte'])
                if 'lte' in field_value:
                    conditions.append(field <= field_value['lte'])
                if 'gt' in field_value:
                    conditions.append(field > field_value['gt'])
                if 'lt' in field_value:
                    conditions.append(field < field_value['lt'])
            else:
                conditions.append(field == field_value)
    
    return conditions