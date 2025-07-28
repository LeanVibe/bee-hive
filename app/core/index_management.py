"""
Index Management System for Dynamic Vector Index Optimization.

This module provides comprehensive index management for vector search
optimization with:
- Dynamic index type selection and optimization
- Automated index maintenance and rebuilding
- Performance monitoring and tuning
- Index statistics and health monitoring
- Parallel index operations and migrations
- Memory-efficient index operations
- Production-ready index lifecycle management
"""

import asyncio
import time
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import numpy as np

from sqlalchemy import select, and_, or_, desc, asc, func, text, Index
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
import redis.asyncio as redis

from ..models.context import Context, ContextType
from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Vector index types supported."""
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"
    BRUTE_FORCE = "brute_force"  # No index, direct computation


class IndexStatus(Enum):
    """Index status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    REBUILDING = "rebuilding"
    FAILED = "failed"
    MISSING = "missing"


class OperationType(Enum):
    """Index operation types."""
    CREATE = "create"
    REBUILD = "rebuild"
    OPTIMIZE = "optimize"
    DROP = "drop"
    ANALYZE = "analyze"
    VACUUM = "vacuum"


@dataclass
class IndexConfiguration:
    """Configuration for vector index."""
    index_name: str
    index_type: IndexType
    table_name: str
    column_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    last_optimized: Optional[datetime] = None
    optimization_schedule: str = "daily"  # daily, weekly, manual


@dataclass
class IndexStatistics:
    """Statistics for a vector index."""
    index_name: str
    index_type: IndexType
    total_vectors: int
    index_size_bytes: int
    avg_query_time_ms: float
    queries_per_second: float
    cache_hit_ratio: float
    fragmentation_ratio: float
    last_vacuum: Optional[datetime]
    last_analyze: Optional[datetime]
    creation_time: Optional[datetime]
    last_updated: datetime


@dataclass
class IndexOperation:
    """Represents an index operation."""
    operation_id: str
    operation_type: OperationType
    index_name: str
    status: str  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    progress_percent: float = 0.0


class IndexPerformanceAnalyzer:
    """Analyzes index performance and suggests optimizations."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.query_patterns: Dict[str, int] = {}
        
    def record_query_performance(
        self, 
        index_name: str, 
        query_time_ms: float,
        query_pattern: str = "general"
    ) -> None:
        """Record query performance for analysis."""
        if index_name not in self.performance_history:
            self.performance_history[index_name] = []
        
        self.performance_history[index_name].append((datetime.utcnow(), query_time_ms))
        
        # Keep only recent history (last 1000 queries)
        if len(self.performance_history[index_name]) > 1000:
            self.performance_history[index_name] = self.performance_history[index_name][-1000:]
        
        # Track query patterns
        self.query_patterns[query_pattern] = self.query_patterns.get(query_pattern, 0) + 1
    
    def analyze_performance_trends(self, index_name: str) -> Dict[str, Any]:
        """Analyze performance trends for an index."""
        if index_name not in self.performance_history:
            return {"error": "No performance data available"}
        
        history = self.performance_history[index_name]
        if len(history) < 10:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate trends
        recent_queries = [query_time for _, query_time in history[-100:]]
        older_queries = [query_time for _, query_time in history[-200:-100]] if len(history) >= 200 else []
        
        current_avg = np.mean(recent_queries)
        previous_avg = np.mean(older_queries) if older_queries else current_avg
        
        trend_direction = "improving" if current_avg < previous_avg else "degrading"
        trend_magnitude = abs(current_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
        
        return {
            "index_name": index_name,
            "current_avg_ms": current_avg,
            "previous_avg_ms": previous_avg,
            "trend_direction": trend_direction,
            "trend_magnitude": trend_magnitude,
            "total_queries": len(history),
            "p95_latency_ms": np.percentile(recent_queries, 95),
            "p99_latency_ms": np.percentile(recent_queries, 99),
            "recommendation": self._generate_performance_recommendation(
                current_avg, trend_direction, trend_magnitude
            )
        }
    
    def _generate_performance_recommendation(
        self, 
        avg_latency: float, 
        trend: str, 
        magnitude: float
    ) -> str:
        """Generate performance recommendation."""
        if avg_latency > 1000:  # > 1 second
            return "Consider index optimization or rebuilding - high latency detected"
        elif trend == "degrading" and magnitude > 0.2:
            return "Performance degrading - schedule index maintenance"
        elif avg_latency > 500:  # > 500ms
            return "Monitor performance closely - approaching slow query threshold"
        else:
            return "Performance within acceptable range"
    
    def suggest_optimal_index_type(
        self, 
        vector_count: int, 
        query_patterns: Dict[str, int],
        memory_constraint_mb: Optional[int] = None
    ) -> Tuple[IndexType, Dict[str, Any]]:
        """Suggest optimal index type based on data characteristics."""
        total_queries = sum(query_patterns.values())
        
        # Analyze query pattern distribution
        similarity_queries = query_patterns.get("similarity", 0)
        hybrid_queries = query_patterns.get("hybrid", 0)
        complex_queries = query_patterns.get("complex", 0)
        
        # Decision logic
        if vector_count < 1000:
            # Small datasets can use brute force
            return IndexType.BRUTE_FORCE, {"reason": "Small dataset, brute force efficient"}
        
        elif vector_count < 10000:
            # Medium datasets work well with IVFFlat
            lists = min(100, max(10, vector_count // 100))
            return IndexType.IVFFLAT, {
                "reason": "Medium dataset, IVFFlat optimal",
                "parameters": {"lists": lists}
            }
        
        else:
            # Large datasets benefit from HNSW
            if memory_constraint_mb and memory_constraint_mb < 1000:
                # Memory constrained, use IVFFlat
                lists = min(200, max(50, vector_count // 1000))
                return IndexType.IVFFLAT, {
                    "reason": "Large dataset with memory constraints",
                    "parameters": {"lists": lists}
                }
            else:
                # Use HNSW for best performance
                m = 16 if complex_queries / total_queries > 0.3 else 12
                ef_construction = 64 if complex_queries / total_queries > 0.3 else 32
                
                return IndexType.HNSW, {
                    "reason": "Large dataset, HNSW for best performance",
                    "parameters": {"m": m, "ef_construction": ef_construction}
                }


class IndexManager:
    """
    Comprehensive index management system for vector search optimization.
    
    Features:
    - Dynamic index type selection and optimization
    - Automated maintenance scheduling and execution
    - Performance monitoring and tuning
    - Parallel operations with progress tracking
    - Health monitoring and alerting
    - Memory-efficient operations
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        maintenance_interval_hours: int = 24,
        performance_threshold_ms: float = 500.0,
        memory_limit_mb: int = 2000
    ):
        """
        Initialize index manager.
        
        Args:
            redis_client: Redis client for coordination
            maintenance_interval_hours: Hours between maintenance cycles
            performance_threshold_ms: Performance threshold for optimization
            memory_limit_mb: Memory limit for index operations
        """
        self.settings = get_settings()
        self.redis_client = redis_client or get_redis_client()
        self.maintenance_interval_hours = maintenance_interval_hours
        self.performance_threshold_ms = performance_threshold_ms
        self.memory_limit_mb = memory_limit_mb
        
        # Components
        self.performance_analyzer = IndexPerformanceAnalyzer()
        
        # State management
        self.managed_indexes: Dict[str, IndexConfiguration] = {}
        self.active_operations: Dict[str, IndexOperation] = {}
        self.index_statistics: Dict[str, IndexStatistics] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._operation_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent operations
        
        # Default index configurations
        self._default_configs = {
            "context_embedding_hnsw": IndexConfiguration(
                index_name="context_embedding_hnsw_idx",
                index_type=IndexType.HNSW,
                table_name="context",
                column_name="embedding",
                parameters={"m": 16, "ef_construction": 64}
            ),
            "context_embedding_ivfflat": IndexConfiguration(
                index_name="context_embedding_ivfflat_idx",
                index_type=IndexType.IVFFLAT,
                table_name="context",
                column_name="embedding",
                parameters={"lists": 100}
            )
        }
    
    async def start(self) -> None:
        """Start index management services."""
        logger.info("Starting index management system")
        
        # Initialize managed indexes
        await self._initialize_managed_indexes()
        
        # Start background services
        self._background_tasks.append(
            asyncio.create_task(self._maintenance_scheduler())
        )
        
        self._background_tasks.append(
            asyncio.create_task(self._performance_monitor())
        )
        
        self._background_tasks.append(
            asyncio.create_task(self._operation_manager())
        )
        
        self._background_tasks.append(
            asyncio.create_task(self._statistics_collector())
        )
    
    async def stop(self) -> None:
        """Stop index management services."""
        logger.info("Stopping index management system")
        
        self._shutdown_event.set()
        
        # Cancel active operations
        for operation in self.active_operations.values():
            if operation.status == "running":
                operation.status = "cancelled"
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def create_index(
        self,
        index_name: str,
        index_type: IndexType,
        table_name: str = "context",
        column_name: str = "embedding",
        parameters: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> str:
        """
        Create a new vector index.
        
        Args:
            index_name: Name of the index
            index_type: Type of index to create
            table_name: Table containing vectors
            column_name: Column containing vectors
            parameters: Index-specific parameters
            force: Force creation even if index exists
            
        Returns:
            Operation ID for tracking
        """
        operation_id = str(uuid.uuid4())
        
        # Check if index already exists
        if not force and await self._index_exists(index_name):
            raise ValueError(f"Index {index_name} already exists. Use force=True to recreate.")
        
        # Create operation
        operation = IndexOperation(
            operation_id=operation_id,
            operation_type=OperationType.CREATE,
            index_name=index_name,
            status="pending",
            parameters={
                "index_type": index_type.value,
                "table_name": table_name,
                "column_name": column_name,
                "index_parameters": parameters or {},
                "force": force
            }
        )
        
        self.active_operations[operation_id] = operation
        
        # Create index configuration
        config = IndexConfiguration(
            index_name=index_name,
            index_type=index_type,
            table_name=table_name,
            column_name=column_name,
            parameters=parameters or {}
        )
        
        self.managed_indexes[index_name] = config
        
        logger.info(f"Scheduled index creation: {index_name} ({index_type.value})")
        return operation_id
    
    async def optimize_index(
        self,
        index_name: str,
        analyze_performance: bool = True
    ) -> str:
        """
        Optimize an existing index.
        
        Args:
            index_name: Name of index to optimize
            analyze_performance: Whether to analyze performance first
            
        Returns:
            Operation ID for tracking
        """
        operation_id = str(uuid.uuid4())
        
        if index_name not in self.managed_indexes:
            raise ValueError(f"Index {index_name} is not managed by this system")
        
        # Create operation
        operation = IndexOperation(
            operation_id=operation_id,
            operation_type=OperationType.OPTIMIZE,
            index_name=index_name,
            status="pending",
            parameters={
                "analyze_performance": analyze_performance
            }
        )
        
        self.active_operations[operation_id] = operation
        
        logger.info(f"Scheduled index optimization: {index_name}")
        return operation_id
    
    async def rebuild_index(
        self,
        index_name: str,
        new_parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Rebuild an existing index, optionally with new parameters.
        
        Args:
            index_name: Name of index to rebuild
            new_parameters: New parameters for the index
            
        Returns:
            Operation ID for tracking
        """
        operation_id = str(uuid.uuid4())
        
        if index_name not in self.managed_indexes:
            raise ValueError(f"Index {index_name} is not managed by this system")
        
        # Create operation
        operation = IndexOperation(
            operation_id=operation_id,
            operation_type=OperationType.REBUILD,
            index_name=index_name,
            status="pending",
            parameters={
                "new_parameters": new_parameters
            }
        )
        
        self.active_operations[operation_id] = operation
        
        logger.info(f"Scheduled index rebuild: {index_name}")
        return operation_id
    
    async def get_index_statistics(
        self,
        index_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for indexes.
        
        Args:
            index_name: Specific index name, or None for all indexes
            
        Returns:
            Index statistics
        """
        if index_name:
            if index_name in self.index_statistics:
                stats = self.index_statistics[index_name]
                return {
                    "index_name": stats.index_name,
                    "index_type": stats.index_type.value,
                    "total_vectors": stats.total_vectors,
                    "index_size_mb": stats.index_size_bytes / (1024 * 1024),
                    "avg_query_time_ms": stats.avg_query_time_ms,
                    "queries_per_second": stats.queries_per_second,
                    "cache_hit_ratio": stats.cache_hit_ratio,
                    "fragmentation_ratio": stats.fragmentation_ratio,
                    "last_vacuum": stats.last_vacuum.isoformat() if stats.last_vacuum else None,
                    "last_analyze": stats.last_analyze.isoformat() if stats.last_analyze else None,
                    "last_updated": stats.last_updated.isoformat()
                }
            else:
                return {"error": f"No statistics available for index {index_name}"}
        
        # Return all index statistics
        all_stats = {}
        for name, stats in self.index_statistics.items():
            all_stats[name] = {
                "index_type": stats.index_type.value,
                "total_vectors": stats.total_vectors,
                "index_size_mb": stats.index_size_bytes / (1024 * 1024),
                "avg_query_time_ms": stats.avg_query_time_ms,
                "last_updated": stats.last_updated.isoformat()
            }
        
        return {
            "total_indexes": len(all_stats),
            "indexes": all_stats,
            "total_vectors": sum(s["total_vectors"] for s in all_stats.values()),
            "total_size_mb": sum(s["index_size_mb"] for s in all_stats.values())
        }
    
    async def get_operation_status(self, operation_id: str) -> Dict[str, Any]:
        """Get status of an index operation."""
        if operation_id not in self.active_operations:
            return {"error": "Operation not found"}
        
        operation = self.active_operations[operation_id]
        
        return {
            "operation_id": operation.operation_id,
            "operation_type": operation.operation_type.value,
            "index_name": operation.index_name,
            "status": operation.status,
            "progress_percent": operation.progress_percent,
            "started_at": operation.started_at.isoformat() if operation.started_at else None,
            "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
            "duration_seconds": operation.duration_seconds,
            "error_message": operation.error_message
        }
    
    async def suggest_index_optimization(
        self,
        index_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get optimization suggestions for indexes.
        
        Args:
            index_name: Specific index to analyze, or None for all
            
        Returns:
            Optimization suggestions
        """
        suggestions = []
        
        indexes_to_analyze = [index_name] if index_name else list(self.managed_indexes.keys())
        
        for idx_name in indexes_to_analyze:
            if idx_name not in self.managed_indexes:
                continue
                
            config = self.managed_indexes[idx_name]
            
            # Analyze performance trends
            performance_analysis = self.performance_analyzer.analyze_performance_trends(idx_name)
            
            if "error" not in performance_analysis:
                suggestion = {
                    "index_name": idx_name,
                    "current_type": config.index_type.value,
                    "performance_analysis": performance_analysis,
                    "recommendations": []
                }
                
                # Generate specific recommendations
                if performance_analysis["current_avg_ms"] > self.performance_threshold_ms:
                    suggestion["recommendations"].append({
                        "type": "performance_optimization",
                        "description": "Consider rebuilding index with optimized parameters",
                        "priority": "high"
                    })
                
                if performance_analysis["trend_direction"] == "degrading":
                    suggestion["recommendations"].append({
                        "type": "maintenance",
                        "description": "Schedule VACUUM and ANALYZE operations",
                        "priority": "medium"
                    })
                
                # Check if different index type might be better
                if idx_name in self.index_statistics:
                    stats = self.index_statistics[idx_name]
                    query_patterns = {"general": 100}  # Simplified for example
                    
                    optimal_type, type_params = self.performance_analyzer.suggest_optimal_index_type(
                        stats.total_vectors, query_patterns, self.memory_limit_mb
                    )
                    
                    if optimal_type != config.index_type:
                        suggestion["recommendations"].append({
                            "type": "index_type_change",
                            "description": f"Consider changing to {optimal_type.value}: {type_params['reason']}",
                            "new_type": optimal_type.value,
                            "parameters": type_params.get("parameters", {}),
                            "priority": "low"
                        })
                
                suggestions.append(suggestion)
        
        return {
            "total_suggestions": len(suggestions),
            "suggestions": suggestions,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _initialize_managed_indexes(self) -> None:
        """Initialize managed indexes from existing database indexes."""
        try:
            async with get_async_session() as session:
                # Query existing indexes
                result = await session.execute(text("""
                    SELECT indexname, indexdef 
                    FROM pg_indexes 
                    WHERE tablename = 'context' 
                    AND indexname LIKE '%embedding%'
                """))
                
                existing_indexes = result.all()
                
                for index_name, index_def in existing_indexes:
                    # Parse index definition to determine type
                    if "hnsw" in index_def.lower():
                        index_type = IndexType.HNSW
                    elif "ivfflat" in index_def.lower():
                        index_type = IndexType.IVFFLAT
                    else:
                        continue  # Skip unknown index types
                    
                    # Create configuration for existing index
                    config = IndexConfiguration(
                        index_name=index_name,
                        index_type=index_type,
                        table_name="context",
                        column_name="embedding",
                        created_at=datetime.utcnow()  # Approximate
                    )
                    
                    self.managed_indexes[index_name] = config
                    logger.info(f"Registered existing index: {index_name} ({index_type.value})")
                
        except Exception as e:
            logger.error(f"Failed to initialize managed indexes: {e}")
    
    async def _index_exists(self, index_name: str) -> bool:
        """Check if an index exists in the database."""
        try:
            async with get_async_session() as session:
                result = await session.execute(text("""
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = :index_name
                """), {"index_name": index_name})
                
                return result.first() is not None
                
        except Exception as e:
            logger.error(f"Failed to check index existence: {e}")
            return False
    
    async def _execute_index_operation(self, operation: IndexOperation) -> None:
        """Execute an index operation."""
        async with self._operation_semaphore:
            operation.status = "running"
            operation.started_at = datetime.utcnow()
            
            try:
                if operation.operation_type == OperationType.CREATE:
                    await self._execute_create_operation(operation)
                elif operation.operation_type == OperationType.REBUILD:
                    await self._execute_rebuild_operation(operation)
                elif operation.operation_type == OperationType.OPTIMIZE:
                    await self._execute_optimize_operation(operation)
                elif operation.operation_type == OperationType.ANALYZE:
                    await self._execute_analyze_operation(operation)
                elif operation.operation_type == OperationType.VACUUM:
                    await self._execute_vacuum_operation(operation)
                
                operation.status = "completed"
                
            except Exception as e:
                operation.status = "failed"
                operation.error_message = str(e)
                logger.error(f"Index operation failed: {operation.operation_id}: {e}")
            
            finally:
                operation.completed_at = datetime.utcnow()
                if operation.started_at:
                    operation.duration_seconds = (
                        operation.completed_at - operation.started_at
                    ).total_seconds()
    
    async def _execute_create_operation(self, operation: IndexOperation) -> None:
        """Execute index creation."""
        params = operation.parameters
        index_type = IndexType(params["index_type"])
        table_name = params["table_name"]
        column_name = params["column_name"]
        index_params = params["index_parameters"]
        
        async with get_async_session() as session:
            # Drop existing index if force is True
            if params.get("force"):
                try:
                    await session.execute(text(f"DROP INDEX IF EXISTS {operation.index_name}"))
                except Exception as e:
                    logger.warning(f"Failed to drop existing index: {e}")
            
            operation.progress_percent = 20.0
            
            # Create the appropriate index
            if index_type == IndexType.HNSW:
                m = index_params.get("m", 16)
                ef_construction = index_params.get("ef_construction", 64)
                
                create_sql = f"""
                CREATE INDEX {operation.index_name} ON {table_name} 
                USING hnsw ({column_name} vector_cosine_ops) 
                WITH (m = {m}, ef_construction = {ef_construction})
                """
                
            elif index_type == IndexType.IVFFLAT:
                lists = index_params.get("lists", 100)
                
                create_sql = f"""
                CREATE INDEX {operation.index_name} ON {table_name} 
                USING ivfflat ({column_name} vector_cosine_ops) 
                WITH (lists = {lists})
                """
            
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            operation.progress_percent = 50.0
            
            # Execute index creation
            await session.execute(text(create_sql))
            await session.commit()
            
            operation.progress_percent = 90.0
            
            # Update configuration
            if operation.index_name in self.managed_indexes:
                config = self.managed_indexes[operation.index_name]
                config.created_at = datetime.utcnow()
                config.parameters.update(index_params)
            
            operation.progress_percent = 100.0
            
            logger.info(f"Created index: {operation.index_name} ({index_type.value})")
    
    async def _execute_rebuild_operation(self, operation: IndexOperation) -> None:
        """Execute index rebuild."""
        # Get current configuration
        config = self.managed_indexes[operation.index_name]
        new_params = operation.parameters.get("new_parameters", {})
        
        async with get_async_session() as session:
            # Drop existing index
            await session.execute(text(f"DROP INDEX IF EXISTS {operation.index_name}"))
            operation.progress_percent = 25.0
            
            # Update parameters if provided
            if new_params:
                config.parameters.update(new_params)
            
            # Recreate index with same or updated parameters
            if config.index_type == IndexType.HNSW:
                m = config.parameters.get("m", 16)
                ef_construction = config.parameters.get("ef_construction", 64)
                
                create_sql = f"""
                CREATE INDEX {operation.index_name} ON {config.table_name} 
                USING hnsw ({config.column_name} vector_cosine_ops) 
                WITH (m = {m}, ef_construction = {ef_construction})
                """
                
            elif config.index_type == IndexType.IVFFLAT:
                lists = config.parameters.get("lists", 100)
                
                create_sql = f"""
                CREATE INDEX {operation.index_name} ON {config.table_name} 
                USING ivfflat ({config.column_name} vector_cosine_ops) 
                WITH (lists = {lists})
                """
            
            operation.progress_percent = 75.0
            
            # Execute rebuild
            await session.execute(text(create_sql))
            await session.commit()
            
            operation.progress_percent = 100.0
            
            # Update configuration
            config.last_optimized = datetime.utcnow()
            
            logger.info(f"Rebuilt index: {operation.index_name}")
    
    async def _execute_optimize_operation(self, operation: IndexOperation) -> None:
        """Execute index optimization."""
        async with get_async_session() as session:
            # Run VACUUM and ANALYZE
            await session.execute(text(f"VACUUM ANALYZE {self.managed_indexes[operation.index_name].table_name}"))
            operation.progress_percent = 50.0
            
            # Update statistics
            await session.execute(text(f"ANALYZE {self.managed_indexes[operation.index_name].table_name}"))
            operation.progress_percent = 100.0
            
            # Update configuration
            config = self.managed_indexes[operation.index_name]
            config.last_optimized = datetime.utcnow()
            
            logger.info(f"Optimized index: {operation.index_name}")
    
    async def _execute_analyze_operation(self, operation: IndexOperation) -> None:
        """Execute ANALYZE operation."""
        async with get_async_session() as session:
            table_name = self.managed_indexes[operation.index_name].table_name
            await session.execute(text(f"ANALYZE {table_name}"))
            operation.progress_percent = 100.0
            
            logger.info(f"Analyzed table for index: {operation.index_name}")
    
    async def _execute_vacuum_operation(self, operation: IndexOperation) -> None:
        """Execute VACUUM operation."""
        async with get_async_session() as session:
            table_name = self.managed_indexes[operation.index_name].table_name
            await session.execute(text(f"VACUUM {table_name}"))
            operation.progress_percent = 100.0
            
            logger.info(f"Vacuumed table for index: {operation.index_name}")
    
    async def _maintenance_scheduler(self) -> None:
        """Background maintenance scheduler."""
        logger.info("Starting index maintenance scheduler")
        
        while not self._shutdown_event.is_set():
            try:
                await self._schedule_maintenance_operations()
                
                # Wait for next maintenance cycle
                await asyncio.sleep(self.maintenance_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance scheduler error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
        
        logger.info("Index maintenance scheduler stopped")
    
    async def _schedule_maintenance_operations(self) -> None:
        """Schedule maintenance operations for managed indexes."""
        current_time = datetime.utcnow()
        
        for index_name, config in self.managed_indexes.items():
            try:
                # Check if maintenance is due
                last_optimized = config.last_optimized or config.created_at
                if not last_optimized:
                    continue
                
                hours_since_optimization = (current_time - last_optimized).total_seconds() / 3600
                
                if config.optimization_schedule == "daily" and hours_since_optimization >= 24:
                    # Schedule optimization
                    operation_id = await self.optimize_index(index_name, analyze_performance=True)
                    logger.info(f"Scheduled maintenance optimization for {index_name}: {operation_id}")
                
                elif config.optimization_schedule == "weekly" and hours_since_optimization >= 168:
                    # Schedule weekly optimization
                    operation_id = await self.optimize_index(index_name, analyze_performance=True)
                    logger.info(f"Scheduled weekly optimization for {index_name}: {operation_id}")
                
            except Exception as e:
                logger.error(f"Failed to schedule maintenance for {index_name}: {e}")
    
    async def _performance_monitor(self) -> None:
        """Background performance monitor."""
        logger.info("Starting index performance monitor")
        
        while not self._shutdown_event.is_set():
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Index performance monitor stopped")
    
    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics for managed indexes."""
        try:
            async with get_async_session() as session:
                for index_name, config in self.managed_indexes.items():
                    # Get index size and statistics
                    size_result = await session.execute(text("""
                        SELECT pg_size_pretty(pg_relation_size(:index_name)) as size_pretty,
                               pg_relation_size(:index_name) as size_bytes
                    """), {"index_name": index_name})
                    
                    size_row = size_result.first()
                    if size_row:
                        index_size_bytes = size_row.size_bytes
                    else:
                        index_size_bytes = 0
                    
                    # Get vector count
                    count_result = await session.execute(text(f"""
                        SELECT COUNT(*) as total_vectors 
                        FROM {config.table_name} 
                        WHERE {config.column_name} IS NOT NULL
                    """))
                    
                    count_row = count_result.first()
                    total_vectors = count_row.total_vectors if count_row else 0
                    
                    # Update statistics
                    self.index_statistics[index_name] = IndexStatistics(
                        index_name=index_name,
                        index_type=config.index_type,
                        total_vectors=total_vectors,
                        index_size_bytes=index_size_bytes,
                        avg_query_time_ms=0.0,  # Would be updated by query performance
                        queries_per_second=0.0,
                        cache_hit_ratio=0.0,
                        fragmentation_ratio=0.0,
                        last_vacuum=None,  # Would need to track this
                        last_analyze=None,  # Would need to track this
                        creation_time=config.created_at,
                        last_updated=datetime.utcnow()
                    )
                    
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    async def _operation_manager(self) -> None:
        """Background operation manager."""
        logger.info("Starting index operation manager")
        
        while not self._shutdown_event.is_set():
            try:
                # Process pending operations
                pending_operations = [
                    op for op in self.active_operations.values()
                    if op.status == "pending"
                ]
                
                for operation in pending_operations:
                    # Check if we can start this operation
                    if len([op for op in self.active_operations.values() if op.status == "running"]) < 2:
                        # Start the operation
                        asyncio.create_task(self._execute_index_operation(operation))
                
                # Clean up completed operations (keep for 1 hour)
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                completed_operations = [
                    op_id for op_id, op in self.active_operations.items()
                    if op.status in ["completed", "failed", "cancelled"] and
                       op.completed_at and op.completed_at < cutoff_time
                ]
                
                for op_id in completed_operations:
                    del self.active_operations[op_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Operation manager error: {e}")
                await asyncio.sleep(30)
        
        logger.info("Index operation manager stopped")
    
    async def _statistics_collector(self) -> None:
        """Background statistics collector."""
        logger.info("Starting index statistics collector")
        
        while not self._shutdown_event.is_set():
            try:
                # Store statistics in Redis for dashboard
                stats_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "managed_indexes": len(self.managed_indexes),
                    "active_operations": len([op for op in self.active_operations.values() if op.status == "running"]),
                    "total_vectors": sum(
                        stats.total_vectors for stats in self.index_statistics.values()
                    ),
                    "total_index_size_mb": sum(
                        stats.index_size_bytes for stats in self.index_statistics.values()
                    ) / (1024 * 1024)
                }
                
                await self.redis_client.set(
                    "index_management:statistics",
                    json.dumps(stats_data),
                    expire=3600
                )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Statistics collector error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Index statistics collector stopped")


# Global instance for application use
_index_manager: Optional[IndexManager] = None


async def get_index_manager() -> IndexManager:
    """
    Get singleton index manager instance.
    
    Returns:
        IndexManager instance
    """
    global _index_manager
    
    if _index_manager is None:
        _index_manager = IndexManager()
        await _index_manager.start()
    
    return _index_manager


async def cleanup_index_manager() -> None:
    """Cleanup index manager resources."""
    global _index_manager
    
    if _index_manager:
        await _index_manager.stop()
        _index_manager = None