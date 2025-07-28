"""
Context Engine Integration Service - Complete Context Management System.

Provides unified interface for all context operations including:
- Embedding generation and semantic search
- Automated consolidation with usage pattern triggers
- Memory management with cleanup policies and optimization
- Intelligent caching for performance
- Context lifecycle management with versioning
- Integration with orchestrator and sleep-wake cycles
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, AsyncIterator
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from sqlalchemy import select, and_, or_, func, update, delete, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..schemas.context import ContextCreate, ContextUpdate, ContextSearchRequest
from ..core.database import get_async_session
from ..core.context_manager import ContextManager, get_context_manager
from ..core.embedding_service import EmbeddingService, get_embedding_service
from ..core.enhanced_context_consolidator import (
    UltraCompressedContextMode, 
    get_ultra_compressed_context_mode,
    CompressionMetrics
)
from ..core.enhanced_vector_search import EnhancedVectorSearchEngine, create_enhanced_search_engine
from ..core.vector_search import SearchFilters, ContextMatch
from ..core.context_analytics import ContextAnalyticsManager
from ..core.redis import get_redis_client
from ..core.config import get_settings

# Import optimization components
from ..core.advanced_vector_search import AdvancedVectorSearchEngine, SimilarityAlgorithm
from ..core.optimized_embedding_pipeline import OptimizedEmbeddingPipeline, get_optimized_embedding_pipeline
from ..core.search_analytics import SearchAnalytics, get_search_analytics
from ..core.index_management import IndexManager
from ..core.hybrid_search_engine import HybridSearchEngine


logger = logging.getLogger(__name__)


class ContextEngineStatus(Enum):
    """Context engine operational status."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class ConsolidationTrigger(Enum):
    """Triggers for context consolidation."""
    USAGE_THRESHOLD = "usage_threshold"
    TIME_BASED = "time_based"
    MEMORY_PRESSURE = "memory_pressure"
    MANUAL = "manual"
    SLEEP_CYCLE = "sleep_cycle"


@dataclass
class ContextEngineConfig:
    """Configuration for context engine behavior."""
    # Consolidation settings
    auto_consolidation_enabled: bool = True
    consolidation_usage_threshold: int = 10
    consolidation_time_threshold_hours: int = 24
    consolidation_batch_size: int = 50
    max_consolidation_concurrency: int = 3
    
    # Memory management
    memory_cleanup_enabled: bool = True
    memory_cleanup_interval_hours: int = 6
    context_retention_days: int = 90
    low_importance_threshold: float = 0.3
    memory_pressure_threshold_mb: int = 1000
    
    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 10000
    cache_cleanup_interval_minutes: int = 30
    
    # Performance
    max_search_time_ms: float = 500.0
    embedding_batch_size: int = 100
    max_concurrent_operations: int = 10
    
    # Analytics
    analytics_enabled: bool = True
    metrics_retention_days: int = 30


@dataclass
class ContextEngineMetrics:
    """Comprehensive metrics for context engine performance."""
    # Operational metrics
    total_contexts: int = 0
    consolidated_contexts: int = 0
    cached_contexts: int = 0
    active_searches: int = 0
    
    # Performance metrics
    avg_search_time_ms: float = 0.0
    avg_consolidation_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Health metrics
    error_rate: float = 0.0
    uptime_hours: float = 0.0
    last_maintenance: Optional[datetime] = None
    
    # Usage statistics
    searches_per_hour: float = 0.0
    consolidations_per_hour: float = 0.0
    contexts_created_per_hour: float = 0.0


class ContextEngineIntegration:
    """
    Comprehensive Context Engine Integration Service.
    
    Provides unified interface for all context management operations with:
    - Intelligent context storage and retrieval
    - Automated consolidation based on usage patterns
    - Memory management and cleanup
    - Performance optimization and caching
    - Integration with orchestrator and agent lifecycle
    """
    
    def __init__(
        self,
        config: Optional[ContextEngineConfig] = None,
        context_manager: Optional[ContextManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        consolidator: Optional[UltraCompressedContextMode] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        """
        Initialize context engine integration.
        
        Args:
            config: Engine configuration
            context_manager: Context manager instance
            embedding_service: Embedding service instance
            consolidator: Context consolidator instance
            redis_client: Redis client for caching
        """
        self.config = config or ContextEngineConfig()
        self.settings = get_settings()
        
        # Core services
        self.context_manager = context_manager or get_context_manager()
        self.embedding_service = embedding_service or get_embedding_service()
        self.consolidator = consolidator or get_ultra_compressed_context_mode()
        self.redis_client = redis_client or get_redis_client()
        
        # Search engine (initialized lazily)
        self._search_engine: Optional[EnhancedVectorSearchEngine] = None
        self._analytics_manager: Optional[ContextAnalyticsManager] = None
        
        # Optimization components (initialized lazily for performance)
        self._advanced_search_engine: Optional[AdvancedVectorSearchEngine] = None
        self._optimized_embedding_pipeline: Optional[OptimizedEmbeddingPipeline] = None
        self._search_analytics: Optional[SearchAnalytics] = None
        self._index_manager: Optional[IndexManager] = None
        self._hybrid_search_engine: Optional[HybridSearchEngine] = None
        
        # State management
        self.status = ContextEngineStatus.INITIALIZING
        self.start_time = datetime.utcnow()
        self.metrics = ContextEngineMetrics()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._consolidation_queue: asyncio.Queue = asyncio.Queue()
        self._cleanup_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self._operation_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._error_counts: Dict[str, int] = defaultdict(int)
        
        # Cache management
        self._context_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Initialization flag
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the context engine and start background services."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Context Engine Integration Service")
            
            # Initialize database session and search engine
            db_session = await get_async_session()
            self._search_engine = await create_enhanced_search_engine(
                db_session=db_session,
                embedding_service=self.embedding_service,
                redis_client=self.redis_client
            )
            
            # Initialize analytics manager
            self._analytics_manager = ContextAnalyticsManager(
                db_session=db_session,
                redis_client=self.redis_client
            )
            
            # Initialize optimization components
            await self._initialize_optimization_components(db_session)
            
            # Start background services
            if self.config.auto_consolidation_enabled:
                self._background_tasks.append(
                    asyncio.create_task(self._consolidation_worker())
                )
            
            if self.config.memory_cleanup_enabled:
                self._background_tasks.append(
                    asyncio.create_task(self._memory_cleanup_worker())
                )
            
            if self.config.cache_enabled:
                self._background_tasks.append(
                    asyncio.create_task(self._cache_cleanup_worker())
                )
            
            # Start metrics collection
            self._background_tasks.append(
                asyncio.create_task(self._metrics_collector())
            )
            
            self.status = ContextEngineStatus.HEALTHY
            self._initialized = True
            
            logger.info("Context Engine Integration Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Context Engine Integration: {e}")
            self.status = ContextEngineStatus.UNHEALTHY
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the context engine and cleanup resources."""
        logger.info("Shutting down Context Engine Integration Service")
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Cleanup optimization components
        await self._cleanup_optimization_components()
        
        # Cleanup resources
        await self.context_manager.cleanup()
        self._context_cache.clear()
        self._cache_timestamps.clear()
        
        self.status = ContextEngineStatus.UNHEALTHY
        self._initialized = False
        
        logger.info("Context Engine Integration Service shutdown complete")
    
    async def _initialize_optimization_components(self, db_session: AsyncSession) -> None:
        """Initialize optimization components for enhanced performance."""
        try:
            logger.info("Initializing optimization components")
            
            # Initialize search analytics
            self._search_analytics = await get_search_analytics()
            
            # Initialize optimized embedding pipeline
            self._optimized_embedding_pipeline = await get_optimized_embedding_pipeline()
            
            # Initialize index manager
            self._index_manager = IndexManager(
                db_session=db_session,
                redis_client=self.redis_client
            )
            
            # Initialize advanced vector search engine
            self._advanced_search_engine = AdvancedVectorSearchEngine(
                db_session=db_session,
                embedding_service=self.embedding_service,
                redis_client=self.redis_client
            )
            
            # Initialize hybrid search engine
            self._hybrid_search_engine = HybridSearchEngine(
                db_session=db_session,
                vector_search_engine=self._advanced_search_engine,
                redis_client=self.redis_client
            )
            
            logger.info("Optimization components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization components: {e}")
            raise
    
    async def _cleanup_optimization_components(self) -> None:
        """Cleanup optimization components."""
        try:
            # Cleanup search analytics
            if self._search_analytics:
                await self._search_analytics.stop()
                self._search_analytics = None
            
            # Cleanup optimized embedding pipeline
            if self._optimized_embedding_pipeline:
                await self._optimized_embedding_pipeline.stop()
                self._optimized_embedding_pipeline = None
            
            # Clear other references
            self._index_manager = None
            self._advanced_search_engine = None
            self._hybrid_search_engine = None
            
            logger.info("Optimization components cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up optimization components: {e}")
    
    async def store_context_enhanced(
        self,
        context_data: ContextCreate,
        generate_embedding: bool = True,
        enable_auto_consolidation: bool = True,
        cache_result: bool = True
    ) -> Context:
        """
        Store context with enhanced features.
        
        Args:
            context_data: Context data to store
            generate_embedding: Whether to generate embedding
            enable_auto_consolidation: Whether to trigger auto consolidation check
            cache_result: Whether to cache the result
            
        Returns:
            Stored context with generated embedding
        """
        start_time = time.perf_counter()
        
        try:
            # Store context using context manager
            context = await self.context_manager.store_context(
                context_data=context_data,
                auto_embed=generate_embedding,
                auto_compress=False  # We'll handle compression separately
            )
            
            # Cache the result
            if cache_result and self.config.cache_enabled:
                await self._cache_context(context)
            
            # Check for auto consolidation trigger
            if enable_auto_consolidation and self.config.auto_consolidation_enabled:
                await self._check_consolidation_triggers(context_data.agent_id)
            
            # Record metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._record_operation_metric("store_context", processing_time)
            
            if self._analytics_manager:
                await self._analytics_manager.record_context_creation(
                    context_id=context.id,
                    agent_id=context_data.agent_id,
                    context_type=context_data.context_type,
                    importance_score=context_data.importance_score,
                    processing_time_ms=processing_time
                )
            
            logger.debug(f"Enhanced context storage completed in {processing_time:.2f}ms")
            return context
            
        except Exception as e:
            self._error_counts["store_context"] += 1
            logger.error(f"Enhanced context storage failed: {e}")
            raise
    
    async def search_contexts_enhanced(
        self,
        request: ContextSearchRequest,
        use_cache: bool = True,
        enable_analytics: bool = True
    ) -> Tuple[List[ContextMatch], Dict[str, Any]]:
        """
        Perform enhanced context search with caching and analytics.
        
        Args:
            request: Search request parameters
            use_cache: Whether to use caching
            enable_analytics: Whether to record analytics
            
        Returns:
            Tuple of (search results, search metadata)
        """
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            if use_cache and self.config.cache_enabled:
                cache_key = self._generate_search_cache_key(request)
                cached_results = await self._get_cached_search_results(cache_key)
                if cached_results:
                    search_time = (time.perf_counter() - start_time) * 1000
                    return cached_results, {
                        "cache_hit": True,
                        "search_time_ms": search_time,
                        "results_count": len(cached_results)
                    }
            
            # Perform enhanced search
            if not self._search_engine:
                await self.initialize()
            
            results, metadata = await self._search_engine.enhanced_search(
                query=request.query,
                agent_id=request.agent_id,
                limit=request.limit,
                filters=SearchFilters(
                    context_types=[request.context_type] if request.context_type else None,
                    min_similarity=request.min_relevance
                ),
                performance_target_ms=self.config.max_search_time_ms
            )
            
            # Cache results
            if use_cache and self.config.cache_enabled and not metadata.get("cache_hit"):
                await self._cache_search_results(cache_key, results)
            
            # Record analytics
            search_time = metadata.get("search_time_ms", 0)
            self._record_operation_metric("search_contexts", search_time)
            
            if enable_analytics and self._analytics_manager:
                for match in results[:5]:  # Record top 5 results
                    await self._analytics_manager.record_context_retrieval(
                        context_id=match.context.id,
                        requesting_agent_id=request.agent_id,
                        query_text=request.query,
                        similarity_score=match.similarity_score,
                        relevance_score=match.relevance_score,
                        rank_position=match.rank,
                        response_time_ms=search_time,
                        retrieval_method=metadata.get("search_method", "enhanced")
                    )
            
            return results, metadata
            
        except Exception as e:
            self._error_counts["search_contexts"] += 1
            logger.error(f"Enhanced context search failed: {e}")
            raise
    
    async def search_contexts_optimized(
        self,
        request: ContextSearchRequest,
        use_advanced_search: bool = True,
        use_hybrid_search: bool = False,
        enable_analytics: bool = True
    ) -> Tuple[List[ContextMatch], Dict[str, Any]]:
        """
        Perform optimized context search with advanced algorithms.
        
        Args:
            request: Search request parameters
            use_advanced_search: Whether to use advanced vector search engine
            use_hybrid_search: Whether to use hybrid search (vector + text)
            enable_analytics: Whether to record analytics
            
        Returns:
            Tuple of (search results, search metadata with performance metrics)
        """
        start_time = time.perf_counter()
        
        try:
            # Ensure optimization components are initialized
            if not self._initialized:
                await self.initialize()
            
            # Record search event for analytics
            if enable_analytics and self._search_analytics:
                from ..core.search_analytics import SearchEventType
                await self._search_analytics.record_search_event(
                    event_type=SearchEventType.QUERY_SUBMITTED,
                    query=request.query,
                    agent_id=str(request.agent_id) if request.agent_id else None
                )
            
            search_results = []
            search_metadata = {}
            
            # Choose search method based on preferences
            if use_hybrid_search and self._hybrid_search_engine:
                # Use hybrid search (vector + text)
                search_results, search_metadata = await self._hybrid_search_engine.hybrid_search(
                    query=request.query,
                    agent_id=request.agent_id,
                    limit=request.limit or 10,
                    filters=SearchFilters(
                        context_type=request.context_type,
                        agent_id=request.agent_id,
                        importance_threshold=request.importance_threshold,
                        date_range=(request.start_date, request.end_date)
                    ) if any([request.context_type, request.importance_threshold, request.start_date]) else None
                )
                search_metadata["search_method"] = "hybrid"
                
            elif use_advanced_search and self._advanced_search_engine:
                # Use advanced vector search
                search_results, search_metadata = await self._advanced_search_engine.ultra_fast_search(
                    query=request.query,
                    agent_id=request.agent_id,
                    limit=request.limit or 10,
                    filters=SearchFilters(
                        context_type=request.context_type,
                        agent_id=request.agent_id,
                        importance_threshold=request.importance_threshold,
                        date_range=(request.start_date, request.end_date)
                    ) if any([request.context_type, request.importance_threshold, request.start_date]) else None,
                    similarity_algorithm=SimilarityAlgorithm.MIXED,
                    performance_target_ms=25.0
                )
                search_metadata["search_method"] = "advanced_vector"
                
            else:
                # Fallback to enhanced search
                search_results, search_metadata = await self.search_contexts_enhanced(
                    request=request,
                    use_cache=True,
                    enable_analytics=False  # Already recorded above
                )
                search_metadata["search_method"] = "enhanced_fallback"
            
            # Record results for analytics
            if enable_analytics and self._search_analytics:
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                await self._search_analytics.record_search_results(
                    query=request.query,
                    results=search_results,
                    processing_time_ms=processing_time_ms,
                    search_metadata=search_metadata,
                    agent_id=str(request.agent_id) if request.agent_id else None
                )
            
            # Add optimization metrics
            search_metadata.update({
                "optimization_enabled": True,
                "advanced_search_used": use_advanced_search and self._advanced_search_engine is not None,
                "hybrid_search_used": use_hybrid_search and self._hybrid_search_engine is not None,
                "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                "results_count": len(search_results)
            })
            
            return search_results, search_metadata
            
        except Exception as e:
            self._error_counts["search_contexts_optimized"] += 1
            logger.error(f"Optimized context search failed: {e}")
            
            # Fallback to enhanced search
            try:
                return await self.search_contexts_enhanced(request=request, enable_analytics=enable_analytics)
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                raise
    
    async def generate_embedding_optimized(
        self,
        text: str,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Generate embedding using optimized pipeline.
        
        Args:
            text: Text to embed
            priority: Processing priority (1=high, 2=medium, 3=low)  
            metadata: Additional metadata
            
        Returns:
            Generated embedding vector
        """
        try:
            # Use optimized embedding pipeline if available
            if self._optimized_embedding_pipeline:
                result = await self._optimized_embedding_pipeline.generate_embedding_optimized(
                    text=text,
                    priority=priority,
                    metadata=metadata
                )
                if result.embedding:
                    return result.embedding
            
            # Fallback to standard embedding service
            return await self.embedding_service.generate_embedding(text)
            
        except Exception as e:
            logger.error(f"Optimized embedding generation failed: {e}")
            # Fallback to standard embedding service
            return await self.embedding_service.generate_embedding(text)
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization metrics."""
        metrics = {
            "optimization_components_status": {
                "advanced_search_engine": self._advanced_search_engine is not None,
                "optimized_embedding_pipeline": self._optimized_embedding_pipeline is not None,
                "search_analytics": self._search_analytics is not None,
                "index_manager": self._index_manager is not None,
                "hybrid_search_engine": self._hybrid_search_engine is not None
            }
        }
        
        # Add performance metrics from each component
        if self._optimized_embedding_pipeline:
            metrics["embedding_pipeline"] = self._optimized_embedding_pipeline.get_performance_metrics()
        
        if self._search_analytics:
            metrics["search_analytics"] = await self._search_analytics.get_performance_summary()
        
        if self._index_manager:
            metrics["index_management"] = await self._index_manager.get_index_status()
        
        return metrics
    
    async def trigger_consolidation(
        self,
        agent_id: UUID,
        trigger_type: ConsolidationTrigger = ConsolidationTrigger.MANUAL,
        target_reduction: float = 0.70
    ) -> CompressionMetrics:
        """
        Trigger context consolidation for an agent.
        
        Args:
            agent_id: Agent ID to consolidate contexts for
            trigger_type: What triggered the consolidation
            target_reduction: Target compression ratio
            
        Returns:
            Compression metrics
        """
        start_time = time.perf_counter()
        
        try:
            logger.info(f"Triggering consolidation for agent {agent_id} ({trigger_type.value})")
            
            # Perform ultra compression
            metrics = await self.consolidator.ultra_compress_agent_contexts(
                agent_id=agent_id,
                target_reduction=target_reduction,
                preserve_critical=True
            )
            
            # Record metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._record_operation_metric("consolidation", processing_time)
            
            # Clear relevant caches
            await self._invalidate_agent_caches(agent_id)
            
            # Record analytics
            if self._analytics_manager:
                await self._analytics_manager.record_consolidation_event(
                    agent_id=agent_id,
                    trigger_type=trigger_type.value,
                    contexts_processed=metrics.contexts_merged + metrics.contexts_archived,
                    compression_ratio=metrics.compression_ratio,
                    processing_time_ms=processing_time
                )
            
            logger.info(
                f"Consolidation completed for agent {agent_id}: "
                f"{metrics.compression_ratio:.1%} reduction in {processing_time:.0f}ms"
            )
            
            return metrics
            
        except Exception as e:
            self._error_counts["consolidation"] += 1
            logger.error(f"Context consolidation failed for agent {agent_id}: {e}")
            raise
    
    async def manage_context_lifecycle(
        self,
        context_id: UUID,
        action: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Manage context lifecycle operations.
        
        Args:
            context_id: Context ID to manage
            action: Lifecycle action (archive, restore, version, etc.)
            **kwargs: Additional parameters for the action
            
        Returns:
            Action result information
        """
        start_time = time.perf_counter()
        
        try:
            result = {"action": action, "context_id": str(context_id), "success": False}
            
            if action == "archive":
                success = await self.context_manager.delete_context(context_id)
                result["success"] = success
                
            elif action == "restore":
                # Implementation for restoring archived context
                result["success"] = await self._restore_context(context_id)
                
            elif action == "version":
                # Implementation for creating context version
                result["version_id"] = await self._create_context_version(context_id, **kwargs)
                result["success"] = True
                
            elif action == "cleanup":
                # Implementation for context cleanup
                result["cleaned_items"] = await self._cleanup_context(context_id, **kwargs)
                result["success"] = True
                
            else:
                raise ValueError(f"Unknown lifecycle action: {action}")
            
            # Record metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._record_operation_metric(f"lifecycle_{action}", processing_time)
            
            result["processing_time_ms"] = processing_time
            return result
            
        except Exception as e:
            self._error_counts[f"lifecycle_{action}"] += 1
            logger.error(f"Context lifecycle action '{action}' failed for {context_id}: {e}")
            raise
    
    async def optimize_memory_usage(
        self,
        agent_id: Optional[UUID] = None,
        force_cleanup: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize memory usage by cleaning up old/unused contexts.
        
        Args:
            agent_id: Specific agent to optimize (optional)
            force_cleanup: Force cleanup regardless of thresholds
            
        Returns:
            Cleanup results
        """
        start_time = time.perf_counter()
        
        try:
            results = {
                "contexts_cleaned": 0,
                "contexts_archived": 0,
                "cache_entries_cleared": 0,
                "memory_freed_mb": 0.0
            }
            
            # Check memory pressure
            current_memory_mb = await self._estimate_memory_usage()
            should_cleanup = (
                force_cleanup or 
                current_memory_mb > self.config.memory_pressure_threshold_mb
            )
            
            if not should_cleanup:
                logger.debug("Memory optimization skipped - no pressure detected")
                return results
            
            logger.info(f"Starting memory optimization (current usage: {current_memory_mb:.1f}MB)")
            
            # Cleanup old contexts
            if agent_id:
                cleaned = await self.context_manager.cleanup_old_contexts(
                    max_age_days=self.config.context_retention_days,
                    min_importance_threshold=self.config.low_importance_threshold
                )
                results["contexts_cleaned"] = cleaned
            else:
                # Cleanup for all agents
                async with get_async_session() as session:
                    agents_result = await session.execute(select(Agent.id))
                    agent_ids = [row[0] for row in agents_result.all()]
                    
                    total_cleaned = 0
                    for aid in agent_ids:
                        cleaned = await self.context_manager.cleanup_old_contexts(
                            max_age_days=self.config.context_retention_days,
                            min_importance_threshold=self.config.low_importance_threshold
                        )
                        total_cleaned += cleaned
                    
                    results["contexts_cleaned"] = total_cleaned
            
            # Clear caches
            cache_cleared = await self._clear_stale_caches()
            results["cache_entries_cleared"] = cache_cleared
            
            # Estimate memory freed
            new_memory_mb = await self._estimate_memory_usage()
            results["memory_freed_mb"] = max(0, current_memory_mb - new_memory_mb)
            
            # Record metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._record_operation_metric("memory_optimization", processing_time)
            
            logger.info(
                f"Memory optimization completed: {results['contexts_cleaned']} contexts cleaned, "
                f"{results['memory_freed_mb']:.1f}MB freed in {processing_time:.0f}ms"
            )
            
            return results
            
        except Exception as e:
            self._error_counts["memory_optimization"] += 1
            logger.error(f"Memory optimization failed: {e}")
            raise
    
    async def get_comprehensive_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the context engine."""
        try:
            health = {
                "status": self.status.value,
                "uptime_hours": (datetime.utcnow() - self.start_time).total_seconds() / 3600,
                "initialized": self._initialized,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Component health
            health["components"] = {}
            
            # Context manager health
            context_health = await self.context_manager.health_check()
            health["components"]["context_manager"] = context_health
            
            # Embedding service health
            embedding_health = await self.embedding_service.health_check()
            health["components"]["embedding_service"] = embedding_health
            
            # Search engine health
            if self._search_engine:
                search_metrics = self._search_engine.get_enhanced_performance_metrics()
                health["components"]["search_engine"] = {
                    "status": "healthy",
                    "metrics": search_metrics
                }
            
            # Redis health
            try:
                await self.redis_client.ping()
                health["components"]["redis"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
            
            # Performance metrics
            health["performance"] = await self._get_performance_metrics()
            
            # Resource usage
            health["resources"] = {
                "memory_usage_mb": await self._estimate_memory_usage(),
                "cache_size": len(self._context_cache),
                "active_tasks": len([t for t in self._background_tasks if not t.done()])
            }
            
            # Error rates
            total_operations = sum(len(metrics) for metrics in self._operation_metrics.values())
            total_errors = sum(self._error_counts.values())
            health["error_rate"] = total_errors / max(1, total_operations)
            
            # Overall status determination
            component_statuses = [comp.get("status") for comp in health["components"].values()]
            if "unhealthy" in component_statuses:
                health["status"] = ContextEngineStatus.UNHEALTHY.value
            elif "degraded" in component_statuses:
                health["status"] = ContextEngineStatus.DEGRADED.value
            else:
                health["status"] = ContextEngineStatus.HEALTHY.value
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": ContextEngineStatus.UNHEALTHY.value,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Background Workers
    
    async def _consolidation_worker(self) -> None:
        """Background worker for automatic context consolidation."""
        logger.info("Starting consolidation worker")
        
        try:
            while True:
                try:
                    # Check all agents for consolidation needs
                    async with get_async_session() as session:
                        agents_result = await session.execute(select(Agent.id))
                        agent_ids = [row[0] for row in agents_result.all()]
                    
                    for agent_id in agent_ids:
                        try:
                            # Check if consolidation is needed
                            if await self._should_consolidate_agent(agent_id):
                                await self._consolidation_queue.put({
                                    "agent_id": agent_id,
                                    "trigger": ConsolidationTrigger.USAGE_THRESHOLD,
                                    "timestamp": datetime.utcnow()
                                })
                        except Exception as e:
                            logger.warning(f"Error checking consolidation for agent {agent_id}: {e}")
                    
                    # Process consolidation queue
                    while not self._consolidation_queue.empty():
                        try:
                            consolidation_job = await asyncio.wait_for(
                                self._consolidation_queue.get(), timeout=1.0
                            )
                            
                            await self.trigger_consolidation(
                                agent_id=consolidation_job["agent_id"],
                                trigger_type=consolidation_job["trigger"]
                            )
                            
                        except asyncio.TimeoutError:
                            break
                        except Exception as e:
                            logger.error(f"Consolidation job failed: {e}")
                    
                    # Wait before next check
                    await asyncio.sleep(self.config.consolidation_time_threshold_hours * 3600)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Consolidation worker error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Consolidation worker stopped")
    
    async def _memory_cleanup_worker(self) -> None:
        """Background worker for memory cleanup operations."""
        logger.info("Starting memory cleanup worker")
        
        try:
            while True:
                try:
                    # Perform memory optimization
                    await self.optimize_memory_usage()
                    
                    # Wait for next cleanup cycle
                    await asyncio.sleep(self.config.memory_cleanup_interval_hours * 3600)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Memory cleanup worker error: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
                    
        except asyncio.CancelledError:
            logger.info("Memory cleanup worker stopped")
    
    async def _cache_cleanup_worker(self) -> None:
        """Background worker for cache cleanup operations."""
        logger.info("Starting cache cleanup worker")
        
        try:
            while True:
                try:
                    await self._clear_stale_caches()
                    await asyncio.sleep(self.config.cache_cleanup_interval_minutes * 60)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cache cleanup worker error: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("Cache cleanup worker stopped")
    
    async def _metrics_collector(self) -> None:
        """Background worker for collecting and updating metrics."""
        logger.info("Starting metrics collector")
        
        try:
            while True:
                try:
                    await self._update_metrics()
                    await asyncio.sleep(60)  # Update metrics every minute
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Metrics collector error: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("Metrics collector stopped")
    
    # Helper Methods
    
    async def _should_consolidate_agent(self, agent_id: UUID) -> bool:
        """Check if an agent needs context consolidation."""
        try:
            async with get_async_session() as session:
                # Count unconsolidated contexts
                unconsolidated_count = await session.scalar(
                    select(func.count(Context.id)).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.is_consolidated == "false",
                            Context.created_at < datetime.utcnow() - timedelta(hours=1)
                        )
                    )
                )
                
                return (unconsolidated_count or 0) >= self.config.consolidation_usage_threshold
                
        except Exception as e:
            logger.error(f"Error checking consolidation need for agent {agent_id}: {e}")
            return False
    
    async def _cache_context(self, context: Context) -> None:
        """Cache context for faster retrieval."""
        if not self.config.cache_enabled:
            return
        
        try:
            cache_key = f"context:{context.id}"
            self._context_cache[cache_key] = context.to_dict()
            self._cache_timestamps[cache_key] = datetime.utcnow()
            
            # Also cache in Redis
            await self.redis_client.setex(
                cache_key,
                self.config.cache_ttl_seconds,
                json.dumps(context.to_dict(), default=str)
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache context {context.id}: {e}")
    
    def _generate_search_cache_key(self, request: ContextSearchRequest) -> str:
        """Generate cache key for search request."""
        import hashlib
        
        key_data = {
            "query": request.query,
            "agent_id": str(request.agent_id) if request.agent_id else None,
            "context_type": request.context_type.value if request.context_type else None,
            "limit": request.limit,
            "min_relevance": request.min_relevance
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    async def _get_cached_search_results(self, cache_key: str) -> Optional[List[ContextMatch]]:
        """Get cached search results."""
        # Implementation would need proper deserialization
        # This is simplified for the example
        return None
    
    async def _cache_search_results(self, cache_key: str, results: List[ContextMatch]) -> None:
        """Cache search results."""
        # Implementation would need proper serialization
        # This is simplified for the example
        pass
    
    async def _invalidate_agent_caches(self, agent_id: UUID) -> None:
        """Invalidate all caches related to an agent."""
        try:
            # Clear memory cache
            keys_to_remove = [
                key for key in self._context_cache.keys()
                if f"agent:{agent_id}" in key
            ]
            
            for key in keys_to_remove:
                self._context_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
            
            # Clear Redis cache
            pattern = f"*agent:{agent_id}*"
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis_client.delete(*keys)
                
        except Exception as e:
            logger.warning(f"Failed to invalidate caches for agent {agent_id}: {e}")
    
    async def _clear_stale_caches(self) -> int:
        """Clear stale cache entries."""
        cleared_count = 0
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.cache_ttl_seconds)
        
        try:
            # Clear memory cache
            stale_keys = [
                key for key, timestamp in self._cache_timestamps.items()
                if timestamp < cutoff_time
            ]
            
            for key in stale_keys:
                self._context_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
                cleared_count += 1
            
            return cleared_count
            
        except Exception as e:
            logger.warning(f"Failed to clear stale caches: {e}")
            return cleared_count
    
    async def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        # Simplified estimation - in production would use more accurate methods
        try:
            import sys
            
            # Estimate based on cache sizes and data structures
            cache_size = len(self._context_cache) * 1024  # Rough estimate
            metrics_size = sum(len(m) for m in self._operation_metrics.values()) * 100
            
            return (cache_size + metrics_size) / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0
    
    def _record_operation_metric(self, operation: str, duration_ms: float) -> None:
        """Record operation performance metric."""
        self._operation_metrics[operation].append(duration_ms)
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {}
        
        for operation, durations in self._operation_metrics.items():
            if durations:
                metrics[operation] = {
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "operation_count": len(durations)
                }
        
        return metrics
    
    async def _update_metrics(self) -> None:
        """Update comprehensive metrics."""
        try:
            # Update context counts
            async with get_async_session() as session:
                self.metrics.total_contexts = await session.scalar(
                    select(func.count(Context.id))
                ) or 0
                
                self.metrics.consolidated_contexts = await session.scalar(
                    select(func.count(Context.id)).where(Context.is_consolidated == "true")
                ) or 0
            
            # Update performance metrics
            search_durations = self._operation_metrics.get("search_contexts", [])
            if search_durations:
                self.metrics.avg_search_time_ms = sum(search_durations) / len(search_durations)
            
            consolidation_durations = self._operation_metrics.get("consolidation", [])
            if consolidation_durations:
                self.metrics.avg_consolidation_time_ms = sum(consolidation_durations) / len(consolidation_durations)
            
            # Update cache metrics
            self.metrics.cached_contexts = len(self._context_cache)
            
            # Update uptime
            self.metrics.uptime_hours = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    # Lifecycle management helpers
    
    async def _restore_context(self, context_id: UUID) -> bool:
        """Restore an archived context."""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    update(Context)
                    .where(Context.id == context_id)
                    .values(context_metadata=func.jsonb_set(
                        Context.context_metadata,
                        '{archived}',
                        'false'
                    ))
                )
                await session.commit()
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to restore context {context_id}: {e}")
            return False
    
    async def _create_context_version(self, context_id: UUID, **kwargs) -> str:
        """Create a version of a context."""
        try:
            version_id = str(uuid4())
            
            async with get_async_session() as session:
                # Get original context
                context = await session.get(Context, context_id)
                if not context:
                    raise ValueError(f"Context {context_id} not found")
                
                # Create version metadata
                version_metadata = {
                    "version_id": version_id,
                    "original_context_id": str(context_id),
                    "created_at": datetime.utcnow().isoformat(),
                    "version_type": kwargs.get("version_type", "manual"),
                    "description": kwargs.get("description", "Context version")
                }
                
                # Update context with version info
                if not context.context_metadata:
                    context.context_metadata = {}
                
                if "versions" not in context.context_metadata:
                    context.context_metadata["versions"] = []
                
                context.context_metadata["versions"].append(version_metadata)
                await session.commit()
                
                return version_id
                
        except Exception as e:
            logger.error(f"Failed to create version for context {context_id}: {e}")
            raise
    
    async def _cleanup_context(self, context_id: UUID, **kwargs) -> int:
        """Cleanup context-related data."""
        try:
            cleaned_items = 0
            
            # Clear caches
            cache_keys = [key for key in self._context_cache.keys() if str(context_id) in key]
            for key in cache_keys:
                self._context_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
                cleaned_items += 1
            
            # Clear Redis caches
            pattern = f"*{context_id}*"
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis_client.delete(*keys)
                cleaned_items += len(keys)
            
            return cleaned_items
            
        except Exception as e:
            logger.error(f"Failed to cleanup context {context_id}: {e}")
            return 0


# Global instance for application use
_context_engine: Optional[ContextEngineIntegration] = None


async def get_context_engine_integration() -> ContextEngineIntegration:
    """
    Get singleton context engine integration instance.
    
    Returns:
        ContextEngineIntegration instance
    """
    global _context_engine
    
    if _context_engine is None:
        _context_engine = ContextEngineIntegration()
        await _context_engine.initialize()
    
    return _context_engine


async def cleanup_context_engine_integration() -> None:
    """Cleanup context engine integration resources."""
    global _context_engine
    
    if _context_engine:
        await _context_engine.shutdown()
        _context_engine = None