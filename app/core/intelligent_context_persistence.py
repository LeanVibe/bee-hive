"""
Intelligent Context Persistence System - Epic 4 Context Engine

Advanced context persistence with intelligent caching, compression, and lifecycle management
for optimal memory usage and retrieval performance in the LeanVibe Agent Hive 2.0.
"""

import asyncio
import time
import json
import hashlib
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import logging
from pathlib import Path

import structlog
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload
import redis.asyncio as redis

from .database import get_async_session
from .redis import get_redis_client
from .config import get_settings
from ..models.context import Context, ContextType
from ..models.agent import Agent
from .context_compression import get_context_compressor, CompressionLevel
from .context_cache_manager import get_context_cache_manager

logger = structlog.get_logger()


class PersistenceStrategy(Enum):
    """Context persistence strategies."""
    IMMEDIATE = "immediate"          # Persist immediately
    BATCH = "batch"                 # Batch persistence
    ADAPTIVE = "adaptive"           # Adaptive based on importance
    DELAYED = "delayed"             # Delayed persistence
    COMPRESSED_ONLY = "compressed_only"  # Only persist compressed


class LifecycleStage(Enum):
    """Context lifecycle stages."""
    ACTIVE = "active"               # Actively used
    WARM = "warm"                   # Recently used
    COLD = "cold"                   # Infrequently used
    FROZEN = "frozen"               # Long-term storage
    ARCHIVED = "archived"           # Historical archive
    EXPIRED = "expired"             # Ready for deletion


@dataclass
class PersistenceMetrics:
    """Metrics for persistence operations."""
    total_contexts: int = 0
    active_contexts: int = 0
    compressed_contexts: int = 0
    cache_hit_rate: float = 0.0
    average_retrieval_time_ms: float = 0.0
    storage_efficiency: float = 0.0
    lifecycle_transitions: Dict[str, int] = field(default_factory=dict)
    compression_savings_mb: float = 0.0


@dataclass
class StorageLocation:
    """Information about where context is stored."""
    primary_location: str           # database, redis, file_system, memory
    secondary_locations: List[str]  # backup locations
    compression_applied: bool
    encryption_applied: bool
    last_accessed: datetime
    access_frequency: int
    storage_size_bytes: int


@dataclass
class ContextLifecycleState:
    """State information for context lifecycle management."""
    context_id: str
    current_stage: LifecycleStage
    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance_score: float
    agent_id: str
    storage_locations: List[StorageLocation]
    compression_level: Optional[CompressionLevel]
    next_transition_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentContextPersistence:
    """
    Intelligent Context Persistence System with advanced lifecycle management.
    
    Features:
    - Adaptive persistence strategies based on usage patterns
    - Intelligent compression and caching
    - Lifecycle-based storage optimization
    - Multi-tier storage architecture
    - Performance-optimized retrieval
    - Automatic cleanup and archival
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.settings = get_settings()
        self.db_session = db_session
        self.redis_client = get_redis_client()
        self.logger = logger.bind(component="intelligent_context_persistence")
        
        # Get component dependencies
        self.compressor = get_context_compressor()
        self.cache_manager = get_context_cache_manager()
        
        # Persistence configuration
        self.default_strategy = PersistenceStrategy.ADAPTIVE
        self.compression_threshold_bytes = 1024  # 1KB
        self.batch_size = 100
        self.max_memory_contexts = 1000
        
        # Lifecycle thresholds (in days)
        self.lifecycle_thresholds = {
            LifecycleStage.WARM: 1,      # Active -> Warm after 1 day
            LifecycleStage.COLD: 7,      # Warm -> Cold after 7 days
            LifecycleStage.FROZEN: 30,   # Cold -> Frozen after 30 days
            LifecycleStage.ARCHIVED: 90, # Frozen -> Archived after 90 days
            LifecycleStage.EXPIRED: 365  # Archived -> Expired after 1 year
        }
        
        # In-memory state tracking
        self.lifecycle_states: Dict[str, ContextLifecycleState] = {}
        self.pending_persistence: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = PersistenceMetrics()
        self.operation_times = defaultdict(list)
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._is_running = False
        
        self.logger.info("💾 Intelligent Context Persistence initialized")
    
    async def initialize(self) -> None:
        """Initialize the persistence system."""
        if self._is_running:
            return
        
        try:
            self.logger.info("🚀 Initializing Intelligent Context Persistence...")
            
            # Load existing lifecycle states
            await self._load_lifecycle_states()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._is_running = True
            self.logger.info("✅ Intelligent Context Persistence initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize persistence system: {e}")
            raise
    
    async def store_context(
        self,
        context: Union[Context, Dict[str, Any]],
        strategy: Optional[PersistenceStrategy] = None,
        force_immediate: bool = False
    ) -> str:
        """
        Store context with intelligent persistence strategy.
        
        Args:
            context: Context to store
            strategy: Persistence strategy to use
            force_immediate: Force immediate persistence
            
        Returns:
            Context ID
        """
        start_time = time.time()
        
        try:
            # Normalize context
            context_data = await self._normalize_context_for_storage(context)
            context_id = context_data["id"]
            
            self.logger.debug(f"💾 Storing context {context_id}")
            
            # Update metrics
            self.metrics.total_contexts += 1
            
            # Determine persistence strategy
            if strategy is None:
                strategy = await self._determine_optimal_strategy(context_data)
            
            # Create or update lifecycle state
            lifecycle_state = await self._create_or_update_lifecycle_state(
                context_id, context_data
            )
            
            # Execute persistence strategy
            if force_immediate or strategy == PersistenceStrategy.IMMEDIATE:
                await self._persist_immediately(context_data, lifecycle_state)
            elif strategy == PersistenceStrategy.BATCH:
                await self._add_to_batch(context_data, lifecycle_state)
            elif strategy == PersistenceStrategy.COMPRESSED_ONLY:
                await self._persist_compressed_only(context_data, lifecycle_state)
            elif strategy == PersistenceStrategy.ADAPTIVE:
                await self._persist_adaptively(context_data, lifecycle_state)
            else:  # DELAYED
                await self._schedule_delayed_persistence(context_data, lifecycle_state)
            
            # Update access patterns
            self._update_access_patterns(context_id)
            
            # Cache in memory if appropriate
            if await self._should_cache_in_memory(context_data, lifecycle_state):
                await self._cache_in_memory(context_data)
            
            processing_time = time.time() - start_time
            self.operation_times["store"].append(processing_time)
            
            self.logger.debug(
                f"✅ Context {context_id} stored using {strategy.value} strategy "
                f"in {processing_time:.3f}s"
            )
            
            return context_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store context: {e}")
            raise
    
    async def retrieve_context(
        self,
        context_id: str,
        include_compressed: bool = True,
        update_access: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve context with intelligent caching and decompression.
        
        Args:
            context_id: Context ID to retrieve
            include_compressed: Whether to include compressed contexts
            update_access: Whether to update access patterns
            
        Returns:
            Context data if found
        """
        start_time = time.time()
        
        try:
            self.logger.debug(f"🔍 Retrieving context {context_id}")
            
            # Check memory cache first
            context_data = await self._retrieve_from_memory(context_id)
            if context_data:
                if update_access:
                    self._update_access_patterns(context_id)
                    await self._update_lifecycle_state_access(context_id)
                
                processing_time = time.time() - start_time
                self.operation_times["retrieve_memory"].append(processing_time)
                self.metrics.cache_hit_rate = self._calculate_cache_hit_rate()
                
                self.logger.debug(f"✅ Context {context_id} retrieved from memory cache")
                return context_data
            
            # Check Redis cache
            context_data = await self._retrieve_from_redis(context_id)
            if context_data:
                # Cache in memory for future access
                await self._cache_in_memory(context_data)
                
                if update_access:
                    self._update_access_patterns(context_id)
                    await self._update_lifecycle_state_access(context_id)
                
                processing_time = time.time() - start_time
                self.operation_times["retrieve_redis"].append(processing_time)
                
                self.logger.debug(f"✅ Context {context_id} retrieved from Redis cache")
                return context_data
            
            # Retrieve from database
            context_data = await self._retrieve_from_database(context_id)
            if context_data:
                # Decompress if needed
                if context_data.get("is_compressed") and include_compressed:
                    context_data = await self._decompress_context(context_data)
                
                # Cache in Redis and memory
                await self._cache_in_redis(context_data)
                if await self._should_cache_in_memory(context_data):
                    await self._cache_in_memory(context_data)
                
                if update_access:
                    self._update_access_patterns(context_id)
                    await self._update_lifecycle_state_access(context_id)
                
                processing_time = time.time() - start_time
                self.operation_times["retrieve_database"].append(processing_time)
                
                self.logger.debug(f"✅ Context {context_id} retrieved from database")
                return context_data
            
            # Context not found
            processing_time = time.time() - start_time
            self.operation_times["retrieve_miss"].append(processing_time)
            
            self.logger.debug(f"❌ Context {context_id} not found")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to retrieve context {context_id}: {e}")
            raise
    
    async def update_context_lifecycle(
        self,
        context_id: str,
        new_stage: Optional[LifecycleStage] = None,
        force_transition: bool = False
    ) -> bool:
        """
        Update context lifecycle stage.
        
        Args:
            context_id: Context ID to update
            new_stage: New lifecycle stage (auto-determined if None)
            force_transition: Force transition regardless of thresholds
            
        Returns:
            True if lifecycle was updated
        """
        try:
            lifecycle_state = self.lifecycle_states.get(context_id)
            if not lifecycle_state:
                self.logger.warning(f"No lifecycle state found for context {context_id}")
                return False
            
            # Determine new stage if not provided
            if new_stage is None:
                new_stage = await self._determine_next_lifecycle_stage(lifecycle_state)
            
            # Check if transition is valid
            if not force_transition and not await self._is_valid_transition(
                lifecycle_state.current_stage, new_stage, lifecycle_state
            ):
                return False
            
            old_stage = lifecycle_state.current_stage
            lifecycle_state.current_stage = new_stage
            lifecycle_state.next_transition_at = await self._calculate_next_transition_time(
                lifecycle_state
            )
            
            # Update metrics
            transition_key = f"{old_stage.value}_to_{new_stage.value}"
            self.metrics.lifecycle_transitions[transition_key] = \
                self.metrics.lifecycle_transitions.get(transition_key, 0) + 1
            
            # Apply stage-specific optimizations
            await self._apply_lifecycle_stage_optimizations(context_id, lifecycle_state)
            
            self.logger.info(
                f"📈 Context {context_id} transitioned from {old_stage.value} "
                f"to {new_stage.value}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update lifecycle for context {context_id}: {e}")
            return False
    
    async def optimize_storage(
        self,
        target_reduction_mb: Optional[float] = None,
        max_processing_time_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Optimize storage usage through intelligent compression and archival.
        
        Args:
            target_reduction_mb: Target storage reduction in MB
            max_processing_time_minutes: Maximum time to spend optimizing
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        max_time = max_processing_time_minutes * 60
        
        try:
            self.logger.info(f"⚡ Starting storage optimization")
            
            optimization_results = {
                "contexts_processed": 0,
                "contexts_compressed": 0,
                "contexts_archived": 0,
                "contexts_deleted": 0,
                "storage_saved_mb": 0.0,
                "processing_time_seconds": 0.0,
                "optimizations_applied": []
            }
            
            # Get contexts eligible for optimization
            optimization_candidates = await self._identify_optimization_candidates()
            
            for context_id, candidate_info in optimization_candidates.items():
                # Check time limit
                if time.time() - start_time > max_time:
                    self.logger.info("⏰ Storage optimization time limit reached")
                    break
                
                # Check target reduction
                if (target_reduction_mb and 
                    optimization_results["storage_saved_mb"] >= target_reduction_mb):
                    self.logger.info("🎯 Storage optimization target reached")
                    break
                
                # Apply optimization
                result = await self._apply_context_optimization(context_id, candidate_info)
                
                # Update results
                optimization_results["contexts_processed"] += 1
                if result["compressed"]:
                    optimization_results["contexts_compressed"] += 1
                if result["archived"]:
                    optimization_results["contexts_archived"] += 1
                if result["deleted"]:
                    optimization_results["contexts_deleted"] += 1
                
                optimization_results["storage_saved_mb"] += result["storage_saved_mb"]
                optimization_results["optimizations_applied"].extend(result["optimizations"])
            
            optimization_results["processing_time_seconds"] = time.time() - start_time
            
            # Update global metrics
            self.metrics.compression_savings_mb += optimization_results["storage_saved_mb"]
            self.metrics.storage_efficiency = await self._calculate_storage_efficiency()
            
            self.logger.info(
                f"✅ Storage optimization complete: {optimization_results['contexts_processed']} "
                f"contexts processed, {optimization_results['storage_saved_mb']:.2f}MB saved"
            )
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Storage optimization failed: {e}")
            raise
    
    async def get_persistence_analytics(self) -> Dict[str, Any]:
        """Get comprehensive persistence analytics."""
        try:
            # Calculate current metrics
            self.metrics.cache_hit_rate = self._calculate_cache_hit_rate()
            self.metrics.average_retrieval_time_ms = self._calculate_average_retrieval_time()
            self.metrics.storage_efficiency = await self._calculate_storage_efficiency()
            
            # Lifecycle distribution
            stage_distribution = defaultdict(int)
            for state in self.lifecycle_states.values():
                stage_distribution[state.current_stage.value] += 1
            
            # Access pattern analysis
            access_analysis = await self._analyze_access_patterns()
            
            # Performance analysis
            performance_analysis = self._analyze_performance_metrics()
            
            analytics = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "total_contexts": self.metrics.total_contexts,
                    "active_contexts": self.metrics.active_contexts,
                    "compressed_contexts": self.metrics.compressed_contexts,
                    "cache_hit_rate": self.metrics.cache_hit_rate,
                    "average_retrieval_time_ms": self.metrics.average_retrieval_time_ms,
                    "storage_efficiency": self.metrics.storage_efficiency,
                    "compression_savings_mb": self.metrics.compression_savings_mb
                },
                "lifecycle_distribution": dict(stage_distribution),
                "lifecycle_transitions": dict(self.metrics.lifecycle_transitions),
                "access_patterns": access_analysis,
                "performance_metrics": performance_analysis,
                "recommendations": await self._generate_optimization_recommendations()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get persistence analytics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the persistence system."""
        if not self._is_running:
            return
        
        self.logger.info("🔄 Shutting down Intelligent Context Persistence...")
        
        self._is_running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Process pending persistence operations
        if self.pending_persistence:
            await self._process_pending_persistence()
        
        # Save lifecycle states
        await self._save_lifecycle_states()
        
        self.logger.info("✅ Intelligent Context Persistence shutdown complete")
    
    # Private helper methods
    
    async def _determine_optimal_strategy(
        self, context_data: Dict[str, Any]
    ) -> PersistenceStrategy:
        """Determine optimal persistence strategy for context."""
        importance = context_data.get("importance_score", 0.5)
        size_bytes = len(json.dumps(context_data).encode('utf-8'))
        context_type = context_data.get("context_type", "general")
        
        # High importance -> immediate persistence
        if importance >= 0.8:
            return PersistenceStrategy.IMMEDIATE
        
        # Large contexts -> compressed only
        if size_bytes > self.compression_threshold_bytes * 10:
            return PersistenceStrategy.COMPRESSED_ONLY
        
        # Critical context types -> immediate
        if context_type in ["error_resolution", "decision", "security"]:
            return PersistenceStrategy.IMMEDIATE
        
        # Default to adaptive
        return PersistenceStrategy.ADAPTIVE
    
    async def _normalize_context_for_storage(
        self, context: Union[Context, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize context for storage."""
        if isinstance(context, Context):
            return {
                "id": str(context.id),
                "content": context.content,
                "context_type": context.context_type.value if context.context_type else "general",
                "agent_id": str(context.agent_id),
                "importance_score": context.importance_score,
                "created_at": context.created_at.isoformat(),
                "metadata": getattr(context, 'metadata', {})
            }
        else:
            return context
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Lifecycle management task
        task = asyncio.create_task(self._background_lifecycle_management())
        self._background_tasks.append(task)
        
        # Batch persistence task
        task = asyncio.create_task(self._background_batch_persistence())
        self._background_tasks.append(task)
        
        # Storage optimization task
        task = asyncio.create_task(self._background_storage_optimization())
        self._background_tasks.append(task)
        
        # Metrics collection task
        task = asyncio.create_task(self._background_metrics_collection())
        self._background_tasks.append(task)
        
        self.logger.info("🔄 Background persistence tasks started")
    
    async def _background_lifecycle_management(self) -> None:
        """Background task for lifecycle management."""
        while self._is_running:
            try:
                await self._process_lifecycle_transitions()
                await asyncio.sleep(3600)  # Check every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background lifecycle management error: {e}")
                await asyncio.sleep(300)  # 5 minutes before retry
    
    async def _background_batch_persistence(self) -> None:
        """Background task for batch persistence."""
        while self._is_running:
            try:
                if self.pending_persistence:
                    await self._process_pending_persistence()
                await asyncio.sleep(60)  # Process every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background batch persistence error: {e}")
                await asyncio.sleep(30)  # 30 seconds before retry
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate."""
        memory_hits = len(self.operation_times.get("retrieve_memory", []))
        redis_hits = len(self.operation_times.get("retrieve_redis", []))
        db_hits = len(self.operation_times.get("retrieve_database", []))
        misses = len(self.operation_times.get("retrieve_miss", []))
        
        total_requests = memory_hits + redis_hits + db_hits + misses
        if total_requests == 0:
            return 0.0
        
        cache_hits = memory_hits + redis_hits
        return cache_hits / total_requests
    
    def _update_access_patterns(self, context_id: str) -> None:
        """Update access patterns for a context."""
        now = datetime.utcnow()
        self.access_patterns[context_id].append(now)
        
        # Keep only recent access patterns (last 100 accesses)
        if len(self.access_patterns[context_id]) > 100:
            self.access_patterns[context_id] = self.access_patterns[context_id][-100:]
    
    # Placeholder implementations for complex methods
    
    async def _load_lifecycle_states(self) -> None:
        """Load existing lifecycle states from storage."""
        pass  # Implementation would load from Redis/Database
    
    async def _save_lifecycle_states(self) -> None:
        """Save lifecycle states to storage."""
        pass  # Implementation would save to Redis/Database
    
    async def _persist_immediately(self, context_data: Dict, lifecycle_state: ContextLifecycleState) -> None:
        """Persist context immediately to database."""
        pass  # Implementation would write to database immediately
    
    async def _retrieve_from_memory(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context from memory cache."""
        return None  # Placeholder implementation
    
    async def _retrieve_from_redis(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context from Redis cache."""
        return None  # Placeholder implementation
    
    async def _retrieve_from_database(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context from database."""
        return None  # Placeholder implementation


# Global persistence instance
_persistence_system: Optional[IntelligentContextPersistence] = None


async def get_intelligent_context_persistence(
    db_session: Optional[AsyncSession] = None
) -> IntelligentContextPersistence:
    """
    Get or create the global intelligent context persistence instance.
    
    Args:
        db_session: Optional database session
        
    Returns:
        IntelligentContextPersistence instance
    """
    global _persistence_system
    
    if _persistence_system is None:
        _persistence_system = IntelligentContextPersistence(db_session)
        await _persistence_system.initialize()
    
    return _persistence_system


async def shutdown_intelligent_context_persistence():
    """Shutdown the global intelligent context persistence system."""
    global _persistence_system
    
    if _persistence_system:
        await _persistence_system.shutdown()
        _persistence_system = None