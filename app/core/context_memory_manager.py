"""
Context Memory Manager - Advanced Memory Management and Cleanup System.

Provides intelligent memory management for context storage with:
- Automated cleanup policies based on usage patterns
- Memory pressure detection and response
- Intelligent archiving and restoration
- Performance-optimized garbage collection
- Memory usage monitoring and optimization
- Context lifecycle management
"""

import asyncio
import logging
import json
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, AsyncIterator
from uuid import UUID
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from sqlalchemy import select, and_, or_, func, delete, update, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class CleanupPolicy(Enum):
    """Cleanup policies for context memory management."""
    CONSERVATIVE = "conservative"  # Keep most contexts, minimal cleanup
    BALANCED = "balanced"         # Balanced cleanup based on usage
    AGGRESSIVE = "aggressive"     # Aggressive cleanup for memory pressure
    EMERGENCY = "emergency"       # Emergency cleanup for critical situations


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CleanupPolicyConfig:
    """Configuration for cleanup policies."""
    policy: CleanupPolicy
    max_age_days: int
    min_importance_threshold: float
    max_contexts_per_agent: int
    memory_threshold_mb: float
    access_count_threshold: int
    consolidation_preference: bool
    preserve_critical_types: Set[ContextType]
    
    @classmethod
    def conservative(cls) -> "CleanupPolicyConfig":
        """Conservative cleanup policy."""
        return cls(
            policy=CleanupPolicy.CONSERVATIVE,
            max_age_days=180,
            min_importance_threshold=0.1,
            max_contexts_per_agent=1000,
            memory_threshold_mb=2000,
            access_count_threshold=1,
            consolidation_preference=True,
            preserve_critical_types={
                ContextType.DECISION,
                ContextType.ARCHITECTURE,
                ContextType.ERROR_RESOLUTION
            }
        )
    
    @classmethod
    def balanced(cls) -> "CleanupPolicyConfig":
        """Balanced cleanup policy."""
        return cls(
            policy=CleanupPolicy.BALANCED,
            max_age_days=90,
            min_importance_threshold=0.3,
            max_contexts_per_agent=500,
            memory_threshold_mb=1000,
            access_count_threshold=2,
            consolidation_preference=True,
            preserve_critical_types={
                ContextType.DECISION,
                ContextType.ARCHITECTURE
            }
        )
    
    @classmethod
    def aggressive(cls) -> "CleanupPolicyConfig":
        """Aggressive cleanup policy."""
        return cls(
            policy=CleanupPolicy.AGGRESSIVE,
            max_age_days=30,
            min_importance_threshold=0.5,
            max_contexts_per_agent=200,
            memory_threshold_mb=500,
            access_count_threshold=5,
            consolidation_preference=False,  # Direct cleanup
            preserve_critical_types={ContextType.DECISION}
        )
    
    @classmethod
    def emergency(cls) -> "CleanupPolicyConfig":
        """Emergency cleanup policy."""
        return cls(
            policy=CleanupPolicy.EMERGENCY,
            max_age_days=7,
            min_importance_threshold=0.8,
            max_contexts_per_agent=50,
            memory_threshold_mb=100,
            access_count_threshold=10,
            consolidation_preference=False,
            preserve_critical_types=set()  # Minimal preservation
        )


@dataclass
class MemoryUsageSnapshot:
    """Snapshot of memory usage metrics."""
    timestamp: datetime
    total_memory_mb: float
    available_memory_mb: float
    context_count: int
    consolidated_context_count: int
    avg_context_size_kb: float
    memory_pressure_level: MemoryPressureLevel
    gc_collections: int
    cache_size_mb: float


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    policy_used: CleanupPolicy
    contexts_archived: int
    contexts_deleted: int
    contexts_consolidated: int
    memory_freed_mb: float
    processing_time_ms: float
    errors_encountered: int
    agents_processed: int


class ContextMemoryManager:
    """
    Advanced memory management system for context storage.
    
    Features:
    - Intelligent cleanup policies based on memory pressure
    - Automated archiving and restoration
    - Memory usage monitoring and optimization
    - Performance-optimized garbage collection
    - Context lifecycle management with preservation rules
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        
        # Configuration
        self.cleanup_policies = {
            CleanupPolicy.CONSERVATIVE: CleanupPolicyConfig.conservative(),
            CleanupPolicy.BALANCED: CleanupPolicyConfig.balanced(),
            CleanupPolicy.AGGRESSIVE: CleanupPolicyConfig.aggressive(),
            CleanupPolicy.EMERGENCY: CleanupPolicyConfig.emergency()
        }
        
        self.current_policy = self.cleanup_policies[CleanupPolicy.BALANCED]
        
        # Memory monitoring
        self.memory_snapshots: deque = deque(maxlen=1000)
        self.cleanup_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.cleanup_performance: Dict[CleanupPolicy, List[float]] = defaultdict(list)
        self.memory_optimization_stats = {
            "total_cleanups": 0,
            "total_memory_freed_mb": 0.0,
            "total_contexts_processed": 0,
            "avg_cleanup_time_ms": 0.0
        }
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Memory pressure thresholds (MB)
        self.memory_thresholds = {
            MemoryPressureLevel.LOW: 500,
            MemoryPressureLevel.MEDIUM: 1000,
            MemoryPressureLevel.HIGH: 2000,
            MemoryPressureLevel.CRITICAL: 4000
        }
    
    async def start_memory_management(self) -> None:
        """Start the memory management system."""
        if self._is_running:
            return
        
        logger.info("Starting context memory management system")
        self._is_running = True
        
        # Start monitoring and cleanup tasks
        self._monitor_task = asyncio.create_task(self._memory_monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_memory_management(self) -> None:
        """Stop the memory management system."""
        if not self._is_running:
            return
        
        logger.info("Stopping context memory management system")
        self._is_running = False
        
        # Cancel background tasks
        for task in [self._monitor_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def perform_memory_cleanup(
        self,
        policy: Optional[CleanupPolicy] = None,
        agent_id: Optional[UUID] = None,
        force_cleanup: bool = False
    ) -> CleanupResult:
        """
        Perform comprehensive memory cleanup.
        
        Args:
            policy: Cleanup policy to use (auto-detected if None)
            agent_id: Specific agent to clean up (all agents if None)
            force_cleanup: Force cleanup regardless of conditions
            
        Returns:
            CleanupResult with details of cleanup performed
        """
        start_time = datetime.utcnow()
        
        try:
            # Determine cleanup policy
            if policy is None:
                memory_pressure = await self._assess_memory_pressure()
                policy = self._select_cleanup_policy(memory_pressure, force_cleanup)
            
            config = self.cleanup_policies[policy]
            
            logger.info(f"Starting memory cleanup with {policy.value} policy")
            
            # Take memory snapshot before cleanup
            before_snapshot = await self._take_memory_snapshot()
            
            result = CleanupResult(
                policy_used=policy,
                contexts_archived=0,
                contexts_deleted=0,
                contexts_consolidated=0,
                memory_freed_mb=0.0,
                processing_time_ms=0.0,
                errors_encountered=0,
                agents_processed=0
            )
            
            # Get agents to process
            if agent_id:
                agents_to_process = [agent_id]
            else:
                async with get_async_session() as session:
                    agents_result = await session.execute(select(Agent.id))
                    agents_to_process = [row[0] for row in agents_result.all()]
            
            # Process each agent
            for aid in agents_to_process:
                try:
                    agent_result = await self._cleanup_agent_contexts(aid, config)
                    
                    result.contexts_archived += agent_result["archived"]
                    result.contexts_deleted += agent_result["deleted"]
                    result.contexts_consolidated += agent_result["consolidated"]
                    result.agents_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error cleaning up agent {aid}: {e}")
                    result.errors_encountered += 1
            
            # Perform system-level cleanup
            await self._perform_system_cleanup(config)
            
            # Take memory snapshot after cleanup
            after_snapshot = await self._take_memory_snapshot()
            
            # Calculate memory freed
            result.memory_freed_mb = max(0, 
                before_snapshot.total_memory_mb - after_snapshot.total_memory_mb
            )
            
            # Calculate processing time
            processing_time = datetime.utcnow() - start_time
            result.processing_time_ms = processing_time.total_seconds() * 1000
            
            # Update performance tracking
            self.cleanup_performance[policy].append(result.processing_time_ms)
            self.cleanup_history.append(result)
            
            # Update statistics
            self.memory_optimization_stats["total_cleanups"] += 1
            self.memory_optimization_stats["total_memory_freed_mb"] += result.memory_freed_mb
            self.memory_optimization_stats["total_contexts_processed"] += (
                result.contexts_archived + result.contexts_deleted + result.contexts_consolidated
            )
            
            # Update average cleanup time
            total_time = (
                self.memory_optimization_stats["avg_cleanup_time_ms"] * 
                (self.memory_optimization_stats["total_cleanups"] - 1) + 
                result.processing_time_ms
            )
            self.memory_optimization_stats["avg_cleanup_time_ms"] = (
                total_time / self.memory_optimization_stats["total_cleanups"]
            )
            
            logger.info(
                f"Memory cleanup completed: {result.contexts_archived} archived, "
                f"{result.contexts_deleted} deleted, {result.memory_freed_mb:.1f}MB freed "
                f"in {result.processing_time_ms:.0f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            result.errors_encountered += 1
            processing_time = datetime.utcnow() - start_time
            result.processing_time_ms = processing_time.total_seconds() * 1000
            return result
    
    async def optimize_memory_usage(
        self,
        target_reduction_mb: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize memory usage with intelligent strategies.
        
        Args:
            target_reduction_mb: Target memory reduction (auto-calculated if None)
            
        Returns:
            Optimization results
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info("Starting memory usage optimization")
            
            # Assess current memory situation
            memory_snapshot = await self._take_memory_snapshot()
            pressure_level = memory_snapshot.memory_pressure_level
            
            optimization_result = {
                "initial_memory_mb": memory_snapshot.total_memory_mb,
                "initial_context_count": memory_snapshot.context_count,
                "pressure_level": pressure_level.value,
                "optimizations_applied": [],
                "memory_freed_mb": 0.0,
                "contexts_optimized": 0,
                "processing_time_ms": 0.0
            }
            
            # Strategy 1: Garbage collection
            gc_freed = await self._perform_garbage_collection()
            optimization_result["optimizations_applied"].append("garbage_collection")
            optimization_result["memory_freed_mb"] += gc_freed
            
            # Strategy 2: Cache optimization
            cache_freed = await self._optimize_caches()
            optimization_result["optimizations_applied"].append("cache_optimization")
            optimization_result["memory_freed_mb"] += cache_freed
            
            # Strategy 3: Context consolidation
            if pressure_level in [MemoryPressureLevel.MEDIUM, MemoryPressureLevel.HIGH]:
                consolidation_result = await self._perform_emergency_consolidation()
                optimization_result["optimizations_applied"].append("emergency_consolidation")
                optimization_result["contexts_optimized"] += consolidation_result["contexts_processed"]
                optimization_result["memory_freed_mb"] += consolidation_result["memory_freed_mb"]
            
            # Strategy 4: Aggressive cleanup if still under pressure
            if pressure_level == MemoryPressureLevel.CRITICAL:
                cleanup_result = await self.perform_memory_cleanup(
                    policy=CleanupPolicy.EMERGENCY,
                    force_cleanup=True
                )
                optimization_result["optimizations_applied"].append("emergency_cleanup")
                optimization_result["contexts_optimized"] += (
                    cleanup_result.contexts_archived + cleanup_result.contexts_deleted
                )
                optimization_result["memory_freed_mb"] += cleanup_result.memory_freed_mb
            
            # Final memory snapshot
            final_snapshot = await self._take_memory_snapshot()
            optimization_result["final_memory_mb"] = final_snapshot.total_memory_mb
            optimization_result["final_context_count"] = final_snapshot.context_count
            
            # Calculate total processing time
            processing_time = datetime.utcnow() - start_time
            optimization_result["processing_time_ms"] = processing_time.total_seconds() * 1000
            
            logger.info(
                f"Memory optimization completed: {optimization_result['memory_freed_mb']:.1f}MB freed, "
                f"{optimization_result['contexts_optimized']} contexts optimized "
                f"in {optimization_result['processing_time_ms']:.0f}ms"
            )
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            processing_time = datetime.utcnow() - start_time
            return {
                "error": str(e),
                "processing_time_ms": processing_time.total_seconds() * 1000
            }
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        try:
            current_snapshot = await self._take_memory_snapshot()
            
            stats = {
                "current_memory": {
                    "total_memory_mb": current_snapshot.total_memory_mb,
                    "available_memory_mb": current_snapshot.available_memory_mb,
                    "context_count": current_snapshot.context_count,
                    "consolidated_context_count": current_snapshot.consolidated_context_count,
                    "avg_context_size_kb": current_snapshot.avg_context_size_kb,
                    "pressure_level": current_snapshot.memory_pressure_level.value,
                    "cache_size_mb": current_snapshot.cache_size_mb
                },
                "cleanup_statistics": self.memory_optimization_stats.copy(),
                "policy_performance": {},
                "memory_trends": self._analyze_memory_trends(),
                "recommendations": await self._generate_memory_recommendations()
            }
            
            # Policy performance statistics
            for policy, times in self.cleanup_performance.items():
                if times:
                    stats["policy_performance"][policy.value] = {
                        "avg_time_ms": sum(times) / len(times),
                        "min_time_ms": min(times),
                        "max_time_ms": max(times),
                        "usage_count": len(times)
                    }
            
            # Recent cleanup history
            recent_cleanups = list(self.cleanup_history)[-10:]  # Last 10 cleanups
            stats["recent_cleanups"] = [
                {
                    "policy": cleanup.policy_used.value,
                    "contexts_processed": (
                        cleanup.contexts_archived + cleanup.contexts_deleted + cleanup.contexts_consolidated
                    ),
                    "memory_freed_mb": cleanup.memory_freed_mb,
                    "processing_time_ms": cleanup.processing_time_ms,
                    "agents_processed": cleanup.agents_processed
                }
                for cleanup in recent_cleanups
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {"error": str(e)}
    
    # Private Methods
    
    async def _memory_monitoring_loop(self) -> None:
        """Background loop for memory monitoring."""
        logger.info("Starting memory monitoring loop")
        
        try:
            while self._is_running:
                try:
                    # Take memory snapshot
                    snapshot = await self._take_memory_snapshot()
                    self.memory_snapshots.append(snapshot)
                    
                    # Check for memory pressure
                    if snapshot.memory_pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                        logger.warning(f"Memory pressure detected: {snapshot.memory_pressure_level.value}")
                        
                        # Trigger immediate optimization for critical pressure
                        if snapshot.memory_pressure_level == MemoryPressureLevel.CRITICAL:
                            asyncio.create_task(self.optimize_memory_usage())
                    
                    # Store snapshot in Redis for analytics
                    await self.redis_client.lpush(
                        "memory_snapshots",
                        json.dumps(asdict(snapshot), default=str)
                    )
                    
                    # Keep only recent snapshots in Redis
                    await self.redis_client.ltrim("memory_snapshots", 0, 999)
                    
                    # Wait before next snapshot
                    await asyncio.sleep(60)  # Monitor every minute
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("Memory monitoring loop stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background loop for scheduled cleanup operations."""
        logger.info("Starting cleanup loop")
        
        try:
            while self._is_running:
                try:
                    # Perform scheduled cleanup based on current policy
                    await self.perform_memory_cleanup()
                    
                    # Wait for next cleanup cycle (configurable interval)
                    await asyncio.sleep(3600)  # Cleanup every hour
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
                    
        except asyncio.CancelledError:
            logger.info("Cleanup loop stopped")
    
    async def _take_memory_snapshot(self) -> MemoryUsageSnapshot:
        """Take a snapshot of current memory usage."""
        try:
            # System memory information
            memory_info = psutil.virtual_memory()
            total_memory_mb = memory_info.total / (1024 * 1024)
            available_memory_mb = memory_info.available / (1024 * 1024)
            
            # Context count information
            async with get_async_session() as session:
                context_count = await session.scalar(select(func.count(Context.id))) or 0
                consolidated_count = await session.scalar(
                    select(func.count(Context.id)).where(Context.is_consolidated == "true")
                ) or 0
            
            # Calculate average context size (rough estimate)
            avg_context_size_kb = 2.0 if context_count == 0 else (
                (total_memory_mb * 1024 * 0.1) / context_count  # Assume 10% of memory for contexts
            )
            
            # Assess memory pressure
            used_memory_mb = total_memory_mb - available_memory_mb
            pressure_level = self._calculate_memory_pressure(used_memory_mb)
            
            # GC information
            gc_stats = gc.get_stats()
            gc_collections = sum(stat.get('collections', 0) for stat in gc_stats)
            
            # Estimate cache size (simplified)
            cache_size_mb = 50.0  # Rough estimate
            
            return MemoryUsageSnapshot(
                timestamp=datetime.utcnow(),
                total_memory_mb=total_memory_mb,
                available_memory_mb=available_memory_mb,
                context_count=context_count,
                consolidated_context_count=consolidated_count,
                avg_context_size_kb=avg_context_size_kb,
                memory_pressure_level=pressure_level,
                gc_collections=gc_collections,
                cache_size_mb=cache_size_mb
            )
            
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {e}")
            return MemoryUsageSnapshot(
                timestamp=datetime.utcnow(),
                total_memory_mb=0.0,
                available_memory_mb=0.0,
                context_count=0,
                consolidated_context_count=0,
                avg_context_size_kb=0.0,
                memory_pressure_level=MemoryPressureLevel.LOW,
                gc_collections=0,
                cache_size_mb=0.0
            )
    
    def _calculate_memory_pressure(self, used_memory_mb: float) -> MemoryPressureLevel:
        """Calculate memory pressure level based on usage."""
        if used_memory_mb >= self.memory_thresholds[MemoryPressureLevel.CRITICAL]:
            return MemoryPressureLevel.CRITICAL
        elif used_memory_mb >= self.memory_thresholds[MemoryPressureLevel.HIGH]:
            return MemoryPressureLevel.HIGH
        elif used_memory_mb >= self.memory_thresholds[MemoryPressureLevel.MEDIUM]:
            return MemoryPressureLevel.MEDIUM
        else:
            return MemoryPressureLevel.LOW
    
    async def _assess_memory_pressure(self) -> MemoryPressureLevel:
        """Assess current memory pressure level."""
        snapshot = await self._take_memory_snapshot()
        return snapshot.memory_pressure_level
    
    def _select_cleanup_policy(
        self,
        memory_pressure: MemoryPressureLevel,
        force_cleanup: bool
    ) -> CleanupPolicy:
        """Select appropriate cleanup policy based on memory pressure."""
        if force_cleanup or memory_pressure == MemoryPressureLevel.CRITICAL:
            return CleanupPolicy.EMERGENCY
        elif memory_pressure == MemoryPressureLevel.HIGH:
            return CleanupPolicy.AGGRESSIVE
        elif memory_pressure == MemoryPressureLevel.MEDIUM:
            return CleanupPolicy.BALANCED
        else:
            return CleanupPolicy.CONSERVATIVE
    
    async def _cleanup_agent_contexts(
        self,
        agent_id: UUID,
        config: CleanupPolicyConfig
    ) -> Dict[str, int]:
        """Clean up contexts for a specific agent."""
        result = {"archived": 0, "deleted": 0, "consolidated": 0}
        
        try:
            async with get_async_session() as session:
                # Get contexts to potentially clean up
                cutoff_date = datetime.utcnow() - timedelta(days=config.max_age_days)
                
                query = select(Context).where(
                    and_(
                        Context.agent_id == agent_id,
                        or_(
                            Context.created_at < cutoff_date,
                            Context.importance_score < config.min_importance_threshold,
                            func.cast(Context.access_count, session.Integer) < config.access_count_threshold
                        )
                    )
                ).order_by(
                    Context.importance_score.asc(),
                    Context.accessed_at.asc()
                ).limit(100)  # Process in batches
                
                contexts_result = await session.execute(query)
                contexts = list(contexts_result.scalars().all())
                
                for context in contexts:
                    try:
                        # Check if context should be preserved
                        if await self._should_preserve_context(context, config):
                            continue
                        
                        # Determine cleanup action
                        if config.consolidation_preference and not context.is_consolidated:
                            # Mark for consolidation instead of deletion
                            context.update_metadata("pending_consolidation", True)
                            result["consolidated"] += 1
                        else:
                            # Archive the context
                            context.update_metadata("archived", True)
                            context.update_metadata("archived_at", datetime.utcnow().isoformat())
                            context.update_metadata("archived_reason", f"cleanup_{config.policy.value}")
                            result["archived"] += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing context {context.id}: {e}")
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up contexts for agent {agent_id}: {e}")
        
        return result
    
    async def _should_preserve_context(
        self,
        context: Context,
        config: CleanupPolicyConfig
    ) -> bool:
        """Determine if a context should be preserved from cleanup."""
        # Always preserve critical context types
        if context.context_type in config.preserve_critical_types:
            return True
        
        # Preserve high importance contexts
        if context.importance_score >= 0.9:
            return True
        
        # Preserve recently accessed contexts
        if context.accessed_at and context.accessed_at > datetime.utcnow() - timedelta(days=7):
            return True
        
        # Preserve contexts with high access count
        access_count = int(context.access_count or 0)
        if access_count >= 50:
            return True
        
        return False
    
    async def _perform_system_cleanup(self, config: CleanupPolicyConfig) -> None:
        """Perform system-level cleanup operations."""
        try:
            # Clean up orphaned contexts
            async with get_async_session() as session:
                # Delete contexts with null agent_id that are old
                cutoff_date = datetime.utcnow() - timedelta(days=config.max_age_days // 2)
                
                await session.execute(
                    delete(Context).where(
                        and_(
                            Context.agent_id.is_(None),
                            Context.created_at < cutoff_date,
                            Context.importance_score < config.min_importance_threshold
                        )
                    )
                )
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error in system cleanup: {e}")
    
    async def _perform_garbage_collection(self) -> float:
        """Perform garbage collection and estimate memory freed."""
        try:
            # Get memory before GC
            memory_before = psutil.virtual_memory().used / (1024 * 1024)
            
            # Force garbage collection
            collected = gc.collect()
            
            # Get memory after GC
            memory_after = psutil.virtual_memory().used / (1024 * 1024)
            
            memory_freed = max(0, memory_before - memory_after)
            
            logger.debug(f"Garbage collection freed {memory_freed:.1f}MB ({collected} objects)")
            return memory_freed
            
        except Exception as e:
            logger.error(f"Error in garbage collection: {e}")
            return 0.0
    
    async def _optimize_caches(self) -> float:
        """Optimize various caches and estimate memory freed."""
        try:
            memory_freed = 0.0
            
            # Clear expired Redis cache entries
            try:
                # This is a simplified implementation
                # In practice, you'd implement more sophisticated cache cleaning
                expired_keys = []
                async for key in self.redis_client.scan_iter(match="*_cache:*"):
                    ttl = await self.redis_client.ttl(key)
                    if ttl == -1:  # No expiration set
                        expired_keys.append(key)
                
                if expired_keys:
                    await self.redis_client.delete(*expired_keys[:1000])  # Limit batch size
                    memory_freed += len(expired_keys) * 0.001  # Rough estimate
                    
            except Exception as e:
                logger.warning(f"Error optimizing Redis cache: {e}")
            
            return memory_freed
            
        except Exception as e:
            logger.error(f"Error optimizing caches: {e}")
            return 0.0
    
    async def _perform_emergency_consolidation(self) -> Dict[str, Any]:
        """Perform emergency context consolidation."""
        try:
            consolidation_result = {
                "contexts_processed": 0,
                "memory_freed_mb": 0.0
            }
            
            # Find agents with high unconsolidated context counts
            async with get_async_session() as session:
                high_context_agents = await session.execute(
                    select(Context.agent_id, func.count(Context.id).label('count'))
                    .where(Context.is_consolidated == "false")
                    .group_by(Context.agent_id)
                    .having(func.count(Context.id) > 20)
                    .order_by(desc('count'))
                    .limit(10)
                )
                
                for agent_id, context_count in high_context_agents:
                    try:
                        # This would integrate with the consolidation service
                        # For now, we'll just mark contexts for consolidation
                        await session.execute(
                            update(Context)
                            .where(
                                and_(
                                    Context.agent_id == agent_id,
                                    Context.is_consolidated == "false"
                                )
                            )
                            .values(context_metadata=func.jsonb_set(
                                Context.context_metadata,
                                '{emergency_consolidation_pending}',
                                'true'
                            ))
                            .execution_options(synchronize_session=False)
                        )
                        
                        consolidation_result["contexts_processed"] += context_count
                        
                    except Exception as e:
                        logger.error(f"Error in emergency consolidation for agent {agent_id}: {e}")
                
                await session.commit()
            
            # Estimate memory freed (rough calculation)
            consolidation_result["memory_freed_mb"] = consolidation_result["contexts_processed"] * 0.5
            
            return consolidation_result
            
        except Exception as e:
            logger.error(f"Error in emergency consolidation: {e}")
            return {"contexts_processed": 0, "memory_freed_mb": 0.0}
    
    def _analyze_memory_trends(self) -> Dict[str, Any]:
        """Analyze memory usage trends from historical snapshots."""
        try:
            if len(self.memory_snapshots) < 2:
                return {"insufficient_data": True}
            
            recent_snapshots = list(self.memory_snapshots)[-60:]  # Last hour of snapshots
            
            # Calculate trends
            memory_values = [s.total_memory_mb for s in recent_snapshots]
            context_counts = [s.context_count for s in recent_snapshots]
            
            memory_trend = "stable"
            if len(memory_values) >= 10:
                if memory_values[-5:] > memory_values[:5]:
                    memory_trend = "increasing"
                elif memory_values[-5:] < memory_values[:5]:
                    memory_trend = "decreasing"
            
            context_trend = "stable"
            if len(context_counts) >= 10:
                if context_counts[-5:] > context_counts[:5]:
                    context_trend = "increasing"
                elif context_counts[-5:] < context_counts[:5]:
                    context_trend = "decreasing"
            
            return {
                "memory_trend": memory_trend,
                "context_trend": context_trend,
                "avg_memory_mb": sum(memory_values) / len(memory_values),
                "avg_context_count": sum(context_counts) / len(context_counts),
                "snapshots_analyzed": len(recent_snapshots)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing memory trends: {e}")
            return {"error": str(e)}
    
    async def _generate_memory_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations."""
        try:
            recommendations = []
            
            # Analyze current state
            current_snapshot = await self._take_memory_snapshot()
            
            # High context count recommendation
            if current_snapshot.context_count > 10000:
                recommendations.append(
                    f"High context count detected ({current_snapshot.context_count}). "
                    "Consider enabling aggressive cleanup policy."
                )
            
            # Low consolidation rate recommendation
            if current_snapshot.context_count > 0:
                consolidation_rate = (
                    current_snapshot.consolidated_context_count / current_snapshot.context_count
                )
                if consolidation_rate < 0.3:
                    recommendations.append(
                        f"Low consolidation rate ({consolidation_rate:.1%}). "
                        "Consider enabling more frequent consolidation."
                    )
            
            # Memory pressure recommendation
            if current_snapshot.memory_pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                recommendations.append(
                    f"Memory pressure detected ({current_snapshot.memory_pressure_level.value}). "
                    "Consider immediate cleanup or increasing memory allocation."
                )
            
            # Performance recommendations based on cleanup history
            if self.cleanup_history:
                avg_cleanup_time = sum(c.processing_time_ms for c in self.cleanup_history) / len(self.cleanup_history)
                if avg_cleanup_time > 30000:  # 30 seconds
                    recommendations.append(
                        f"Cleanup operations are slow (avg: {avg_cleanup_time:.0f}ms). "
                        "Consider smaller batch sizes or more frequent cleanups."
                    )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [f"Error generating recommendations: {e}"]


# Global instance for application use
_memory_manager: Optional[ContextMemoryManager] = None


def get_context_memory_manager() -> ContextMemoryManager:
    """
    Get singleton context memory manager instance.
    
    Returns:
        ContextMemoryManager instance
    """
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = ContextMemoryManager()
    
    return _memory_manager


async def start_memory_management() -> None:
    """Start context memory management."""
    memory_manager = get_context_memory_manager()
    await memory_manager.start_memory_management()


async def stop_memory_management() -> None:
    """Stop context memory management."""
    global _memory_manager
    
    if _memory_manager:
        await _memory_manager.stop_memory_management()
        _memory_manager = None