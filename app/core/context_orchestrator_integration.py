"""
Context Orchestrator Integration - Sleep-Wake Cycle Context Management.

Provides seamless integration between the context engine and orchestrator with:
- Sleep-wake cycle context consolidation
- Session state preservation and restoration
- Memory consolidation during sleep phases  
- Context warming during wake phases
- Orchestrator event handling and synchronization
- Performance-optimized session transitions
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..core.config import get_settings
from ..core.context_engine_integration import (
    ContextEngineIntegration,
    get_context_engine_integration
)
from ..core.context_consolidation_triggers import (
    ConsolidationTriggerManager,
    ConsolidationTrigger,
    TriggerType,
    TriggerPriority,
    get_consolidation_trigger_manager
)
from ..core.context_memory_manager import (
    ContextMemoryManager,
    get_context_memory_manager
)
from ..core.context_cache_manager import (
    ContextCacheManager,
    get_context_cache_manager
)
from ..core.context_lifecycle_manager import (
    ContextLifecycleManager,
    VersionAction,
    get_context_lifecycle_manager
)


logger = logging.getLogger(__name__)


class OrchestratorEvent(Enum):
    """Orchestrator events that trigger context operations."""
    AGENT_SLEEP_INITIATED = "agent_sleep_initiated"
    AGENT_SLEEP_COMPLETED = "agent_sleep_completed"
    AGENT_WAKE_INITIATED = "agent_wake_initiated"
    AGENT_WAKE_COMPLETED = "agent_wake_completed"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    MEMORY_PRESSURE_DETECTED = "memory_pressure_detected"
    CONSOLIDATION_REQUESTED = "consolidation_requested"


class SleepPhase(Enum):
    """Sleep phases for context processing."""
    LIGHT_SLEEP = "light_sleep"        # Basic consolidation
    DEEP_SLEEP = "deep_sleep"          # Heavy consolidation and cleanup
    REM_SLEEP = "rem_sleep"            # Pattern analysis and optimization


@dataclass
class SleepCycleContext:
    """Context information for sleep cycle processing."""
    agent_id: UUID
    session_id: Optional[UUID]
    sleep_initiated_at: datetime
    expected_wake_time: Optional[datetime]
    sleep_phase: SleepPhase
    contexts_count: int
    unconsolidated_count: int
    memory_usage_mb: float
    last_activity_at: datetime
    consolidation_priority: TriggerPriority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": str(self.agent_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "sleep_initiated_at": self.sleep_initiated_at.isoformat(),
            "expected_wake_time": self.expected_wake_time.isoformat() if self.expected_wake_time else None,
            "sleep_phase": self.sleep_phase.value,
            "contexts_count": self.contexts_count,
            "unconsolidated_count": self.unconsolidated_count,
            "memory_usage_mb": self.memory_usage_mb,
            "last_activity_at": self.last_activity_at.isoformat(),
            "consolidation_priority": self.consolidation_priority.value
        }


@dataclass
class WakeContext:
    """Context information for wake cycle processing."""
    agent_id: UUID
    session_id: Optional[UUID]
    wake_initiated_at: datetime
    sleep_duration_minutes: float
    contexts_processed_during_sleep: int
    memory_freed_mb: float
    consolidation_ratio: float
    cache_warmup_required: bool
    priority_contexts: List[str]  # Context IDs to prioritize
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": str(self.agent_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "wake_initiated_at": self.wake_initiated_at.isoformat(),
            "sleep_duration_minutes": self.sleep_duration_minutes,
            "contexts_processed_during_sleep": self.contexts_processed_during_sleep,
            "memory_freed_mb": self.memory_freed_mb,
            "consolidation_ratio": self.consolidation_ratio,
            "cache_warmup_required": self.cache_warmup_required,
            "priority_contexts": self.priority_contexts
        }


class ContextOrchestratorIntegration:
    """
    Integration layer between context engine and orchestrator.
    
    Features:
    - Sleep-wake cycle context management
    - Automated consolidation during sleep phases
    - Context warming during wake phases
    - Session state preservation and restoration
    - Performance-optimized transitions
    - Event-driven orchestrator integration
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        
        # Core services (initialized lazily)
        self._context_engine: Optional[ContextEngineIntegration] = None
        self._trigger_manager: Optional[ConsolidationTriggerManager] = None
        self._memory_manager: Optional[ContextMemoryManager] = None
        self._cache_manager: Optional[ContextCacheManager] = None
        self._lifecycle_manager: Optional[ContextLifecycleManager] = None
        
        # Sleep-wake cycle tracking
        self.active_sleep_cycles: Dict[UUID, SleepCycleContext] = {}
        self.wake_contexts: Dict[UUID, WakeContext] = {}
        
        # Event handlers
        self.event_handlers: Dict[OrchestratorEvent, List[Callable]] = {
            event: [] for event in OrchestratorEvent
        }
        
        # Performance metrics
        self.integration_metrics = {
            "sleep_cycles_processed": 0,
            "wake_cycles_processed": 0,
            "contexts_consolidated_during_sleep": 0,
            "contexts_warmed_during_wake": 0,
            "avg_sleep_processing_time_ms": 0.0,
            "avg_wake_processing_time_ms": 0.0,
            "memory_optimized_mb": 0.0
        }
        
        # Background tasks
        self._sleep_monitor_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def initialize(self) -> None:
        """Initialize the orchestrator integration."""
        try:
            logger.info("Initializing Context Orchestrator Integration")
            
            # Initialize core services
            self._context_engine = await get_context_engine_integration()
            self._trigger_manager = get_consolidation_trigger_manager()
            self._memory_manager = get_context_memory_manager()
            self._cache_manager = get_context_cache_manager()
            self._lifecycle_manager = get_context_lifecycle_manager()
            
            # Register default event handlers
            await self._register_default_handlers()
            
            # Start monitoring
            self._is_running = True
            self._sleep_monitor_task = asyncio.create_task(self._sleep_monitor_loop())
            
            logger.info("Context Orchestrator Integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Context Orchestrator Integration: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator integration."""
        try:
            logger.info("Shutting down Context Orchestrator Integration")
            
            self._is_running = False
            
            # Cancel background tasks
            if self._sleep_monitor_task and not self._sleep_monitor_task.done():
                self._sleep_monitor_task.cancel()
                try:
                    await self._sleep_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Clear active cycles
            self.active_sleep_cycles.clear()
            self.wake_contexts.clear()
            
            logger.info("Context Orchestrator Integration shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Context Orchestrator Integration: {e}")
    
    async def handle_agent_sleep_initiated(
        self,
        agent_id: UUID,
        session_id: Optional[UUID] = None,
        expected_wake_time: Optional[datetime] = None,
        sleep_phase: SleepPhase = SleepPhase.LIGHT_SLEEP
    ) -> SleepCycleContext:
        """
        Handle agent sleep initiation.
        
        Args:
            agent_id: Agent entering sleep
            session_id: Current session ID
            expected_wake_time: When agent is expected to wake
            sleep_phase: Type of sleep phase
            
        Returns:
            Sleep cycle context information
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Initiating sleep cycle for agent {agent_id} ({sleep_phase.value})")
            
            # Gather context information
            context_stats = await self._gather_agent_context_stats(agent_id)
            
            # Create sleep cycle context
            sleep_context = SleepCycleContext(
                agent_id=agent_id,
                session_id=session_id,
                sleep_initiated_at=start_time,
                expected_wake_time=expected_wake_time,
                sleep_phase=sleep_phase,
                contexts_count=context_stats["total_contexts"],
                unconsolidated_count=context_stats["unconsolidated_contexts"],
                memory_usage_mb=context_stats["estimated_memory_mb"],
                last_activity_at=context_stats["last_activity"],
                consolidation_priority=self._determine_consolidation_priority(context_stats)
            )
            
            # Store sleep context
            self.active_sleep_cycles[agent_id] = sleep_context
            await self._store_sleep_context(sleep_context)
            
            # Create recovery point before sleep processing
            if self._lifecycle_manager:
                await self._lifecycle_manager.create_recovery_point(
                    context_id=agent_id,  # Using agent_id as context for session-level recovery
                    recovery_type="pre_sleep",
                    metadata={
                        "sleep_phase": sleep_phase.value,
                        "contexts_count": context_stats["total_contexts"],
                        "session_id": str(session_id) if session_id else None
                    }
                )
            
            # Trigger sleep-specific consolidation
            if sleep_context.unconsolidated_count > 0:
                await self._trigger_sleep_consolidation(sleep_context)
            
            # Emit event
            await self._emit_event(OrchestratorEvent.AGENT_SLEEP_INITIATED, {
                "agent_id": str(agent_id),
                "sleep_context": sleep_context.to_dict()
            })
            
            # Update metrics
            self.integration_metrics["sleep_cycles_processed"] += 1
            
            logger.info(
                f"Sleep cycle initiated for agent {agent_id}: "
                f"{sleep_context.unconsolidated_count} contexts to process"
            )
            
            return sleep_context
            
        except Exception as e:
            logger.error(f"Error initiating sleep cycle for agent {agent_id}: {e}")
            raise
    
    async def handle_agent_sleep_completed(
        self,
        agent_id: UUID,
        consolidation_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle agent sleep completion.
        
        Args:
            agent_id: Agent completing sleep
            consolidation_results: Results from consolidation process
            
        Returns:
            Sleep completion results
        """
        try:
            logger.info(f"Completing sleep cycle for agent {agent_id}")
            
            # Get sleep context
            sleep_context = self.active_sleep_cycles.get(agent_id)
            if not sleep_context:
                logger.warning(f"No active sleep context found for agent {agent_id}")
                return {"error": "No active sleep context"}
            
            # Calculate sleep duration
            sleep_duration = datetime.utcnow() - sleep_context.sleep_initiated_at
            sleep_duration_minutes = sleep_duration.total_seconds() / 60
            
            # Gather post-sleep statistics
            post_sleep_stats = await self._gather_agent_context_stats(agent_id)
            
            # Calculate processing results
            contexts_processed = (
                sleep_context.unconsolidated_count - post_sleep_stats["unconsolidated_contexts"]
            )
            memory_freed = max(0, sleep_context.memory_usage_mb - post_sleep_stats["estimated_memory_mb"])
            
            # Create completion results
            completion_results = {
                "agent_id": str(agent_id),
                "sleep_duration_minutes": sleep_duration_minutes,
                "contexts_processed": contexts_processed,
                "memory_freed_mb": memory_freed,
                "consolidation_ratio": contexts_processed / max(1, sleep_context.contexts_count),
                "sleep_phase": sleep_context.sleep_phase.value,
                "post_sleep_stats": post_sleep_stats
            }
            
            # Store completion results
            await self._store_sleep_completion_results(agent_id, completion_results)
            
            # Clean up sleep context
            self.active_sleep_cycles.pop(agent_id, None)
            
            # Update metrics
            self.integration_metrics["contexts_consolidated_during_sleep"] += contexts_processed
            self.integration_metrics["memory_optimized_mb"] += memory_freed
            
            # Emit event
            await self._emit_event(OrchestratorEvent.AGENT_SLEEP_COMPLETED, completion_results)
            
            logger.info(
                f"Sleep cycle completed for agent {agent_id}: "
                f"{contexts_processed} contexts processed, {memory_freed:.1f}MB freed"
            )
            
            return completion_results
            
        except Exception as e:
            logger.error(f"Error completing sleep cycle for agent {agent_id}: {e}")
            return {"error": str(e)}
    
    async def handle_agent_wake_initiated(
        self,
        agent_id: UUID,
        session_id: Optional[UUID] = None,
        priority_contexts: Optional[List[str]] = None
    ) -> WakeContext:
        """
        Handle agent wake initiation.
        
        Args:
            agent_id: Agent waking up
            session_id: New session ID
            priority_contexts: Priority context IDs to warm
            
        Returns:
            Wake context information
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Initiating wake cycle for agent {agent_id}")
            
            # Get sleep completion results
            sleep_results = await self._get_sleep_completion_results(agent_id)
            
            # Calculate sleep duration
            sleep_duration_minutes = 0.0
            if sleep_results:
                sleep_duration_minutes = sleep_results.get("sleep_duration_minutes", 0.0)
            
            # Create wake context
            wake_context = WakeContext(
                agent_id=agent_id,
                session_id=session_id,
                wake_initiated_at=start_time,
                sleep_duration_minutes=sleep_duration_minutes,
                contexts_processed_during_sleep=sleep_results.get("contexts_processed", 0) if sleep_results else 0,
                memory_freed_mb=sleep_results.get("memory_freed_mb", 0.0) if sleep_results else 0.0,
                consolidation_ratio=sleep_results.get("consolidation_ratio", 0.0) if sleep_results else 0.0,
                cache_warmup_required=sleep_duration_minutes > 60,  # Warm cache if slept > 1 hour
                priority_contexts=priority_contexts or []
            )
            
            # Store wake context
            self.wake_contexts[agent_id] = wake_context
            await self._store_wake_context(wake_context)
            
            # Perform cache warming if required
            if wake_context.cache_warmup_required and self._cache_manager:
                warmup_task = asyncio.create_task(
                    self._perform_cache_warmup(agent_id, priority_contexts)
                )
                # Don't await - run in background
            
            # Restore session state if needed
            if session_id:
                await self._restore_session_state(agent_id, session_id)
            
            # Emit event
            await self._emit_event(OrchestratorEvent.AGENT_WAKE_INITIATED, {
                "agent_id": str(agent_id),
                "wake_context": wake_context.to_dict()
            })
            
            # Update metrics
            self.integration_metrics["wake_cycles_processed"] += 1
            
            logger.info(
                f"Wake cycle initiated for agent {agent_id}: "
                f"cache warmup {'required' if wake_context.cache_warmup_required else 'not required'}"
            )
            
            return wake_context
            
        except Exception as e:
            logger.error(f"Error initiating wake cycle for agent {agent_id}: {e}")
            raise
    
    async def handle_agent_wake_completed(
        self,
        agent_id: UUID,
        warmup_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle agent wake completion.
        
        Args:
            agent_id: Agent completing wake
            warmup_results: Results from cache warming
            
        Returns:
            Wake completion results
        """
        try:
            logger.info(f"Completing wake cycle for agent {agent_id}")
            
            # Get wake context
            wake_context = self.wake_contexts.get(agent_id)
            if not wake_context:
                logger.warning(f"No active wake context found for agent {agent_id}")
                return {"error": "No active wake context"}
            
            # Calculate wake duration
            wake_duration = datetime.utcnow() - wake_context.wake_initiated_at
            wake_duration_ms = wake_duration.total_seconds() * 1000
            
            # Create completion results
            completion_results = {
                "agent_id": str(agent_id),
                "wake_duration_ms": wake_duration_ms,
                "contexts_warmed": warmup_results.get("contexts_warmed", 0) if warmup_results else 0,
                "cache_hit_rate_improvement": warmup_results.get("hit_rate_improvement", 0.0) if warmup_results else 0.0,
                "session_restored": wake_context.session_id is not None,
                "wake_context": wake_context.to_dict()
            }
            
            # Store completion results
            await self._store_wake_completion_results(agent_id, completion_results)
            
            # Clean up wake context
            self.wake_contexts.pop(agent_id, None)
            
            # Update metrics
            if warmup_results:
                self.integration_metrics["contexts_warmed_during_wake"] += warmup_results.get("contexts_warmed", 0)
            
            # Update average processing times
            current_avg = self.integration_metrics["avg_wake_processing_time_ms"]
            total_wakes = self.integration_metrics["wake_cycles_processed"]
            new_avg = ((current_avg * (total_wakes - 1)) + wake_duration_ms) / total_wakes
            self.integration_metrics["avg_wake_processing_time_ms"] = new_avg
            
            # Emit event
            await self._emit_event(OrchestratorEvent.AGENT_WAKE_COMPLETED, completion_results)
            
            logger.info(
                f"Wake cycle completed for agent {agent_id}: "
                f"{completion_results['contexts_warmed']} contexts warmed in {wake_duration_ms:.0f}ms"
            )
            
            return completion_results
            
        except Exception as e:
            logger.error(f"Error completing wake cycle for agent {agent_id}: {e}")
            return {"error": str(e)}
    
    async def handle_session_started(
        self,
        agent_id: UUID,
        session_id: UUID,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle session start event.
        
        Args:
            agent_id: Agent starting session
            session_id: New session ID
            session_metadata: Session metadata
            
        Returns:
            Session start results
        """
        try:
            logger.info(f"Starting session {session_id} for agent {agent_id}")
            
            # Create session recovery point
            if self._lifecycle_manager:
                await self._lifecycle_manager.create_recovery_point(
                    context_id=agent_id,
                    recovery_type="session_start",
                    metadata={
                        "session_id": str(session_id),
                        "session_metadata": session_metadata or {},
                        "started_at": datetime.utcnow().isoformat()
                    }
                )
            
            # Warm relevant contexts for new session
            if self._cache_manager:
                await self._cache_manager.warm_cache_for_agent(agent_id, context_limit=20)
            
            # Emit event
            await self._emit_event(OrchestratorEvent.SESSION_STARTED, {
                "agent_id": str(agent_id),
                "session_id": str(session_id),
                "metadata": session_metadata or {}
            })
            
            return {
                "agent_id": str(agent_id),
                "session_id": str(session_id),
                "recovery_point_created": True,
                "cache_warmed": True
            }
            
        except Exception as e:
            logger.error(f"Error handling session start for agent {agent_id}: {e}")
            return {"error": str(e)}
    
    async def handle_session_ended(
        self,
        agent_id: UUID,
        session_id: UUID,
        session_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle session end event.
        
        Args:
            agent_id: Agent ending session
            session_id: Session ID ending
            session_summary: Summary of session
            
        Returns:
            Session end results
        """
        try:
            logger.info(f"Ending session {session_id} for agent {agent_id}")
            
            # Check if consolidation is needed
            consolidation_triggered = False
            if self._trigger_manager:
                triggers = await self._trigger_manager.check_agent_triggers(agent_id)
                if triggers:
                    # Trigger consolidation for session end
                    consolidation_triggered = True
                    asyncio.create_task(self._process_consolidation_triggers(triggers))
            
            # Invalidate session-specific caches
            if self._cache_manager:
                await self._cache_manager.invalidate_agent_contexts(agent_id)
            
            # Create session end recovery point
            if self._lifecycle_manager:
                await self._lifecycle_manager.create_recovery_point(
                    context_id=agent_id,
                    recovery_type="session_end",
                    metadata={
                        "session_id": str(session_id),
                        "session_summary": session_summary,
                        "ended_at": datetime.utcnow().isoformat(),
                        "consolidation_triggered": consolidation_triggered
                    }
                )
            
            # Emit event
            await self._emit_event(OrchestratorEvent.SESSION_ENDED, {
                "agent_id": str(agent_id),
                "session_id": str(session_id),
                "session_summary": session_summary,
                "consolidation_triggered": consolidation_triggered
            })
            
            return {
                "agent_id": str(agent_id),
                "session_id": str(session_id),
                "consolidation_triggered": consolidation_triggered,
                "caches_invalidated": True,
                "recovery_point_created": True
            }
            
        except Exception as e:
            logger.error(f"Error handling session end for agent {agent_id}: {e}")
            return {"error": str(e)}
    
    async def register_event_handler(
        self,
        event: OrchestratorEvent,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register event handler for orchestrator events.
        
        Args:
            event: Event to handle
            handler: Handler function
        """
        self.event_handlers[event].append(handler)
        logger.info(f"Registered handler for event {event.value}")
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        try:
            status = {
                "is_running": self._is_running,
                "active_sleep_cycles": len(self.active_sleep_cycles),
                "active_wake_contexts": len(self.wake_contexts),
                "integration_metrics": self.integration_metrics.copy(),
                "component_status": {},
                "recent_events": await self._get_recent_events()
            }
            
            # Component status
            if self._context_engine:
                engine_health = await self._context_engine.get_comprehensive_health_status()
                status["component_status"]["context_engine"] = engine_health["status"]
            
            if self._memory_manager:
                memory_stats = await self._memory_manager.get_memory_statistics()
                status["component_status"]["memory_manager"] = "healthy" if "error" not in memory_stats else "unhealthy"
            
            if self._cache_manager:
                cache_stats = await self._cache_manager.get_cache_statistics()
                status["component_status"]["cache_manager"] = "healthy" if "error" not in cache_stats else "unhealthy"
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {"error": str(e)}
    
    # Private Methods
    
    async def _gather_agent_context_stats(self, agent_id: UUID) -> Dict[str, Any]:
        """Gather context statistics for an agent."""
        try:
            async with get_async_session() as session:
                # Total contexts
                total_contexts = await session.scalar(
                    select(func.count(Context.id)).where(Context.agent_id == agent_id)
                ) or 0
                
                # Unconsolidated contexts
                unconsolidated_contexts = await session.scalar(
                    select(func.count(Context.id)).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.is_consolidated == "false"
                        )
                    )
                ) or 0
                
                # Last activity
                last_activity = await session.scalar(
                    select(func.max(Context.accessed_at)).where(Context.agent_id == agent_id)
                ) or datetime.utcnow()
                
                # Estimate memory usage (simplified)
                estimated_memory_mb = total_contexts * 0.1  # Rough estimate
                
                return {
                    "total_contexts": total_contexts,
                    "unconsolidated_contexts": unconsolidated_contexts,
                    "estimated_memory_mb": estimated_memory_mb,
                    "last_activity": last_activity
                }
                
        except Exception as e:
            logger.error(f"Error gathering context stats for agent {agent_id}: {e}")
            return {
                "total_contexts": 0,
                "unconsolidated_contexts": 0,
                "estimated_memory_mb": 0.0,
                "last_activity": datetime.utcnow()
            }
    
    def _determine_consolidation_priority(self, context_stats: Dict[str, Any]) -> TriggerPriority:
        """Determine consolidation priority based on context statistics."""
        unconsolidated = context_stats["unconsolidated_contexts"]
        memory_mb = context_stats["estimated_memory_mb"]
        
        if unconsolidated >= 100 or memory_mb >= 500:
            return TriggerPriority.CRITICAL
        elif unconsolidated >= 50 or memory_mb >= 200:
            return TriggerPriority.HIGH
        elif unconsolidated >= 20 or memory_mb >= 100:
            return TriggerPriority.MEDIUM
        else:
            return TriggerPriority.LOW
    
    async def _trigger_sleep_consolidation(self, sleep_context: SleepCycleContext) -> None:
        """Trigger consolidation during sleep cycle."""
        try:
            if not self._trigger_manager:
                return
            
            # Register sleep cycle trigger
            trigger = await self._trigger_manager.register_sleep_cycle_trigger(
                agent_id=sleep_context.agent_id,
                sleep_context_count=sleep_context.unconsolidated_count,
                expected_wake_time=sleep_context.expected_wake_time
            )
            
            if trigger and self._context_engine:
                # Execute consolidation
                consolidation_result = await self._context_engine.trigger_consolidation(
                    agent_id=sleep_context.agent_id,
                    trigger_type=trigger.trigger_type,
                    target_reduction=0.7  # 70% reduction target for sleep
                )
                
                logger.info(
                    f"Sleep consolidation completed for agent {sleep_context.agent_id}: "
                    f"{consolidation_result.compression_ratio:.1%} reduction achieved"
                )
                
        except Exception as e:
            logger.error(f"Error triggering sleep consolidation: {e}")
    
    async def _perform_cache_warmup(
        self,
        agent_id: UUID,
        priority_contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform cache warmup during wake cycle."""
        try:
            if not self._cache_manager:
                return {"contexts_warmed": 0}
            
            # Warm general agent contexts
            warmed_count = await self._cache_manager.warm_cache_for_agent(
                agent_id=agent_id,
                context_limit=30
            )
            
            # Warm priority contexts if specified
            priority_warmed = 0
            if priority_contexts:
                for context_id_str in priority_contexts:
                    try:
                        context_id = UUID(context_id_str)
                        context = await self._cache_manager.get_context(context_id, use_cache=False)
                        if context:
                            await self._cache_manager.store_context(context)
                            priority_warmed += 1
                    except Exception as e:
                        logger.warning(f"Error warming priority context {context_id_str}: {e}")
            
            total_warmed = warmed_count + priority_warmed
            
            logger.info(f"Cache warmup completed for agent {agent_id}: {total_warmed} contexts warmed")
            
            return {
                "contexts_warmed": total_warmed,
                "general_contexts": warmed_count,
                "priority_contexts": priority_warmed,
                "hit_rate_improvement": 0.2  # Estimated improvement
            }
            
        except Exception as e:
            logger.error(f"Error performing cache warmup: {e}")
            return {"contexts_warmed": 0, "error": str(e)}
    
    async def _restore_session_state(self, agent_id: UUID, session_id: UUID) -> None:
        """Restore session state during wake cycle."""
        try:
            # This would integrate with session management
            # For now, just log the restoration
            logger.info(f"Restoring session state for agent {agent_id}, session {session_id}")
            
        except Exception as e:
            logger.error(f"Error restoring session state: {e}")
    
    async def _process_consolidation_triggers(self, triggers: List[ConsolidationTrigger]) -> None:
        """Process consolidation triggers."""
        try:
            if not self._context_engine:
                return
            
            for trigger in triggers:
                try:
                    await self._context_engine.trigger_consolidation(
                        agent_id=trigger.agent_id,
                        trigger_type=trigger.trigger_type
                    )
                except Exception as e:
                    logger.error(f"Error processing consolidation trigger {trigger.trigger_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing consolidation triggers: {e}")
    
    async def _store_sleep_context(self, sleep_context: SleepCycleContext) -> None:
        """Store sleep context in Redis."""
        try:
            key = f"sleep_context:{sleep_context.agent_id}"
            await self.redis_client.setex(
                key,
                86400,  # 24 hours
                json.dumps(sleep_context.to_dict(), default=str)
            )
        except Exception as e:
            logger.error(f"Error storing sleep context: {e}")
    
    async def _store_wake_context(self, wake_context: WakeContext) -> None:
        """Store wake context in Redis."""
        try:
            key = f"wake_context:{wake_context.agent_id}"
            await self.redis_client.setex(
                key,
                86400,  # 24 hours
                json.dumps(wake_context.to_dict(), default=str)
            )
        except Exception as e:
            logger.error(f"Error storing wake context: {e}")
    
    async def _store_sleep_completion_results(
        self,
        agent_id: UUID,
        results: Dict[str, Any]
    ) -> None:
        """Store sleep completion results."""
        try:
            key = f"sleep_results:{agent_id}"
            await self.redis_client.setex(
                key,
                86400,
                json.dumps(results, default=str)
            )
        except Exception as e:
            logger.error(f"Error storing sleep completion results: {e}")
    
    async def _store_wake_completion_results(
        self,
        agent_id: UUID,
        results: Dict[str, Any]
    ) -> None:
        """Store wake completion results."""
        try:
            key = f"wake_results:{agent_id}"
            await self.redis_client.setex(
                key,
                86400,
                json.dumps(results, default=str)
            )
        except Exception as e:
            logger.error(f"Error storing wake completion results: {e}")
    
    async def _get_sleep_completion_results(self, agent_id: UUID) -> Optional[Dict[str, Any]]:
        """Get sleep completion results."""
        try:
            key = f"sleep_results:{agent_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data.decode())
            return None
        except Exception as e:
            logger.error(f"Error getting sleep completion results: {e}")
            return None
    
    async def _emit_event(self, event: OrchestratorEvent, data: Dict[str, Any]) -> None:
        """Emit orchestrator event."""
        try:
            # Store event in Redis for external consumption
            event_data = {
                "event": event.value,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.lpush(
                "orchestrator_events",
                json.dumps(event_data, default=str)
            )
            
            # Keep only recent events
            await self.redis_client.ltrim("orchestrator_events", 0, 999)
            
            # Call registered handlers
            handlers = self.event_handlers.get(event, [])
            for handler in handlers:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.value}: {e}")
                    
        except Exception as e:
            logger.error(f"Error emitting event {event.value}: {e}")
    
    async def _get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent orchestrator events."""
        try:
            events = await self.redis_client.lrange("orchestrator_events", 0, limit - 1)
            return [json.loads(event.decode()) for event in events]
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        try:
            # Example: log all events
            async def log_event_handler(data: Dict[str, Any]) -> None:
                logger.info(f"Orchestrator event: {data}")
            
            for event in OrchestratorEvent:
                await self.register_event_handler(event, log_event_handler)
                
        except Exception as e:
            logger.error(f"Error registering default handlers: {e}")
    
    async def _sleep_monitor_loop(self) -> None:
        """Background loop for monitoring sleep cycles."""
        logger.info("Starting sleep monitor loop")
        
        try:
            while self._is_running:
                try:
                    # Monitor active sleep cycles for completion
                    current_time = datetime.utcnow()
                    
                    for agent_id, sleep_context in list(self.active_sleep_cycles.items()):
                        # Check if sleep should be completed (based on expected wake time or duration)
                        if sleep_context.expected_wake_time and current_time >= sleep_context.expected_wake_time:
                            logger.info(f"Auto-completing sleep cycle for agent {agent_id}")
                            await self.handle_agent_sleep_completed(agent_id)
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in sleep monitor loop: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("Sleep monitor loop stopped")


# Global instance for application use
_orchestrator_integration: Optional[ContextOrchestratorIntegration] = None


async def get_context_orchestrator_integration() -> ContextOrchestratorIntegration:
    """
    Get singleton context orchestrator integration instance.
    
    Returns:
        ContextOrchestratorIntegration instance
    """
    global _orchestrator_integration
    
    if _orchestrator_integration is None:
        _orchestrator_integration = ContextOrchestratorIntegration()
        await _orchestrator_integration.initialize()
    
    return _orchestrator_integration


async def cleanup_context_orchestrator_integration() -> None:
    """Cleanup context orchestrator integration resources."""
    global _orchestrator_integration
    
    if _orchestrator_integration:
        await _orchestrator_integration.shutdown()
        _orchestrator_integration = None