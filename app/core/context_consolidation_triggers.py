"""
Context Consolidation Triggers - Smart Automated Context Management.

Provides intelligent triggers for context consolidation based on:
- Usage patterns and access frequency
- Memory pressure and resource constraints
- Time-based scheduling and agent activity
- Sleep-wake cycle integration
- Performance threshold monitoring
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, AsyncIterator
from uuid import UUID
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.config import get_settings
from ..core.redis import get_redis_client


logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of consolidation triggers."""
    USAGE_THRESHOLD = "usage_threshold"
    TIME_SCHEDULE = "time_schedule"
    MEMORY_PRESSURE = "memory_pressure"
    AGENT_ACTIVITY = "agent_activity"
    SLEEP_CYCLE = "sleep_cycle"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class TriggerPriority(Enum):
    """Priority levels for consolidation triggers."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConsolidationTrigger:
    """Represents a consolidation trigger event."""
    trigger_id: str
    trigger_type: TriggerType
    priority: TriggerPriority
    agent_id: UUID
    triggered_at: datetime
    expected_processing_time_ms: float
    context_count_estimate: int
    memory_pressure_mb: float
    trigger_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trigger to dictionary."""
        return {
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type.value,
            "priority": self.priority.value,
            "agent_id": str(self.agent_id),
            "triggered_at": self.triggered_at.isoformat(),
            "expected_processing_time_ms": self.expected_processing_time_ms,
            "context_count_estimate": self.context_count_estimate,
            "memory_pressure_mb": self.memory_pressure_mb,
            "trigger_metadata": self.trigger_metadata
        }


@dataclass
class AgentUsagePattern:
    """Represents agent usage patterns for trigger decisions."""
    agent_id: UUID
    contexts_created_per_hour: float
    contexts_accessed_per_hour: float
    avg_session_duration_minutes: float
    peak_activity_hours: List[int]
    consolidation_frequency_hours: float
    last_consolidation: Optional[datetime]
    current_unconsolidated_count: int
    memory_usage_mb: float
    is_active: bool


class ConsolidationTriggerManager:
    """
    Manages intelligent triggers for context consolidation.
    
    Features:
    - Usage pattern analysis and smart triggers
    - Memory pressure monitoring
    - Time-based scheduling with agent activity awareness
    - Sleep-wake cycle integration
    - Emergency consolidation for resource protection
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        
        # Configuration
        self.usage_threshold_contexts = 15
        self.memory_pressure_threshold_mb = 500
        self.max_consolidation_age_hours = 6
        self.emergency_threshold_contexts = 100
        self.consolidation_batch_size = 50
        
        # State tracking
        self.agent_patterns: Dict[UUID, AgentUsagePattern] = {}
        self.trigger_history: deque = deque(maxlen=1000)
        self.active_triggers: Dict[str, ConsolidationTrigger] = {}
        
        # Performance tracking
        self.trigger_success_rates: Dict[TriggerType, float] = {}
        self.avg_processing_times: Dict[TriggerType, float] = {}
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start_monitoring(self) -> None:
        """Start the trigger monitoring system."""
        if self._is_running:
            return
        
        logger.info("Starting consolidation trigger monitoring")
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop the trigger monitoring system."""
        if not self._is_running:
            return
        
        logger.info("Stopping consolidation trigger monitoring")
        self._is_running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def check_all_triggers(self) -> List[ConsolidationTrigger]:
        """
        Check all trigger conditions and return triggered consolidations.
        
        Returns:
            List of consolidation triggers that should be executed
        """
        triggered_consolidations = []
        
        try:
            # Update agent patterns
            await self._update_agent_patterns()
            
            # Check each trigger type
            for agent_id, pattern in self.agent_patterns.items():
                # Usage threshold triggers
                usage_triggers = await self._check_usage_triggers(agent_id, pattern)
                triggered_consolidations.extend(usage_triggers)
                
                # Memory pressure triggers
                memory_triggers = await self._check_memory_triggers(agent_id, pattern)
                triggered_consolidations.extend(memory_triggers)
                
                # Time-based triggers
                time_triggers = await self._check_time_triggers(agent_id, pattern)
                triggered_consolidations.extend(time_triggers)
                
                # Activity-based triggers
                activity_triggers = await self._check_activity_triggers(agent_id, pattern)
                triggered_consolidations.extend(activity_triggers)
                
                # Emergency triggers
                emergency_triggers = await self._check_emergency_triggers(agent_id, pattern)
                triggered_consolidations.extend(emergency_triggers)
            
            # Sort by priority
            triggered_consolidations.sort(
                key=lambda t: (t.priority.value, t.triggered_at),
                reverse=True
            )
            
            # Store active triggers
            for trigger in triggered_consolidations:
                self.active_triggers[trigger.trigger_id] = trigger
                self.trigger_history.append(trigger)
            
            if triggered_consolidations:
                logger.info(f"Generated {len(triggered_consolidations)} consolidation triggers")
            
            return triggered_consolidations
            
        except Exception as e:
            logger.error(f"Error checking consolidation triggers: {e}")
            return []
    
    async def check_agent_triggers(self, agent_id: UUID) -> List[ConsolidationTrigger]:
        """
        Check consolidation triggers for a specific agent.
        
        Args:
            agent_id: Agent ID to check triggers for
            
        Returns:
            List of consolidation triggers for the agent
        """
        try:
            # Update pattern for this agent
            pattern = await self._get_agent_pattern(agent_id)
            if not pattern:
                return []
            
            self.agent_patterns[agent_id] = pattern
            
            triggers = []
            
            # Check all trigger types for this agent
            triggers.extend(await self._check_usage_triggers(agent_id, pattern))
            triggers.extend(await self._check_memory_triggers(agent_id, pattern))
            triggers.extend(await self._check_time_triggers(agent_id, pattern))
            triggers.extend(await self._check_activity_triggers(agent_id, pattern))
            triggers.extend(await self._check_emergency_triggers(agent_id, pattern))
            
            return triggers
            
        except Exception as e:
            logger.error(f"Error checking triggers for agent {agent_id}: {e}")
            return []
    
    async def register_sleep_cycle_trigger(
        self,
        agent_id: UUID,
        sleep_context_count: int,
        expected_wake_time: Optional[datetime] = None
    ) -> Optional[ConsolidationTrigger]:
        """
        Register a sleep cycle consolidation trigger.
        
        Args:
            agent_id: Agent entering sleep cycle
            sleep_context_count: Number of contexts to consolidate
            expected_wake_time: When agent is expected to wake
            
        Returns:
            Consolidation trigger if conditions are met
        """
        try:
            # Check if consolidation is warranted
            if sleep_context_count < 5:
                logger.debug(f"Skipping sleep cycle consolidation for agent {agent_id} - too few contexts")
                return None
            
            # Create sleep cycle trigger
            trigger = ConsolidationTrigger(
                trigger_id=f"sleep_{agent_id}_{int(datetime.utcnow().timestamp())}",
                trigger_type=TriggerType.SLEEP_CYCLE,
                priority=TriggerPriority.HIGH,
                agent_id=agent_id,
                triggered_at=datetime.utcnow(),
                expected_processing_time_ms=sleep_context_count * 200,  # Estimate 200ms per context
                context_count_estimate=sleep_context_count,
                memory_pressure_mb=0.0,
                trigger_metadata={
                    "sleep_cycle": True,
                    "expected_wake_time": expected_wake_time.isoformat() if expected_wake_time else None,
                    "consolidation_reason": "agent_sleep_cycle"
                }
            )
            
            self.active_triggers[trigger.trigger_id] = trigger
            self.trigger_history.append(trigger)
            
            logger.info(f"Registered sleep cycle trigger for agent {agent_id} ({sleep_context_count} contexts)")
            return trigger
            
        except Exception as e:
            logger.error(f"Error registering sleep cycle trigger for agent {agent_id}: {e}")
            return None
    
    async def register_manual_trigger(
        self,
        agent_id: UUID,
        priority: TriggerPriority = TriggerPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsolidationTrigger:
        """
        Register a manual consolidation trigger.
        
        Args:
            agent_id: Agent to consolidate contexts for
            priority: Trigger priority
            metadata: Additional trigger metadata
            
        Returns:
            Manual consolidation trigger
        """
        try:
            # Get context count estimate
            async with get_async_session() as session:
                context_count = await session.scalar(
                    select(func.count(Context.id)).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.is_consolidated == "false"
                        )
                    )
                ) or 0
            
            trigger = ConsolidationTrigger(
                trigger_id=f"manual_{agent_id}_{int(datetime.utcnow().timestamp())}",
                trigger_type=TriggerType.MANUAL,
                priority=priority,
                agent_id=agent_id,
                triggered_at=datetime.utcnow(),
                expected_processing_time_ms=context_count * 150,
                context_count_estimate=context_count,
                memory_pressure_mb=0.0,
                trigger_metadata=metadata or {"manual_trigger": True}
            )
            
            self.active_triggers[trigger.trigger_id] = trigger
            self.trigger_history.append(trigger)
            
            logger.info(f"Registered manual trigger for agent {agent_id}")
            return trigger
            
        except Exception as e:
            logger.error(f"Error registering manual trigger for agent {agent_id}: {e}")
            raise
    
    async def complete_trigger(
        self,
        trigger_id: str,
        success: bool,
        processing_time_ms: float,
        contexts_processed: int
    ) -> None:
        """
        Mark a trigger as completed and update performance metrics.
        
        Args:
            trigger_id: ID of completed trigger
            success: Whether trigger execution was successful
            processing_time_ms: Actual processing time
            contexts_processed: Number of contexts actually processed
        """
        try:
            trigger = self.active_triggers.pop(trigger_id, None)
            if not trigger:
                logger.warning(f"Trigger {trigger_id} not found in active triggers")
                return
            
            # Update success rates
            trigger_type = trigger.trigger_type
            if trigger_type not in self.trigger_success_rates:
                self.trigger_success_rates[trigger_type] = 0.8  # Default success rate
            
            # Exponential moving average for success rate
            current_rate = self.trigger_success_rates[trigger_type]
            new_rate = 0.9 * current_rate + 0.1 * (1.0 if success else 0.0)
            self.trigger_success_rates[trigger_type] = new_rate
            
            # Update processing time averages
            if trigger_type not in self.avg_processing_times:
                self.avg_processing_times[trigger_type] = processing_time_ms
            else:
                current_avg = self.avg_processing_times[trigger_type]
                new_avg = 0.9 * current_avg + 0.1 * processing_time_ms
                self.avg_processing_times[trigger_type] = new_avg
            
            # Store completion data in Redis for analytics
            completion_data = {
                "trigger_id": trigger_id,
                "trigger_type": trigger_type.value,
                "success": success,
                "processing_time_ms": processing_time_ms,
                "contexts_processed": contexts_processed,
                "expected_processing_time_ms": trigger.expected_processing_time_ms,
                "expected_context_count": trigger.context_count_estimate,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.lpush(
                "consolidation_trigger_completions",
                json.dumps(completion_data, default=str)
            )
            
            logger.info(
                f"Completed trigger {trigger_id} - Success: {success}, "
                f"Time: {processing_time_ms:.0f}ms, Contexts: {contexts_processed}"
            )
            
        except Exception as e:
            logger.error(f"Error completing trigger {trigger_id}: {e}")
    
    async def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trigger statistics."""
        try:
            stats = {
                "active_triggers": len(self.active_triggers),
                "total_triggers_generated": len(self.trigger_history),
                "trigger_success_rates": {
                    t.value: rate for t, rate in self.trigger_success_rates.items()
                },
                "avg_processing_times_ms": {
                    t.value: time_ms for t, time_ms in self.avg_processing_times.items()
                },
                "agents_monitored": len(self.agent_patterns),
                "is_monitoring": self._is_running
            }
            
            # Recent trigger breakdown
            recent_triggers = list(self.trigger_history)[-100:]  # Last 100 triggers
            trigger_counts = defaultdict(int)
            for trigger in recent_triggers:
                trigger_counts[trigger.trigger_type.value] += 1
            
            stats["recent_trigger_distribution"] = dict(trigger_counts)
            
            # Agent-specific statistics
            agent_stats = {}
            for agent_id, pattern in self.agent_patterns.items():
                agent_stats[str(agent_id)] = {
                    "contexts_created_per_hour": pattern.contexts_created_per_hour,
                    "contexts_accessed_per_hour": pattern.contexts_accessed_per_hour,
                    "current_unconsolidated_count": pattern.current_unconsolidated_count,
                    "memory_usage_mb": pattern.memory_usage_mb,
                    "is_active": pattern.is_active,
                    "last_consolidation": pattern.last_consolidation.isoformat() if pattern.last_consolidation else None
                }
            
            stats["agent_patterns"] = agent_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting trigger statistics: {e}")
            return {"error": str(e)}
    
    # Private Methods
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for trigger checking."""
        logger.info("Starting consolidation trigger monitoring loop")
        
        try:
            while self._is_running:
                try:
                    # Check for new triggers
                    triggers = await self.check_all_triggers()
                    
                    if triggers:
                        logger.info(f"Monitoring generated {len(triggers)} new triggers")
                    
                    # Wait before next check (configurable interval)
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Consolidation trigger monitoring loop stopped")
    
    async def _update_agent_patterns(self) -> None:
        """Update usage patterns for all agents."""
        try:
            async with get_async_session() as session:
                # Get all active agents
                agents_result = await session.execute(select(Agent.id))
                agent_ids = [row[0] for row in agents_result.all()]
                
                # Update patterns for each agent
                for agent_id in agent_ids:
                    pattern = await self._get_agent_pattern(agent_id)
                    if pattern:
                        self.agent_patterns[agent_id] = pattern
                        
        except Exception as e:
            logger.error(f"Error updating agent patterns: {e}")
    
    async def _get_agent_pattern(self, agent_id: UUID) -> Optional[AgentUsagePattern]:
        """Get usage pattern for specific agent."""
        try:
            async with get_async_session() as session:
                # Time windows for analysis
                now = datetime.utcnow()
                one_hour_ago = now - timedelta(hours=1)
                one_day_ago = now - timedelta(days=1)
                
                # Context creation rate
                contexts_created_hour = await session.scalar(
                    select(func.count(Context.id)).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.created_at >= one_hour_ago
                        )
                    )
                ) or 0
                
                # Context access rate
                contexts_accessed_hour = await session.scalar(
                    select(func.count(Context.id)).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.accessed_at >= one_hour_ago
                        )
                    )
                ) or 0
                
                # Current unconsolidated count
                unconsolidated_count = await session.scalar(
                    select(func.count(Context.id)).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.is_consolidated == "false"
                        )
                    )
                ) or 0
                
                # Last consolidation time
                last_consolidated_context = await session.scalar(
                    select(Context.consolidated_at).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.is_consolidated == "true"
                        )
                    ).order_by(desc(Context.consolidated_at))
                )
                
                # Estimate memory usage (simplified)
                estimated_memory_mb = unconsolidated_count * 0.1  # Rough estimate
                
                return AgentUsagePattern(
                    agent_id=agent_id,
                    contexts_created_per_hour=float(contexts_created_hour),
                    contexts_accessed_per_hour=float(contexts_accessed_hour),
                    avg_session_duration_minutes=60.0,  # Placeholder
                    peak_activity_hours=[9, 10, 11, 14, 15, 16],  # Business hours
                    consolidation_frequency_hours=6.0,
                    last_consolidation=last_consolidated_context,
                    current_unconsolidated_count=unconsolidated_count,
                    memory_usage_mb=estimated_memory_mb,
                    is_active=(contexts_created_hour > 0 or contexts_accessed_hour > 0)
                )
                
        except Exception as e:
            logger.error(f"Error getting agent pattern for {agent_id}: {e}")
            return None
    
    async def _check_usage_triggers(
        self,
        agent_id: UUID,
        pattern: AgentUsagePattern
    ) -> List[ConsolidationTrigger]:
        """Check usage-based consolidation triggers."""
        triggers = []
        
        try:
            # Check if unconsolidated context count exceeds threshold
            if pattern.current_unconsolidated_count >= self.usage_threshold_contexts:
                trigger = ConsolidationTrigger(
                    trigger_id=f"usage_{agent_id}_{int(datetime.utcnow().timestamp())}",
                    trigger_type=TriggerType.USAGE_THRESHOLD,
                    priority=TriggerPriority.MEDIUM,
                    agent_id=agent_id,
                    triggered_at=datetime.utcnow(),
                    expected_processing_time_ms=pattern.current_unconsolidated_count * 150,
                    context_count_estimate=pattern.current_unconsolidated_count,
                    memory_pressure_mb=pattern.memory_usage_mb,
                    trigger_metadata={
                        "threshold_contexts": self.usage_threshold_contexts,
                        "actual_contexts": pattern.current_unconsolidated_count,
                        "contexts_created_per_hour": pattern.contexts_created_per_hour
                    }
                )
                triggers.append(trigger)
                
        except Exception as e:
            logger.error(f"Error checking usage triggers for agent {agent_id}: {e}")
        
        return triggers
    
    async def _check_memory_triggers(
        self,
        agent_id: UUID,
        pattern: AgentUsagePattern
    ) -> List[ConsolidationTrigger]:
        """Check memory pressure triggers."""
        triggers = []
        
        try:
            # Check if memory usage exceeds threshold
            if pattern.memory_usage_mb >= self.memory_pressure_threshold_mb:
                trigger = ConsolidationTrigger(
                    trigger_id=f"memory_{agent_id}_{int(datetime.utcnow().timestamp())}",
                    trigger_type=TriggerType.MEMORY_PRESSURE,
                    priority=TriggerPriority.HIGH,
                    agent_id=agent_id,
                    triggered_at=datetime.utcnow(),
                    expected_processing_time_ms=pattern.current_unconsolidated_count * 200,
                    context_count_estimate=pattern.current_unconsolidated_count,
                    memory_pressure_mb=pattern.memory_usage_mb,
                    trigger_metadata={
                        "memory_threshold_mb": self.memory_pressure_threshold_mb,
                        "actual_memory_mb": pattern.memory_usage_mb,
                        "memory_pressure_ratio": pattern.memory_usage_mb / self.memory_pressure_threshold_mb
                    }
                )
                triggers.append(trigger)
                
        except Exception as e:
            logger.error(f"Error checking memory triggers for agent {agent_id}: {e}")
        
        return triggers
    
    async def _check_time_triggers(
        self,
        agent_id: UUID,
        pattern: AgentUsagePattern
    ) -> List[ConsolidationTrigger]:
        """Check time-based consolidation triggers."""
        triggers = []
        
        try:
            # Check if enough time has passed since last consolidation
            if pattern.last_consolidation:
                hours_since_consolidation = (
                    datetime.utcnow() - pattern.last_consolidation
                ).total_seconds() / 3600
                
                if (hours_since_consolidation >= self.max_consolidation_age_hours and
                    pattern.current_unconsolidated_count >= 5):
                    
                    trigger = ConsolidationTrigger(
                        trigger_id=f"time_{agent_id}_{int(datetime.utcnow().timestamp())}",
                        trigger_type=TriggerType.TIME_SCHEDULE,
                        priority=TriggerPriority.LOW,
                        agent_id=agent_id,
                        triggered_at=datetime.utcnow(),
                        expected_processing_time_ms=pattern.current_unconsolidated_count * 120,
                        context_count_estimate=pattern.current_unconsolidated_count,
                        memory_pressure_mb=pattern.memory_usage_mb,
                        trigger_metadata={
                            "max_age_hours": self.max_consolidation_age_hours,
                            "hours_since_last": hours_since_consolidation,
                            "last_consolidation": pattern.last_consolidation.isoformat()
                        }
                    )
                    triggers.append(trigger)
            
            elif pattern.current_unconsolidated_count >= 10:
                # No previous consolidation but enough contexts
                trigger = ConsolidationTrigger(
                    trigger_id=f"initial_{agent_id}_{int(datetime.utcnow().timestamp())}",
                    trigger_type=TriggerType.TIME_SCHEDULE,
                    priority=TriggerPriority.MEDIUM,
                    agent_id=agent_id,
                    triggered_at=datetime.utcnow(),
                    expected_processing_time_ms=pattern.current_unconsolidated_count * 120,
                    context_count_estimate=pattern.current_unconsolidated_count,
                    memory_pressure_mb=pattern.memory_usage_mb,
                    trigger_metadata={
                        "initial_consolidation": True,
                        "context_count": pattern.current_unconsolidated_count
                    }
                )
                triggers.append(trigger)
                
        except Exception as e:
            logger.error(f"Error checking time triggers for agent {agent_id}: {e}")
        
        return triggers
    
    async def _check_activity_triggers(
        self,
        agent_id: UUID,
        pattern: AgentUsagePattern
    ) -> List[ConsolidationTrigger]:
        """Check activity-based consolidation triggers."""
        triggers = []
        
        try:
            # Check if agent is very active and could benefit from consolidation
            if (pattern.contexts_created_per_hour >= 5 and
                pattern.current_unconsolidated_count >= 20):
                
                trigger = ConsolidationTrigger(
                    trigger_id=f"activity_{agent_id}_{int(datetime.utcnow().timestamp())}",
                    trigger_type=TriggerType.AGENT_ACTIVITY,
                    priority=TriggerPriority.MEDIUM,
                    agent_id=agent_id,
                    triggered_at=datetime.utcnow(),
                    expected_processing_time_ms=pattern.current_unconsolidated_count * 180,
                    context_count_estimate=pattern.current_unconsolidated_count,
                    memory_pressure_mb=pattern.memory_usage_mb,
                    trigger_metadata={
                        "contexts_created_per_hour": pattern.contexts_created_per_hour,
                        "contexts_accessed_per_hour": pattern.contexts_accessed_per_hour,
                        "high_activity_consolidation": True
                    }
                )
                triggers.append(trigger)
                
        except Exception as e:
            logger.error(f"Error checking activity triggers for agent {agent_id}: {e}")
        
        return triggers
    
    async def _check_emergency_triggers(
        self,
        agent_id: UUID,
        pattern: AgentUsagePattern
    ) -> List[ConsolidationTrigger]:
        """Check emergency consolidation triggers."""
        triggers = []
        
        try:
            # Check if context count is extremely high (emergency consolidation)
            if pattern.current_unconsolidated_count >= self.emergency_threshold_contexts:
                trigger = ConsolidationTrigger(
                    trigger_id=f"emergency_{agent_id}_{int(datetime.utcnow().timestamp())}",
                    trigger_type=TriggerType.EMERGENCY,
                    priority=TriggerPriority.CRITICAL,
                    agent_id=agent_id,
                    triggered_at=datetime.utcnow(),
                    expected_processing_time_ms=pattern.current_unconsolidated_count * 100,  # Faster emergency processing
                    context_count_estimate=pattern.current_unconsolidated_count,
                    memory_pressure_mb=pattern.memory_usage_mb,
                    trigger_metadata={
                        "emergency_threshold": self.emergency_threshold_contexts,
                        "actual_count": pattern.current_unconsolidated_count,
                        "emergency_consolidation": True
                    }
                )
                triggers.append(trigger)
                
        except Exception as e:
            logger.error(f"Error checking emergency triggers for agent {agent_id}: {e}")
        
        return triggers


# Global instance for application use
_trigger_manager: Optional[ConsolidationTriggerManager] = None


def get_consolidation_trigger_manager() -> ConsolidationTriggerManager:
    """
    Get singleton consolidation trigger manager instance.
    
    Returns:
        ConsolidationTriggerManager instance
    """
    global _trigger_manager
    
    if _trigger_manager is None:
        _trigger_manager = ConsolidationTriggerManager()
    
    return _trigger_manager


async def start_consolidation_monitoring() -> None:
    """Start consolidation trigger monitoring."""
    trigger_manager = get_consolidation_trigger_manager()
    await trigger_manager.start_monitoring()


async def stop_consolidation_monitoring() -> None:
    """Stop consolidation trigger monitoring."""
    global _trigger_manager
    
    if _trigger_manager:
        await _trigger_manager.stop_monitoring()
        _trigger_manager = None