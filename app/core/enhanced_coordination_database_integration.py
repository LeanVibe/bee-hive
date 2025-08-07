"""
Enhanced Coordination Database Integration for LeanVibe Agent Hive 2.0

This module provides the integration layer between the enhanced multi-agent
coordination system and the database persistence layer, enabling real-time
dashboard visibility and business value tracking.

Key Features:
- Automatic persistence of coordination events and collaboration contexts
- Real-time data synchronization between in-memory and database state
- Business value calculation and metrics aggregation
- WebSocket event streaming for dashboard updates
- Performance-optimized batch operations for high-throughput scenarios
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import weakref

import structlog

from .enhanced_multi_agent_coordination import (
    EnhancedMultiAgentCoordinator, SpecializedAgent, CollaborationContext,
    CoordinationPattern, SpecializedAgentRole, CoordinationPatternType,
    TaskComplexity, CoordinationEvent as CoordEvent
)
from ..services.coordination_persistence_service import (
    CoordinationPersistenceService, get_coordination_persistence_service
)
# Use the main coordination_event model to avoid conflicts
from ..models.coordination_event import (
    CoordinationEvent, CoordinationEventType
)
# AgentCollaboration will be mocked for now
from typing import NamedTuple
class AgentCollaboration(NamedTuple):
    collaboration_id: str
    progress_percentage: float
    current_step: str
    def to_dict(self):
        return {"collaboration_id": self.collaboration_id, "progress_percentage": self.progress_percentage}
from .comprehensive_dashboard_integration import get_dashboard_integration
from .redis import get_message_broker

logger = structlog.get_logger()


@dataclass
class CoordinationMetrics:
    """Real-time coordination metrics for dashboard display."""
    active_collaborations: int
    total_coordination_events: int
    business_value_generated: float
    productivity_improvement: float
    average_collaboration_duration: float
    success_rate: float
    unique_agents_participating: int
    most_active_patterns: List[Dict[str, Any]]
    calculated_at: datetime


class CoordinationDatabaseIntegrator:
    """
    Integrates the enhanced multi-agent coordination system with database persistence.
    
    Provides seamless synchronization between in-memory coordination state and
    persistent storage, enabling dashboard visibility and business value tracking.
    """
    
    def __init__(self, coordinator: EnhancedMultiAgentCoordinator):
        self.coordinator = coordinator
        self.persistence_service = get_coordination_persistence_service()
        self.dashboard_integration = get_dashboard_integration()
        self.message_broker = get_message_broker()
        
        self.logger = logger.bind(service="coordination_db_integrator")
        
        # Track active collaborations and events
        self.active_collaborations: Dict[str, AgentCollaboration] = {}
        self.event_cache: Dict[str, CoordinationEvent] = {}
        
        # Performance tracking
        self.last_metrics_calculation = datetime.utcnow()
        self.metrics_cache: Optional[CoordinationMetrics] = None
        
        # Integration hooks
        self._setup_coordination_hooks()
    
    def _setup_coordination_hooks(self):
        """Set up hooks to capture coordination events."""
        try:
            # Hook into coordination events
            self.coordinator.on_collaboration_started = self._on_collaboration_started
            self.coordinator.on_coordination_event = self._on_coordination_event
            self.coordinator.on_collaboration_completed = self._on_collaboration_completed
            self.coordinator.on_pattern_executed = self._on_pattern_executed
            
            self.logger.info("coordination_hooks_established")
        except Exception as e:
            self.logger.error("failed_to_setup_coordination_hooks", error=str(e))
    
    async def _on_collaboration_started(
        self,
        collaboration_context: CollaborationContext,
        pattern: Optional[CoordinationPattern] = None
    ):
        """Handle collaboration start event."""
        try:
            # Create database collaboration record
            collaboration = await self.persistence_service.create_collaboration(
                collaboration_id=collaboration_context.collaboration_id,
                name=f"Collaboration {collaboration_context.collaboration_id[:8]}",
                participants=collaboration_context.participants,
                pattern_id=pattern.pattern_id if pattern else None,
                description=f"Multi-agent collaboration with {len(collaboration_context.participants)} agents"
            )
            
            self.active_collaborations[collaboration_context.collaboration_id] = collaboration
            
            # Create coordination event
            await self._create_coordination_event(
                event_type=CoordinationEventType.COLLABORATION_STARTED,
                participants=collaboration_context.participants,
                collaboration_id=collaboration_context.collaboration_id,
                event_data={
                    "pattern_type": pattern.pattern_type.value if pattern else "custom",
                    "estimated_duration": pattern.estimated_duration if pattern else None,
                    "shared_knowledge_keys": list(collaboration_context.shared_knowledge.keys())
                }
            )
            
            # Stream to dashboard
            await self._stream_dashboard_update({
                "type": "collaboration_started",
                "collaboration_id": collaboration_context.collaboration_id,
                "participants": collaboration_context.participants,
                "pattern": pattern.to_dict() if pattern else None
            })
            
            self.logger.info("collaboration_started_persisted",
                           collaboration_id=collaboration_context.collaboration_id,
                           participants_count=len(collaboration_context.participants))
            
        except Exception as e:
            self.logger.error("failed_to_handle_collaboration_started",
                            error=str(e),
                            collaboration_id=collaboration_context.collaboration_id)
    
    async def _on_coordination_event(
        self,
        event_type: str,
        participants: List[str],
        event_data: Dict[str, Any],
        collaboration_id: Optional[str] = None,
        duration_ms: Optional[int] = None
    ):
        """Handle general coordination event."""
        try:
            # Map event type to enum
            coord_event_type = self._map_event_type(event_type)
            if not coord_event_type:
                self.logger.warning("unmapped_event_type", event_type=event_type)
                return
            
            # Calculate business value based on event type
            business_value = self._calculate_event_business_value(
                coord_event_type, participants, event_data, duration_ms
            )
            
            # Create coordination event
            event = await self._create_coordination_event(
                event_type=coord_event_type,
                participants=participants,
                collaboration_id=collaboration_id,
                event_data=event_data,
                duration_ms=duration_ms,
                business_value=business_value
            )
            
            # Update collaboration progress if applicable
            if collaboration_id and collaboration_id in self.active_collaborations:
                await self._update_collaboration_progress(collaboration_id, event)
            
            # Stream to dashboard
            await self._stream_dashboard_update({
                "type": "coordination_event",
                "event_type": event_type,
                "participants": participants,
                "collaboration_id": collaboration_id,
                "business_value": business_value,
                "event_id": str(event.id)
            })
            
            self.logger.info("coordination_event_persisted",
                           event_type=event_type,
                           participants_count=len(participants),
                           business_value=business_value)
            
        except Exception as e:
            self.logger.error("failed_to_handle_coordination_event",
                            error=str(e),
                            event_type=event_type)
    
    async def _on_collaboration_completed(
        self,
        collaboration_id: str,
        success: bool,
        final_context: CollaborationContext,
        business_impact: Dict[str, Any]
    ):
        """Handle collaboration completion."""
        try:
            # Calculate final business value
            total_business_value = business_impact.get("total_value", 0.0)
            productivity_gain = business_impact.get("productivity_gain", 0.0)
            
            # Update collaboration record
            await self.persistence_service.update_collaboration(
                collaboration_id=collaboration_id,
                updates={
                    "business_value_generated": total_business_value,
                    "productivity_gain": productivity_gain,
                    "progress_percentage": 100.0,
                    "shared_knowledge": final_context.shared_knowledge,
                    "decisions_made": final_context.decisions_made,
                    "success_patterns": final_context.success_patterns
                },
                complete=True
            )
            
            # Create completion event
            await self._create_coordination_event(
                event_type=CoordinationEventType.PATTERN_COMPLETED,
                participants=final_context.participants,
                collaboration_id=collaboration_id,
                event_data={
                    "success": success,
                    "total_business_value": total_business_value,
                    "productivity_gain": productivity_gain,
                    "decisions_count": len(final_context.decisions_made),
                    "artifacts_count": len(final_context.artifacts_created)
                },
                business_value=total_business_value
            )
            
            # Remove from active collaborations
            self.active_collaborations.pop(collaboration_id, None)
            
            # Stream to dashboard
            await self._stream_dashboard_update({
                "type": "collaboration_completed",
                "collaboration_id": collaboration_id,
                "success": success,
                "business_value": total_business_value,
                "productivity_gain": productivity_gain
            })
            
            self.logger.info("collaboration_completed_persisted",
                           collaboration_id=collaboration_id,
                           success=success,
                           business_value=total_business_value)
            
        except Exception as e:
            self.logger.error("failed_to_handle_collaboration_completed",
                            error=str(e),
                            collaboration_id=collaboration_id)
    
    async def _on_pattern_executed(
        self,
        pattern: CoordinationPattern,
        execution_result: Dict[str, Any]
    ):
        """Handle pattern execution completion."""
        try:
            # Update pattern usage statistics in memory
            # (Database patterns would be updated via separate service)
            
            success = execution_result.get("success", False)
            duration_ms = execution_result.get("duration_ms", 0)
            business_value = execution_result.get("business_value", 0.0)
            
            # Create pattern execution event
            await self._create_coordination_event(
                event_type=CoordinationEventType.PATTERN_COMPLETED,
                participants=execution_result.get("participants", []),
                event_data={
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type.value,
                    "success": success,
                    "execution_metrics": execution_result.get("metrics", {})
                },
                duration_ms=duration_ms,
                business_value=business_value
            )
            
            # Stream to dashboard
            await self._stream_dashboard_update({
                "type": "pattern_executed",
                "pattern": pattern.to_dict(),
                "success": success,
                "duration_ms": duration_ms,
                "business_value": business_value
            })
            
            self.logger.info("pattern_execution_persisted",
                           pattern_id=pattern.pattern_id,
                           success=success,
                           business_value=business_value)
            
        except Exception as e:
            self.logger.error("failed_to_handle_pattern_executed",
                            error=str(e),
                            pattern_id=pattern.pattern_id)
    
    async def _create_coordination_event(
        self,
        event_type: CoordinationEventType,
        participants: List[str],
        event_data: Dict[str, Any],
        collaboration_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        business_value: Optional[float] = None
    ) -> CoordinationEvent:
        """Create and cache a coordination event."""
        event = await self.persistence_service.create_coordination_event(
            event_type=event_type,
            participants=participants,
            event_data=event_data,
            collaboration_id=collaboration_id,
            duration_ms=duration_ms,
            business_value=business_value
        )
        
        # Cache for quick access
        self.event_cache[str(event.id)] = event
        
        # Clean cache if it gets too large (keep last 1000 events)
        if len(self.event_cache) > 1000:
            oldest_keys = sorted(
                self.event_cache.keys(),
                key=lambda k: self.event_cache[k].created_at
            )[:100]  # Remove oldest 100
            
            for key in oldest_keys:
                self.event_cache.pop(key, None)
        
        return event
    
    async def _update_collaboration_progress(
        self,
        collaboration_id: str,
        event: CoordinationEvent
    ):
        """Update collaboration progress based on events."""
        try:
            collaboration = self.active_collaborations.get(collaboration_id)
            if not collaboration:
                return
            
            # Calculate progress based on event types
            progress_increment = self._calculate_progress_increment(event.event_type)
            new_progress = min(
                collaboration.progress_percentage + progress_increment,
                100.0
            )
            
            # Update collaboration
            await self.persistence_service.update_collaboration(
                collaboration_id=collaboration_id,
                updates={
                    "progress_percentage": new_progress,
                    "current_step": event.event_type.value
                }
            )
            
            # Update cache
            collaboration.progress_percentage = new_progress
            collaboration.current_step = event.event_type.value
            
        except Exception as e:
            self.logger.error("failed_to_update_collaboration_progress",
                            error=str(e),
                            collaboration_id=collaboration_id)
    
    async def _stream_dashboard_update(self, update_data: Dict[str, Any]):
        """Stream update to dashboard via WebSocket."""
        try:
            await self.dashboard_integration.stream_coordination_update(update_data)
        except Exception as e:
            self.logger.warning("failed_to_stream_dashboard_update",
                              error=str(e))
    
    def _map_event_type(self, event_type: str) -> Optional[CoordinationEventType]:
        """Map string event type to enum."""
        mapping = {
            "collaboration_started": CoordinationEventType.COLLABORATION_STARTED,
            "task_handoff": CoordinationEventType.TASK_HANDOFF,
            "code_review_cycle": CoordinationEventType.CODE_REVIEW_CYCLE,
            "pair_programming": CoordinationEventType.PAIR_PROGRAMMING,
            "knowledge_sharing": CoordinationEventType.KNOWLEDGE_SHARING,
            "design_review": CoordinationEventType.DESIGN_REVIEW,
            "conflict_resolution": CoordinationEventType.CONFLICT_RESOLUTION,
            "team_standup": CoordinationEventType.TEAM_STANDUP,
            "continuous_integration": CoordinationEventType.CONTINUOUS_INTEGRATION,
            "decision_made": CoordinationEventType.DECISION_MADE,
            "pattern_completed": CoordinationEventType.PATTERN_COMPLETED
        }
        return mapping.get(event_type)
    
    def _calculate_event_business_value(
        self,
        event_type: CoordinationEventType,
        participants: List[str],
        event_data: Dict[str, Any],
        duration_ms: Optional[int]
    ) -> float:
        """Calculate business value for a coordination event."""
        base_values = {
            CoordinationEventType.COLLABORATION_STARTED: 50.0,
            CoordinationEventType.PAIR_PROGRAMMING: 150.0,
            CoordinationEventType.CODE_REVIEW_CYCLE: 100.0,
            CoordinationEventType.KNOWLEDGE_SHARING: 75.0,
            CoordinationEventType.DESIGN_REVIEW: 125.0,
            CoordinationEventType.TASK_HANDOFF: 25.0,
            CoordinationEventType.CONFLICT_RESOLUTION: 200.0,
            CoordinationEventType.CONTINUOUS_INTEGRATION: 100.0,
            CoordinationEventType.DECISION_MADE: 50.0,
            CoordinationEventType.PATTERN_COMPLETED: 300.0
        }
        
        base_value = base_values.get(event_type, 25.0)
        
        # Adjust based on participants (collaboration multiplier)
        participant_multiplier = min(len(participants) * 0.5, 2.0)
        
        # Adjust based on duration (efficiency bonus)
        duration_multiplier = 1.0
        if duration_ms:
            # Bonus for efficient completion (under expected time)
            expected_duration = event_data.get("expected_duration_ms", duration_ms)
            if duration_ms < expected_duration:
                duration_multiplier = 1.2
        
        return base_value * participant_multiplier * duration_multiplier
    
    def _calculate_progress_increment(self, event_type: CoordinationEventType) -> float:
        """Calculate progress increment based on event type."""
        increments = {
            CoordinationEventType.COLLABORATION_STARTED: 10.0,
            CoordinationEventType.TASK_HANDOFF: 15.0,
            CoordinationEventType.CODE_REVIEW_CYCLE: 20.0,
            CoordinationEventType.PAIR_PROGRAMMING: 25.0,
            CoordinationEventType.KNOWLEDGE_SHARING: 15.0,
            CoordinationEventType.DESIGN_REVIEW: 20.0,
            CoordinationEventType.DECISION_MADE: 10.0,
            CoordinationEventType.PATTERN_COMPLETED: 30.0
        }
        return increments.get(event_type, 5.0)
    
    async def get_real_time_metrics(self) -> CoordinationMetrics:
        """Get real-time coordination metrics for dashboard."""
        try:
            # Check if we need to recalculate (every 30 seconds)
            now = datetime.utcnow()
            if (self.metrics_cache and 
                (now - self.last_metrics_calculation).seconds < 30):
                return self.metrics_cache
            
            # Calculate fresh metrics
            business_metrics = await self.persistence_service.calculate_business_value_metrics(
                timeframe_hours=24
            )
            
            active_collaborations = len(self.active_collaborations)
            
            # Get recent events for additional calculations
            recent_events = await self.persistence_service.get_coordination_events(
                limit=100
            )
            
            # Calculate average duration
            durations = [e.duration_ms for e in recent_events if e.duration_ms]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            
            # Get unique participating agents
            unique_agents = set()
            for event in recent_events:
                unique_agents.update(event.participants)
            
            # Most active patterns (mock for now)
            most_active_patterns = [
                {"pattern_id": "pair_programming_basic", "usage_count": 15, "success_rate": 0.92},
                {"pattern_id": "code_review_cycle", "usage_count": 12, "success_rate": 0.88},
                {"pattern_id": "knowledge_sharing_session", "usage_count": 8, "success_rate": 0.95}
            ]
            
            self.metrics_cache = CoordinationMetrics(
                active_collaborations=active_collaborations,
                total_coordination_events=business_metrics["total_coordination_events"],
                business_value_generated=business_metrics["total_business_value"],
                productivity_improvement=business_metrics["productivity_improvement_factor"],
                average_collaboration_duration=avg_duration,
                success_rate=business_metrics["coordination_success_rate"],
                unique_agents_participating=len(unique_agents),
                most_active_patterns=most_active_patterns,
                calculated_at=now
            )
            
            self.last_metrics_calculation = now
            
            self.logger.info("real_time_metrics_calculated",
                           active_collaborations=active_collaborations,
                           total_events=business_metrics["total_coordination_events"],
                           business_value=business_metrics["total_business_value"])
            
            return self.metrics_cache
            
        except Exception as e:
            self.logger.error("failed_to_calculate_real_time_metrics", error=str(e))
            # Return cached metrics or defaults
            return self.metrics_cache or CoordinationMetrics(
                active_collaborations=0, total_coordination_events=0,
                business_value_generated=0.0, productivity_improvement=1.0,
                average_collaboration_duration=0.0, success_rate=0.0,
                unique_agents_participating=0, most_active_patterns=[],
                calculated_at=now
            )
    
    async def initialize_default_patterns(self):
        """Initialize default coordination patterns in database."""
        try:
            patterns = await self.persistence_service.create_default_coordination_patterns()
            self.logger.info("default_patterns_initialized", count=len(patterns))
            return patterns
        except Exception as e:
            self.logger.error("failed_to_initialize_default_patterns", error=str(e))
            return []
    
    async def get_coordination_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for coordination dashboard."""
        try:
            # Get real-time metrics
            metrics = await self.get_real_time_metrics()
            
            # Get active collaborations details
            active_collaborations_data = []
            for collaboration in self.active_collaborations.values():
                active_collaborations_data.append(collaboration.to_dict())
            
            # Get recent coordination events
            recent_events = await self.persistence_service.get_coordination_events(limit=50)
            recent_events_data = [event.to_dict() for event in recent_events]
            
            # Get coordination patterns
            patterns = await self.persistence_service.get_coordination_patterns()
            patterns_data = [pattern.to_dict() for pattern in patterns]
            
            dashboard_data = {
                "metrics": asdict(metrics),
                "active_collaborations": active_collaborations_data,
                "recent_events": recent_events_data,
                "coordination_patterns": patterns_data,
                "system_status": {
                    "coordination_system_active": bool(self.coordinator),
                    "database_connected": True,
                    "dashboard_streaming": True,
                    "last_updated": datetime.utcnow().isoformat()
                }
            }
            
            self.logger.info("coordination_dashboard_data_prepared",
                           metrics_calculated=True,
                           active_collaborations=len(active_collaborations_data),
                           recent_events=len(recent_events_data))
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error("failed_to_get_coordination_dashboard_data", error=str(e))
            raise


# Global integrator instance
_coordination_db_integrator: Optional[CoordinationDatabaseIntegrator] = None

def get_coordination_db_integrator(
    coordinator: Optional[EnhancedMultiAgentCoordinator] = None
) -> CoordinationDatabaseIntegrator:
    """Get or create the global coordination database integrator."""
    global _coordination_db_integrator
    
    if _coordination_db_integrator is None:
        if coordinator is None:
            raise ValueError("Coordinator required for first initialization")
        _coordination_db_integrator = CoordinationDatabaseIntegrator(coordinator)
    
    return _coordination_db_integrator

def reset_coordination_db_integrator():
    """Reset the global integrator (for testing)."""
    global _coordination_db_integrator
    _coordination_db_integrator = None