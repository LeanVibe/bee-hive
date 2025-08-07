"""
Coordination Persistence Service for LeanVibe Agent Hive 2.0

Handles database persistence for the enhanced multi-agent coordination system,
providing seamless integration between in-memory coordination state and
persistent storage for dashboard visibility and business value tracking.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager

import structlog
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_

from ..core.database import get_async_session
from ..models.coordination import (
    CoordinationEvent, CoordinationPattern, AgentCollaboration,
    CoordinationEventType, CoordinationPatternType, SpecializedAgentRole, TaskComplexity
)
from ..models.session import Session, SessionStatus
from ..models.agent import Agent, AgentStatus

logger = structlog.get_logger()


class CoordinationPersistenceService:
    """
    Service for persisting and retrieving coordination data.
    
    Provides high-level methods for saving coordination events, managing
    collaboration contexts, and querying business value metrics.
    """
    
    def __init__(self):
        self.logger = logger.bind(service="coordination_persistence")
    
    async def create_coordination_event(
        self,
        event_type: CoordinationEventType,
        participants: List[str],
        event_data: Dict[str, Any],
        session_id: Optional[str] = None,
        collaboration_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        business_value: Optional[float] = None
    ) -> CoordinationEvent:
        """Create and persist a coordination event."""
        async with get_async_session() as session:
            try:
                event = CoordinationEvent(
                    event_type=event_type,
                    participants=participants,
                    session_id=uuid.UUID(session_id) if session_id else None,
                    collaboration_id=collaboration_id,
                    primary_agent_id=participants[0] if participants else None,
                    event_data=event_data,
                    duration_ms=duration_ms,
                    business_value=business_value,
                    status="active"
                )
                
                session.add(event)
                await session.commit()
                await session.refresh(event)
                
                self.logger.info("coordination_event_created", 
                               event_id=str(event.id), 
                               event_type=event_type.value,
                               participants_count=len(participants))
                
                return event
                
            except Exception as e:
                await session.rollback()
                self.logger.error("failed_to_create_coordination_event", 
                                error=str(e), event_type=event_type.value)
                raise
    
    async def update_coordination_event(
        self,
        event_id: str,
        updates: Dict[str, Any],
        complete: bool = False
    ) -> Optional[CoordinationEvent]:
        """Update an existing coordination event."""
        async with get_async_session() as session:
            try:
                event = await session.get(CoordinationEvent, uuid.UUID(event_id))
                if not event:
                    return None
                
                # Apply updates
                for key, value in updates.items():
                    if hasattr(event, key):
                        setattr(event, key, value)
                
                if complete:
                    event.completed_at = datetime.utcnow()
                    event.status = "completed"
                
                await session.commit()
                await session.refresh(event)
                
                self.logger.info("coordination_event_updated", 
                               event_id=event_id, updates_count=len(updates))
                
                return event
                
            except Exception as e:
                await session.rollback()
                self.logger.error("failed_to_update_coordination_event", 
                                error=str(e), event_id=event_id)
                raise
    
    async def create_collaboration(
        self,
        collaboration_id: str,
        name: str,
        participants: List[str],
        pattern_id: Optional[str] = None,
        session_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> AgentCollaboration:
        """Create and persist an agent collaboration."""
        async with get_async_session() as session:
            try:
                collaboration = AgentCollaboration(
                    collaboration_id=collaboration_id,
                    name=name,
                    description=description,
                    participants=participants,
                    pattern_id=pattern_id,
                    session_id=uuid.UUID(session_id) if session_id else None,
                    primary_agent_id=participants[0] if participants else None,
                    status="active"
                )
                
                session.add(collaboration)
                await session.commit()
                await session.refresh(collaboration)
                
                self.logger.info("collaboration_created", 
                               collaboration_id=collaboration_id,
                               participants_count=len(participants))
                
                return collaboration
                
            except Exception as e:
                await session.rollback()
                self.logger.error("failed_to_create_collaboration", 
                                error=str(e), collaboration_id=collaboration_id)
                raise
    
    async def update_collaboration(
        self,
        collaboration_id: str,
        updates: Dict[str, Any],
        complete: bool = False
    ) -> Optional[AgentCollaboration]:
        """Update an existing collaboration."""
        async with get_async_session() as session:
            try:
                stmt = select(AgentCollaboration).where(
                    AgentCollaboration.collaboration_id == collaboration_id
                )
                result = await session.execute(stmt)
                collaboration = result.scalar_one_or_none()
                
                if not collaboration:
                    return None
                
                # Apply updates
                for key, value in updates.items():
                    if hasattr(collaboration, key):
                        setattr(collaboration, key, value)
                
                if complete:
                    collaboration.actual_completion = datetime.utcnow()
                    collaboration.status = "completed"
                
                await session.commit()
                await session.refresh(collaboration)
                
                self.logger.info("collaboration_updated", 
                               collaboration_id=collaboration_id)
                
                return collaboration
                
            except Exception as e:
                await session.rollback()
                self.logger.error("failed_to_update_collaboration", 
                                error=str(e), collaboration_id=collaboration_id)
                raise
    
    async def get_active_collaborations(
        self, 
        session_id: Optional[str] = None
    ) -> List[AgentCollaboration]:
        """Get all active collaborations, optionally filtered by session."""
        async with get_async_session() as session:
            try:
                stmt = select(AgentCollaboration).where(
                    AgentCollaboration.status == "active"
                )
                
                if session_id:
                    stmt = stmt.where(AgentCollaboration.session_id == uuid.UUID(session_id))
                
                result = await session.execute(stmt)
                collaborations = result.scalars().all()
                
                self.logger.info("active_collaborations_retrieved", 
                               count=len(collaborations),
                               session_id=session_id)
                
                return list(collaborations)
                
            except Exception as e:
                self.logger.error("failed_to_get_active_collaborations", 
                                error=str(e), session_id=session_id)
                raise
    
    async def get_coordination_events(
        self,
        session_id: Optional[str] = None,
        collaboration_id: Optional[str] = None,
        event_type: Optional[CoordinationEventType] = None,
        limit: int = 100
    ) -> List[CoordinationEvent]:
        """Get coordination events with optional filtering."""
        async with get_async_session() as session:
            try:
                stmt = select(CoordinationEvent).order_by(
                    CoordinationEvent.created_at.desc()
                ).limit(limit)
                
                if session_id:
                    stmt = stmt.where(CoordinationEvent.session_id == uuid.UUID(session_id))
                
                if collaboration_id:
                    stmt = stmt.where(CoordinationEvent.collaboration_id == collaboration_id)
                
                if event_type:
                    stmt = stmt.where(CoordinationEvent.event_type == event_type)
                
                result = await session.execute(stmt)
                events = result.scalars().all()
                
                self.logger.info("coordination_events_retrieved", 
                               count=len(events),
                               session_id=session_id,
                               collaboration_id=collaboration_id)
                
                return list(events)
                
            except Exception as e:
                self.logger.error("failed_to_get_coordination_events", 
                                error=str(e))
                raise
    
    async def calculate_business_value_metrics(
        self,
        timeframe_hours: int = 24,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate business value metrics from coordination data."""
        async with get_async_session() as session:
            try:
                # Calculate time range
                start_time = datetime.utcnow() - timedelta(hours=timeframe_hours)
                
                # Base query
                base_query = select(CoordinationEvent).where(
                    CoordinationEvent.created_at >= start_time
                )
                
                if session_id:
                    base_query = base_query.where(
                        CoordinationEvent.session_id == uuid.UUID(session_id)
                    )
                
                # Get all events in timeframe
                result = await session.execute(base_query)
                events = result.scalars().all()
                
                # Calculate metrics
                total_events = len(events)
                total_business_value = sum(
                    event.business_value or 0.0 for event in events
                )
                
                # Productivity metrics
                collaboration_events = [
                    e for e in events 
                    if e.event_type in [
                        CoordinationEventType.PAIR_PROGRAMMING,
                        CoordinationEventType.CODE_REVIEW_CYCLE,
                        CoordinationEventType.KNOWLEDGE_SHARING
                    ]
                ]
                
                avg_collaboration_duration = 0.0
                if collaboration_events:
                    durations = [e.duration_ms for e in collaboration_events if e.duration_ms]
                    if durations:
                        avg_collaboration_duration = sum(durations) / len(durations)
                
                # Success rate
                completed_events = [e for e in events if e.status == "completed"]
                success_rate = len(completed_events) / total_events if total_events > 0 else 0.0
                
                metrics = {
                    "timeframe_hours": timeframe_hours,
                    "total_coordination_events": total_events,
                    "total_business_value": total_business_value,
                    "average_business_value_per_event": total_business_value / total_events if total_events > 0 else 0.0,
                    "collaboration_events_count": len(collaboration_events),
                    "average_collaboration_duration_ms": avg_collaboration_duration,
                    "coordination_success_rate": success_rate,
                    "productivity_improvement_factor": min(success_rate * 2.0, 3.4),  # Cap at 340%
                    "estimated_annual_value": total_business_value * (365 * 24 / timeframe_hours) if timeframe_hours > 0 else 0.0,
                    "active_agents": len(set(
                        agent_id 
                        for event in events 
                        for agent_id in event.participants
                    )),
                    "event_types_distribution": self._calculate_event_type_distribution(events),
                    "calculated_at": datetime.utcnow().isoformat()
                }
                
                self.logger.info("business_value_metrics_calculated", 
                               timeframe_hours=timeframe_hours,
                               total_events=total_events,
                               total_value=total_business_value)
                
                return metrics
                
            except Exception as e:
                self.logger.error("failed_to_calculate_business_value_metrics", 
                                error=str(e))
                raise
    
    async def get_agent_coordination_summary(
        self, 
        agent_id: str,
        timeframe_hours: int = 24
    ) -> Dict[str, Any]:
        """Get coordination summary for a specific agent."""
        async with get_async_session() as session:
            try:
                start_time = datetime.utcnow() - timedelta(hours=timeframe_hours)
                
                # Get events where agent participated
                stmt = select(CoordinationEvent).where(
                    and_(
                        CoordinationEvent.created_at >= start_time,
                        func.array_position(CoordinationEvent.participants, agent_id).isnot(None)
                    )
                )
                
                result = await session.execute(stmt)
                events = result.scalars().all()
                
                # Calculate agent-specific metrics
                total_events = len(events)
                total_business_value = sum(event.business_value or 0.0 for event in events)
                
                collaboration_count = len([
                    e for e in events 
                    if e.event_type in [
                        CoordinationEventType.PAIR_PROGRAMMING,
                        CoordinationEventType.CODE_REVIEW_CYCLE
                    ]
                ])
                
                leadership_count = len([
                    e for e in events 
                    if e.primary_agent_id == agent_id
                ])
                
                unique_collaborators = set()
                for event in events:
                    unique_collaborators.update([
                        p for p in event.participants if p != agent_id
                    ])
                
                summary = {
                    "agent_id": agent_id,
                    "timeframe_hours": timeframe_hours,
                    "total_coordination_events": total_events,
                    "collaboration_events": collaboration_count,
                    "leadership_events": leadership_count,
                    "unique_collaborators": len(unique_collaborators),
                    "total_business_value_contributed": total_business_value,
                    "average_value_per_event": total_business_value / total_events if total_events > 0 else 0.0,
                    "collaboration_frequency": collaboration_count / timeframe_hours if timeframe_hours > 0 else 0.0,
                    "leadership_ratio": leadership_count / total_events if total_events > 0 else 0.0,
                    "calculated_at": datetime.utcnow().isoformat()
                }
                
                self.logger.info("agent_coordination_summary_calculated", 
                               agent_id=agent_id,
                               total_events=total_events,
                               collaborations=collaboration_count)
                
                return summary
                
            except Exception as e:
                self.logger.error("failed_to_get_agent_coordination_summary", 
                                error=str(e), agent_id=agent_id)
                raise
    
    def _calculate_event_type_distribution(self, events: List[CoordinationEvent]) -> Dict[str, int]:
        """Calculate distribution of event types."""
        distribution = {}
        for event in events:
            event_type = event.event_type.value
            distribution[event_type] = distribution.get(event_type, 0) + 1
        return distribution
    
    async def get_coordination_patterns(
        self, 
        active_only: bool = True
    ) -> List[CoordinationPattern]:
        """Get coordination patterns."""
        async with get_async_session() as session:
            try:
                stmt = select(CoordinationPattern)
                
                if active_only:
                    stmt = stmt.where(CoordinationPattern.is_active == "active")
                
                result = await session.execute(stmt)
                patterns = result.scalars().all()
                
                self.logger.info("coordination_patterns_retrieved", 
                               count=len(patterns), active_only=active_only)
                
                return list(patterns)
                
            except Exception as e:
                self.logger.error("failed_to_get_coordination_patterns", 
                                error=str(e))
                raise
    
    async def create_default_coordination_patterns(self) -> List[CoordinationPattern]:
        """Create default coordination patterns if they don't exist."""
        async with get_async_session() as session:
            try:
                # Check if patterns already exist
                stmt = select(func.count(CoordinationPattern.id))
                result = await session.execute(stmt)
                existing_count = result.scalar()
                
                if existing_count > 0:
                    self.logger.info("coordination_patterns_already_exist", count=existing_count)
                    return await self.get_coordination_patterns()
                
                # Create default patterns
                default_patterns = [
                    CoordinationPattern(
                        pattern_id="pair_programming_basic",
                        name="Basic Pair Programming",
                        description="Two agents collaborate on implementation tasks",
                        pattern_type=CoordinationPatternType.PAIR_PROGRAMMING,
                        required_roles=["developer", "reviewer"],
                        complexity_level=TaskComplexity.MODERATE,
                        coordination_steps=[
                            {"step": "initialize", "description": "Set up shared context"},
                            {"step": "collaborate", "description": "Work together on implementation"},
                            {"step": "review", "description": "Review and validate results"}
                        ],
                        success_metrics={"code_quality": 0.9, "time_efficiency": 0.8},
                        estimated_duration_minutes=120
                    ),
                    CoordinationPattern(
                        pattern_id="code_review_cycle",
                        name="Multi-Agent Code Review",
                        description="Structured code review with multiple specialized agents",
                        pattern_type=CoordinationPatternType.CODE_REVIEW_CYCLE,
                        required_roles=["developer", "reviewer", "tester"],
                        complexity_level=TaskComplexity.COMPLEX,
                        coordination_steps=[
                            {"step": "submit", "description": "Submit code for review"},
                            {"step": "review", "description": "Conduct thorough review"},
                            {"step": "test", "description": "Validate functionality"},
                            {"step": "approve", "description": "Final approval and merge"}
                        ],
                        success_metrics={"code_quality": 0.95, "defect_reduction": 0.9},
                        estimated_duration_minutes=90
                    ),
                    CoordinationPattern(
                        pattern_id="knowledge_sharing_session",
                        name="Knowledge Sharing Session",
                        description="Agents share expertise and best practices",
                        pattern_type=CoordinationPatternType.KNOWLEDGE_SHARING,
                        required_roles=["architect", "developer", "product"],
                        complexity_level=TaskComplexity.SIMPLE,
                        coordination_steps=[
                            {"step": "preparation", "description": "Prepare knowledge to share"},
                            {"step": "presentation", "description": "Present insights"},
                            {"step": "discussion", "description": "Collaborative discussion"},
                            {"step": "documentation", "description": "Document key learnings"}
                        ],
                        success_metrics={"knowledge_transfer": 0.85, "team_alignment": 0.9},
                        estimated_duration_minutes=60
                    )
                ]
                
                for pattern in default_patterns:
                    session.add(pattern)
                
                await session.commit()
                
                # Refresh patterns
                for pattern in default_patterns:
                    await session.refresh(pattern)
                
                self.logger.info("default_coordination_patterns_created", 
                               count=len(default_patterns))
                
                return default_patterns
                
            except Exception as e:
                await session.rollback()
                self.logger.error("failed_to_create_default_coordination_patterns", 
                                error=str(e))
                raise


# Global service instance
_coordination_persistence_service: Optional[CoordinationPersistenceService] = None

def get_coordination_persistence_service() -> CoordinationPersistenceService:
    """Get the global coordination persistence service instance."""
    global _coordination_persistence_service
    if _coordination_persistence_service is None:
        _coordination_persistence_service = CoordinationPersistenceService()
    return _coordination_persistence_service