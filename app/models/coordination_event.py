"""
Coordination Event model for LeanVibe Agent Hive 2.0

Tracks sophisticated multi-agent coordination activities, collaboration patterns,
and business value metrics for the enhanced coordination system.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Integer, ForeignKey, Float
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class CoordinationEventType(Enum):
    """Types of coordination events."""
    COLLABORATION_STARTED = "collaboration_started"
    COLLABORATION_COMPLETED = "collaboration_completed"
    COLLABORATION_FAILED = "collaboration_failed"
    AGENT_HANDOFF = "agent_handoff"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    CODE_REVIEW_CYCLE = "code_review_cycle"
    PAIR_PROGRAMMING = "pair_programming"
    DESIGN_REVIEW = "design_review"
    CONTINUOUS_INTEGRATION = "continuous_integration"
    CONFLICT_RESOLUTION = "conflict_resolution"
    DECISION_POINT = "decision_point"
    BUSINESS_VALUE_GENERATED = "business_value_generated"


class CollaborationQuality(Enum):
    """Quality levels for collaboration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    FAILED = "failed"


class CoordinationEvent(Base):
    """
    Records sophisticated multi-agent coordination activities.
    
    This model tracks the actual collaborative work between specialized agents,
    providing detailed visibility into autonomous development processes.
    """
    
    __tablename__ = "coordination_events"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    event_type = Column(ENUM(CoordinationEventType, name='coordinationeventtype'), nullable=False, index=True)
    collaboration_id = Column(String(255), nullable=True, index=True)  # Links related events
    
    # Session and agent coordination details
    session_id = Column(DatabaseAgnosticUUID(), ForeignKey("sessions.id"), nullable=True, index=True)
    participating_agents = Column(UUIDArray(), nullable=False, default=list)
    primary_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=True, index=True)
    coordination_pattern = Column(String(100), nullable=True, index=True)
    
    # Event context and outcomes
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    context = Column(JSON, nullable=True, default=dict)
    outcomes = Column(JSON, nullable=True, default=dict)
    artifacts_created = Column(StringArray(), nullable=True, default=list)
    
    # Quality and performance metrics
    quality_score = Column(Float, nullable=True)  # 0.0 to 1.0
    collaboration_efficiency = Column(Float, nullable=True)  # 0.0 to 1.0
    communication_count = Column(Integer, nullable=False, default=0)
    decisions_made_count = Column(Integer, nullable=False, default=0)
    knowledge_shared_count = Column(Integer, nullable=False, default=0)
    
    # Business value tracking
    estimated_time_saved = Column(Integer, nullable=True)  # minutes
    quality_improvement = Column(Float, nullable=True)  # 0.0 to 1.0
    business_value_score = Column(Float, nullable=True)  # calculated ROI
    cost_efficiency = Column(Float, nullable=True)  # cost per hour saved
    
    # Execution timing
    duration_seconds = Column(Float, nullable=True)
    estimated_duration = Column(Integer, nullable=True)  # seconds
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status and metadata
    success = Column(String(5), nullable=False, default="true")  # boolean as string for compatibility
    error_message = Column(Text, nullable=True)
    event_metadata = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    session = relationship("Session", back_populates="coordination_events")
    primary_agent = relationship("Agent", foreign_keys=[primary_agent_id])
    
    def __repr__(self) -> str:
        return f"<CoordinationEvent(id={self.id}, type='{self.event_type}', collaboration='{self.collaboration_id}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert coordination event to dictionary for serialization."""
        return {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "collaboration_id": self.collaboration_id,
            "participating_agents": [str(agent_id) for agent_id in (self.participating_agents or [])],
            "primary_agent_id": str(self.primary_agent_id) if self.primary_agent_id else None,
            "coordination_pattern": self.coordination_pattern,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "outcomes": self.outcomes,
            "artifacts_created": self.artifacts_created or [],
            "quality_score": self.quality_score,
            "collaboration_efficiency": self.collaboration_efficiency,
            "communication_count": self.communication_count,
            "decisions_made_count": self.decisions_made_count,
            "knowledge_shared_count": self.knowledge_shared_count,
            "estimated_time_saved": self.estimated_time_saved,
            "quality_improvement": self.quality_improvement,
            "business_value_score": self.business_value_score,
            "cost_efficiency": self.cost_efficiency,
            "duration_seconds": self.duration_seconds,
            "estimated_duration": self.estimated_duration,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success.lower() == "true",
            "error_message": self.error_message,
            "event_metadata": self.event_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def create_collaboration_started(cls, 
                                   collaboration_id: str,
                                   pattern: str,
                                   participating_agents: List[str],
                                   primary_agent_id: str,
                                   title: str,
                                   context: Dict[str, Any]) -> 'CoordinationEvent':
        """Create a collaboration started event."""
        return cls(
            event_type=CoordinationEventType.COLLABORATION_STARTED,
            collaboration_id=collaboration_id,
            coordination_pattern=pattern,
            participating_agents=[uuid.UUID(agent_id) for agent_id in participating_agents],
            primary_agent_id=uuid.UUID(primary_agent_id),
            title=title,
            context=context,
            started_at=datetime.utcnow()
        )
    
    @classmethod
    def create_collaboration_completed(cls,
                                     collaboration_id: str,
                                     pattern: str,
                                     participating_agents: List[str],
                                     primary_agent_id: str,
                                     title: str,
                                     outcomes: Dict[str, Any],
                                     quality_score: float,
                                     efficiency: float,
                                     duration: float,
                                     business_value: float) -> 'CoordinationEvent':
        """Create a collaboration completed event."""
        return cls(
            event_type=CoordinationEventType.COLLABORATION_COMPLETED,
            collaboration_id=collaboration_id,
            coordination_pattern=pattern,
            participating_agents=[uuid.UUID(agent_id) for agent_id in participating_agents],
            primary_agent_id=uuid.UUID(primary_agent_id),
            title=title,
            outcomes=outcomes,
            quality_score=quality_score,
            collaboration_efficiency=efficiency,
            duration_seconds=duration,
            business_value_score=business_value,
            completed_at=datetime.utcnow(),
            success="true"
        )
    
    def calculate_business_value(self) -> float:
        """Calculate comprehensive business value score."""
        if not self.duration_seconds or not self.quality_score:
            return 0.0
        
        # Base value from time efficiency
        time_efficiency = min(1.0, (self.estimated_duration or self.duration_seconds) / max(self.duration_seconds, 1))
        
        # Quality bonus
        quality_bonus = self.quality_score * 0.3
        
        # Collaboration efficiency bonus
        efficiency_bonus = (self.collaboration_efficiency or 0.5) * 0.2
        
        # Knowledge sharing bonus
        knowledge_bonus = min(0.2, (self.knowledge_shared_count or 0) * 0.05)
        
        # Decision making bonus
        decision_bonus = min(0.1, (self.decisions_made_count or 0) * 0.02)
        
        return min(1.0, time_efficiency + quality_bonus + efficiency_bonus + knowledge_bonus + decision_bonus)
    
    def update_business_metrics(self):
        """Update calculated business value metrics."""
        self.business_value_score = self.calculate_business_value()
        
        if self.duration_seconds and self.estimated_time_saved:
            # Calculate cost efficiency (business value per time invested)
            time_invested_hours = self.duration_seconds / 3600
            time_saved_hours = self.estimated_time_saved / 60
            if time_invested_hours > 0:
                self.cost_efficiency = time_saved_hours / time_invested_hours
        
        # Update quality improvement based on outcomes
        if self.outcomes and 'quality_metrics' in self.outcomes:
            metrics = self.outcomes['quality_metrics']
            self.quality_improvement = metrics.get('improvement_score', 0.0)


class BusinessValueMetric(Base):
    """
    Aggregated business value metrics for coordination activities.
    
    Provides high-level ROI and productivity metrics for dashboard display.
    """
    
    __tablename__ = "business_value_metrics"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    metric_type = Column(String(50), nullable=False, index=True)  # 'daily', 'weekly', 'monthly'
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Productivity metrics
    total_collaborations = Column(Integer, nullable=False, default=0)
    successful_collaborations = Column(Integer, nullable=False, default=0)
    average_quality_score = Column(Float, nullable=True)
    average_efficiency = Column(Float, nullable=True)
    
    # Time and cost metrics
    total_time_saved_hours = Column(Float, nullable=False, default=0.0)
    total_coordination_time_hours = Column(Float, nullable=False, default=0.0)
    cost_efficiency_ratio = Column(Float, nullable=True)  # time saved / time invested
    
    # Business value metrics
    total_business_value = Column(Float, nullable=False, default=0.0)
    average_business_value = Column(Float, nullable=True)
    roi_percentage = Column(Float, nullable=True)  # return on investment
    
    # Pattern effectiveness
    most_effective_pattern = Column(String(100), nullable=True)
    pattern_success_rates = Column(JSON, nullable=True, default=dict)
    
    # Agent performance
    top_performing_agents = Column(UUIDArray(), nullable=True, default=list)
    agent_collaboration_scores = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<BusinessValueMetric(type='{self.metric_type}', period={self.period_start} to {self.period_end})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert business value metric to dictionary for serialization."""
        return {
            "id": str(self.id),
            "metric_type": self.metric_type,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "total_collaborations": self.total_collaborations,
            "successful_collaborations": self.successful_collaborations,
            "success_rate": self.successful_collaborations / max(self.total_collaborations, 1),
            "average_quality_score": self.average_quality_score,
            "average_efficiency": self.average_efficiency,
            "total_time_saved_hours": self.total_time_saved_hours,
            "total_coordination_time_hours": self.total_coordination_time_hours,
            "cost_efficiency_ratio": self.cost_efficiency_ratio,
            "total_business_value": self.total_business_value,
            "average_business_value": self.average_business_value,
            "roi_percentage": self.roi_percentage,
            "most_effective_pattern": self.most_effective_pattern,
            "pattern_success_rates": self.pattern_success_rates,
            "top_performing_agents": [str(agent_id) for agent_id in (self.top_performing_agents or [])],
            "agent_collaboration_scores": self.agent_collaboration_scores,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def calculate_for_period(cls, 
                           period_start: datetime, 
                           period_end: datetime,
                           events: List[CoordinationEvent]) -> 'BusinessValueMetric':
        """Calculate business value metrics for a given period."""
        metric_type = "daily" if (period_end - period_start).days <= 1 else \
                     "weekly" if (period_end - period_start).days <= 7 else "monthly"
        
        total_collaborations = len(events)
        successful_collaborations = len([e for e in events if e.success == "true"])
        
        if not events:
            return cls(
                metric_type=metric_type,
                period_start=period_start,
                period_end=period_end,
                total_collaborations=0,
                successful_collaborations=0
            )
        
        # Calculate averages
        quality_scores = [e.quality_score for e in events if e.quality_score is not None]
        efficiency_scores = [e.collaboration_efficiency for e in events if e.collaboration_efficiency is not None]
        business_values = [e.business_value_score for e in events if e.business_value_score is not None]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else None
        avg_business_value = sum(business_values) / len(business_values) if business_values else None
        
        # Calculate time metrics
        total_time_saved = sum(e.estimated_time_saved or 0 for e in events) / 60  # convert to hours
        total_coordination_time = sum(e.duration_seconds or 0 for e in events) / 3600  # convert to hours
        
        cost_efficiency = total_time_saved / max(total_coordination_time, 0.01)
        roi_percentage = ((total_time_saved - total_coordination_time) / max(total_coordination_time, 0.01)) * 100
        
        # Pattern effectiveness
        pattern_counts = {}
        pattern_successes = {}
        for event in events:
            pattern = event.coordination_pattern or 'unknown'
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            if event.success == "true":
                pattern_successes[pattern] = pattern_successes.get(pattern, 0) + 1
        
        pattern_success_rates = {
            pattern: pattern_successes.get(pattern, 0) / count
            for pattern, count in pattern_counts.items()
        }
        
        most_effective_pattern = max(pattern_success_rates.items(), key=lambda x: x[1])[0] \
                               if pattern_success_rates else None
        
        return cls(
            metric_type=metric_type,
            period_start=period_start,
            period_end=period_end,
            total_collaborations=total_collaborations,
            successful_collaborations=successful_collaborations,
            average_quality_score=avg_quality,
            average_efficiency=avg_efficiency,
            total_time_saved_hours=total_time_saved,
            total_coordination_time_hours=total_coordination_time,
            cost_efficiency_ratio=cost_efficiency,
            total_business_value=sum(business_values),
            average_business_value=avg_business_value,
            roi_percentage=roi_percentage,
            most_effective_pattern=most_effective_pattern,
            pattern_success_rates=pattern_success_rates
        )